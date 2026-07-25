"""Native-Windows psmux backend for the terminal-multiplexer seam.

psmux (a Rust/ConPTY tmux re-implementation) speaks the tmux CLI through its
own distinctly-named ``psmux`` binary, so this leaf points the base's spawn
seam at that name, keeps every argv construction in :mod:`.tmux_base`, and
swaps only the shell dialect (PowerShell instead of POSIX sh) via the base's
hooks, plus the handful of behaviors where psmux diverges from tmux:
window-level ``-e`` is accepted but silently dropped, an attaching
``new-session`` is refused by a nesting guard when run from inside a psmux
pane, ``kill-session`` ignores the ``=name`` exact-match form, a quoted
command string does not survive psmux's outer re-parse (so shell source
travels as ``pwsh -EncodedCommand``), and ``pipe-pane`` strips every
dash-flag token from the piped command (so the log sink travels as a
positional sidecar ``.ps1``). Window ids are minted per server (one
server per session), so ``new_window``/``list_window_ids`` hand back
session-qualified ``session:@N`` ids that route to the owning server from
any caller (psmux/psmux#483) — the TUI-side mint/list surfaces
(``new_parked_window``, ``list_windows`` field ids) deliberately still hand
back bare ids, a different process context tracked in #291.
``available()`` additionally gates on
the reported version: psmux releases up to 3.3.6 kill recycled PIDs during
pane/session teardown without a process-identity check, which can take down
an unrelated long-lived process mid-run. See :mod:`.multiplexer` for the
contract.
"""

from __future__ import annotations

import base64
import os
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

from .tmux_base import PARKED_RETURN_DETACH, BaseTmuxBackend, TmuxError

_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _pwsh_quote(value: str) -> str:
    # A single-quoted PowerShell literal: no interpolation, and the only escape
    # is doubling the quote itself.
    return "'" + value.replace("'", "''") + "'"


class PsmuxMultiplexer(BaseTmuxBackend):
    """psmux backend — tmux-family argv from the base, PowerShell dialect and
    the documented psmux divergences here.

    Registered by :func:`~.multiplexer._load_builtin_backends` for ``win32``,
    mirroring :class:`~.tmux_backend.TmuxMultiplexer`.
    """

    # psmux ships psmux/pmux/tmux binaries built from the same source; spawning
    # the distinct psmux name never collides with another tmux-family install
    # (e.g. a tmux-windows port owning ``tmux`` on the same PATH).
    _BINARY = "psmux"
    # psmux emits UTF-8; decoding with the console codepage (cp1252) garbles
    # format-string output, and a stray byte must degrade visibly, not raise.
    _ENCODING = "utf-8"
    _ERRORS = "backslashreplace"

    # ------------------------------------------- shell dialect (PowerShell)

    # A command pwsh could not even start (not recognized) still runs the rest of
    # the source but leaves $LASTEXITCODE unset — coalesce with a plain `if`
    # (works on any PowerShell version) rather than the PS7-only `??` syntax.
    _EXIT_CAPTURE = "$ec = if ($null -eq $LASTEXITCODE) { 1 } else { $LASTEXITCODE }"
    _ECHO = "Write-Host"
    _PARK = "Read-Host"

    def _join_argv(self, argv: list[str]) -> str:
        # The call operator runs a quoted executable with quoted args verbatim.
        # A bare `& ` is a pwsh parse error, so refuse an empty argv here rather
        # than shipping a window that dies on launch.
        if not argv:
            raise TmuxError("empty command")
        return "& " + " ".join(_pwsh_quote(arg) for arg in argv)

    def _source_prefix(self) -> str:
        # psmux windows inherit the claude environment of whichever process
        # cold-started the psmux server (teammate mode, session ids, SSE ports).
        # Clear it so a CLI launched here starts fresh instead of impersonating
        # that session.
        return (
            "Get-ChildItem Env: | Where-Object { $_.Name -like 'CLAUDE_CODE_*' "
            "-or $_.Name -like 'CLAUDECODE*' -or $_.Name -eq 'PSMUX_CLAUDE_TEAMMATE_MODE' } "
            "| ForEach-Object { Remove-Item ('Env:' + $_.Name) }; "
        )

    def _shell_wrap(self, source: str) -> list[str]:
        # psmux joins the trailing argv and re-parses it through an outer shell,
        # which strips embedded quoting; -EncodedCommand (base64 of UTF-16LE) is
        # the lossless transport for arbitrary shell source.
        encoded = base64.b64encode(source.encode("utf-16-le")).decode("ascii")
        return ["pwsh", "-NoProfile", "-EncodedCommand", encoded]

    def _parked_trailer(self, return_opt: str) -> str:
        # The base's trailer re-expressed in pwsh — the tmux verbs are protocol-
        # identical across the family. Errors go to $null: a client or pane that
        # is already gone means the window just parks as-is.
        mux = self._BINARY
        return (
            f"$ret = {mux} show-options -wqv {_pwsh_quote(return_opt)} 2>$null; "
            f"if ($ret -eq '{PARKED_RETURN_DETACH}') {{ {mux} detach-client 2>$null }} "
            f"elseif ($ret) {{ {mux} switch-client -t $ret 2>$null; "
            f"if ($LASTEXITCODE -ne 0) {{ {mux} switch-client -l 2>$null }} }}"
        )

    def _window_launch(self, env: dict[str, str], command: str) -> list[str]:
        # psmux accepts `new-window -e` but silently drops it, so the env rides
        # an in-source prelude instead. `command` arrives POSIX-quoted (callers
        # shlex-quote each arg), so split it here and re-quote for pwsh.
        for key in env:
            if not _ENV_NAME.fullmatch(key):
                raise TmuxError(f"invalid environment variable name: {key!r}")
        try:
            argv = shlex.split(command)
        except ValueError as exc:
            raise TmuxError(f"unparseable command: {exc}") from exc
        prelude = "".join(f"$env:{key} = {_pwsh_quote(value)}; " for key, value in env.items())
        source = self._source_prefix() + prelude + self._join_argv(argv)
        return self._shell_wrap(source)

    # ------------------------------------------------- psmux divergences

    def new_session(
        self, name: str, cwd: Path, cols: int | None = None, lines: int | None = None
    ) -> None:
        # psmux's nesting guard refuses new-session from inside a psmux pane
        # (current builds only for an attaching one, older builds no-op'd `-d`
        # too — exit 0, nothing created); the documented bypass is one env var
        # on the create call, kept as a cheap belt. The create env copies the
        # parent env rather than building from scratch — Windows children need
        # SystemRoot etc. — but scrubs the claude session vars (the same names
        # _source_prefix clears per window): this call may cold-start the psmux
        # server, whose env every window then inherits.
        env = {
            k: v
            for k, v in os.environ.items()
            if not (
                k.upper().startswith(("CLAUDE_CODE_", "CLAUDECODE"))
                or k.upper() == "PSMUX_CLAUDE_TEAMMATE_MODE"
            )
        }
        env["PSMUX_ALLOW_NESTING"] = "1"
        geometry = ["-x", str(cols), "-y", str(lines)] if cols and lines else []
        try:
            proc = self._run(
                ["new-session", "-d", "-s", name, "-c", str(cwd), *geometry],
                check=False,
                env=env,
            )
        except (subprocess.TimeoutExpired, OSError) as exc:
            raise TmuxError(f"{self._BINARY} new-session failed: {exc}") from exc
        if proc.returncode != 0:
            raise TmuxError(f"{self._BINARY} new-session failed: {proc.stderr.strip()}")
        # Belt for the nesting guard's historical no-op mode (exit 0, nothing
        # created): verify the session exists so the failure blames session
        # creation, not the next verb's "can't find session".
        if not self.has_session(name):
            raise TmuxError(
                f"{self._BINARY} new-session exited 0 but session {name!r} was not "
                "created (nesting guard no-op?)"
            )

    def kill_session(self, name: str) -> None:
        # psmux ignores the `=name` exact-match form for kill-session; plain-name
        # targeting works. Same best-effort tolerance as the base.
        if not shutil.which(self._BINARY):
            return
        try:
            self._run(["kill-session", "-t", name], check=False)
        except (subprocess.SubprocessError, OSError):
            pass

    @staticmethod
    def _qualified_window_id(session: str, window_id: str) -> str:
        # psmux mints window ids per server (one server per session), so a bare
        # `@N` replayed as a `-t` target routes by the caller's $TMUX — from a
        # ctl pane that is the wrong server entirely (#254). A `session:@N`
        # target routes by the session's port file instead (psmux/psmux#483:
        # qualified targets are at full parity; bare ids are a permanent model
        # boundary). Degrade to the bare id when the session name contains `:`
        # (it would split at the wrong colon) — the #221 rule — or when the id
        # is falsy (a failure sentinel must pass through unchanged).
        #
        # Composed here rather than via self.target() — which would emit
        # `=session:@N`, and does parse (psmux strips the `=`) — because the
        # seam requires target() to stay a stable *by-name* token and `@N` is
        # an id. Do not "unify" the two grammars: current_return_target's
        # `=session:%N` is a target(), this is not.
        if not window_id or ":" in session:
            return window_id
        return f"{session}:{window_id}"

    def new_window(
        self, session: str, name: str, cwd: Path, env: dict[str, str], command: str
    ) -> str:
        # Session-qualify the minted id (see _qualified_window_id) so every
        # downstream `-t` consumer inherits an unambiguous target. Post-process
        # the base's return rather than reformatting `-F` so the argv stays
        # byte-identical to the base's.
        window_id = super().new_window(session, name, cwd, env, command)
        return self._qualified_window_id(session, window_id)

    def list_window_ids(self, session: str) -> list[str]:
        # psmux's list-windows emits bare `@N` lines; qualify them identically
        # to new_window or window_alive's membership check (native_id in
        # list_window_ids) would read every window as dead.
        return [
            self._qualified_window_id(session, window_id)
            for window_id in super().list_window_ids(session)
        ]

    def current_return_target(self) -> str | None:
        # psmux runs one server per session, so a bare pane id recorded on a
        # control-session window is session-local at replay: at best
        # unresolvable, at worst colliding with a real control-session pane and
        # landing the client on the wrong one with exit 0 (psmux/psmux#483).
        # Qualify with the session: psmux's parse_target accepts a pane id in
        # the window slot of `=session:%N` and (on releases carrying the #483
        # fix) forwards it to the owning server, where the id resolves in
        # exactly the right per-server id space. tmux must NOT receive this
        # form — its window resolver has no pane-id handling, so the qualified
        # target errors and the return degrades to `switch-client -l` — which
        # is why composition is backend-owned rather than seam-uniform.
        pane = self.current_pane_id()
        if not pane:
            return None
        session = self.current_session()
        # Degrade to the bare id (the base default, pre-#483 hazard and all)
        # rather than None when the session probe fails, answers empty, or
        # answers a name the `=session:%N` grammar cannot carry: a resolvable
        # own pane means we ARE inside the multiplexer, so recording "detach"
        # would strand the client.
        if not session or ":" in session:
            return pane
        return self.target(session, pane)

    def pipe_pane(self, window_id: str, log_file: Path) -> None:
        # The base's POSIX `cat >>` sink assumes a POSIX host shell, and psmux
        # strips every dash-flag token from the piped command before spawning it
        # (psmux/psmux#482) — any `pwsh -EncodedCommand` transport dies on launch
        # — so the sink source lives in a sidecar .ps1 invoked positionally. The
        # sink is byte-exact like `cat >>` (raw stream copy: no console decode of
        # the pane bytes, no re-encode, no CRLF normalization) and flushes per
        # chunk: the run log is live-tailed for activity detection, and a
        # buffered copy never surfaces bytes — pipe EOF is unreliable on psmux.
        # Known ceilings until upstream restores a flag transport: whether a
        # spaced or $-bearing path survives psmux's quote re-parse is untested,
        # an AllSigned/Restricted execution policy refuses the unsigned .ps1,
        # and a spawn race that exits 0 still yields a silent empty log — the
        # warning below covers surfaced failures only.
        sink_file = log_file.with_name(log_file.name + ".sink.ps1")
        if any(char in str(sink_file) for char in ("$", "`")):
            print(
                f"warning: pipe-pane log capture failed for {window_id}: "
                "sidecar path contains PowerShell interpolation syntax",
                file=sys.stderr,
            )
            return
        sink = (
            "$in = [System.Console]::OpenStandardInput()\n"
            f"$out = [System.IO.File]::Open({_pwsh_quote(str(log_file))}, "
            "'Append', 'Write', 'Read')\n"
            "$buf = New-Object byte[] 4096\n"
            "while (($n = $in.Read($buf, 0, $buf.Length)) -gt 0) "
            "{ $out.Write($buf, 0, $n); $out.Flush() }\n"
            "$out.Dispose()\n"
        )
        try:
            sink_file.write_text(sink, encoding="utf-8")
            self._tmux("pipe-pane", "-t", window_id, "-o", f'pwsh "{sink_file}"')
        except (TmuxError, OSError, UnicodeEncodeError) as exc:
            # Best-effort, as the base: a window that died on launch (or psmux's
            # first-pipe-after-new-window spawn race, noted in psmux/psmux#482)
            # is not a setup failure — but say so, or an empty run log is
            # unexplainable.
            print(
                f"warning: pipe-pane log capture failed for {window_id}: {exc}",
                file=sys.stderr,
            )

    # Releases up to this version can force-kill a recycled PID during pane
    # teardown and let orphaned servers accumulate — engine-fatal, so they
    # must never be selected.
    _LAST_UNSUPPORTED = (3, 3, 6)
    # Class-level default; instances shadow it on first probe. Never assign on
    # the class outside tests — that would poison every future instance.
    _version_ok: bool | None = None

    def available(self) -> bool:
        # Every window launch needs pwsh alongside the psmux binary itself.
        # The version gate fails closed: psmux prints `tmux X.Y.Z` (the tmux
        # prefix is kept deliberately for tmux-version parsers), and an old or
        # unidentifiable install reads as unusable. A forced backend name (env
        # var or policy) still bypasses this probe, with a warning at the
        # launch gates (see multiplexer.mux_usable). The gate verdict is cached
        # on the instance so repeated availability polls don't each spawn a
        # version query; the lru-cached selected instance re-probes a swapped
        # install only on restart (detect_multiplexers' fresh instances
        # re-probe every call).
        if not all(shutil.which(exe) for exe in (self._BINARY, "pwsh")):
            return False
        if self._version_ok is None:
            # A missing patch segment reads as 0 — psmux hardwires three-part
            # Cargo semver today, but real tmux versions are two-part and the
            # compat prefix invites upstream to mirror that format someday.
            reported = re.match(r"tmux (\d+)\.(\d+)(?:\.(\d+))?", self.version() or "")
            self._version_ok = bool(reported) and (
                tuple(int(part or 0) for part in reported.groups()) > self._LAST_UNSUPPORTED
            )
        return self._version_ok
