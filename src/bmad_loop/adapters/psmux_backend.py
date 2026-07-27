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
server per session), so every id this backend hands back —
``new_window``, ``list_window_ids``, ``new_parked_window``, the
``window_id`` columns of ``list_windows``, and ``current_window_id`` — is
session-qualified to ``session:@N`` (degrading to the bare id only where
the grammar cannot carry the session name — see ``_qualified_window_id``)
and routes to the owning server from any caller (psmux/psmux#483).
``select_window`` is the one verb that
cannot take that form: psmux validates a scoped target's window part
CLI-side against window index/name only, so the override resolves the id
back to an index first. Per-window user options do not exist at all, so
the window-option verbs, the ``@``-prefixed columns of ``list_windows``
and the parked trailer route through a substitute channel — see the
``per-window option channel (#310)`` block below for the model and its
rules. ``detach-client`` and ``switch-client`` report dispatch rather than
effect — every arm exits 0 whether or not a client moved — so both seam
booleans are measured against the session's attached-client count instead of
read off the exit code; see the ``client verbs: observed effect (#317)``
block. ``available()`` additionally gates on
the reported version: psmux releases up to 3.3.6 kill recycled PIDs during
pane/session teardown without a process-identity check, which can take down
an unrelated long-lived process mid-run. The psmux behaviors cited in this
module were read from the psmux source at tag ``v3.3.7``. See
:mod:`.multiplexer` for the contract.
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
        #
        # The base reads the return target with `show-options -wqv`, which is
        # dead on psmux (#310), so this reads the session-scoped id-keyed option
        # instead. Unlike every other caller of that channel, the trailer cannot
        # be handed its own window id — it is built before the window exists —
        # so it probes for it in-pane. `-t $env:TMUX_PANE` pins the probe to the
        # pane's own window (psmux sets TMUX_PANE in every pane, and a bare `%N`
        # target resolves globally via DisplayMessageById); a target-less probe
        # would resolve the server's *active* window, which is another window's
        # key the moment focus moves after Enter. No `-t <session>`: running
        # inside the pane, $TMUX already routes to this window's own server.
        #
        # Both captures go through "$(...)".Trim(): a bare capture yields an
        # array when psmux emits more than one line, and `-eq` on an array
        # filters instead of comparing — silently taking neither branch.
        mux = self._BINARY
        probe = (
            '"$(' + mux + " display-message -p -t $env:TMUX_PANE '#{window_id}' 2>$null)\".Trim()"
        )
        read_key = '"$(' + mux + ' show-options -qv $key 2>$null)".Trim()'
        return (
            f"$wid = {probe}; "
            # A failed probe skips the return AND the key free; the orphan
            # sweep at a later parked-window launch reclaims the key once the
            # window is gone.
            f"if ($wid) {{ $key = {_pwsh_quote(return_opt + '_')} + $wid; "
            f"$ret = {read_key}; "
            f"if ($ret -eq '{PARKED_RETURN_DETACH}') {{ {mux} detach-client 2>$null }} "
            # The switch leg is still inert at 3.3.7 (psmux/psmux#483: the
            # server splits the target without parse_target); only the detach
            # return actually moves a client today. Kept: it is the correct
            # verb the moment upstream lands the fix.
            f"elseif ($ret) {{ {mux} switch-client -t $ret 2>$null; "
            f"if ($LASTEXITCODE -ne 0) {{ {mux} switch-client -l 2>$null }} }} "
            # Free the key on the way out: the ctl session outlives every run,
            # so a key left behind is one this server carries for its whole life.
            f"{mux} set-option -u $key 2>$null }}"
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
        # boundary). Degrade to the bare id when the session name is empty or
        # contains `:` (the grammar cannot carry it: `":@N"` parses sessionless
        # and `a:b:@N` splits at the wrong colon) — the #221 rule — or when the
        # id is falsy (a failure sentinel must pass through unchanged).
        #
        # Composed here rather than via self.target() — which would emit
        # `=session:@N`, and does parse (psmux strips the `=`) — because the
        # seam requires target() to stay a stable *by-name* token and `@N` is
        # an id. Do not "unify" the two grammars: current_return_target's
        # `=session:%N` is a target(), this is not.
        if not window_id or not session or ":" in session:
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

    def new_parked_window(
        self, session: str, name: str, cwd: Path, argv: list[str], return_opt: str
    ) -> str:
        # Same per-server ambiguity as new_window, worse context: the launcher
        # usually runs OUTSIDE any pane, so a bare `@N` replayed as a `-t` target
        # falls through to the most-recent-session fallback rather than the
        # session that just minted it.
        window_id = super().new_parked_window(session, name, cwd, argv, return_opt)
        # Launch time is the reconcile point for keys whose window is gone —
        # Enter-dismissing a parked window closes it without kill_window ever
        # running, so its keys would otherwise outlive it (#310).
        self._sweep_orphan_keys(session)
        return self._qualified_window_id(session, window_id)

    def list_windows(self, session: str, fields: list[str]) -> list[tuple[str, ...]]:
        # Two corrections on the columns the prune reads. `window_id` is
        # qualified — the prune replays those values as kill-window targets, and
        # a bare one can hit another server's identically-numbered window. An
        # `@` field never reaches psmux: `#{@name}` expands from the one
        # per-server map, so every row would carry the same value. Probe the
        # window id in its place and fill from the id-keyed options, fetched as
        # ONE full listing — a flat extra call per list_windows, regardless of
        # row or column count (#310).
        opt_columns = {i: field for i, field in enumerate(fields) if field.startswith("@")}
        probe_fields = [
            "window_id" if i in opt_columns else field for i, field in enumerate(fields)
        ]
        rows = super().list_windows(session, probe_fields)
        id_columns = {i for i, field in enumerate(fields) if field == "window_id"}
        if not opt_columns and not id_columns:
            return rows
        # The #221 degrade: an empty or `:`-bearing session cannot be routed
        # with `-t`, and an unrouted read would answer from whichever server
        # the fallback picks — fill "" without issuing reads at all.
        degraded = not session or ":" in session
        options = self._scoped_options(session) if opt_columns and not degraded else None
        if opt_columns and not degraded and options is None:
            # Say it: every column degrades to "unset" at once, so a prune reads
            # as if nothing were ever tagged. Visible on the CLI prune (cli.py,
            # incl. the --dry-run this most affects); NOT under the TUI, which
            # captures stderr for the app's whole run (tui/app.py's run_tui
            # note) — the select_window precedent below, same deliberate ceiling.
            print(
                f"warning: show-options listing failed on {session}; "
                "option columns read as unset",
                file=sys.stderr,
            )
        out: list[tuple[str, ...]] = []
        for row in rows:
            values = list(row)
            for i, option in opt_columns.items():
                # values[i] is the bare `@N` probed in the option column's place.
                digits = self._id_digits(values[i])
                if degraded or not digits or options is None:
                    # Unreadable degrades to "" ("unset") rather than to a
                    # "read failed" sentinel: the caller's untagged fallback
                    # proves ownership on its own (the ctl prune claims an
                    # untagged window only when the run dir exists under THIS
                    # project, and run ids are unique — runs.new_run_id), so a
                    # third state in the column would buy only the skip of our
                    # own dead window.
                    values[i] = ""
                else:
                    values[i] = options.get(self._scoped_option_key(option, digits), "")
            for i in id_columns:
                values[i] = self._qualified_window_id(session, values[i])
            out.append(tuple(values))
        return out

    # ------------------------------------ per-window option channel (#310)
    #
    # Per-window user options do not exist on psmux: there is one scope per
    # server, and every `-w` read of an `@` name returns '' before the map is
    # consulted (a 14-name builtin allowlist gates it). Deliberate, per
    # psmux/psmux#321 — a boundary to route around, not a bug to wait out.
    #
    # Substitute: a SESSION-scoped option whose key carries the window id
    # (`@bmad_project_@3` for `@3`). Two rules keep it correct —
    #   - `-t <session>` on every out-of-pane write and read (the parked
    #     trailer runs in-pane and rides `$TMUX` instead — see
    #     `_parked_trailer`). psmux picks the server from the target, and
    #     without an explicit session it falls back to $TMUX / most-recent —
    #     i.e. some other server. The session comes from the qualified ids
    #     this backend already mints, which is why #310 lands after #291.
    #   - the key ends with the full id (`_@3`, never bare digits): the ctl
    #     server loads the user's psmux config, so the map is shared, and the
    #     `_@` shape is what keeps foreign config options out of the generic
    #     cleanup sweeps. Not airtight — psmux accepts any `@` name, so a
    #     hand-written `@theme_@3` WOULD match — but no naming convention in
    #     the tmux/psmux ecosystem puts `@` mid-name, and a bmad-owned prefix
    #     guard would hardcode caller names into the backend. The session
    #     stays out of the key; routing already carries it.
    #
    # Builtin window options keep the base's `-w` argv — psmux accepts those
    # allowlisted names (only `automatic-rename` has true per-window storage;
    # the rest read the global value), and rerouting them into this channel
    # would break reads that work today.

    # A window target as the seam composes it: `session:@N`, `=session:@N`, or
    # `=session:<window-name>`. Deliberately looser than _QUALIFIED_ID, which
    # only admits the id form — set_return_pane passes a name token.
    _SESSION_WINDOW = re.compile(r"^(?P<session>[^:]+):(?P<window>.+)$")

    @staticmethod
    def _id_digits(window_id: str) -> str:
        """`@3` -> `3`; `""` for anything that is not a bare window id."""
        return window_id[1:] if PsmuxMultiplexer._BARE_ID.fullmatch(window_id) else ""

    def _option_scope(self, target: str) -> tuple[str, str] | None:
        """``(session, id-digits)`` for a window target, or None when it carries
        no session and so cannot be routed to a specific server.

        A sessionless target reaches here only where `_qualified_window_id`
        degraded. Guessing a server for it is the misrouting qualification
        exists to prevent, so it returns None and the caller declines to act.
        """
        match = self._SESSION_WINDOW.fullmatch(target.removeprefix("="))
        if match is None:
            return None
        session, window = match["session"], match["window"]
        digits = self._id_digits(window)
        if not digits:
            # A name token; same one-round-trip renumber race as _window_index.
            digits = self._id_digits(self._window_id_for_name(session, window) or "")
        return (session, digits) if digits else None

    def _window_id_for_name(self, session: str, name: str) -> str | None:
        """Bare window id of the window called ``name`` in ``session``, or None."""
        # super(): the base emits unqualified ids, the form the key needs.
        for win_id, win_name in super().list_windows(session, ["window_id", "window_name"]):
            if win_name == name:
                return win_id
        return None

    @staticmethod
    def _scoped_option_key(option: str, digits: str) -> str:
        return f"{option}_@{digits}"

    # The channel's key suffix. Anchored: `@bmad_project_@13` must not read as
    # a key of window `@3`, and a foreign `@color_3` (no `_@` shape) never
    # matches (see the namespace note in the channel comment above).
    _KEY_SUFFIX = re.compile(r"_@(\d+)$")

    def _read_scoped(self, session: str, option: str, digits: str) -> str | None:
        # No `-w`: the session-scoped read is the one that reaches the map.
        # None = transport failure, "" = unset. The ABC read admits only "", so
        # the caller collapses them — the distinction survives as a warning.
        try:
            proc = self._run(
                ["show-options", "-qv", "-t", session, self._scoped_option_key(option, digits)],
                check=False,
            )
        except (subprocess.SubprocessError, OSError):
            return None
        return proc.stdout.strip() if proc.returncode == 0 else None

    def _warn_unroutable(self, verb: str, target: str, option: str) -> None:
        # Silence would leave a write that looks like it landed but did not.
        print(
            f"warning: {verb} {option} skipped — {target} does not resolve to a "
            "window id on a routable session (unqualified target, unknown window "
            "name, or the name-resolve listing failed), so the per-window option "
            "channel is unavailable",
            file=sys.stderr,
        )

    @staticmethod
    def _transportable(value: str) -> bool:
        # The value crosses psmux's CLI→server control line: the client wraps a
        # spaced value in double quotes escaping only `"`, and the server
        # tokenizer treats a bare `'` as a quote opener, drops `-`-leading
        # tokens, and collapses `\\` inside double quotes. A value that cannot
        # survive that hop verbatim is refused loudly instead of stored
        # corrupted — a tag that reads back different from what the prune
        # compares against makes the window silently unprunable.
        #
        # The two branches deliberately ban DIFFERENT shapes. Inside the
        # client's double quotes `'` is literal and a mid-token `;` survives,
        # so a spaced `O'Brien Files` or `a; b` path passes. But the one-shot
        # chain splitter (config.rs `split_chained_commands`) is NOT
        # quote-aware: it cuts on whitespace-delimited `;`/`\;` TOKENS, so
        # `a ; b` stores as `a` and hands the rest to the server as a command
        # (live-verified on 3.3.7; psmux/psmux#499). `\\` collapses and a `"` can close the
        # wrapper early (client-escaped `\"` after a backslash reads back as
        # `\\` + closing quote). So the spaced branch refuses `"`, `\\`, a
        # trailing `\`, and standalone `;`/`\;` tokens. Outside double quotes
        # the tokenizer's escape branch never fires (commands.rs:690 requires
        # in_double_quotes) — an unquoted backslash is pushed literally, psmux
        # being Windows-native — so the unspaced branch does not ban `\\` and a
        # spaceless UNC path (`\\srv\share`) rides verbatim.
        # Non-ASCII-space whitespace (NBSP, tab, …) is refused outright: the
        # client quotes only on ASCII `' '` (main.rs `s.contains(' ')`) while
        # the server tokenizer splits on Unicode `is_whitespace()` — an NBSP in
        # an unquoted value splits the token server-side. Leading/trailing
        # ASCII space is refused too: it survives the wire, but this backend's
        # own reads strip/Trim, so the value could never read back equal.
        if not value or value.startswith("-") or any(c.isspace() and c != " " for c in value):
            return False
        if value != value.strip():
            return False
        if " " in value:  # will be double-quoted by the psmux client
            if '"' in value or "\\\\" in value or value.endswith("\\"):
                return False
            return all(tok not in (";", "\\;") for tok in value.split())
        return not any(c in value for c in ";'\"")

    def _write_scoped(self, verb: list[str], key: str) -> None:
        # Both mutating verbs, one body. A write that silently failed re-opens
        # the mis-scoped prune this channel exists to close, and a silently
        # failed `-u` leaves a live return key that replays the return move when
        # the window's command exits — so failures are said out loud either way
        # (the verbs stay best-effort: warn, never raise).
        label = " ".join(verb[: verb.index("-t")])  # `set-option` / `set-option -u`
        try:
            proc = self._run(verb, check=False)
            if proc.returncode != 0:
                print(f"warning: {label} {key} failed: {proc.stderr.strip()}", file=sys.stderr)
        except (subprocess.SubprocessError, OSError) as exc:
            print(f"warning: {label} {key} failed: {exc}", file=sys.stderr)

    def set_window_option(self, target: str, option: str, value: str) -> None:
        if not option.startswith("@"):
            super().set_window_option(target, option, value)
            return
        # Scope first, transport second: an unroutable target has nothing to
        # write AND nothing to free, and resolving here keeps the refusal path
        # from re-entering unset_window_option and warning twice about a verb
        # the caller never issued.
        scope = self._option_scope(target)
        if scope is None:
            self._warn_unroutable("set-option", target, option)
            return
        session, digits = scope
        key = self._scoped_option_key(option, digits)
        if not self._transportable(value):
            print(
                f"warning: set-option {option} skipped — value does not survive "
                "psmux's control-line transport verbatim; the key is freed and "
                "the option reads as unset",
                file=sys.stderr,
            )
            # Free any prior value: a refused REwrite must not leave the stale
            # one to be replayed later (e.g. a parked return target).
            self._write_scoped(["set-option", "-u", "-t", session, key], key)
            return
        self._write_scoped(["set-option", "-t", session, key, value], key)

    def unset_window_option(self, target: str, option: str) -> None:
        if not option.startswith("@"):
            super().unset_window_option(target, option)
            return
        scope = self._option_scope(target)
        if scope is None:
            self._warn_unroutable("set-option -u", target, option)
            return
        session, digits = scope
        # `-u` genuinely frees the key: the server's SetOptionUnset handler
        # removes `@`-prefixed names from the map (verified at v3.3.7).
        key = self._scoped_option_key(option, digits)
        self._write_scoped(["set-option", "-u", "-t", session, key], key)

    def show_window_option(self, target: str, option: str) -> str:
        if not option.startswith("@"):
            return super().show_window_option(target, option)
        scope = self._option_scope(target)
        if scope is None:
            # An unroutable target warns for reads the same as for writes —
            # no verb is sent; "" already means "unset" to every caller.
            self._warn_unroutable("show-options", target, option)
            return ""
        value = self._read_scoped(scope[0], option, scope[1])
        if value is None:
            # Transport failure, not a miss. The return value still degrades
            # to "" ("unset") — that is all the ABC read admits — but say so.
            print(
                f"warning: show-options {option} failed on {target}; treating as unset",
                file=sys.stderr,
            )
            return ""
        return value

    def _scoped_options(self, session: str) -> dict[str, str] | None:
        """All `@`-prefixed options on ``session`` as ``{key: value}``, or None
        on any failure (distinct from {} — an empty map is a real answer).
        Live-verified on 3.3.7: `show-options -q -t <session>` lists user
        options as `@key "value"` lines; accepted channel values can contain
        neither `"` nor newlines (see _transportable), so the quote strip is
        lossless for every value this backend wrote. Known hole: one server
        request handler (the plugin-drain copy, server/mod.rs:465 at v3.3.7)
        answers this listing empty-with-success while keys exist, so a
        surprising {} is possible and is not proof that no keys are set."""
        try:
            proc = self._run(["show-options", "-q", "-t", session], check=False)
        except (subprocess.SubprocessError, OSError):
            return None
        if proc.returncode != 0:
            return None
        options: dict[str, str] = {}
        for line in proc.stdout.splitlines():
            name, _, rest = line.partition(" ")
            if not name.startswith("@"):
                continue
            if len(rest) >= 2 and rest.startswith('"') and rest.endswith('"'):
                rest = rest[1:-1]
            options[name] = rest
        return options

    def _sweep_orphan_keys(self, session: str) -> None:
        # Free `_@N` keys whose window no longer exists (Enter-dismissed parked
        # windows never pass through kill_window). The `_@` shape keeps foreign
        # config options out of the sweep; window ids are never recycled within
        # a server, so a swept key cannot belong to a future window.
        #
        # Order matters: keys are snapshotted BEFORE the live-window listing, so
        # a window minted-and-tagged between the two calls has its key outside
        # the snapshot and cannot be swept as a false orphan.
        try:
            options = self._scoped_options(session)
            if options is None:
                return
            live = set(super().list_window_ids(session))  # base = bare ids
            if not live:
                # A session being swept just minted a window, so an empty live
                # list is a failed probe, not an empty session — treating it as
                # truth would sweep every key, live windows included.
                return
            for name in options:
                match = self._KEY_SUFFIX.search(name)
                if match and f"@{match.group(1)}" not in live:
                    self._run(["set-option", "-u", "-t", session, name], check=False)
        except (subprocess.SubprocessError, OSError, TmuxError) as exc:
            # Reconcile is opportunistic; the mint must never fail on it. But a
            # sweep that fails every launch leaks keys for the server's whole
            # life, so the failure is at least visible.
            print(f"warning: orphan-key sweep failed on {session}: {exc}", file=sys.stderr)

    def kill_window(self, target: str) -> None:
        # Free the window's id-keyed session options before the kill: the ctl
        # session outlives every run, so a leaked key lives as long as the
        # server. Discovery is generic by the `_@<digits>` suffix — the backend
        # must not know which option names callers use. Best-effort throughout:
        # cleanup failure never blocks the kill. Known ceiling, accepted: the
        # kill itself is best-effort, so a kill that then fails leaves a live
        # window without its keys — the prune's untagged run-dir fallback still
        # scopes the retry, and a lost return key just parks the window as-is.
        # Cost, accepted: one listing round-trip per kill, two for a name target
        # (name-resolve, then keys; agent-window kills pay it too, for
        # nothing); skip-by-session-name if that ever measures.
        scope = self._option_scope(target)
        if scope is not None:
            session, digits = scope
            suffix = f"_@{digits}"
            try:
                for name in self._scoped_options(session) or ():
                    if name.endswith(suffix):
                        self._run(["set-option", "-u", "-t", session, name], check=False)
            except (subprocess.SubprocessError, OSError):
                pass
        super().kill_window(target)

    # What _qualified_window_id composes: `<session>:@<n>`. The session part
    # excludes `:` because that is exactly when qualification degrades to a bare
    # id; requiring `@<digits>` keeps a `=session:window-name` token out.
    _QUALIFIED_ID = re.compile(r"^(?P<session>[^:]+):(?P<window_id>@\d+)$")
    _BARE_ID = re.compile(r"^@\d+$")

    def current_window_id(self) -> str | None:
        # Must match list_windows' window_id form: the ctl prune skips its own
        # window by comparing them, so a mismatch makes a prune run from inside a
        # ctl window kill that window. Hence ONE probe rather than a window id
        # plus a separate session name — a split probe can resolve the id and
        # fail the session, and rows are qualified from the session list_windows
        # was *passed*, so neither a bare id nor None could ever equal one.
        probed = self._display_message("#{session_name}:#{window_id}")
        if not probed:
            return None
        if self._QUALIFIED_ID.fullmatch(probed):
            return probed
        # A `:`-bearing (or empty) session name can't carry the grammar — the
        # #221 degrade, which list_windows applies to its rows identically. A
        # malformed id is no target at all; half-parsing one would aim a kill.
        _, _, window_id = probed.rpartition(":")
        return window_id if self._BARE_ID.fullmatch(window_id) else None

    def select_window(self, target: str) -> None:
        # psmux checks a scoped target's window part CLI-side before sending, and
        # only ever matches `#{window_index}`/`#{window_name}` — never
        # `#{window_id}` — so a qualified id exits 1 with "can't find window"
        # while the server itself resolves ids fine (FocusWindowById). Translate
        # to the index that check accepts and the same window is focused.
        # The `=` strip matters: target(session, window) composes `=session:@N`,
        # which would capture the session as `=session` and look up `==session`
        # (psmux's parse_target removes only one `=`), silently focusing nothing.
        match = self._QUALIFIED_ID.fullmatch(target.removeprefix("="))
        if match is not None:
            session = match["session"]
            index = self._window_index(session, match["window_id"])
            if index is None:
                # The unresolved id sent below is exactly the form that CLI-side
                # check rejects, and its exit 1 lands in a pipe the base
                # discards — so the send is known-futile and the caller's later
                # attach shows whichever window happens to be current. Guessing
                # an index instead could focus an unrelated window, so warn and
                # send: the pipe-pane sidecar precedent, but a weaker one, and
                # deliberately not load-bearing. Today's only caller of the
                # qualified-id path (select_ctl_window_id, from the TUI's resolve
                # launch) runs inside the Textual app, which captures stderr for
                # the app's whole run — see the run_tui note in tui/app.py, which
                # pre-trips the forced-backend warning for exactly this reason —
                # so this emission is invisible there. It is kept for the CLI-side
                # callers tui/launch.py is textual-free to allow; a TUI-visible
                # miss would need a return value, which is more seam than a
                # best-effort focus change is worth.
                print(
                    f"warning: select-window could not resolve {target} to a "
                    "window index; the window will not be focused",
                    file=sys.stderr,
                )
            else:
                target = f"{session}:{index}"
        # A name token or a bare id goes through untouched.
        super().select_window(target)

    def _window_index(self, session: str, window_id: str) -> str | None:
        """Display index of ``window_id`` within ``session``, or None when it
        cannot be resolved.

        Known ceiling: psmux can renumber windows between this lookup and the
        verb using the answer. The race is one round-trip wide and the caller is
        a best-effort focus change, so no lock; targeting by name instead would
        reopen the ctl-window name collisions stable ids exist to avoid.
        """
        # `window_index` is the display index — the field psmux's own existence
        # check compares against, so matching it exactly is the point.
        for row in super().list_windows(session, ["window_id", "window_index"]):
            if row[0] == window_id:
                return row[1] or None
        return None

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

    # ------------------------------- client verbs: observed effect (#317)
    #
    # psmux's client verbs report *dispatch*, not effect. At ``v3.3.7`` both CLI
    # arms end in ``send_control(cmd)?; return Ok(())``, so the only nonzero exit
    # is an unreachable session server: ``detach-client`` succeeds with zero
    # clients attached (and a flag-less detach is promoted server-side to
    # detach-all), and *no* form of ``switch-client`` — ``-t`` included —
    # carries a server reply. Later builds narrow this but do not close it: a
    # response path landed after the tag and only for ``-t``, leaving ``-l``
    # exit-0 regardless. Taking those exit codes as the seam's booleans is the
    # rc-0 no-op (#228) reached through Python instead of a shell fallback, and
    # it is what ``tui.launch.return_attached_client`` would consume to decide a
    # human has been handed their terminal back.
    #
    # So the exit code is discarded and the effect is measured: count the
    # clients attached to this session before and after the verb, and answer on
    # the DELTA. Never on an absolute count — a ``-t`` read against a session
    # whose server is gone answers from whichever server the fallback picks, so
    # "zero attached" on its own proves nothing (#315). A drop cannot be
    # manufactured that way: it needs two successful reads of the same session,
    # the first of them nonzero.
    #
    # Unobservable answers False, never a vacuous True. For the detach that is
    # safe in both directions — the caller's UNREACHABLE and RETURNED agree that
    # nobody is left at this terminal, and the only cost is the parked trailer
    # re-issuing a detach that no-ops. For the switch it is also the truth
    # today: the switch leg is inert at 3.3.7 (psmux/psmux#483), so no client
    # ever moves. When upstream lands that fix the probe reports it without a
    # code change here — which is why this measures rather than hardcoding.
    #
    # Residue: a switch whose target pane lives in THIS session moves the client
    # between windows without changing the session's attached count, so it reads
    # as no effect. Reachable only when the return target was recorded from
    # inside a control-session window; the caller then keeps prompting, which is
    # the safe direction.

    def _attached_clients(self, session: str) -> int | None:
        """Clients attached to ``session``, or None when psmux cannot say.

        Self-detecting on purpose: an unsupported format field cannot answer a
        plausible integer, so a psmux build that does not carry
        ``#{session_attached}`` degrades to None instead of a wrong count.
        """
        try:
            proc = self._run(
                ["display-message", "-p", "-t", session, "#{session_attached}"], check=False
            )
        except (subprocess.SubprocessError, OSError):
            return None
        if proc.returncode != 0:
            return None
        text = proc.stdout.strip()
        return int(text) if text.isdigit() else None

    def _client_left(self, verb: list[str]) -> bool:
        """Run a client verb and answer whether a client left this session."""
        session = self.current_session()
        if not session:
            # Not inside a pane (or the probe failed): there is no "this
            # session" to measure against, and no client of ours to move.
            return False
        before = self._attached_clients(session)
        try:
            self._run(verb, check=False)
        except (subprocess.SubprocessError, OSError):
            return False
        after = self._attached_clients(session)
        if before is None or after is None:
            return False
        return before > 0 and after < before

    def detach_client(self) -> bool:
        return self._client_left(["detach-client"])

    def switch_client(self, target: str, last_fallback: bool = False) -> bool:
        if self._client_left(["switch-client", "-t", target]):
            return True
        return last_fallback and self._client_left(["switch-client", "-l"])

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
