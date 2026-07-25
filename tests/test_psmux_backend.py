"""psmux backend unit tests.

Deterministic: the single subprocess seam (``tmux_base.subprocess.run``) is
mocked, so these run on any OS. Shell source shipped as ``-EncodedCommand`` is
decoded back (base64 → UTF-16LE) to assert its composition.
"""

import base64
import os
import subprocess

import pytest

from bmad_loop.adapters import psmux_backend, tmux_base
from bmad_loop.adapters.multiplexer import MultiplexerError, get_multiplexer
from bmad_loop.adapters.psmux_backend import PsmuxMultiplexer


class _RecordRun:
    """Stand-in for subprocess.run that records every spawn's argv and kwargs."""

    def __init__(self, returncode: int = 0, stderr: str = "", stdout: str = ""):
        self.calls: list[tuple[list, dict]] = []
        self.returncode = returncode
        self.stderr = stderr
        self.stdout = stdout

    @property
    def argv(self):
        return self.calls[-1][0]

    @property
    def kwargs(self):
        return self.calls[-1][1]

    def __call__(self, argv, **kwargs):
        self.calls.append((argv, kwargs))
        return subprocess.CompletedProcess(
            argv, self.returncode, stdout=self.stdout, stderr=self.stderr
        )


@pytest.fixture
def rec(monkeypatch):
    recorder = _RecordRun()
    monkeypatch.setattr(tmux_base.subprocess, "run", recorder)
    return recorder


def _decode(encoded: str) -> str:
    return base64.b64decode(encoded).decode("utf-16-le")


def _pwsh_payload(argv: list) -> str:
    """Assert the trailing args are a pwsh -EncodedCommand launch; return the
    decoded shell source."""
    assert argv[-4:-1] == ["pwsh", "-NoProfile", "-EncodedCommand"]
    return _decode(argv[-1])


# ------------------------------------------------------------------ decoding


def test_run_decodes_utf8_with_backslashreplace(rec):
    PsmuxMultiplexer()._run(["list-windows"])
    assert rec.kwargs["encoding"] == "utf-8"
    assert rec.kwargs["errors"] == "backslashreplace"


# ---------------------------------------------------------------- new_window


def test_new_window_ships_env_and_command_as_encoded_pwsh(rec, tmp_path):
    PsmuxMultiplexer().new_window(
        "s", "n", tmp_path, {"A": "x y", "B": "it's"}, "claude -p 'hi there'"
    )

    # the tmux-family scaffolding is the base's, spawned via the psmux binary,
    # with no -e flags (psmux drops them)
    assert rec.argv[:12] == [
        "psmux",
        "new-window",
        "-t",
        "=s:",
        "-n",
        "n",
        "-c",
        str(tmp_path),
        "-P",
        "-F",
        "#{window_id}",
        "pwsh",
    ]
    assert "-e" not in rec.argv

    source = _pwsh_payload(rec.argv)
    # teammate-clear prelude, then env prelude, then the call-operator command
    assert source.index("Remove-Item") < source.index("$env:A")
    assert "'CLAUDE_CODE_*'" in source
    assert "'CLAUDECODE*'" in source
    assert "'PSMUX_CLAUDE_TEAMMATE_MODE'" in source
    assert "$env:A = 'x y'; " in source
    assert "$env:B = 'it''s'; " in source
    assert source.endswith("& 'claude' '-p' 'hi there'")


def test_new_window_rejects_invalid_env_name(rec, tmp_path):
    mux = PsmuxMultiplexer()
    for bad in ("A-B", "1X", "A B", "", "SAFE\n"):
        with pytest.raises(MultiplexerError):
            mux.new_window("s", "n", tmp_path, {bad: "v"}, "cmd")
    assert rec.calls == []  # rejected before any spawn


def test_new_window_rejects_malformed_command(rec, tmp_path):
    mux = PsmuxMultiplexer()
    # unbalanced quote (shlex can't split it) and an empty command (`& ` alone
    # is a pwsh parse error) both fail as the seam type, before any spawn
    for bad in ("claude -p 'x", "", "   "):
        with pytest.raises(MultiplexerError):
            mux.new_window("s", "n", tmp_path, {}, bad)
    assert rec.calls == []


def test_new_parked_window_rejects_empty_argv(rec, tmp_path):
    with pytest.raises(MultiplexerError):
        PsmuxMultiplexer().new_parked_window("s", "n", tmp_path, [], "")
    assert rec.calls == []


def test_new_window_literalizes_shell_operators(rec, tmp_path):
    # the seam's `command` is a POSIX-quoted argv join, not a shell line: pwsh
    # re-quoting turns would-be operators into literal arguments
    PsmuxMultiplexer().new_window("s", "n", tmp_path, {}, "a && b | c")
    source = _pwsh_payload(rec.argv)
    assert source.endswith("& 'a' '&&' 'b' '|' 'c'")


def test_new_window_env_values_stay_inert_literals(rec, tmp_path):
    # Env values are attacker-shaped strings from the caller's perspective:
    # pwsh must receive each one as a single-quoted literal with no room for
    # interpolation, subexpression, or quote breakout.
    hostile = {
        "A": "it's",
        "B": "line1\nline2",
        "C": "$(Remove-Item x)",
        "D": "`; Write-Host pwned",
        "E": "",
        "F": "'; Remove-Item -Recurse 'C:\\ #",
    }
    PsmuxMultiplexer().new_window("s", "n", tmp_path, hostile, "prog")
    source = _pwsh_payload(rec.argv)
    for key, value in hostile.items():
        assert f"$env:{key} = '{value.replace(chr(39), chr(39) * 2)}'; " in source
    # with doubled quotes collapsed, every remaining quote must pair up — an
    # odd count means some value broke out of its literal
    assert source.replace("''", "").count("'") % 2 == 0


# ------------------------------------------- session-qualified window ids (#254)
# psmux mints window ids per server (one server per session), so a bare `@N`
# replayed as a `-t` target routes by the caller's $TMUX — the wrong server
# from a ctl pane. new_window and list_window_ids must therefore emit the
# `session:@N` form symmetrically (psmux/psmux#483), or window_alive's
# membership check reads every window as dead.


def _window_fake(monkeypatch, new_window_id: str = "@2\n", listed: str = "@1\n@2\n"):
    """Script new-window to print an id and list-windows to list ids."""

    def fake(argv, **kwargs):
        out = new_window_id if argv[1] == "new-window" else listed
        return subprocess.CompletedProcess(argv, 0, stdout=out, stderr="")

    monkeypatch.setattr(tmux_base.subprocess, "run", fake)


def test_new_window_returns_session_qualified_id(monkeypatch, tmp_path):
    _window_fake(monkeypatch)
    assert PsmuxMultiplexer().new_window("s", "n", tmp_path, {}, "prog") == "s:@2"


def test_list_window_ids_returns_session_qualified_ids(monkeypatch):
    _window_fake(monkeypatch)
    assert PsmuxMultiplexer().list_window_ids("s") == ["s:@1", "s:@2"]


def test_qualification_degrades_to_bare_on_colon_session(monkeypatch, tmp_path):
    # A `:` in the session name would split the target at the wrong colon on
    # replay — both methods degrade to the bare id identically (the #221 rule).
    _window_fake(monkeypatch)
    mux = PsmuxMultiplexer()
    assert mux.new_window("a:b", "n", tmp_path, {}, "prog") == "@2"
    assert mux.list_window_ids("a:b") == ["@1", "@2"]


def test_new_window_falsy_id_passes_through_unqualified(monkeypatch, tmp_path):
    # An empty minted id is a failure sentinel, not a target — qualifying it
    # would forge "s:" out of nothing.
    _window_fake(monkeypatch, new_window_id="")
    assert PsmuxMultiplexer().new_window("s", "n", tmp_path, {}, "prog") == ""


def test_window_alive_accepts_new_window_id(monkeypatch, tmp_path):
    # Symmetry is the whole contract: the id new_window mints must be found by
    # list_window_ids, or the engine's liveness probe declares an instant crash.
    _window_fake(monkeypatch)
    mux = PsmuxMultiplexer()
    assert mux.window_alive("s", mux.new_window("s", "n", tmp_path, {}, "prog")) is True


def test_qualified_id_reaches_the_pipe_pane_target(rec, tmp_path):
    # #254 is about the `-t` argv, not the return value: the consumer must replay
    # the minted id verbatim, and this backend's pipe_pane override (the one verb
    # it reimplements) must not re-derive a bare target of its own.
    rec.stdout = "@2\n"
    mux = PsmuxMultiplexer()
    mux.pipe_pane(mux.new_window("s", "n", tmp_path, {}, "prog"), tmp_path / "run.log")
    assert rec.argv[1:4] == ["pipe-pane", "-t", "s:@2"]


def test_list_window_ids_transport_failure_still_raises(monkeypatch):
    # Qualification must not soften the liveness contract: a transport failure
    # raises rather than answering [] (which would read as "session crashed").
    def timeout(*_a, **_k):
        raise subprocess.TimeoutExpired(["psmux"], 30)

    monkeypatch.setattr(tmux_base.subprocess, "run", timeout)
    with pytest.raises(MultiplexerError):
        PsmuxMultiplexer().list_window_ids("s")


# --------------------------------------------------------------- new_session


def test_new_session_bypasses_nesting_guard(rec, monkeypatch, tmp_path):
    monkeypatch.setenv("CLAUDE_CODE_SSE_PORT", "1234")
    monkeypatch.setenv("CLAUDECODE", "1")
    monkeypatch.setenv("PSMUX_CLAUDE_TEAMMATE_MODE", "tmux")
    monkeypatch.setenv("Claude_Code_Mixed", "mixed")
    before = dict(os.environ)
    PsmuxMultiplexer().new_session("s", tmp_path, cols=80, lines=24)

    create_argv, create_kwargs = rec.calls[0]
    assert create_argv == [
        "psmux",
        "new-session",
        "-d",
        "-s",
        "s",
        "-c",
        str(tmp_path),
        "-x",
        "80",
        "-y",
        "24",
    ]
    # the no-op belt: create is verified by a has-session probe afterwards
    assert rec.argv == ["psmux", "has-session", "-t", "=s"]
    assert create_kwargs["env"]["PSMUX_ALLOW_NESTING"] == "1"
    # the claude session vars are scrubbed from the create env (the psmux server
    # this call may cold-start would otherwise hand them to every window)
    assert "CLAUDE_CODE_SSE_PORT" not in create_kwargs["env"]
    assert "CLAUDECODE" not in create_kwargs["env"]
    assert "PSMUX_CLAUDE_TEAMMATE_MODE" not in create_kwargs["env"]
    assert "Claude_Code_Mixed" not in create_kwargs["env"]
    # the bypass var and the scrub are confined to the child spawn
    assert dict(os.environ) == before


def test_new_session_omits_geometry_when_unset(rec, tmp_path):
    PsmuxMultiplexer().new_session("s", tmp_path)
    create_argv = rec.calls[0][0]
    assert "-x" not in create_argv
    assert "-y" not in create_argv


def test_new_session_exit_zero_noop_raises(monkeypatch, tmp_path):
    # The nesting guard's historical failure mode: new-session exits 0 having
    # created nothing. The belt verifies and blames session creation directly.
    def fake(argv, **kwargs):
        rc = 1 if argv[1] == "has-session" else 0
        return subprocess.CompletedProcess(argv, rc, stdout="", stderr="")

    monkeypatch.setattr(tmux_base.subprocess, "run", fake)
    with pytest.raises(MultiplexerError, match="was not created"):
        PsmuxMultiplexer().new_session("s", tmp_path)


def test_new_session_failure_raises_multiplexer_error(monkeypatch, tmp_path):
    monkeypatch.setattr(tmux_base.subprocess, "run", _RecordRun(returncode=1, stderr="boom"))
    with pytest.raises(MultiplexerError):
        PsmuxMultiplexer().new_session("s", tmp_path)

    def timeout(*_a, **_k):
        raise subprocess.TimeoutExpired(["tmux"], 30)

    monkeypatch.setattr(tmux_base.subprocess, "run", timeout)
    with pytest.raises(MultiplexerError):
        PsmuxMultiplexer().new_session("s", tmp_path)


# --------------------------------------------------------------- kill_session


def test_kill_session_uses_plain_target(rec, monkeypatch):
    # strict which-stub: the guard must probe the psmux binary, not a
    # copy-pasted "tmux"
    monkeypatch.setattr(
        psmux_backend.shutil,
        "which",
        lambda name: "C:\\bin\\psmux.exe" if name == "psmux" else None,
    )
    PsmuxMultiplexer().kill_session("s")
    assert rec.argv == ["psmux", "kill-session", "-t", "s"]  # no `=` — psmux ignores it


def test_kill_session_no_binary_no_spawn(rec, monkeypatch):
    monkeypatch.setattr(psmux_backend.shutil, "which", lambda _name: None)
    PsmuxMultiplexer().kill_session("s")
    assert rec.calls == []


# ------------------------------------------------------- return target (#221)
# psmux runs one server per session, so the parked-window return target must be
# session-qualified: a bare %N replayed from the control session is at best
# unresolvable, at worst collides with a real control-session pane
# (psmux/psmux#483). The seam default (bare pane id) stays correct for tmux —
# whose switch-client rejects the qualified form — so the composition lives in
# this backend's override.


def _probe_fake(monkeypatch, answers: dict[str, tuple[int, str]]):
    """Script the display-message probes: fmt -> (returncode, stdout)."""

    def fake(argv, **kwargs):
        rc, out = answers[argv[-1]]
        return subprocess.CompletedProcess(argv, rc, stdout=out, stderr="")

    monkeypatch.setenv("TMUX", "/tmp/psmux-1000/default,123,0")  # inside psmux
    monkeypatch.setattr(tmux_base.subprocess, "run", fake)


def test_return_target_session_qualified(monkeypatch):
    _probe_fake(monkeypatch, {"#{pane_id}": (0, "%9\n"), "#{session_name}": (0, "main\n")})
    assert PsmuxMultiplexer().current_return_target() == "=main:%9"


def test_return_target_none_outside_mux(monkeypatch):
    monkeypatch.delenv("TMUX", raising=False)
    monkeypatch.setattr(tmux_base.subprocess, "run", _RecordRun())
    assert PsmuxMultiplexer().current_return_target() is None


def test_return_target_none_on_empty_pane(monkeypatch):
    _probe_fake(monkeypatch, {"#{pane_id}": (0, "\n"), "#{session_name}": (0, "main\n")})
    assert PsmuxMultiplexer().current_return_target() is None


def test_return_target_bare_pane_when_session_probe_fails(monkeypatch):
    # A resolvable own pane means we ARE inside the multiplexer; a failed
    # session-name probe degrades to the bare pane id, never to None (which
    # callers would record as "detach" and strand the client).
    _probe_fake(monkeypatch, {"#{pane_id}": (0, "%9\n"), "#{session_name}": (1, "")})
    assert PsmuxMultiplexer().current_return_target() == "%9"


def test_return_target_bare_pane_on_empty_session_name(monkeypatch):
    # rc-0 empty stdout from the session probe must degrade the same way a
    # failed probe does — a "=:%9" target would misparse at replay.
    _probe_fake(monkeypatch, {"#{pane_id}": (0, "%9\n"), "#{session_name}": (0, "\n")})
    assert PsmuxMultiplexer().current_return_target() == "%9"


def test_return_target_bare_pane_on_unqualifiable_session_name(monkeypatch):
    # A session name the `=session:%N` grammar cannot carry (a `:` would split
    # at the wrong colon on replay) degrades to the bare id too.
    _probe_fake(monkeypatch, {"#{pane_id}": (0, "%9\n"), "#{session_name}": (0, "a:b\n")})
    assert PsmuxMultiplexer().current_return_target() == "%9"


# ------------------------------------------------------------- parked window


def test_new_parked_window_composes_pwsh_source(rec, tmp_path):
    PsmuxMultiplexer().new_parked_window("s", "n", tmp_path, ["claude", "--resume"], "%3")

    source = _pwsh_payload(rec.argv)
    prefix_end = source.index("& 'claude' '--resume'")
    assert "Remove-Item" in source[:prefix_end]  # teammate-clear prelude first
    # A not-recognized command leaves $LASTEXITCODE unset but the source keeps
    # running, so the banner needs a fallback code that also works before pwsh 7.
    assert (
        "& 'claude' '--resume'; "
        "$ec = if ($null -eq $LASTEXITCODE) { 1 } else { $LASTEXITCODE }; "
        'Write-Host "[bmad-loop exited $ec — press enter]"; Read-Host; ' in source
    )
    # trailer: same tmux-family verbs as the POSIX one, pwsh control flow,
    # issued through the psmux binary
    assert "$ret = psmux show-options -wqv '%3' 2>$null; " in source
    assert "if ($ret -eq 'detach') { psmux detach-client 2>$null }" in source
    assert "psmux switch-client -t $ret 2>$null" in source
    assert "psmux switch-client -l 2>$null" in source


# ------------------------------------------------------------------ pipe_pane


def test_pipe_pane_ships_positional_sidecar_sink(rec, tmp_path):
    log = tmp_path / "win's.log"
    PsmuxMultiplexer().pipe_pane("@1", log)

    assert rec.argv[:5] == ["psmux", "pipe-pane", "-t", "@1", "-o"]
    # psmux strips every dash-flag token from the piped command, so the sink
    # must be a purely positional launch of a sidecar script
    sidecar = tmp_path / "win's.log.sink.ps1"
    assert rec.argv[5] == f'pwsh "{sidecar}"'
    sink = sidecar.read_text(encoding="utf-8")
    # byte-exact raw stream copy (no console decode / re-encode / CRLF mangling),
    # flushed per chunk so the live tail sees bytes incrementally
    quoted = str(log).replace(chr(39), chr(39) * 2)
    assert f"[System.IO.File]::Open('{quoted}', 'Append', 'Write', 'Read')" in sink
    assert "$in.Read($buf, 0, $buf.Length)" in sink
    assert "$out.Flush()" in sink


def test_pipe_pane_swallows_failure_with_warning(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(tmux_base.subprocess, "run", _RecordRun(returncode=1, stderr="gone"))
    assert PsmuxMultiplexer().pipe_pane("@1", tmp_path / "log") is None
    assert "pipe-pane log capture failed" in capsys.readouterr().err


def test_pipe_pane_sidecar_write_failure_warns_without_spawning(rec, capsys, tmp_path):
    # An unwritable sidecar path (missing log dir) must warn and skip the psmux
    # call — never raise
    assert PsmuxMultiplexer().pipe_pane("@1", tmp_path / "absent" / "log") is None
    assert rec.calls == []
    assert "pipe-pane log capture failed" in capsys.readouterr().err


@pytest.mark.parametrize("syntax", ["$name", "`name"])
def test_pipe_pane_rejects_interpolating_sidecar_path(rec, capsys, tmp_path, syntax):
    log = tmp_path / f"{syntax}.log"
    assert PsmuxMultiplexer().pipe_pane("@1", log) is None
    assert rec.calls == []
    assert not log.with_name(log.name + ".sink.ps1").exists()
    assert "PowerShell interpolation syntax" in capsys.readouterr().err


# ------------------------------------------------------------------ selection


def test_available_requires_psmux_pwsh_and_supported_version(monkeypatch):
    # Only psmux + pwsh may be probed — a tmux drop-in is deliberately not
    # required, so a which() stub answering for anything else must not matter.
    monkeypatch.setattr(
        psmux_backend.shutil,
        "which",
        lambda name: f"C:\\bin\\{name}.exe" if name in ("psmux", "pwsh") else None,
    )
    monkeypatch.setattr(PsmuxMultiplexer, "version", lambda self: "tmux 3.3.7")
    assert PsmuxMultiplexer().available() is True

    # 3.3.6 and older force-kill recycled PIDs on teardown — unusable
    monkeypatch.setattr(PsmuxMultiplexer, "version", lambda self: "tmux 3.3.6")
    assert PsmuxMultiplexer().available() is False

    monkeypatch.setattr(PsmuxMultiplexer, "version", lambda self: "tmux 3.4.0")
    assert PsmuxMultiplexer().available() is True

    # multi-digit segments compare numerically, not lexicographically
    monkeypatch.setattr(PsmuxMultiplexer, "version", lambda self: "tmux 3.10.0")
    assert PsmuxMultiplexer().available() is True
    monkeypatch.setattr(PsmuxMultiplexer, "version", lambda self: "tmux 10.0")
    assert PsmuxMultiplexer().available() is True

    # a suffixed newer release still clears the strictly-greater gate
    monkeypatch.setattr(PsmuxMultiplexer, "version", lambda self: "tmux 3.3.7-rc0")
    assert PsmuxMultiplexer().available() is True

    # a two-part compat version (tmux's own format) reads as patch 0
    monkeypatch.setattr(PsmuxMultiplexer, "version", lambda self: "tmux 3.4")
    assert PsmuxMultiplexer().available() is True

    monkeypatch.setattr(PsmuxMultiplexer, "version", lambda self: "tmux 3.3")
    assert PsmuxMultiplexer().available() is False

    # unidentifiable version fails closed
    for garbled in (None, "", "tmux next-3.4", "psmux 9.9.9"):
        monkeypatch.setattr(PsmuxMultiplexer, "version", lambda self, v=garbled: v)
        assert PsmuxMultiplexer().available() is False


def test_available_composes_real_version_probe(monkeypatch):
    # End-to-end through the real version() seam (no version() stub): the gate
    # must survive `psmux -V` composition, including trailing-newline stripping.
    monkeypatch.setattr(
        psmux_backend.shutil,
        "which",
        lambda name: f"C:\\bin\\{name}.exe" if name in ("psmux", "pwsh") else None,
    )
    monkeypatch.setattr(tmux_base.shutil, "which", lambda name: f"C:\\bin\\{name}.exe")
    rec = _RecordRun(stdout="tmux 3.3.7\n")
    monkeypatch.setattr(tmux_base.subprocess, "run", rec)
    assert PsmuxMultiplexer().available() is True
    assert rec.argv == ["psmux", "-V"]


def test_available_caches_version_gate_per_instance(monkeypatch):
    monkeypatch.setattr(
        psmux_backend.shutil,
        "which",
        lambda name: f"C:\\bin\\{name}.exe" if name in ("psmux", "pwsh") else None,
    )
    calls = 0

    def probe(self):
        nonlocal calls
        calls += 1
        return "tmux 3.3.7"

    monkeypatch.setattr(PsmuxMultiplexer, "version", probe)
    mux = PsmuxMultiplexer()
    assert mux.available() is True
    assert mux.available() is True
    assert calls == 1  # repeated polls must not respawn the version query


def test_available_missing_binary_short_circuits_version_probe(monkeypatch):
    def no_probe(self):
        raise AssertionError("version() must not spawn when a binary is missing")

    monkeypatch.setattr(PsmuxMultiplexer, "version", no_probe)
    for absent in ("pwsh", "psmux"):
        monkeypatch.setattr(
            psmux_backend.shutil, "which", lambda name, a=absent: None if name == a else "x"
        )
        assert PsmuxMultiplexer().available() is False


def test_registry_selects_psmux_when_forced(monkeypatch):
    monkeypatch.setenv("BMAD_LOOP_MUX_BACKEND", "psmux")
    get_multiplexer.cache_clear()
    try:
        assert isinstance(get_multiplexer(), PsmuxMultiplexer)
    finally:
        get_multiplexer.cache_clear()  # don't leak the forced pick to other tests
