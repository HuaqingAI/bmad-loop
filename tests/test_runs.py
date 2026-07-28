"""Run-directory helper tests."""

import json
import os
import re
import subprocess
import sys
import tarfile

import pytest
from conftest import escalated_run, git

from bmad_loop import platform_util, runs, verify
from bmad_loop.adapters import tmux_base
from bmad_loop.journal import load_state, save_state
from bmad_loop.model import RunState
from bmad_loop.process_host import ProcessHost


def _make_run(project, run_id, with_state=True):
    run_dir = project / ".bmad-loop" / "runs" / run_id
    run_dir.mkdir(parents=True)
    if with_state:
        (run_dir / "state.json").write_text("{}")
    return run_dir


def _make_state_run(project, run_id, **state_kwargs):
    run_dir = project / ".bmad-loop" / "runs" / run_id
    save_state(
        run_dir,
        RunState(
            run_id=run_id,
            project=str(project),
            started_at="2026-06-11T10:00:00",
            **state_kwargs,
        ),
    )
    return run_dir


def _dead_pid() -> int:
    # A process that exits immediately, cross-platform (POSIX `true` isn't on
    # Windows). The interpreter is always present and on every host.
    proc = subprocess.Popen([sys.executable, "-c", ""])
    proc.wait()
    return proc.pid


class _FakeHost(ProcessHost):
    """A ProcessHost for driving stop_run's escalation deterministically without
    spawning real processes. ``alive`` / ``identity`` may be a value or a zero-arg
    callable (so they can change between the stop-time read and the post-grace
    check). A real subclass on purpose: ``alive_and_ours`` and ``liveness_of``
    are inherited, so these tests exercise the production decision table instead
    of a hand-copied mirror that could silently drift."""

    def __init__(self, *, alive, identity=1.0, on_terminate=None):
        self._alive = alive
        self._identity = identity
        self.on_terminate = on_terminate
        self.terminated: list[int] = []
        self.force_killed: list[int] = []

    def terminate(self, pid):
        self.terminated.append(pid)
        if self.on_terminate is not None:
            self.on_terminate(pid)

    def force_kill(self, pid):
        self.force_killed.append(pid)

    def is_alive(self, pid):
        return self._alive() if callable(self._alive) else self._alive

    def identity(self, pid):
        return self._identity() if callable(self._identity) else self._identity

    def hook_interpreter(self):
        return "python3"


def test_list_run_dirs_sorted_and_filtered(tmp_path):
    _make_run(tmp_path, "20260611-120000-bbbb")
    _make_run(tmp_path, "20260610-090000-aaaa")
    _make_run(tmp_path, "20260612-080000-cccc", with_state=False)  # no state.json
    listed = runs.list_run_dirs(tmp_path)
    assert [d.name for d in listed] == ["20260610-090000-aaaa", "20260611-120000-bbbb"]


def test_list_run_dirs_missing(tmp_path):
    assert runs.list_run_dirs(tmp_path) == []
    assert runs.latest_run_dir(tmp_path) is None


def test_latest_run_dir(tmp_path):
    _make_run(tmp_path, "20260610-090000-aaaa")
    newest = _make_run(tmp_path, "20260611-120000-bbbb")
    assert runs.latest_run_dir(tmp_path) == newest


def test_new_run_id_format():
    assert re.fullmatch(r"\d{8}-\d{6}-[0-9a-f]{4}", runs.new_run_id())


def test_is_valid_run_id_is_identity_for_generated_ids():
    """The validator must accept everything our one legitimate producer emits —
    and those ids must survive both sanitizers byte-for-byte, since a run id is a
    directory name (safe_segment) and a git ref component (safe_ref_segment) at once."""
    for _ in range(20):
        run_id = runs.new_run_id()
        assert runs.is_valid_run_id(run_id)
        assert platform_util.safe_segment(run_id) == run_id
        assert platform_util.safe_ref_segment(run_id) == run_id


@pytest.mark.parametrize("value", ["r1", "RID", "a", "A_b-C9", "x" * platform_util.MAX_SEGMENT])
def test_is_valid_run_id_accepts(value):
    assert runs.is_valid_run_id(value)


@pytest.mark.parametrize(
    "value",
    [
        "",  # empty
        "..",  # traversal
        "../x",
        "..\\x",
        "/etc/passwd",  # posix absolute
        "C:\\windows",  # windows drive-absolute
        "C:rel",  # windows drive-relative
        "a/b",  # posix separator
        "a\\b",  # windows separator
        "-lead",  # leading dash (git porcelain option-lookalike)
        "_lead",  # leading underscore: only [A-Za-z0-9] may start an id
        ".hidden",  # leading dot
        "a.b",  # dot: mangles a multiplexer session name
        "a:b",  # colon: mangles a multiplexer session name, illegal in a git ref
        "a b",  # whitespace
        "a\tb",
        "a\nb",
        "a\x00b",  # control char
        "trailing.",  # windows drops trailing dots
        "trailing ",  # ...and trailing spaces
        'a"b',
        "a<b",
        "a>b",
        "a|b",
        "a?b",
        "a*b",
        "a~b",
        "a^b",
        "a[b",
        "a@{b",
        "CON",  # reserved windows device basenames, any case, with or without ext
        "nul",
        "COM1",
        "x" * (platform_util.MAX_SEGMENT + 1),  # over the segment cap
    ],
)
def test_is_valid_run_id_rejects(value):
    assert not runs.is_valid_run_id(value)


def test_write_pid(tmp_path):
    runs.write_pid(tmp_path)
    tokens = (tmp_path / "engine.pid").read_text().split()
    assert tokens[0] == str(os.getpid())
    # identity is persisted as an optional second token so a reused pid can later be
    # told from our engine; Linux always provides one (via /proc starttime).
    if sys.platform.startswith("linux"):
        assert len(tokens) == 2 and float(tokens[1]) > 0
    elif len(tokens) > 1:
        assert float(tokens[1]) > 0


@pytest.mark.usefixtures("force_tmux_backend")  # attach_argv goes through the seam
def test_attach_argv_outside_tmux(monkeypatch):
    monkeypatch.delenv("TMUX", raising=False)
    assert runs.attach_argv("r1") == ["tmux", "attach", "-t", "=bmad-loop-r1"]


@pytest.mark.usefixtures("force_tmux_backend")  # attach_argv goes through the seam
def test_attach_argv_inside_tmux(monkeypatch):
    monkeypatch.setenv("TMUX", "/tmp/tmux-1000/default,123,0")
    assert runs.attach_argv("r1") == ["tmux", "switch-client", "-t", "=bmad-loop-r1"]


# --------------------------------------------------------- resolution / liveness


def test_run_dir_for_and_is_run(tmp_path):
    run_dir = _make_run(tmp_path, "r1")
    assert runs.run_dir_for(tmp_path, "r1") == run_dir
    assert runs.is_run(run_dir)
    assert not runs.is_run(tmp_path / ".bmad-loop" / "runs" / "nope")


def test_short_ref():
    assert runs.short_ref("20260620-143025-a1b2") == "a1b2"


def test_resolve_run_dir_exact_and_partial(tmp_path):
    target = _make_run(tmp_path, "20260620-143025-a1b2")
    _make_run(tmp_path, "20260619-101010-c3d4")
    # exact full id
    assert runs.resolve_run_dir(tmp_path, "20260620-143025-a1b2") == target
    # full trailing segment
    assert runs.resolve_run_dir(tmp_path, "a1b2") == target
    # prefix of the trailing segment
    assert runs.resolve_run_dir(tmp_path, "a1") == target
    # a longer tail of the id (endswith)
    assert runs.resolve_run_dir(tmp_path, "025-a1b2") == target


def test_resolve_run_dir_no_match(tmp_path):
    _make_run(tmp_path, "20260620-143025-a1b2")
    with pytest.raises(runs.RunRefError, match="no such run: zzzz"):
        runs.resolve_run_dir(tmp_path, "zzzz")


def test_resolve_run_dir_ambiguous(tmp_path):
    _make_run(tmp_path, "20260620-143025-a1b2")
    _make_run(tmp_path, "20260619-101010-a1c9")
    with pytest.raises(runs.RunRefError, match="ambiguous run ref 'a1' matches 2 runs"):
        runs.resolve_run_dir(tmp_path, "a1")


@pytest.mark.parametrize("ref", ["../../outside", "../outside", "a/b", "a\\b"])
def test_resolve_run_dir_never_escapes_the_runs_dir(tmp_path, ref):
    """The exact branch recomposes `project / RUNS_DIR / ref` from the raw ref, so a
    ref carrying separators or `..` must never reach it — otherwise
    `bmad-loop delete ../../x` rmtree's any outside directory that happens to hold a
    state.json. Such refs fall through to partial matching, which can only yield a
    name `list_run_dirs` enumerated, i.e. an immediate child of the runs dir.

    Stated as containment rather than no-match because `a\\b` is one legal directory
    name on POSIX (inside the runs dir, so it legitimately resolves) and a nested
    path on Windows (where it must not)."""
    project = tmp_path / "proj"
    _make_run(project, "20260620-143025-a1b2")
    runs_dir = (project / ".bmad-loop" / "runs").resolve()
    # plant a state.json exactly where the un-gated exact branch would land
    planted = (project / ".bmad-loop" / "runs" / ref).resolve()
    planted.mkdir(parents=True, exist_ok=True)
    (planted / "state.json").write_text("{}")

    try:
        got = runs.resolve_run_dir(project, ref)
    except runs.RunRefError as e:
        assert "no such run" in str(e)
    else:
        assert got.resolve().parent == runs_dir  # an enumerated run dir, never an escape
    assert (planted / "state.json").is_file()  # never consumed as a run


def test_resolve_run_dir_absolute_ref_never_escapes(tmp_path):
    """`run_dir_for(project, "/abs")` is `Path("/abs")` — `/`-join discards the
    project prefix entirely, so an absolute ref escapes without needing a `..`."""
    project = tmp_path / "proj"
    _make_run(project, "20260620-143025-a1b2")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "state.json").write_text("{}")

    with pytest.raises(runs.RunRefError, match="no such run"):
        runs.resolve_run_dir(project, str(outside))
    assert (outside / "state.json").is_file()


def test_resolve_run_dir_exact_wins_over_ambiguity(tmp_path):
    # An exact id resolves even when another run's id ends with it (which would
    # otherwise be an ambiguous partial match).
    exact = _make_run(tmp_path, "20260620-143025-a1b2")
    _make_run(tmp_path, "20260101-000000-20260620-143025-a1b2")  # ends with the exact id
    assert runs.resolve_run_dir(tmp_path, "20260620-143025-a1b2") == exact


def test_read_pid_missing_and_garbage(tmp_path):
    run_dir = _make_run(tmp_path, "r1")
    assert runs.read_pid(run_dir) is None
    (run_dir / "engine.pid").write_text("not-a-pid")
    assert runs.read_pid(run_dir) is None
    (run_dir / "engine.pid").write_text("4242")
    assert runs.read_pid(run_dir) == 4242


def test_engine_alive(tmp_path):
    run_dir = _make_run(tmp_path, "r1")
    assert runs.engine_alive(run_dir) is False  # no pid file
    runs.write_pid(run_dir)  # this test process: alive
    assert runs.engine_alive(run_dir) is True
    (run_dir / "engine.pid").write_text(str(_dead_pid()))
    assert runs.engine_alive(run_dir) is False


def test_read_pid_identity_forms(tmp_path):
    run_dir = _make_run(tmp_path, "r1")
    assert runs.read_pid_identity(run_dir) == (None, None)  # missing
    (run_dir / "engine.pid").write_text("4242")  # legacy: pid only
    assert runs.read_pid_identity(run_dir) == (4242, None)
    (run_dir / "engine.pid").write_text("4242 678.5")  # pid + identity
    assert runs.read_pid_identity(run_dir) == (4242, 678.5)
    (run_dir / "engine.pid").write_text("not-a-pid 1.0")  # unparseable pid
    assert runs.read_pid_identity(run_dir) == (None, None)


def test_engine_liveness(tmp_path, monkeypatch):
    run_dir = _make_run(tmp_path, "r1")
    assert runs.engine_liveness(run_dir) == "dead"  # no pid file → nothing to gate on

    (run_dir / "engine.pid").write_text("4242 100.0")

    def use(host):
        monkeypatch.setattr(runs, "get_process_host", lambda: host)

    use(_FakeHost(alive=True, identity=100.0))
    assert runs.engine_liveness(run_dir) == "alive"  # identity matches

    use(_FakeHost(alive=True, identity=999.0))
    assert runs.engine_liveness(run_dir) == "dead"  # reused pid: identity differs

    # live pid whose identity is unreadable (win32 ERROR_ACCESS_DENIED) → unknown, not dead
    use(_FakeHost(alive=True, identity=None))
    assert runs.engine_liveness(run_dir) == "unknown"

    class _Boom:  # an unexpected probe failure degrades to unknown, never a false dead
        def liveness_of(self, pid, identity):
            raise RuntimeError("probe blew up")

    use(_Boom())
    assert runs.engine_liveness(run_dir) == "unknown"

    # A misconfigured host (get_process_host itself raising) is a hard error, not a
    # flaky per-pid probe — it must propagate, never mask as 'unknown'.
    from bmad_loop.process_host import ProcessHostError

    def _boom_host():
        raise ProcessHostError("BMAD_LOOP_PROCESS_HOST matches no registered host")

    monkeypatch.setattr(runs, "get_process_host", _boom_host)
    with pytest.raises(ProcessHostError):
        runs.engine_liveness(run_dir)


@pytest.mark.parametrize("identity_token", ["garbage", "nan", "inf", "-inf"])
def test_engine_alive_malformed_identity_fails_closed(tmp_path, monkeypatch, identity_token):
    # Two tokens means "identity was intended"; if token 2 is corrupt, do not
    # degrade to legacy bare-existence liveness and report a reused pid as alive.
    run_dir = _make_run(tmp_path, "r1")
    (run_dir / "engine.pid").write_text(f"4242 {identity_token}")
    monkeypatch.setattr(runs, "get_process_host", lambda: _FakeHost(alive=True, identity=123.0))
    assert runs.engine_alive(run_dir) is False


def test_engine_alive_reused_pid_reads_dead(tmp_path, monkeypatch):
    # A stranger inherited the recorded pid: identity no longer matches → dead.
    run_dir = _make_run(tmp_path, "r1")
    (run_dir / "engine.pid").write_text("4242 123.0")
    monkeypatch.setattr(runs, "get_process_host", lambda: _FakeHost(alive=True, identity=999.0))
    assert runs.engine_alive(run_dir) is False
    monkeypatch.setattr(runs, "get_process_host", lambda: _FakeHost(alive=True, identity=123.0))
    assert runs.engine_alive(run_dir) is True


def test_engine_alive_legacy_pid_degrades_to_existence(tmp_path, monkeypatch):
    # A legacy pid file (no identity token) can only fall back to bare existence.
    run_dir = _make_run(tmp_path, "r1")
    (run_dir / "engine.pid").write_text("4242")
    monkeypatch.setattr(runs, "get_process_host", lambda: _FakeHost(alive=True))
    assert runs.engine_alive(run_dir) is True
    monkeypatch.setattr(runs, "get_process_host", lambda: _FakeHost(alive=False))
    assert runs.engine_alive(run_dir) is False


# ---------------------------------------------------------------- stop / delete


def test_stop_run_already_finished(tmp_path, monkeypatch):
    monkeypatch.setattr(runs, "kill_session", lambda _rid: None)
    run_dir = _make_state_run(tmp_path, "r1", finished=True)
    assert runs.stop_run(run_dir) is False
    assert load_state(run_dir).stopped is False


def test_stop_run_no_pid_falls_back_to_mark(tmp_path, monkeypatch):
    killed = []
    monkeypatch.setattr(runs, "kill_session", lambda rid: killed.append(rid))
    run_dir = _make_state_run(tmp_path, "r1")  # no engine.pid -> legacy/dead
    assert runs.stop_run(run_dir) is True
    assert load_state(run_dir).stopped is True
    assert killed == ["r1"]
    journal = (run_dir / "journal.jsonl").read_text()
    assert "run-stop" in journal and '"fallback": true' in journal


def test_stop_run_dead_pid_falls_back(tmp_path, monkeypatch):
    monkeypatch.setattr(runs, "kill_session", lambda _rid: None)
    run_dir = _make_state_run(tmp_path, "r1")
    (run_dir / "engine.pid").write_text(str(_dead_pid()))
    assert runs.stop_run(run_dir) is True
    assert load_state(run_dir).stopped is True


def test_stop_run_signals_live_process(tmp_path, monkeypatch):
    monkeypatch.setattr(runs, "kill_session", lambda _rid: None)
    run_dir = _make_state_run(tmp_path, "r1")
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        (run_dir / "engine.pid").write_text(str(proc.pid))
        assert runs.stop_run(run_dir) is True
        # the process received SIGTERM and is gone
        assert proc.poll() is not None or proc.wait(timeout=5) is not None
        assert load_state(run_dir).stopped is True
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=10)


def test_stop_run_respects_engine_written_stopped(tmp_path, monkeypatch):
    """When a live engine exits having already marked the run stopped, stop_run
    trusts it and does not re-journal a fallback entry."""
    monkeypatch.setattr(runs, "kill_session", lambda _rid: None)
    run_dir = _make_state_run(tmp_path, "r1")
    (run_dir / "engine.pid").write_text("4242")

    def _mark_stopped(_pid):
        # emulate the engine handler marking stopped, then dying on SIGTERM
        st = load_state(run_dir)
        st.stopped = True
        save_state(run_dir, st)

    host = _FakeHost(alive=False, on_terminate=_mark_stopped)
    monkeypatch.setattr(runs, "get_process_host", lambda: host)
    assert runs.stop_run(run_dir) is True
    assert load_state(run_dir).stopped is True
    assert host.force_killed == []  # exited gracefully — no escalation
    # trusted the engine: no fallback journal entry written
    journal = run_dir / "journal.jsonl"
    assert not journal.exists() or "fallback" not in journal.read_text()


def test_stop_run_force_kills_wedged_engine(tmp_path, monkeypatch):
    """An engine that ignores SIGTERM past the grace window is force-killed, then
    marked stopped — as long as its pid identity still matches what we recorded."""
    monkeypatch.setattr(runs, "kill_session", lambda _rid: None)
    monkeypatch.setattr(runs, "_STOP_WAIT_S", 0.05)
    monkeypatch.setattr(runs, "_STOP_POLL_S", 0.01)
    run_dir = _make_state_run(tmp_path, "r1")
    (run_dir / "engine.pid").write_text("4242 123.0")  # persisted identity

    host = _FakeHost(alive=True, identity=123.0)  # never exits, identity stable
    monkeypatch.setattr(runs, "get_process_host", lambda: host)
    assert runs.stop_run(run_dir) is True
    assert host.force_killed == [4242]
    assert load_state(run_dir).stopped is True


def test_stop_run_force_kills_wedged_legacy_engine(tmp_path, monkeypatch):
    """A legacy pid file (no persisted identity) can still force-kill a wedged
    engine: the forced path falls back to a stop-time identity sample (today's
    behavior) rather than refusing outright — no capability regression for
    pre-upgrade runs."""
    monkeypatch.setattr(runs, "kill_session", lambda _rid: None)
    monkeypatch.setattr(runs, "_STOP_WAIT_S", 0.05)
    monkeypatch.setattr(runs, "_STOP_POLL_S", 0.01)
    run_dir = _make_state_run(tmp_path, "r1")
    (run_dir / "engine.pid").write_text("4242")  # legacy: pid only, no identity token

    host = _FakeHost(alive=True, identity=555.0)  # never exits, identity stable
    monkeypatch.setattr(runs, "get_process_host", lambda: host)
    assert runs.stop_run(run_dir) is True
    assert host.force_killed == [4242]
    assert load_state(run_dir).stopped is True


def test_stop_run_refuses_force_kill_on_identity_mismatch(tmp_path, monkeypatch):
    """If the pid is still 'alive' but its identity changed during the grace window
    (possible pid reuse), refuse to force-kill and raise StopRunError instead."""
    monkeypatch.setattr(runs, "kill_session", lambda _rid: None)
    monkeypatch.setattr(runs, "_STOP_WAIT_S", 0.05)
    monkeypatch.setattr(runs, "_STOP_POLL_S", 0.01)
    run_dir = _make_state_run(tmp_path, "r1")
    (run_dir / "engine.pid").write_text("4242 123.0")  # persisted identity at run start

    # matches the persisted identity at stop entry, then changes before the
    # post-grace force-kill check (pid reused mid-grace).
    identities = iter([123.0, 999.0])
    host = _FakeHost(alive=True, identity=lambda: next(identities))
    monkeypatch.setattr(runs, "get_process_host", lambda: host)
    with pytest.raises(runs.StopRunError):
        runs.stop_run(run_dir)
    assert host.force_killed == []


def test_stop_run_refuses_force_kill_without_identity(tmp_path, monkeypatch):
    """On a platform that can't provide an identity (None), a wedged engine can't
    be safely force-killed — raise StopRunError rather than risk a reused pid."""
    monkeypatch.setattr(runs, "kill_session", lambda _rid: None)
    monkeypatch.setattr(runs, "_STOP_WAIT_S", 0.05)
    monkeypatch.setattr(runs, "_STOP_POLL_S", 0.01)
    run_dir = _make_state_run(tmp_path, "r1")
    (run_dir / "engine.pid").write_text("4242")

    host = _FakeHost(alive=True, identity=None)
    monkeypatch.setattr(runs, "get_process_host", lambda: host)
    with pytest.raises(runs.StopRunError):
        runs.stop_run(run_dir)
    assert host.force_killed == []


def test_stop_run_clean_stop_on_pre_stop_pid_reuse(tmp_path, monkeypatch):
    """If the recorded pid was reused by an unrelated process before stop_run
    ran, don't signal the stranger — fall back to a clean mark-stopped, with no
    StopRunError and no terminate/force-kill."""
    killed = []
    monkeypatch.setattr(runs, "kill_session", lambda rid: killed.append(rid))
    run_dir = _make_state_run(tmp_path, "r1")
    (run_dir / "engine.pid").write_text("4242 123.0")  # recorded identity 123.0

    host = _FakeHost(alive=True, identity=999.0)  # alive, but identity differs → reused
    monkeypatch.setattr(runs, "get_process_host", lambda: host)
    assert runs.stop_run(run_dir) is True
    assert host.terminated == [] and host.force_killed == []  # stranger never signalled
    assert load_state(run_dir).stopped is True
    assert killed == ["r1"]
    assert '"fallback": true' in (run_dir / "journal.jsonl").read_text()


# ---------------------------------------------------------------- graceful stop


def _use_host(monkeypatch, host):
    monkeypatch.setattr(runs, "get_process_host", lambda: host)


def test_request_graceful_stop_writes_file_when_alive(tmp_path, monkeypatch):
    run_dir = _make_state_run(tmp_path, "r1")
    (run_dir / "engine.pid").write_text("4242 100.0")  # identity matches the live host
    _use_host(monkeypatch, _FakeHost(alive=True, identity=100.0))
    assert runs.request_graceful_stop(run_dir) == "requested"
    assert runs.graceful_stop_requested(run_dir)
    body = json.loads((run_dir / runs.STOP_REQUEST_FILE).read_text())
    assert body["mode"] == "graceful"
    assert body["requested_at"]  # an ISO timestamp is stamped
    # written atomically — no staging temp left behind
    assert not (run_dir / (runs.STOP_REQUEST_FILE + ".tmp")).exists()


def test_request_graceful_stop_idempotent_keeps_timestamp(tmp_path, monkeypatch):
    run_dir = _make_state_run(tmp_path, "r1")
    (run_dir / "engine.pid").write_text("4242 100.0")
    _use_host(monkeypatch, _FakeHost(alive=True, identity=100.0))
    # a request already on disk with a distinctive timestamp must be left untouched
    existing = json.dumps({"requested_at": "2000-01-01T00:00:00", "mode": "graceful"})
    (run_dir / runs.STOP_REQUEST_FILE).write_text(existing)
    assert runs.request_graceful_stop(run_dir) == "already-pending"
    assert (run_dir / runs.STOP_REQUEST_FILE).read_text() == existing  # original preserved


def test_request_graceful_stop_refuses_dead_engine(tmp_path):
    run_dir = _make_state_run(tmp_path, "r1")  # no engine.pid → liveness reads "dead"
    with pytest.raises(runs.GracefulStopError, match="no live engine"):
        runs.request_graceful_stop(run_dir)
    assert not runs.graceful_stop_requested(run_dir)  # nothing written on refusal


def test_request_graceful_stop_refuses_finished_run(tmp_path):
    run_dir = _make_state_run(tmp_path, "r1", finished=True)
    with pytest.raises(runs.GracefulStopError, match="already finished"):
        runs.request_graceful_stop(run_dir)
    assert not runs.graceful_stop_requested(run_dir)


def test_request_graceful_stop_unknown_liveness_is_unverifiable(tmp_path, monkeypatch):
    run_dir = _make_state_run(tmp_path, "r1")
    (run_dir / "engine.pid").write_text("4242 100.0")
    # a live pid whose identity is unreadable (win32 access-denied) reads "unknown":
    # the request is still written, but the caller can't confirm a consumer.
    _use_host(monkeypatch, _FakeHost(alive=True, identity=None))
    assert runs.request_graceful_stop(run_dir) == "requested-unverifiable"
    assert runs.graceful_stop_requested(run_dir)


def test_clear_graceful_stop_removes_or_noops(tmp_path):
    run_dir = _make_run(tmp_path, "r1")
    assert runs.clear_graceful_stop(run_dir) is False  # nothing pending → no-op, never raises
    (run_dir / runs.STOP_REQUEST_FILE).write_text("{}")
    assert runs.clear_graceful_stop(run_dir) is True  # present → removed
    assert not runs.graceful_stop_requested(run_dir)
    assert runs.clear_graceful_stop(run_dir) is False  # already gone → no-op again


def test_stop_run_clears_pending_graceful_request(tmp_path, monkeypatch):
    """A hard stop supersedes a pending graceful request: the control file is
    cleared even on the no-live-engine mark-stopped fallback path, so a later
    resume doesn't re-honor the stop the operator escalated past."""
    monkeypatch.setattr(runs, "kill_session", lambda _rid: None)
    run_dir = _make_state_run(tmp_path, "r1")  # no engine.pid → mark-stopped fallback
    (run_dir / runs.STOP_REQUEST_FILE).write_text("{}")  # a graceful request pending
    assert runs.stop_run(run_dir) is True
    assert load_state(run_dir).stopped is True
    assert not runs.graceful_stop_requested(run_dir)


# ---------------------------------------------------------------- prune sessions


def test_mux_sessions_no_tmux(monkeypatch):
    # mux_sessions now delegates to the multiplexer backend; patch its seam.
    monkeypatch.setattr(tmux_base.shutil, "which", lambda _name: None)
    assert runs.mux_sessions() == []


def test_mux_sessions_no_server(monkeypatch):
    monkeypatch.setattr(tmux_base.shutil, "which", lambda _name: "/usr/bin/tmux")
    monkeypatch.setattr(
        tmux_base.subprocess,
        "run",
        lambda *a, **k: subprocess.CompletedProcess(a, 1, stdout="", stderr="no server"),
    )
    assert runs.mux_sessions() == []


def test_prunable_sessions_partitions(tmp_path, monkeypatch):
    mine = runs.project_tag(tmp_path)
    # live run: real run dir with this process's pid, tagged ours
    live = _make_state_run(tmp_path, "live-1")
    runs.write_pid(live)
    # finished run: run dir exists but dead pid, tagged ours
    finished = _make_state_run(tmp_path, "fin-1")
    (finished / "engine.pid").write_text(str(_dead_pid()))
    # orphan tagged ours: session's run dir is gone -> still prunable
    # untagged finished run: ownership proven by the run dir under this project
    untag_fin = _make_state_run(tmp_path, "untag-fin")
    (untag_fin / "engine.pid").write_text(str(_dead_pid()))

    sessions = [
        "bmad-loop-live-1",
        "bmad-loop-fin-1",
        "bmad-loop-orphan-1",
        "bmad-loop-other-1",  # another project's live run
        "bmad-loop-untag-fin",  # pre-upgrade session, no tag
        "bmad-loop-untag-orphan",  # pre-upgrade, no tag, no run dir here
        "bmad-loop-ctl",  # control session: never a candidate
        "unrelated",  # not ours
    ]
    monkeypatch.setattr(runs, "mux_sessions", lambda: sessions)
    monkeypatch.setattr(
        runs,
        "session_project_tags",
        lambda: {
            "bmad-loop-live-1": mine,
            "bmad-loop-fin-1": mine,
            "bmad-loop-orphan-1": mine,
            "bmad-loop-other-1": "/some/other/project",
            # untag-* and unrelated intentionally absent (no tag)
        },
    )
    prunable, alive, unknown = runs.prunable_sessions(tmp_path)
    # other-1 (foreign tag) and untag-orphan (unprovable) are skipped entirely
    assert sorted(prunable) == ["fin-1", "orphan-1", "untag-fin"]
    assert alive == ["live-1"]
    assert unknown == set()


def test_prunable_sessions_skips_invalid_run_ids(tmp_path, monkeypatch):
    """A session name is untrusted input (anyone can create one). Stripping the
    prefix off `bmad-loop-../../x` would hand `run_dir_for` a traversing id, and a
    tagged session would then steer engine_liveness — and prune_sessions' kill — at
    a path outside the runs dir. Reject before recomposing."""
    mine = runs.project_tag(tmp_path)
    good = _make_state_run(tmp_path, "fin-1")
    (good / "engine.pid").write_text(str(_dead_pid()))

    sessions = ["bmad-loop-fin-1", "bmad-loop-../../x", "bmad-loop-a.b", "bmad-loop-"]
    monkeypatch.setattr(runs, "mux_sessions", lambda: sessions)
    monkeypatch.setattr(runs, "session_project_tags", lambda: dict.fromkeys(sessions, mine))

    prunable, live, unknown = runs.prunable_sessions(tmp_path)
    assert prunable == ["fin-1"]
    assert live == [] and unknown == set()


def test_prunable_sessions_flags_unknown(tmp_path, monkeypatch):
    # live pid, unreadable identity (win32 ERROR_ACCESS_DENIED) → prunable anyway
    # (unknown never blocks cleanup) but flagged so frontends can warn.
    mine = runs.project_tag(tmp_path)
    odd = _make_state_run(tmp_path, "odd-1")
    (odd / "engine.pid").write_text("4242 123.0")
    monkeypatch.setattr(runs, "mux_sessions", lambda: ["bmad-loop-odd-1"])
    monkeypatch.setattr(runs, "session_project_tags", lambda: {"bmad-loop-odd-1": mine})
    monkeypatch.setattr(runs, "get_process_host", lambda: _FakeHost(alive=True, identity=None))
    prunable, live, unknown = runs.prunable_sessions(tmp_path)
    assert prunable == ["odd-1"]
    assert live == []
    assert unknown == {"odd-1"}


def test_prune_sessions_dry_run_kills_nothing(tmp_path, monkeypatch):
    finished = _make_state_run(tmp_path, "fin-1")
    (finished / "engine.pid").write_text(str(_dead_pid()))
    killed: list[str] = []
    monkeypatch.setattr(runs, "kill_session", lambda rid: killed.append(rid))
    monkeypatch.setattr(runs, "mux_sessions", lambda: ["bmad-loop-fin-1"])
    monkeypatch.setattr(
        runs, "session_project_tags", lambda: {"bmad-loop-fin-1": runs.project_tag(tmp_path)}
    )
    assert runs.prune_sessions(tmp_path, dry_run=True) == (["fin-1"], [], set())
    assert killed == []
    assert runs.prune_sessions(tmp_path) == (["fin-1"], [], set())
    assert killed == ["fin-1"]


def test_prune_sessions_returns_unknown_from_same_sample(tmp_path, monkeypatch):
    # the unknown subset must come from the partition prune_sessions itself
    # killed, so a frontend warning built from it never names an unpruned session
    mine = runs.project_tag(tmp_path)
    odd = _make_state_run(tmp_path, "odd-1")
    (odd / "engine.pid").write_text("4242 123.0")
    killed: list[str] = []
    monkeypatch.setattr(runs, "kill_session", lambda rid: killed.append(rid))
    monkeypatch.setattr(runs, "mux_sessions", lambda: ["bmad-loop-odd-1"])
    monkeypatch.setattr(runs, "session_project_tags", lambda: {"bmad-loop-odd-1": mine})
    monkeypatch.setattr(runs, "get_process_host", lambda: _FakeHost(alive=True, identity=None))
    assert runs.prune_sessions(tmp_path) == (["odd-1"], [], {"odd-1"})
    assert killed == ["odd-1"]


def test_delete_run(tmp_path):
    run_dir = _make_state_run(tmp_path, "r1")
    runs.delete_run(run_dir)
    assert not run_dir.exists()


def _escalated_run(tmp_path, spec_text, *, restore_patch_stale=None, git_project=False):
    """conftest's builder with this module's shape: the spec is written first (so
    `git_project=True` commits it), and only `(run_dir, spec)` comes back."""
    spec = tmp_path / "spec.md"
    spec.write_text(spec_text, encoding="utf-8")
    run = escalated_run(
        tmp_path,
        "r1",
        story_key="1-1-a",
        attempt=2,
        started_at="2026-06-11T10:00:00",
        spec_file=str(spec),
        restore_patch=restore_patch_stale,
        git_project=git_project,
    )
    return run.run_dir, spec


_SPEC_WITH_ARR = (
    "---\ntitle: t\nstatus: blocked\n---\n\n## Intent\n\nbody\n"
    "\n## Auto Run Result\n\n- Status: blocked\n\nboom\n"
)


def test_rearm_restore_mode_sets_in_review_strips_arr_and_latches(tmp_path):
    from bmad_loop.journal import Journal
    from bmad_loop.model import Phase

    run_dir, spec = _escalated_run(tmp_path, _SPEC_WITH_ARR)
    runs.rearm_escalation(run_dir, restore_patch="artifacts/attempt.patch")

    task = load_state(run_dir).tasks["1-1-a"]
    assert task.phase == Phase.PENDING and task.attempt == 0
    assert task.restore_patch == "artifacts/attempt.patch"
    text = spec.read_text()
    assert "status: in-review" in text  # in-review routes step-01 -> step-04
    assert "## Auto Run Result" not in text  # stale terminal section stripped
    entry = [e for e in Journal(run_dir).entries() if e["kind"] == "story-escalation-resolved"][-1]
    assert entry["restore"] is True


def test_rearm_plain_mode_sets_ready_for_dev_and_clears_stale_latch(tmp_path):
    from bmad_loop.journal import Journal
    from bmad_loop.model import Phase

    # a stale latch from a prior restore attempt the human then chose to redo fresh
    run_dir, spec = _escalated_run(tmp_path, _SPEC_WITH_ARR, restore_patch_stale="old.patch")
    runs.rearm_escalation(run_dir)  # no restore_patch => from-scratch

    task = load_state(run_dir).tasks["1-1-a"]
    assert task.phase == Phase.PENDING
    assert task.restore_patch is None  # stale latch cleared
    assert "status: ready-for-dev" in spec.read_text()
    entry = [e for e in Journal(run_dir).entries() if e["kind"] == "story-escalation-resolved"][-1]
    assert entry["restore"] is False


def test_rearm_aborts_when_the_spec_status_cannot_be_reopened(tmp_path):
    """The seam that proves the silent-`False` defect mattered. This spec reads as
    `status: blocked` — the reader resolves the block scalar fine — so it clears
    every gate ahead of the flip, and the flip is the one thing that cannot
    happen. Under the old line scanner that was a `False` nobody read: the re-drive
    was dispatched, step-01 saw the unchanged terminal status and routed the
    session to "ingest as context, do not resume", and the story re-wedged.

    Nothing may be persisted: `save_state` runs BELOW this point, so the task must
    still be ESCALATED at attempt 2 and the escalation still armed for a retry."""
    from bmad_loop.model import Phase

    spec_text = (
        "---\ntitle: t\nstatus: |\n  blocked\n---\n\n## Intent\n\nbody\n"
        "\n## Auto Run Result\n\n- Status: blocked\n\nboom\n"
    )
    run_dir, spec = _escalated_run(tmp_path, spec_text)
    assert verify.status_of(verify.read_frontmatter(spec)) == "blocked"  # the reader is fine

    with pytest.raises(runs.RearmError, match="re-open story spec"):
        runs.rearm_escalation(run_dir, restore_patch="artifacts/attempt.patch")

    assert spec.read_text(encoding="utf-8") == spec_text  # byte-identical
    task = load_state(run_dir).tasks["1-1-a"]
    assert task.phase == Phase.ESCALATED and task.attempt == 2  # nothing persisted
    assert task.restore_patch is None  # the latch never landed either


def test_rearm_resets_followup_reviews_spent(tmp_path):
    """A human-resolved re-drive gets a fresh damping budget: rearm_escalation
    zeroes followup_reviews_spent alongside review_cycle, so the clean rebuild
    against the corrected spec can honor a follow-up again."""
    run_dir, _ = _escalated_run(tmp_path, _SPEC_WITH_ARR)
    # seed a spent damping budget from the escalated attempt
    state = load_state(run_dir)
    state.tasks["1-1-a"].followup_reviews_spent = 3
    state.tasks["1-1-a"].review_cycle = 2
    save_state(run_dir, state)

    runs.rearm_escalation(run_dir)

    task = load_state(run_dir).tasks["1-1-a"]
    assert task.followup_reviews_spent == 0
    assert task.review_cycle == 0  # reset in lockstep with the counter


# --------------------------------------------- #90: abandoned restore-latch residue


def _stale_restore_tree(tmp_path, *, latch="artifacts/attempt.patch"):
    """An escalation whose latched restore already applied: `newfile.txt` is the
    patch's untracked creation, `human.txt` is the resolve session's own file."""
    run_dir, spec = _escalated_run(
        tmp_path, _SPEC_WITH_ARR, restore_patch_stale=latch, git_project=True
    )
    patch = tmp_path / "artifacts" / "attempt.patch"
    patch.parent.mkdir(parents=True, exist_ok=True)
    patch.write_text(
        "diff --git a/newfile.txt b/newfile.txt\n"
        "new file mode 100644\n"
        "index 0000000..1111111\n"
        "--- /dev/null\n"
        "+++ b/newfile.txt\n"
        "@@ -0,0 +1 @@\n"
        "+from the abandoned attempt\n",
        encoding="utf-8",
    )
    (tmp_path / "newfile.txt").write_text("from the abandoned attempt\n")  # the applied residue
    (tmp_path / "human.txt").write_text("from the resolve session\n")
    return run_dir, spec, patch


def _kinds(run_dir, prefix="stale-restore-"):
    from bmad_loop.journal import Journal

    return [e for e in Journal(run_dir).entries() if e["kind"].startswith(prefix)]


def test_rearm_excludes_stale_restore_residue_from_baseline_snapshot(tmp_path):
    """The abandoned attempt's applied new files must NOT be blessed as
    pre-existing, or finalize_commit's `add -A` sweeps them into the corrected
    story's commit. The resolve session's own untracked file still is."""
    run_dir, _spec, patch = _stale_restore_tree(tmp_path)

    runs.rearm_escalation(run_dir)  # from-scratch re-arm replaces the latch

    task = load_state(run_dir).tasks["1-1-a"]
    assert "human.txt" in task.baseline_untracked
    assert "newfile.txt" not in task.baseline_untracked
    assert (tmp_path / "newfile.txt").exists()  # rearm deletes nothing; the re-drive's reset does
    excluded = _kinds(run_dir, "stale-restore-excluded")
    assert len(excluded) == 1
    assert excluded[0]["files"] == ["newfile.txt"]
    assert excluded[0]["patch"] == str(patch)


def test_rearm_re_latching_the_same_patch_still_excludes_its_residue(tmp_path):
    """Re-arming a restore onto the same patch: the first application's files are
    still residue (and `git apply` would otherwise fail with 'already exists')."""
    run_dir, _spec, _patch = _stale_restore_tree(tmp_path)

    runs.rearm_escalation(run_dir, restore_patch="artifacts/attempt.patch")

    task = load_state(run_dir).tasks["1-1-a"]
    assert task.restore_patch == "artifacts/attempt.patch"
    assert "human.txt" in task.baseline_untracked
    assert "newfile.txt" not in task.baseline_untracked
    assert _kinds(run_dir, "stale-restore-excluded")


def test_rearm_missing_stale_patch_degrades_loudly_without_raising(tmp_path):
    """A deleted patch file must never wedge resolve: journal the degrade and fall
    back to the pre-#90 snapshot (everything untracked counts as pre-existing)."""
    run_dir, _spec, patch = _stale_restore_tree(tmp_path)
    patch.unlink()
    (tmp_path / "committed.txt").write_text("from the escalated attempt\n")
    git(tmp_path, "add", "committed.txt")
    git(tmp_path, "commit", "-q", "-m", "attempt commit")

    runs.rearm_escalation(run_dir)  # must not raise RearmError

    task = load_state(run_dir).tasks["1-1-a"]
    assert {"human.txt", "newfile.txt"} <= set(task.baseline_untracked)  # full snapshot
    unparseable = _kinds(run_dir, "stale-restore-unparseable")
    assert len(unparseable) == 1
    assert "FileNotFoundError" in unparseable[0]["error"]
    assert not _kinds(run_dir, "stale-restore-excluded")
    # the unreadable patch must not also cost the human the commits warning
    assert _kinds(run_dir, "stale-restore-commits")


def test_rearm_without_a_stale_latch_journals_no_stale_restore_events(tmp_path):
    run_dir, _spec = _escalated_run(tmp_path, _SPEC_WITH_ARR, git_project=True)
    (tmp_path / "human.txt").write_text("from the resolve session\n")

    runs.rearm_escalation(run_dir, restore_patch="artifacts/attempt.patch")

    assert "human.txt" in load_state(run_dir).tasks["1-1-a"].baseline_untracked
    assert _kinds(run_dir) == []


def test_rearm_warns_about_commits_below_the_refreshed_baseline(tmp_path):
    """The worse variant: commits made above the OLD baseline become the re-drive's
    permanent starting point. Warn-only — a mechanical revert would claw back the
    resolve session's own blessed commits, which live in the same range."""
    run_dir, _spec, _patch = _stale_restore_tree(tmp_path)
    (tmp_path / "committed.txt").write_text("from the escalated attempt\n")
    git(tmp_path, "add", "committed.txt")
    git(tmp_path, "commit", "-q", "-m", "attempt commit")
    old_baseline = load_state(run_dir).tasks["1-1-a"].baseline_commit

    runs.rearm_escalation(run_dir)

    task = load_state(run_dir).tasks["1-1-a"]
    assert task.baseline_commit != old_baseline  # baseline advanced past the commit
    warned = _kinds(run_dir, "stale-restore-commits")
    assert len(warned) == 1
    assert warned[0]["old_baseline"] == old_baseline
    assert warned[0]["commits"] == [git(tmp_path, "rev-parse", "HEAD")]


def test_archive_run(tmp_path):
    run_dir = _make_state_run(tmp_path, "20260611-100000-aaaa")
    (run_dir / "journal.jsonl").write_text('{"kind":"x"}\n')
    dest = runs.archive_run(tmp_path, run_dir)

    assert dest == tmp_path / ".bmad-loop" / "archive" / "20260611-100000-aaaa.tar.gz"
    assert dest.is_file()
    assert not run_dir.exists()  # original removed
    assert not dest.with_suffix(".tar.gz.tmp").exists()  # temp cleaned via replace
    with tarfile.open(dest) as tar:
        names = tar.getnames()
    assert "20260611-100000-aaaa/state.json" in names
    assert "20260611-100000-aaaa/journal.jsonl" in names
