"""tui.launch builds exact tmux/CLI argv — verified against monkeypatched
subprocess so no real tmux server is touched, plus one real-subprocess
sanity check of the captured path.

The tmux invocations now live in the multiplexer backend (launch drives the
seam), so the tmux subprocess/which seams are patched on ``tmux_base`` (the
shared backend base where the spawn primitive lives); the captured read-only
path still shells out from ``launch`` itself."""

from __future__ import annotations

import json
import os
import shlex
import signal
import subprocess
import sys
from pathlib import Path

import pytest

from bmad_loop import runs
from bmad_loop.adapters import tmux_base
from bmad_loop.adapters.multiplexer import MultiplexerError, get_multiplexer
from bmad_loop.tui import launch

# Every test here asserts tmux-specific argv/behaviour through the multiplexer
# seam. An installed external backend can match win32 (the herdr adapter does),
# where tmux does not — get_multiplexer() would then not bottom-fall-back to
# tmux — so pin tmux by name (a no-op on a stock POSIX box).
pytestmark = pytest.mark.usefixtures("force_tmux_backend")


class FakeRun:
    """Records argv; scripts the returncode of `tmux has-session`."""

    def __init__(self, has_session_rc: int = 1):
        self.calls: list[list[str]] = []
        self.has_session_rc = has_session_rc

    def __call__(self, argv, **kwargs):
        self.calls.append(list(argv))
        rc = self.has_session_rc if argv[1] == "has-session" else 0
        out = "@7\n" if argv[1] == "new-window" else ""
        return subprocess.CompletedProcess(argv, rc, stdout=out, stderr="")

    def by_verb(self, verb: str) -> list[list[str]]:
        return [c for c in self.calls if c[1] == verb]


@pytest.fixture
def fake_run(monkeypatch) -> FakeRun:
    fake = FakeRun()
    monkeypatch.setattr(tmux_base.subprocess, "run", fake)
    monkeypatch.setattr(tmux_base.shutil, "which", lambda name: f"/usr/bin/{name}")
    # These tests pin the POSIX tmux argv shapes; force that backend so they
    # hold on hosts where platform selection would pick another (win32 → psmux).
    monkeypatch.setenv("BMAD_LOOP_MUX_BACKEND", "tmux")
    get_multiplexer.cache_clear()
    yield fake
    get_multiplexer.cache_clear()


def expected_cli(*tail: str) -> str:
    return shlex.join([sys.executable, "-m", "bmad_loop.cli", *tail])


def test_start_run_detached_argv(fake_run, tmp_path: Path):
    launch.start_run_detached(tmp_path, "RID", epic=2, story="1-2-x", max_stories=3)

    nw0 = fake_run.by_verb("new-window")[0]
    assert nw0[nw0.index("-F") + 1] == "#{window_id}"

    # control session was missing: has-session, new-session, new-window, then
    # the project tag is stamped on the new window so cross-project cleanup
    # never closes it
    assert [c[1] for c in fake_run.calls] == [
        "has-session",
        "new-session",
        "new-window",
        "set-option",
    ]
    from bmad_loop import runs

    assert fake_run.by_verb("set-option")[0] == [
        "tmux",
        "set-option",
        "-w",
        "-t",
        "@7",
        runs.PROJECT_OPTION,
        runs.project_tag(tmp_path),
    ]
    ns = fake_run.by_verb("new-session")[0]
    assert ns == [
        "tmux",
        "new-session",
        "-d",
        "-s",
        "bmad-loop-ctl",
        "-c",
        str(tmp_path),
    ]

    nw = fake_run.by_verb("new-window")[0]
    assert nw[:2] == ["tmux", "new-window"]
    assert "-d" in nw
    assert nw[nw.index("-t") + 1] == "=bmad-loop-ctl:"
    assert nw[nw.index("-n") + 1] == "run-RID"
    assert nw[nw.index("-c") + 1] == str(tmp_path)
    assert nw[-3:-1] == ["sh", "-c"]
    shell = nw[-1]
    assert (
        expected_cli(
            "run",
            "--project",
            str(tmp_path),
            "--run-id",
            "RID",
            "--epic",
            "2",
            "--story",
            "1-2-x",
            "--max-stories",
            "3",
        )
        in shell
    )
    assert "read -r" in shell  # window stays open showing the exit status
    # after the read, return the attached client to where it came from: switch a
    # same-tmux client back to its pane, or detach a throwaway external client
    assert "@bmad_return_pane" in shell
    assert "switch-client" in shell
    assert "detach-client" in shell


def test_start_run_detached_argv_stories(fake_run, tmp_path: Path):
    launch.start_run_detached(tmp_path, "RID", spec="_bmad-output/epic-1")
    shell = fake_run.by_verb("new-window")[0][-1]
    assert (
        expected_cli(
            "run",
            "--project",
            str(tmp_path),
            "--run-id",
            "RID",
            "--spec",
            "_bmad-output/epic-1",
        )
        in shell
    )


def test_start_run_omits_blank_filters(fake_run, tmp_path: Path):
    launch.start_run_detached(tmp_path, "RID")
    shell = fake_run.by_verb("new-window")[0][-1]
    assert expected_cli("run", "--project", str(tmp_path), "--run-id", "RID") in shell
    for flag in ("--epic", "--story", "--max-stories"):
        assert flag not in shell


def test_start_sweep_detached_flags(fake_run, tmp_path: Path):
    launch.start_sweep_detached(tmp_path, "RID", no_prompt=True, decisions_only=True, max_bundles=2)
    nw = fake_run.by_verb("new-window")[0]
    assert nw[nw.index("-n") + 1] == "sweep-RID"
    shell = nw[-1]
    assert (
        expected_cli(
            "sweep",
            "--project",
            str(tmp_path),
            "--run-id",
            "RID",
            "--no-prompt",
            "--decisions-only",
            "--max-bundles",
            "2",
        )
        in shell
    )


def test_resume_detached_argv(fake_run, tmp_path: Path):
    launch.resume_detached(tmp_path, "RID")
    nw = fake_run.by_verb("new-window")[0]
    assert nw[nw.index("-n") + 1] == "resume-RID"
    assert expected_cli("resume", "--project", str(tmp_path), "RID") in nw[-1]


def test_existing_ctl_session_reused(monkeypatch, tmp_path: Path):
    fake = FakeRun(has_session_rc=0)
    monkeypatch.setattr(tmux_base.subprocess, "run", fake)
    monkeypatch.setattr(tmux_base.shutil, "which", lambda name: f"/usr/bin/{name}")
    launch.resume_detached(tmp_path, "RID")
    assert [c[1] for c in fake.calls] == ["has-session", "new-window", "set-option"]


def test_launch_without_mux_raises(monkeypatch, tmp_path: Path):
    monkeypatch.delenv("BMAD_LOOP_MUX_BACKEND", raising=False)
    get_multiplexer.cache_clear()
    monkeypatch.setattr(tmux_base.shutil, "which", lambda name: None)
    assert not launch.mux_available()
    with pytest.raises(launch.LaunchError, match="multiplexer backend unavailable"):
        launch.start_run_detached(tmp_path, "RID")


def test_forced_launch_bypasses_availability(fake_run, monkeypatch, capsys, tmp_path: Path):
    from bmad_loop.adapters import multiplexer as mux_mod

    monkeypatch.setattr(mux_mod, "_FORCED_UNUSABLE_WARNED", False)
    monkeypatch.setattr(tmux_base.shutil, "which", lambda name: None)
    launch.start_run_detached(tmp_path, "RID")
    assert fake_run.by_verb("new-window")
    # trusted, but not silently: the bypass names itself once on stderr
    assert "forced multiplexer backend" in capsys.readouterr().err


def test_observers_follow_forced_backend(fake_run, monkeypatch):
    """The observer gates (mux_available feeds attach/ctl-window/prune) must
    share the launch preflight's forced-aware rule — launch working while
    attach reports "nothing to attach to" would be a silent split."""
    from bmad_loop.adapters import multiplexer as mux_mod

    monkeypatch.setattr(mux_mod, "_usable", lambda mux: False)
    assert launch.mux_available() is True  # fake_run's fixture forces tmux by env


def test_new_window_failure_raises(monkeypatch, tmp_path: Path):
    def failing_run(argv, **kwargs):
        rc = 1 if argv[1] in ("has-session", "new-window") else 0
        return subprocess.CompletedProcess(argv, rc, stdout="", stderr="boom")

    monkeypatch.setattr(tmux_base.subprocess, "run", failing_run)
    monkeypatch.setattr(tmux_base.shutil, "which", lambda name: f"/usr/bin/{name}")
    with pytest.raises(launch.LaunchError, match="new-window.*failed: boom"):
        launch.start_run_detached(tmp_path, "RID")


def test_ensure_ctl_session_probe_failure_raises_launch_error(monkeypatch, tmp_path: Path):
    # has_session is raiser-side: a transport failure (timeout / missing binary) on
    # the ctl-session probe must convert to LaunchError so the TUI's launch/resume/
    # resolve handlers (which catch LaunchError) surface a toast instead of crashing
    # on the raw MultiplexerError that would otherwise slip past their except clause.
    def failing_run(argv, **kwargs):
        if argv[1] == "has-session":
            raise OSError("backend server not reachable")
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    monkeypatch.setattr(tmux_base.subprocess, "run", failing_run)
    monkeypatch.setattr(tmux_base.shutil, "which", lambda name: f"/usr/bin/{name}")
    with pytest.raises(launch.LaunchError, match="ctl-session setup failed"):
        launch.start_run_detached(tmp_path, "RID")


def test_session_exists(monkeypatch):
    fake = FakeRun(has_session_rc=0)
    monkeypatch.setattr(tmux_base.subprocess, "run", fake)
    assert launch.session_exists("bmad-loop-x")
    assert fake.calls[0] == ["tmux", "has-session", "-t", "=bmad-loop-x"]


def _ctl_listing(monkeypatch, rows: str) -> list[list[str]]:
    """Script the ctl-session window listing; returns the recorded argv."""
    calls: list[list[str]] = []

    def fake(argv, **kwargs):
        calls.append(list(argv))
        out = rows if argv[1] == "list-windows" else ""
        return subprocess.CompletedProcess(argv, 0, stdout=out, stderr="")

    monkeypatch.setattr(tmux_base.subprocess, "run", fake)
    monkeypatch.setattr(tmux_base.shutil, "which", lambda name: f"/usr/bin/{name}")
    return calls


def _write_record(project: Path, run_id: str, win_id: str) -> Path:
    """Stand in for a launch having minted `win_id` for this run."""
    run_dir = runs.run_dir_for(project, run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    record = run_dir / launch._CTL_WINDOW_FILE
    record.write_text(win_id, encoding="utf-8")
    return record


def test_ctl_window_id_matches_run_id_suffix(monkeypatch, tmp_path: Path):
    # The id, not the name: consumers replay the value as select/kill/option
    # targets, where a by-name resolve can land on a duplicate. With no record
    # of what the run's last launch minted, the answer is the first match.
    _ctl_listing(monkeypatch, "@1\trun-AAAA\n@2\tsweep-RID\n@3\tresume-BBBB\n")
    assert launch.ctl_window_id(tmp_path, "RID") == "@2"
    assert launch.ctl_window_id(tmp_path, "CCCC") is None


def test_ctl_window_id_prefers_the_window_the_last_launch_minted(monkeypatch, tmp_path: Path):
    # #482: `e` over a parked run leaves `run-RID` in front of the live
    # `resume-RID`, and the scan alone answers the parked corpse. The recorded
    # id names the window we actually created.
    _ctl_listing(monkeypatch, "@1\trun-RID\n@2\tresume-RID\n")
    _write_record(tmp_path, "RID", "@2")
    assert launch.ctl_window_id(tmp_path, "RID") == "@2"


def test_ctl_window_id_ignores_a_record_the_listing_no_longer_shows(monkeypatch, tmp_path: Path):
    # The recorded window was killed (`x`) or pruned. Replaying a target that no
    # longer resolves is the dangerous kind of stale — an unresolvable `-t`
    # lands on the *active* window — so fall back to a window that exists.
    _ctl_listing(monkeypatch, "@1\trun-RID\n")
    _write_record(tmp_path, "RID", "@2")
    assert launch.ctl_window_id(tmp_path, "RID") == "@1"


def test_ctl_window_id_ignores_a_record_that_now_names_another_run(monkeypatch, tmp_path: Path):
    # A backend that reuses a freed window id must not let a stale record hand
    # back a foreign run's window: the record is re-proved against the name too.
    _ctl_listing(monkeypatch, "@2\trun-OTHER\n@5\tresume-RID\n")
    _write_record(tmp_path, "RID", "@2")
    assert launch.ctl_window_id(tmp_path, "RID") == "@5"


def test_ctl_window_id_none_when_no_window_carries_the_run_id(monkeypatch, tmp_path: Path):
    # A record can never resurrect a run whose windows are all gone.
    _ctl_listing(monkeypatch, "@1\trun-OTHER\n@3\tshell\n")
    _write_record(tmp_path, "RID", "@1")
    assert launch.ctl_window_id(tmp_path, "RID") is None


def test_ctl_window_id_unreadable_record_falls_back(monkeypatch, tmp_path: Path):
    # An unreadable hint is not an error — it just leaves the name scan.
    _ctl_listing(monkeypatch, "@1\trun-RID\n@2\tresume-RID\n")
    run_dir = runs.run_dir_for(tmp_path, "RID")
    (run_dir / launch._CTL_WINDOW_FILE).mkdir(parents=True)  # a dir, not a file
    assert launch.ctl_window_id(tmp_path, "RID") == "@1"


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX FIFOs")
def test_read_record_does_not_block_on_a_fifo(tmp_path: Path):
    # A session can replace its own workspace-writable record with a FIFO, and
    # opening one for reading blocks until somebody writes. action_attach reads
    # this on Textual's event loop, so that freezes the dashboard on a keypress.
    # O_NONBLOCK returns immediately; the S_ISREG check on the opened descriptor
    # then rejects it. Under an alarm because a regression here HANGS the suite —
    # and with a handler that RAISES, so the ablation fails this test rather than
    # letting the default SIGALRM disposition kill the whole pytest process.
    run_dir = runs.run_dir_for(tmp_path, "RID")
    run_dir.mkdir(parents=True)
    os.mkfifo(run_dir / launch._CTL_WINDOW_FILE)

    # NOT TimeoutError: that is a subclass of OSError, so _read_ctl_window's own
    # `except OSError` swallows it and the ablated code still returns None — the
    # first version of this test passed against the bug, five seconds slower.
    class Blocked(Exception):
        pass

    def _blocked(_signum, _frame):
        raise Blocked("_read_ctl_window blocked on a FIFO")

    previous = signal.signal(signal.SIGALRM, _blocked)
    signal.setitimer(signal.ITIMER_REAL, 5)
    try:
        assert launch._read_ctl_window(tmp_path, "RID") is None
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX FIFOs")
def test_read_record_rejects_a_fifo_that_already_has_data(tmp_path: Path):
    # The case only the S_ISREG check catches, and the reason it is not redundant
    # with the other two guards: a writer holding the FIFO open with bytes queued
    # means the open does not block (so O_NONBLOCK is not what refuses it) and
    # the path is not a link (so O_NOFOLLOW is not either). Without the check the
    # queued bytes are simply read, letting a session forge the record through a
    # pipe it controls rather than a file. Ablate S_ISREG and this returns "@2".
    run_dir = runs.run_dir_for(tmp_path, "RID")
    run_dir.mkdir(parents=True)
    fifo = run_dir / launch._CTL_WINDOW_FILE
    os.mkfifo(fifo)

    writer = os.open(fifo, os.O_RDWR | os.O_NONBLOCK)  # RDWR: no peer needed
    try:
        os.write(writer, b"@2")
        assert launch._read_ctl_window(tmp_path, "RID") is None
    finally:
        os.close(writer)


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX device nodes")
def test_read_record_rejects_a_non_regular_file(tmp_path: Path):
    # The S_ISREG check on the opened descriptor, which O_NOFOLLOW alone does
    # not give (a link is only one way to reach a device). `/dev/zero` is the
    # case that bites hardest: an endless source raises MemoryError, which is
    # not an OSError and so escapes this function's "never raises" promise
    # entirely, out through action_attach, which has no handler at all.
    run_dir = runs.run_dir_for(tmp_path, "RID")
    run_dir.mkdir(parents=True)
    (run_dir / launch._CTL_WINDOW_FILE).symlink_to("/dev/zero")

    assert launch._read_ctl_window(tmp_path, "RID") is None


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX symlinks")
def test_read_record_does_not_follow_a_symlink(tmp_path: Path):
    # O_NOFOLLOW: the name is read, not wherever it points. The target here is a
    # perfectly ordinary file holding a perfectly plausible window id, so every
    # other guard passes it — only the no-follow refuses. Symmetry with the
    # write side, which replaces the name rather than the link's target.
    run_dir = runs.run_dir_for(tmp_path, "RID")
    run_dir.mkdir(parents=True)
    elsewhere = tmp_path / "somewhere-else"
    elsewhere.write_text("@99", encoding="utf-8")
    (run_dir / launch._CTL_WINDOW_FILE).symlink_to(elsewhere)

    assert launch._read_ctl_window(tmp_path, "RID") is None


def test_read_record_is_bounded(tmp_path: Path):
    # The cap stands on its own, without the flags: a plain regular file can be
    # arbitrarily large, and a hint is at most a window id either way.
    run_dir = runs.run_dir_for(tmp_path, "RID")
    run_dir.mkdir(parents=True)
    (run_dir / launch._CTL_WINDOW_FILE).write_text("@" + "9" * 5_000_000, encoding="utf-8")

    recorded = launch._read_ctl_window(tmp_path, "RID")
    assert recorded is not None and len(recorded) <= launch._MAX_RECORD_BYTES


def test_ctl_window_id_invalid_utf8_record_falls_back(monkeypatch, tmp_path: Path):
    _ctl_listing(monkeypatch, "@1\trun-RID\n@2\tresume-RID\n")
    record = _write_record(tmp_path, "RID", "@2")
    record.write_bytes(b"\xff")
    assert launch.ctl_window_id(tmp_path, "RID") == "@1"


def test_ctl_window_id_skips_empty_id_rows(monkeypatch, tmp_path: Path):
    # An empty id must never be returned as a target — an empty `-t` resolves
    # against the current window. psmux's qualifier passes a falsy id through.
    _ctl_listing(monkeypatch, "\tsweep-RID\n@7\tsweep-RID\n")
    assert launch.ctl_window_id(tmp_path, "RID") == "@7"


def test_kill_ctl_window_kills_by_resolved_id_not_a_name_token(monkeypatch, tmp_path: Path):
    # The kill replays the id this listing resolved, never a `=session:name`
    # token the backend would resolve again. With no record the scan picks the
    # first match (`@7`); what the id buys is that a rename or a new window
    # between two verbs cannot re-point the second.
    calls = _ctl_listing(monkeypatch, "@2\trun-x\n@7\tsweep-RID\n@9\tsweep-RID\n")
    launch.kill_ctl_window(tmp_path, "RID")
    assert ["tmux", "kill-window", "-t", "@7"] in calls


def test_attach_plan_selects_and_returns_the_recorded_window(monkeypatch, tmp_path: Path):
    # #482's first two consequences: the window the attach lands on, and the one
    # its return_window stamps @bmad_return_pane on, are the same live window.
    calls = _ctl_listing(monkeypatch, "@1\trun-RID\n@2\tresume-RID\n")
    _write_record(tmp_path, "RID", "@2")
    monkeypatch.setattr(launch, "session_exists", lambda s: False)
    monkeypatch.setattr(launch, "decision_pending", lambda rd: False)
    plan = launch.attach_plan(tmp_path, "RID")
    assert plan is not None
    _argv, return_window = plan
    assert return_window == "@2"
    assert ["tmux", "select-window", "-t", "@2"] in calls


def test_kill_ctl_window_follows_the_record(monkeypatch, tmp_path: Path):
    # #482's third consequence: `x` must not close the parked window and leave
    # the live one running.
    calls = _ctl_listing(monkeypatch, "@1\trun-RID\n@2\tresume-RID\n")
    _write_record(tmp_path, "RID", "@2")
    launch.kill_ctl_window(tmp_path, "RID")
    assert ["tmux", "kill-window", "-t", "@2"] in calls


def test_ctl_window_id_no_session_or_tmux(monkeypatch, tmp_path: Path):
    def fake(argv, **kwargs):
        return subprocess.CompletedProcess(argv, 1, stdout="", stderr="no session")

    monkeypatch.setattr(tmux_base.subprocess, "run", fake)
    monkeypatch.setattr(tmux_base.shutil, "which", lambda name: f"/usr/bin/{name}")
    assert launch.ctl_window_id(tmp_path, "RID") is None
    monkeypatch.setattr(tmux_base.shutil, "which", lambda name: None)
    assert launch.ctl_window_id(tmp_path, "RID") is None  # no subprocess call attempted


def test_set_return_pane_argv(fake_run):
    launch.set_return_pane("=bmad-loop-ctl:sweep-RID", "%9")
    assert fake_run.calls == [
        ["tmux", "set-option", "-w", "-t", "=bmad-loop-ctl:sweep-RID", "@bmad_return_pane", "%9"]
    ]


def test_current_return_target_bare_pane_on_tmux(monkeypatch):
    # The launch helper delegates to the backend; on tmux the seam default
    # answers the bare pane id — globally unique under the one-server model,
    # and the only form tmux's switch-client actually resolves (its window
    # resolver rejects a pane id in the `session:%N` slot). The qualified
    # composition is a psmux override, pinned in test_psmux_backend.
    def fake(argv, **kwargs):
        assert argv[-1] == "#{pane_id}"  # exactly one probe, no session probe
        return subprocess.CompletedProcess(argv, 0, stdout="%9\n", stderr="")

    monkeypatch.setenv("TMUX", "/tmp/tmux-1000/default,123,0")  # inside tmux
    monkeypatch.setattr(tmux_base.subprocess, "run", fake)
    assert launch.current_return_target() == "%9"


def test_current_return_target_none_outside_tmux(monkeypatch):
    # Outside tmux the TMUX guard answers None WITHOUT shelling out: against a
    # live server, display-message would answer for some OTHER client's session
    # and misreport a plain shell as being inside tmux.
    def boom(*_a, **_k):
        raise AssertionError("outside tmux, current_* must not shell out")

    monkeypatch.delenv("TMUX", raising=False)
    monkeypatch.setattr(tmux_base.subprocess, "run", boom)
    assert launch.current_return_target() is None
    assert launch.current_session() is None


def test_current_return_target_none_on_transport_failure(monkeypatch):
    def fake(argv, **kwargs):
        return subprocess.CompletedProcess(argv, 1, stdout="", stderr="no server")

    monkeypatch.setenv("TMUX", "/tmp/tmux-1000/default,123,0")
    monkeypatch.setattr(tmux_base.subprocess, "run", fake)
    assert launch.current_return_target() is None


def test_current_return_target_none_on_empty_pane(monkeypatch):
    # rc-0 empty stdout from the pane probe must answer None, not "" — the
    # seam default's `or None` guard, which callers map to RETURN_DETACH.
    monkeypatch.setenv("TMUX", "/tmp/tmux-1000/default,123,0")
    monkeypatch.setattr(
        tmux_base.subprocess,
        "run",
        lambda argv, **kwargs: subprocess.CompletedProcess(argv, 0, stdout="\n", stderr=""),
    )
    assert launch.current_return_target() is None


def test_start_detached_returns_window_id(fake_run, tmp_path: Path):
    assert launch.start_resolve_detached(tmp_path, "RID") == "@7"


def _make_run(project: Path, run_id: str = "RID") -> Path:
    """A run dir runs.is_run accepts — the state a resume/resolve launches over."""
    run_dir = runs.run_dir_for(project, run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "state.json").write_text("{}", encoding="utf-8")
    return run_dir


def test_start_detached_records_the_window_it_minted(fake_run, tmp_path: Path):
    run_dir = _make_run(tmp_path)
    launch.resume_detached(tmp_path, "RID")
    assert (run_dir / launch._CTL_WINDOW_FILE).read_text(encoding="utf-8") == "@7"


def test_start_detached_records_nothing_without_a_run(fake_run, tmp_path: Path):
    # A fresh `run` mints the only window carrying its run id — nothing to
    # disambiguate — and the record must never conjure a directory that
    # runs.is_run would then report as not a run. The explicit skip keeps this
    # expected case out of the OSError swallow; this test pins the outcome.
    launch.start_run_detached(tmp_path, "RID")
    assert not runs.run_dir_for(tmp_path, "RID").exists()


def test_no_record_into_a_dir_that_is_not_a_run(fake_run, tmp_path: Path):
    # The case the is_run guard actually gates (the missing-dir sibling above is
    # also covered by the OSError swallow — deleting the guard leaves it green):
    # a run-dir-shaped directory without state.json (pruned, partial). Here the
    # write would *succeed*, so only the guard keeps the sidecar out.
    run_dir = runs.run_dir_for(tmp_path, "RID")
    run_dir.mkdir(parents=True)
    launch.resume_detached(tmp_path, "RID")
    assert not (run_dir / launch._CTL_WINDOW_FILE).exists()


def test_start_detached_survives_an_unwritable_record(fake_run, tmp_path: Path, monkeypatch):
    # The window is already running by the time the record is written, so a
    # failed write degrades to the name scan rather than failing the launch.
    from bmad_loop import platform_util

    run_dir = _make_run(tmp_path)
    (run_dir / launch._CTL_WINDOW_FILE).mkdir()  # a dir, not a file
    # On win32 the replace-over-a-directory denial looks like the transient
    # sharing violation atomic_replace retries; skip the ~5s backoff.
    monkeypatch.setattr(platform_util, "_REPLACE_ATTEMPTS", 1)
    assert launch.start_resolve_detached(tmp_path, "RID") == "@7"


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX symlinks")
def test_symlinked_record_is_replaced_not_followed(fake_run, tmp_path: Path):
    # `atomic_write_text` follows a symlink under its default contract, and the
    # run dir lives under the project root every coding session can write — so a
    # session that plants a link here would aim this *host-side* write at any
    # path the user can write, reach a workspace-confined adapter otherwise
    # denies it. The write must land on the name, never on the link's target.
    run_dir = _make_run(tmp_path)
    outside = tmp_path / "pyproject.toml"
    outside.write_text("[project]\n", encoding="utf-8")
    record = run_dir / launch._CTL_WINDOW_FILE
    record.symlink_to(outside)

    assert launch.resume_detached(tmp_path, "RID") == "@7"  # the launch still succeeds
    assert outside.read_text(encoding="utf-8") == "[project]\n"  # not redirected
    # Clobbered, not refused: the record self-heals into a plain file, so the
    # next launch does not trip over a link left in place.
    assert not record.is_symlink()
    assert record.read_text(encoding="utf-8") == "@7"


def test_resume_reports_a_record_that_did_not_survive(fake_run, tmp_path: Path, monkeypatch):
    # The window id was captured, so start_detached returns it — but the record
    # did not land, which leaves ctl_window_id on the same ambiguous scan an
    # uncaptured id does. One signal for both, or the rest of the degradation
    # hides behind the success toast.
    _make_run(tmp_path)

    def boom(*_a, **_k):
        raise OSError("read-only file system")

    monkeypatch.setattr(launch, "atomic_write_text", boom)
    assert launch.resume_detached(tmp_path, "RID") is None


def test_resume_returns_the_id_when_the_record_survives(fake_run, tmp_path: Path):
    # The other half of the signal: a recorded window is reported plainly, so
    # the warning stays specific to real degradation.
    _make_run(tmp_path)
    assert launch.resume_detached(tmp_path, "RID") == "@7"


def test_failed_record_forgets_the_previous_one(fake_run, tmp_path: Path, monkeypatch):
    # A launch that cannot record the window it minted must not leave the
    # *previous* launch's id authoritative — that id names a window this launch
    # just superseded, so the honest state is no record at all.
    run_dir = _make_run(tmp_path)
    _write_record(tmp_path, "RID", "@2")

    def boom(*_a, **_k):
        raise OSError("disk full")

    monkeypatch.setattr(launch, "atomic_write_text", boom)
    launch.resume_detached(tmp_path, "RID")
    assert not (run_dir / launch._CTL_WINDOW_FILE).exists()


def test_failed_record_survives_a_non_oserror(fake_run, tmp_path: Path, monkeypatch):
    """`OSError` was too narrow to keep the docstring's promise that a failed
    write must not fail the launch. `atomic_write_text` resolves the path before
    its own try, and below 3.13 `Path.resolve` reports a symlink loop as
    `RuntimeError` — so a run dir reached through a looping link crashed the
    launch of a window that is *already running*, on the 3.11/3.12 legs.

    The fault is injected rather than built from a real symlink loop on purpose:
    3.13+ resolves loops without raising, so a loop-based version would pass on
    the interpreter this suite usually runs and only ever fail on the older legs
    — green here, red in CI, for a guard that was never exercised. Same reasoning
    as tests/test_engine.py's `test_failed_rollback_does_not_displace_the_commit_failure`.
    """
    run_dir = _make_run(tmp_path)
    _write_record(tmp_path, "RID", "@2")

    def boom(*_a, **_k):
        raise RuntimeError("Symlink loop from '/x'")

    monkeypatch.setattr(launch, "atomic_write_text", boom)
    launch.resume_detached(tmp_path, "RID")  # must not raise
    assert not (run_dir / launch._CTL_WINDOW_FILE).exists()


def test_record_survives_a_raising_window_tag(fake_run, tmp_path: Path, monkeypatch):
    # Record-before-tag ordering: the seam declares set_window_option
    # best-effort, but a non-conforming backend raising from it must not cost
    # the record — swap the two calls in start_detached and this fails.
    run_dir = _make_run(tmp_path)

    def boom(self, *_a, **_k):
        raise MultiplexerError("tag failed")

    monkeypatch.setattr(type(get_multiplexer()), "set_window_option", boom)
    with pytest.raises(MultiplexerError):
        launch.resume_detached(tmp_path, "RID")
    assert (run_dir / launch._CTL_WINDOW_FILE).read_text(encoding="utf-8") == "@7"


def test_uncaptured_window_id_forgets_the_previous_record(monkeypatch, tmp_path: Path):
    # new-window answered no id: nothing to record, and the stale record must go.
    run_dir = _make_run(tmp_path)
    _write_record(tmp_path, "RID", "@2")

    def fake(argv, **kwargs):
        # rc 0 throughout, incl. has-session: the ctl session exists, and
        # new-window succeeds but answers no id on stdout.
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    monkeypatch.setattr(tmux_base.subprocess, "run", fake)
    monkeypatch.setattr(tmux_base.shutil, "which", lambda name: f"/usr/bin/{name}")
    assert launch.start_resolve_detached(tmp_path, "RID") is None
    assert not (run_dir / launch._CTL_WINDOW_FILE).exists()


def test_record_round_trips_a_session_qualified_id(monkeypatch, tmp_path: Path):
    # The re-prove is a pure string match, so any qualified form works as long
    # as the mint and the window_id column agree (multiplexer's symmetry note);
    # `session:@N` is the shape psmux actually emits on both sides.
    _ctl_listing(
        monkeypatch,
        "bmad-loop-ctl:@1\trun-RID\nbmad-loop-ctl:@2\tresume-RID\n",
    )
    _write_record(tmp_path, "RID", "bmad-loop-ctl:@2")
    assert launch.ctl_window_id(tmp_path, "RID") == "bmad-loop-ctl:@2"


def test_record_with_trailing_newline_still_matches(monkeypatch, tmp_path: Path):
    # A newline-terminated record (hand-edited, foreign writer) must not fail
    # the `recorded in matches` check and silently answer the parked corpse.
    _ctl_listing(monkeypatch, "@1\trun-RID\n@2\tresume-RID\n")
    _write_record(tmp_path, "RID", "@2\n")
    assert launch.ctl_window_id(tmp_path, "RID") == "@2"


def test_prune_ctl_windows(monkeypatch, tmp_path: Path):
    from bmad_loop import runs

    mine = runs.project_tag(tmp_path)
    # one live run (this process's pid); the others have no run dir
    live = tmp_path / ".bmad-loop" / "runs" / "20260101-000000-live"
    live.mkdir(parents=True)
    (live / "state.json").write_text("{}")
    runs.write_pid(live)

    # window format is window_id\twindow_name\t@bmad_project
    windows = (
        "@1\t0\t\n"  # the session's initial shell — not a run window
        f"@2\trun-20260101-000000-live\t{mine}\n"  # live run, ours — keep
        f"@3\tsweep-20260101-000000-dead\t{mine}\n"  # tagged-ours orphan — kill
        "@5\tsweep-20260101-000000-other\t/some/other/project\n"  # another project — skip
        f"@4\tresume-20260101-000000-cur\t{mine}\n"  # matches, but is the current window
    )
    killed: list[list[str]] = []

    def fake(argv, **kwargs):
        verb = argv[1]
        if verb == "has-session":
            return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")
        if verb == "display-message":  # we are sitting in @4
            return subprocess.CompletedProcess(argv, 0, stdout="@4\n", stderr="")
        if verb == "list-windows":
            return subprocess.CompletedProcess(argv, 0, stdout=windows, stderr="")
        if verb == "kill-window":
            killed.append(list(argv))
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    monkeypatch.setenv("TMUX", "/tmp/tmux-1000/default,123,0")  # we sit in a pane of @4
    monkeypatch.setattr(tmux_base.subprocess, "run", fake)
    monkeypatch.setattr(tmux_base.shutil, "which", lambda name: f"/usr/bin/{name}")

    assert launch.prunable_ctl_windows(tmp_path) == ["sweep-20260101-000000-dead"]
    assert killed == []  # dry-run view kills nothing
    assert launch.prune_ctl_windows(tmp_path) == ["sweep-20260101-000000-dead"]
    assert killed == [["tmux", "kill-window", "-t", "@3"]]


def test_prune_ctl_windows_skips_invalid_run_ids(monkeypatch, tmp_path: Path):
    """A ctl-window name is untrusted input (anyone can rename a tmux window).
    Stripping the kind prefix off `run-../../x` would hand run_dir_for a
    traversing id, steering the liveness read — and, for an untagged window,
    the run-dir ownership fallback — at a path outside the runs dir. Reject
    before recomposing (mirrors runs.prunable_sessions)."""
    from bmad_loop import runs

    mine = runs.project_tag(tmp_path)
    # a real runs dir, so the traversal has an existing anchor to climb from
    (tmp_path / ".bmad-loop" / "runs").mkdir(parents=True)
    # where the un-gated recomposition of `run-../../planted` would land: an
    # outside dir whose state.json would otherwise claim the untagged window
    planted = tmp_path / "planted"
    planted.mkdir()
    (planted / "state.json").write_text("{}")

    windows = (
        f"@2\tsweep-20260101-000000-dead\t{mine}\n"  # legit orphan — still killed
        f"@3\trun-../../x\t{mine}\n"  # traversal — skipped
        f"@5\tsweep-a.b\t{mine}\n"  # invalid charset — skipped
        "@6\trun-../../planted\t\n"  # untagged — outside state.json must not claim it
    )
    killed: list[list[str]] = []

    def fake(argv, **kwargs):
        verb = argv[1]
        if verb == "has-session":
            return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")
        if verb == "display-message":  # current window is none of the rows
            return subprocess.CompletedProcess(argv, 0, stdout="@1\n", stderr="")
        if verb == "list-windows":
            return subprocess.CompletedProcess(argv, 0, stdout=windows, stderr="")
        if verb == "kill-window":
            killed.append(list(argv))
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    monkeypatch.setenv("TMUX", "/tmp/tmux-1000/default,123,0")
    monkeypatch.setattr(tmux_base.subprocess, "run", fake)
    monkeypatch.setattr(tmux_base.shutil, "which", lambda name: f"/usr/bin/{name}")

    assert launch.prunable_ctl_windows(tmp_path) == ["sweep-20260101-000000-dead"]
    assert launch.prune_ctl_windows(tmp_path) == ["sweep-20260101-000000-dead"]
    assert killed == [["tmux", "kill-window", "-t", "@2"]]


def test_prune_ctl_windows_no_session(monkeypatch, tmp_path: Path):
    def fake(argv, **kwargs):  # has-session reports the ctl session is gone
        return subprocess.CompletedProcess(argv, 1, stdout="", stderr="")

    monkeypatch.setattr(tmux_base.subprocess, "run", fake)
    monkeypatch.setattr(tmux_base.shutil, "which", lambda name: f"/usr/bin/{name}")
    assert launch.prune_ctl_windows(tmp_path) == []


def test_select_ctl_window_id_argv(fake_run):
    launch.select_ctl_window_id("@7")
    assert fake_run.calls == [["tmux", "select-window", "-t", "@7"]]


def test_in_ctl_session(monkeypatch):
    # in_ctl_session is backend-honest: it trusts current_session(), which is
    # None whenever this process is not inside the selected multiplexer (the
    # old direct TMUX sniff lives in the tmux backend's _display_message now —
    # see test_in_ctl_session_outside_tmux).
    monkeypatch.setattr(launch, "current_session", lambda: "bmad-loop-ctl")
    assert launch.in_ctl_session() is True
    monkeypatch.setattr(launch, "current_session", lambda: "some-other-session")
    assert launch.in_ctl_session() is False
    monkeypatch.setattr(launch, "current_session", lambda: None)
    assert launch.in_ctl_session() is False  # not inside the multiplexer


def test_in_ctl_session_outside_tmux(monkeypatch):
    # End-to-end through the real tmux backend: outside tmux (no TMUX env) the
    # backend's current_session() is None without shelling out, even when a
    # live server would answer display-message for some other client.
    def boom(*_a, **_k):
        raise AssertionError("outside tmux, in_ctl_session must not shell out")

    monkeypatch.delenv("TMUX", raising=False)
    monkeypatch.setattr(tmux_base.subprocess, "run", boom)
    assert launch.in_ctl_session() is False


def test_detach_client_argv(fake_run):
    launch.detach_client()
    assert fake_run.calls == [["tmux", "detach-client"]]


def _return_fake(
    monkeypatch, *, win="@5", option="=main:%9", switch_rc=0, fallback_rc=0, detach_rc=0
):
    """Script tmux for return_attached_client: display-message -> window id,
    show-options -> the recorded RETURN_OPTION, switch-client -t -> switch_rc,
    switch-client -l -> fallback_rc, detach-client -> detach_rc.
    return_attached_client runs inside a ctl window, so TMUX is set (the
    backend's current_window_id answers None otherwise)."""
    monkeypatch.setenv("TMUX", "/tmp/tmux-1000/default,123,0")
    calls: list[list[str]] = []

    def fake(argv, **kwargs):
        calls.append(list(argv))
        verb = argv[1]
        if verb == "display-message":
            out, rc = (f"{win}\n", 0) if win is not None else ("", 1)
        elif verb == "show-options":
            out, rc = (f"{option}\n" if option else "", 0)
        elif verb == "switch-client" and argv[2] == "-t":
            out, rc = "", switch_rc
        elif verb == "switch-client" and argv[2] == "-l":
            out, rc = "", fallback_rc
        elif verb == "detach-client":
            out, rc = "", detach_rc
        else:
            out, rc = "", 0
        return subprocess.CompletedProcess(argv, rc, stdout=out, stderr="")

    monkeypatch.setattr(tmux_base.subprocess, "run", fake)
    monkeypatch.setattr(tmux_base.shutil, "which", lambda name: f"/usr/bin/{name}")
    return calls


def test_return_attached_client_switches_to_pane(monkeypatch):
    calls = _return_fake(monkeypatch, option="=main:%9")
    assert launch.return_attached_client() is launch.ReturnOutcome.RETURNED
    assert ["tmux", "switch-client", "-t", "=main:%9"] in calls
    assert ["tmux", "set-option", "-wu", "-t", "@5", "@bmad_return_pane"] in calls
    assert ["tmux", "switch-client", "-l"] not in calls  # no fallback when -t works
    assert not any(c[1] == "detach-client" for c in calls)


def test_return_attached_client_switch_fallback(monkeypatch):
    calls = _return_fake(monkeypatch, option="=main:%9", switch_rc=1)
    assert launch.return_attached_client() is launch.ReturnOutcome.RETURNED
    assert ["tmux", "switch-client", "-l"] in calls
    # the fallback returned a client too, so the option is consumed — without
    # this the unset could regress to primary-success-only and stay green
    assert ["tmux", "set-option", "-wu", "-t", "@5", "@bmad_return_pane"] in calls


def test_return_attached_client_switch_fails_stays_attended(monkeypatch):
    """Stale target plus no last client: the client never left this window, so
    the human is still in front of it — ATTENDED, and RETURN_OPTION stays set
    or the post-exit trailer loses its retry."""
    calls = _return_fake(monkeypatch, option="=main:%9", switch_rc=1, fallback_rc=1)
    assert launch.return_attached_client() is launch.ReturnOutcome.ATTENDED
    assert ["tmux", "switch-client", "-l"] in calls  # fallback was attempted
    assert not any(c[1] == "set-option" for c in calls)  # option survives


def test_return_attached_client_detach_fails_is_unreachable(monkeypatch):
    """`detach-client` fails only when there is no current client, so a failed
    detach is positive evidence that nobody is watching — the opposite of a
    failed switch, and NOT the same answer. RETURN_OPTION still survives."""
    calls = _return_fake(monkeypatch, option="detach", detach_rc=1)
    assert launch.return_attached_client() is launch.ReturnOutcome.UNREACHABLE
    assert ["tmux", "detach-client"] in calls
    assert not any(c[1] == "set-option" for c in calls)


def test_return_attached_client_detaches(monkeypatch):
    calls = _return_fake(monkeypatch, option="detach")
    assert launch.return_attached_client() is launch.ReturnOutcome.RETURNED
    assert ["tmux", "detach-client"] in calls
    assert ["tmux", "set-option", "-wu", "-t", "@5", "@bmad_return_pane"] in calls
    assert not any(c[1] == "switch-client" for c in calls)


def test_return_attached_client_noop_when_unset(monkeypatch):
    """No return target recorded — a plain foreground sweep. Nothing was
    attempted, so nothing can be concluded about who is at the terminal: the
    conservative ATTENDED, never UNREACHABLE."""
    calls = _return_fake(monkeypatch, option="")
    assert launch.return_attached_client() is launch.ReturnOutcome.ATTENDED
    assert not any(c[1] in ("switch-client", "detach-client", "set-option") for c in calls)


def test_return_attached_client_noop_without_tmux(monkeypatch):
    # This is a NEGATIVE gate test: the module-wide force_tmux_backend pin makes
    # mux_usable trust the backend regardless of available(), so drop the pin
    # (and the pinned selection) or — inside a real tmux session, TMUX set —
    # the trusted path reaches display-message and shells out after all.
    monkeypatch.delenv("BMAD_LOOP_MUX_BACKEND", raising=False)
    get_multiplexer.cache_clear()
    ran: list = []
    monkeypatch.setattr(tmux_base.shutil, "which", lambda name: None)
    monkeypatch.setattr(tmux_base.subprocess, "run", lambda *a, **k: ran.append(a))
    assert launch.return_attached_client() is launch.ReturnOutcome.ATTENDED
    assert ran == []  # never shells out when tmux is missing


def test_decision_pending_true(tmp_path: Path):
    from bmad_loop.journal import Journal

    rd = tmp_path / "run"
    j = Journal(rd)
    j.append("triage-done")
    j.append("decision-pending", dw_id="DW-90", question="?")
    assert launch.decision_pending(rd) is True


def test_decision_pending_false_after_answer(tmp_path: Path):
    from bmad_loop.journal import Journal

    rd = tmp_path / "run"
    j = Journal(rd)
    j.append("decision-pending", dw_id="DW-90", question="?")
    j.append("decision-answered", dw_id="DW-90", key="1")
    assert launch.decision_pending(rd) is False


def test_decision_pending_false_when_empty(tmp_path: Path):
    assert launch.decision_pending(tmp_path / "missing") is False


def test_attach_plan_prefers_ctl_when_decision_pending(monkeypatch):
    monkeypatch.delenv("TMUX", raising=False)
    monkeypatch.setattr(launch, "ctl_window_id", lambda proj, rid: "@2")
    monkeypatch.setattr(launch, "session_exists", lambda s: True)
    monkeypatch.setattr(launch, "decision_pending", lambda rd: True)
    selected: list[str] = []
    monkeypatch.setattr(launch, "select_ctl_window_id", lambda w: selected.append(w))
    argv, return_window = launch.attach_plan(Path("/proj"), "RID")
    assert argv == ["tmux", "attach", "-t", "=bmad-loop-ctl"]
    assert return_window == "@2"
    assert selected == ["@2"]


def test_attach_plan_prefers_ctl_when_no_agent_session(monkeypatch):
    monkeypatch.delenv("TMUX", raising=False)
    monkeypatch.setattr(launch, "ctl_window_id", lambda proj, rid: "@2")
    monkeypatch.setattr(launch, "session_exists", lambda s: False)
    monkeypatch.setattr(launch, "decision_pending", lambda rd: False)
    monkeypatch.setattr(launch, "select_ctl_window_id", lambda w: None)
    argv, return_window = launch.attach_plan(Path("/proj"), "RID")
    assert argv == ["tmux", "attach", "-t", "=bmad-loop-ctl"]
    assert return_window == "@2"


def test_attach_plan_agent_session_when_no_decision(monkeypatch):
    monkeypatch.delenv("TMUX", raising=False)
    monkeypatch.setattr(launch, "ctl_window_id", lambda proj, rid: None)
    monkeypatch.setattr(launch, "session_exists", lambda s: True)
    monkeypatch.setattr(launch, "decision_pending", lambda rd: False)
    assert launch.attach_plan(Path("/proj"), "RID") == (
        ["tmux", "attach", "-t", "=bmad-loop-RID"],
        None,
    )


def test_attach_plan_none_when_nothing_to_attach(monkeypatch):
    monkeypatch.setattr(launch, "ctl_window_id", lambda proj, rid: None)
    monkeypatch.setattr(launch, "session_exists", lambda s: False)
    monkeypatch.setattr(launch, "decision_pending", lambda rd: False)
    assert launch.attach_plan(Path("/proj"), "RID") is None


def test_run_captured_merges_streams(monkeypatch):
    def fake(argv, **kwargs):
        assert argv[:3] == [sys.executable, "-m", "bmad_loop.cli"]
        assert argv[3:] == ["validate", "--project", "/p"]
        # encoding= puts subprocess in text mode without setting the `text`
        # kwarg, so assert on the decoding that is actually pinned. UTF-8 at
        # errors="replace" is the point: text=True would decode with the
        # locale encoding at errors="strict" (the #200 failure family).
        assert kwargs.get("capture_output")
        assert kwargs.get("encoding") == "utf-8" and kwargs.get("errors") == "replace"
        return subprocess.CompletedProcess(argv, 1, stdout="ok line", stderr="FAIL line\n")

    monkeypatch.setattr(launch.subprocess, "run", fake)
    rc, out = launch.run_captured(["validate", "--project", "/p"])
    assert rc == 1
    assert out == "ok line\nFAIL line\n"


def test_run_captured_streams_keeps_stderr_off_stdout(monkeypatch):
    """The reason the seam exists: a caller parsing stdout as one JSON document
    must not receive a dependency's stderr warning appended to it. Merged, this
    is exactly the input that makes json.loads raise "Extra data"."""
    payload = '{"schema_version": 1, "ok": true}'

    def fake(argv, **kwargs):
        return subprocess.CompletedProcess(
            argv, 0, stdout=payload, stderr="DeprecationWarning: whatever\n"
        )

    monkeypatch.setattr(launch.subprocess, "run", fake)
    rc, out, err = launch.run_captured_streams(["validate", "--project", "/p", "--json"])
    assert rc == 0
    assert out == payload
    assert "DeprecationWarning" in err
    assert json.loads(out) == {"schema_version": 1, "ok": True}
    # and the merging caller still gets the blob it wants, from the same call
    assert launch.run_captured(["validate", "--project", "/p"])[1] == (
        payload + "\nDeprecationWarning: whatever\n"
    )


def test_run_captured_real_subprocess():
    """End-to-end: the module really is invocable as `python -m bmad_loop.cli`."""
    rc, out = launch.run_captured(["--version"])
    assert rc == 0
    assert "bmad-loop" in out


def test_run_captured_streams_real_subprocess():
    """The separated form against the real CLI: a `--json` document parses off
    stdout alone, with stderr empty (the machine.py purity contract)."""
    rc, out, err = launch.run_captured_streams(["--version"])
    assert rc == 0
    assert "bmad-loop" in out
    assert err == ""
