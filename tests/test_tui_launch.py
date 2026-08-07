"""tui.launch builds exact tmux/CLI argv — verified against monkeypatched
subprocess so no real tmux server is touched, plus one real-subprocess
sanity check of the captured path.

The tmux invocations now live in the multiplexer backend (launch drives the
seam), so the tmux subprocess/which seams are patched on ``tmux_base`` (the
shared backend base where the spawn primitive lives); the captured read-only
path still shells out from ``launch`` itself."""

from __future__ import annotations

import json
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

from bmad_loop.adapters import tmux_base
from bmad_loop.adapters.multiplexer import get_multiplexer
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


def test_ctl_window_id_matches_run_id_suffix(monkeypatch):
    # The id, not the name: consumers replay the value as select/kill/option
    # targets, where a by-name resolve can land on a duplicate.
    def fake(argv, **kwargs):
        out = "@1\trun-AAAA\n@2\tsweep-RID\n@3\tresume-BBBB\n" if argv[1] == "list-windows" else ""
        return subprocess.CompletedProcess(argv, 0, stdout=out, stderr="")

    monkeypatch.setattr(tmux_base.subprocess, "run", fake)
    monkeypatch.setattr(tmux_base.shutil, "which", lambda name: f"/usr/bin/{name}")
    assert launch.ctl_window_id("RID") == "@2"
    assert launch.ctl_window_id("CCCC") is None


def test_ctl_window_id_skips_empty_id_rows(monkeypatch):
    # An empty id must never be returned as a target — an empty `-t` resolves
    # against the current window. psmux's qualifier passes a falsy id through.
    def fake(argv, **kwargs):
        out = "\tsweep-RID\n@7\tsweep-RID\n" if argv[1] == "list-windows" else ""
        return subprocess.CompletedProcess(argv, 0, stdout=out, stderr="")

    monkeypatch.setattr(tmux_base.subprocess, "run", fake)
    monkeypatch.setattr(tmux_base.shutil, "which", lambda name: f"/usr/bin/{name}")
    assert launch.ctl_window_id("RID") == "@7"


def test_kill_ctl_window_kills_by_resolved_id_not_a_name_token(monkeypatch):
    # The kill replays the id this listing resolved, never a `=session:name`
    # token the backend would resolve again. Which of two same-named windows
    # the scan picks is unchanged (first match, `@7`); what the id buys is that
    # a rename or a new window between two verbs cannot re-point the second.
    calls: list[list[str]] = []

    def fake(argv, **kwargs):
        calls.append(list(argv))
        out = "@2\trun-x\n@7\tsweep-RID\n@9\tsweep-RID\n" if argv[1] == "list-windows" else ""
        return subprocess.CompletedProcess(argv, 0, stdout=out, stderr="")

    monkeypatch.setattr(tmux_base.subprocess, "run", fake)
    monkeypatch.setattr(tmux_base.shutil, "which", lambda name: f"/usr/bin/{name}")
    launch.kill_ctl_window("RID")
    assert ["tmux", "kill-window", "-t", "@7"] in calls


def test_ctl_window_id_no_session_or_tmux(monkeypatch):
    def fake(argv, **kwargs):
        return subprocess.CompletedProcess(argv, 1, stdout="", stderr="no session")

    monkeypatch.setattr(tmux_base.subprocess, "run", fake)
    monkeypatch.setattr(tmux_base.shutil, "which", lambda name: f"/usr/bin/{name}")
    assert launch.ctl_window_id("RID") is None
    monkeypatch.setattr(tmux_base.shutil, "which", lambda name: None)
    assert launch.ctl_window_id("RID") is None  # no subprocess call attempted


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
    monkeypatch.setattr(launch, "ctl_window_id", lambda rid: "@2")
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
    monkeypatch.setattr(launch, "ctl_window_id", lambda rid: "@2")
    monkeypatch.setattr(launch, "session_exists", lambda s: False)
    monkeypatch.setattr(launch, "decision_pending", lambda rd: False)
    monkeypatch.setattr(launch, "select_ctl_window_id", lambda w: None)
    argv, return_window = launch.attach_plan(Path("/proj"), "RID")
    assert argv == ["tmux", "attach", "-t", "=bmad-loop-ctl"]
    assert return_window == "@2"


def test_attach_plan_agent_session_when_no_decision(monkeypatch):
    monkeypatch.delenv("TMUX", raising=False)
    monkeypatch.setattr(launch, "ctl_window_id", lambda rid: None)
    monkeypatch.setattr(launch, "session_exists", lambda s: True)
    monkeypatch.setattr(launch, "decision_pending", lambda rd: False)
    assert launch.attach_plan(Path("/proj"), "RID") == (
        ["tmux", "attach", "-t", "=bmad-loop-RID"],
        None,
    )


def test_attach_plan_none_when_nothing_to_attach(monkeypatch):
    monkeypatch.setattr(launch, "ctl_window_id", lambda rid: None)
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
