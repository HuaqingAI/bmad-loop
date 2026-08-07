"""Live psmux gate: cross-project ctl-window prune isolation on a real server.

The unit acceptance test (test_psmux_backend) proves the candidate scan with a
faked subprocess; what no unit layer can prove is the transport half — a real
psmux server holding two projects' windows on one shared control session, a
prune in one project, and the other project's window *and its option keys*
surviving the kill. Same zero-token contract as test_opencode_live: parked
windows run a plain `exit 0`, no coding CLI is ever launched.

Windows-local by construction (psmux registers for win32 only); skipped
everywhere else, and when psmux is absent or an unsupported version.
"""

from __future__ import annotations

import re
import shutil
import sys
import uuid
from pathlib import Path

import pytest

from bmad_loop import runs
from bmad_loop.adapters.psmux_backend import PsmuxMultiplexer
from bmad_loop.adapters.tmux_base import TmuxError
from bmad_loop.tui import launch

HAVE_PSMUX = sys.platform == "win32" and shutil.which("psmux") is not None
pytestmark = pytest.mark.skipif(not HAVE_PSMUX, reason="requires Windows with psmux on PATH")

PARKED_ARGV = ["pwsh", "-NoProfile", "-Command", "exit 0"]  # zero tokens, parks on read


def test_prune_kills_only_the_owning_projects_window(tmp_path: Path, monkeypatch):
    mux = PsmuxMultiplexer()
    if not mux.available():
        pytest.skip("psmux present but not an admitted version")
    session = f"bmad-loop-test-{uuid.uuid4().hex[:8]}"
    proj_a = tmp_path / "proj-a"
    proj_b = tmp_path / "proj-b"
    proj_a.mkdir()
    proj_b.mkdir()
    body_ok = False
    try:
        mux.new_session(session, tmp_path)
        win_a = mux.new_parked_window(session, "run-20260726-1", proj_a, PARKED_ARGV, "@r")
        win_b = mux.new_parked_window(session, "run-20260726-2", proj_b, PARKED_ARGV, "@r")
        # A degraded mint hands back "" or a bare id — fail loud here rather
        # than let rsplit or a mis-scoped option write misdiagnose the prune.
        assert re.fullmatch(r"[^:]+:@\d+", win_a), win_a
        assert re.fullmatch(r"[^:]+:@\d+", win_b), win_b
        mux.set_window_option(win_a, runs.PROJECT_OPTION, runs.project_tag(proj_a))
        mux.set_window_option(win_b, runs.PROJECT_OPTION, runs.project_tag(proj_b))
        # set_window_option declines silently (warn-only): prove the tags
        # landed before asserting anything about the prune's discrimination.
        assert mux.show_window_option(win_a, runs.PROJECT_OPTION) == runs.project_tag(proj_a)
        assert mux.show_window_option(win_b, runs.PROJECT_OPTION) == runs.project_tag(proj_b)
        # A hand-written user option carrying window A's digits in the OLD
        # (pre-marker) shape: the seam's sweeps must never claim it.
        digits_a = win_a.rsplit("@", 1)[1]
        foreign = f"@theme_@{digits_a}"
        proc = mux._run(["set-option", "-t", session, foreign, "user"], check=False)
        assert proc.returncode == 0, proc.stderr

        monkeypatch.setattr(launch, "CTL_SESSION", session)
        monkeypatch.setattr(launch, "get_multiplexer", lambda: mux)
        # Pin "outside any pane": when pytest itself runs inside psmux the
        # target-less probe can resolve the test's own window and exclude it.
        monkeypatch.setattr(mux, "current_window_id", lambda: None)
        monkeypatch.setattr(runs, "engine_alive", lambda _dir: False)

        assert launch.prune_ctl_windows(proj_a) == ["run-20260726-1"]

        live = mux.list_window_ids(session)
        assert win_a not in live
        assert win_b in live
        options = mux._scoped_options(session) or {}
        key_a = mux._scoped_option_key(runs.PROJECT_OPTION, digits_a)
        key_b = mux._scoped_option_key(runs.PROJECT_OPTION, win_b.rsplit("@", 1)[1])
        assert key_a not in options  # freed by the verified kill
        assert options.get(key_b) == runs.project_tag(proj_b)  # untouched
        assert options.get(foreign) == "user"  # foreign key survives every sweep
        body_ok = True
    finally:
        # kill_session is a best-effort backstop; verify it worked so a real
        # server never leaks silently off a green (or already-failing) run.
        mux.kill_session(session)
        try:
            leaked = mux.has_session(session)
        except TmuxError:
            leaked = True
        if leaked:
            note = f"live-gate session {session} survived teardown; kill it manually"
            # Only raise when the body passed. pytest.fail() here on an
            # already-failing run replaces the real diagnostic — the leak takes
            # over the summary line and the assertion drops to a chained
            # "during handling" frame — so that run keeps the warning instead.
            if body_ok:
                pytest.fail(note)
            print(f"warning: {note}", file=sys.stderr)
