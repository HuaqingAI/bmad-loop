"""Unit tests for the RecoveryFlow collaborator (issue #244 PR 2/2).

RecoveryFlow was carved out of Engine's rollback/preserve cluster. These
exercise it in isolation — built from narrow deps + stub engine callbacks, no
Engine instance — which is the point of the extraction. End-to-end behavior
under a real Engine stays covered by test_engine.py.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from conftest import git

from bmad_loop import verify
from bmad_loop.gates import ATTENTION_FILE
from bmad_loop.model import Phase, StoryTask
from bmad_loop.policy import GatesPolicy, LimitsPolicy, NotifyPolicy, Policy, ScmPolicy
from bmad_loop.recovery_flow import RecoveryFlow
from bmad_loop.verify import GitError, rev_parse_head
from bmad_loop.workspace import Workspace

QUIET = NotifyPolicy(desktop=False, file=True)


def _policy(**scm) -> Policy:
    return Policy(
        gates=GatesPolicy(mode="none"),
        notify=QUIET,
        scm=ScmPolicy(**scm),
        limits=LimitsPolicy(),
    )


class _RecordingJournal:
    def __init__(self) -> None:
        self.entries: list[tuple[str, dict]] = []

    def append(self, event: str, **fields) -> None:
        self.entries.append((event, fields))

    def events(self) -> list[str]:
        return [e for e, _ in self.entries]

    def fields(self, event: str) -> dict:
        for e, f in self.entries:
            if e == event:
                return f
        raise KeyError(event)


class _Pause(Exception):
    """Stand-in for the engine's RunPaused, raised by the injected escalation_pause
    (and the escalate callback) so these tests need not import the engine."""

    def __init__(self, reason: str, story_key: str) -> None:
        super().__init__(reason)
        self.reason = reason
        self.story_key = story_key


def _fake_workspace(root: Path, *, output=None, impl=None, plan=None):
    """A duck-typed Workspace for the pure `protected_relpaths` predicate — root +
    the three artifact folders, no git required."""
    root = Path(root)
    paths = SimpleNamespace(
        output_folder=output if output is not None else root / "_bmad-output",
        implementation_artifacts=(
            impl if impl is not None else root / "_bmad-output" / "implementation-artifacts"
        ),
        planning_artifacts=(
            plan if plan is not None else root / "_bmad-output" / "planning-artifacts"
        ),
    )
    return SimpleNamespace(root=root, paths=paths)


def _make_flow(
    *,
    workspace,
    paths=None,
    policy: Policy | None = None,
    state=None,
    journal: _RecordingJournal | None = None,
    run_dir: Path | None = None,
):
    """Build a RecoveryFlow wired to recording stubs. The returned flow carries a
    ``.calls`` namespace tallying the injected callbacks for assertions. ``paths``
    defaults to the workspace root (so the flow reads as "main checkout"); pass a
    different ``repo_root`` to simulate a mounted unit worktree."""
    calls = SimpleNamespace(saves=0, emits=[], pauses=[], escalates=[])

    def _save() -> None:
        calls.saves += 1

    def _emit(stage, task=None, **fields):
        calls.emits.append(stage)
        return None

    def _escalate(task, reason) -> None:
        calls.escalates.append((task.story_key, reason))
        raise _Pause(reason, task.story_key)

    def _pause(reason, story_key="", *, cause=None):
        calls.pauses.append((reason, story_key))
        raise _Pause(reason, story_key)

    flow = RecoveryFlow(
        paths=paths if paths is not None else SimpleNamespace(repo_root=workspace.root),
        policy=policy if policy is not None else _policy(),
        state=state if state is not None else SimpleNamespace(run_id="run-1"),
        journal=journal if journal is not None else _RecordingJournal(),
        run_dir=run_dir if run_dir is not None else workspace.root,
        workspace_get=lambda: workspace,
        emit=_emit,
        save=_save,
        escalate=_escalate,
        escalation_pause=_pause,
    )
    flow.calls = calls
    return flow


def _task(repo: Path, story_key: str = "1-1-a") -> StoryTask:
    task = StoryTask(story_key=story_key, epic=1)
    task.baseline_commit = rev_parse_head(repo)
    task.baseline_untracked = []
    return task


# --------------------------------------------------------------- protected paths


def test_protected_relpaths_lists_bmad_folders(tmp_path):
    flow = _make_flow(workspace=_fake_workspace(tmp_path))
    assert flow.protected_relpaths() == (
        "_bmad-output",
        "_bmad-output/implementation-artifacts",
        "_bmad-output/planning-artifacts",
    )


def test_protected_relpaths_skips_folders_outside_repo(tmp_path):
    # a planning folder configured outside the repo raises ValueError on
    # relative_to and is dropped — nothing to protect there.
    outside = tmp_path.parent / "elsewhere" / "planning"
    ws = _fake_workspace(tmp_path, plan=outside)
    flow = _make_flow(workspace=ws)
    assert flow.protected_relpaths() == (
        "_bmad-output",
        "_bmad-output/implementation-artifacts",
    )


def test_protected_relpaths_drops_repo_root_prefix(tmp_path):
    # a folder == repo root would relativize to "." and, used as a keep/exclude
    # prefix, cover the whole tree — it must be dropped.
    ws = _fake_workspace(tmp_path, output=tmp_path)
    flow = _make_flow(workspace=ws)
    assert "." not in flow.protected_relpaths()
    assert "_bmad-output/implementation-artifacts" in flow.protected_relpaths()


# --------------------------------------------------------------- rollback_or_pause


def test_rollback_skips_clean_tree(project):
    ws = Workspace.default(project)
    flow = _make_flow(workspace=ws, policy=_policy(rollback_on_failure=False))
    task = _task(project.project)  # HEAD == baseline, no untracked

    flow.rollback_or_pause(task)  # must NOT raise

    assert "rollback-skipped-clean" in flow.journal.events()
    assert flow.calls.pauses == []
    assert flow.calls.emits == []  # no pre/post_rollback on the clean short-circuit


def test_rollback_auto_resets_when_flag_on(project):
    repo = project.project
    ws = Workspace.default(project)
    flow = _make_flow(workspace=ws, policy=_policy(rollback_on_failure=True))
    task = _task(repo)
    (repo / "src.txt").write_text("committed attempt\n")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "attempt work")

    flow.rollback_or_pause(task)  # must NOT raise

    assert rev_parse_head(repo) == task.baseline_commit  # reset to baseline
    assert flow.calls.emits == ["pre_rollback", "post_rollback"]
    assert flow.calls.pauses == []
    assert "rollback-auto" in flow.journal.events()


def test_rollback_off_pauses_and_leaves_tree(project):
    repo = project.project
    ws = Workspace.default(project)
    flow = _make_flow(workspace=ws, policy=_policy(rollback_on_failure=False))
    task = _task(repo)
    (repo / "dirty.txt").write_text("uncommitted work\n")  # untracked → dirty

    with pytest.raises(_Pause):
        flow.rollback_or_pause(task)

    assert (repo / "dirty.txt").exists()  # tree untouched
    assert flow.calls.pauses  # escalation_pause fired
    assert "rollback-manual-required" in flow.journal.events()


def test_rollback_in_unit_worktree_auto_recovers_even_when_off(project, tmp_path):
    # workspace.root != paths.repo_root ⇒ a mounted unit worktree: rollback OFF
    # still auto-recovers (the flag gates in-place recovery only, #161).
    repo = project.project
    ws = Workspace.default(project)
    flow = _make_flow(
        workspace=ws,
        paths=SimpleNamespace(repo_root=tmp_path / "some-other-main-checkout"),
        policy=_policy(rollback_on_failure=False),
    )
    task = _task(repo)
    (repo / "src.txt").write_text("worktree attempt\n")

    flow.rollback_or_pause(task)  # must NOT pause despite rollback OFF

    assert flow.calls.pauses == []
    assert flow.calls.emits == ["pre_rollback", "post_rollback"]
    assert "rollback-auto" in flow.journal.events()


def test_rollback_resolved_cause_auto_recovers_when_off(project):
    repo = project.project
    ws = Workspace.default(project)
    flow = _make_flow(workspace=ws, policy=_policy(rollback_on_failure=False))
    task = _task(repo)
    (repo / "src.txt").write_text("re-drive edit\n")

    flow.rollback_or_pause(task, cause="resolved")  # human-initiated → never pauses

    assert flow.calls.pauses == []
    assert "rollback-auto" in flow.journal.events()


def test_rollback_dirty_check_git_fault_degrades_to_dirty(project, monkeypatch):
    # #156: an un-determinable dirty check assumes dirty — OFF then pauses rather
    # than skip-clean, and never crashes.
    repo = project.project
    ws = Workspace.default(project)
    flow = _make_flow(workspace=ws, policy=_policy(rollback_on_failure=False))
    task = _task(repo)

    def boom(*a, **k):
        raise GitError("git diff timed out")

    monkeypatch.setattr(verify, "attempt_dirty", boom)

    with pytest.raises(_Pause):
        flow.rollback_or_pause(task)

    assert "rollback-dirty-check-failed" in flow.journal.events()
    assert "rollback-skipped-clean" not in flow.journal.events()


# --------------------------------------------------------------- preserve refs


def test_preserve_attempt_commits_parks_committed_work(project):
    repo = project.project
    ws = Workspace.default(project)
    flow = _make_flow(workspace=ws)
    task = _task(repo)
    (repo / "src.txt").write_text("committed\n")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "attempt commit")

    flow.preserve_attempt_commits(task, allow_pause=True)

    assert "attempt-commits-preserved" in flow.journal.events()
    ref = flow.journal.fields("attempt-commits-preserved")["ref"]
    assert ref.startswith("attempt-preserve/")
    git(repo, "rev-parse", "--verify", ref)  # the recovery branch exists


def test_preserve_attempt_commits_noop_without_commits(project):
    ws = Workspace.default(project)
    flow = _make_flow(workspace=ws)
    task = _task(project.project)  # HEAD == baseline, nothing committed above it

    flow.preserve_attempt_commits(task, allow_pause=True)

    assert "attempt-commits-preserved" not in flow.journal.events()
    assert flow.calls.pauses == []


def test_preserve_attempt_commits_pauses_when_ref_fails_and_allowed(project, monkeypatch):
    repo = project.project
    ws = Workspace.default(project)
    flow = _make_flow(workspace=ws)
    task = _task(repo)
    (repo / "src.txt").write_text("committed\n")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "attempt commit")

    def no_ref(*a, **k):
        raise GitError("branch creation failed")

    monkeypatch.setattr(verify, "preserve_commits", no_ref)

    with pytest.raises(_Pause):
        flow.preserve_attempt_commits(task, allow_pause=True)

    assert "attempt-preserve-failed" in flow.journal.events()
    # preserve_failed=True routes through the committed-work notice
    assert "could not be auto-preserved" in flow.calls.pauses[-1][0]


def test_preserve_attempt_commits_no_pause_on_redrive(project, monkeypatch):
    repo = project.project
    ws = Workspace.default(project)
    flow = _make_flow(workspace=ws)
    task = _task(repo)
    (repo / "src.txt").write_text("committed\n")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "attempt commit")

    monkeypatch.setattr(verify, "preserve_commits", lambda *a, **k: None)  # ref did not take

    flow.preserve_attempt_commits(task, allow_pause=False)  # re-drive: never pauses

    assert "attempt-preserve-failed" in flow.journal.events()
    assert flow.calls.pauses == []


def test_preserve_attempt_worktree_snapshots_dirty_tree(project):
    repo = project.project
    ws = Workspace.default(project)
    flow = _make_flow(workspace=ws)
    task = _task(repo)
    task.attempt = 2
    (repo / "src.txt").write_text("uncommitted edit\n")  # tracked file dirtied

    flow.preserve_attempt_worktree(task)

    assert "attempt-worktree-preserved" in flow.journal.events()
    ref = flow.journal.fields("attempt-worktree-preserved")["ref"]
    assert ref.startswith("refs/attempt-preserve-dirty/")
    git(repo, "rev-parse", "--verify", ref)  # the snapshot ref exists


# --------------------------------------------------------------- safe_reset


def test_safe_reset_reverts_tracked_and_keeps_baseline(project):
    repo = project.project
    ws = Workspace.default(project)
    flow = _make_flow(workspace=ws)
    task = _task(repo)
    (repo / "src.txt").write_text("committed then to be reset\n")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "attempt")

    flow.safe_reset(task)

    assert rev_parse_head(repo) == task.baseline_commit


# --------------------------------------------------------------- prune


def test_prune_preserve_refs_disabled_when_keep_zero(project):
    ws = Workspace.default(project)
    flow = _make_flow(workspace=ws, policy=_policy(preserve_keep=0))
    flow.prune_preserve_refs()
    assert flow.journal.events() == []


def test_prune_preserve_refs_journals_deleted_per_family(project, monkeypatch):
    ws = Workspace.default(project)
    flow = _make_flow(workspace=ws, policy=_policy(preserve_keep=3))
    monkeypatch.setattr(verify, "prune_preserve_refs", lambda repo, keep: ["a", "b"])
    monkeypatch.setattr(verify, "prune_preserve_dirty_refs", lambda repo, keep: [])

    flow.prune_preserve_refs()

    assert flow.journal.fields("attempt-preserve-pruned")["count"] == 2
    # empty deletion for the other family journals nothing
    assert "attempt-preserve-dirty-pruned" not in flow.journal.events()


def test_prune_preserve_refs_error_journaled_and_other_family_still_runs(project, monkeypatch):
    ws = Workspace.default(project)
    flow = _make_flow(workspace=ws, policy=_policy(preserve_keep=3))

    def stuck(repo, keep):
        exc = RuntimeError("update-ref stuck")
        exc.deleted = ["r1"]  # a partial prune already deleted this before stalling
        exc.failed = ["r2"]
        raise exc

    monkeypatch.setattr(verify, "prune_preserve_refs", stuck)
    monkeypatch.setattr(verify, "prune_preserve_dirty_refs", lambda repo, keep: ["d1"])

    flow.prune_preserve_refs()  # a failure in one family must never crash or skip the other

    events = flow.journal.events()
    assert "attempt-preserve-pruned" in events  # partial deletions stay auditable
    assert flow.journal.fields("attempt-preserve-prune-failed")["failed"] == ["r2"]
    assert "attempt-preserve-dirty-pruned" in events  # the second family still ran


# --------------------------------------------------------------- manual recovery


def test_pause_stopped_wording_no_commits(project, tmp_path):
    ws = Workspace.default(project)
    flow = _make_flow(workspace=ws, run_dir=tmp_path)
    task = _task(project.project)

    with pytest.raises(_Pause) as excinfo:
        flow.pause_for_manual_recovery(task, task.baseline_commit)

    reason = excinfo.value.reason
    assert "failed" not in reason
    assert "manual rollback needed" in reason
    assert flow.calls.saves == 1
    assert "rollback-manual-required" in flow.journal.events()
    # notify wrote a line to the run dir's attention file (QUIET file=True)
    assert "manual rollback for 1-1-a" in (tmp_path / ATTENTION_FILE).read_text()


def test_pause_committed_wording_names_at_risk_commits(project, tmp_path):
    repo = project.project
    ws = Workspace.default(project)
    flow = _make_flow(workspace=ws, run_dir=tmp_path)
    task = _task(repo)
    (repo / "src.txt").write_text("committed work above baseline\n")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "finished but unfolded")

    with pytest.raises(_Pause) as excinfo:
        flow.pause_for_manual_recovery(task, task.baseline_commit)

    reason = excinfo.value.reason
    assert "committed work above its baseline" in reason
    assert "do NOT reset before checking" in reason


def test_pause_preserve_failed_wording(project, tmp_path):
    ws = Workspace.default(project)
    flow = _make_flow(workspace=ws, run_dir=tmp_path)
    task = _task(project.project)

    with pytest.raises(_Pause) as excinfo:
        flow.pause_for_manual_recovery(task, task.baseline_commit, preserve_failed=True)

    assert "could not be auto-preserved" in excinfo.value.reason


# --------------------------------------------------------------- restore_patch


def test_restore_patch_noop_without_latch(project):
    ws = Workspace.default(project)
    flow = _make_flow(workspace=ws)
    task = _task(project.project)  # restore_patch is None by default

    flow.restore_patch(task)

    assert flow.journal.events() == []
    assert flow.calls.escalates == []


def test_restore_patch_applies_and_journals(project, monkeypatch):
    ws = Workspace.default(project)
    flow = _make_flow(workspace=ws)
    task = _task(project.project)
    task.restore_patch = "patch.diff"
    applied = []
    monkeypatch.setattr(verify, "resolve_restore_path", lambda raw, root: Path(root) / raw)
    monkeypatch.setattr(verify, "apply_patch", lambda repo, patch: applied.append(patch))

    flow.restore_patch(task)

    assert applied  # the patch was applied
    assert "attempt-restored" in flow.journal.events()
    assert flow.calls.escalates == []


def test_restore_patch_escalates_on_apply_failure(project, monkeypatch):
    ws = Workspace.default(project)
    flow = _make_flow(workspace=ws)
    task = _task(project.project)
    task.restore_patch = "patch.diff"
    task.phase = Phase.DEV_RUNNING  # the call-site invariant restore_patch relies on
    monkeypatch.setattr(verify, "resolve_restore_path", lambda raw, root: Path(root) / raw)

    def boom(repo, patch):
        raise GitError("does not apply")

    monkeypatch.setattr(verify, "apply_patch", boom)

    with pytest.raises(_Pause):
        flow.restore_patch(task)

    assert task.phase == Phase.DEV_VERIFY  # stepped to the escalatable phase first
    assert flow.calls.escalates  # routed through the engine's escalation
    assert "attempt-restore-failed" in flow.journal.events()
    assert "attempt-restored" not in flow.journal.events()  # never reached on failure
