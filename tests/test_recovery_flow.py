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
from conftest import git, refuse_to_resolve

from bmad_loop import verify
from bmad_loop.bmadconfig import ProjectPaths
from bmad_loop.gates import ATTENTION_FILE
from bmad_loop.model import Phase, StoryTask
from bmad_loop.policy import GatesPolicy, LimitsPolicy, NotifyPolicy, Policy, ScmPolicy
from bmad_loop.recovery_flow import PRESERVE_REF_PROBE_LIMIT, RecoveryFlow
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
        calls.pauses.append((reason, story_key, cause))
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


def _tracked_spec(
    project: ProjectPaths,
    *,
    name: str = "spec-1-1-a.md",
    status: str = "ready-for-dev",
    body: str = "baseline intent\n",
) -> Path:
    """Create and commit one ordinary spec, returning the attempt baseline file."""
    spec = project.implementation_artifacts / name
    spec.parent.mkdir(parents=True, exist_ok=True)
    spec.write_text(f"---\nstatus: {status}\n---\n\n{body}")
    git(project.project, "add", "-A")
    git(project.project, "commit", "-q", "-m", "tracked spec baseline")
    return spec


def _status(spec: Path) -> str:
    return verify.status_of(verify.read_frontmatter(spec))


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
    # A folder == repo root would relativize to "." and, used as a preserve
    # prefix, keep the whole tree through a reset — it must be dropped.
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


def test_rollback_dirty_check_oserror_degrades_to_dirty(project, monkeypatch):
    # #343: spawn faults now arrive typed as GitSpawnError, but the guard keeps a
    # plain-OSError net for any untyped fault out of the probe. It must degrade
    # exactly like the GitError above — this is the first git call on the rollback
    # path, so an unguarded fault here crashes before any preserve step can run.
    repo = project.project
    ws = Workspace.default(project)
    flow = _make_flow(workspace=ws, policy=_policy(rollback_on_failure=False))
    task = _task(repo)

    def boom(*a, **k):
        raise OSError(24, "Too many open files")

    monkeypatch.setattr(verify, "attempt_dirty", boom)

    with pytest.raises(_Pause):
        flow.rollback_or_pause(task)

    assert "rollback-dirty-check-failed" in flow.journal.events()
    assert "rollback-skipped-clean" not in flow.journal.events()


def test_bound_lifecycle_only_spec_is_normalized_and_reads_git_clean(project):
    """T8: the one-file attempt binding recognizes only its own lifecycle delta.

    Ablation: replace `owned_exclude` with `()` in the first dirty probe and this
    test fails by taking the rollback-off manual-pause path.
    """
    repo = project.project
    spec = _tracked_spec(project)
    task = _task(repo)
    task.dispatched_spec_file = str(spec)
    verify.set_frontmatter_status(spec, "in-progress")
    flow = _make_flow(
        workspace=Workspace.default(project), policy=_policy(rollback_on_failure=False)
    )

    flow.rollback_or_pause(task)

    assert _status(spec) == "ready-for-dev"
    assert git(repo, "status", "--porcelain") == ""
    assert flow.journal.events() == ["rollback-skipped-clean"]
    assert flow.calls.pauses == []
    assert flow.calls.emits == []


def test_plain_owned_spec_with_substantive_residue_still_pauses(project):
    """T9: status normalization does not authorize a plain attempt's body edit.

    Ablation: remove the `redrive` term from the owned-dirty event guard and this
    test fails because the plain attempt returns instead of pausing.
    """
    repo = project.project
    spec = _tracked_spec(project)
    task = _task(repo)
    task.dispatched_spec_file = str(spec)
    spec.write_text("---\nstatus: in-progress\n---\n\nhuman substantive correction\n")
    flow = _make_flow(
        workspace=Workspace.default(project), policy=_policy(rollback_on_failure=False)
    )

    with pytest.raises(_Pause):
        flow.rollback_or_pause(task)

    assert _status(spec) == "ready-for-dev"
    assert "human substantive correction" in spec.read_text()
    assert "rollback-owned-spec-normalized" not in flow.journal.events()
    assert "rollback-skipped-clean" not in flow.journal.events()
    assert flow.calls.emits == []


def test_bound_spec_exclusion_does_not_hide_sibling_source_residue(project):
    """T10/source: source debris keeps the ordinary rollback policy reachable.

    INVERSE ablation: replace the exact-file exclusion with `.` and this test
    fails because the source is initially hidden and the owned status is rewritten.
    """
    repo = project.project
    spec = _tracked_spec(project)
    task = _task(repo)
    task.dispatched_spec_file = str(spec)
    verify.set_frontmatter_status(spec, "in-progress")
    (repo / "src.txt").write_text("attempt source residue\n")
    flow = _make_flow(
        workspace=Workspace.default(project), policy=_policy(rollback_on_failure=False)
    )

    with pytest.raises(_Pause):
        flow.rollback_or_pause(task)

    assert _status(spec) == "in-progress"  # no normalization while sibling debris exists
    assert "rollback-skipped-clean" not in flow.journal.events()


def test_bound_spec_exclusion_does_not_hide_sibling_artifact_residue(project):
    """T10/artifact: a whole-folder exclusion must not return through this path.

    INVERSE ablation: pass `protected` to the first dirty probe and this test
    fails because the sibling artifact is initially hidden and the owned status
    is rewritten despite that residue.
    """
    repo = project.project
    spec = _tracked_spec(project)
    task = _task(repo)
    task.dispatched_spec_file = str(spec)
    verify.set_frontmatter_status(spec, "in-progress")
    (project.implementation_artifacts / "sibling-result.md").write_text("attempt residue\n")
    flow = _make_flow(
        workspace=Workspace.default(project), policy=_policy(rollback_on_failure=False)
    )

    with pytest.raises(_Pause):
        flow.rollback_or_pause(task)

    assert _status(spec) == "in-progress"
    assert "rollback-skipped-clean" not in flow.journal.events()


def test_bound_spec_exclusion_does_not_hide_run_created_untracked_residue(project):
    """T10/untracked: an unrelated run-created path remains attempt dirtiness.

    INVERSE ablation: add `run-created.tmp` beside the owned-spec exclusion and
    this test fails because normalization runs while unrelated residue exists.
    """
    repo = project.project
    spec = _tracked_spec(project)
    task = _task(repo)
    task.dispatched_spec_file = str(spec)
    verify.set_frontmatter_status(spec, "in-progress")
    (repo / "run-created.tmp").write_text("attempt residue\n")
    flow = _make_flow(
        workspace=Workspace.default(project), policy=_policy(rollback_on_failure=False)
    )

    with pytest.raises(_Pause):
        flow.rollback_or_pause(task)

    assert _status(spec) == "in-progress"
    assert (repo / "run-created.tmp").is_file()
    assert "rollback-skipped-clean" not in flow.journal.events()


def test_unbound_spec_flip_retains_existing_dirty_policy(project):
    """T11: late `spec_file` cannot substitute for attempt-scoped ownership.

    INVERSE ablations: add the implementation-artifacts folder to the first dirty
    probe's exclusions, or substitute late `task.spec_file` ownership; either makes
    this test fail by emitting `rollback-skipped-clean`.
    """
    repo = project.project
    spec = _tracked_spec(project)
    task = _task(repo)
    task.spec_file = str(spec)  # deliberately late/accepted ownership only
    verify.set_frontmatter_status(spec, "in-progress")
    flow = _make_flow(
        workspace=Workspace.default(project), policy=_policy(rollback_on_failure=False)
    )

    with pytest.raises(_Pause):
        flow.rollback_or_pause(task)

    assert _status(spec) == "in-progress"
    assert "rollback-skipped-clean" not in flow.journal.events()
    assert "rollback-owned-spec-normalized" not in flow.journal.events()


def test_latched_redrive_reports_owned_corrected_spec_as_still_dirty(project):
    """T12: a human-corrected spec is authorized, preserved, and never called clean.

    Ablation: delete the owned-dirty event arm and this test fails because the
    latched redrive falls through to the ordinary reset/pause policy.
    """
    repo = project.project
    spec = _tracked_spec(project)
    task = _task(repo)
    task.dispatched_spec_file = str(spec)
    task.resolved_redrive = True
    spec.write_text("---\nstatus: ready-for-dev\n---\n\nhuman corrected intent\n")
    flow = _make_flow(
        workspace=Workspace.default(project), policy=_policy(rollback_on_failure=False)
    )

    flow.rollback_or_pause(task)

    assert _status(spec) == "ready-for-dev"
    assert "human corrected intent" in spec.read_text()
    assert verify.attempt_dirty(repo, task.baseline_commit, task.baseline_untracked)
    assert flow.journal.fields("rollback-owned-spec-normalized") == {
        "story_key": task.story_key,
        "spec": str(spec.resolve()),
        "status": "ready-for-dev",
        "checkout_dirty": True,
    }
    assert "rollback-skipped-clean" not in flow.journal.events()
    assert flow.calls.pauses == []
    assert flow.calls.emits == []


def test_post_normalization_probe_fault_cannot_authorize_owned_dirty(project, monkeypatch):
    """A failed real-checkout re-probe stays fail-safe dirty on a resolved reset.

    Ablation: remove `dirty_probe_succeeded` from the owned-dirty event guard and
    this test fails because an unproven checkout bypasses the actual rollback.
    """
    repo = project.project
    spec = _tracked_spec(project)
    task = _task(repo)
    task.dispatched_spec_file = str(spec)
    spec.write_text("---\nstatus: ready-for-dev\n---\n\nhuman corrected intent\n")
    real_attempt_dirty = verify.attempt_dirty
    probes = 0

    def fault_second_probe(*args, **kwargs):
        nonlocal probes
        probes += 1
        if probes == 2:
            raise verify.GitError("post-normalization probe failed")
        return real_attempt_dirty(*args, **kwargs)

    monkeypatch.setattr(verify, "attempt_dirty", fault_second_probe)
    flow = _make_flow(
        workspace=Workspace.default(project), policy=_policy(rollback_on_failure=False)
    )

    flow.rollback_or_pause(task, cause="resolved")

    assert probes == 2
    assert "rollback-dirty-check-failed" in flow.journal.events()
    assert "rollback-auto" in flow.journal.events()
    assert "rollback-skipped-clean" not in flow.journal.events()
    assert "rollback-owned-spec-normalized" not in flow.journal.events()
    assert flow.calls.emits == ["pre_rollback", "post_rollback"]
    assert verify.attempt_dirty(repo, task.baseline_commit, task.baseline_untracked)


def test_patch_restore_redrive_normalizes_owned_spec_to_in_review(project):
    """T13: the restore latch selects `in-review`, never from-scratch readiness.

    Ablation: replace the restore-latched target selection with unconditional
    `ready-for-dev` and this test fails on both the spec and journal status.
    """
    repo = project.project
    spec = _tracked_spec(project, status="in-review")
    task = _task(repo)
    task.dispatched_spec_file = str(spec)
    task.restore_patch = "intent-gap.patch"
    task.resolved_redrive = True
    spec.write_text("---\nstatus: in-progress\n---\n\nrestored human correction\n")
    flow = _make_flow(
        workspace=Workspace.default(project), policy=_policy(rollback_on_failure=False)
    )

    flow.rollback_or_pause(task)

    assert _status(spec) == "in-review"
    assert "restored human correction" in spec.read_text()
    event = flow.journal.fields("rollback-owned-spec-normalized")
    assert event["status"] == "in-review"
    assert event["checkout_dirty"] is True
    assert "rollback-skipped-clean" not in flow.journal.events()
    assert flow.calls.emits == []


def test_owned_spec_without_visible_status_fails_the_post_write_oracle(project):
    """T14/False: a writer no-op is not repair success without the target oracle.

    Ablation: delete the post-write `status_of(read_frontmatter(...))` comparison
    and this test fails by reaching ordinary rollback policy instead of the typed
    repair error.
    """
    repo = project.project
    spec = _tracked_spec(project)
    task = _task(repo)
    task.dispatched_spec_file = str(spec)
    spec.write_text("---\ntitle: no status here\n---\n\nbaseline intent\n")
    flow = _make_flow(
        workspace=Workspace.default(project), policy=_policy(rollback_on_failure=False)
    )

    with pytest.raises(verify.FrontmatterWriteError, match="could not normalize"):
        flow.rollback_or_pause(task)

    assert "rollback-skipped-clean" not in flow.journal.events()
    assert "rollback-owned-spec-normalized" not in flow.journal.events()
    assert flow.calls.pauses == []


def test_owned_spec_with_unsafe_status_shape_propagates_writer_error(project):
    """T14/write: repair-write refusal is never caught as failed observation.

    Ablation: catch `FrontmatterWriteError` around the normalization write and
    return from recovery; this test fails because the unsafe-shape error vanishes.
    """
    repo = project.project
    spec = _tracked_spec(project)
    task = _task(repo)
    task.dispatched_spec_file = str(spec)
    spec.write_text("---\n{status: in-progress, keep: 1}\n---\nbaseline intent\n")
    flow = _make_flow(
        workspace=Workspace.default(project), policy=_policy(rollback_on_failure=False)
    )

    with pytest.raises(verify.FrontmatterWriteError, match="no in-place line edit"):
        flow.rollback_or_pause(task)

    assert _status(spec) == "in-progress"
    assert "rollback-skipped-clean" not in flow.journal.events()
    assert "rollback-owned-spec-normalized" not in flow.journal.events()


def test_owned_spec_outside_trusted_roots_is_not_excluded_or_mutated(project):
    """T15/outside: an in-repo path is not trusted merely because Git can name it.

    Ablation: delete the `spec_within_roots` refusal in `_attempt_owned_spec` and
    this test fails because the out-of-project file is normalized and called clean.
    """
    repo = project.project
    trusted = repo / "trusted-project"
    trusted_impl = trusted / "_bmad-output" / "implementation-artifacts"
    trusted_plan = trusted / "_bmad-output" / "planning-artifacts"
    trusted_impl.mkdir(parents=True)
    trusted_plan.mkdir(parents=True)
    paths = ProjectPaths(
        project=trusted,
        implementation_artifacts=trusted_impl,
        planning_artifacts=trusted_plan,
        repo_root=repo,
    )
    outside = repo / "outside-trusted-project.md"
    outside.write_text("---\nstatus: ready-for-dev\n---\nbody\n")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "outside binding baseline")
    task = _task(repo)
    task.dispatched_spec_file = str(outside)
    verify.set_frontmatter_status(outside, "in-progress")
    flow = _make_flow(
        workspace=Workspace(root=repo, paths=paths),
        paths=paths,
        policy=_policy(rollback_on_failure=False),
    )

    with pytest.raises(_Pause):
        flow.rollback_or_pause(task)

    assert _status(outside) == "in-progress"
    assert "rollback-skipped-clean" not in flow.journal.events()
    assert "rollback-owned-spec-normalized" not in flow.journal.events()


def test_missing_attempt_binding_does_not_fall_back_to_late_spec(project):
    """T15/missing: a stale attempt binding cannot borrow accepted ownership.

    INVERSE ablation: fall back to `task.spec_file` when the dispatched path is
    missing and this test fails because the late spec is normalized and called clean.
    """
    repo = project.project
    spec = _tracked_spec(project)
    task = _task(repo)
    task.spec_file = str(spec)
    task.dispatched_spec_file = str(project.implementation_artifacts / "missing.md")
    verify.set_frontmatter_status(spec, "in-progress")
    flow = _make_flow(
        workspace=Workspace.default(project), policy=_policy(rollback_on_failure=False)
    )

    with pytest.raises(_Pause):
        flow.rollback_or_pause(task)

    assert _status(spec) == "in-progress"
    assert "rollback-skipped-clean" not in flow.journal.events()
    assert "rollback-owned-spec-normalized" not in flow.journal.events()


def test_non_file_attempt_binding_is_refused(project):
    """T15/non-file: a directory can never become an exact owned-spec exclusion.

    INVERSE ablation: accept the resolved candidate without its `is_file` guard
    and this test fails on the attempted directory frontmatter repair.
    """
    repo = project.project
    spec = _tracked_spec(project)
    task = _task(repo)
    task.dispatched_spec_file = str(project.implementation_artifacts)
    verify.set_frontmatter_status(spec, "in-progress")
    flow = _make_flow(
        workspace=Workspace.default(project), policy=_policy(rollback_on_failure=False)
    )

    with pytest.raises(_Pause):
        flow.rollback_or_pause(task)

    assert _status(spec) == "in-progress"
    assert "rollback-skipped-clean" not in flow.journal.events()
    assert "rollback-owned-spec-normalized" not in flow.journal.events()


def test_ambiguous_relative_attempt_binding_is_refused(project):
    """T15/ambiguous: two live interpretations cannot confer attempt ownership.

    Ablation: weaken the unique-candidate guard to accept the first live file and
    this test fails because the project-relative candidate is normalized as owned.
    """
    repo = project.project
    project_candidate = repo / "spec.md"
    artifact_candidate = project.implementation_artifacts / "spec.md"
    for candidate in (project_candidate, artifact_candidate):
        candidate.write_text("---\nstatus: ready-for-dev\n---\nbody\n")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "ambiguous spec baseline")
    task = _task(repo)
    task.dispatched_spec_file = "spec.md"
    verify.set_frontmatter_status(project_candidate, "in-progress")
    flow = _make_flow(
        workspace=Workspace.default(project), policy=_policy(rollback_on_failure=False)
    )

    with pytest.raises(_Pause):
        flow.rollback_or_pause(task)

    assert _status(project_candidate) == "in-progress"
    assert _status(artifact_candidate) == "ready-for-dev"
    assert "rollback-skipped-clean" not in flow.journal.events()
    assert "rollback-owned-spec-normalized" not in flow.journal.events()


def test_binding_resolution_fault_is_fail_safe_dirty(project, monkeypatch):
    """T15/unsafe: uncertain ownership cannot become a mutation/exclusion grant.

    Ablation: let `_attempt_owned_spec` continue with the unresolved candidate
    after `Path.resolve` raises and this test fails before the manual-pause policy.
    """
    repo = project.project
    spec = _tracked_spec(project)
    task = _task(repo)
    task.dispatched_spec_file = str(spec)
    verify.set_frontmatter_status(spec, "in-progress")
    refuse_to_resolve(monkeypatch, spec)
    flow = _make_flow(
        workspace=Workspace.default(project), policy=_policy(rollback_on_failure=False)
    )

    with pytest.raises(_Pause):
        flow.rollback_or_pause(task)

    assert _status(spec) == "in-progress"
    assert "rollback-skipped-clean" not in flow.journal.events()
    assert "rollback-owned-spec-normalized" not in flow.journal.events()


def test_external_configured_artifact_spec_is_trusted_without_git_exclusion(
    project, tmp_path, monkeypatch
):
    """A configured external artifact root is writable but never a Git pathspec.

    Ablations: restrict trusted roots to the project tree, or return the external
    absolute path as a Git exclusion; either makes this test fail before both the
    trusted normalization and empty-exclusion assertions can hold.
    """
    repo = project.project
    external_impl = tmp_path / "external-artifacts"
    external_impl.mkdir()
    paths = ProjectPaths(
        project=repo,
        implementation_artifacts=external_impl,
        planning_artifacts=project.planning_artifacts,
        output_folder=project.output_folder,
        repo_root=repo,
    )
    spec = external_impl / "spec-1-1-a.md"
    spec.write_text("---\nstatus: in-progress\n---\nexternal intent\n")
    task = _task(repo)
    task.dispatched_spec_file = str(spec)
    seen_excludes: list[tuple[str, ...]] = []
    real_attempt_dirty = verify.attempt_dirty

    def recording_attempt_dirty(*args, **kwargs):
        seen_excludes.append(kwargs.get("exclude", ()))
        return real_attempt_dirty(*args, **kwargs)

    monkeypatch.setattr(verify, "attempt_dirty", recording_attempt_dirty)
    flow = _make_flow(
        workspace=Workspace(root=repo, paths=paths),
        paths=paths,
        policy=_policy(rollback_on_failure=False),
    )

    flow.rollback_or_pause(task)

    assert _status(spec) == "ready-for-dev"
    assert seen_excludes == [(), ()]
    assert flow.journal.events() == ["rollback-skipped-clean"]
    assert flow.calls.pauses == []


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
    assert task.preserve_ref == ref  # #333: the ref reaches run state, not just the journal


def test_preserve_attempt_commits_noop_without_commits(project):
    ws = Workspace.default(project)
    flow = _make_flow(workspace=ws)
    task = _task(project.project)  # HEAD == baseline, nothing committed above it

    flow.preserve_attempt_commits(task, allow_pause=True)

    assert "attempt-commits-preserved" not in flow.journal.events()
    assert flow.calls.pauses == []
    assert task.preserve_ref is None  # nothing parked → nothing to point the operator at


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
    # the ref never took — run state must not name one (#333)
    assert task.preserve_ref is None


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


def _commit_something(repo: Path) -> None:
    (repo / "src.txt").write_text("committed\n")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "attempt commit")


@pytest.mark.parametrize(
    "make_exc",
    # Factories, not instances: a parametrized exception *instance* is built once at
    # collection and shared by every case, and re-raising it mutates its
    # __traceback__ — which under pytest-randomly's shuffling made these cases
    # couple to each other and fail for the wrong reason.
    [lambda: GitError("git log timed out"), lambda: OSError(24, "Too many open files")],
    ids=["giterror", "oserror"],
)
def test_preserve_attempt_commits_pauses_when_range_unenumerable(project, monkeypatch, make_exc):
    # #343: `commits_above` carried no guard at all, so even a plain GitError (a
    # translated git timeout) crashed the rollback here. An un-determinable range
    # must refuse the reset, never fall through the `not commits` early return.
    repo = project.project
    ws = Workspace.default(project)
    flow = _make_flow(workspace=ws)
    task = _task(repo)
    _commit_something(repo)

    def boom(*a, **k):
        raise make_exc()

    monkeypatch.setattr(verify, "commits_above", boom)

    with pytest.raises(_Pause):
        flow.preserve_attempt_commits(task, allow_pause=True)

    assert "attempt-preserve-enumerate-failed" in flow.journal.events()
    assert "could not be auto-preserved" in flow.calls.pauses[-1][0]
    assert task.preserve_ref is None  # nothing parked → nothing to point the operator at


def test_preserve_attempt_commits_pauses_when_head_read_fails(project, monkeypatch):
    # The second unguarded call (#343): the range enumerated fine, but the tip the
    # ref would park at could not be read — still work we cannot park.
    repo = project.project
    ws = Workspace.default(project)
    flow = _make_flow(workspace=ws)
    task = _task(repo)
    _commit_something(repo)

    def boom(*a, **k):
        raise GitError("git rev-parse timed out")

    monkeypatch.setattr(verify, "rev_parse_head", boom)

    with pytest.raises(_Pause):
        flow.preserve_attempt_commits(task, allow_pause=True)

    assert "attempt-preserve-enumerate-failed" in flow.journal.events()
    assert task.preserve_ref is None


def test_preserve_attempt_commits_unenumerable_no_pause_on_redrive(project, monkeypatch):
    # The re-drive contract forbids pausing even here: a human directed the discard.
    repo = project.project
    ws = Workspace.default(project)
    flow = _make_flow(workspace=ws)
    task = _task(repo)
    _commit_something(repo)

    def boom(*a, **k):
        raise OSError(24, "Too many open files")

    monkeypatch.setattr(verify, "commits_above", boom)

    flow.preserve_attempt_commits(task, allow_pause=False)  # must not raise

    assert "attempt-preserve-enumerate-failed" in flow.journal.events()
    assert flow.calls.pauses == []


def test_preserve_attempt_commits_ref_oserror_treated_as_failure(project, monkeypatch):
    # Sibling of the GitError case above: the ref write can also fail untyped (#343).
    repo = project.project
    ws = Workspace.default(project)
    flow = _make_flow(workspace=ws)
    task = _task(repo)
    _commit_something(repo)

    def boom(*a, **k):
        raise OSError(12, "Cannot allocate memory")

    monkeypatch.setattr(verify, "preserve_commits", boom)

    with pytest.raises(_Pause):
        flow.preserve_attempt_commits(task, allow_pause=True)

    assert "attempt-preserve-failed" in flow.journal.events()
    assert task.preserve_ref is None


def test_preserve_attempt_worktree_snapshots_dirty_tree(project):
    repo = project.project
    ws = Workspace.default(project)
    flow = _make_flow(workspace=ws)
    task = _task(repo)
    task.attempt = 2
    (repo / "src.txt").write_text("uncommitted edit\n")  # tracked file dirtied

    flow.preserve_attempt_worktree(task, allow_pause=False)

    assert "attempt-worktree-preserved" in flow.journal.events()
    ref = flow.journal.fields("attempt-worktree-preserved")["ref"]
    assert ref.startswith("refs/attempt-preserve-dirty/")
    git(repo, "rev-parse", "--verify", ref)  # the snapshot ref exists
    assert task.preserve_ref == ref  # #333


def test_dirty_preserve_ref_wins_and_subsumes_the_commits_branch(project):
    """Both families fire on one rollback: commits above baseline are parked on an
    `attempt-preserve/*` branch and the still-dirty tree on a dirty snapshot. The
    dirty ref is the one `preserve_ref` keeps, because it is committed parented at
    the attempt's HEAD and therefore already contains the branch — so the single
    `git merge --ff-only <preserve_ref>` the defer notice prints recovers the
    whole attempt, not half of it."""
    repo = project.project
    ws = Workspace.default(project)
    flow = _make_flow(workspace=ws, policy=_policy(rollback_on_failure=True))
    task = _task(repo)
    (repo / "src.txt").write_text("committed\n")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "attempt commit")
    (repo / "src.txt").write_text("committed then edited\n")  # uncommitted on top

    flow.rollback_or_pause(task)

    branch = flow.journal.fields("attempt-commits-preserved")["ref"]
    dirty = flow.journal.fields("attempt-worktree-preserved")["ref"]
    assert task.preserve_ref == dirty != branch
    # subsumption: the branch tip is an ancestor of the dirty snapshot
    git(repo, "merge-base", "--is-ancestor", branch, dirty)
    # Ablation target for the partial marker: hoist `preserve_partial = True` out of
    # preserve_attempt_worktree's `except` and this fails. A park that captured
    # everything must never be libelled as commits-only.
    assert task.preserve_partial is False


def _fail_snapshot(monkeypatch, exc=None):
    """Make verify.snapshot_worktree raise, the way a full disk or a git timeout
    inside `add -u`/`write-tree`/`update-ref` would."""

    def _fail(*_a, **_k):
        raise exc if exc is not None else verify.GitError("simulated commit-tree failure")

    monkeypatch.setattr(verify, "snapshot_worktree", _fail)


def test_snapshot_failure_leaves_a_commits_only_ref_flagged_partial(project, monkeypatch):
    """The snapshot raises *after* the commits branch was already parked, and on a
    re-drive the reset runs anyway. `preserve_ref` then names the commits branch
    alone, so the whole-attempt promise no longer holds — `preserve_partial` records
    that, and the defer notice downgrades its claim instead of telling the operator
    a `merge --ff-only` restores work the reset just destroyed.

    Since #340 a *plain* rollback refuses this reset, so the re-drive
    (`cause="resolved"`, contractually pause-free) is the surviving path where the
    partial park is reachable — which is exactly why #338's downgrade is narrowed
    rather than superseded. `rollback_on_failure` is left OFF so the re-drive is
    what carries the auto-recover arm here, not the policy flag.

    Ablation target: delete `task.preserve_partial = True` from
    preserve_attempt_worktree's `except verify.GitError` block and this fails."""
    repo = project.project
    ws = Workspace.default(project)
    flow = _make_flow(workspace=ws)
    task = _task(repo)
    (repo / "src.txt").write_text("committed\n")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "attempt commit")
    (repo / "src.txt").write_text("committed then edited\n")  # the half that will be lost

    _fail_snapshot(monkeypatch)

    flow.rollback_or_pause(task, cause="resolved")

    assert flow.calls.pauses == []  # a re-drive never pauses, even on a preserve failure
    assert "attempt-worktree-preserve-failed" in flow.journal.events()
    assert task.preserve_ref == flow.journal.fields("attempt-commits-preserved")["ref"]
    assert task.preserve_ref.startswith("attempt-preserve/")
    assert task.preserve_partial is True
    # why it matters: the reset ran regardless, so the tree is back at baseline and
    # the ref the notice names carries the committed half ONLY — the uncommitted
    # edit survives nowhere, which is exactly what the un-downgraded notice's
    # `merge --ff-only` would have implied it could restore
    assert "committed" not in (repo / "src.txt").read_text()
    assert git(repo, "show", f"{task.preserve_ref}:src.txt") == "committed"


def test_snapshot_failure_pauses_when_work_would_be_lost(project, monkeypatch):
    """#340: a failed dirty snapshot refuses the reset instead of destroying the
    uncommitted work it existed to capture. Unlike the commits path's orphaned
    objects, a tracked edit a `reset --hard` discards is unrecoverable, so the
    safety net becomes a gate.

    Ablation target: delete the `pause_for_manual_recovery(..., snapshot_failed=True)`
    call from preserve_attempt_worktree's `except` and this fails."""
    repo = project.project
    ws = Workspace.default(project)
    flow = _make_flow(workspace=ws, policy=_policy(rollback_on_failure=True))
    task = _task(repo)
    (repo / "src.txt").write_text("uncommitted work\n")
    (repo / "new.txt").write_text("run-created untracked\n")
    _fail_snapshot(monkeypatch)

    with pytest.raises(_Pause):
        flow.rollback_or_pause(task)

    assert "attempt-worktree-preserve-failed" in flow.journal.events()
    # the whole point: the tree is untouched, so the work is still there to rescue
    assert (repo / "src.txt").read_text() == "uncommitted work\n"
    assert (repo / "new.txt").is_file()
    notice = flow.calls.pauses[0][0]
    assert "uncommitted work could not be auto-preserved" in notice
    assert "attempt-worktree-preserve-failed" in notice  # names the diagnosis breadcrumb
    # rescue first, discard second — and step 3 is what lets the pause terminate
    # (reset, resume, rollback-skipped-clean). Anchor on the command, not the bare
    # phrase: the prose above it also says `reset --hard`, so index() would match there.
    assert notice.index("Save what you want to keep") < notice.index(
        f'git -C "{repo}" reset --hard'
    )


def test_snapshot_failure_proceeds_when_nothing_would_be_lost(project, monkeypatch):
    """The refusal is gated on there being something left to lose. An attempt that
    committed everything has a tree clean vs HEAD, so the reset destroys nothing and
    a snapshot fault must not halt an unattended run over it (cf. #123's false-pause
    complaint).

    INVERSE ablation — `pauses == []` would also pass if the pause were simply
    unreachable. Make the pause unconditional (drop the `_reset_would_destroy`
    guard) and this must fail."""
    repo = project.project
    ws = Workspace.default(project)
    flow = _make_flow(workspace=ws, policy=_policy(rollback_on_failure=True))
    task = _task(repo)
    (repo / "src.txt").write_text("committed\n")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "attempt commit")  # nothing left uncommitted
    _fail_snapshot(monkeypatch)

    flow.rollback_or_pause(task)

    assert flow.calls.pauses == []
    assert "attempt-worktree-preserve-failed" in flow.journal.events()
    assert rev_parse_head(repo) == task.baseline_commit  # the harmless reset ran
    # and the committed half is still recoverable by name
    assert git(repo, "show", f"{task.preserve_ref}:src.txt") == "committed"


def test_snapshot_failure_probe_fault_pauses(project, monkeypatch):
    """A git fault in the at-risk probe itself must read as work-at-risk, never as
    permission to reset — the same fail-safe direction as the dirty check (#156).

    Ablation target: flip `_reset_would_destroy`'s `except verify.GitError` to
    `return False` and this fails."""
    repo = project.project
    ws = Workspace.default(project)
    flow = _make_flow(workspace=ws, policy=_policy(rollback_on_failure=True))
    task = _task(repo)
    (repo / "src.txt").write_text("uncommitted work\n")
    _fail_snapshot(monkeypatch)

    def _fail_probe(*_a, **_k):
        raise verify.GitError("simulated rev-parse failure")

    monkeypatch.setattr(verify, "rev_parse_head", _fail_probe)

    with pytest.raises(_Pause):
        flow.rollback_or_pause(task)

    assert (repo / "src.txt").read_text() == "uncommitted work\n"  # not reset


def test_probe_oserror_also_fails_safe(project, monkeypatch):
    """The probe runs immediately after a snapshot fault, against the same git
    binary, so the EMFILE/ENOMEM that broke the capture is likely to break the probe
    too. Catching only GitError here would undo the broadening one frame up and
    crash the rollback anyway — the asymmetry, not either half, is the defect.

    Ablation target: narrow `_reset_would_destroy`'s `except` back to
    `verify.GitError` and this fails with the raw OSError."""
    repo = project.project
    ws = Workspace.default(project)
    flow = _make_flow(workspace=ws, policy=_policy(rollback_on_failure=True))
    task = _task(repo)
    (repo / "src.txt").write_text("uncommitted work\n")
    _fail_snapshot(monkeypatch, OSError(24, "Too many open files"))

    def _fail_probe(*_a, **_k):
        raise OSError(24, "Too many open files")

    monkeypatch.setattr(verify, "rev_parse_head", _fail_probe)

    with pytest.raises(_Pause):  # the typed pause, not the OSError
        flow.rollback_or_pause(task)

    assert (repo / "src.txt").read_text() == "uncommitted work\n"  # not reset


def test_snapshot_failure_never_pauses_on_redrive(project, monkeypatch):
    """The re-drive's pause-free contract outranks the #340 gate: the operator
    already directed this discard through the resolve workflow, so a failed capture
    journals and lets the reset run.

    Ablation target: delete `if not allow_pause: return` from the `except` and this
    fails."""
    repo = project.project
    ws = Workspace.default(project)
    flow = _make_flow(workspace=ws)
    task = _task(repo)
    (repo / "src.txt").write_text("uncommitted work\n")
    _fail_snapshot(monkeypatch)

    flow.rollback_or_pause(task, cause="resolved")

    assert flow.calls.pauses == []
    assert task.preserve_partial is True  # still latched — the notice must downgrade
    assert (repo / "src.txt").read_text() == "original\n"  # reset ran


def test_snapshot_oserror_degrades_into_the_typed_path(project, monkeypatch):
    """`snapshot_worktree` can raise a plain OSError outright — ENOSPC/EMFILE from
    its TemporaryDirectory — a filesystem fault the #343 spawn translation cannot
    cover. Preservation is observation, not a repair write, so it degrades into
    the same journal-and-decide path a GitError takes rather than crashing the
    run mid-rollback.

    Ablation target: narrow the `except` back to `verify.GitError` and this fails
    with the raw OSError."""
    repo = project.project
    ws = Workspace.default(project)
    flow = _make_flow(workspace=ws, policy=_policy(rollback_on_failure=True))
    task = _task(repo)
    (repo / "src.txt").write_text("uncommitted work\n")
    _fail_snapshot(monkeypatch, OSError(24, "Too many open files"))

    with pytest.raises(_Pause):  # the typed pause, not the OSError
        flow.rollback_or_pause(task)

    entry = flow.journal.fields("attempt-worktree-preserve-failed")
    assert "Too many open files" in entry["error"]  # errno detail kept as a breadcrumb
    assert (repo / "src.txt").read_text() == "uncommitted work\n"


def test_ref_probe_git_fault_degrades_like_a_failed_snapshot(project, monkeypatch):
    """The free-refname probe spawns git before the snapshot does, so it is the
    first place a spawn/timeout fault can surface. `ref_exists` deliberately does
    not swallow those (mistaking "git could not run" for "the name is free" would
    overwrite the very snapshot the probe exists to protect), so the probe has to
    sit inside the handler that turns a preservation fault into a pause.

    Ablation target: move the probe loop back above the `try` and this fails with
    the raw GitSpawnError instead of the typed pause."""
    repo = project.project
    ws = Workspace.default(project)
    flow = _make_flow(workspace=ws, policy=_policy(rollback_on_failure=True))
    task = _task(repo)
    (repo / "src.txt").write_text("uncommitted work\n")

    def _fail(*_a, **_k):
        raise verify.GitSpawnError("git: command not found")

    monkeypatch.setattr(verify, "ref_exists", _fail)

    with pytest.raises(_Pause):  # the typed pause, not the GitSpawnError
        flow.rollback_or_pause(task)

    entry = flow.journal.fields("attempt-worktree-preserve-failed")
    assert "command not found" in entry["error"]
    assert (repo / "src.txt").read_text() == "uncommitted work\n"  # work not reset away


def test_ref_probe_is_bounded_and_refuses_to_reuse_an_occupied_name(project, monkeypatch):
    """The probe terminates on its own — the ref set is finite and the serial only
    climbs — but terminating is not the same as being bounded: without a cap the
    iteration count is whatever the namespace happens to hold, one git spawn each,
    inside a crash-recovery path. `PRESERVE_REF_PROBE_LIMIT` bounds the scan, and
    exhausting it must RAISE rather than fall through to the last candidate;
    reusing an occupied name is the exact data loss the probe exists to prevent
    (#349). Exhaustion is a preservation fault like any other, so it degrades into
    the typed pause instead of crashing the rollback.

    `ref_exists` is answered True for a few calls PAST the limit, so removing the
    cap makes this fail on the missing pause rather than hanging the suite.

    Ablation target: delete the `serial > PRESERVE_REF_PROBE_LIMIT` raise and the
    probe walks past the bound to a free name, the snapshot succeeds, and the
    `pytest.raises(_Pause)` fails."""
    repo = project.project
    ws = Workspace.default(project)
    flow = _make_flow(workspace=ws, policy=_policy(rollback_on_failure=True))
    task = _task(repo)
    (repo / "src.txt").write_text("uncommitted work\n")

    probed: list[str] = []

    def _occupied(_repo, refname: str) -> bool:
        probed.append(refname)
        return len(probed) <= PRESERVE_REF_PROBE_LIMIT + 5

    snapshots: list[str] = []

    def _snapshot(_repo, refname, **_k):
        snapshots.append(refname)
        return refname

    monkeypatch.setattr(verify, "ref_exists", _occupied)
    monkeypatch.setattr(verify, "snapshot_worktree", _snapshot)

    with pytest.raises(_Pause):  # the typed pause, not a raw error and not a hang
        flow.rollback_or_pause(task)

    # The bound is the assertion: the base name plus -r2..-r<limit>, then stop.
    assert len(probed) == PRESERVE_REF_PROBE_LIMIT
    assert probed[-1].endswith(f"-r{PRESERVE_REF_PROBE_LIMIT}")
    assert not snapshots  # never wrote over any of the names it found occupied

    entry = flow.journal.fields("attempt-worktree-preserve-failed")
    assert "no free snapshot refname" in entry["error"]  # names the exhaustion
    assert "scm.preserve_keep" in entry["error"]  # and the operator's remedy
    assert (repo / "src.txt").read_text() == "uncommitted work\n"  # work not reset away


def test_notice_probe_oserror_does_not_swallow_the_pause(project, monkeypatch):
    """`pause_for_manual_recovery`'s advisory `commits_above` probe runs while the
    fault that broke the snapshot is still in force, so it is the likeliest place
    for a second EMFILE. Its own comment says a git fault there must not block the
    pause — an uncaught OSError did exactly that, losing a pause the caller had
    already decided to take.

    Ablation target: narrow that probe's `except` back to `verify.GitError` and this
    fails with the raw OSError instead of pausing."""
    repo = project.project
    ws = Workspace.default(project)
    flow = _make_flow(workspace=ws, policy=_policy(rollback_on_failure=True))
    task = _task(repo)
    (repo / "src.txt").write_text("uncommitted work\n")
    _fail_snapshot(monkeypatch, OSError(24, "Too many open files"))

    # Only the *notice's* probe may fail: preserve_attempt_commits calls
    # commits_above first, and breaking that would abort the rollback earlier and
    # never reach the code under test.
    real_commits_above = verify.commits_above
    seen = {"n": 0}

    def _flaky(*a, **k):
        seen["n"] += 1
        if seen["n"] == 1:
            return real_commits_above(*a, **k)
        raise OSError(24, "Too many open files")

    monkeypatch.setattr(verify, "commits_above", _flaky)

    with pytest.raises(_Pause):
        flow.rollback_or_pause(task)

    assert seen["n"] >= 2  # the advisory probe really was reached and really failed
    notice = flow.calls.pauses[0][0]
    assert "uncommitted work could not be auto-preserved" in notice  # shape (d) intact
    assert (repo / "src.txt").read_text() == "uncommitted work\n"  # not reset


def test_snapshot_failure_pause_names_the_unit_worktree(project, tmp_path, monkeypatch):
    """#161 compatibility: a preserve-failure pause can fire while a unit worktree is
    mounted, and every instruction must target that tree. Naming the main checkout
    there quotes a HEAD the attempt never moved and invites a destructive reset of a
    tree the operator never worked in.

    Ablation target: change `pause_for_manual_recovery`'s `root` back to
    `self.paths.repo_root` and this fails."""
    repo = project.project
    ws = Workspace.default(project)
    main_checkout = tmp_path / "some-other-main-checkout"
    flow = _make_flow(
        workspace=ws,
        paths=SimpleNamespace(repo_root=main_checkout),
        policy=_policy(rollback_on_failure=False),  # in-worktree recovery ignores it
    )
    task = _task(repo)
    (repo / "src.txt").write_text("worktree attempt\n")
    _fail_snapshot(monkeypatch)

    with pytest.raises(_Pause):
        flow.rollback_or_pause(task)

    notice = flow.calls.pauses[0][0]
    assert str(repo) in notice
    assert str(main_checkout) not in notice


def test_rollback_clears_a_previous_attempts_preserve_ref(project, monkeypatch):
    """Ablation target: delete the `task.preserve_ref = None` at the top of
    RecoveryFlow's auto-recover arm and this fails. A later rollback that parks
    nothing — here the ref simply fails to take on a pause-free re-drive — must not
    leave the *earlier* attempt's ref standing, or the defer notice sends the
    operator to work that is not the deferred attempt's."""
    repo = project.project
    ws = Workspace.default(project)
    flow = _make_flow(workspace=ws, policy=_policy(rollback_on_failure=True))
    task = _task(repo)
    (repo / "src.txt").write_text("attempt 1\n")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "attempt 1")

    flow.rollback_or_pause(task)
    stale = task.preserve_ref
    assert stale and stale.startswith("attempt-preserve/")

    task.attempt = 1
    (repo / "src.txt").write_text("attempt 2\n")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "attempt 2")
    monkeypatch.setattr(verify, "preserve_commits", lambda *a, **k: None)  # ref did not take

    flow.rollback_or_pause(task, cause="resolved")  # re-drive: never pauses

    assert "attempt-preserve-failed" in flow.journal.events()
    assert task.preserve_ref is None


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


def test_safe_reset_preflight_failure_journals_and_pauses_redrive(project, monkeypatch):
    """A typed cleanup-preflight refusal is journaled and passed to the injected
    pause as its exact cause; the resolved re-drive stops before post-rollback or
    any destructive reset can run.

    Ablation target: delete the `except RollbackPreflightError` journal/pause block
    and this test fails on the uncaught typed error; delete only `_pause` and it
    fails because `post_rollback` continues and the expected pause is absent.
    """
    repo = project.project
    ws = Workspace.default(project)
    flow = _make_flow(workspace=ws)
    task = _task(repo)
    created = repo / "uncertain" / "created.txt"
    created.parent.mkdir()
    created.write_text("run-created\n")
    (repo / "src.txt").write_text("tracked attempt\n")
    refuse_to_resolve(monkeypatch, created)
    monkeypatch.setattr(flow, "preserve_attempt_commits", lambda *args, **kwargs: None)
    monkeypatch.setattr(flow, "preserve_attempt_worktree", lambda *args, **kwargs: None)

    with pytest.raises(_Pause):
        flow.rollback_or_pause(task, cause="resolved")

    failure = flow.journal.fields("rollback-reset-failed")
    assert "preflight rollback cleanup" in failure["error"]
    assert len(flow.calls.pauses) == 1
    reason, story_key, cause = flow.calls.pauses[0]
    assert story_key == task.story_key
    assert isinstance(cause, verify.RollbackPreflightError)
    assert cause is not None and cause.__cause__ is not None
    assert str(cause) in reason
    assert flow.calls.emits == ["pre_rollback"]  # no post-reset re-drive continuation
    assert (repo / "src.txt").read_text() == "tracked attempt\n"
    assert created.read_text() == "run-created\n"


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


def test_restore_patch_escalates_on_apply_oserror(project, monkeypatch):
    # #343: the patch is read from disk, so an ENOENT/EACCES arrives untyped and
    # must still escalate rather than crash past the escalation.
    ws = Workspace.default(project)
    flow = _make_flow(workspace=ws)
    task = _task(project.project)
    task.restore_patch = "patch.diff"
    task.phase = Phase.DEV_RUNNING
    monkeypatch.setattr(verify, "resolve_restore_path", lambda raw, root: Path(root) / raw)

    def boom(repo, patch):
        raise OSError(2, "No such file or directory")

    monkeypatch.setattr(verify, "apply_patch", boom)

    with pytest.raises(_Pause):
        flow.restore_patch(task)

    assert task.phase == Phase.DEV_VERIFY
    assert flow.calls.escalates
    assert "attempt-restore-failed" in flow.journal.events()
