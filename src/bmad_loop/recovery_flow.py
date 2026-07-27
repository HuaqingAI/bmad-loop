"""Attempt rollback + recovery-ref preservation flow.

Extracted from :class:`bmad_loop.engine.Engine` (issue #244, PR 2/2): the
rollback/preserve cluster is an independent recovery state machine. It lives
here as a collaborator built from narrow dependencies (repo paths, the policy,
run state, journal, run dir) plus a getter for the engine's swappable active
workspace and a handful of engine callbacks (emit a plugin hook, save state,
escalate a task, and escalation-pause). The collaborator never receives the
whole ``Engine`` — it cannot reach engine internals beyond those callables.

``Engine`` keeps same-name private methods that delegate here, so its tests and
the ``SweepEngine``/``StoriesEngine`` subclasses (which override nothing in this
cluster) see an unchanged surface.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Callable, NoReturn

from . import gates, verify
from .model import Phase
from .platform_util import safe_ref_segment
from .statemachine import advance

if TYPE_CHECKING:
    from .bmadconfig import ProjectPaths
    from .journal import Journal
    from .model import RunState, StoryTask
    from .policy import Policy
    from .workspace import Workspace


class RecoveryFlow:
    """Roll back or pause a stopped/abandoned attempt, parking any work it did on
    named recovery refs before the reset.

    Built once per engine from narrow deps + engine callbacks (see module
    docstring). Behavior is identical to the cluster it was carved out of; the
    only structural changes are that engine-owned effects go through injected
    callables: ``emit`` fires a plugin hook (late-bound so a monkeypatched
    ``Engine._emit`` still wins), ``save`` persists run state, ``escalate``
    routes an intent-gap restore failure through the engine's escalation, and
    ``escalation_pause`` raises the engine's ``RunPaused`` (injected so this
    module need not import ``engine`` — that would reintroduce a runtime<->engine
    import cycle). ``workspace_get`` reads the engine's live (worktree-swappable)
    active workspace."""

    def __init__(
        self,
        *,
        paths: ProjectPaths,
        policy: Policy,
        state: RunState,
        journal: Journal,
        run_dir: Path,
        workspace_get: Callable[[], Workspace],
        emit: Callable[..., object],
        save: Callable[[], None],
        escalate: Callable[[StoryTask, str], None],
        escalation_pause: Callable[..., NoReturn],
    ) -> None:
        self.paths = paths
        self.policy = policy
        self.state = state
        self.journal = journal
        self.run_dir = run_dir
        # Read live (a getter, not a captured ref) so a run that swaps the
        # engine's `self.workspace` to a mounted unit worktree is seen here.
        self._workspace_get = workspace_get
        # Injected late-bound so a test patching `engine._emit` still wins
        # (recovery_flow's own binding wouldn't).
        self._emit = emit
        self._save = save
        self._escalate = escalate
        self._pause = escalation_pause

    def protected_relpaths(self) -> tuple[str, ...]:
        """Repo-relative posix paths of the BMAD artifact folders. These are
        orchestrator-owned: never counted as a dev attempt's dirtiness (the
        resolve workflow corrects the frozen spec here) and preserved through
        rollback. Folders configured outside the repo are skipped — nothing to
        protect there."""
        workspace = self._workspace_get()
        out: list[str] = []
        for protected in (
            workspace.paths.output_folder,
            workspace.paths.implementation_artifacts,
            workspace.paths.planning_artifacts,
        ):
            try:
                rel = protected.relative_to(workspace.root).as_posix()
            except ValueError:
                continue  # configured outside the repo; nothing to protect here
            # "." (folder == repo root) as an exclude/keep prefix would cover the
            # whole tree — drop it so a misconfig can't disable the dirty check.
            if rel and rel != ".":
                out.append(rel)
        return tuple(out)

    def rollback_or_pause(self, task: StoryTask, *, cause: str = "stopped") -> None:
        """Recover from an attempt that won't proceed.

        No-op when the tree is already at the attempt's baseline (nothing this
        attempt touched, ignoring orchestrator-owned artifact folders): neither a
        reset nor a pause is needed. This is also what lets the manual-recovery
        instructions terminate — after the operator resets and resumes, the
        now-clean tree skips straight through instead of re-pausing on the
        still-set ``baseline_commit``.

        A ``cause="resolved"`` re-drive is human-initiated (the operator ran the
        resolve workflow and re-armed the story), so it always auto-recovers and
        never pauses, regardless of ``scm.rollback_on_failure``. For the entire
        re-drive (``task.resolved_redrive``, latched at resume and cleared once the
        correction is committed) the BMAD artifact folders are treated as
        orchestrator-owned: excluded from the dirty check (the corrected spec must
        not read as a failed attempt) and preserved through every reset — so a
        later mid-re-drive retry/defer reset can't silently revert the correction.

        Otherwise (a stopped/abandoned attempt) recovery depends on where the
        attempt ran. Inside a mounted unit worktree it always auto-recovers: the
        worktree is disposable, the attempt's work is parked on preserve refs
        before the reset, and ``scm.rollback_on_failure`` gates *in-place*
        (isolation="none") recovery only (#161). In the main checkout the flag
        governs: OFF (default) leaves the working tree untouched and emits a bold
        manual-recovery notice that pauses the run (stop-and-wait); ON does a
        clean reset to baseline. Either way pre-existing untracked files are
        preserved; there is no blanket ``git clean``."""
        workspace = self._workspace_get()
        resolved = cause == "resolved"
        # preserve the corrected spec for the whole re-drive, not just the first
        # reset; the auto-recover (pause-vs-reset) decision below is unaffected.
        redrive = resolved or task.resolved_redrive
        protected = self.protected_relpaths() if redrive else ()
        # Un-determinable dirty check (git timeout/failure, #156) ⇒ assume dirty:
        # never skip recovery on an unproven "clean", never crash the run. The
        # normal branching below then decides — OFF pauses (worktree kept), ON /
        # resolved auto-recovers behind its preserve steps, keeping the re-drive
        # pause-free contract intact.
        dirty = True
        if task.baseline_commit:
            try:
                dirty = verify.attempt_dirty(
                    workspace.root,
                    task.baseline_commit,
                    task.baseline_untracked,
                    exclude=protected,
                )
            except verify.GitError as exc:
                self.journal.append(
                    "rollback-dirty-check-failed", story_key=task.story_key, error=str(exc)
                )
        if task.baseline_commit and not dirty:
            self.journal.append("rollback-skipped-clean", story_key=task.story_key)
            return
        # A mounted unit worktree is disposable by design: its branch never
        # touches the operator's checkout and the attempt's work is parked on
        # recovery refs before any reset. `scm.rollback_on_failure` gates
        # *in-place* (isolation="none") recovery only — pausing here would emit
        # main-checkout reset instructions for a tree the operator never works
        # in (#161). Compared by path, not by policy: a worktree-mode call that
        # reaches this method *outside* a mounted unit (e.g. resume with no
        # worktree recorded) still targets the main checkout and must pause.
        in_unit_worktree = workspace.root != self.paths.repo_root
        if resolved or in_unit_worktree or self.policy.scm.rollback_on_failure:
            # `preserve_ref` names where *this* rollback parked the attempt. Clear
            # it first: a later attempt that parks nothing (clean tree, no commits
            # above baseline, or a preserve failure) must not inherit the previous
            # attempt's ref — the defer notice would then send the operator to work
            # that is not the deferred attempt's.
            task.preserve_ref = None
            self.journal.append(
                "rollback-auto",
                story_key=task.story_key,
                baseline=task.baseline_commit or "",
                note="reverting tracked changes + run-created untracked files",
            )
            # Give a plugin (the Unity engine) a chance to quiesce before the reset
            # rewrites tracked files under it — e.g. save + close open scenes so a
            # git reset --hard can't leave a shared Editor showing a run-freezing
            # "scene changed on disk" modal. Observe-only, like pre_worktree_teardown:
            # the returned ctx is ignored and never routed through _vetoed — a failed
            # quiesce must never block a rollback.
            self._emit("pre_rollback", task)
            # A re-drive (resolved / mid-re-drive) is contractually pause-free, so it
            # preserves best-effort but never blocks; a plain rollback pauses rather
            # than reset past work it could not park.
            self.preserve_attempt_commits(task, allow_pause=not redrive)
            # Park the attempt's uncommitted diff too, so the reset below (and its
            # untracked cleanup) can't silently destroy in-progress work. Runs only
            # if preserve_attempt_commits did not pause (plain-rollback preserve
            # failure); best-effort, never blocks.
            self.preserve_attempt_worktree(task)
            self.safe_reset(task, preserve=protected)
            # Refresh the plugin's view of the now-reset tree (the Unity engine
            # re-imports assets). Observe-only for the same reason as pre_rollback.
            self._emit("post_rollback", task)
            return
        self.pause_for_manual_recovery(task, task.baseline_commit or "")
        return  # unreachable: pause_for_manual_recovery always raises

    def safe_reset(self, task: StoryTask, *, preserve: tuple[str, ...] = ()) -> None:
        """Revert tracked changes to the task baseline and remove only the
        untracked files this run created — never a blanket `git clean`. Used by
        the gated/resolved rollback and by internal ledger recovery (sweep
        migration), which restores the orchestrator's own state and must not
        pause. The BMAD artifact folders are always kept from untracked deletion;
        ``preserve`` (set only on a resolved re-drive) additionally keeps their
        *tracked* content alive through the reset, so a just-corrected spec is not
        reverted. Sweep passes no ``preserve`` — it wants the broken ledger gone."""
        workspace = self._workspace_get()
        verify.safe_rollback(
            workspace.root,
            task.baseline_commit or "",
            baseline_untracked=task.baseline_untracked,
            keep=(".bmad-loop", *self.protected_relpaths()),
            preserve=preserve,
        )

    def restore_patch(self, task: StoryTask) -> None:
        """Re-apply the latched intent-gap patch (BMAD-METHOD #2564) onto the
        baseline tree so the re-driven session resumes review (step-04) on the
        restored diff instead of re-implementing. No-op unless a restore is latched.

        Applied from inside `_dev_phase`'s loop, right before each dispatch that
        runs against a fresh baseline (the first attempt and every non-fixable
        rollback retry — the loop gates this on ``feedback is None``, so a
        fixable-feedback retry that KEEPS the attempt's tree is never double-applied,
        and the patch file's own untracked/tracked content is excluded from
        `baseline_untracked` because that snapshot is taken before the first apply).
        This is the plan's "apply after every baseline reset" seam, placed here
        rather than in `_rollback_or_pause` so the patch always lands after the
        clean baseline_untracked snapshot — avoiding a mid-re-drive reset preserving
        the patch's own new files and then colliding with the re-apply.

        On apply failure we escalate rather than dispatch a session onto a
        half-restored tree; the task is mid-dispatch (DEV_RUNNING), so step it to
        the escalatable DEV_VERIFY phase first (`_escalate` raises RunPaused)."""
        if not task.restore_patch:
            return
        workspace = self._workspace_get()
        patch = verify.resolve_restore_path(task.restore_patch, workspace.root)
        try:
            verify.apply_patch(workspace.root, patch)
        except verify.GitError as e:
            self.journal.append(
                "attempt-restore-failed",
                story_key=task.story_key,
                patch=task.restore_patch,
                error=str(e),
            )
            # Call-site invariant: `_dev_phase` advances the task to DEV_RUNNING
            # immediately before dispatch, and this runs on that path only — so the
            # step to DEV_VERIFY is unconditional. It is required because `_escalate`
            # cannot transition out of DEV_RUNNING directly.
            advance(task, Phase.DEV_VERIFY)
            self._escalate(task, f"intent-gap restore patch failed to apply: {e}")
        self.journal.append("attempt-restored", story_key=task.story_key, patch=task.restore_patch)

    def prune_preserve_refs(self) -> None:
        """Bounded retention for both recovery-ref families at run start — the
        attempt-preserve/* branches and the refs/attempt-preserve-dirty/*
        worktree snapshots: keep the newest scm.preserve_keep of each by
        committer date, delete the tail (mirrors the runs/cleanup retention
        knobs — without it the refs grow unbounded on a long-lived project).
        Best-effort: a git failure is journalled per family and never blocks or
        pauses the run — the refs are a safety net, not run state — and a
        failure in one family never skips the other. preserve_keep = 0 disables
        pruning entirely."""
        keep = self.policy.scm.preserve_keep
        if keep <= 0:
            return
        workspace = self._workspace_get()
        for family, prune in (
            ("attempt-preserve", verify.prune_preserve_refs),
            ("attempt-preserve-dirty", verify.prune_preserve_dirty_refs),
        ):
            try:
                deleted = prune(workspace.root, keep)
            except Exception as exc:  # housekeeping must never crash the
                # run: a git timeout/OSError here would otherwise escape to the crash
                # handler, so anything beyond the expected GitError is journalled too
                # A partial prune (PrunePreserveError) already deleted refs before one
                # stuck — that destructive half must stay structurally auditable, not
                # buried in the error string.
                partial = getattr(exc, "deleted", [])
                if partial:
                    self.journal.append(f"{family}-pruned", count=len(partial), refs=partial)
                failed = getattr(exc, "failed", [])
                if failed:
                    self.journal.append(f"{family}-prune-failed", error=str(exc), failed=failed)
                else:
                    self.journal.append(f"{family}-prune-failed", error=str(exc))
                continue
            if deleted:
                self.journal.append(f"{family}-pruned", count=len(deleted), refs=deleted)

    def preserve_attempt_commits(self, task: StoryTask, *, allow_pause: bool) -> None:
        """Before an auto-rollback's hard reset, park any commits the attempt made
        above its baseline under a named recovery ref, so `reset --hard baseline`
        can't silently orphan committed work (it survives `git gc` and is
        recoverable by name, not just the reflog). No-op when the attempt added no
        commits — an uncommitted-only revert is the intended, non-destructive case.

        If commits exist but the ref cannot be created: with ``allow_pause`` (a
        plain rollback) refuse to reset — pause for manual recovery rather than
        destroy the work. On a re-drive (``allow_pause=False``) the caller's
        contract forbids pausing, so journal the failure and let the reset proceed
        (the re-drive is a human-directed discard of the failed attempt)."""
        baseline = task.baseline_commit
        if not baseline:
            return
        workspace = self._workspace_get()
        commits = verify.commits_above(workspace.root, baseline)
        if not commits:
            return
        head = verify.rev_parse_head(workspace.root)  # the tip the recovery ref parks at
        # run_id can be an arbitrary user `--run-id`; ref-sanitize it (same
        # identity-for-clean-ids / digest-for-dirty contract as the unit branches) so
        # an exotic/overlong id can't blow the ref-name limit, fail `git branch`, and
        # drop the recovery ref (which on a re-drive would then reset past the work
        # anyway).
        slug = safe_ref_segment(self.state.run_id)
        try:
            ref = verify.preserve_commits(
                workspace.root,
                baseline,
                f"attempt-preserve/{slug}-{head[:8]}",
                commits=commits,
            )
        except verify.GitError:
            ref = None  # branch creation failed — treat as a preservation failure
        if ref is None:
            # commits exist (just enumerated) but the ref did not take.
            self.journal.append("attempt-preserve-failed", story_key=task.story_key, head=head)
            if allow_pause:
                # the commits at HEAD could not be parked — the notice must NOT tell
                # the operator to blindly `reset --hard` (that would discard them).
                self.pause_for_manual_recovery(task, baseline, preserve_failed=True)
            return  # re-drive: never pause — proceed to the (human-directed) reset
        task.preserve_ref = ref
        self.journal.append(
            "attempt-commits-preserved", story_key=task.story_key, ref=ref, count=len(commits)
        )

    def preserve_attempt_worktree(self, task: StoryTask) -> None:
        """Before an auto-rollback's hard reset, park the attempt's *uncommitted*
        working-tree changes (tracked edits + run-created untracked files) under a
        named recovery ref, so `reset --hard baseline` and its untracked cleanup
        can't silently destroy in-progress work. Complements
        `_preserve_attempt_commits` (which parks *committed* work above baseline);
        together they cover the whole attempt. No-op when the tree is clean vs HEAD
        — the intended non-destructive uncommitted-revert case. Best-effort: a
        capture failure is journaled but never blocks the (human-directed re-drive
        or policy-gated) reset — the recovery ref is a safety net, not a gate."""
        baseline = task.baseline_commit
        if not baseline:
            return
        workspace = self._workspace_get()
        # Same ref-sanitized slug as preserve_attempt_commits so an exotic/overlong
        # --run-id can't blow the ref-name limit and drop the ref.
        slug = safe_ref_segment(self.state.run_id)
        # ``baseline_commit`` is fixed across the whole dev retry loop, so keying the
        # ref on the baseline alone would make a 2nd dirty rollback reuse the name and
        # orphan the 1st attempt's snapshot. ``task.attempt`` only ever increments
        # (never resets), so it uniquely discriminates each retry's recovery ref.
        ref = f"refs/attempt-preserve-dirty/{slug}-{baseline[:8]}-{task.attempt}"
        try:
            parked = verify.snapshot_worktree(
                workspace.root, ref, baseline_untracked=task.baseline_untracked
            )
        except verify.GitError as exc:
            # Keep the git failure detail (commit-tree/update-ref stderr): if the
            # following reset destroys work, this is the only breadcrumb explaining
            # why the safety-net snapshot couldn't be captured.
            self.journal.append(
                "attempt-worktree-preserve-failed", story_key=task.story_key, error=str(exc)
            )
            return
        if parked:
            # Last writer wins over preserve_attempt_commits' branch on purpose:
            # the snapshot is commit-tree'd parented at the attempt's HEAD
            # (verify.snapshot_worktree), so it already contains the commits that
            # branch points at — one ref recovers the whole attempt.
            task.preserve_ref = parked
            self.journal.append("attempt-worktree-preserved", story_key=task.story_key, ref=parked)

    def pause_for_manual_recovery(
        self, task: StoryTask, baseline: str, *, preserve_failed: bool = False
    ) -> None:
        """Leave the tree untouched, surface bold manual-recovery instructions, and
        pause the run. Always raises RunPaused. Three notice shapes: (a, default)
        the OFF path for a stopped/abandoned in-place attempt with no commits of
        its own — plain manual-rollback steps; (b, ``preserve_failed``) rollback is
        ON/resolved but the attempt's commits above baseline could not be parked on
        a recovery ref, so an automatic ``reset --hard`` would silently discard
        them — a distinct notice that names the at-risk commits and never tells the
        operator to blindly reset; (c) the OFF path but the attempt COMMITTED work
        above its baseline (#100: a completed session whose run died before the
        orchestrator folded the result) — instructing a bare ``reset --hard`` there
        would discard finished, possibly already-pushed commits, so this notice
        tells the operator to save and check integration state first. A *resolved*
        escalation never reaches here — `_rollback_or_pause` auto-recovers that
        human-initiated re-drive regardless of `scm.rollback_on_failure`."""
        workspace = self._workspace_get()
        short = baseline[:12] or "<baseline_commit>"
        # Name the tree every instruction targets. Usually the main checkout,
        # but a preserve-failure pause can fire while a unit worktree is
        # mounted — a bare "current HEAD" / `git reset --hard` there reads as
        # the operator's own checkout (whose HEAD is typically *at* the
        # baseline, making the quoted commit range empty) and invites a
        # destructive reset of a tree the attempt never touched (#161).
        root = workspace.root
        commits: list[str] = []
        if baseline:
            # advisory probe: a git fault here must not block the pause itself
            try:
                commits = verify.commits_above(root, baseline)
            except verify.GitError:
                commits = []
        if preserve_failed:
            notice = (
                "**ACTION REQUIRED — commits could not be auto-preserved**\n"
                f"Story **{task.story_key}**'s attempt committed work above its "
                "baseline, but a recovery ref for those commits could not be created, "
                "so the automatic rollback was refused rather than `reset --hard` "
                "past (and discard) them. **Your commits are intact at the current "
                f"HEAD of `{root}`.**\n"
                f'  1. **Save them first** — e.g. `git -C "{root}" branch my-rescue '
                f"HEAD` (the commits are `{short}..HEAD` there).\n"
                "  2. Only once they are safe, discard the attempt if you want to: "
                f'`git -C "{root}" reset --hard {short}`, then review/remove leftover '
                "untracked files.\n"
                f"Then run `bmad-loop resume {self.state.run_id}`."
            )
        elif commits:
            notice = (
                "**ACTION REQUIRED — manual recovery needed (committed work present)**\n"
                f"Story **{task.story_key}**'s attempt was stopped with auto-rollback "
                "OFF, and it **committed work above its baseline**. **Your commits "
                f"are intact at the current HEAD of `{root}`.** They may already be "
                "integrated or pushed to a remote — do NOT reset before checking.\n"
                f'  1. **Save them first** — e.g. `git -C "{root}" branch my-rescue '
                f"HEAD` (the commits are `{short}..HEAD` there).\n"
                "  2. Check whether they are already integrated (merged, pushed to "
                "a remote, referenced by open PRs) before discarding anything.\n"
                "  3. Only if you decide to discard the attempt: "
                f'`git -C "{root}" reset --hard {short}`, then review/remove leftover '
                "untracked files.\n"
                f"Then run `bmad-loop resume {self.state.run_id}`."
            )
        else:
            why = (
                f"Story **{task.story_key}**'s attempt was stopped and auto-rollback "
                f"is OFF, so the working tree at `{root}` was left exactly as-is "
                "for you to inspect.\n"
            )
            notice = (
                "**ACTION REQUIRED — manual rollback needed**\n"
                f"{why}"
                "To discard this attempt yourself:\n"
                "  1. **BACK UP any untracked files you want to keep** — the reset "
                "below deletes uncommitted work.\n"
                f'  2. `git -C "{root}" reset --hard {short}` then review/remove '
                "leftover untracked files.\n"
                "  3. **Restore the files you backed up in step 1.**\n"
                f"Then run `bmad-loop resume {self.state.run_id}`. To let the "
                "orchestrator do a safe automatic rollback next time, enable "
                "`[scm] rollback_on_failure` (it discards the attempt's uncommitted "
                "work but never deletes pre-existing untracked files)."
            )
        self.journal.append(
            "rollback-manual-required",
            story_key=task.story_key,
            baseline=baseline,
            commits=len(commits),
        )
        gates.notify(
            self.policy,
            self.run_dir,
            f"ACTION REQUIRED: manual rollback for {task.story_key}",
            notice,
        )
        self._save()
        self._pause(notice, task.story_key)
