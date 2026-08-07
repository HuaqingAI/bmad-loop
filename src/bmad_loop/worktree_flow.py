"""Per-unit worktree isolation + integration flow.

Extracted from :class:`bmad_loop.engine.Engine` (issue #244, findings F-3/F-9a):
``Engine`` was a god-class whose worktree/integration cluster is an independent
state machine. It lives here as a collaborator built from narrow dependencies
(repo paths, the policy, run state, journal, plugin registry, loaded adapters)
plus a handful of engine callbacks (emit a plugin hook, save state, run the
per-unit ready gate, carry isolated ledger writes, escalate-pause, and get/set
the engine's active workspace).
The collaborator never receives the whole ``Engine`` — it cannot reach engine
internals beyond those callables.

``Engine`` keeps same-name private methods that delegate here, so its tests and
the ``SweepEngine``/``StoriesEngine`` subclasses see an unchanged surface.

``provision_worktree`` (previously in ``install.py``) is rehomed here too so the
runtime control loop no longer imports the installer; ``install.py`` re-exports
it lazily for its own tests.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from importlib import resources
from pathlib import Path
from typing import TYPE_CHECKING, Callable, NoReturn

from . import gates, verify
from .install import (
    _REVIEW_LAYER_SKILLS,
    BASE_SKILLS,
    BMAD_DIR,
    BMAD_SCRIPTS_SEED_REL,
    BMAD_SEED_EXCLUDES,
    CENTRAL_CONFIG_REL,
    DEV_PRIMITIVE_MARKERS,
    DEV_PRIMITIVE_ROLES,
    HOOK_SCRIPT_REL,
    MERGED_REVIEW_SKILL,
    MODULE_SKILLS,
    RENDER_DIR_REL,
    RENDERER_SCRIPT_MARKER,
    RENDERER_SCRIPT_UNIT_REL,
    RENDERER_SEED_SENTINELS,
    _absent_renderer_sources,
    _copy_traversable,
    _is_dir,
    _is_file,
    _occupied,
    _renderer_unit_required,
    _walk_traversable_files,
    _worktree_local_exclude,
    dev_primitive_or_default,
    merge_hooks,
    missing_stories_support,
    renderer_stub_resolved,
    resolve_review_layers,
)
from .model import Phase
from .process_host import get_process_host
from .workspace import (
    UnitWorkspace,
    Workspace,
    close_unit_workspace,
    discard_worktree,
    unit_worktrees_dir,
)

if TYPE_CHECKING:
    from .adapters.base import CodingCLIAdapter
    from .adapters.profile import CLIProfile
    from .bmadconfig import ProjectPaths
    from .journal import Journal
    from .model import RunState, StoryTask
    from .plugins import PluginRegistry
    from .policy import Policy


# CLI profile name -> the agent id the Unity-MCP CLI's `setup-mcp` expects (see
# `unity-mcp-cli setup-mcp --list`). All but claude differ only by claude's
# "-code" suffix; codex/gemini/cursor and any custom profile pass through as-is.
_SETUP_MCP_AGENT_IDS = {"claude": "claude-code"}


def _setup_mcp_agent_id(profile_name: str) -> str:
    """Map a CLI profile name to its Unity-MCP `setup-mcp` agent id."""
    return _SETUP_MCP_AGENT_IDS.get(profile_name, profile_name)


def _worktree_skill_copy_candidates(repo_root: Path, tree: str) -> tuple[str, ...]:
    """Every upstream skill worth best-effort copying into ``tree``."""
    resolved = resolve_review_layers(repo_root, tree)
    return tuple(dict.fromkeys((*BASE_SKILLS, *(resolved.skills() if resolved else ()))))


def _required_worktree_skills(repo_root: Path, tree: str) -> tuple[str, ...]:
    """Upstream skills whose absence deterministically stalls this tree's run.

    Match :func:`install.missing_base_skills`: gate the selected dev primitive and
    resolved required review skills only. If the review shape is unknown, prefer a
    present merged reviewer; otherwise require the standalone fallback reviewers.
    Catalog-only and advisory skills remain copy candidates but never arm the fatal
    pre-dispatch completeness gate.
    """
    resolved = resolve_review_layers(repo_root, tree)
    if resolved is not None:
        review_skills = tuple(resolved.required)
    elif (repo_root / tree / MERGED_REVIEW_SKILL / "SKILL.md").is_file():
        review_skills = (MERGED_REVIEW_SKILL,)
    else:
        review_skills = tuple(sorted(_REVIEW_LAYER_SKILLS))
    return tuple(dict.fromkeys((dev_primitive_or_default(repo_root, tree), *review_skills)))


def _drop_inert_tracked_file_patterns(
    worktree: Path, patterns: set[str]
) -> tuple[set[str], str | None]:
    """Drop shield patterns that name a TRACKED REGULAR FILE, keeping the rest.

    Such a pattern shields nothing and costs something (#392, reported from production
    by an external user whose repo-hygiene gate then blocked the story's commit).
    Measured on git 2.55.0 with the shield's own private-exclude + worktree-scoped
    `core.excludesFile` shape, a tracked `.codex/hooks.json`, and its pattern present:

    * `git add -A` STAGES a modification to it anyway — ignore rules are consulted only
      for untracked paths, so the pattern never did the job it is here for;
    * `git ls-files -ci --exclude-standard` inside the worktree REPORTS it, which is the
      tracked-and-ignored state hygiene gates reject.

    So the pattern is pure cost and dropping it is free. This is the second half of #384
    — its reporter proposed it as their option 3 ("skip any pattern whose path already
    contains tracked files … costs nothing and removes the surprising case entirely");
    PR #385 landed option 1 (scope + lifetime) and this half was dropped rather than
    rejected, which is how a second reporter hit it.

    A tracked DIRECTORY keeps its pattern: measured, that one really does hide new
    children, which is the shield working. Its tracked children still answer `-ci`, and
    no pattern shape avoids that — `dir/*`, `dir/**` and a trailing negation all
    measured identical to `dir`, since gitignore cannot re-include under an excluded
    parent, while the one shape that cleared the report leaked a new file into the
    commit. Not a case this function can fix; the tradeoff favors the shield.

    Patterns ending in `/` are directory-shaped by construction (`RENDER_DIR_REL`) and
    are never probed.

    Degrades by KEEPING every pattern when git cannot answer: the shield staying too
    wide is cosmetic, while dropping a pattern on a guess can leak the orchestrator's
    own seeded files into a story commit. Returns the surviving patterns and a reason
    for the caller to journal, or None when nothing went wrong.

    NOT-A-REPO IS SILENT, and it is the ordinary case rather than an edge one: many
    callers provision plain non-repo directories, where there is no index, no shield and
    no `git add -A` to be wrong about. The same `rev-parse --absolute-git-dir` gate
    `_worktree_local_exclude` skips on is asked FIRST, so this cannot emit a degrade per
    call and train operators past the one that matters. Only a repo that answers that
    probe and then fails the per-path one is reported."""
    try:
        if verify.git_bytes(worktree, "rev-parse", "--absolute-git-dir").returncode != 0:
            return patterns, None
    except (verify.GitError, OSError):
        return patterns, None
    kept: set[str] = set()
    unprobed: list[str] = []
    for pattern in patterns:
        rel = pattern.lstrip("/")
        if not rel or pattern.endswith("/"):
            kept.add(pattern)
            continue
        try:
            if not verify.path_tracked_file(worktree, rel):
                kept.add(pattern)
        except (verify.GitError, OSError) as e:
            kept.add(pattern)
            unprobed.append(f"{rel} ({e})")
    if unprobed:
        return kept, (
            "worktree git-add shield could not check whether these paths are tracked, "
            "so their patterns were kept; a tracked file among them will read as "
            f"ignored to repo-hygiene checks (#392): {'; '.join(sorted(unprobed))}"
        )
    return kept, None


def _seed_bmad_tree(worktree: Path, repo_root: Path) -> list[str]:
    """Merge the repo's project-local BMAD surface into an isolated worktree.

    Renderer-backed skills receive the worktree as their project root and do not
    walk upward for ``_bmad``. Copy every usable file except generated render output,
    per-file and without clobbering checkout content. The shared Traversable walk is
    intentional: unlike ``rglob``, it descends a symlinked child directory, allowing
    the result-side completeness predicates to see every file the copier considered.

    Returns the paths that need the worktree-local git-add shield: the root when it
    was absent before seeding, otherwise each file that actually landed.
    """
    src_root = repo_root / BMAD_DIR
    if not _is_dir(src_root):
        return []
    dst_root = worktree / BMAD_DIR
    had_bmad = _is_dir(dst_root)
    try:
        tops = sorted(src_root.iterdir(), key=lambda entry: entry.name)
    except OSError:
        return []

    seeded: list[str] = []
    for top in tops:
        if top.name in BMAD_SEED_EXCLUDES:
            continue
        for rel, src in _walk_traversable_files(top, top.name):
            if not _is_file(src):
                continue
            dst = dst_root.joinpath(*rel.split("/"))
            if _copy_traversable(
                src,
                dst,
                skip_existing=True,
                worktree=worktree,
                repo_root=repo_root,
            ):
                seeded.append(f"{BMAD_DIR}/{rel}")
    if not seeded:
        return []
    return [BMAD_DIR] if not had_bmad else seeded


def _bmad_scripts_seed_incomplete(worktree: Path, repo_root: Path) -> bool:
    """Whether a required repo renderer unit member missed the worktree.

    Match the renderer preflight's content-keyed required-file predicate. Arbitrary
    sibling scripts are still merge-seeded, but their absence cannot prove the
    renderer will HALT and therefore must not arm the CRITICAL escalation gate.
    """
    return any(
        _renderer_unit_required(repo_root, rel)
        and _is_file(repo_root / rel)
        and not _is_file(worktree / rel)
        for rel in RENDERER_SCRIPT_UNIT_REL
    )


def _central_config_seed_incomplete(worktree: Path, repo_root: Path) -> bool:
    """Whether the repo's required renderer config failed to reach the worktree."""
    return _is_file(repo_root / CENTRAL_CONFIG_REL) and not _is_file(worktree / CENTRAL_CONFIG_REL)


def base_skills_seed_incomplete(worktree: Path, repo_root: Path, trees: Sequence[str]) -> list[str]:
    """Required upstream skill rels present in the repo but absent in the worktree.

    ``BASE_SKILLS`` is a copy-if-present catalog, not a requirement set. Gate only
    the selected primitive and required review skills; advisory/conditional and
    inactive catalog entries are still provisioned best-effort, but their absence
    cannot prove the dev/review session will stall. Unknown review shapes use the
    same merged-or-standalone fallback as the run-start preflight.

    Within those active skills, mirror the deterministic run-start contract rather
    than treating every descendant as fatal: reviewers require ``SKILL.md``; the dev
    primitive additionally requires its defined markers and any renderer snapshot
    sources its worktree copy resolves. Other descendants still copy best-effort,
    but their absence does not prove the outer session will write no artifact.

    A missing ``SKILL.md`` reports the coarse skill rel. A repo directory without
    ``SKILL.md`` remains the run-start preflight's concern and cannot produce a false
    CRITICAL escalation here. Stories mode adds its content-keyed dispatch probe at
    the caller, where the run mode is available.
    """
    missing: list[str] = []
    for tree in dict.fromkeys(trees):
        primitive = dev_primitive_or_default(repo_root, tree)
        for skill in _required_worktree_skills(repo_root, tree):
            repo_skill = repo_root / tree / skill
            worktree_skill = worktree / tree / skill
            if not _is_file(repo_skill / "SKILL.md"):
                # Distinguish an unreadable directory from a skill the repo simply
                # does not carry. The walk yields only the former as a directory leaf.
                if any(_is_dir(src) for _, src in _walk_traversable_files(repo_skill)):
                    missing.append(f"{tree}/{skill}")
                continue
            if not _is_file(worktree_skill / "SKILL.md"):
                missing.append(f"{tree}/{skill}")
                continue
            if skill != primitive:
                continue
            missing.extend(
                f"{tree}/{skill}/{rel}"
                for rel in DEV_PRIMITIVE_MARKERS
                if _is_file(repo_skill / rel) and not _is_file(worktree_skill / rel)
            )
            missing.extend(
                f"{tree}/{skill}/{rel}"
                for rel in _absent_renderer_sources(worktree_skill)
                if _is_file(repo_skill.joinpath(*rel.split("/")))
            )
    return missing


def worktree_seed_undelivered(
    worktree: Path,
    repo_root: Path,
    seed_files: Sequence[str] = (),
    seed_globs: Sequence[str] = (),
    config_paths: Sequence[str] = (),
) -> list[str]:
    """Seed rels the repo carries that never reached the worktree.

    Source containment is deliberately *not* an eligibility requirement: a source
    symlink resolving outside the repo is the canonical entry the seed loop refuses
    and this result check must report. Delivery does require every usable source
    entry to remain within the repo, matching the copier. Destination containment is
    likewise required so a path outside the worktree cannot masquerade as delivery.

    Hook configs need a different result question because provisioning writes the
    hook registration itself after seeding. For those rels, existence proves nothing;
    source escape, a symlinked destination, or destination escape proves the seed was
    refused. This report is informational and is never an escalation gate.
    """
    worktree = worktree.resolve()
    repo_root = repo_root.resolve()
    rels = [str(rel) for rel in seed_files]
    for pattern in seed_globs:
        rels.extend(
            match.relative_to(repo_root).as_posix() for match in sorted(repo_root.glob(pattern))
        )
    hook_configs = {Path(rel) for rel in config_paths}

    def contained(path: Path, root: Path) -> bool:
        try:
            return path.resolve().is_relative_to(root)
        except (OSError, RuntimeError):
            return False

    def delivered(src: Path, dst: Path) -> bool:
        """Whether every usable source descendant has a matching destination."""
        if not contained(src, repo_root) or not contained(dst, worktree):
            return False
        if _is_file(src):
            return _is_file(dst)
        if not _is_dir(src) or not _is_dir(dst):
            return False

        complete = True

        def visit_dir(rel: str, source) -> bool:
            nonlocal complete
            if not isinstance(source, Path) or not contained(source, repo_root):
                complete = False
                return False
            target = dst.joinpath(*rel.split("/")) if rel else dst
            if not contained(target, worktree) or not _is_dir(target):
                complete = False
            return True

        for child_rel, child in _walk_traversable_files(src, _visit_dir=visit_dir):
            target = dst.joinpath(*child_rel.split("/")) if child_rel else dst
            if _is_file(child):
                if (
                    not isinstance(child, Path)
                    or not contained(child, repo_root)
                    or not contained(target, worktree)
                    or not _is_file(target)
                ):
                    complete = False
            elif _is_dir(child):
                # Directories are yielded only when enumeration was refused, so
                # their unknown descendants cannot be claimed as delivered.
                complete = False
        return complete

    undelivered: list[str] = []
    for rel in dict.fromkeys(rels):
        src = repo_root / rel
        if not (_is_file(src) or _is_dir(src)):
            continue
        dst = worktree / rel
        if Path(rel) in hook_configs:
            try:
                destination_is_link = dst.is_symlink()
            except OSError:
                destination_is_link = True
            if not contained(src, repo_root) or not contained(dst, worktree) or destination_is_link:
                undelivered.append(rel)
            continue
        if delivered(src, dst):
            continue
        undelivered.append(rel)
    return undelivered


def provision_worktree(
    worktree: Path,
    profiles: Sequence[CLIProfile],
    repo_root: Path,
    seed_files: Sequence[str] = (),
    seed_globs: Sequence[str] = (),
    *,
    on_degraded: Callable[[str], None] | None = None,
) -> list[str]:
    """Make a freshly-created git worktree a self-sufficient bmad-loop project.

    A worktree checks out tracked files only, but the skill trees (.claude/skills,
    .agents/skills), the hook config, and the project's gitignored MCP/CLI configs
    are absent from the checkout. Without them the bundled bmad-loop-* skills are missing,
    the Stop-signal hook never fires, and isolated sessions can't reach their MCP
    server. Lay the bundled skills + signal hook into the worktree for the active
    CLI profiles, and copy the `seed_files` configs in from the main repo. The
    upstream skills the orchestrator drives (BASE_SKILLS: bmad-dev-auto + the review
    hunters, plus whatever review layers this project's own config names) are not
    bundled in the wheel, so they are copied from the MAIN REPO's installed tree
    instead — together with `_bmad/custom/`, the customization those layers resolve
    through, so the isolated run resolves the same layer set the preflight
    validated. Quiet (no stdout) — unlike `install_into` this runs inside the
    engine loop under a TUI. No-op when there's nothing to do.

    seed_globs are project-relative glob patterns (e.g. ".claude/skills/*") expanded
    against the main repo; every match is copied into the worktree under the same
    relative path, copy-when-absent like seed_files. A game-engine plugin uses these
    to pull its MCP-generated skill tree (gitignored, so absent from the checkout)
    into a per_worktree Editor's checkout.

    A `seed_files` entry naming a DIRECTORY whose destination already exists is
    seeded child by child: the children the checkout lacks are copied in, the ones
    it carries are left untouched. A worktree checks out tracked files, so such a
    dir always exists and the entry would otherwise be a total no-op (issue #230).

    Kept safe against the unit's eventual `git add -A` commit:
    - skills + seed files are copied only when ABSENT — at FILE granularity, so a
      project that commits its own skill tree (e.g. .agents/) or config keeps it
      untouched (no diff merged back);
    - the hook points at the MAIN repo's already-installed relay via an absolute
      path (the relay locates the run dir from $BMAD_LOOP_RUN_DIR, not its own
      location), so nothing is written into the worktree's .bmad-loop/;
    - everything we wrote is excluded from git, in a file private to THIS worktree
      (`.git/worktrees/<id>/info/exclude`, activated per-worktree) that dies with it
      when the worktree is removed. It is never the repository-wide
      `.git/info/exclude`: that file is shared with the operator's own checkout and
      permanent, so shielding through it hid every new file under a tool dir from
      their `git add -A` forever (#384). That write is best-effort: when git can't
      be queried at all it is skipped silently, but any fault after that — including
      a refusal to scope the shield, and a shield that was written but which git does
      not resolve to — is reported to `on_degraded` (once, with the reason) rather
      than swallowed, and the shield is skipped rather than widened back to the
      shared file. An unshielded worktree lets `git add -A` stage the tool files, so
      it must not fail invisibly.
    Skill trees, the per-CLI hook config, and the seeded configs all live in dirs
    projects gitignore — but the exclude shields them even when a project doesn't.

    seed_files are copied BEFORE the hook step so a seeded settings file that is
    also a hook config_path (.claude/settings.json, .gemini/settings.json) keeps its
    real content and just gets the Stop hook merged in, rather than being created empty.

    The repo's `_bmad/` surface is also merge-seeded, excluding generated render
    output. Renderer and upstream-skill completeness failures share the return
    channel with ordinary no-op seeds so they are journaled, but the caller re-probes
    the skill result and content-gates the renderer sentinels before escalating.

    Returns the `seed_files` entries that copied NOTHING because everything they
    name was already present, plus reserved completeness reports. A directory entry
    that seeded even one child is not a no-op and is not reported.
    """
    if not profiles and not seed_files and not seed_globs and not _is_dir(repo_root / BMAD_DIR):
        return []
    worktree = worktree.resolve()
    repo_root = repo_root.resolve()
    relay = repo_root / HOOK_SCRIPT_REL
    skills_root = resources.files("bmad_loop.data").joinpath("skills")

    # project gitignored MCP/CLI configs: copy from the main repo when absent.
    # Resolve-and-contain guards against an `..`/absolute entry escaping either tree.
    seeded: list[str] = []
    # Entries that named a real source but copied nothing, because every path they
    # name already exists. Reported to the caller (this function is quiet by
    # contract — it runs under a TUI) because the no-op is otherwise silent: an
    # entry that reads as applied configuration is not. Per-CHILD skips inside a
    # directory entry are deliberately not reported — the checkout is expected to
    # carry its tracked children, so that is routine rather than a
    # misconfiguration, exactly like the glob-expanded matches below.
    skipped: list[str] = []
    for rel in seed_files:
        src = (repo_root / rel).resolve()
        raw = worktree / rel
        dst = raw.resolve()
        if not src.is_relative_to(repo_root) or not dst.is_relative_to(worktree):
            continue
        if not (_is_file(src) or _is_dir(src)):
            continue
        if _occupied(dst):
            # A live symlink remains the ordinary existing-destination no-op. This
            # arm must precede the raw-vs-resolved refusal below so it stays named in
            # `skipped` rather than turning into a silent drop.
            if dst != raw:
                skipped.append(str(rel))
                continue
            # File entries keep the classic copy-when-absent skip. A destination
            # that is not a directory while the source is (the checkout carries a
            # FILE where the seed names a dir) is a type mismatch: recursing would
            # try to mkdir over the file, so the entry is skipped whole instead.
            if not _is_dir(src) or not _is_dir(raw):
                skipped.append(str(rel))
                continue
            # A DIRECTORY whose destination exists is the case #230 reported: the
            # checkout carries some tracked child, so the whole entry used to be a
            # no-op — including the gitignored children that are absent and would
            # clobber nothing. Recurse instead, copying only what is missing.
            if not _copy_traversable(
                src,
                raw,
                skip_existing=True,
                worktree=worktree,
                repo_root=repo_root,
            ):
                # every child was already present: still a total no-op, still
                # reported. Only a PARTIAL seed stops being reported.
                skipped.append(str(rel))
                continue
            # Partially seeded, so the entry must still reach `patterns` below:
            # the children we just wrote have to stay out of the unit's
            # `git add -A`. Excluding the whole dir is safe — an exclude does not
            # untrack the tracked children that were already there.
            seeded.append(rel)
            continue
        # `resolve()` is non-strict, so a dangling leaf or parent link answers for
        # its target. Never mkdir/copy through it. Existing live links were handled
        # by the skip arm above, preserving copy-when-absent reporting.
        if dst != raw:
            continue
        if _copy_traversable(
            src,
            raw,
            skip_existing=True,
            worktree=worktree,
            repo_root=repo_root,
        ):
            seeded.append(rel)

    # glob-seeded trees (e.g. an engine plugin's MCP skill dirs): expand each
    # pattern against the main repo and copy matches in, same contain guard +
    # copy-when-absent semantics. rel is taken from the unresolved match so the
    # worktree path mirrors the repo layout; resolve only guards containment.
    for pattern in seed_globs:
        for match in sorted(repo_root.glob(pattern)):
            rel = match.relative_to(repo_root)
            src = match.resolve()
            raw = worktree / rel
            dst = raw.resolve()
            if not src.is_relative_to(repo_root) or not dst.is_relative_to(worktree):
                continue
            if not (_is_file(src) or _is_dir(src)) or _occupied(dst):
                continue
            if dst != raw:
                continue
            if not _copy_traversable(
                src,
                raw,
                skip_existing=True,
                worktree=worktree,
                repo_root=repo_root,
            ):
                continue
            # as_posix so the exclude pattern anchors on Windows too (os.sep would not)
            seeded.append(rel.as_posix())

    # Renderer-backed skills are handed the worktree as their project root. Merge
    # the repo's project-local BMAD surface after explicit seeds (operator intent wins
    # on collisions) and reserve the two renderer sentinels for result-side checks.
    seeded_bmad = _seed_bmad_tree(worktree, repo_root)
    skipped = [rel for rel in skipped if rel not in RENDERER_SEED_SENTINELS]
    if _bmad_scripts_seed_incomplete(worktree, repo_root):
        skipped.append(BMAD_SCRIPTS_SEED_REL)
    if _central_config_seed_incomplete(worktree, repo_root):
        skipped.append(CENTRAL_CONFIG_REL)

    # bundled skills into each CLI's skill tree (deduped: codex+gemini share one);
    # never clobber a skill the checkout already carries (tracked or pre-existing).
    for tree in dict.fromkeys(p.skill_tree for p in profiles):
        tree_dir = worktree / tree
        for skill in MODULE_SKILLS:
            dst = tree_dir / skill
            _copy_traversable(
                skills_root.joinpath(skill),
                dst,
                skip_existing=True,
                worktree=worktree,
            )
        # Known gap, deliberately restated rather than folded into the CRITICAL gate
        # below: wheel MODULE_SKILLS have no result-side completeness predicate. Story
        # worktrees deterministically dispatch the upstream dev/review skills, while
        # these bundled operator/triage skills are a different invocation surface; a
        # partial wheel copy therefore cannot honestly reuse the guaranteed-stall gate.
        # Closing that observability gap needs its own required-consumer contract.
        # The orchestrator-driven upstream skills are not in the wheel; copy them
        # from the MAIN REPO's installed tree (same tree path) so an isolated
        # worktree can still resolve the dev primitive and the review layers. Skip
        # silently when the main repo lacks them — the run-start preflight reports
        # it.
        #
        # BASE_SKILLS names BOTH primitive eras, which is what carries the skill
        # across the rename: the resolution below returns REVIEW skills only, so a
        # primitive the catalog did not name would be silently left behind (the
        # is_dir guard swallows the miss) and every isolated session would stall on
        # an Unknown command.
        #
        # BASE_SKILLS is only the floor. The review layers this project actually
        # invokes are read from the installed primitive — resolved on disk, so a
        # renamed project resolves its own layers rather than degrading to the
        # static catalog — exactly as the preflight reads them, so a reviewer named
        # by a project override (a custom or renamed skill) is provisioned too.
        # Validating a skill here and then not copying it is how preflight passes in
        # the main checkout while the isolated review fails on a skill that was
        # never there.
        for skill in _worktree_skill_copy_candidates(repo_root, tree):
            dst = tree_dir / skill
            src = (repo_root / tree / skill).resolve()
            if not src.is_relative_to(repo_root) or not _is_dir(src):
                continue
            _copy_traversable(
                src,
                dst,
                skip_existing=True,
                worktree=worktree,
                repo_root=repo_root,
            )

    # Re-ask the result rather than trusting copy bookkeeping. A user-authored seed
    # can happen to spell a skill rel, so only this disk predicate may arm the gate.
    skipped.extend(
        base_skills_seed_incomplete(
            worktree, repo_root, [profile.skill_tree for profile in profiles]
        )
    )

    # per-CLI signal-hook registration, baked to the main repo's relay (absolute).
    # Hookless profiles (HTTP/SSE transport) have no config to merge.
    for profile in profiles:
        if profile.hookless:
            continue
        raw_config_path = worktree / profile.hooks.config_path
        # Refuse before mkdir, read, or write: hook commands are worktree-specific
        # and must never mutate a shared dotfile through a live or dangling link.
        # Inspect every component: a non-strict resolve either leaves a symlink cycle
        # unresolved — and so textually equal to the raw path — or raises RuntimeError,
        # which the except below takes; which of the two happens is interpreter-version
        # dependent. The resolved comparison additionally refuses ``..`` and absolute
        # profiles.
        refused = not raw_config_path.is_relative_to(worktree)
        cursor = raw_config_path
        try:
            while not refused and cursor != worktree:
                if cursor.is_symlink():
                    refused = True
                    break
                cursor = cursor.parent
            config_path = raw_config_path.resolve()
        except (OSError, RuntimeError):
            continue
        if refused or config_path != raw_config_path or not config_path.is_relative_to(worktree):
            continue
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config: dict = {}
        if _is_file(config_path):
            try:
                config = json.loads(config_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                config = {}
        host = get_process_host()
        interp = host.hook_interpreter()
        registrations = {
            native: f"{interp} {host.shell_quote(str(relay))} {canonical}"
            for native, canonical in profile.hooks.events.items()
        }
        config, changed = merge_hooks(config, registrations, profile.hooks.dialect)
        if changed:
            config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    # Shield exactly the paths we wrote (skill trees + hook configs + seeded
    # configs) from the unit's `git add -A`, in case a project doesn't gitignore
    # its tool dirs. Scoped to this worktree and expiring with it — these are
    # paths projects legitimately TRACK, so a repo-wide exclude here went on
    # hiding their new files from the operator's own checkout (#384).
    patterns = {f"/{p.skill_tree}" for p in profiles}
    # hookless profiles have no config_path, so there is nothing to shield: their
    # empty string would render as a bare "/", which git strips to a zero-length
    # pattern (the trailing slash becomes MUSTBEDIR) that matches nothing — inert,
    # unlike "/*" or "*", which do blanket. A junk line in a generated file, then,
    # not a worktree-wide exclusion.
    patterns |= {f"/{p.hooks.config_path}" for p in profiles if not p.hookless}
    patterns |= {f"/{rel}" for rel in seeded}
    patterns |= {f"/{rel}" for rel in seeded_bmad}
    if f"/{BMAD_DIR}" not in patterns:
        # The renderer may create or rewrite this generated directory during the
        # session, after provisioning has finished. Give it a dedicated transient
        # shield unless the blanket root shield already subsumes it: `/_bmad` prunes
        # the directory before git descends, so `/_bmad/render/` would provably never
        # be consulted. Avoiding that inert sibling keeps the worktree-local exclude
        # precise; the file and its lines disappear with this worktree.
        patterns.add(f"/{RENDER_DIR_REL}/")
    patterns, tracked_degrade = _drop_inert_tracked_file_patterns(worktree, patterns)
    if tracked_degrade is not None and on_degraded is not None:
        on_degraded(tracked_degrade)
    reason = _worktree_local_exclude(worktree, sorted(patterns))
    if reason is not None and on_degraded is not None:
        on_degraded(reason)
    return skipped


class WorktreeFlow:
    """Provision, drive, integrate and reclaim per-unit git worktrees.

    Built once per engine from narrow deps + engine callbacks (see module
    docstring). Behavior is identical to the cluster it was carved out of; the
    only structural changes are that engine-owned effects go through injected
    callables: ``emit`` fires a plugin hook (late-bound so a monkeypatched
    ``Engine._emit`` still wins), ``save`` persists run state, ``gate_unit`` runs
    the per-unit ready gate, ``carry_isolated_ledger_writes`` applies Engine-owned
    ledger bookkeeping after a successful merge, ``workspace_get``/``workspace_set``
    read and swap the engine's active workspace, and ``escalation_pause`` raises the
    engine's ``RunPaused`` (injected so this module need not import ``engine`` — that
    would reintroduce a runtime<->engine import cycle)."""

    def __init__(
        self,
        *,
        paths: ProjectPaths,
        policy: Policy,
        state: RunState,
        journal: Journal,
        run_dir: Path,
        registry: PluginRegistry,
        adapters_get: Callable[[], dict[str, CodingCLIAdapter]],
        open_unit_workspace: Callable[..., UnitWorkspace],
        emit: Callable[..., object],
        save: Callable[[], None],
        gate_unit: Callable[[StoryTask], bool],
        carry_isolated_ledger_writes: Callable[[StoryTask], None],
        escalation_pause: Callable[..., NoReturn],
        workspace_get: Callable[[], Workspace],
        workspace_set: Callable[[Workspace], None],
    ) -> None:
        self.paths = paths
        self.policy = policy
        self.state = state
        self.journal = journal
        self.run_dir = run_dir
        self._registry = registry
        # Read live (a getter, not a captured dict) so a test that rebinds
        # `engine.adapters` after construction is still seen here.
        self._adapters_get = adapters_get
        # Injected late-bound so a test patching the `engine.open_unit_workspace`
        # module global still wins (worktree_flow's own binding wouldn't).
        self._open_unit_workspace = open_unit_workspace
        self._emit = emit
        self._save = save
        self._gate_unit = gate_unit
        self._carry_isolated_ledger_writes = carry_isolated_ledger_writes
        self._pause = escalation_pause
        self._workspace_get = workspace_get
        self._workspace_set = workspace_set

    @property
    def isolated(self) -> bool:
        return self.policy.scm.isolation == "worktree"

    def ensure_target_branch(self) -> None:
        """Resolve (once, at run start) the branch every unit merges back into.

        No-op unless isolation=worktree. Default target is the branch checked out
        now; a configured target is created if missing and checked out in the
        main repo (merges land on whatever the main repo has checked out, and a
        unit worktree must never check out the target itself). Pinned in state so
        resume keeps targeting the same branch."""
        if not self.isolated or self.state.target_branch:
            return
        if self.policy.scm.failed_diff_unlimited:
            # the safety cap is off; make sure the operator knows a failed unit
            # could write a very large forensic patch.
            self.journal.append(
                "scm-failed-diff-unlimited",
                note="failed-unit diff capture is uncapped (scm.failed_diff_unlimited); "
                "changes.patch may be very large",
            )
        repo = self.paths.repo_root
        configured = self.policy.scm.target_branch.strip()
        if configured:
            if not verify.branch_exists(repo, configured):
                try:
                    verify.create_branch(repo, configured, "HEAD")
                except verify.GitError as e:
                    # e.g. an unborn repo (no commit to base a branch on).
                    self._pause(f"cannot create target branch {configured!r}: {e}", cause=e)
                self.journal.append("target-branch-created", branch=configured)
            if verify.current_branch(repo) != configured:
                verify.checkout_branch(repo, configured)
                self.journal.append("target-branch-checkout", branch=configured)
            self.state.target_branch = configured
        else:
            current = verify.current_branch(repo)
            if current == "HEAD":
                # detached HEAD has no branch to merge into; merges would land on
                # an unreferenced commit. Require a real branch (or a configured
                # target) before isolating work into worktrees.
                self._pause(
                    "isolation=worktree on a detached HEAD: check out a branch or "
                    "set scm.target_branch before running"
                )
            self.state.target_branch = current
        self.journal.append("target-branch", branch=self.state.target_branch)
        self._save()

    def worktree_profiles(self) -> list[CLIProfile]:
        """The distinct CLI profiles of the dev + review adapters, for provisioning
        their skills/hooks into a worktree. Adapters without a `profile` (e.g. test
        fakes) contribute nothing, so provisioning is a no-op for them.

        The role set is :data:`install.DEV_PRIMITIVE_ROLES` — the same constant
        `cli._skill_trees` gates on — rather than a local pair, so the provisioned
        set and the gated set cannot drift apart. A tree gated but not provisioned
        ships a session into the `Unknown command` stall the preflight exists to
        catch; a tree provisioned but not gated refuses runs over a skill no session
        reads."""
        seen: dict[str, CLIProfile] = {}
        adapters = self._adapters_get()
        for adapter in (adapters[role] for role in DEV_PRIMITIVE_ROLES):
            profile = getattr(adapter, "profile", None)
            if profile is not None and profile.name not in seen:
                seen[profile.name] = profile
        return list(seen.values())

    def engine_agent_ids(self) -> list[str]:
        """The Unity-MCP `setup-mcp` agent ids for every CLI that runs in a
        worktree (dev + review). A worktree can host more than one agent — e.g.
        dev=claude, review=codex — and each reads its own MCP config file, so the
        per_worktree setup must point every one of them at the worktree's Editor,
        not just the dev agent. Deduped, order-preserving; empty for test fakes."""
        ids: list[str] = []
        for profile in self.worktree_profiles():
            agent = _setup_mcp_agent_id(profile.name)
            if agent not in ids:
                ids.append(agent)
        return ids

    def _ledger_seed(self, worktree: Path) -> tuple[str, ...]:
        """The deferred-work ledger, when a worktree checkout cannot deliver it.

        `git worktree add` checks out TRACKED files only, so a project that
        gitignores its ledger — the default shape — gets a unit worktree with
        none. The orchestrator is the ledger's single writer under the generic
        dev path and writes through ``self.workspace.paths``, i.e. that missing
        copy: ``mark_done`` returns False on an absent file,
        ``verify_review_bundle`` reads the same absent file and never sees the
        ids `done`, so the bundle defers on a fixable retry and `open_ids`
        re-bundles the same work for ever (#426). No DONE-leg carry can rescue
        that — the unit never reaches DONE. Seeding moves the failure onto a leg
        that has one, where ``SweepEngine._carry_isolated_ledger_writes`` applies
        the close to the main checkout.

        The copy is not what delivers the close: every seeded rel is shielded
        from the unit's ``git add -A``, so the worktree's flip never rides the
        merge. The seeded ledger exists so the GATE can read the orchestrator's
        own write; the carry stays the delivery path, hence
        ``sweep-bundle-close-carry-uncommitted`` on a run that lands.

        Excluded: a ledger already in the checkout (without the exclusion the copy
        would be a no-op the seed loop reports as ``worktree-seed-skipped`` on every
        tracked-ledger project); one absent from the main checkout (dropped
        silently, so the entry would be invisible rather than merely inert — the
        common case, since the first harvest is what CREATES the file); and one
        resolving outside the project tree, which for an out-of-tree artifacts DIR
        ``ProjectPaths.rebased`` leaves unmoved, so the worktree already reads it.
        Presence is asked of the WORKTREE, not of git: that is the predicate the
        seed loop itself decides on, and unlike ``verify.path_tracked`` it costs no
        subprocess and cannot raise.

        A ledger that is itself a symlink keeps the dir in-tree, so ``rebased``
        moves it and the worktree path does not exist: the exclusion is still right
        for an out-of-repo target (``provision_worktree`` refuses that source
        whatever rel it is handed), but an in-repo target is seeded to the WRONG
        path and still hits #426 (#462). Resolving also names the target of a
        TRACKED ledger symlink whose target is untracked — the only path the seed
        loop will write, since it refuses to copy through a link. ``relative_to``
        decides PLACEMENT; containment is re-checked against both roots at the copy
        site.

        Deduped against ``scm.worktree_seed`` by the caller.
        """
        ledger = self.paths.deferred_work
        repo = self.paths.repo_root
        try:
            rel = ledger.resolve().relative_to(repo.resolve()).as_posix()
        except (OSError, RuntimeError, ValueError):
            return ()
        if not _is_file(ledger) or _is_file(worktree / rel):
            return ()
        return (rel,)

    def run_isolated(self, task: StoryTask, drive: Callable[[StoryTask], None]) -> None:
        """Run one unit's `drive` body in a fresh per-unit worktree, then merge
        it back into the target branch. `drive` either returns (DONE/DEFERRED →
        integrate) or raises RunPaused (spec-approval gate / escalation → leave
        the worktree mounted for resume/inspection, integration skipped)."""
        try:
            unit = self._open_unit_workspace(
                self.paths.repo_root,
                self.paths,
                self.state.run_id,
                task.story_key,
                self.state.target_branch,
                self.policy.scm.branch_per,
                self.run_dir,
            )
        except verify.GitSpawnError as e:
            # a spawn fault is machine-wide, not this unit's: deferring would
            # march the whole queue into DEFERRED one notification at a time
            # and end the run "finished" over a broken environment (#194/#343).
            self._pause(
                f"cannot spawn git while opening a worktree for {task.story_key}: {e}",
                task.story_key,
                cause=e,
            )
        except verify.GitError as e:
            # could not mount a worktree (e.g. branch_per=run with a kept-failed
            # unit still holding the shared branch). Defer this unit rather than
            # crash the whole run; the operator can free the branch and re-run.
            task.defer_reason = f"could not open worktree: {e}"
            task.phase = Phase.DEFERRED  # deliberate: no legal move from PENDING
            self.journal.append("worktree-open-failed", story_key=task.story_key, error=str(e))
            gates.notify(
                self.policy, self.run_dir, f"worktree open failed: {task.story_key}", str(e)
            )
            self._save()
            return
        task.worktree_path = str(unit.path)
        self.journal.append(
            "worktree-opened", story_key=task.story_key, branch=unit.branch, path=str(unit.path)
        )
        task.branch = unit.branch
        # A worktree checks out tracked files only, but the bmad-loop-* skill
        # trees + signal-hook config are typically gitignored, so they are absent
        # from the fresh checkout. Re-lay them into the worktree so the bundled
        # bmad-loop-* skills are present and the Stop-signal hook fires. Also seed the loaded
        # adapters' gitignored MCP/CLI configs so isolated sessions can reach their
        # MCP server (seed_adapter_defaults) plus any extra project-listed paths.
        profiles = self.worktree_profiles()
        scm = self.policy.scm
        seeds: list[str] = []
        if scm.seed_adapter_defaults:
            for profile in profiles:
                seeds.extend(profile.seed_files)
                # The hook config is seeded from the SAME list that shields it. It was
                # only ever in the shield set (`provision_worktree`'s `patterns`), never
                # here, so whether it got seeded depended on a profile happening to name
                # it twice: claude's `seed_files` carries `.claude/settings.json`, which
                # is also its `config_path`, and codex's does not carry
                # `.codex/hooks.json` (#471).
                #
                # What seeding fixes, stated only as far as it is measured: the hook
                # step in `provision_worktree` creates and writes that config whether
                # or not it was seeded (`merge_hooks` on an absent file returns
                # changed=True), so bmad-loop's OWN Stop hook registers either way.
                # What an unseeded worktree loses is the PROJECT's hook configuration —
                # the session runs against a file holding the relay registrations alone.
                # #471's reported stall is consistent with the CLI declining hooks from
                # a config it has not trusted (codex.toml's `first_run_note`), but that
                # mechanism is UNCONFIRMED and nothing here rests on it; the issue's own
                # stated mechanism — the file being absent — is false at this line.
                #
                # Deriving it from `config_path` rather than restating it per profile is
                # what keeps a future profile from regressing the same way. Hookless
                # profiles have no config to seed. The seed loop skips an occupied
                # destination, so a project that TRACKS this path keeps its checked-out
                # copy untouched.
                if not profile.hookless and profile.hooks.config_path:
                    seeds.append(profile.hooks.config_path)
        seeds.extend(scm.worktree_seed)
        seeds.extend(self._ledger_seed(unit.path))
        # plugins (e.g. the Unity engine) may prime an isolated checkout with
        # gitignored paths they need — e.g. an MCP-generated skill tree + client
        # config so the worktree's Editor MCP is reachable. Aggregate every loaded
        # plugin's declared seeds.
        seeds.extend(self._registry.seed_files())
        seed_files = list(dict.fromkeys(seeds))  # dedupe, preserve order
        seed_globs = self._registry.seed_globs()
        skipped_seeds = provision_worktree(
            unit.path,
            profiles,
            self.paths.repo_root,
            seed_files=seed_files,
            seed_globs=seed_globs,
            on_degraded=lambda msg: self._exclude_degraded(task.story_key, msg),
        )
        if skipped_seeds:
            # A seed entry whose destination already exists is a no-op. Harmless for
            # a file the checkout legitimately carries, but a directory entry is
            # skipped WHOLE the moment any child is tracked — so a `worktree_seed`
            # that looks applied can be copying nothing. Journal it; provision is
            # quiet by contract (it runs under the TUI).
            self.journal.append(
                "worktree-seed-skipped", story_key=task.story_key, entries=skipped_seeds
            )

        # A dropped arbitrary seed is observable but not fatal. Unlike the two gates
        # below, bmad-loop cannot know that a user/plugin config is required by the
        # session, and its usual trigger is a healthy shared/dotfile-managed config.
        # Hook config destinations cannot prove delivery because provisioning writes
        # the Stop registration itself after the seed step.
        undelivered_seeds = worktree_seed_undelivered(
            unit.path,
            self.paths.repo_root,
            seed_files=seed_files,
            seed_globs=seed_globs,
            config_paths=[p.hooks.config_path for p in profiles if not p.hookless],
        )
        if undelivered_seeds:
            self.journal.append(
                "worktree-seed-dropped", story_key=task.story_key, entries=undelivered_seeds
            )

        # Missing upstream skill content is a determinate stall for both inline and
        # renderer-era primitives. Re-probe disk rather than trusting skipped_seeds,
        # which a user-authored seed rel could otherwise forge. Check this first so a
        # wholly absent primitive is not misdiagnosed as a renderer-surface problem.
        trees = [p.skill_tree for p in profiles]
        absent_skills = base_skills_seed_incomplete(unit.path, self.paths.repo_root, trees)
        if absent_skills:
            reason = (
                "the worktree is missing required upstream skill contract files the repo has "
                f"({', '.join(absent_skills)}) — the session would stall having "
                "written nothing: on `Unknown command` when the whole skill is "
                "absent, or at a required primitive marker or renderer source. "
                "The usual cause is a required skill directory or file symlinked "
                "to a shared BMad install outside the repo, which worktree seeding "
                "cannot follow"
            )
            self.escalate_unit(task, reason)  # always raises RunPaused

        # Stories mode has a stricter, content-keyed primitive contract than sprint
        # mode. Re-run that exact preflight against the mounted worktree: a step-01
        # through-link can pass in the main checkout yet be refused during copying,
        # while a tracked stale worktree copy proves that existence alone is not
        # enough. Either shape would HALT a folder+id dispatch before writing a spec.
        stories_support = (
            missing_stories_support(unit.path, trees) if self.state.source == "stories" else []
        )
        if stories_support:
            short_dispatch = []
            for finding in stories_support:
                detail = finding.detail or {}
                rel = f"{detail['tree']}/{detail['skill']}/{detail['file']}"
                marker = detail.get("marker")
                short_dispatch.append(f"{rel} (missing {marker!r})" if marker else rel)
            reason = (
                "the worktree's dev primitive does not satisfy stories-mode dispatch "
                f"support ({', '.join(short_dispatch)}) — the folder+id session would "
                "HALT without writing a spec. The usual cause is a required router "
                "file symlinked to a shared BMad install outside the repo, which "
                "worktree seeding cannot follow"
            )
            self.escalate_unit(task, reason)  # always raises RunPaused

        # Provisioning owns these exact sentinel strings, but only a content-confirmed
        # renderer stub consumes the surface. The conjunct is load-bearing: inline
        # pre-#2601 SKILL.md projects may carry the same repo paths and must proceed.
        short_surface = [rel for rel in RENDERER_SEED_SENTINELS if rel in skipped_seeds]
        if short_surface and renderer_stub_resolved(self.paths.project, trees):
            reason = (
                f"the dev primitive renders via {RENDERER_SCRIPT_MARKER} but the "
                "worktree's renderer surface came up short of the repo's "
                f"({', '.join(short_surface)}) — the session would HALT without "
                "writing a spec. The usual cause is a symlinked _bmad/ pointing "
                "outside the repo, which worktree seeding cannot follow"
            )
            self.escalate_unit(task, reason)  # always raises RunPaused

        self._save()
        prev = self._workspace_get()
        self._workspace_set(unit.workspace)
        try:
            # A plugin (e.g. the Unity engine) may launch the unit's managed Editor
            # at pre_worktree_setup + wait for its MCP at pre_ready_gate before
            # driving. A veto (defer) at either stage leaves the task DEFERRED and
            # skips drive(); both fall through to _integrate_unit, which tears the
            # (empty) worktree down via the DEFERRED path.
            if self._gate_unit(task):
                self._emit("post_worktree_setup", task)
                drive(task)
        finally:
            # always run teardown — on success, on a deferral, and on a RunPaused
            # (spec gate / escalation) propagating through — before the workspace is
            # restored, so a managed Editor never outlives its worktree. Teardown
            # stages are observe-only (a veto here cannot un-tear-down).
            self._emit("pre_worktree_teardown", task)
            self._emit("post_worktree_teardown", task)
            self._workspace_set(prev)
        # reached only on a normal return (DONE or DEFERRED); a RunPaused from the
        # spec gate or an escalation propagates past here, leaving the worktree up.
        self.integrate_unit(task, unit)

    def _exclude_degraded(self, story_key: str, msg: str) -> None:
        """The git-add shield was owed for this unit's worktree and did not happen.

        Journaled AND notified, the way `worktree-open-failed` is above. The notify
        is the half that was missing: the shield's whole degrade policy is to SKIP
        rather than widen — activating over patterns it could not copy would shadow
        the operator's own excludes — and skipping is only defensible if the operator
        finds out. A run that ends "finished" with a journal line nobody reads is how
        the provisioned tool files reach a story's merge unnoticed.

        `gates.notify` is best-effort and never raises, and is inert unless
        `notify.file`/`notify.desktop` is configured, so this cannot break a run —
        which matters, because it is called from inside provisioning.
        """
        self.journal.append("worktree-exclude-degraded", story_key=story_key, error=msg)
        gates.notify(self.policy, self.run_dir, f"worktree exclude degraded: {story_key}", msg)

    def failed_diff_max_bytes(self) -> int | None:
        """Per-untracked-file size cap for a failed unit's forensic patch, in
        bytes — or None when the operator lifted the cap (scm.failed_diff_unlimited)."""
        scm = self.policy.scm
        if scm.failed_diff_unlimited:
            return None
        return scm.failed_diff_max_mb * 1_048_576

    def integrate_unit(self, task: StoryTask, unit: UnitWorkspace) -> None:
        self._emit("pre_integrate", task)
        scm = self.policy.scm
        # AWAITING_OPERATOR merges beside DONE: a parked story CARRIES A COMMIT
        # (that is what separates it from DEFERRED/ESCALATED), and stranding that
        # commit on a torn-down unit branch would lose finished work over an
        # obligation that lives outside the repo entirely. The human's remaining
        # actions are recorded in the registry, not in this worktree.
        if task.phase in (Phase.DONE, Phase.AWAITING_OPERATOR):
            # Merge the unit branch into the target branch locally. We open PRs
            # ourselves by hand once the branch has landed; the orchestrator only
            # commits the worktree onto the selected target.
            self.merge_local(task, unit)
            # Engine-authored ledger writes normally ride the unit commit, but a
            # gitignored ledger is omitted by finalize_commit's `git add -A` and
            # would otherwise disappear with successful teardown. Carry only after
            # merge: a tracked ledger reaches the target through the merge first,
            # where Engine's all-status provenance scan can deduplicate it.
            self._carry_isolated_ledger_writes(task)
            # Phase is already terminal and persisted before integration, so make
            # the carry completion durable here. Engine replays an unlatched carry
            # after a crash in the merge-to-carry window.
            task.isolated_ledger_carried = True
            self._save()
        else:  # DEFERRED — capture the diff, keep or drop per keep_failed
            patch = close_unit_workspace(
                unit,
                success=False,
                keep_failed=scm.keep_failed,
                run_dir=self.run_dir,
                unit_key=task.story_key,
                delete_branch=scm.delete_branch,
                detach_kept=scm.branch_per == "run",
                diff_max_file_bytes=self.failed_diff_max_bytes(),
                on_teardown_degraded=lambda msg: self.journal.append(
                    "worktree-teardown-degraded", story_key=task.story_key, error=msg
                ),
            )
            self.journal.append(
                "unit-closed",
                story_key=task.story_key,
                branch=unit.branch,
                kept=scm.keep_failed,
                patch=str(patch) if patch else None,
            )

    def merge_local(
        self,
        task: StoryTask,
        unit: UnitWorkspace,
        *,
        replay: bool = False,
        replay_strategy: str | None = None,
    ) -> None:
        """Merge a DONE unit's branch into the target branch from the main repo."""
        if not replay:
            self._emit("pre_merge", task)
        scm = self.policy.scm
        merge_strategy = scm.merge_strategy if replay_strategy is None else replay_strategy
        repo = self.paths.repo_root
        target = self.state.target_branch
        source = task.commit_sha or verify.rev_parse_head(unit.path)
        merge_ref = unit.branch
        if replay:
            current_source = verify.rev_parse_head(unit.path)
            if current_source != source:
                reason = (
                    f"merge replay of {unit.branch} into {target} blocked: the unit "
                    f"branch advanced from recorded source {source} to {current_source}; "
                    "the later commits were not produced or verified by the completed "
                    "session and the branch was preserved for manual recovery"
                )
                self.keep_branch_and_escalate(task, unit, reason)  # always raises RunPaused
                return
            # Pin both the collision allowlist and the merge operand to the
            # write-ahead SHA. The branch can move after the check; replay must
            # integrate only the commit the completed session actually proved.
            merge_ref = source
        # A per_worktree Unity Editor can leak asset writes into the *main*
        # checkout (see the unity plugin's worktree setup), dirtying the target with the very
        # files this branch already committed. Reconcile that first: clean only
        # the leaked copies of incoming files; refuse (escalate) if anything dirty
        # falls outside this branch's path set — that may be real operator work.
        try:
            cleaned = verify.clean_incoming_collisions(repo, target, merge_ref)
        except (verify.GitError, OSError) as e:
            # OSError joins GitError because clean_incoming_collisions mutates the
            # checkout directly (unlink/iterdir/rmdir) — a non-spawn FS fault the
            # #343 chokepoint cannot translate. Crashing here would strand a DONE
            # unit mid-merge; the keep-branch escalation is the point of this guard.
            if isinstance(e, (verify.GitSpawnError, OSError)):
                # environment fault (spawn failure or direct-FS error) — there may
                # be no stray files at all, so no "clean them" guidance: the inner
                # error is the diagnosis.
                reason = (
                    f"merge of {unit.branch} into {target} blocked: could not "
                    f"reconcile the target checkout ({e}) — fix the underlying "
                    f"fault, then `bmad-loop resume {self.state.run_id}`"
                )
            else:
                reason = (
                    f"merge of {unit.branch} into {target} blocked: the target checkout has "
                    f"uncommitted changes that are not part of this branch (likely a Unity "
                    f"Editor wrote into the main project) — clean them, then "
                    f"`bmad-loop resume {self.state.run_id}`. {e}"
                )
            self.keep_branch_and_escalate(task, unit, reason)  # always raises RunPaused
            return
        if cleaned:
            self.journal.append(
                "merge-target-cleaned",
                story_key=task.story_key,
                branch=unit.branch,
                paths=cleaned,
            )
        if not replay:
            # The task is already terminal and durable here. Record integration
            # intent immediately before git so a host loss after merge success but
            # before `unit-merged` can safely re-run the merge instead of losing a
            # gitignored ledger when the stale worktree is reclaimed.
            self.journal.append(
                "unit-merge-started",
                story_key=task.story_key,
                branch=unit.branch,
                target=target,
                strategy=merge_strategy,
                source=source,
            )
        try:
            verify.merge_branch(
                repo,
                merge_ref,
                strategy=merge_strategy,
                message=self.merge_message(task),
                allow_empty_squash=replay,
            )
        except verify.GitError as e:
            # genuine content conflict against the target: keep the branch for
            # manual merge. The unit committed cleanly (phase is already DONE,
            # which has no legal transition), so escalate directly.
            reason = (
                f"merge of {unit.branch} into {target} failed "
                f"(content conflict against the target): {e}"
            )
            self.keep_branch_and_escalate(task, unit, reason)  # always raises RunPaused
            return  # defensive: never fall through to the success teardown below
        self.journal.append(
            "unit-merged",
            story_key=task.story_key,
            branch=unit.branch,
            target=self.state.target_branch,
            strategy=merge_strategy,
            source=source,
        )
        self._emit("post_merge", task)
        close_unit_workspace(
            unit,
            success=True,
            keep_failed=scm.keep_failed,
            run_dir=self.run_dir,
            unit_key=task.story_key,
            delete_branch=scm.delete_branch,
            on_teardown_degraded=lambda msg: self.journal.append(
                "worktree-teardown-degraded", story_key=task.story_key, error=msg
            ),
        )

    def keep_branch_and_escalate(self, task: StoryTask, unit: UnitWorkspace, reason: str) -> None:
        """Preserve a DONE unit's branch (no delete, kept for manual merge) and
        escalate. Shared by the two merge-back failure paths: a target dirtied
        with stray work, and a genuine content conflict."""
        close_unit_workspace(
            unit,
            success=False,
            keep_failed=True,
            run_dir=self.run_dir,
            unit_key=task.story_key,
            delete_branch=False,
            diff_max_file_bytes=self.failed_diff_max_bytes(),
        )
        self.escalate_unit(task, reason)  # always raises RunPaused

    def escalate_unit(self, task: StoryTask, reason: str) -> None:
        """Mark a unit ESCALATED, notify, and pause the run.

        Callers escalate outside a legal transition, before dispatch or after a
        completed merge attempt, so the phase is set directly rather than advanced.
        """
        task.phase = Phase.ESCALATED
        self.journal.append("story-escalated", story_key=task.story_key, reason=reason)
        gates.notify(
            self.policy,
            self.run_dir,
            f"CRITICAL escalation: {task.story_key}",
            f"{reason} — resolve, then `bmad-loop resume {self.state.run_id}`",
        )
        self._save()
        self._pause(reason, task.story_key)

    def merge_message(self, task: StoryTask) -> str:
        return f"Merge {task.branch} into {self.state.target_branch} (bmad-loop)"

    def gc_run_worktrees(self) -> None:
        """Reclaim this run's worktree scaffolding once it finishes cleanly.

        DONE units drop their worktree at merge time; this is a safety net for a
        worktree leaked by a crash between merge and teardown, plus it prunes
        stale git admin entries and removes the now-empty run worktree dir.
        Worktrees deliberately kept for inspection (a kept-failed/escalated unit)
        are left in place and journaled so the operator can find them."""
        if not self.isolated:
            return
        repo = self.paths.repo_root
        for task in self.state.tasks.values():
            if task.phase == Phase.DONE and task.worktree_path:
                wt = Path(task.worktree_path)
                if wt.is_dir():
                    discard_worktree(repo, task.worktree_path, task.branch, run_dir=self.run_dir)
            elif task.terminal and task.worktree_path and Path(task.worktree_path).is_dir():
                # kept on purpose (keep_failed): leave it, but surface where.
                self.journal.append(
                    "worktree-kept", story_key=task.story_key, path=task.worktree_path
                )
        verify.worktree_prune(repo)
        worktrees_parent = unit_worktrees_dir(self.run_dir)
        if worktrees_parent.is_dir() and not any(worktrees_parent.iterdir()):
            worktrees_parent.rmdir()

    def reopen_unit(self, task: StoryTask) -> UnitWorkspace:
        """Reconstruct the UnitWorkspace for an in-flight unit on resume, from
        the worktree path + branch persisted on the task. The worktree must still
        be mounted — if it was pruned out from under us we cannot safely reuse it,
        so escalate rather than run a session in a missing directory."""
        wt = Path(task.worktree_path)
        if not wt.is_dir():
            self.escalate_unit(
                task,
                f"worktree for {task.story_key} is gone ({wt}); cannot resume in place",
            )
        # spec_file is persisted relative to the worktree (model.to_dict) so the
        # state stays portable; re-absolutize it against the reopened worktree.
        if task.spec_file and not Path(task.spec_file).is_absolute():
            task.spec_file = str(wt / task.spec_file)
        return UnitWorkspace(
            workspace=Workspace(root=wt, paths=self.paths.rebased(wt)),
            repo_root=self.paths.repo_root,
            branch=task.branch,
            path=wt,
            baseline=task.baseline_commit or "",
        )
