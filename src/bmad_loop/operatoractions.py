"""Project-level index of stories parked at ``awaiting-operator`` (#335).

A park commits everything an agent could do and records what a human still owes
in the spec's ``operator_actions:`` frontmatter. That frontmatter, plus the
board's ``awaiting-operator`` token, is the *committed truth*. This module adds
an index over it — ``.bmad-loop/operator-actions.json``, keyed by story key — so
``bmad-loop confirm`` can list every outstanding obligation across epics and runs
without re-reading every spec, and can name the run and commit a park came from
(provenance the spec itself does not carry).

Deliberately machine-local
--------------------------
Unlike ``.bmad-loop/decisions.json``, this store is NOT committed. It is written
from inside a story's commit window, where every way of committing it is wrong:

- committing it on its own path shifts HEAD past the ``commit_sha`` the park just
  stamped; and under ``scm.isolation = "worktree"`` it would advance the *target*
  branch while the unit branch is still out, so the merge-back fails outright
  under ``scm.merge_strategy = "ff"``;
- folding it into the story's own commit is only possible in the worktree, and
  the human reads the index at the project root;
- leaving it merely untracked dirties the tree that the epic-boundary auto-sweep
  refuses to run on, and the next story's ``git add -A`` would sweep one story's
  bookkeeping into another story's commit.

So the writer registers the path in the repository's local git exclude instead.
The index stays invisible to git, and none of the above can happen.

The cost, stated plainly because ``confirm`` has to answer for it: the index
lives on the machine that ran the orchestrator, so that is where a park is
confirmed. A fresh clone has none, and cannot rebuild one — ``spec_file`` is
reported by the dev session in its result JSON and is not derivable from a story
key, so the path to a parked story's spec survives only here. ``confirm``
therefore refuses rather than guessing, and says why. Confirming from a second
machine needs a *committed* index and the park-commit sequencing to go with it;
that is a follow-up, not something to fake with a glob.

Because the index can still drift from the committed truth it points at (a
hand-edited spec, a ``git revert``, a story re-driven to ``done``), ``validate``
carries ``operator.registry-stale`` and ``operator.actions-malformed``. Both are
warnings: an index is only safe to keep out of the commit if something reports
its drift and nothing gates on it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from . import sprintstatus, verify
from .bmadconfig import ProjectPaths
from .frontmatter import operator_actions_of, read_frontmatter, status_of
from .install import _worktree_local_exclude
from .platform_util import atomic_replace

STORE_REL = Path(".bmad-loop") / "operator-actions.json"
AWAITING_OPERATOR = verify.AWAITING_OPERATOR


def store_path(project: Path) -> Path:
    return project / STORE_REL


# --------------------------------------------------------------- store I/O


def load(project: Path) -> dict[str, dict]:
    """The park index, ``{story_key: {actions, spec_file, commit, run_id,
    parked_at}}``. Tolerant of a missing or malformed file (returns ``{}``): the
    index is a convenience over committed truth, so an unreadable one degrades to
    "no index" rather than blocking a human from confirming their own work."""
    path = store_path(project)
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    return data if isinstance(data, dict) else {}


def _write_store(project: Path, data: dict) -> None:
    path = store_path(project)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    atomic_replace(tmp, path)
    _exclude_from_git(project)


def _exclude_from_git(project: Path) -> None:
    """Keep the index out of git's view (see the module docstring for why).

    Reuses the helper worktree provisioning already excludes its seeded tool
    files with; it resolves ``--git-common-dir``, so the pattern lands in the
    MAIN repository's exclude file even when the park is committing inside a
    linked worktree — which is exactly the case that needs it. Best-effort by
    construction (the helper returns quietly when git cannot be queried), and
    that is the right failure mode: an index that could not be excluded is still
    a correct index."""
    _worktree_local_exclude(project, [STORE_REL.as_posix()])


# ------------------------------------------------------------ record + drop


def record_park(
    project: Path,
    story_key: str,
    *,
    actions: list[str],
    spec_file: str,
    commit: str,
    run_id: str,
    parked_at: str,
) -> None:
    """Index a story the run just parked. Re-parking the same key overwrites its
    entry rather than accumulating: a story owes whatever its latest park says it
    owes, and a stale action list is worse than none."""
    data = load(project)
    data[story_key] = {
        "actions": list(actions),
        "spec_file": spec_file,
        "commit": commit,
        "run_id": run_id,
        "parked_at": parked_at,
    }
    _write_store(project, data)


def drop(project: Path, story_key: str) -> bool:
    """Remove a story's entry, returning whether one was there. No write when the
    key is absent, so confirming a story the index never knew about (a fresh
    clone) leaves no file behind."""
    data = load(project)
    if story_key not in data:
        return False
    del data[story_key]
    _write_store(project, data)
    return True


# ------------------------------------------------- joining index to truth


@dataclass(frozen=True)
class ParkedStory:
    """One index entry joined back to the committed truth it points at.

    The index is what `confirm` can *find*; the spec and the board are what it is
    allowed to *believe*. Keeping both readings on one object is what lets the
    command refuse precisely — "the index says parked, the board says done" is a
    different message, and a different remedy, from "no such story".

    ``spec_status`` / ``board_status`` are None when that side could not be read
    at all (spec missing or unreadable, board missing or the key absent), which
    is deliberately distinct from a side that reads some *other* status."""

    story_key: str
    actions: tuple[str, ...]
    spec_path: Path | None
    spec_status: str | None
    board_status: str | None
    commit: str
    run_id: str
    parked_at: str

    @property
    def confirmable(self) -> bool:
        """Whether both sides of the committed truth still describe a park with
        something owed. Everything else is drift, and `confirm` refuses it rather
        than flipping a board on the index's word alone."""
        return (
            bool(self.actions)
            and self.spec_status == AWAITING_OPERATOR
            and self.board_status == AWAITING_OPERATOR
        )

    def drift(self) -> str | None:
        """Why this entry is not confirmable, phrased for a human, or None when
        it is. One sentence per cause, most-fundamental first — a missing spec
        explains a missing status, so it must be reported instead of it."""
        if self.spec_path is None:
            return "the index records no spec file for it"
        if self.spec_status is None:
            return f"its spec is missing or unreadable ({self.spec_path})"
        if self.spec_status != AWAITING_OPERATOR:
            return f"its spec now says status: {self.spec_status}"
        if self.board_status is None:
            return "it is not on the sprint board"
        if self.board_status != AWAITING_OPERATOR:
            return f"the board now says {self.board_status}"
        if not self.actions:
            return "it declares no readable actions"
        return None


def resolve(project: Path, paths: ProjectPaths) -> list[ParkedStory]:
    """Every index entry joined to its spec and board status, sorted by key.

    Reading degrades rather than raises throughout: this backs `confirm --list`
    and a `validate` warning, and neither may be the thing that crashes on a spec
    someone deleted. The actions come from the SPEC when it can be read and from
    the index only as a fallback — the spec is the committed truth, so a spec
    edited after the park shows the human what they actually owe now."""
    entries = load(project)
    out: list[ParkedStory] = []
    for key in sorted(entries):
        entry = entries[key] if isinstance(entries[key], dict) else {}
        spec_path = _spec_path(entry, paths)
        spec_fm = _spec_frontmatter(spec_path)
        spec_actions = operator_actions_of(spec_fm) if spec_fm is not None else ()
        out.append(
            ParkedStory(
                story_key=key,
                actions=spec_actions or actions_of(entry),
                spec_path=spec_path,
                spec_status=status_of(spec_fm) if spec_fm is not None else None,
                board_status=_board_status(paths, key),
                commit=str(entry.get("commit") or ""),
                run_id=str(entry.get("run_id") or ""),
                parked_at=str(entry.get("parked_at") or ""),
            )
        )
    return out


def _spec_path(entry: dict, paths: ProjectPaths) -> Path | None:
    spec_file = str(entry.get("spec_file") or "")
    if not spec_file:
        return None
    try:
        return verify.resolve_spec_path(spec_file, paths)
    except (OSError, ValueError):
        return None


def _spec_frontmatter(spec_path: Path | None) -> dict | None:
    """The spec's frontmatter, or None when there is no readable spec there."""
    if spec_path is None:
        return None
    try:
        if not spec_path.is_file():
            return None
        return read_frontmatter(spec_path)
    except (OSError, UnicodeDecodeError):
        return None


def _board_status(paths: ProjectPaths, key: str) -> str | None:
    try:
        return sprintstatus.story_status(paths.sprint_status, key)
    except (sprintstatus.SprintStatusError, OSError, UnicodeDecodeError):
        return None


def actions_of(entry: dict) -> tuple[str, ...]:
    """The actions an index entry declares, read through the *same* normalizer
    the spec frontmatter goes through (:func:`frontmatter.operator_actions_of`).

    Sharing the reading is the point: the index is written from a spec and
    compared back against one, so a shape that collapses to ``()`` on the spec
    side must collapse to ``()`` here too. Two readings would let a hand-edited
    index disagree with the spec about what a human owes."""
    return operator_actions_of({"operator_actions": entry.get("actions")})
