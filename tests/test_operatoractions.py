"""The project-level operator-actions index (#335, part 3 of 4).

The index is what `bmad-loop confirm` can find; the spec and the board are what
it is allowed to believe. These tests pin both halves: the store round-trip, and
the join back onto the committed truth that decides whether an entry is
confirmable or drifted.
"""

from __future__ import annotations

import json

import pytest
from conftest import install_bmad_config, spec_path, write_spec, write_sprint

from bmad_loop import operatoractions, verify

ACTIONS = ["buy example.com at the registrar", "publish the _acme-challenge TXT record"]


def _index_only(paths, key="1-1-a", **kw):
    """Index a story WITHOUT writing anything else, so a test about git's view of
    the index is not confounded by the untracked spec/config a park also leaves."""
    operatoractions.record_park(
        paths.project,
        key,
        actions=kw.get("actions", ACTIONS),
        spec_file=kw.get("spec_file", "spec.md"),
        commit="abcdef1234567890",
        run_id="run-1",
        parked_at="2026-07-28",
    )


def _park(paths, key="1-1-a", *, actions=None, status="awaiting-operator", board=None):
    """Index a story and write the committed state it points at."""
    if not (paths.project / "_bmad" / "bmm" / "config.yaml").is_file():
        install_bmad_config(paths)
    acts = ACTIONS if actions is None else actions
    spec = spec_path(paths, key)
    write_spec(spec, status, "base", operator_actions=acts)
    board_now = {}
    if paths.sprint_status.is_file():
        import yaml

        board_now = yaml.safe_load(paths.sprint_status.read_text())["development_status"]
    write_sprint(paths, {**board_now, "epic-1": "in-progress", key: board or status})
    operatoractions.record_park(
        paths.project,
        key,
        actions=list(acts) if isinstance(acts, list) else [],
        spec_file=spec.relative_to(paths.project).as_posix(),
        commit="abcdef1234567890",
        run_id="run-1",
        parked_at="2026-07-28",
    )
    return spec


# ------------------------------------------------------------------ store I/O


def test_record_and_load_round_trip(project):
    _park(project)
    data = operatoractions.load(project.project)
    assert list(data) == ["1-1-a"]
    assert data["1-1-a"]["actions"] == ACTIONS
    assert data["1-1-a"]["commit"] == "abcdef1234567890"
    assert data["1-1-a"]["run_id"] == "run-1"
    assert data["1-1-a"]["parked_at"] == "2026-07-28"


def test_re_parking_replaces_rather_than_accumulates(project):
    """A story owes whatever its LATEST park says it owes. Appending would leave
    a human acknowledging actions an earlier attempt declared and this one
    dropped."""
    _park(project)
    operatoractions.record_park(
        project.project,
        "1-1-a",
        actions=["only this one now"],
        spec_file="spec.md",
        commit="c2",
        run_id="run-2",
        parked_at="2026-07-29",
    )
    data = operatoractions.load(project.project)
    assert data["1-1-a"]["actions"] == ["only this one now"]
    assert data["1-1-a"]["commit"] == "c2"


def test_drop_removes_the_entry_and_reports_whether_it_was_there(project):
    _park(project)
    assert operatoractions.drop(project.project, "1-1-a") is True
    assert operatoractions.load(project.project) == {}
    # a second drop is a no-op, not an error — confirming a story the index never
    # knew about must not create a file just to delete from it
    assert operatoractions.drop(project.project, "1-1-a") is False
    assert operatoractions.drop(project.project, "never-existed") is False


@pytest.mark.parametrize(
    ("content", "why"),
    [
        ("{ not json", "malformed JSON"),
        ("[1, 2, 3]", "a JSON array where the store is an object"),
        ('"a string"', "a bare JSON scalar"),
    ],
)
def test_load_degrades_on_an_unusable_store(project, content, why):
    """The index is a convenience over committed truth, so an unreadable one
    reads as "no index" rather than blocking a human from confirming their own
    work."""
    store = operatoractions.store_path(project.project)
    store.parent.mkdir(parents=True, exist_ok=True)
    store.write_text(content)
    assert operatoractions.load(project.project) == {}, why


def test_load_of_an_absent_store_is_empty(project):
    assert operatoractions.load(project.project) == {}


def test_the_store_is_written_atomically_and_parsable(project):
    _park(project)
    raw = operatoractions.store_path(project.project).read_text()
    assert json.loads(raw)  # valid JSON, not a half-written temp
    assert not list(operatoractions.store_path(project.project).parent.glob("*.tmp"))


# --------------------------------------------------- keeping git out of it


def test_writing_the_index_leaves_the_worktree_clean(project):
    """The index is written from INSIDE a story's commit window. If git could see
    it, the epic-boundary auto-sweep would refuse to run (it requires a clean
    tree) and the next story's `git add -A` would sweep one story's bookkeeping
    into another story's commit.

    Ablation: delete the `_exclude_from_git` call in `_write_store` and this
    fails — the untracked index shows up in `git status` and the tree is dirty."""
    assert verify.worktree_clean(project.project)
    _index_only(project)
    assert operatoractions.store_path(project.project).is_file()  # it really is on disk
    assert verify.worktree_clean(project.project)


def test_the_exclude_pattern_lands_in_the_repository_exclude_file(project):
    _index_only(project)
    exclude = (project.project / ".git" / "info" / "exclude").read_text()
    assert ".bmad-loop/operator-actions.json" in exclude


def test_the_exclude_is_not_appended_twice(project):
    _index_only(project)
    _index_only(project, "1-2-b")
    exclude = (project.project / ".git" / "info" / "exclude").read_text()
    assert exclude.count(".bmad-loop/operator-actions.json") == 1


def test_the_store_write_survives_a_non_git_project(tmp_path):
    """Best-effort by construction: an index that could not be excluded is still
    a correct index, so a project with no git at all must not raise."""
    operatoractions.record_park(
        tmp_path,
        "1-1-a",
        actions=["do the thing"],
        spec_file="spec.md",
        commit="",
        run_id="r",
        parked_at="2026-07-28",
    )
    assert operatoractions.load(tmp_path)["1-1-a"]["actions"] == ["do the thing"]


# ------------------------------------------------- reading actions back


@pytest.mark.parametrize(
    ("stored", "expected", "why"),
    [
        (["a", "b"], ("a", "b"), "the ordinary list"),
        (["a", "a", "b"], ("a", "b"), "deduped, order preserved"),
        (["  a  ", ""], ("a",), "stripped, blanks dropped"),
        ("a string", (), "a bare string is not a list of actions"),
        ([{"action": "x", "check": "dig"}], (), "the deliberate v2 shape reads as nothing in v1"),
        ([None], (), "None is not an action"),
        (None, (), "absent"),
    ],
)
def test_actions_of_shares_the_frontmatter_reading(stored, expected, why):
    """The index is written from a spec and compared back against one, so a shape
    that collapses to () on the spec side must collapse to () here too."""
    assert operatoractions.actions_of({"actions": stored}) == expected, why


# --------------------------------------------- joining the index to truth


def test_resolve_joins_the_index_to_spec_and_board(project):
    _park(project)
    (story,) = operatoractions.resolve(project.project, project)
    assert story.story_key == "1-1-a"
    assert story.actions == tuple(ACTIONS)
    assert story.spec_status == "awaiting-operator"
    assert story.board_status == "awaiting-operator"
    assert story.commit == "abcdef1234567890"
    assert story.confirmable and story.drift() is None


def test_resolve_prefers_the_specs_actions_over_the_indexed_copy(project):
    """The spec is the committed truth. A spec edited after the park must show the
    human what they owe NOW, not what the run recorded then."""
    spec = _park(project)
    write_spec(spec, "awaiting-operator", "base", operator_actions=["the revised action"])
    (story,) = operatoractions.resolve(project.project, project)
    assert story.actions == ("the revised action",)


def test_resolve_falls_back_to_the_indexed_actions_when_the_spec_has_none(project):
    """A spec whose declaration was lost still shows what the run recorded — the
    fallback is what keeps a human from being told they owe nothing."""
    spec = _park(project)
    write_spec(spec, "awaiting-operator", "base")  # no operator_actions
    (story,) = operatoractions.resolve(project.project, project)
    assert story.actions == tuple(ACTIONS)


def test_resolve_is_sorted_by_story_key(project):
    _park(project, "1-2-b")
    _park(project, "1-1-a")
    assert [s.story_key for s in operatoractions.resolve(project.project, project)] == [
        "1-1-a",
        "1-2-b",
    ]


@pytest.mark.parametrize(
    ("spec_status", "board", "expect_drift"),
    [
        ("awaiting-operator", "awaiting-operator", None),
        ("done", "awaiting-operator", "its spec now says status: done"),
        ("awaiting-operator", "done", "the board now says done"),
        ("in-progress", "in-progress", "its spec now says status: in-progress"),
    ],
)
def test_drift_names_the_side_that_disagrees(project, spec_status, board, expect_drift):
    """ "The index says parked, the board says done" is a different message and a
    different remedy from "no such story" — so both sides are reported, and the
    spec is reported first because a spec that moved on explains everything."""
    _park(project, status=spec_status, board=board)
    (story,) = operatoractions.resolve(project.project, project)
    assert story.drift() == expect_drift
    assert story.confirmable is (expect_drift is None)


def test_drift_reports_a_missing_spec_before_a_missing_status(project):
    spec = _park(project)
    spec.unlink()
    (story,) = operatoractions.resolve(project.project, project)
    assert story.spec_status is None
    assert "missing or unreadable" in (story.drift() or "")
    assert not story.confirmable


def test_drift_reports_an_entry_with_no_spec_path(project):
    install_bmad_config(project)
    write_sprint(project, {"1-1-a": "awaiting-operator"})
    operatoractions.record_park(
        project.project,
        "1-1-a",
        actions=ACTIONS,
        spec_file="",
        commit="c",
        run_id="r",
        parked_at="2026-07-28",
    )
    (story,) = operatoractions.resolve(project.project, project)
    assert story.spec_path is None
    assert story.drift() == "the index records no spec file for it"


def test_drift_reports_a_story_absent_from_the_board(project):
    _park(project)
    write_sprint(project, {"epic-1": "in-progress"})  # story key gone
    (story,) = operatoractions.resolve(project.project, project)
    assert story.board_status is None
    assert story.drift() == "it is not on the sprint board"


def test_a_park_declaring_nothing_readable_is_not_confirmable(project):
    """A park is DEFINED by owing at least one action. An entry with none is not
    something a human can acknowledge, so it must not read as confirmable."""
    _park(project, actions=[])
    (story,) = operatoractions.resolve(project.project, project)
    assert story.actions == ()
    assert story.drift() == "it declares no readable actions"
    assert not story.confirmable


def test_resolve_degrades_on_an_unreadable_board(project):
    """`resolve` backs a listing and a validate warning; neither may be the thing
    that crashes on a board someone broke."""
    _park(project)
    project.sprint_status.write_text("{{ not yaml")
    (story,) = operatoractions.resolve(project.project, project)
    assert story.board_status is None
    assert not story.confirmable


def test_resolve_tolerates_a_non_dict_entry(project):
    """A hand-mangled index must degrade, not raise, on the way to the warning
    that reports it."""
    install_bmad_config(project)
    write_sprint(project, {"1-1-a": "awaiting-operator"})
    store = operatoractions.store_path(project.project)
    store.parent.mkdir(parents=True, exist_ok=True)
    store.write_text(json.dumps({"1-1-a": "not a record"}))
    (story,) = operatoractions.resolve(project.project, project)
    assert story.actions == () and story.spec_path is None


def test_verify_exports_the_same_token(project):
    """One spelling of the token across the park and its exit."""
    assert operatoractions.AWAITING_OPERATOR == verify.AWAITING_OPERATOR == "awaiting-operator"
