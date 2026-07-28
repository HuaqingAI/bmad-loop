import pytest
from conftest import write_sprint

from bmad_loop import sprintstatus


def test_load_classifies_keys(project):
    write_sprint(
        project,
        {
            "epic-1": "in-progress",
            "1-1-user-auth": "done",
            "1-2-account-mgmt": "ready-for-dev",
            "epic-1-retrospective": "optional",
            "epic-2": "backlog",
            "2-1-personality": "backlog",
            "epic-2-retrospective": "optional",
            "epic-1-retro-item-1-test-design": "done",
            "epic-2-retro-item-3-fts5-research": "backlog",
            "weird-key": "huh",
        },
    )
    ss = sprintstatus.load(project.sprint_status)
    assert ss.epics == {1: "in-progress", 2: "backlog"}
    assert [s.key for s in ss.stories] == [
        "1-1-user-auth",
        "1-2-account-mgmt",
        "2-1-personality",
    ]
    assert ss.stories[1].epic == 1 and ss.stories[1].num == 2
    assert ss.retros == {1: "optional", 2: "optional"}
    assert ss.unknown_keys == ("weird-key",)


def test_load_split_story_keys(project):
    # BMAD splits an oversized story into 2-6a / 2-6b (issue #144); both halves
    # must parse as stories — not fall into unknown_keys — and keep file order.
    write_sprint(
        project,
        {
            "2-5-intact": "done",
            "2-6a-build-structure": "backlog",
            "2-6b-extend-structure": "backlog",
            "2-7-later": "backlog",
            "2-6ab-not-a-split": "backlog",  # multi-letter: not the convention
            "2-6A-not-lower": "backlog",  # uppercase: not the convention
        },
    )
    ss = sprintstatus.load(project.sprint_status)
    assert [s.key for s in ss.stories] == [
        "2-5-intact",
        "2-6a-build-structure",
        "2-6b-extend-structure",
        "2-7-later",
    ]
    a, b = ss.stories[1], ss.stories[2]
    assert (a.epic, a.num, a.suffix, a.slug) == (2, 6, "a", "build-structure")
    assert (b.epic, b.num, b.suffix, b.slug) == (2, 6, "b", "extend-structure")
    assert ss.stories[0].suffix == ""  # whole stories carry no suffix
    assert ss.unknown_keys == ("2-6ab-not-a-split", "2-6A-not-lower")
    # the split halves are picked in file order, before later stories
    assert sprintstatus.next_actionable(ss).key == "2-6a-build-structure"


def test_load_classifies_retro_items(project):
    write_sprint(
        project,
        {
            "epic-1-retrospective": "done",
            "epic-1-retro-item-1-test-design-in-stories": "done",
            "epic-5-retro-item-2-singleflight-inflight-guard-helper": "backlog",
        },
    )
    ss = sprintstatus.load(project.sprint_status)
    # retro action items are recognized, not dumped into unknown_keys
    assert ss.unknown_keys == ()
    assert ss.retros == {1: "done"}  # plain retrospective key is unaffected
    assert [(r.key, r.epic, r.num, r.slug, r.status) for r in ss.retro_items] == [
        ("epic-1-retro-item-1-test-design-in-stories", 1, 1, "test-design-in-stories", "done"),
        (
            "epic-5-retro-item-2-singleflight-inflight-guard-helper",
            5,
            2,
            "singleflight-inflight-guard-helper",
            "backlog",
        ),
    ]


def test_retro_items_do_not_become_actionable_stories(project):
    # recognition only: retro items must not leak into story selection
    write_sprint(project, {"epic-3-retro-item-1-do-a-thing": "backlog"})
    ss = sprintstatus.load(project.sprint_status)
    assert ss.stories == ()
    assert sprintstatus.next_actionable(ss) is None


def test_legacy_drafted_maps_to_ready(project):
    write_sprint(project, {"1-1-x": "drafted"})
    ss = sprintstatus.load(project.sprint_status)
    assert ss.stories[0].status == "ready-for-dev"


def test_next_actionable_order_and_skip(project):
    write_sprint(
        project,
        {"1-1-a": "done", "1-2-b": "ready-for-dev", "1-3-c": "backlog"},
    )
    ss = sprintstatus.load(project.sprint_status)
    assert sprintstatus.next_actionable(ss).key == "1-2-b"
    assert sprintstatus.next_actionable(ss, skip={"1-2-b"}).key == "1-3-c"
    assert sprintstatus.next_actionable(ss, skip={"1-2-b", "1-3-c"}) is None


def test_awaiting_operator_is_ordered_just_below_done():
    """The token's whole contract is its POSITION: last stop before `done`, so
    confirming a parked story is a forward advance and nothing can regress `done`
    back into it. Asserting the index relationships rather than the literal tuple
    keeps this from breaking every time a status is added elsewhere."""
    order = sprintstatus.STATUS_ORDER
    assert order.index("awaiting-operator") == order.index("done") - 1
    assert order.index("review") < order.index("awaiting-operator")


def test_parked_story_is_never_picked_as_next_actionable(project):
    """A parked story's agent-doable work is already committed — picking it up
    again would redo finished work while the human's external actions stay
    outstanding. The board must walk straight past it to the next real story.

    Ablation (repo rule, inverse form — the gate here is an ABSENCE, so the
    ablation ADDS rather than deletes): put "awaiting-operator" into
    ACTIONABLE_STATUSES and this test must fail on the first assert. It does —
    which is what proves the assertions gate on actionability rather than on
    file order (1-1-a is first in the file, so a test that merely returned the
    first pickable story would pass either way).
    """
    write_sprint(
        project,
        {"1-1-a": "awaiting-operator", "1-2-b": "ready-for-dev"},
    )
    ss = sprintstatus.load(project.sprint_status)
    assert sprintstatus.next_actionable(ss).key == "1-2-b"
    # and with the only other story taken, there is nothing left to run — the
    # parked story is not a fallback either
    assert sprintstatus.next_actionable(ss, skip={"1-2-b"}) is None


def test_parked_story_cannot_be_targeted_by_a_selector(project):
    """`--story` naming a parked story is a user error with a specific message,
    not a silent re-drive: select_actionable filters on the same set."""
    write_sprint(project, {"1-1-a": "awaiting-operator"})
    ss = sprintstatus.load(project.sprint_status)
    with pytest.raises(
        sprintstatus.SprintStatusError,
        match=r"story 1-1 matched 1-1-a but its status is 'awaiting-operator'",
    ):
        sprintstatus.select_actionable(ss, None, "1-1")


def test_next_actionable_epic_filter(project):
    # document order (epic 5 before epic 9), not numeric; the epic filter returns
    # epic 9's first actionable story even though 5-1 is earlier in the file.
    write_sprint(
        project,
        {"5-1-e5": "backlog", "9-0-x": "ready-for-dev", "9-1-y": "backlog"},
    )
    ss = sprintstatus.load(project.sprint_status)
    assert sprintstatus.next_actionable(ss).key == "5-1-e5"  # unfiltered = file order
    assert sprintstatus.next_actionable(ss, epic=9).key == "9-0-x"
    assert sprintstatus.next_actionable(ss, skip={"9-0-x"}, epic=9).key == "9-1-y"
    assert sprintstatus.next_actionable(ss, skip={"9-0-x", "9-1-y"}, epic=9) is None


def test_story_status_reread(project):
    write_sprint(project, {"1-1-a": "in-progress"})
    assert sprintstatus.story_status(project.sprint_status, "1-1-a") == "in-progress"
    assert sprintstatus.story_status(project.sprint_status, "9-9-z") is None


def test_parse_selector_forms():
    # full key — exact match intent
    sel = sprintstatus.parse_selector(None, "3-1-user-auth")
    assert (sel.epic, sel.num, sel.key, sel.slug) == (3, 1, "3-1-user-auth", None)
    # short refs: hyphen and dot are equivalent
    for ref in ("3-1", "3.1"):
        sel = sprintstatus.parse_selector(None, ref)
        assert (sel.epic, sel.num, sel.key, sel.slug) == (3, 1, None, None)
    # bare number resolves against --epic
    sel = sprintstatus.parse_selector(3, "1")
    assert (sel.epic, sel.num, sel.slug) == (3, 1, None)
    # slug fragment
    sel = sprintstatus.parse_selector(None, "user-auth")
    assert (sel.epic, sel.num, sel.slug) == (None, None, "user-auth")
    # epic only — not targeted
    sel = sprintstatus.parse_selector(3, None)
    assert sel.epic == 3 and not sel.is_targeted


def test_parse_selector_split_suffix():
    # every numeric form carries the split suffix through to the selector
    sel = sprintstatus.parse_selector(None, "2-6a-build-structure")
    assert (sel.epic, sel.num, sel.suffix, sel.key) == (2, 6, "a", "2-6a-build-structure")
    for ref in ("2-6a", "2.6a"):
        sel = sprintstatus.parse_selector(None, ref)
        assert (sel.epic, sel.num, sel.suffix, sel.key, sel.slug) == (2, 6, "a", None, None)
    sel = sprintstatus.parse_selector(2, "6a")
    assert (sel.epic, sel.num, sel.suffix) == (2, 6, "a")
    # suffix-less forms leave suffix None — the whole-family wildcard
    for epic, ref in [(None, "2-6"), (None, "2.6"), (2, "6"), (None, "2-6-whole-slug")]:
        sel = sprintstatus.parse_selector(epic, ref)
        assert sel.suffix is None, ref


def test_parse_selector_bare_number_needs_epic():
    with pytest.raises(sprintstatus.SprintStatusError, match="ambiguous story '1'"):
        sprintstatus.parse_selector(None, "1")


def test_parse_selector_epic_conflict():
    with pytest.raises(sprintstatus.SprintStatusError, match="conflicts"):
        sprintstatus.parse_selector(2, "3-1")
    with pytest.raises(sprintstatus.SprintStatusError, match="conflicts"):
        sprintstatus.parse_selector(2, "3-1-user-auth")


def test_select_actionable_short_ref_and_epic_story(project):
    write_sprint(
        project,
        {"3-1-user-auth": "ready-for-dev", "3-2-foo": "backlog", "4-1-bar": "backlog"},
    )
    ss = sprintstatus.load(project.sprint_status)
    for epic, story in [(None, "3-1"), (None, "3.1"), (3, "1"), (None, "user-auth")]:
        got = sprintstatus.select_actionable(ss, epic, story)
        assert [s.key for s in got] == ["3-1-user-auth"]
    # epic only selects every actionable story in the epic
    assert [s.key for s in sprintstatus.select_actionable(ss, 3, None)] == [
        "3-1-user-auth",
        "3-2-foo",
    ]


def test_select_actionable_split_suffix(project):
    write_sprint(
        project,
        {
            "2-6a-build-structure": "backlog",
            "2-6b-extend-structure": "backlog",
            "2-7-later": "backlog",
        },
    )
    ss = sprintstatus.load(project.sprint_status)
    # a suffixed ref selects exactly its half — never the sibling
    for epic, story in [(None, "2-6a"), (None, "2.6a"), (2, "6a")]:
        got = sprintstatus.select_actionable(ss, epic, story)
        assert [s.key for s in got] == ["2-6a-build-structure"]
    assert [s.key for s in sprintstatus.select_actionable(ss, None, "2.6b")] == [
        "2-6b-extend-structure"
    ]
    # the plain short ref selects the whole split family, in file order
    assert [s.key for s in sprintstatus.select_actionable(ss, None, "2-6")] == [
        "2-6a-build-structure",
        "2-6b-extend-structure",
    ]
    # a suffix that doesn't exist matches nothing
    with pytest.raises(sprintstatus.SprintStatusError, match="no story matches '2-6c'"):
        sprintstatus.select_actionable(ss, None, "2-6c")


def test_select_actionable_split_suffix_not_actionable(project):
    write_sprint(
        project,
        {"2-6a-build-structure": "done", "2-6b-extend-structure": "backlog"},
    )
    ss = sprintstatus.load(project.sprint_status)
    with pytest.raises(
        sprintstatus.SprintStatusError,
        match=r"story 2-6a matched 2-6a-build-structure but its status is 'done'",
    ):
        sprintstatus.select_actionable(ss, None, "2-6a")
    # the family ref still finds the remaining actionable half
    got = sprintstatus.select_actionable(ss, None, "2-6")
    assert [s.key for s in got] == ["2-6b-extend-structure"]


def test_select_actionable_targeted_not_actionable(project):
    write_sprint(project, {"3-1-user-auth": "ready-for-dev", "3-2-foo": "done"})
    ss = sprintstatus.load(project.sprint_status)
    with pytest.raises(
        sprintstatus.SprintStatusError,
        match=r"story 3-2 matched 3-2-foo but its status is 'done'",
    ):
        sprintstatus.select_actionable(ss, None, "3-2")


def test_select_actionable_ambiguous_slug(project):
    write_sprint(project, {"3-1-user-auth": "backlog", "4-2-admin-auth": "backlog"})
    ss = sprintstatus.load(project.sprint_status)
    with pytest.raises(sprintstatus.SprintStatusError, match="ambiguous"):
        sprintstatus.select_actionable(ss, None, "auth")


def test_select_actionable_no_match(project):
    write_sprint(project, {"3-1-user-auth": "backlog"})
    ss = sprintstatus.load(project.sprint_status)
    with pytest.raises(sprintstatus.SprintStatusError, match="no story matches"):
        sprintstatus.select_actionable(ss, None, "9-9")


def test_missing_file_raises(project):
    with pytest.raises(sprintstatus.SprintStatusError, match="not found"):
        sprintstatus.load(project.sprint_status)


def test_malformed_yaml_raises(project):
    project.sprint_status.write_text("development_status: [unclosed")
    with pytest.raises(sprintstatus.SprintStatusError, match="not valid YAML"):
        sprintstatus.load(project.sprint_status)


def test_missing_map_raises(project):
    project.sprint_status.write_text("project: x\n")
    with pytest.raises(sprintstatus.SprintStatusError, match="development_status"):
        sprintstatus.load(project.sprint_status)
