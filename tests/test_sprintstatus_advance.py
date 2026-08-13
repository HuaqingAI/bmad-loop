"""Tests for the orchestrator-owned sprint-status writer (generic-skill path)."""

from pathlib import Path

from bmad_loop import sprintstatus

SPRINT = """\
# Sprint status — do not hand-edit casually
generated: 01-06-2026 10:00
last_updated: 01-06-2026 10:00

# STATUS DEFINITIONS
#   backlog -> ready-for-dev -> in-progress -> review -> done
development_status:
  epic-3: backlog
  3-1-login: done
  3-2-digest-delivery: backlog  # the next story
  epic-4: in-progress
  4-1-thing: review

# WORKFLOW NOTES
# keep these comments
"""


def _write(tmp_path: Path) -> Path:
    p = tmp_path / "sprint-status.yaml"
    p.write_text(SPRINT, encoding="utf-8")
    return p


def test_advance_to_in_progress_lifts_backlog_epic(tmp_path):
    p = _write(tmp_path)
    out = sprintstatus.advance(p, "3-2-digest-delivery", "in-progress")
    assert out == "in-progress"
    assert sprintstatus.story_status(p, "3-2-digest-delivery") == "in-progress"
    assert sprintstatus.load(p).epics[3] == "in-progress"  # epic lifted


def test_advance_split_story_lifts_backlog_epic(tmp_path):
    # a split-story key (issue #144) must advance and lift its epic like any other
    text = (
        "last_updated: 01-06-2026 10:00\n"
        "development_status:\n"
        "  epic-2: backlog\n"
        "  2-6a-build-structure: backlog\n"
        "  2-6b-extend-structure: backlog\n"
    )
    p = tmp_path / "sprint-status.yaml"
    p.write_text(text, encoding="utf-8")
    out = sprintstatus.advance(p, "2-6a-build-structure", "in-progress")
    assert out == "in-progress"
    assert sprintstatus.story_status(p, "2-6a-build-structure") == "in-progress"
    assert sprintstatus.load(p).epics[2] == "in-progress"  # epic lifted
    assert sprintstatus.story_status(p, "2-6b-extend-structure") == "backlog"  # sibling untouched


def test_advance_preserves_comments_and_structure(tmp_path):
    p = _write(tmp_path)
    sprintstatus.advance(p, "3-2-digest-delivery", "in-progress")
    text = p.read_text()
    assert "# STATUS DEFINITIONS" in text
    assert "# WORKFLOW NOTES" in text
    assert "# the next story" in text  # inline comment survived
    assert "# keep these comments" in text


def test_advance_never_regresses(tmp_path):
    p = _write(tmp_path)
    out = sprintstatus.advance(p, "4-1-thing", "in-progress")  # currently review
    assert out == "review"
    assert sprintstatus.story_status(p, "4-1-thing") == "review"


def test_advance_confirms_a_parked_story_forward_to_done(tmp_path):
    """The exit move `bmad-loop confirm` will need: because `awaiting-operator`
    sits below `done` in STATUS_ORDER, completing a parked story is an ordinary
    forward advance through the sole writer — no invariant exception required."""
    p = _write(tmp_path)
    sprintstatus.advance(p, "3-2-digest-delivery", "awaiting-operator")
    assert sprintstatus.story_status(p, "3-2-digest-delivery") == "awaiting-operator"

    out = sprintstatus.advance(p, "3-2-digest-delivery", "done")

    assert out == "done"
    assert sprintstatus.story_status(p, "3-2-digest-delivery") == "done"


def test_advance_never_regresses_done_into_awaiting_operator(tmp_path):
    """The other half of the ordering: once a story is `done`, nothing walks the
    board back to `awaiting-operator`. This is a real hardening, not a restatement
    — before the token joined STATUS_ORDER it was unordered, so the never-regress
    guard's `target in STATUS_ORDER` arm short-circuited and this write went
    through. (Demoting a done story is Phase 4's `operator.on_review_demotion`
    question, and it will need its own deliberate, allowlisted writer.)"""
    p = _write(tmp_path)
    before = p.read_text()

    out = sprintstatus.advance(p, "3-1-login", "awaiting-operator")  # already done

    assert out == "done"
    assert p.read_text() == before


def test_advance_returns_current_when_line_not_rewritable(tmp_path):
    """A quoted story key parses via YAML (story_status finds it) but the line-edit
    writer can't rewrite it. advance() must report the unchanged status, not falsely
    claim it reached target, and must leave the file untouched."""
    text = (
        "last_updated: 01-06-2026 10:00\n"
        "development_status:\n"
        "  epic-5: in-progress\n"
        "  '5-1-quoted': ready-for-dev\n"
    )
    p = tmp_path / "sprint-status.yaml"
    p.write_text(text, encoding="utf-8")
    before = p.read_text()

    out = sprintstatus.advance(p, "5-1-quoted", "in-progress", now="02-06-2026 09:00")

    assert out == "ready-for-dev"  # current status, not the requested target
    assert p.read_text() == before  # nothing rewritten — not even last_updated


def test_advance_idempotent_done(tmp_path):
    p = _write(tmp_path)
    out = sprintstatus.advance(p, "3-1-login", "done")  # already done
    assert out == "done"
    assert sprintstatus.story_status(p, "3-1-login") == "done"


def test_advance_to_review(tmp_path):
    p = _write(tmp_path)
    out = sprintstatus.advance(p, "3-2-digest-delivery", "review")
    assert out == "review"
    assert sprintstatus.story_status(p, "3-2-digest-delivery") == "review"
    # epic NOT lifted for non-in-progress targets
    assert sprintstatus.load(p).epics[3] == "backlog"


def test_advance_done_does_not_touch_epic(tmp_path):
    p = _write(tmp_path)
    sprintstatus.advance(p, "3-2-digest-delivery", "done")
    assert sprintstatus.load(p).epics[3] == "backlog"


def test_advance_epic_not_lifted_when_not_backlog(tmp_path):
    p = _write(tmp_path)
    sprintstatus.advance(p, "4-1-thing", "in-progress")  # regresses -> no-op anyway
    # epic-4 was in-progress; ensure unchanged
    assert sprintstatus.load(p).epics[4] == "in-progress"


def test_advance_refreshes_last_updated(tmp_path):
    p = _write(tmp_path)
    sprintstatus.advance(p, "3-2-digest-delivery", "in-progress", now="22-06-2026 14:30")
    text = p.read_text()
    assert "last_updated: 22-06-2026 14:30" in text
    assert "generated: 01-06-2026 10:00" in text  # generated untouched


def test_advance_story_not_found(tmp_path):
    p = _write(tmp_path)
    assert sprintstatus.advance(p, "9-9-ghost", "in-progress") is None


def test_advance_missing_file(tmp_path):
    assert sprintstatus.advance(tmp_path / "ghost.yaml", "3-2-x", "in-progress") is None


# ================================================= the value / comment split
#
# #366. `_set_mapping_value` decides where a line's scalar ends and a trailing
# inline comment begins, and NOTHING checks that guess afterwards: `advance` has
# no oracle at all, and the one a caller might reach for cannot see this class of
# error anyway — `yaml.safe_load` strips comments before it could compare, so a
# line rewritten with a comment invented out of the tail of a quoted value
# re-parses as a perfectly clean `3-2-x: done`. (Proven by ablation on the sibling
# defect, PR #365, whose three verification gates all passed the fabricated
# comment.) The pattern is therefore the gate here, and these tests hold it.
#
# Called directly rather than through `advance` wherever the shape under test is
# a REFUSAL: `advance` answers a refused line and a story already at target with
# the same unchanged status, so only the writer's own return separates them.
# Every assertion is on the FULL resulting text — a substring or a re-parse is
# blind to exactly the fabrication these are here to catch.


def test_a_hash_inside_a_quoted_value_never_becomes_a_comment(tmp_path):
    """The case #366 is about, end to end through the sole writer. `"a # b"`
    carries no comment — the `#` is scalar text — so a split guessed from the last
    ` #` on the line writes `3-2-x: done # b"`, promoting the tail of the value
    into a comment the board never had and truncating the value it came from. A
    quote-led remainder is taken whole instead, which drops nothing that was
    ever a comment."""
    p = tmp_path / "sprint-status.yaml"
    board = (
        'last_updated: 01-06-2026 10:00\ndevelopment_status:\n  3-2-x: "a # b"\n  3-3-y: backlog\n'
    )
    p.write_text(board, encoding="utf-8")

    assert sprintstatus.advance(p, "3-2-x", "done") == "done"

    assert p.read_bytes().decode() == board.replace('"a # b"', "done")


def test_a_quoted_value_is_replaced_whole_with_no_comment_carried(tmp_path):
    """The writer's own half of the case above: the write SUCCEEDS (a quoted
    hand-edit is still a value the orchestrator owns and replaces), and what it
    leaves behind is the bare target and nothing else."""
    lines = ['  3-2-x: "a # b"  # real comment\n']

    assert sprintstatus._set_mapping_value(lines, "3-2-x", "done") is True

    # the trailing comment goes too: nothing here can tell a closing quote from
    # a quote inside the scalar, so a comment after one is dropped, not guessed.
    assert "".join(lines) == "  3-2-x: done\n"


def test_a_value_with_internal_spaces_is_matched_whole(tmp_path):
    """Why this board cannot borrow `frontmatter._VALUE_COMMENT_RE`'s
    conservative token class: `last_updated` is a bare scalar WITH SPACES, and a
    token gate would refuse it — the timestamp refresh would silently stop
    happening (`test_advance_refreshes_last_updated` is the advance-level half)."""
    lines = ["last_updated: 01-06-2026 10:00\n"]

    assert sprintstatus._set_mapping_value(lines, "last_updated", "22-06-2026 14:30") is True

    assert "".join(lines) == "last_updated: 22-06-2026 14:30\n"


def test_an_inline_comment_carries_with_its_authored_separator(tmp_path):
    """The preservation the split exists to make possible, unchanged by #366: an
    unquoted value cedes the FIRST whitespace-preceded `#`, and the whitespace
    that separates it comes through as authored (two spaces here), so a
    hand-aligned comment column is not reflowed by a status flip."""
    lines = ["  3-2-digest-delivery: backlog  # the next story\n"]

    assert sprintstatus._set_mapping_value(lines, "3-2-digest-delivery", "in-progress") is True

    assert "".join(lines) == "  3-2-digest-delivery: in-progress  # the next story\n"


def test_a_hash_glued_to_the_value_stays_part_of_the_value(tmp_path):
    """YAML needs whitespace before a `#` for it to open a comment, so
    `backlog#x` is the single scalar `backlog#x`. The value is replaced whole and
    `#x` is not carried forward as a comment the board never had."""
    lines = ["  3-2-x: backlog#x\n"]

    assert sprintstatus._set_mapping_value(lines, "3-2-x", "done") is True

    assert "".join(lines) == "  3-2-x: done\n"


def test_a_line_with_trailing_whitespace_and_no_comment_is_refused(tmp_path):
    """Characterization, not a requirement — but pinned so the split cannot
    change it by accident. Both arms end at a non-space character, so a value
    with trailing whitespace and no comment is a remainder neither can account
    for, and the line is left exactly as authored rather than rewritten a few
    invisible characters shorter. `advance` then reports the unchanged status
    (`test_advance_returns_current_when_line_not_rewritable` is that half)."""
    trailing = "  3-2-x: backlog  \n"
    quoted_trailing = "  3-2-x: 'backlog' \n"

    for line in (trailing, quoted_trailing):
        lines = [line]
        assert sprintstatus._set_mapping_value(lines, "3-2-x", "done") is False
        assert "".join(lines) == line
