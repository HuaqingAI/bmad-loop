"""Ledger parsing and editing: deferredwork.py."""

from pathlib import Path

import pytest

from bmad_loop import deferredwork
from bmad_loop.deferredwork import (
    _ISO_DATE_RE,
    LINE_BREAK_RE,
    SEVERITY_ALIASES,
    append_decision,
    append_entry,
    classify,
    field_line_present,
    field_severity,
    has_legacy,
    mark_done,
    mark_done_many,
    mark_done_many_reopenable,
    mark_open,
    next_seq,
    open_ids,
    parse_declaration,
    parse_ledger,
    parse_legacy,
)

OPERATION_ID = "run-20260803T120000/dw-fix"

LEDGER = """\
# Deferred Work

### DW-1: Harden unicode handling

origin: quick-dev split of spec-3-2-digest.md, 2026-06-01
location: src/strings.py:40
reason: out of scope for the digest story.
status: open

### DW-2: Old closed item

origin: code review of spec-1-1.md, 2026-05-20
location: src/foo.py:10
reason: pre-existing.
status: done 2026-05-25

### DW-3: Needs human decision

origin: code review of spec-2-2.md, 2026-06-05
location: src/retry.py:12
reason: auto-mode: needs human decision
status: open
seen-again: 2026-06-08 (code review of spec-2-3.md)
"""


def write_ledger(tmp_path: Path, text: str = LEDGER) -> Path:
    path = tmp_path / "deferred-work.md"
    path.write_text(text, encoding="utf-8")
    return path


def close_reopenable(
    path: Path,
    dw_id: str,
    note: str,
    date: str = "2026-06-11",
    operation_id: str = OPERATION_ID,
) -> None:
    assert mark_done_many_reopenable(path, [dw_id], date, note, operation_id) == [dw_id]


def test_parse_ledger_entries():
    entries = parse_ledger(LEDGER)
    assert [e.id for e in entries] == ["DW-1", "DW-2", "DW-3"]
    assert entries[0].title == "Harden unicode handling"
    assert entries[0].open
    assert not entries[1].open
    assert entries[1].status == "done 2026-05-25"
    assert entries[2].open  # seen-again line does not affect status


def test_open_ids():
    assert open_ids(LEDGER) == {"DW-1", "DW-3"}


def test_parse_tolerates_freeform_sections():
    text = (
        "## Deferred from: code review of story 0.3 (2026-06-08)\n\n"
        "- W1-b — some freeform item with no DW format\n\n" + LEDGER
    )
    assert open_ids(text) == {"DW-1", "DW-3"}


def test_entry_ends_at_next_section_heading():
    text = LEDGER + "\n## Notes\n\nstatus: open\n"
    entries = parse_ledger(text)
    # the stray status line under "## Notes" must not leak into DW-3
    assert entries[-1].id == "DW-3"
    assert "## Notes" not in entries[-1].body


def test_entry_without_status_is_not_open():
    text = "### DW-7: Malformed entry\n\norigin: somewhere\n"
    entries = parse_ledger(text)
    assert entries[0].status == ""
    assert not entries[0].open
    assert open_ids(text) == set()


def test_mark_done_touches_only_target(tmp_path):
    path = write_ledger(tmp_path)
    assert mark_done(path, "DW-1", "2026-06-11", "guards added in src/strings.py")
    text = path.read_text(encoding="utf-8")
    entries = {e.id: e for e in parse_ledger(text)}
    assert entries["DW-1"].status == "done 2026-06-11"
    assert "resolution: guards added in src/strings.py" in entries["DW-1"].body
    assert entries["DW-3"].open
    assert "resolution:" not in entries["DW-3"].body
    assert entries["DW-2"].status == "done 2026-05-25"


def test_mark_done_idempotent(tmp_path):
    path = write_ledger(tmp_path)
    assert mark_done(path, "DW-1", "2026-06-11", "fixed")
    snapshot = path.read_text(encoding="utf-8")
    assert not mark_done(path, "DW-1", "2026-06-12", "fixed again")
    assert path.read_text(encoding="utf-8") == snapshot


def test_mark_done_missing_entry(tmp_path):
    path = write_ledger(tmp_path)
    snapshot = path.read_text(encoding="utf-8")
    assert not mark_done(path, "DW-99", "2026-06-11", "n/a")
    assert path.read_text(encoding="utf-8") == snapshot


def test_mark_open_round_trips_one_reopenable_close_character_for_character(tmp_path):
    path = write_ledger(tmp_path)
    before = path.read_text(encoding="utf-8")
    note = "resolved by sweep bundle dw-fix"
    close_reopenable(path, "DW-1", note)
    assert mark_open(path, "DW-1", note, OPERATION_ID)
    assert path.read_text(encoding="utf-8") == before


@pytest.mark.parametrize(
    "status_line",
    ["status:   open", "status: open # keep this annotation", "status:\topen   # annotated"],
)
def test_mark_open_restores_the_original_parser_accepted_status_line(tmp_path, status_line):
    text = f"### DW-1: formatted\n\norigin: session\n{status_line}\n"
    path = write_ledger(tmp_path, text)

    close_reopenable(path, "DW-1", "by dw-a")
    assert mark_open(path, "DW-1", "by dw-a", OPERATION_ID)

    assert path.read_text(encoding="utf-8") == text


def test_reopenable_close_persists_adjacent_resolution_and_undo_marker(tmp_path):
    path = write_ledger(tmp_path)
    close_reopenable(path, "DW-1", "by dw-a")

    lines = parse_ledger(path.read_text(encoding="utf-8"))[0].body.splitlines()
    status_at = lines.index("status: done 2026-06-11")
    assert lines[status_at + 1] == "resolution: by dw-a"
    assert lines[status_at + 2].startswith("resolution-undo: ")
    assert len(lines[status_at + 2].split()) == 4


def test_standard_close_retains_its_existing_ledger_shape(tmp_path):
    path = write_ledger(tmp_path, "### DW-1: standard\n\nstatus: open\n")
    assert mark_done_many(path, ["DW-1"], "2026-06-11", "by dw-a") == ["DW-1"]

    assert path.read_text(encoding="utf-8") == (
        "### DW-1: standard\n\nstatus: done 2026-06-11\nresolution: by dw-a\n"
    )


def test_reopenable_close_refuses_an_original_status_with_a_splitlines_break(tmp_path):
    text = "### DW-1: unsafe\n\nstatus: open\u2028injected: value\n"
    path = write_ledger(tmp_path, text)

    assert mark_done_many_reopenable(path, ["DW-1"], "2026-06-11", "by dw-a", OPERATION_ID) == []
    assert path.read_text(encoding="utf-8") == text


def test_mark_open_touches_only_the_target_entry(tmp_path):
    path = write_ledger(tmp_path)
    close_reopenable(path, "DW-1", "by dw-a")
    close_reopenable(path, "DW-3", "by dw-a")
    assert mark_open(path, "DW-1", "by dw-a", OPERATION_ID)

    entries = {e.id: e for e in parse_ledger(path.read_text(encoding="utf-8"))}
    assert entries["DW-1"].open
    assert "resolution:" not in entries["DW-1"].body
    assert entries["DW-2"].status == "done 2026-05-25"
    assert entries["DW-3"].status == "done 2026-06-11"


def test_mark_open_refuses_a_missing_ledger(tmp_path):
    assert not mark_open(tmp_path / "nope.md", "DW-1", "by dw-a", OPERATION_ID)


def test_mark_open_refuses_a_missing_entry(tmp_path):
    path = write_ledger(tmp_path)
    snapshot = path.read_text(encoding="utf-8")
    assert not mark_open(path, "DW-99", "by dw-a", OPERATION_ID)
    assert path.read_text(encoding="utf-8") == snapshot


def test_mark_open_refuses_an_entry_that_is_still_open(tmp_path):
    """The open guard is independent of note matching: an LLM-written open entry
    can carry a matching resolution line, but the orchestrator must not delete it."""
    path = write_ledger(
        tmp_path,
        "### DW-1: partially handled\n\nstatus: open\nresolution: by dw-a\n",
    )
    snapshot = path.read_text(encoding="utf-8")
    assert not mark_open(path, "DW-1", "by dw-a", OPERATION_ID)
    assert path.read_text(encoding="utf-8") == snapshot


def test_mark_open_refuses_a_status_less_entry(tmp_path):
    """parse_ledger tolerates this shape. Without the explicit guard, a future
    call from _defer raises AttributeError and crashes instead of deferring."""
    path = write_ledger(
        tmp_path,
        "### DW-1: malformed but tolerated\n\norigin: session\nresolution: by dw-a\n",
    )
    snapshot = path.read_text(encoding="utf-8")
    assert parse_ledger(snapshot)[0].status == ""
    assert not mark_open(path, "DW-1", "by dw-a", OPERATION_ID)
    assert path.read_text(encoding="utf-8") == snapshot


@pytest.mark.parametrize("status", ["malformed", "done someday", "done 2026-02-30"])
def test_mark_open_refuses_a_noncanonical_status(tmp_path, status):
    path = write_ledger(
        tmp_path,
        f"### DW-1: malformed but tolerated\n\nstatus: {status}\nresolution: by dw-a\n",
    )
    snapshot = path.read_text(encoding="utf-8")
    assert not mark_open(path, "DW-1", "by dw-a", OPERATION_ID)
    assert path.read_text(encoding="utf-8") == snapshot


def test_mark_open_refuses_a_closed_entry_without_a_resolution(tmp_path):
    path = write_ledger(tmp_path)
    snapshot = path.read_text(encoding="utf-8")
    assert not mark_open(path, "DW-2", "", OPERATION_ID)
    assert path.read_text(encoding="utf-8") == snapshot


def test_mark_open_refuses_a_legacy_close_with_the_same_note_but_no_marker(tmp_path):
    path = write_ledger(tmp_path)
    assert mark_done(path, "DW-1", "2026-06-11", "by dw-a")
    snapshot = path.read_text(encoding="utf-8")
    assert not mark_open(path, "DW-1", "by dw-a", OPERATION_ID)
    assert path.read_text(encoding="utf-8") == snapshot


def test_mark_open_reopens_only_closes_from_this_operation(tmp_path):
    """A reused story note does not make an earlier run's close ours to undo."""
    path = write_ledger(tmp_path)
    note = "resolved by story 1-1"
    close_reopenable(path, "DW-1", note, operation_id="earlier-run/dw-fix")

    marked = mark_done_many_reopenable(path, ["DW-1", "DW-3"], "2026-06-11", note, OPERATION_ID)
    assert marked == ["DW-3"]
    assert not mark_open(path, "DW-1", note, OPERATION_ID)
    assert mark_open(path, "DW-3", note, OPERATION_ID)

    entries = {e.id: e for e in parse_ledger(path.read_text(encoding="utf-8"))}
    assert entries["DW-1"].status == "done 2026-06-11"
    assert entries["DW-3"].open


def test_mark_open_refuses_the_wrong_operation_id(tmp_path):
    path = write_ledger(tmp_path)
    close_reopenable(path, "DW-1", "by dw-a")
    snapshot = path.read_text(encoding="utf-8")

    assert not mark_open(path, "DW-1", "by dw-a", "another-run/dw-fix")
    assert path.read_text(encoding="utf-8") == snapshot


def test_mark_open_recomputes_operation_identity_after_a_crash(tmp_path):
    """No return-side receipt is needed after the close has reached disk."""
    path = write_ledger(tmp_path)
    operation_id = f"run-20260803T120000/{'dw-fix'}"
    assert mark_done_many_reopenable(path, ["DW-1"], "2026-06-11", "by dw-a", operation_id) == [
        "DW-1"
    ]

    # Simulate rehydration from already-persisted run + task identity. The marker
    # on disk, not the close call's returned list, carries the original line.
    replay_operation_id = "/".join(["run-20260803T120000", "dw-fix"])
    assert mark_open(path, "DW-1", "by dw-a", replay_operation_id)


def test_operation_id_line_breaks_are_encoded_inside_the_marker(tmp_path):
    path = write_ledger(tmp_path)
    operation_id = "run-20260803\nreview\u2028dw-fix"
    close_reopenable(path, "DW-1", "by dw-a", operation_id=operation_id)
    text = path.read_text(encoding="utf-8")

    assert operation_id not in text
    assert sum(line.startswith("resolution-undo: ") for line in text.splitlines()) == 1
    assert mark_open(path, "DW-1", "by dw-a", operation_id)


def test_reopenable_close_refuses_an_empty_operation_id_without_writing(tmp_path):
    path = write_ledger(tmp_path)
    snapshot = path.read_text(encoding="utf-8")

    with pytest.raises(ValueError, match="operation_id must not be empty"):
        mark_done_many_reopenable(path, ["DW-1"], "2026-06-11", "by dw-a", "")
    assert path.read_text(encoding="utf-8") == snapshot


@pytest.mark.parametrize(
    "prior_status_payload",
    [
        "not-hex",
        b"\xff".hex(),
        "status: done 2026-06-11".encode().hex(),
        "status: open\ninjected: value".encode().hex(),
        "status: open\u2028injected: value".encode().hex(),
    ],
)
def test_mark_open_refuses_a_malformed_or_non_open_prior_status_payload(
    tmp_path, prior_status_payload
):
    path = write_ledger(tmp_path)
    close_reopenable(path, "DW-1", "by dw-a")
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)
    marker_at = next(i for i, line in enumerate(lines) if line.startswith("resolution-undo: "))
    marker_prefix = lines[marker_at].rstrip("\n").rsplit(" ", 1)[0]
    lines[marker_at] = f"{marker_prefix} {prior_status_payload}\n"
    path.write_text("".join(lines), encoding="utf-8")
    snapshot = path.read_text(encoding="utf-8")

    assert not mark_open(path, "DW-1", "by dw-a", OPERATION_ID)
    assert path.read_text(encoding="utf-8") == snapshot


@pytest.mark.parametrize("tamper", ["status-date", "marker-date", "marker-adjacency"])
def test_mark_open_refuses_a_tampered_done_status_or_marker(tmp_path, tamper):
    path = write_ledger(tmp_path)
    close_reopenable(path, "DW-1", "by dw-a")
    text = path.read_text(encoding="utf-8")
    if tamper == "status-date":
        text = text.replace("status: done 2026-06-11", "status: done 2026-06-12", 1)
    elif tamper == "marker-date":
        marker = next(line for line in text.splitlines() if line.startswith("resolution-undo: "))
        text = text.replace(marker, marker.replace("2026-06-11", "2026-06-12"), 1)
    else:
        text = text.replace(
            "\nresolution-undo:", "\ndecision: 2026-06-11 keep\nresolution-undo:", 1
        )
    path.write_text(text, encoding="utf-8")
    snapshot = path.read_text(encoding="utf-8")

    assert not mark_open(path, "DW-1", "by dw-a", OPERATION_ID)
    assert path.read_text(encoding="utf-8") == snapshot


def test_mark_open_requires_the_resolution_directly_below_status(tmp_path):
    """A later matching resolution belongs to a different annotation. Searching
    the whole entry would turn this specific undo into a general reopen."""
    path = write_ledger(tmp_path, "### DW-1: closed elsewhere\n\nstatus: open\n")
    close_reopenable(path, "DW-1", "by dw-a")
    path.write_text(
        path.read_text(encoding="utf-8").replace(
            "\nresolution: by dw-a", "\ndecision: 2026-06-11 keep\nresolution: by dw-a", 1
        ),
        encoding="utf-8",
    )
    snapshot = path.read_text(encoding="utf-8")
    assert not mark_open(path, "DW-1", "by dw-a", OPERATION_ID)
    assert path.read_text(encoding="utf-8") == snapshot


def test_mark_open_tolerates_reformatted_resolution_whitespace(tmp_path):
    path = write_ledger(tmp_path, "### DW-1: closed then reflowed\n\nstatus: open\n")
    close_reopenable(path, "DW-1", "by dw-a")
    path.write_text(
        path.read_text(encoding="utf-8").replace("resolution: by dw-a", "resolution:   by dw-a  "),
        encoding="utf-8",
    )
    assert mark_open(path, "DW-1", "by dw-a", OPERATION_ID)
    entry = parse_ledger(path.read_text(encoding="utf-8"))[0]
    assert entry.open and "resolution:" not in entry.body


@pytest.mark.parametrize("note", ["  padded note  ", "line one\nline two"])
def test_mark_open_uses_the_same_note_normalization_as_mark_done(tmp_path, note):
    """Main's mark_done flattens line breaks and permits padded clean text. The
    undo accepts the original caller value, not only the rendered ledger line."""
    path = write_ledger(tmp_path)
    before = path.read_text(encoding="utf-8")
    close_reopenable(path, "DW-1", note)
    assert mark_open(path, "DW-1", note, OPERATION_ID)
    assert path.read_text(encoding="utf-8") == before


def test_mark_open_is_idempotent_after_the_undo(tmp_path):
    path = write_ledger(tmp_path)
    close_reopenable(path, "DW-1", "by dw-a")
    assert mark_open(path, "DW-1", "by dw-a", OPERATION_ID)
    snapshot = path.read_text(encoding="utf-8")
    assert not mark_open(path, "DW-1", "by dw-a", OPERATION_ID)
    assert path.read_text(encoding="utf-8") == snapshot


def test_mark_open_write_failure_raises_and_keeps_the_closed_entry(tmp_path, monkeypatch):
    path = write_ledger(tmp_path)
    close_reopenable(path, "DW-1", "by dw-a")
    closed = path.read_text(encoding="utf-8")

    def boom(path, text):
        raise OSError("no space left")

    monkeypatch.setattr(deferredwork, "atomic_write_text", boom)
    with pytest.raises(OSError, match="no space left"):
        mark_open(path, "DW-1", "by dw-a", OPERATION_ID)
    assert path.read_text(encoding="utf-8") == closed


def test_reopenable_close_write_failure_raises_and_keeps_the_open_entry(tmp_path, monkeypatch):
    path = write_ledger(tmp_path)
    before = path.read_text(encoding="utf-8")

    def boom(path, text):
        raise OSError("no space left")

    monkeypatch.setattr(deferredwork, "atomic_write_text", boom)
    with pytest.raises(OSError, match="no space left"):
        mark_done_many_reopenable(path, ["DW-1", "DW-3"], "2026-06-11", "by dw-a", OPERATION_ID)
    assert path.read_text(encoding="utf-8") == before


def test_append_decision(tmp_path):
    path = write_ledger(tmp_path)
    assert append_decision(path, "DW-3", "2026-06-11", "Keep cap", "frozen intent stands")
    entries = {e.id: e for e in parse_ledger(path.read_text(encoding="utf-8"))}
    assert "decision: 2026-06-11 Keep cap — frozen intent stands" in entries["DW-3"].body
    assert entries["DW-3"].open  # decision alone does not close
    assert "decision:" not in entries["DW-1"].body


def test_append_decision_then_mark_done(tmp_path):
    path = write_ledger(tmp_path)
    assert append_decision(path, "DW-3", "2026-06-11", "Close", "")
    assert mark_done(path, "DW-3", "2026-06-11", "closed by human decision")
    entries = {e.id: e for e in parse_ledger(path.read_text(encoding="utf-8"))}
    assert entries["DW-3"].status == "done 2026-06-11"
    assert "decision: 2026-06-11 Close" in entries["DW-3"].body


def test_append_decision_missing_file(tmp_path):
    assert not append_decision(tmp_path / "nope.md", "DW-1", "2026-06-11", "x", "y")
    assert not mark_done(tmp_path / "nope.md", "DW-1", "2026-06-11", "x")


# ------------------------------------------------- line-break injection (#305)
#
# The ledger is line-oriented and every mutator interpolates its arguments, so a
# value carrying a break injects ledger lines. Free text is SANITIZED, never
# refused, and nothing upstream rejects it either: the close paths call these
# writers bare (sweep._close_resolved, decisions.apply_pre_answer), so a raise
# ends the sweep run as crashed. Only the orchestrator-owned enumerables (date,
# status, severity) raise.

# Derived from the oracle, never typed. A literal U+2028 in a source file is
# trivially normalized to a plain space by an editor or a tool, and the two are
# then indistinguishable by inspection — which silently turns a line-break case
# into a whitespace case that passes for the wrong reason. The bound is safe:
# the exhaustive test below proves the whole split set lives under it.
BREAK_CHARS = tuple(chr(c) for c in range(0x2030) if len(("a" + chr(c) + "b").splitlines()) > 1)
# Multi-character runs, which `BREAK_CHARS` cannot reach: every entry there is
# one character, and the break-only cases collapse to "" before the quantifier
# matters. Without these, deleting `+` from LINE_BREAK_RE leaves the suite
# green while every CRLF value silently gains a second space — and CRLF is the
# common real-world break, being what a Windows-authored result.json carries.
BREAK_RUNS = ("\r\n", "\n\n")


def test_line_break_set_is_exactly_what_str_splitlines_splits_on():
    """`LINE_BREAK_RE` must cover every character `str.splitlines()` splits on,
    because `parse_legacy` scans the ledger with it while `parse_ledger` matches
    with `re.MULTILINE` — a member missed here splits an entry for one reader and
    is invisible to the other.

    Checked by enumeration rather than by listing the members: written as source
    escapes these are easy to get wrong, and a `\\u2028` silently normalized to a
    plain space would widen the class to collapse ordinary spaces — quietly
    breaking the byte-identity guarantee instead of the injection guard."""
    splits = {chr(c) for c in range(0x110000) if len(("a" + chr(c) + "b").splitlines()) > 1}
    matches = {chr(c) for c in range(0x110000) if LINE_BREAK_RE.fullmatch(chr(c))}
    assert matches == splits
    assert " " not in matches and "\t" not in matches  # ordinary whitespace is not a break


@pytest.mark.parametrize("brk", [*BREAK_CHARS, *BREAK_RUNS])
def test_mark_done_many_sanitizes_an_injected_note(tmp_path, brk):
    """Driven through `mark_done_many`, not the `mark_done` wrapper: that is the
    entry point `Engine._apply_deferred_closes` uses, so a guard placed on the
    wrapper would be inert for every story close."""
    path = write_ledger(tmp_path)
    before = len(parse_ledger(path.read_text(encoding="utf-8")))

    note = f"fixed{brk}### DW-99: injected{brk}status: open"

    assert mark_done_many(path, ["DW-1"], "2026-06-11", note) == ["DW-1"]

    text = path.read_text(encoding="utf-8")
    entries = {e.id: e for e in parse_ledger(text)}
    assert len(entries) == before  # no phantom DW-99 minted
    assert set(entries) == {"DW-1", "DW-2", "DW-3"}
    assert entries["DW-1"].status == "done 2026-06-11"
    # the injected text survives as prose on one line — sanitized, not dropped
    assert "resolution: fixed ### DW-99: injected status: open" in entries["DW-1"].body


@pytest.mark.parametrize("brk", [*BREAK_CHARS, *BREAK_RUNS])
def test_mark_done_sanitizes_a_note_that_would_double_the_status_line(tmp_path, brk):
    """The quiet half of the bug, and the half a status assertion cannot see:
    `STATUS_RE` takes the FIRST match, so an injected `status:` line leaves both
    `entry.status` and `entry.open` reporting exactly what they should. The damage
    is structural — the entry ends up carrying two status lines, so the ledger no
    longer says one thing about it, and which line wins depends on which reader
    looks. Asserted by counting lines the way `str.splitlines()` does, which is
    why the U+2028 case belongs here too."""
    path = write_ledger(tmp_path)

    assert mark_done(path, "DW-1", "2026-06-11", f"fixed{brk}status: open")

    (entry,) = [e for e in parse_ledger(path.read_text(encoding="utf-8")) if e.id == "DW-1"]
    assert entry.status == "done 2026-06-11"
    assert [line for line in entry.body.splitlines() if line.startswith("status:")] == [
        "status: done 2026-06-11"
    ]


@pytest.mark.parametrize("run", BREAK_RUNS)
def test_a_multi_character_break_run_collapses_to_exactly_one_space(tmp_path, run):
    """`LINE_BREAK_RE`'s `+` quantifier, which nothing else pins: a run of breaks
    must become ONE space, not one per character. CRLF is the case that matters —
    it is what a Windows-authored `result.json` carries, so it is the most likely
    break to reach these writers at all."""
    path = write_ledger(tmp_path)

    assert mark_done(path, "DW-1", "2026-06-11", f"fixed{run}in src/foo.py")

    (entry,) = [e for e in parse_ledger(path.read_text(encoding="utf-8")) if e.id == "DW-1"]
    assert "resolution: fixed in src/foo.py" in entry.body


def test_mark_done_sanitizes_a_note_that_would_truncate_the_entry(tmp_path):
    """The sharpest shape: a break followed by the flat appender's opening line.
    `FLAT_ENTRY_RE` bounds a canonical entry on exactly that line (#304), so an
    unsanitized note cuts DW-1's span short there and the tail re-surfaces as a
    phantom *legacy* item — visible to the sweep's migration trigger, `--dry-run`
    leftovers and the TUI as work nobody filed."""
    path = write_ledger(tmp_path)

    assert mark_done(path, "DW-1", "2026-06-11", "fixed\n- source_spec: `x.md`")

    text = path.read_text(encoding="utf-8")
    (entry,) = [e for e in parse_ledger(text) if e.id == "DW-1"]
    assert "resolution: fixed - source_spec: `x.md`" in entry.body
    assert parse_legacy(text) == []


@pytest.mark.parametrize(
    "date",
    [
        "nope",
        "2026-6-11",
        "20260611",
        "2026-02-30",
        "2026-06-11\nx",
        "\u0662\u0660\u0662\u0666-\u0660\u0666-\u0660\u0669",
    ],
    ids=["prose", "unpadded", "compact", "impossible-day", "with-break", "arabic-indic-digits"],
)
def test_mark_done_many_raises_on_a_bad_date_without_writing(tmp_path, date):
    """`date` is orchestrator-owned (`Engine._today()`), so a bad value is a
    programmer bug — the one place in these writers that raises. The digits are
    pinned to ASCII by the pattern itself (`[0-9]`, not `\\d`), so the shape
    check refuses an Arabic-Indic date rather than leaning on `fromisoformat`."""
    path = write_ledger(tmp_path)
    snapshot = path.read_text(encoding="utf-8")

    with pytest.raises(ValueError, match="date must be YYYY-MM-DD"):
        mark_done_many(path, ["DW-1"], date, "fixed")

    assert path.read_text(encoding="utf-8") == snapshot


def test_the_date_pattern_itself_refuses_non_ascii_digits():
    """`\\d` matches Arabic-Indic, fullwidth and mathematical digit forms; `[0-9]`
    does not. `date.fromisoformat` refuses them a moment later either way, so this
    pins the *pattern* rather than the outcome — the docstring's claim is about
    the regex, and only a direct assertion can hold it to that."""
    assert _ISO_DATE_RE.fullmatch("2026-06-09")
    for foreign in (
        "\u0662\u0660\u0662\u0666-\u0660\u0666-\u0660\u0669",
        "\uff12\uff10\uff12\uff16-\uff10\uff16-\uff10\uff19",
    ):
        assert not _ISO_DATE_RE.fullmatch(foreign)


def test_bad_date_raises_even_with_no_ledger_on_disk(tmp_path):
    """Validated at function entry, ahead of the `is_file()` short-circuit: a
    guard that only fires when the ledger happens to exist is one an absent
    fixture hides."""
    missing = tmp_path / "nope.md"

    with pytest.raises(ValueError, match="date must be YYYY-MM-DD"):
        mark_done_many(missing, ["DW-1"], "nope", "fixed")
    with pytest.raises(ValueError, match="date must be YYYY-MM-DD"):
        append_decision(missing, "DW-1", "nope", "Keep", "detail")


def test_append_decision_sanitizes_label_and_detail(tmp_path):
    path = write_ledger(tmp_path)

    assert append_decision(
        path, "DW-3", "2026-06-11", "Keep\ncap", f"Keep{BREAK_CHARS[-1]}status: done 2026-01-01"
    )

    text = path.read_text(encoding="utf-8")
    entries = {e.id: e for e in parse_ledger(text)}
    assert "decision: 2026-06-11 Keep cap — Keep status: done 2026-01-01" in entries["DW-3"].body
    assert len([line for line in text.splitlines() if line.startswith("decision:")]) == 1
    assert entries["DW-3"].open  # the injected `status:` did not close it


@pytest.mark.parametrize("detail", [*BREAK_CHARS, " \n ", "\r\n\t"])
def test_append_decision_drops_the_separator_for_a_break_only_detail(tmp_path, detail):
    """A detail that sanitizes to nothing must take the ` — ` with it. The
    ordering is the guard: sanitizing after the emptiness test would leave the
    entry carrying a dangling separator promising a detail that is not there."""
    path = write_ledger(tmp_path)

    assert append_decision(path, "DW-3", "2026-06-11", "Keep cap", detail)

    (line,) = [
        ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.startswith("decision:")
    ]
    assert line == "decision: 2026-06-11 Keep cap"
    assert "—" not in line


def test_writers_leave_a_clean_value_byte_identical(tmp_path):
    """No reformatting of values that never carried a break — the sanitizer must
    not become an incidental whitespace normalizer for existing ledgers."""
    path = write_ledger(tmp_path)
    padded = "  fixed  in  src/foo.py\t"

    assert mark_done(path, "DW-1", "2026-06-11", padded)
    assert append_decision(path, "DW-3", "2026-06-11", " Keep ", padded)

    text = path.read_text(encoding="utf-8")
    assert f"resolution: {padded}" in text
    assert f"decision: 2026-06-11  Keep  — {padded}" in text


# ------------------------------------------------------------------- legacy
#
# Fixtures are condensed verbatim from four real pre-DW project ledgers,
# one per shape the parser must handle.

# id'd bullets under "## Deferred from:" sections; bold/bracket done markers
LIGHTS_OUT = """\

## Deferred from: code review of gdd.md (2026-06-08)

- W1 — **RESOLVED 2026-06-09**: Validate the dual-clock squeeze is mathematically survivable. Epic-0 tuning gate. [MAJOR] — harness PASS and human **GO** decision. [CLOSED]
- W2 — Gloom engagement/retention once Top-Gloom is out of reach. Playtest watch (already a Success Metric). [MINOR]

## Deferred from: code review of story 0.3 (2026-06-08)

- W1 — duplicate native id in another section. [MINOR→MAJOR if missed]
- **W-1.2-c** — CLOSED: `moveSpeed` finite > 0 law (+ test).

## 2026-06-09 — epics-review absorb, 3-layer code review (spec-apply-epics-review)

- D-1 — `ShouldForceReturnToPool` cap-equality boundary unpinned. [MINOR]
"""

# "### D-CAP-001: title — RESOLVED" entry headings with field bullets
STORY_MAKER = """\
# Deferred Work

## From Epic 8 capstone live run (2026-06-11)

### D-CAP-001: claim entity references never resolved to canonical bible entity_ids — RESOLVED
- **Severity:** high (V0-blocker — broke approve→merge)
- **Detail:** Story 7.1 `extract_claims` derives references from draft text alone.
- **Resolution:** `spec-dw-cap-001-entity-id-resolution` — deterministic resolver added.

### D-CAP-002: an *ambiguous* claim reference is labeled `[NEW]` to the proposal composer
- **Severity:** low (rare — needs two bible entities sharing a slug)
- **Detail:** the label set is binary (KNOWN/NEW).

---

## D-8.6-001 — fact_key prefix not enforced against the character's entity_id (Story 8.6)

- **Surfaced by:** Story 8.6 retro review (Blind Hunter), 2026-06-04.
- **Severity:** low — a mis-prefixed key still lands in this character's own delta file.
"""

# strikethrough sections/bullets, bold-titled open bullets, no native ids
NOTEY = """\
# Deferred Work

## ~~Deferred from: Epic 1 — Instant Note Capture (2026-04-03)~~ DONE

### ~~Cluster 2: Frontend Core (Stories 1.6–1.10)~~ DONE

~~Depends on: Backend Foundation (Stories 1.1–1.5)~~

- ~~**Story 1.6** — Design Token System (CSS Custom Properties)~~

### ~~Cluster 3: Window & Daemon (Stories 1.11–1.14)~~ DONE

### Deferred from: code review of 3-2-full-text-search-tauri-command (2026-04-06) — 1 open item remaining

- ~~**Snippet `<mark>` HTML tags — XSS risk** — raw `<mark>` HTML injected.~~ → Verified safe: rendered as React text nodes
- **`i64`-to-`number` precision loss** — Specta maps Rust `i64` to JS `number`. IDs beyond 2^53 lose precision silently.
"""

# topic sections with status-suffixed headings and marker-suffixed bullets
MUDCEPTION = """\
# Deferred Work

## Epic 0: Validation (remaining stories)

- ~~**0-1**: Project scaffold — monorepo structure, workspace initialization~~ DONE
- ~~**0-2**: Hello-world SpacetimeDB module — room/exit tables~~ DONE

## Auth Improvements (deferred from 0-6 review)

- ~~OIDC id_token stored as plaintext in user:// — consider OS keychain~~ DOCUMENTED
- Add client_disconnected reducer to clean up orphaned player rows

## Notification Table Visibility (deferred from Story 1.5 review — RESOLVED)

## Config Externalization (split from Epic 0 deferred — DONE)

- ~~Move OIDC client ID to external config file~~ DONE (web already had .env)
"""


def test_legacy_lights_out_shape():
    entries = parse_legacy(LIGHTS_OUT)
    assert [e.id for e in entries] == ["W1", "W2", "W1", "W-1.2-c", "D-1"]
    w1, w2, w1_dup, w12c, d1 = entries
    assert w1.done and w1.severity == "high"  # [MAJOR], **RESOLVED**/[CLOSED]
    assert w1.title.startswith("Validate the dual-clock squeeze")
    assert not w2.done and w2.severity == "low"  # [MINOR]
    assert not w1_dup.done and w1_dup.severity == "low"  # [MINOR→MAJOR ...]
    assert w1.key != w1_dup.key  # same native id, different sections
    assert w12c.done  # plain "CLOSED:" prefix after the bold id
    assert w12c.title.startswith("`moveSpeed` finite > 0 law")
    assert not d1.done
    assert "epics-review absorb" in d1.section  # dated heading is a section


def test_legacy_story_maker_shape():
    entries = parse_legacy(STORY_MAKER)
    assert [e.id for e in entries] == ["D-CAP-001", "D-CAP-002", "D-8.6-001"]
    cap1, cap2, d86 = entries
    assert cap1.done  # "— RESOLVED" heading suffix
    assert cap1.title.endswith("bible entity_ids")  # suffix trimmed
    assert cap1.severity == "high"  # from "- **Severity:** high"
    assert not cap2.done and cap2.severity == "low"  # [NEW] is not a severity
    assert not d86.done and d86.severity == "low"  # "—"-separated heading id
    # field bullets are entry body, not standalone items
    assert "Surfaced by" in d86.body
    assert "**Detail:**" in cap1.body


def test_legacy_notey_shape():
    entries = parse_legacy(NOTEY)
    assert len(entries) == 4
    story16, cluster3, xss, i64 = entries
    assert story16.done  # struck bullet under a struck section
    assert story16.title == "Story 1.6 — Design Token System (CSS Custom Properties)"
    # item-less done section emits itself; the parent epic heading (which has
    # child headings) does not
    assert cluster3.done and "Cluster 3" in cluster3.title
    assert not any("Epic 1" in e.title for e in entries)
    assert xss.done and xss.title == "Snippet `<mark>` HTML tags — XSS risk"
    assert not i64.done  # the one open bullet in a "1 open item remaining" section
    assert i64.title == "`i64`-to-`number` precision loss"
    assert i64.id == ""


def test_legacy_mudception_shape():
    entries = parse_legacy(MUDCEPTION)
    assert len(entries) == 6
    assert [e.id for e in entries[:2]] == ["0-1", "0-2"]
    assert all(e.done for e in entries[:2])  # "Epic 0:" is a section, not an id
    oidc, reducer, notif, config = entries[2:]
    assert oidc.done  # ~~...~~ DOCUMENTED
    assert not reducer.done  # plain bullet in an open section
    assert notif.done and "Notification Table Visibility" in notif.title
    assert config.done  # bullet under a "(... — DONE)" section


def test_legacy_ignores_canonical_ledger():
    assert parse_legacy(LEDGER) == []
    assert not has_legacy(LEDGER)
    assert has_legacy(NOTEY)


def test_mixed_ledger_keeps_both_views_separate():
    mixed = (
        LEDGER + "\n## Deferred from: code review of spec-9-9 (2026-06-12)\n\n"
        "- ~~**Old fixed thing** — was broken, now fixed~~ → fixed in 9.9\n"
        "- **New open thing** — `parser.py` mishandles em-dashes — needs a guard\n"
    )
    assert open_ids(mixed) == {"DW-1", "DW-3"}  # strict view unchanged
    entries = parse_legacy(mixed)
    assert [e.done for e in entries] == [True, False]
    assert all("DW-" not in e.body for e in entries)


def test_flat_append_after_canonical_stays_visible_to_legacy_parser():
    text = (
        "# Deferred Work\n\n"
        "### DW-1: canonical\n\n"
        "origin: test\nlocation: n/a\nreason: test\nstatus: open\n\n"
        "- source_spec: `spec-next.md`\n"
        "  summary: later flat finding\n"
        "  evidence: must not be swallowed by DW-1\n"
    )

    (canonical,) = parse_ledger(text)
    (legacy,) = parse_legacy(text)

    assert "later flat finding" not in canonical.body
    assert legacy.title == "later flat finding"


# The canonical-span boundary has to recognize every flat shape parse_legacy
# does (#304): a boundary keyed on the full three-line block leaves the bug in
# place for the partial ones — and those are not hypothetical, they are the
# shapes test_flat_appender_missing_summary_falls_back and
# test_flat_appender_in_done_section_is_done already pin. Each case asserts both
# halves of the fix: the block reaches parse_legacy AND DW-1 keeps its status.
FLAT_TAIL_SHAPES = {
    "full-block": "- source_spec: `s.md`\n  summary: finding\n  evidence: e\n",
    "no-summary": "- source_spec: `s.md`\n  evidence: orphaned note\n",
    "no-evidence": "- source_spec: `s.md`\n  summary: finding\n",
    "swapped-order": "- source_spec: `s.md`\n  evidence: e\n  summary: finding\n",
    "field-interleaved": "- source_spec: `s.md`\n  severity: high\n  summary: finding\n",
    "star-bullet": "* source_spec: `s.md`\n  summary: finding\n  evidence: e\n",
    "tab-after-marker": "-\tsource_spec: `s.md`\n  summary: finding\n  evidence: e\n",
    "padded-marker": "-   source_spec: `s.md`\n  summary: finding\n  evidence: e\n",
    "uppercase-field": "- SOURCE_SPEC: `s.md`\n  summary: finding\n  evidence: e\n",
}


@pytest.mark.parametrize("shape", sorted(FLAT_TAIL_SHAPES), ids=sorted(FLAT_TAIL_SHAPES))
def test_flat_append_visible_for_every_shape_parse_legacy_accepts(shape):
    text = (
        "# Deferred Work\n\n### DW-1: canonical\n\n"
        "origin: test\nlocation: n/a\nreason: test\nstatus: open\n\n" + FLAT_TAIL_SHAPES[shape]
    )

    (canonical,) = parse_ledger(text)
    (legacy,) = parse_legacy(text)

    assert canonical.open, "the boundary must not cost DW-1 its status"
    assert "source_spec" not in canonical.body
    assert legacy.body.lstrip().startswith(("- ", "* ", "-\t"))


def test_flat_boundary_never_truncates_above_the_entry_status_line():
    """A flat-shaped bullet *inside* an entry, ahead of its `status:`, must not
    move the span end above that status: an entry parsing as neither open nor
    done is a lost tracked entry, the same failure class in the other direction
    (open_ids() drops it, classify() reports it malformed)."""
    text = (
        "### DW-1: canonical\n\norigin: test\nlocation: n/a\n"
        "reason: the appender writes this shape, quoted here as evidence\n"
        "- source_spec: `quoted-example.md`\n"
        "  summary: quoted, not a real finding\n"
        "  evidence: prose inside DW-1, not an appended block\n"
        "status: open\n"
    )

    (canonical,) = parse_ledger(text)

    assert canonical.status == "open" and canonical.open
    assert "quoted-example.md" in canonical.body
    assert open_ids(text) == {"DW-1"}
    assert parse_legacy(text) == []


def test_flat_boundary_still_applies_after_a_quoted_block_inside_the_entry():
    """The in-entry bullet does not grant immunity to what follows it: a real
    appended block after the entry's `status:` is still bounded out."""
    text = (
        "### DW-1: canonical\n\norigin: test\nreason: prose\n"
        "- source_spec: `quoted-example.md`\n  summary: quoted\n  evidence: e\n"
        "status: open\n\n"
        "- source_spec: `real.md`\n  summary: real finding\n  evidence: e\n"
    )

    (canonical,) = parse_ledger(text)
    (legacy,) = parse_legacy(text)

    assert canonical.open
    assert "quoted-example.md" in canonical.body and "real finding" not in canonical.body
    assert legacy.title == "real finding"


def test_flat_boundary_exposes_the_block_when_the_entry_has_no_status_line():
    """No status line means nothing to protect — the entry is already malformed,
    so bounding the block out of it costs nothing and rescues the finding."""
    text = "# Deferred Work\n\n### DW-1: malformed\n\norigin: test\nreason: test\n\n" + (
        "- source_spec: `s.md`\n  summary: rescued finding\n  evidence: e\n"
    )

    (canonical,) = parse_ledger(text)
    (legacy,) = parse_legacy(text)

    assert canonical.status == "" and not canonical.open
    assert legacy.title == "rescued finding"


def test_flat_append_boundary_accepts_crlf():
    text = "\r\n".join(
        [
            "# Deferred Work",
            "",
            "### DW-1: canonical",
            "",
            "origin: test",
            "location: n/a",
            "reason: test",
            "status: open",
            "",
            "- source_spec: `spec-next.md`",
            "  summary: later flat finding",
            "  evidence: must not be swallowed by DW-1",
            "",
        ]
    )

    (canonical,) = parse_ledger(text)
    (legacy,) = parse_legacy(text)

    assert canonical.open
    assert "later flat finding" not in canonical.body
    assert legacy.title == "later flat finding"


@pytest.mark.parametrize("gap", ["\n", ""], ids=["blank-line", "no-blank-line"])
def test_flat_append_between_two_canonical_entries(gap):
    """The realistic sequence: the inner session defers (flat) and the
    orchestrator then refiles a canonical entry at EOF, sandwiching the block."""
    text = (
        "# Deferred Work\n\n### DW-1: a\n\norigin: t\nreason: t\nstatus: open\n"
        + gap
        + "- source_spec: `s.md`\n  summary: sandwiched finding\n  evidence: e\n"
        + "\n### DW-2: b\n\norigin: t\nreason: t\nstatus: open\n"
    )

    assert open_ids(text) == {"DW-1", "DW-2"}
    (legacy,) = parse_legacy(text)
    assert legacy.title == "sandwiched finding"


def test_two_flat_blocks_after_the_last_entry_are_both_visible():
    block = "- source_spec: `s.md`\n  summary: %s\n  evidence: e\n"
    text = (
        "# Deferred Work\n\n### DW-1: a\n\norigin: t\nreason: t\nstatus: open\n\n"
        + (block % "first")
        + (block % "second")
    )

    assert open_ids(text) == {"DW-1"}
    assert [e.title for e in parse_legacy(text)] == ["first", "second"]


def test_flat_append_followed_by_a_legacy_section_keeps_both():
    text = (
        "# Deferred Work\n\n### DW-1: a\n\norigin: t\nreason: t\nstatus: open\n\n"
        "- source_spec: `s.md`\n  summary: flat finding\n  evidence: e\n"
        "\n## Deferred from: an older review (2026-06-01)\n\n"
        "- W1 — a freeform legacy item\n"
    )

    assert open_ids(text) == {"DW-1"}
    assert [e.title for e in parse_legacy(text)] == ["flat finding", "a freeform legacy item"]


def test_legacy_item_does_not_swallow_masked_canonical_neighbor():
    text = (
        "## Deferred from: somewhere (2026-06-01)\n\n"
        "- open legacy item directly above a DW entry\n"
        "### DW-9: Canonical\n\nstatus: open\n"
    )
    entries = parse_legacy(text)
    assert len(entries) == 1
    assert "DW-9" not in entries[0].body and "status:" not in entries[0].body
    assert open_ids(text) == {"DW-9"}


def test_legacy_keys_stable_under_unrelated_edits():
    before = {e.title: e.key for e in parse_legacy(MUDCEPTION)}
    edited = MUDCEPTION.replace(
        "- ~~**0-1**: Project scaffold — monorepo structure, workspace initialization~~ DONE\n",
        "",
    )
    after = {e.title: e.key for e in parse_legacy(edited)}
    for title, key in after.items():
        assert before[title] == key


def test_legacy_prose_and_rules_are_not_items():
    text = (
        "# Deferred Work\n\nSome intro prose, not an item.\n\n---\n\n"
        "## Open section\n\nNarrative paragraph under a section.\n\n- real item one\n"
    )
    entries = parse_legacy(text)
    assert [e.title for e in entries] == ["real item one"]


def test_field_severity_forms():
    assert field_severity("severity: HIGH") == "high"
    assert field_severity("- **Severity:** medium (scoped)") == "medium"
    assert field_severity("priority: blocker") == "critical"
    assert field_severity("severity: n/a") is None
    assert field_severity("no field here") is None


# the generic bmad-dev-auto review appender flat shape (step-04 deferral)
FLAT_APPENDER = """\
# Deferred Work

## Deferred from: spec-3-2-digest (2026-06-20)

- source_spec: `spec-3-2-digest.md`
  summary: Digest scheduler ignores the user timezone offset
  evidence: `schedule.py` hardcodes UTC; surfaced while reviewing the diff
- source_spec: `spec-3-2-digest.md`
  summary: No retry on transient SMTP failures
  evidence: send() raises and the run aborts with no backoff
"""


def test_flat_appender_uses_summary_as_title():
    entries = parse_legacy(FLAT_APPENDER)
    assert len(entries) == 2
    tz, smtp = entries
    assert tz.title == "Digest scheduler ignores the user timezone offset"
    assert smtp.title == "No retry on transient SMTP failures"
    # flat entries are freshly-appended findings: open, no native id, no severity
    assert not tz.done and not smtp.done
    assert tz.id == "" and smtp.id == ""
    assert tz.severity is None
    # source_spec / evidence stay in the body for the migrating session to read
    assert "source_spec" in tz.body and "evidence" in tz.body
    assert has_legacy(FLAT_APPENDER)


def test_flat_appender_missing_summary_falls_back():
    text = "## Deferred\n\n- source_spec: `spec-x.md`\n  evidence: orphaned note\n"
    (entry,) = parse_legacy(text)
    assert entry.title.startswith("source_spec:")  # no summary → keep the raw line
    assert not entry.done


def test_flat_appender_in_done_section_is_done():
    text = (
        "## Deferred from: old review (2026-06-01) — DONE\n\n"
        "- source_spec: `spec-y.md`\n  summary: Already handled upstream\n"
    )
    (entry,) = parse_legacy(text)
    assert entry.done and entry.title == "Already handled upstream"


# ------------------------------------------------------- append_entry / next_seq


def test_next_seq_past_highest():
    text = "### DW-3: a\nstatus: open\n\n### DW-7: b\nstatus: done 2026-01-01\n"
    assert next_seq(text) == 8


def test_next_seq_empty_starts_at_one():
    assert next_seq("") == 1
    assert next_seq("# Deferred Work\n") == 1


def test_append_entry_numbers_and_writes(tmp_path):
    p = tmp_path / "deferred-work.md"
    p.write_text("# Deferred Work\n\n### DW-4: existing\norigin: test\nstatus: open\n")
    new_id = append_entry(
        p,
        title="follow-up still recommended for dw-x",
        origin="review-budget-followup",
        source_spec="spec-foo.md",
        reason="review budget exhausted, work committed",
        severity="low",
    )
    assert new_id == "DW-5"
    entries = {e.id: e for e in parse_ledger(p.read_text())}
    assert "DW-5" in entries and entries["DW-5"].open
    body = entries["DW-5"].body
    assert "origin: review-budget-followup" in body
    assert "location: n/a" in body
    assert "source_spec: `spec-foo.md`" in body
    assert "severity: low" in body
    assert "follow-up still recommended for dw-x" in body


def test_append_entry_preserves_a_supplied_location(tmp_path):
    """`n/a` is only the default the orchestrator's own refiles take — the field
    is documented as `<file:line or component>` and must round-trip verbatim."""
    p = tmp_path / "deferred-work.md"
    assert (
        append_entry(
            p,
            title="follow-up",
            origin="code-review",
            source_spec="spec-foo.md",
            reason="still open",
            location="src/foo.py:12",
        )
        == "DW-1"
    )
    (entry,) = parse_ledger(p.read_text())
    assert "location: src/foo.py:12" in entry.body


def test_append_entry_idempotent_for_open_origin_and_spec(tmp_path):
    p = tmp_path / "deferred-work.md"
    p.write_text("# Deferred Work\n")
    first = append_entry(
        p, title="t", origin="review-budget-followup", source_spec="spec-foo.md", reason="r"
    )
    assert first == "DW-1"
    again = append_entry(
        p, title="t2", origin="review-budget-followup", source_spec="spec-foo.md", reason="r2"
    )
    assert again is None  # an open entry with the same origin+spec already exists
    assert len(parse_ledger(p.read_text())) == 1
    # a different source_spec is not blocked
    other = append_entry(
        p, title="t3", origin="review-budget-followup", source_spec="spec-bar.md", reason="r3"
    )
    assert other == "DW-2"


def test_append_entry_not_blocked_when_prior_is_done(tmp_path):
    p = tmp_path / "deferred-work.md"
    p.write_text(
        "### DW-1: t\norigin: review-budget-followup\n"
        "source_spec: `spec-foo.md`\nstatus: done 2026-01-01\n"
    )
    new_id = append_entry(
        p, title="t2", origin="review-budget-followup", source_spec="spec-foo.md", reason="r"
    )
    assert new_id == "DW-2"  # prior entry is done, not open → re-file allowed


def test_append_entry_creates_missing_ledger(tmp_path):
    p = tmp_path / "sub" / "deferred-work.md"
    new_id = append_entry(p, title="t", origin="o", source_spec="s.md", reason="r")
    assert new_id == "DW-1" and p.is_file()


def test_append_entry_idempotency_ignores_incidental_substring(tmp_path):
    """An unrelated open entry that merely *mentions* the origin marker and the
    spec filename in its `reason:` prose must not suppress a legitimately new
    entry — dedup matches the canonical field lines, not raw body substrings."""
    p = tmp_path / "deferred-work.md"
    p.write_text(
        "### DW-1: unrelated\norigin: code review\n"
        "reason: see the origin: review-budget-followup note re spec-foo.md for context\n"
        "status: open\n"
    )
    new_id = append_entry(
        p, title="t", origin="review-budget-followup", source_spec="spec-foo.md", reason="r"
    )
    assert new_id == "DW-2"  # not suppressed by the incidental mentions


@pytest.mark.parametrize("field", ["title", "origin", "source_spec", "reason", "location"])
@pytest.mark.parametrize("brk", [*BREAK_CHARS, *BREAK_RUNS])
def test_append_entry_sanitizes_free_text_into_one_line(tmp_path, field, brk):
    """One canonical entry, every field on its own line — no injected heading and
    no second `status:` line, whichever field carried the break."""
    p = tmp_path / "deferred-work.md"
    values = {
        "title": "follow-up",
        "origin": "code-review",
        "source_spec": "spec-foo.md",
        "reason": "still open",
        "location": "src/foo.py:12",
    }
    values[field] += f"{brk}### DW-99: injected{brk}status: done 2026-01-01"

    assert append_entry(p, **values) == "DW-1"

    text = p.read_text(encoding="utf-8")
    (entry,) = parse_ledger(text)  # one entry: no phantom DW-99 heading
    assert entry.id == "DW-1" and entry.open  # not closed by the injected status
    lines = text.splitlines()
    assert len(lines) == 6  # heading + origin/location/source_spec/reason/status
    assert sum(line.startswith("### ") for line in lines) == 1
    assert sum(line.startswith("status:") for line in lines) == 1


@pytest.mark.parametrize("title", [*BREAK_CHARS, *BREAK_RUNS, " \n ", "\r\n\t", "  ", "\t", "\xa0"])
def test_append_entry_names_an_entry_with_no_usable_title(tmp_path, title):
    """Two routes to the same unusable heading. A break-only title collapses to
    nothing, and `### DW-1: ` is a heading `HEADING_RE`'s `(.+?)` does not match:
    the caller is handed an id no reader can find, with `next_seq` already past
    it. A whitespace-only title never reaches the sanitizer at all — no break, so
    the byte-identity fast path returns it unchanged and still truthy — and
    parses to a heading that renders blank in `status`, `--json` and the TUI.
    Both get a name; neither gets a space."""
    p = tmp_path / "deferred-work.md"

    assert append_entry(p, title=title, origin="o", source_spec="s.md", reason="r") == "DW-1"

    text = p.read_text(encoding="utf-8")
    (entry,) = parse_ledger(text)  # findable, and the id was not burned
    assert entry.id == "DW-1"
    assert entry.title == "(untitled DW-1)"  # identifiable, not blank
    assert next_seq(text) == 2
    assert len(text.splitlines()) == 6


def test_append_entry_leaves_an_already_empty_title_as_it_was(tmp_path):
    """The substitution is scoped to a title that *had* content and lost it to
    sanitizing. An empty title in, empty title out — that hole predates this
    guard, and widening the fix to cover it would change bytes for a value that
    never carried a break."""
    p = tmp_path / "deferred-work.md"

    assert append_entry(p, title="", origin="o", source_spec="s.md", reason="r") == "DW-1"

    assert p.read_text(encoding="utf-8").startswith("### DW-1: \n")


def test_append_entry_idempotence_survives_sanitizing(tmp_path):
    """Sanitizing must happen BEFORE the idempotence scan: that scan compares the
    caller's value against the stored line via `field_line_present`, so
    sanitizing afterwards would compare raw against sanitized and append a fresh
    entry on every replay of the same defer."""
    p = tmp_path / "deferred-work.md"
    dirty = "spec-foo.md\nstatus: open"

    first = append_entry(
        p, title="t", origin="review-budget-followup", source_spec=dirty, reason="r"
    )
    again = append_entry(
        p, title="t2", origin="review-budget-followup", source_spec=dirty, reason="r2"
    )

    assert first == "DW-1"
    assert again is None
    assert len(parse_ledger(p.read_text(encoding="utf-8"))) == 1


@pytest.mark.parametrize("status", ["", "done", "closed", "open\n### DW-99: injected", "OPEN"])
def test_append_entry_raises_on_a_noncanonical_status_without_writing(tmp_path, status):
    p = tmp_path / "deferred-work.md"

    with pytest.raises(ValueError, match="status must be"):
        append_entry(p, title="t", origin="o", source_spec="s.md", reason="r", status=status)

    assert not p.exists()


def test_append_entry_raises_on_an_impossible_done_date_without_writing(tmp_path):
    p = tmp_path / "deferred-work.md"

    with pytest.raises(ValueError, match="date must be YYYY-MM-DD"):
        append_entry(
            p, title="t", origin="o", source_spec="s.md", reason="r", status="done 2026-02-30"
        )

    assert not p.exists()


@pytest.mark.parametrize("severity", ["urgent", "blocker", "low\nseverity: high"])
def test_append_entry_raises_on_a_noncanonical_severity_without_writing(tmp_path, severity):
    """The whitelist is `SEVERITY_ALIASES`'s normalization targets. `blocker` is
    an inbound alias the *parser* accepts from LLM-written ledgers, not a value
    this writer may emit — so it raises like any other programmer bug."""
    p = tmp_path / "deferred-work.md"

    with pytest.raises(ValueError, match="severity must be one of"):
        append_entry(p, title="t", origin="o", source_spec="s.md", reason="r", severity=severity)

    assert not p.exists()


def test_append_entry_accepts_every_canonical_severity(tmp_path):
    """Pins the whitelist against its source: a `SEVERITY_ALIASES` value that the
    writer refuses would mean the two have drifted."""
    for i, severity in enumerate(sorted(set(SEVERITY_ALIASES.values())), start=1):
        p = tmp_path / f"ledger-{i}.md"
        assert (
            append_entry(
                p, title="t", origin="o", source_spec="s.md", reason="r", severity=severity
            )
            == "DW-1"
        )
        assert f"severity: {severity}" in p.read_text(encoding="utf-8")


def test_append_entry_leaves_a_clean_value_byte_identical(tmp_path):
    p = tmp_path / "deferred-work.md"

    assert (
        append_entry(
            p,
            title="follow-up  for  dw-x",
            origin="review-budget-followup",
            source_spec="spec-foo.md",
            reason="review budget exhausted, work committed",
            location="src/foo.py:12",
            severity="low",
        )
        == "DW-1"
    )

    assert p.read_text(encoding="utf-8") == (
        "### DW-1: follow-up  for  dw-x\n"
        "origin: review-budget-followup\n"
        "location: src/foo.py:12\n"
        "source_spec: `spec-foo.md`\n"
        "severity: low\n"
        "reason: review budget exhausted, work committed\n"
        "status: open\n"
    )


def test_field_line_present_matches_field_not_substring():
    body = (
        "### DW-1: x\norigin: review-budget-followup\n"
        "source_spec: `spec-foo.md`\nreason: mentions spec-foobar.md and review-budget-followup-x\n"
        "status: open\n"
    )
    # exact field-line matches (plain and backtick-wrapped)
    assert field_line_present(body, "origin", "review-budget-followup")
    assert field_line_present(body, "source_spec", "spec-foo.md")
    # a superstring value must not match the shorter field line
    assert not field_line_present(body, "origin", "review-budget")
    # a value that only appears incidentally inside `reason:` is not a field line
    assert not field_line_present(body, "source_spec", "spec-foobar.md")


# ------------------------------ closes_deferred declaration primitives (#234)


def test_parse_declaration_normalizes_items_and_dedupes():
    """Lenient about items, exactly as the id parser is: an LLM-authored manifest
    may emit an unquoted id as a string and a bare number as an int."""
    ids, error = parse_declaration(["DW-1", " DW-2 ", 5, "", "DW-1"])
    assert ids == ("DW-1", "DW-2", "5")  # stripped, blanks dropped, order-preserving dedupe
    assert error is None


def test_parse_declaration_defaults_empty_when_absent():
    assert parse_declaration(None) == ((), None)
    assert parse_declaration([]) == ((), None)


def test_parse_declaration_rejects_a_non_list_container():
    """A string is iterable, so a lenient reading would silently turn one id into
    a list of characters. Callers pick the severity; the reading is shared."""
    ids, error = parse_declaration("DW-1")
    assert ids == ()
    assert error is not None and "must be a list" in error and "str" in error


def test_classify_partitions_open_done_unknown_and_malformed():
    """Four outcomes, not two: `not open` hides a satisfied declaration (a resume
    re-driving a landed close, which must stay silent) and an entry whose status
    line cannot be read at all (which must not)."""
    text = (
        "# Deferred Work\n\n"
        "### DW-1: a\nstatus: open\n\n"
        "### DW-2: b\nstatus: done 2026-06-01\n\n"
        "### DW-3: c\nstatus: in-progress\n\n"
        "### DW-4: d\nreason: no status line at all\n"
    )
    declared = classify(text, ["DW-1", "DW-2", "DW-3", "DW-4", "DW-99"])

    assert declared.open_ids == ("DW-1",)
    assert declared.already_done == ("DW-2",)
    assert declared.unknown == ("DW-99",)
    assert declared.malformed == ("DW-3", "DW-4")


DUPLICATE_OPEN_FIRST = (
    "# Deferred Work\n\n"
    "### DW-1: the first copy\nstatus: open\n\n"
    "### DW-1: the second copy\nstatus: done 2026-06-01\n"
)
DUPLICATE_DONE_FIRST = (
    "# Deferred Work\n\n"
    "### DW-1: the first copy\nstatus: done 2026-06-01\n\n"
    "### DW-1: the second copy\nstatus: open\n"
)


@pytest.mark.parametrize(
    ("text", "expected_open", "expected_done"),
    [
        (DUPLICATE_OPEN_FIRST, ("DW-1",), ()),
        (DUPLICATE_DONE_FIRST, (), ("DW-1",)),
    ],
    ids=["open-first", "done-first"],
)
def test_classify_reads_the_same_duplicate_the_mutation_writes(text, expected_open, expected_done):
    """`classify` must name the entry `_find_entry` acts on. Indexing last-wins
    while the mutation takes the first made them disagree, and BOTH orders then
    closed nothing while saying nothing: done-first classified `open`, then
    `_apply_done` refused the done copy it found first (so nothing was marked and
    not even an unmatched warning followed); open-first classified `already_done`
    and never attempted the write (#284 round-6 review, finding 4)."""
    declared = classify(text, ["DW-1"])

    assert declared.open_ids == expected_open
    assert declared.already_done == expected_done
    assert declared.unknown == ()
    assert declared.duplicates == ("DW-1",)  # reported either way — the ledger is corrupt


@pytest.mark.parametrize(
    ("text", "expected", "after"),
    [
        (DUPLICATE_OPEN_FIRST, ["DW-1"], ["done", "done"]),
        (DUPLICATE_DONE_FIRST, [], ["done", "open"]),
    ],
    ids=["open-first", "done-first"],
)
def test_mark_done_many_agrees_with_classify_on_a_duplicated_id(tmp_path, text, expected, after):
    """The other half of the same contract: what `classify` calls open is exactly
    what the write marks, so an id is never reported closed without being closed —
    nor, as before, classified open and then silently left open.

    The done-first ledger still ends with an open copy nobody touched. That is the
    honest outcome of a corrupt ledger (#286) and it is why the close is journaled
    as a duplicate rather than passed over."""
    p = tmp_path / "deferred-work.md"
    p.write_text(text, encoding="utf-8")
    declared = classify(text, ["DW-1"])

    assert list(mark_done_many(p, declared.open_ids, "2026-07-24", "note")) == expected

    entries = parse_ledger(p.read_text(encoding="utf-8"))
    assert [e.status.split()[0] for e in entries] == after


def test_classify_reports_no_duplicates_for_a_well_formed_ledger():
    declared = classify(LEDGER, ["DW-1", "DW-2"])

    assert declared.duplicates == ()


def test_mark_done_many_writes_once_and_reports_what_landed(tmp_path):
    p = tmp_path / "deferred-work.md"
    p.write_text(LEDGER, encoding="utf-8")

    # DW-2 is already done in the fixture, DW-99 is absent — both skipped, neither
    # an error: only the entries that were open get marked, in the order given.
    marked = mark_done_many(
        p, ["DW-1", "DW-3", "DW-2", "DW-99"], "2026-07-24", "resolved by story 1"
    )

    assert marked == ["DW-1", "DW-3"]
    entries = {e.id: e for e in parse_ledger(p.read_text(encoding="utf-8"))}
    for dw in ("DW-1", "DW-3"):
        assert entries[dw].status == "done 2026-07-24"
        assert "resolution: resolved by story 1" in entries[dw].body


def test_mark_done_many_is_all_or_nothing_on_a_write_failure(tmp_path, monkeypatch):
    """A per-id read-modify-write loop leaves earlier marks on disk when it raises
    partway through — a half-applied closure the caller never gets to journal, so
    the ledger claims resolutions the run has no record of."""
    p = tmp_path / "deferred-work.md"
    p.write_text(LEDGER, encoding="utf-8")
    before = p.read_bytes()
    monkeypatch.setattr(
        "bmad_loop.deferredwork.atomic_write_text",
        lambda path, text: (_ for _ in ()).throw(OSError("disk full")),
    )

    # the raise must REACH the caller, not just leave the file alone: a swallowed
    # error returning [] reads as "nothing declared was open" — the engine disarms
    # its restore snapshot and no journal line ever records the failed write
    with pytest.raises(OSError):
        mark_done_many(p, ["DW-1", "DW-3"], "2026-07-24", "note")

    assert p.read_bytes() == before  # nothing partially applied


def test_mark_done_many_skips_an_already_done_entry(tmp_path):
    """Idempotent for a resume that re-drives a close that already landed: no
    second resolution line, and the id is not reported as newly marked."""
    p = tmp_path / "deferred-work.md"
    p.write_text(LEDGER, encoding="utf-8")
    mark_done_many(p, ["DW-1"], "2026-07-24", "resolved by story 1")

    again = mark_done_many(p, ["DW-1"], "2026-07-25", "resolved by story 1")

    assert again == []
    body = next(e for e in parse_ledger(p.read_text(encoding="utf-8")) if e.id == "DW-1").body
    assert body.count("resolution: resolved by story 1") == 1
