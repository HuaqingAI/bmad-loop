"""`frontmatter.set_frontmatter_status` — the verified in-place status writer.

Two halves. The CHARACTERIZATION half was written against the pre-rewrite line
scanner and is green on both sides of it: it pins the formatting-preserving
minimal edit, and the two deliberate non-preservations (a quoted value and a
trailing inline comment are both dropped) that a "make it a YAML round-trip"
refactor would silently change out from under callers three files away. The
BEHAVIOR half pins the rewrite: a writer that verifies its own edit by
re-parsing, and RAISES rather than returning a `False` nobody reads when the
reader can see a status it cannot safely move.

`set_frontmatter_status`'s original tests live in `tests/test_resolve.py:63-138`,
next to `set_frontmatter_field`'s. They were deliberately left there —
characterize before restructuring, and moving them would hide this diff behind a
rename. Consolidating them here is filed as a follow-up.
"""

import pytest
import yaml

from bmad_loop import frontmatter, verify

_PLAIN = "---\ntitle: List command\nstatus: in-review\nowner: amelia\n---\n\n# Spec\n\nbody\n"


def _spec(tmp_path, text: str, name: str = "spec.md"):
    """Write a spec as raw bytes — `write_text` would relay line endings, and
    several of these fixtures are about exactly which bytes survive."""
    path = tmp_path / name
    path.write_bytes(text.encode("utf-8"))
    return path


# ============================================================ characterization
#
# Green before AND after the rewrite. Anything that fails here is a change to the
# contract callers already depend on, not a change to the defect being fixed.


def test_a_plain_flip_changes_the_status_line_and_nothing_else(tmp_path):
    """The whole point of a line edit over a YAML round-trip: field order,
    comments, quoting and body survive byte-for-byte."""
    spec = _spec(tmp_path, _PLAIN)
    assert frontmatter.set_frontmatter_status(spec, "done") is True
    assert spec.read_bytes().decode() == _PLAIN.replace("status: in-review", "status: done")


def test_flipping_to_the_status_already_there_returns_false_and_does_not_write(tmp_path):
    """Idempotence is observable, not just a return value: a no-op write would
    churn mtime and make every caller's spec look freshly edited to the artifact
    scans that key on it."""
    spec = _spec(tmp_path, _PLAIN.replace("in-review", "done"))
    before = spec.stat().st_mtime_ns
    assert frontmatter.set_frontmatter_status(spec, "done") is False
    assert spec.stat().st_mtime_ns == before


def test_a_quoted_value_is_written_back_unquoted(tmp_path):
    """A deliberate non-preservation, and the most load-bearing pin in this file.
    `conftest.write_spec` writes `status: '<v>'`, and substring assertions read
    the result back as an unquoted `status: done` (tests/test_runs.py:772,
    tests/test_stories_e2e.py:594,793). A refactor that "also preserves the
    value's quotes" breaks those three from here."""
    spec = _spec(tmp_path, "---\nstatus: 'in-review'\n---\nbody\n")
    assert frontmatter.set_frontmatter_status(spec, "done") is True
    assert spec.read_bytes().decode() == "---\nstatus: done\n---\nbody\n"


def test_a_trailing_inline_comment_on_the_status_line_is_dropped(tmp_path):
    """The other deliberate non-preservation. Keeping it needs to know where the
    scalar ends — a `#` inside a quoted value is not a comment — and a wrong
    guess turns a working spec into a refused write. Filed as a follow-up."""
    spec = _spec(tmp_path, "---\nstatus: in-review  # set by step-03\n---\nbody\n")
    assert frontmatter.set_frontmatter_status(spec, "done") is True
    assert spec.read_bytes().decode() == "---\nstatus: done\n---\nbody\n"


def test_a_status_prefixed_key_is_never_targeted(tmp_path):
    spec = _spec(tmp_path, "---\nstatus_note: keep me\nstatus: in-review\n---\nbody\n")
    assert frontmatter.set_frontmatter_status(spec, "done") is True
    assert spec.read_bytes().decode() == "---\nstatus_note: keep me\nstatus: done\n---\nbody\n"


def test_a_commented_out_status_line_is_never_targeted(tmp_path):
    spec = _spec(tmp_path, "---\n# status: draft\nstatus: in-review\n---\nbody\n")
    assert frontmatter.set_frontmatter_status(spec, "done") is True
    assert spec.read_bytes().decode() == "---\n# status: draft\nstatus: done\n---\nbody\n"


def test_no_frontmatter_block_returns_false_with_the_bytes_unchanged(tmp_path):
    spec = _spec(tmp_path, "# just a heading\n")
    assert frontmatter.set_frontmatter_status(spec, "done") is False
    assert spec.read_bytes() == b"# just a heading\n"


def test_a_missing_file_returns_false(tmp_path):
    assert frontmatter.set_frontmatter_status(tmp_path / "nope.md", "done") is False


def test_a_block_with_no_status_key_returns_false(tmp_path):
    """Contrast with `verify.set_frontmatter_field`, which INSERTS a missing key.
    The status helper never invents a status."""
    spec = _spec(tmp_path, "---\ntitle: t\n---\nbody\n")
    assert frontmatter.set_frontmatter_status(spec, "done") is False
    assert spec.read_bytes().decode() == "---\ntitle: t\n---\nbody\n"


def test_a_present_but_empty_status_is_filled(tmp_path):
    """A bmad-dev-auto template can leave the line blank. It reads as a status
    the writer must be able to move, not as a missing key."""
    spec = _spec(tmp_path, "---\nstatus:\ntitle: t\n---\nbody\n")
    assert frontmatter.set_frontmatter_status(spec, "done") is True
    assert spec.read_bytes().decode() == "---\nstatus: done\ntitle: t\n---\nbody\n"


def test_indentation_on_the_status_line_survives(tmp_path):
    spec = _spec(tmp_path, "---\n  status: in-review\n---\nbody\n")
    assert frontmatter.set_frontmatter_status(spec, "done") is True
    assert spec.read_bytes().decode() == "---\n  status: done\n---\nbody\n"


# ================================================================== behavior
#
# The rewrite. Every shape below reads as a real status through
# `read_frontmatter`, and the old line scanner answered each of them either with
# a silent `False` nobody read or with a corrupting write.

# name -> (spec text, the status `read_frontmatter` sees before AND after)
_UNREWRITABLE = {
    "flow-mapping": ("---\n{status: in-review, keep: 1}\n---\nbody\n", "in-review"),
    "block-scalar": ("---\nstatus: |\n  in-review\ntitle: t\n---\nbody\n", "in-review"),
    "value-on-the-next-line": ("---\nstatus:\n  in-review\ntitle: t\n---\nbody\n", "in-review"),
    "value-wrapped-over-two-lines": (
        "---\nstatus: awaiting\n  operator\ntitle: t\n---\nbody\n",
        "awaiting operator",
    ),
    "anchor-another-key-aliases": (
        "---\nstatus: &a in-review\nother: *a\n---\nbody\n",
        "in-review",
    ),
    # The one shape the "every other key unchanged" comparison is the SOLE gate
    # for — the status the reader resolves comes from a merged anchor block, so
    # the only line an edit can reach belongs to `defaults:`, shared state this
    # story does not own. The trial produces the right top-level status and
    # rewrites someone else's mapping to get it; every other check here passes it.
    # Ablating the comparison turns this row into a silent wrong-target write.
    "status-merged-in-from-an-anchor": (
        "---\ndefaults: &d\n  status: in-review\n  owner: amelia\n<<: *d\n---\nbody\n",
        "in-review",
    ),
}


@pytest.mark.parametrize("shape", sorted(_UNREWRITABLE), ids=sorted(_UNREWRITABLE))
def test_a_status_no_line_edit_can_move_raises_and_leaves_the_file_alone(tmp_path, shape):
    """The old scanner's two failure modes, one per row: a silent no-op (flow
    mapping) or a corrupting write (everything else — `status: done in-review`,
    a block scalar with its indicator eaten, an alias left dangling). The file
    must come out byte-identical and still READ as the status it had, so nothing
    downstream can mistake a refused write for a landed one."""
    text, reads_as = _UNREWRITABLE[shape]
    spec = _spec(tmp_path, text)
    with pytest.raises(frontmatter.FrontmatterWriteError):
        frontmatter.set_frontmatter_status(spec, "done")
    assert spec.read_bytes() == text.encode("utf-8")
    assert frontmatter.status_of(frontmatter.read_frontmatter(spec)) == reads_as


def test_an_unparseable_block_raises_where_the_reader_degrades(tmp_path):
    """The doctrine asymmetry, stated as a test. `read_frontmatter` turns this
    same input into `{}` (pinned at tests/test_verify.py:1980) because observation
    may degrade. A writer that did the same would conclude "no status here" from
    a block it could not read and report a repair it never made."""
    text = "---\nstatus: in-review\ntitle: [unclosed\n---\nbody\n"
    spec = _spec(tmp_path, text)
    assert frontmatter.read_frontmatter(spec) == {}  # the reader degrades...
    with pytest.raises(frontmatter.FrontmatterWriteError, match="does not parse as YAML"):
        frontmatter.set_frontmatter_status(spec, "done")  # ...the writer does not
    assert spec.read_bytes() == text.encode("utf-8")


def test_a_nested_status_is_not_the_status_and_is_never_written(tmp_path):
    """The old scanner rewrote the FIRST line that looked like `status:`, so a
    `meta:` block carrying one got the write and the story's real status never
    moved. Here the reader sees no top-level status, so there is nothing to
    change — and the decoy is left exactly as authored."""
    text = "---\nmeta:\n  status: in-review\n---\nbody\n"
    spec = _spec(tmp_path, text)
    assert frontmatter.set_frontmatter_status(spec, "done") is False
    assert spec.read_bytes() == text.encode("utf-8")


@pytest.mark.parametrize(
    ("decoy", "label"),
    [
        ("meta:\n  status: keep\n", "nested-key"),
        ("notes: |\n  status: keep this verbatim\n", "literal-block"),
    ],
    ids=["nested-key", "literal-block"],
)
def test_a_decoy_before_the_real_status_does_not_capture_the_write(tmp_path, decoy, label):
    """Candidates are iterated, not broken on at the first match. The decoy comes
    FIRST on purpose: under the old `break` it took the write and the real status
    line was never reached."""
    text = f"---\n{decoy}status: in-review\n---\nbody\n"
    spec = _spec(tmp_path, text)
    assert frontmatter.set_frontmatter_status(spec, "done") is True
    assert spec.read_bytes().decode() == f"---\n{decoy}status: done\n---\nbody\n"
    assert frontmatter.status_of(frontmatter.read_frontmatter(spec)) == "done"


@pytest.mark.parametrize(
    ("key", "value"),
    [('"status"', "in-review"), ("'status'", "in-review"), ("status ", "in-review")],
    ids=["double-quoted-key", "single-quoted-key", "space-before-colon"],
)
def test_a_key_spelling_yaml_accepts_is_rewritten_with_its_formatting_kept(tmp_path, key, value):
    """All three read as `status` through `read_frontmatter`; all three were a
    silent no-op under `lstrip().startswith("status:")`. The key's own spelling
    survives the rewrite — only the value moves. (A TAB before the colon is not
    in this list on purpose: PyYAML rejects it outright, so it is not a shape the
    reader ever accepts — it lands on the unparseable-block raise above.)"""
    spec = _spec(tmp_path, f"---\n{key}: {value}\n---\nbody\n")
    assert frontmatter.set_frontmatter_status(spec, "done") is True
    assert spec.read_bytes().decode() == f"---\n{key}: done\n---\nbody\n"
    assert frontmatter.status_of(frontmatter.read_frontmatter(spec)) == "done"


def test_a_key_that_merely_ends_where_status_does_is_not_a_status_line(tmp_path):
    """`(?P=q)` backreference: an unquoted key ending in a stray quote char reads
    as the key `status"`, not `status`. The reader sees no top-level status, so
    there is nothing to change — and nothing is written to a key nobody asked
    about."""
    text = '---\nstatus": in-review\ntitle: t\n---\nbody\n'
    spec = _spec(tmp_path, text)
    assert frontmatter.read_frontmatter(spec) == {'status"': "in-review", "title": "t"}
    assert frontmatter.set_frontmatter_status(spec, "done") is False
    assert spec.read_bytes() == text.encode("utf-8")


def test_verify_re_exports_the_same_exception_object(tmp_path):
    """`verify` re-exports the frontmatter names so every `verify.<name>` call
    site stays valid; an `except verify.FrontmatterWriteError` must catch what
    `frontmatter` raises, which only holds if it is the same class."""
    assert verify.FrontmatterWriteError is frontmatter.FrontmatterWriteError
    spec = _spec(tmp_path, "---\n{status: in-review}\n---\nbody\n")
    with pytest.raises(verify.FrontmatterWriteError):
        verify.set_frontmatter_status(spec, "done")


def test_the_verified_edit_is_never_a_yaml_round_trip(tmp_path):
    """`yaml.safe_load` is the oracle, never the serializer. Round-tripping this
    block through `yaml.safe_dump` would reorder the keys, drop the comment, and
    re-quote the values — the whole reason the writer is a line edit."""
    text = (
        "---\n"
        "title: 'restore --- review'  # keep this comment\n"
        "tags: [a, b]\n"
        "status: in-review\n"
        "owner: amelia\n"
        "---\n\nbody\n"
    )
    spec = _spec(tmp_path, text)
    assert frontmatter.set_frontmatter_status(spec, "done") is True
    assert spec.read_bytes().decode() == text.replace("status: in-review", "status: done")
    fm = frontmatter.read_frontmatter(spec)
    assert fm["status"] == "done" and fm["tags"] == ["a", "b"]
    assert list(fm) == ["title", "tags", "status", "owner"]  # order survives
    assert yaml.safe_dump(fm) not in spec.read_text()  # ...and it is not a dump
