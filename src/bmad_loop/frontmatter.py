"""Pure spec-frontmatter parsing: read the YAML ``---``…``---`` block, normalize
the status token, and rewrite ``status:`` in place.

Zero git/subprocess dependencies (only stdlib + PyYAML) so pure domain modules
(``stories``, ``devcontract``) can read spec status without importing ``verify``
and dragging in its whole git surface (assessment finding F-1). ``verify``
re-exports these names, so every existing ``verify.<name>`` / ``from .verify
import <name>`` call site stays valid. (The same docstring rule bars importing
``platform_util`` here — it pulls in ``subprocess`` — so this module's writes use
``write_text``, not ``atomic_replace``.)

READER AND WRITER DEGRADE IN OPPOSITE DIRECTIONS, deliberately. `read_frontmatter`
turns an unparseable or undecodable block into ``{}``: it runs on the observation
path, where the orchestrator's job is to classify what it finds, and every status
gate then reads ``""`` and answers with a clean retry. `set_frontmatter_status`
runs on the *repair* path and does the opposite — when it can see a status it
cannot safely rewrite, it RAISES `FrontmatterWriteError`. That is AGENTS.md's
"observation may degrade, repair writes must raise", and it is what makes a
``False`` return mean one thing only: there was nothing to change.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import yaml


def _split_frontmatter(text: str) -> tuple[str, str, str] | None:
    """Split a document into ``(before, block, after)`` around its YAML
    frontmatter, where ``before + block + after == text`` exactly.

    The opening and closing ``---`` are recognized ONLY as standalone delimiter
    lines (``line.rstrip() == "---"``), so a ``---`` substring inside a scalar
    value (e.g. ``title: 'restore --- review'``) is never mistaken for the
    closing boundary — the flaw a plain ``text.split("---", 2)`` has. ``before``
    is the opening delimiter line, ``block`` is the YAML content between the
    delimiters, and ``after`` begins with the closing delimiter line; callers
    rewrite ``block`` and reconstruct the file byte-for-byte. Returns ``None``
    when the text has no opening delimiter line or no closing delimiter line.
    """
    lines = text.splitlines(keepends=True)  # "".join(lines) == text
    if not lines or lines[0].rstrip() != "---":
        return None
    for i in range(1, len(lines)):
        if lines[i].rstrip() == "---":
            return lines[0], "".join(lines[1:i]), "".join(lines[i:])
    return None


def read_frontmatter(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        # A non-UTF-8 file carries no readable frontmatter — degrade exactly like
        # unparseable YAML below. Every status gate then reads status "" and
        # returns a clean retry/repair outcome instead of crashing mid-verify
        # (UnicodeDecodeError is a ValueError, so it slipped past callers'
        # except-OSError guards).
        return {}
    split = _split_frontmatter(text)
    if split is None:
        return {}
    try:
        doc = yaml.safe_load(split[1])
    except yaml.YAMLError:
        return {}
    return doc if isinstance(doc, dict) else {}


def status_of(fm: dict[str, Any]) -> str:
    """Normalized spec status from a frontmatter dict: stripped + lowercased.

    The single point all spec-frontmatter status gates read through, so casing
    never decides a gate — the spec template and sprint-status tokens are
    lowercase, so a stray ``Done``/``In-Review`` from a hand-edited spec still
    matches. (``devcontract`` keeps its own lowercasing; it parses skill-written
    prose where casing genuinely varies.)
    """
    return str(fm.get("status", "")).strip().lower()


def operator_actions_of(fm: dict[str, Any]) -> tuple[str, ...]:
    """The external, human-only actions a spec's ``operator_actions:`` frontmatter
    declares — normalized, order-preserving, deduped.

    Lives here rather than in ``devcontract`` because both sides of the park
    contract need the same reading and ``verify`` cannot import ``devcontract``
    (the dependency runs the other way).

    Strict about the container, lenient about each scalar item — the
    ``closes_deferred`` reading (:func:`deferredwork.parse_declaration`), for the
    same reason: a bare ``operator_actions: buy the domain`` is iterable, so a
    lenient container reading would silently turn one instruction into a list of
    characters. Items that are themselves containers are dropped rather than
    stringified: ``[{action: ..., check: ...}]`` is the deliberate v2 shape (a
    per-action verification command), and ``str()``-ing it would hand a human a
    line of Python repr as their instruction. ``None`` items drop for the same
    reason — ``str(None)`` is the word "None", not an action.

    Every malformed shape therefore collapses to ``()``, which the verify gates
    read as "declared nothing" and answer with one fixable retry naming the
    expected shape — a park is *defined* by owing at least one action, so an
    empty reading can never be mistaken for a valid park.
    """
    raw = fm.get("operator_actions")
    if not isinstance(raw, list):
        return ()
    items = (str(x).strip() for x in raw if x is not None and not isinstance(x, (list, dict)))
    return tuple(dict.fromkeys(a for a in items if a))


class FrontmatterWriteError(Exception):
    """A frontmatter block carries a key the reader can see but no minimal line
    edit can safely rewrite.

    Raised, not returned, because ``False`` already means "nothing to change" and
    no caller reads the return value — a ``False`` refusal would be invisible at
    every call site, which is the failure this exists to stop rather than
    relocate. `runs.rearm_escalation` translates it to `RearmError` (already
    surfaced by the CLI and the TUI); `cli.cmd_confirm` catches it and names the
    recoverable state; anything else lands on `cli.main`'s backstop as a clean
    one-liner."""


# A frontmatter ``<key>:`` line. Anchored, so only indentation may precede it —
# a `#` comment line and a `- ` list item are structurally excluded rather than
# excluded by a special case. Matching quotes around the key (`"status": x`),
# and whitespace before the colon (`status : x`), because YAML accepts both and
# the old `lstrip().startswith("status:")` scan silently skipped them. The
# `status_note:` exclusion is structural too: after the key the pattern demands
# a quote-or-whitespace-or-colon, and `_` is none of those.
def _key_line_re(key: str) -> re.Pattern[str]:
    return re.compile(rf"^[ \t]*(?P<q>['\"]?){re.escape(key)}(?P=q)[ \t]*:")


_STATUS_KEY_RE = _key_line_re("status")

# Builds the replacement for a matched key line, line ending included. A hook
# rather than one fixed rendering because the two writers on this helper preserve
# deliberately different things: `set_frontmatter_status` drops the value's quotes
# and any trailing comment (its callers read the result back as a bare
# `status: done`), while `devcontract.reset_spec_status` keeps both (its own tests
# pin them). What they share is the VERIFICATION, which is the part that was
# wrong in all three writers — not the formatting, which each already had right.
LineRenderer = Callable[[str, "re.Match[str]", str], str]


def _replace_value(line: str, m: re.Match[str], value: str) -> str:
    """Keep everything through the colon verbatim — indent, a quoted key,
    whitespace before the colon — then ``<space><value>``.

    The gap after the colon is normalized rather than preserved because a bare
    ``status:`` has none to preserve, and filling it would write ``status:done``,
    which is not the key at all."""
    return f"{line[: m.end()]} {value}" + ("\n" if line.endswith("\n") else "")


def _edit_frontmatter_block(
    block: str,
    key: str,
    value: str,
    *,
    pattern: re.Pattern[str] | None = None,
    render: LineRenderer = _replace_value,
    insert: bool = False,
) -> str | None:
    """Rewrite ``<key>:`` inside a frontmatter block, verifying the edit MEANS
    what it was supposed to mean. Returns the new block, None when there is
    nothing to change, and raises `FrontmatterWriteError` otherwise.

    Enumerating the shapes a line scan must not touch is a losing game — a flow
    mapping, a block scalar, a value continued on the next line, an anchor
    another key aliases, a nested key of the same name, the key quoted inside
    ANOTHER key's literal block. Each of those had the old scanner either write
    nothing or corrupt the spec, and every enumeration written for this (mine and
    both reviewers') missed shapes the next one caught.

    So this does not widen a pattern. It makes the trial edit, re-parses it with
    ``yaml.safe_load`` as an ORACLE, and keeps it only if the block still parses
    as a mapping, its top-level ``key`` is exactly ``value``, and **every other
    key is unchanged**. YAML is never used as a serializer — the edit stays the
    formatting-preserving single-line replacement — so one gate replaces six and
    it also rejects shapes nobody has enumerated. The other-keys comparison is
    the half no pattern-widening design has: it is what catches an edit with the
    right effect on ``key`` and a wrong effect somewhere else.

    Candidates are ITERATED, not broken on at the first match. That is what fixes
    the wrong-target write: a decoy line inside another key's literal block fails
    verification and the real key is still reached.

    The parse is of the block the READER would see, so the two agree on what is
    there. An unparseable block raises rather than returning None: the reader
    degrades it to ``{}`` because observation may, but a writer that concluded
    "no status here" from a block it could not read would report success for a
    spec it never touched.

    ``insert`` adds the key as the block's last line when the reader sees no such
    top-level key — what `verify.set_frontmatter_field` and
    `devcontract.reset_spec_status` need and `set_frontmatter_status` must never
    do. It is gated on the READER's view rather than on a scan miss, which is the
    other half of the same defect: a scan that missed a quoted key then appended
    a SECOND one, and the file ended up with two."""
    pattern = _key_line_re(key) if pattern is None else pattern
    try:
        original = yaml.safe_load(block)
    except yaml.YAMLError as e:
        raise FrontmatterWriteError(
            f"the frontmatter block does not parse as YAML, so a {key!r} edit "
            f"cannot be verified ({e.__class__.__name__}: {e})"
        ) from e
    rest = {k: v for k, v in original.items() if k != key} if isinstance(original, dict) else {}
    if not isinstance(original, dict) or key not in original:
        if not insert:
            return None  # the reader sees no such top-level key — nothing to change
        nl = "\r\n" if block.endswith("\r\n") else "\n"
        return _verified(block + f"{key}: {value}{nl}", key, value, rest)
    if original[key] == value:
        return None  # already at the target — idempotent no-op, no write
    lines = block.splitlines(keepends=True)
    for i, line in enumerate(lines):
        m = pattern.match(line)
        if m is None:
            continue
        trial = list(lines)
        trial[i] = render(line, m, value)
        candidate = _verified("".join(trial), key, value, rest)
        if candidate is not None:
            return candidate
    raise FrontmatterWriteError(
        f"the frontmatter carries {key!r} in a shape no in-place line edit can "
        f"safely rewrite to {value!r} (a flow mapping, a block scalar, a value "
        f"continued on the next line, or an anchor another key aliases) — set it "
        f"as a plain `{key}: <value>` line and re-run"
    )


def _verified(candidate: str, key: str, value: str, rest: dict[str, Any]) -> str | None:
    """``candidate`` if it means what the edit intended, else None.

    Three conditions, and the third is the one no pattern-widening design has:
    the block still parses as a mapping, its top-level ``key`` is exactly
    ``value``, and every OTHER key is unchanged. Without the last one an edit
    with the right effect on ``key`` and a wrong effect elsewhere passes — a
    ``status`` merged in from an anchor block is only reachable by rewriting the
    anchor, which is shared state the story does not own."""
    try:
        parsed = yaml.safe_load(candidate)
    except yaml.YAMLError:
        return None  # the edit broke the block — this line is not a scalar key
    if not isinstance(parsed, dict) or parsed.get(key) != value:
        return None  # edited something that is not the key the reader resolves
    if {k: v for k, v in parsed.items() if k != key} != rest:
        return None  # right key, collateral damage — e.g. another key's block
    return candidate


def set_frontmatter_status(path: Path, status: str) -> bool:
    """Rewrite the `status:` field in a spec's `---`…`---` frontmatter block.

    A minimal in-place line replacement (not a YAML round-trip) so the spec's
    formatting, comments, and field order survive — only the status value
    changes, and the edit is verified by re-parsing before it lands (see
    `_edit_frontmatter_block`).

    Returns True when the file was rewritten. Returns False for **nothing to
    change** only: no file, no frontmatter block, no top-level `status` for
    `read_frontmatter` to see, or already at the target. Raises
    `FrontmatterWriteError` when the reader CAN see a status the edit cannot
    safely move — `False` never means "I failed".

    Two deliberate non-preservations, both pinned in tests/test_frontmatter.py:
    the value's own quotes are dropped (`status: 'x'` -> `status: done`), because
    the standard fixture shape is quoted and callers read the result back
    unquoted; and a trailing inline comment on the status line is dropped,
    because knowing where the scalar ends is exactly the guesswork this rewrite
    exists to stop making.
    """
    if not path.is_file():
        return False
    text = path.read_text(encoding="utf-8")
    split = _split_frontmatter(text)
    if split is None:
        return False
    before, block, after = split
    edited = _edit_frontmatter_block(block, "status", status, pattern=_STATUS_KEY_RE)
    if edited is None:
        return False
    path.write_text(before + edited + after, encoding="utf-8")
    return True
