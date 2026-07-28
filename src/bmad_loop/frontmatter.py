"""Pure spec-frontmatter parsing: read the YAML ``---``…``---`` block, normalize
the status token, and rewrite ``status:`` in place.

Zero git/subprocess dependencies (only stdlib + PyYAML) so pure domain modules
(``stories``, ``devcontract``) can read spec status without importing ``verify``
and dragging in its whole git surface (assessment finding F-1). ``verify``
re-exports these names, so every existing ``verify.<name>`` / ``from .verify
import <name>`` call site stays valid.
"""

from __future__ import annotations

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


def set_frontmatter_status(path: Path, status: str) -> bool:
    """Rewrite the `status:` field in a spec's `---`…`---` frontmatter block.

    A minimal in-place line replacement (not a YAML round-trip) so the spec's
    formatting, comments, and field order survive — only the status value
    changes. Returns True when the file was rewritten, False when it has no
    frontmatter or already carries `status`. Idempotent.
    """
    if not path.is_file():
        return False
    text = path.read_text(encoding="utf-8")
    split = _split_frontmatter(text)
    if split is None:
        return False
    before, block, after = split
    block_lines = block.splitlines(keepends=True)
    replaced = False
    for i, line in enumerate(block_lines):
        stripped = line.lstrip()
        if stripped.startswith("status:") and not stripped.startswith("status_"):
            indent = line[: len(line) - len(stripped)]
            newline = "\n" if line.endswith("\n") else ""
            block_lines[i] = f"{indent}status: {status}{newline}"
            replaced = True
            break
    if not replaced:
        return False
    rebuilt = before + "".join(block_lines) + after
    if rebuilt == text:  # already at the target value — idempotent no-op
        return False
    path.write_text(rebuilt, encoding="utf-8")
    return True
