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
    if not text.startswith("---"):
        return {}
    parts = text.split("---", 2)
    if len(parts) < 3:
        return {}
    try:
        doc = yaml.safe_load(parts[1])
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
    if not text.startswith("---"):
        return False
    parts = text.split("---", 2)
    if len(parts) < 3:
        return False
    block_lines = parts[1].splitlines(keepends=True)
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
    rebuilt = parts[0] + "---" + "".join(block_lines) + "---" + parts[2]
    if rebuilt == text:  # already at the target value — idempotent no-op
        return False
    path.write_text(rebuilt, encoding="utf-8")
    return True
