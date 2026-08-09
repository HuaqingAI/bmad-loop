"""Whether a markdown offset sits inside a fenced code block.

A leaf module with no bmad-loop imports, deliberately: the two readers that need
this — `devcontract` (is a `## Auto Run Result` heading real, or quoted?) and
`deferredwork` (is a `gate:` line a declaration, or an example?) — sit on opposite
sides of an import edge (`devcontract` imports `deferredwork`), so neither can
host it for the other. `devcontract._section_headings` already argued the case in
prose: "a second copy of `_fenced`'s open-marker walk is exactly the kind of
near-duplicate that drifts." This module is that argument taken one step further
once a second subsystem needed the same walk.
"""

from __future__ import annotations

import re

# A fence line: up to three spaces of indent, then a maximal run of >= 3 backticks
# or tildes (its char AND length both matter per CommonMark), then the rest of the
# line — an info string on an opener; on a close, only whitespace is allowed.
FENCE_LINE_RE = re.compile(r"^ {0,3}(`{3,}|~{3,})([^\n]*)$", re.MULTILINE)


def fenced(text: str, offset: int, *, unclosed_hides_rest: bool = True) -> bool:
    """True when ``offset`` falls inside a ``` / ~~~ fenced code block.

    A fence opens on a line of three-or-more backticks or tildes (indentable up
    to three spaces; a tab would make an indented code block instead). Per
    CommonMark it closes only on a later line using the SAME character, at least
    as long as the opener, with no trailing non-whitespace — so a shorter run, a
    different fence char, or an info-bearing line inside the block is content,
    not a close. Tracking the open fence's char+length (not a bare line-parity
    count) is what stops a nested-or-mismatched inner fence from flipping state
    early and exposing a quoted heading as a real one.

    ``unclosed_hides_rest`` decides the one case CommonMark leaves to the reader:
    a fence that opens and never closes. The two callers need opposite answers,
    and both are choosing the direction where being wrong is survivable, so this
    is a parameter rather than a policy:

    - ``True`` (``devcontract``): everything after the opener is content. Reading
      a quoted ``## Auto Run Result`` as a real section is a *destructive* misread
      — it strips or terminates a spec — so an ambiguous tail must stay inert.
    - ``False`` (``deferredwork``): the opener is ordinary text. A `gate:` line
      below a stray fence must keep gating, because a gate lost in silence is the
      exact failure that field exists to end; a spurious refusal in an entry whose
      markdown is already malformed is the cheaper wrong answer.
    """
    open_marker: str | None = None
    for m in FENCE_LINE_RE.finditer(text):
        if m.start() >= offset:
            break
        marker, rest = m.group(1), m.group(2)
        if open_marker is None:
            open_marker = marker  # opening fence — an info string is allowed
        elif marker[0] == open_marker[0] and len(marker) >= len(open_marker) and not rest.strip():
            open_marker = None  # valid closing fence
        # else: a shorter / mismatched / info-bearing fence line — literal content
    if open_marker is None:
        return False
    return unclosed_hides_rest or _closes_later(text, offset, open_marker)


def _closes_later(text: str, offset: int, open_marker: str) -> bool:
    """Whether the fence open at ``offset`` is ever validly closed after it.

    Only consulted under ``unclosed_hides_rest=False``, and only when the offset
    is inside an open fence — so the walk above has already paid for the prefix
    and this pays for the remainder exactly once per query.
    """
    for m in FENCE_LINE_RE.finditer(text, offset):
        marker, rest = m.group(1), m.group(2)
        if marker[0] == open_marker[0] and len(marker) >= len(open_marker) and not rest.strip():
            return True
    return False
