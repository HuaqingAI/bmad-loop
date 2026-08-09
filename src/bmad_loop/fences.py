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


def _delimits(marker: str, rest: str) -> bool:
    """Whether a matched line delimits a fence at all, or is ordinary text.

    CommonMark forbids a backtick anywhere in the info string of a BACKTICK fence,
    and only there — the rule exists so that inline code is not read as opening a
    block. Tilde fences carry no such restriction. Checked here rather than folded
    into `FENCE_LINE_RE` so the pattern stays one readable alternation instead of
    two near-identical arms with different info-string classes.

    The miss runs the wrong way for `deferredwork`: a line of prose quoting a
    fence opens a block CommonMark never opens, and a real `gate:` above the next
    closing run is then read as an example and silently stops gating — the lost
    gate the field exists to prevent, not the spurious refusal it tolerates.
    """
    return marker[0] == "~" or "`" not in rest


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
    return any(
        s <= offset < e for s, e in fenced_spans(text, unclosed_hides_rest=unclosed_hides_rest)
    )


def fenced_spans(text: str, *, unclosed_hides_rest: bool = True) -> list[tuple[int, int]]:
    """Half-open ``[start, end)`` ranges of ``text`` that sit inside a fenced block.

    The walk itself, which `fenced()` reduces to one offset and
    `deferredwork.parse_legacy` blanks out wholesale before scanning line by line.
    Keeping it here is the point of the module: a reader that needs the ranges and
    a reader that needs one answer must not disagree about where a block ends.

    The bounds follow the delimiters' roles rather than their extents. A span opens
    one character past the opener's line start, so the opener itself reads as
    outside the block — it is markup that a scanner may still want to see. It ends
    one character past the closer's line start, which puts the closer *inside*: the
    scanners this serves anchor at column 0, and a closing delimiter is the one
    line of a block that can never be mistaken for the content it terminates.
    """
    spans: list[tuple[int, int]] = []
    open_marker: str | None = None
    start = 0
    for m in FENCE_LINE_RE.finditer(text):
        marker, rest = m.group(1), m.group(2)
        if not _delimits(marker, rest):
            continue  # inline code, not a fence line
        if open_marker is None:
            open_marker, start = marker, m.start() + 1  # opener — an info string is allowed
        elif marker[0] == open_marker[0] and len(marker) >= len(open_marker) and not rest.strip():
            spans.append((start, m.start() + 1))  # valid closing fence
            open_marker = None
        # else: a shorter / mismatched / info-bearing fence line — literal content
    if open_marker is not None and unclosed_hides_rest:
        # Past the last offset, not up to it: an unclosed fence has no end, and
        # `fenced()` answered True for an offset at or beyond `len(text)` before
        # these ranges existed. Slicing clamps, so a mask is unaffected either way.
        spans.append((start, len(text) + 1))
    return spans
