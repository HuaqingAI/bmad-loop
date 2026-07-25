"""Translate the generic `bmad-dev-auto` skill's output into the orchestrator's
result.json contract.

Alex Verhovsky's upstream `bmad-dev-auto` skill (BMAD-METHOD PR #2500) is a
decoupled autonomous-coding primitive: it writes NO result.json. Its outcome
lives in the spec it produced — `status:` in the frontmatter (the machine-
consumable signal) plus an appended `## Auto Run Result` prose section (intended
for an LLM deciding how to handle failure). This module is the thin Python shim
that turns that on-disk spec into the legacy result dict that verify.py /
escalation.py already consume, so the rest of the pipeline stays unchanged.

DOCTRINE — never trust prose for a gate. The frontmatter `status:` read straight
off disk is authoritative; the `## Auto Run Result` prose is only used to route
the blocked→PAUSE decision and to carry a human-readable detail. Where the two
disagree we surface it (`status_consistent=False`) so the caller can fail safe
(treat a mismatch as a retry rather than silently proceeding). Every real
deterministic gate (git baseline, worktree-changed, sprint advancement, dw_id
match) still runs in verify.py against actual on-disk state.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .platform_util import atomic_replace
from .verify import DEV_WORKFLOW, read_frontmatter

# The section the skill appends on EVERY terminal path (success and blocked),
# per its step-02/03/04 finalize instructions. Its presence is our completion
# marker on the spec-watch fallback; the `Status:` line within it is the only
# field we parse structurally — everything else is free prose.
AUTO_RUN_HEADING_RE = re.compile(r"^##\s+Auto Run Result\s*$", re.MULTILINE)
# `Status:` possibly bulleted ("- Status: blocked") / bolded ("**Status:** done"),
# case-insensitive on the label, value is the first token on the line.
STATUS_LINE_RE = re.compile(
    r"^\s*(?:[-*]\s*)?(?:\*\*)?status(?:\*\*)?\s*:\s*(?:\*\*)?\s*([A-Za-z-]+)",
    re.IGNORECASE | re.MULTILINE,
)

# Terminal frontmatter statuses the skill can leave behind.
DONE = "done"
BLOCKED = "blocked"

# The status a plan-halt dispatch leaves behind: under folder+id dispatch a
# `Halt after planning.` directive makes the skill HALT right after the
# Ready-for-Development gate, at `ready-for-dev`. This is a *successful* terminal
# outcome for that leg (the plan is done, awaiting human review / implementation)
# — but ONLY when the caller asked for the halt. Without the directive the same
# status is a died-mid-flight non-terminal (it stays in RECONCILABLE_FROM), so
# the seam is gated on `plan_halt`, never on the status alone.
PLAN_HALT_STATUS = "ready-for-dev"

# Frontmatter statuses a half-finalized generic spec may be reconciled FROM when
# its prose terminal `## Auto Run Result` Status is `done`. Deliberately an
# allowlist: anything else (already-`done`, `blocked`, or an unknown custom token)
# is left untouched, so reconciliation can never override a status the skill set on
# purpose. `""` covers a blank or missing frontmatter `status:` — `reset_spec_status`
# fills/inserts the line in that case. `in-review` is included because step-04 sets
# it transiently at the start of a review pass; the skill self-finalizes to `done`,
# so a spec left AT `in-review` with a prose `done` result is a mid-review interrupt
# safe to reconcile forward. The intent-gap patch-restore re-drive (BMAD-METHOD
# #2564) deliberately re-arms the spec TO `in-review` before a session (so step-01
# routes straight to step-04 on the restored diff) — but that is the pre-session
# status the re-driven skill then advances past; it never LEAVES a spec at
# `in-review` as a terminal, so the reconcile allowlist semantics are unchanged.
RECONCILABLE_FROM = frozenset({"", "draft", "ready-for-dev", "in-progress", "in-review"})

# The leading `---\n …frontmatter… \n---` block, captured in three parts so the
# body can be rewritten while the fences stay byte-identical.
_FRONTMATTER_RE = re.compile(r"\A(---\r?\n)(.*?\r?\n)(---[ \t]*\r?\n)", re.DOTALL)
# A frontmatter `status:` line, preserving indent, the `: ` gap, optional quotes,
# and any trailing inline comment. Only the value token is rewritten. The value is
# `*` (not `+`) so a present-but-empty status (`status:` / `status: ""`) is matched
# and filled — a bmad-dev-auto template can leave it blank.
_FM_STATUS_RE = re.compile(
    r"^(?P<pre>[ \t]*status[ \t]*:[ \t]*)(?P<q>['\"]?)(?P<val>[A-Za-z-]*)(?P=q)(?P<rest>.*)$",
    re.MULTILINE,
)

# The skill's no-spec fallback artifact (HALT when {spec_file} is unknown/missing):
# `{implementation_artifacts}/bmad-dev-auto-result-<slug-or-timestamp>.md`. It
# carries a terminal frontmatter `status:` but no `## Auto Run Result` heading.
FALLBACK_RESULT_PREFIX = "bmad-dev-auto-result-"


@dataclass(frozen=True)
class AutoRunResult:
    """Parsed `## Auto Run Result` section. `present` is False when the spec has
    no such section yet (the session has not reached a terminal step)."""

    present: bool
    status: str  # lowercased Status: value, or "" when absent/unparsed
    detail: str  # the prose body after the heading, trimmed (human-readable)


# A fence line: up to three spaces of indent, then a maximal run of >= 3 backticks
# or tildes (its char AND length both matter per CommonMark), then the rest of the
# line — an info string on an opener; on a close, only whitespace is allowed.
_FENCE_LINE_RE = re.compile(r"^ {0,3}(`{3,}|~{3,})([^\n]*)$", re.MULTILINE)


def _fenced(text: str, offset: int) -> bool:
    """True when `offset` falls inside a ``` / ~~~ fenced code block.

    A fence opens on a line of three-or-more backticks or tildes (indentable up
    to three spaces; a tab would make an indented code block instead). Per
    CommonMark it closes only on a later line using the SAME character, at least
    as long as the opener, with no trailing non-whitespace — so a shorter run, a
    different fence char, or an info-bearing line inside the block is content,
    not a close. Tracking the open fence's char+length (not a bare line-parity
    count) is what stops a nested-or-mismatched inner fence from flipping state
    early and exposing a quoted `## Auto Run Result` as a real heading — a
    destructive misread on the strip path."""
    open_marker: str | None = None
    for m in _FENCE_LINE_RE.finditer(text):
        if m.start() >= offset:
            break
        marker, rest = m.group(1), m.group(2)
        if open_marker is None:
            open_marker = marker  # opening fence — an info string is allowed
        elif marker[0] == open_marker[0] and len(marker) >= len(open_marker) and not rest.strip():
            open_marker = None  # valid closing fence
        # else: a shorter / mismatched / info-bearing fence line — literal content
    return open_marker is not None


def _section_headings(text: str) -> list[re.Match[str]]:
    """`AUTO_RUN_HEADING_RE` matches that are real section headings. A heading
    quoted inside a fenced code block (a frozen intent showing an example of the
    terminal section, a log excerpt) is documentation, not structure — treating
    it as terminal would let such a spec read as a result artifact from the
    agent's first save (#52)."""
    return [m for m in AUTO_RUN_HEADING_RE.finditer(text) if not _fenced(text, m.start())]


def _next_heading_start(text: str, offset: int) -> int:
    """Offset of the first non-fenced same-level (`## `) heading at/after
    `offset`, or end-of-text — the shared section boundary. Fenced `## ` lines
    inside the section (quoted shell comments, log output) are content, not
    boundaries."""
    for nxt in re.finditer(r"^##\s", text, re.MULTILINE):
        if nxt.start() >= offset and not _fenced(text, nxt.start()):
            return nxt.start()
    return len(text)


def parse_auto_run_result(text: str) -> AutoRunResult:
    """Tolerantly extract the trailing `## Auto Run Result` section from a spec.

    Reads the LAST real (non-fenced) such heading (the finalize step appends; a
    re-derivation loop could in principle append more than one — the last is the
    live outcome) and pulls its `Status:` value plus the remaining prose as
    detail, spanning to the next real same-level heading.
    """
    matches = _section_headings(text)
    if not matches:
        return AutoRunResult(present=False, status="", detail="")
    last = matches[-1]
    body = text[last.end() : _next_heading_start(text, last.end())]
    status_m = STATUS_LINE_RE.search(body)
    status = status_m.group(1).strip().lower() if status_m else ""
    return AutoRunResult(present=True, status=status, detail=body.strip())


@dataclass(frozen=True)
class SynthResult:
    """A synthesized result.json plus the cross-check signal. `result_json` is
    None when the spec has not terminated yet (no `## Auto Run Result` and no
    terminal frontmatter status), i.e. nothing to translate."""

    result_json: dict[str, Any] | None
    status_consistent: bool


def _read_text_or_empty(path: Path) -> str:
    """Read a spec on the *read-back* path, degrading an unreadable file to "".

    An absent, binary/truncated, or unreadable spec carries no parseable result
    section, so it reads exactly like a spec that has not terminated yet — the
    caller then waits, nudges, or keeps its verdict. Never a crash: this runs on
    the observation path, where the orchestrator's job is to classify what it
    finds, not to trust it. (UnicodeDecodeError is a ValueError, not an OSError.)

    The *repair* path deliberately does the opposite — `reset_spec_status` and
    `strip_auto_run_result` let an unreadable spec raise, because silently
    skipping a rewrite leaves the spec in a state the caller believes it fixed.
    """
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return ""


def synthesize_result(
    spec_path: Path,
    *,
    story_key: str | None,
    dw_ids: list[str] | None = None,
    plan_halt: bool = False,
) -> SynthResult:
    """Build the legacy result dict from the generic skill's on-disk spec.

    Returns ``SynthResult(None, True)`` when the spec carries no terminal signal
    yet (caller should keep waiting / treat the session as not-yet-complete).
    The dict's ``workflow`` is forged to ``auto-dev`` so verify.py's anti-wrong-
    skill guard passes; ``baseline_commit`` is taken from the skill's
    ``baseline_revision`` frontmatter (its name for the same thing). A blocked
    outcome is rendered as a single CRITICAL escalation so ``decide_dev`` PAUSEs
    unchanged — the generic skill has no severity tiers, and per the integration
    decision every block maps to PAUSE.

    ``plan_halt`` is the stories-mode expected-terminal seam: on the first leg of
    a ``spec_checkpoint`` dispatch the caller sends ``Halt after planning.`` and
    the skill HALTs at ``ready-for-dev``. Passing ``plan_halt=True`` treats that
    status as a *successful* terminal — the returned dict carries
    ``status="ready-for-dev"``, no escalation, and a ``plan_halt=True`` marker so
    verify/engine expect a planned (not implemented) spec. Without ``plan_halt``
    the default is unchanged: ``ready-for-dev`` is non-terminal (died mid-flight)
    and returns ``SynthResult(None, True)``. This composes with the engine's
    ``_reconcile_generic_terminal_status`` — that path only reconciles a spec
    whose prose ``## Auto Run Result`` says ``done`` while the frontmatter lags,
    so a plan-halt ``ready-for-dev`` (no such prose) is never reconciled to
    ``done`` and this leg's success outcome is not clobbered.
    """
    try:
        fm = read_frontmatter(spec_path)
    except OSError:
        # Same degrade as `_read_text_or_empty` below, for the same reason: this is
        # the read-back path. An unreadable spec is not evidence a session
        # finished, so treat it exactly like one that has not terminated yet — the
        # caller keeps polling, and a fault that persists past the grace window
        # lands as a stall/timeout verdict that `_post_kill_reconcile` can still
        # rescue. Crashing here would take the whole run down for a spec the CLI
        # merely had open for writing.
        return SynthResult(result_json=None, status_consistent=True)
    fm_status = str(fm.get("status", "")).strip().lower()
    arr = parse_auto_run_result(_read_text_or_empty(spec_path))

    terminal = (DONE, BLOCKED, PLAN_HALT_STATUS) if plan_halt else (DONE, BLOCKED)
    # Not terminal yet: no result section AND frontmatter not at a terminal state.
    if not arr.present and fm_status not in terminal:
        return SynthResult(result_json=None, status_consistent=True)

    # Authoritative status = frontmatter (read off disk). Prose status only
    # cross-checks it. When the prose is present and disagrees, flag it.
    status = fm_status or arr.status
    consistent = (not arr.present) or (not arr.status) or (arr.status == status)

    # The skill names the baseline `baseline_revision`; verify reads `baseline_commit`.
    baseline = str(fm.get("baseline_commit", fm.get("baseline_revision", ""))).strip()

    escalations: list[dict[str, Any]] = []
    if status == BLOCKED or arr.status == BLOCKED:
        detail = arr.detail or "generic dev session reported a blocked outcome"
        escalations.append({"type": "blocked", "severity": "CRITICAL", "detail": detail[:2000]})

    result: dict[str, Any] = {
        "workflow": DEV_WORKFLOW,
        "story_key": story_key,
        "spec_file": str(spec_path),
        "baseline_commit": baseline,
        "status": status,
        "escalations": escalations,
    }
    if dw_ids:
        result["dw_ids"] = list(dw_ids)
    # bmad-dev-auto (BMAD-METHOD PR #2505) self-reviews inline and, on a `done`
    # exit, sets `followup_review_recommended: true` when its review-driven
    # changes warrant an independent second-opinion pass. The skill never sets it
    # on a blocked exit, so only carry it through on `done`.
    if status == DONE:
        result["followup_review_recommended"] = bool(fm.get("followup_review_recommended", False))
    # Mark the clean plan-halt success so verify/engine expect a planned spec
    # (status ready-for-dev, no implementation work). Never marked when a block
    # escalation is present — that routes to PAUSE, not a plan-review pause.
    if plan_halt and status == PLAN_HALT_STATUS and not escalations:
        result["plan_halt"] = True
    return SynthResult(result_json=result, status_consistent=consistent)


def find_result_artifact(impl_artifacts: Path, *, since_ns: int) -> Path | None:
    """Spec-watch fallback: locate THIS session's output artifact.

    This is how the GenericDevAdapter acquires its result: the generic skill
    writes no result.json, so on the session's Stop event we locate the spec it
    produced. The common case is a `spec-*.md` carrying a terminal `## Auto Run
    Result` section (appended by the skill's HALT on success AND blocked, when a
    spec exists). The skill's no-spec fallback — `bmad-dev-auto-result-*.md`,
    written when intent was too unclear to even create a spec — carries a
    terminal frontmatter `status:` but NO `## Auto Run Result` heading, so it is
    matched by filename instead. Scans `impl_artifacts` for the most-recently-
    modified qualifying markdown modified at/after `since_ns` (the session launch
    floor, so a stale prior artifact can't be mistaken for this run's output).
    Returns None when nothing qualifies.
    """
    if not impl_artifacts.is_dir():
        return None
    best: tuple[int, Path] | None = None
    for path in impl_artifacts.glob("*.md"):
        try:
            mtime_ns = path.stat().st_mtime_ns
        except OSError:
            continue
        if not is_result_artifact(path, since_ns=since_ns):
            continue
        if best is None or mtime_ns > best[0]:
            best = (mtime_ns, path)
    return best[1] if best else None


def is_result_artifact(path: Path, *, since_ns: int) -> bool:
    """Whether ONE file qualifies as a session's result artifact — the per-path
    predicate `find_result_artifact` applies to each glob hit, factored out so a
    caller that already KNOWS which spec the session owed (the orchestrator hands
    the review session its path in the prompt) can test that one file instead of
    scanning a directory shared with every concurrent run (#261).

    Qualifies when the file was modified at/after ``since_ns`` (the session-launch
    floor) AND either is the by-name no-spec fallback (`bmad-dev-auto-result-*`,
    which carries no heading by design) or carries a real, non-fenced
    ``## Auto Run Result`` heading — a fence-quoted example must not qualify the
    spec (#52). An unreadable/undecodable candidate cannot be SHOWN to carry a
    terminal section, so it does not qualify (UnicodeDecodeError is a ValueError,
    so a bare ``except OSError`` let a torn mid-write spec crash the scan)."""
    try:
        if path.stat().st_mtime_ns < since_ns:
            return False
    except OSError:
        return False
    if path.name.startswith(FALLBACK_RESULT_PREFIX):
        return True
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return False
    return bool(_section_headings(text))


def find_frontmatter_candidates(impl_artifacts: Path, *, since_ns: int) -> list[Path]:
    """Missing-marker fallback scan (#224): specs this session finalized to a
    terminal frontmatter ``status:`` WITHOUT appending the ``## Auto Run Result``
    marker `find_result_artifact` keys on. The skill's HALT instructions make the
    append unconditional, but compliance is intermittent — without this scan such
    a spec is invisible to the harvest and a finished story rides stall-nudges to
    timeout, then loses its work to a retry/DEFER cycle.

    A candidate must be modified at/after `since_ns` (same session-launch floor
    as the marker scan), carry ZERO real (non-fenced) marker headings, not be the
    no-spec fallback file (that one is already matched by name on the normal
    path), and have frontmatter ``status:`` of ``done`` or ``blocked``. Returns
    ALL matches, most-recent first — the caller refuses to guess between several
    and must apply its own stability fingerprint before synthesizing, because a
    terminal frontmatter under a live window is weaker evidence than the marker.
    """
    if not impl_artifacts.is_dir():
        return []
    found: list[tuple[int, Path]] = []
    for path in impl_artifacts.glob("*.md"):
        try:
            mtime_ns = path.stat().st_mtime_ns
        except OSError:
            continue
        if not is_frontmatter_candidate(path, since_ns=since_ns):
            continue
        found.append((mtime_ns, path))
    found.sort(key=lambda t: t[0], reverse=True)
    return [p for _, p in found]


def is_frontmatter_candidate(path: Path, *, since_ns: int) -> bool:
    """Whether ONE file qualifies for the missing-marker fallback — the per-path
    predicate `find_frontmatter_candidates` applies to each glob hit, factored out
    for the same reason as `is_result_artifact`: a caller holding the spec path the
    session actually owed can test that file alone instead of a shared directory
    (#261).

    Qualifies when the file is modified at/after ``since_ns``, is NOT the no-spec
    fallback (already matched by name on the marker path), carries ZERO real
    (non-fenced) marker headings — one would put it in `find_result_artifact`'s
    territory — and has a frontmatter ``status:`` of ``done`` or ``blocked``. Any
    unreadable/undecodable read degrades to False, never an exception."""
    if path.name.startswith(FALLBACK_RESULT_PREFIX):
        return False
    try:
        if path.stat().st_mtime_ns < since_ns:
            return False
    except OSError:
        return False
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return False
    if _section_headings(text):
        return False  # carries a real marker — the normal scan's territory
    try:
        fm = read_frontmatter(path)
    except OSError:
        return False
    return str(fm.get("status", "")).strip().lower() in (DONE, BLOCKED)


def _atomic_write_spec(spec_path: Path, text: str) -> None:
    """Rewrite ``spec_path`` with ``text`` via a same-directory temp file + atomic
    rename, so an interrupted / short / disk-full write can never truncate the
    canonical spec — a failed repair must lose no work (fault injection on the old
    truncating ``write_text`` reduced a 46-byte spec to 12). Bytes are written
    verbatim (``write_bytes``, not ``write_text``): every caller here has already
    captured and preserved the file's own line endings, and ``write_text``'s
    ``newline=None`` default would re-translate ``\\n``→``\\r\\n`` on Windows. The
    ``.tmp`` sibling ends in ``.tmp`` (not ``.md``), so the ``*.md`` artifact scans
    never see it. On any failure the temp file is removed and the error re-raised —
    the callers impose best-effort, the writer never swallows."""
    tmp = spec_path.with_suffix(spec_path.suffix + ".tmp")
    try:
        tmp.write_bytes(text.encode("utf-8"))
        atomic_replace(tmp, spec_path)
    except BaseException:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise


def reset_spec_status(spec_path: Path, new_status: str) -> bool:
    """Rewrite the frontmatter ``status:`` value of a spec in place.

    Used by the generic-skill repair path: bmad-dev-auto self-finalizes a spec to
    ``done``/``in-review``, and its step-01 routes such a spec to "ingest as
    context, do not resume" — so to repair in place the orchestrator must re-open
    the spec by flipping its status back to ``in-progress``. A minimal line edit
    (not a YAML round-trip): preserves quote style and any trailing inline comment,
    and touches ONLY the first frontmatter block — never a ``Status:`` line in the
    prose body (e.g. the ``## Auto Run Result`` section). A present-but-empty status
    is filled, and a frontmatter block with NO ``status:`` line at all gets one
    inserted before the closing fence (the skill's template can leave it blank or
    absent). Returns True on a real change, False when the spec is absent, has no
    frontmatter block, or is already at ``new_status``."""
    if not spec_path.is_file():
        return False
    text = spec_path.read_text(encoding="utf-8")
    fm = _FRONTMATTER_RE.match(text)
    if not fm:
        return False
    head, body, tail = fm.group(1), fm.group(2), fm.group(3)
    changed = False

    def _repl(m: re.Match[str]) -> str:
        nonlocal changed
        if m.group("val") == new_status:
            return m.group(0)
        changed = True
        # Guarantee `key: value` spacing: a bare `status:` (no trailing space)
        # would otherwise fill to `status:done` — invalid YAML, the key is lost.
        pre = m.group("pre")
        if not pre.endswith((" ", "\t")):
            pre += " "
        # When the value was blank with a trailing inline comment, `rest` begins at
        # the `#`; abutting the value (`status: done# c`) makes the `#` part of the
        # scalar instead of a comment. Re-insert a separating space.
        rest = m.group("rest")
        if rest.startswith("#"):
            rest = " " + rest
        return f"{pre}{m.group('q')}{new_status}{m.group('q')}{rest}"

    if _FM_STATUS_RE.search(body):
        new_body = _FM_STATUS_RE.sub(_repl, body, count=1)
    else:
        # No status: line at all — insert one before the closing fence, matching
        # the block's line ending. `body` always ends with a newline (captured by
        # _FRONTMATTER_RE), so this lands on its own line.
        nl = "\r\n" if body.endswith("\r\n") else "\n"
        new_body = f"{body}status: {new_status}{nl}"
        changed = True
    if not changed:
        return False
    _atomic_write_spec(spec_path, head + new_body + tail + text[fm.end() :])
    return True


def strip_auto_run_result(spec_path: Path) -> bool:
    """Remove every ``## Auto Run Result`` section from a spec, in place.

    Companion to `reset_spec_status` on the re-drive path: re-opening a spec by
    flipping only its frontmatter would leave the stale terminal section behind,
    and `find_result_artifact` keys on that heading — the re-driven session's
    very first save of the spec would then qualify as a terminal result. Each
    section spans its heading to the next same-level heading (the shared
    `parse_auto_run_result` boundary) or end-of-file; headings quoted inside
    fenced code blocks are ignored on both ends. Returns True when a section was
    removed, False when the spec is absent or no section was present.

    Only an absent spec is guarded (a clean no-op, mirroring
    `verify.set_frontmatter_status`); a present-but-unreadable spec or a failing
    write is left to raise. Silently skipping the strip after the caller has
    already flipped the frontmatter status would leave the re-opened spec carrying
    its stale terminal section — the exact state that makes the re-driven session's
    first save read as a result — so that failure must surface, not be swallowed."""
    if not spec_path.is_file():
        return False
    text = spec_path.read_text(encoding="utf-8")
    matches = _section_headings(text)
    if not matches:
        return False
    kept: list[str] = []
    pos = 0
    for m in matches:
        if m.start() < pos:
            continue  # heading inside a section already being removed
        kept.append(text[pos : m.start()])
        pos = _next_heading_start(text, m.end())
    kept.append(text[pos:])
    _atomic_write_spec(spec_path, "".join(kept))
    return True


# Provenance stamped into a synthesized `## Auto Run Result` section so a human
# (or a later re-derivation) can tell an orchestrator-repaired marker from one
# the skill wrote itself. Single line (no internal newlines) so the writer's
# line-ending detection governs every break in the appended block.
ORCHESTRATOR_SYNTH_NOTE = (
    "_Appended by the bmad-loop orchestrator (missing-marker repair, #224): the "
    "session finalized this spec's frontmatter without its `## Auto Run Result` "
    "marker, so the orchestrator synthesized the result from the frontmatter and "
    "appended this section._"
)


def append_auto_run_result(spec_path: Path, status: str, *, detail: str = "") -> bool:
    """Append a synthesized ``## Auto Run Result`` marker section — the inverse of
    `strip_auto_run_result`.

    The missing-marker fallback (#224) synthesizes a session's result from a spec
    that carries a terminal frontmatter ``status:`` but never got the marker the
    skill owed; this writer brings that spec back into contract. Before the append
    a marker-less terminal spec sits in `find_frontmatter_candidates`' territory
    (it requires ZERO real headings); after it the spec carries exactly one real
    ``## Auto Run Result`` heading and moves into `find_result_artifact`'s
    (requires >= 1), so a later re-read is harvested on the normal marker path
    instead of the fallback scan, and the next review launch strips it exactly
    like a skill-written marker (#160).

    Returns False when the spec is absent, or when a REAL (non-fenced)
    ``## Auto Run Result`` heading is already present — idempotence, and the same
    #52 symmetry the strip honors: a fence-quoted heading (a frozen intent's
    example) is documentation, not a marker, so it does NOT block the append. A
    present-but-unreadable spec RAISES rather than no-ops (the repair-write
    doctrine `strip_auto_run_result` and `reset_spec_status` share — the CALLER
    imposes best-effort); silently skipping would leave the spec in a state the
    caller believes it repaired.

    Newline handling follows `reset_spec_status`'s intent: the file's line ending
    is detected (CRLF vs LF) and reused for the appended block, and a missing
    trailing newline is added so the heading can never glue onto the last body line
    (``...body## Auto Run Result``). The spec is read as raw bytes rather than via
    `read_text`, whose universal-newline translation would both hide a CRLF file's
    ending and silently rewrite its whole body to LF — an in-place repair must not
    mutate line endings it did not author. The section is ``## Auto Run Result`` /
    blank / ``Status: <status>`` / blank / the provenance note (plus an optional
    detail paragraph). ``status`` is normalized lowercase and MUST be the spec's
    own frontmatter ``status`` — the caller passes exactly that — so
    `synthesize_result`'s ``consistent`` cross-check holds on every later re-read."""
    if not spec_path.is_file():
        return False
    # Raw read (not read_text): preserve the file's exact line endings, and let an
    # undecodable spec raise UnicodeDecodeError like the strip's read (repair-write
    # doctrine — the caller imposes best-effort, never the writer).
    text = spec_path.read_bytes().decode("utf-8")
    if _section_headings(text):
        return False  # a real marker is already present — idempotent
    status = status.strip().lower()
    nl = "\r\n" if "\r\n" in text else "\n"
    # Ensure the heading lands after a newline the scan can recognize. A file
    # already ending in "\n" (LF, or the "\n" of a CRLF) is left byte-for-byte
    # intact, so a strip of this exact append round-trips. A file ending in a BARE
    # "\r" is the trap: it reads as "already terminated", but AUTO_RUN_HEADING_RE's
    # "^" (MULTILINE) only matches after "\n" — a heading glued directly after "\r"
    # is invisible to the scan. Complete such a bare CR to CRLF (preserving the
    # authored CR) so the heading begins on a recognized line boundary.
    if text.endswith("\r"):
        text += "\n"
    elif text and not text.endswith("\n"):
        text += nl
    section = f"## Auto Run Result{nl}{nl}Status: {status}{nl}{nl}{ORCHESTRATOR_SYNTH_NOTE}{nl}"
    if detail:
        section += f"{nl}{detail.strip()}{nl}"
    _atomic_write_spec(spec_path, text + section)
    return True
