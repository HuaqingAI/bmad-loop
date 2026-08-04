"""Deterministic reading and editing of the deferred-work ledger.

The ledger (`{implementation_artifacts}/deferred-work.md`) is append-only
markdown in the canonical form documented at
bmad-loop-sweep/deferred-work-format.md: `### DW-<seq>: <title>` headings with
`origin:`/`location:`/`reason:`/`status:` field lines. The inner bmad-dev-auto
session appends flatter entries that the orchestrator normalizes on sweep. The
orchestrator never trusts an LLM to have edited it — status flips and decision
records happen here, and gates re-read the file from disk.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date as calendar_date
from pathlib import Path

from .platform_util import atomic_write_text

HEADING_RE = re.compile(r"^### (DW-\d+): (.+?)\s*$", re.MULTILINE)
ANY_HEADING_RE = re.compile(r"^#{1,6} ", re.MULTILINE)
# The flat appender's opening line, in the two forms this module needs it: as a
# bullet in the raw ledger (FLAT_ENTRY_RE, the canonical-span boundary in
# parse_ledger) and as bullet *content* after `_BULLET_RE` has stripped the
# marker (`_FLAT_SOURCE_RE`, legacy section). One shape, two anchors — they have
# to agree, or a block the legacy parser recognizes stays invisible to it (#304).
# Keyed on the opening line alone, deliberately: also requiring the block's
# `summary:`/`evidence:` lines would narrow the boundary below the parser's own
# recognition, leaving the bug in place for every partial shape it accepts.
_FLAT_SOURCE_BODY = r"source_spec:[ \t]"
FLAT_ENTRY_RE = re.compile(rf"^[-*][ \t]+{_FLAT_SOURCE_BODY}", re.IGNORECASE | re.MULTILINE)
STATUS_RE = re.compile(r"^status:[ \t]*(.*)$", re.MULTILINE)
# Everything `str.splitlines()` splits on, not `\n` alone (#305). The writers
# below interpolate their arguments into a line-oriented file, so a break in a
# value injects ledger lines. The C1/Unicode members are load-bearing rather
# than decorative: `parse_legacy` scans with `splitlines()` while `parse_ledger`
# matches with `re.MULTILINE`, so a U+2028 splits an entry for one reader and is
# invisible to the other — the two then disagree about what the ledger says.
LINE_BREAK_RE = re.compile(r"[\n\r\v\f\x1c-\x1e\x85\u2028\u2029]+")
# The writers' date shape. Deliberately a separate literal from the legacy
# parser's `_DATE_TOKEN_RE`, which happens to look similar today: that one
# decides whether a freeform heading is a dated section, and tightening what the
# orchestrator will *write* must never quietly retune what `parse_legacy` reads.
# Spelled `[0-9]` rather than `\d`, which also matches Arabic-Indic, fullwidth
# and mathematical digit forms — the ledger's readers understand none of them.
_ISO_DATE_RE = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}")


@dataclass(frozen=True)
class DWEntry:
    id: str
    title: str
    status: str  # the status field value, "" when the line is missing
    body: str  # full entry text including the heading
    span: tuple[int, int]  # char offsets of the entry in the ledger text

    @property
    def open(self) -> bool:
        return self.status.split()[0] == "open" if self.status else False


def parse_ledger(text: str) -> list[DWEntry]:
    """Extract DW entries; non-conforming sections are skipped, an entry
    without a status line parses with status "" (not open)."""
    entries = []
    headings = list(HEADING_RE.finditer(text))
    for i, m in enumerate(headings):
        end = headings[i + 1].start() if i + 1 < len(headings) else len(text)
        # an entry also ends at any intervening heading (e.g. a "## Deferred
        # from:" section header between freeform and DW-format content)
        other = ANY_HEADING_RE.search(text, m.end(), end)
        if other:
            end = other.start()
        # ...and at a flat appender block, which belongs to no canonical entry
        # (#304). This span is what parse_legacy() masks out before scanning, so
        # absorbing the block hides the finding from every reader of the ledger.
        # Searched from the entry's own `status:` line, never from above it:
        # truncating over the status leaves the entry reading as neither open nor
        # done (open_ids() drops it, classify() calls it malformed), which trades
        # one lost flat block for one lost tracked entry. An entry with no status
        # line has nothing to protect, so the whole span is fair game.
        status_m = STATUS_RE.search(text, m.end(), end)
        flat = FLAT_ENTRY_RE.search(text, status_m.end() if status_m else m.end(), end)
        if flat:
            end = flat.start()
        body = text[m.start() : end]
        status_m = STATUS_RE.search(body)
        entries.append(
            DWEntry(
                id=m.group(1),
                title=m.group(2),
                status=status_m.group(1).strip() if status_m else "",
                body=body,
                span=(m.start(), end),
            )
        )
    return entries


def open_ids(text: str) -> set[str]:
    return {e.id for e in parse_ledger(text) if e.open}


def parse_declaration(raw: object) -> tuple[tuple[str, ...], str | None]:
    """The single reading of a ``closes_deferred:`` declaration (#234), shared by
    the ``stories.yaml`` parser, the engine's close hook, and ``validate``.

    Returns the normalized ids plus an error describing a wrong *container*.
    Missing / YAML-null is an empty declaration, not an error.

    Strict about the container, lenient about each item. A bare
    ``closes_deferred: DW-1`` is a schema error rather than a silently-wrapped
    single id — a string is iterable, so a lenient reading would quietly turn one
    id into a list of characters — while items are ``str()``-normalized and
    stripped, because an LLM-authored manifest may emit an unquoted ``DW-1`` as a
    string but a bare ``5`` as an int. Blanks drop and duplicates collapse
    (order-preserving): both are noise, not a contradiction.

    Callers decide the severity: the manifest parser raises, the engine journals,
    ``validate`` warns. What they must NOT do is disagree — before this, a wrong
    container was a hard schema error in ``stories.yaml`` and a silent empty
    declaration in frontmatter, so the same mistake either failed the parse or
    vanished depending on which file it was made in.

    Whether an id names a real entry is not decided here; that needs the ledger
    (:func:`classify`).
    """
    if raw is None:
        return (), None
    if not isinstance(raw, list):
        return (), f"must be a list of deferred-work ids (got {type(raw).__name__})"
    return tuple(dict.fromkeys(item for item in (str(x).strip() for x in raw) if item)), None


@dataclass(frozen=True)
class Declared:
    """How declared ids line up against one ledger snapshot (#234).

    Four outcomes, not two, because "not open" hides two very different cases.
    ``already_done`` is a satisfied declaration — a resume re-driving a close that
    already landed — and must stay silent. ``malformed`` is an entry that exists
    but carries neither an ``open`` nor a ``done`` status: nothing can be marked,
    and saying nothing would leave the operator believing it was.

    ``duplicates`` cross-cuts the other four: it names the declared ids the ledger
    carries more than once, whichever bucket they landed in. A duplicate id is a
    corrupt ledger (#286), and the entry this classification describes is only one
    of them — so the close is reported, never silent.
    """

    open_ids: tuple[str, ...] = ()
    already_done: tuple[str, ...] = ()
    unknown: tuple[str, ...] = ()
    malformed: tuple[str, ...] = ()
    duplicates: tuple[str, ...] = ()


def classify(text: str, ids: Sequence[str]) -> Declared:
    """Partition `ids` against a single ledger snapshot, preserving order.

    Classifying from a snapshot rather than from :func:`mark_done`'s return value
    is deliberate: that return conflates "already done" with "absent from the
    ledger", and those need opposite treatment (silence vs. a warning).

    **The FIRST entry of a duplicated id wins**, because that is the one
    :func:`_find_entry` — and so every mutation in this module — acts on. Indexing
    last-wins instead made the two disagree, and a ledger carrying one `DW-1` open
    and another done then closed nothing while saying nothing, in either order: a
    done-first ledger classified the id `open`, sent it to
    :func:`mark_done_many`, and had :func:`_apply_done` refuse the done copy it
    found first (marked nothing, so not even an unmatched warning); an open-first
    ledger classified it `already_done` and never attempted the write at all
    (#284 round-6 review, finding 4). The duplicate itself is reported through
    ``duplicates`` rather than swallowed — one id naming two entries is a fault
    about the ledger, not an answer about the work."""
    by_id: dict[str, DWEntry] = {}
    duplicated: set[str] = set()
    for e in parse_ledger(text):
        if e.id in by_id:
            duplicated.add(e.id)
            continue  # first wins: `_find_entry` mutates that one
        by_id[e.id] = e
    buckets: dict[str, list[str]] = {"open": [], "done": [], "unknown": [], "malformed": []}
    for dw_id in ids:
        entry = by_id.get(dw_id)
        if entry is None:
            buckets["unknown"].append(dw_id)
            continue
        word = entry.status.split()[0] if entry.status else ""
        buckets[word if word in ("open", "done") else "malformed"].append(dw_id)
    return Declared(
        open_ids=tuple(buckets["open"]),
        already_done=tuple(buckets["done"]),
        unknown=tuple(buckets["unknown"]),
        malformed=tuple(buckets["malformed"]),
        duplicates=tuple(dw_id for dw_id in dict.fromkeys(ids) if dw_id in duplicated),
    )


def _find_entry(text: str, dw_id: str) -> DWEntry | None:
    for entry in parse_ledger(text):
        if entry.id == dw_id:
            return entry
    return None


def _insert_after_status(text: str, entry: DWEntry, line: str) -> str:
    """Insert a field line right after the entry's status line (or at the end
    of the entry when no status line exists)."""
    status_m = STATUS_RE.search(entry.body)
    if status_m:
        pos = entry.span[0] + status_m.end()
        return text[:pos] + "\n" + line + text[pos:]
    insert_at = entry.span[0] + len(entry.body.rstrip())
    return text[:insert_at] + "\n" + line + text[insert_at:]


def _one_line(value: str) -> str:
    """Collapse every run of line-break characters in `value` to a single space.

    The whole of the #305 fix. These writers interpolate their arguments into a
    line-oriented file, so a value carrying a break mints a phantom
    `### DW-<n>` entry, truncates the entry's span at :data:`FLAT_ENTRY_RE` and
    re-surfaces the tail as a legacy item, or leaves the entry carrying two
    `status:` lines.

    Note what the last shape does *not* do: `STATUS_RE` takes the first match, so
    an injected `status:` never changes what `parse_ledger` reports. A test that
    asserts on `entry.status` therefore passes with this guard deleted — the
    observable is the line structure.

    Sanitizes; never raises, and nothing upstream rejects on a break either. The
    close paths call these writers bare (`sweep._close_resolved`,
    `decisions.apply_pre_answer`), so a `ValueError` would end the sweep as
    crashed; refusing the same text back at `validate_triage` only moved the
    stoppage to a pause. Collapsing is lossless enough — the ledger wants one
    line anyway — so this is the fix, and the skill docs are guidance that
    reduces occurrences without gating on them.

    A value with no break is returned **untouched**, so an existing ledger is
    never reformatted and a clean write is byte-identical to before the guard.
    The trailing `.strip()` removes all surrounding whitespace, not merely the
    space a leading or trailing break left behind — which is why it must stay on
    the far side of that fast path.

    A break-only value therefore sanitizes to `""`. Keeping it non-empty *here*
    could only yield bare whitespace, which trades an unfindable entry for an
    unidentifiable one, so each caller handles its own empties — and by two
    different strategies, which is why neither belongs in this helper.
    :func:`append_entry` **substitutes**, naming a vanished title
    `(untitled DW-<n>)` so the id it just burned stays findable.
    :func:`append_decision` **drops**, shedding the ` — ` separator along with an
    empty detail rather than promising one that is not there. Its `label` needs
    neither: every member of :data:`LINE_BREAK_RE` is `str.isspace()`, and
    `validate_triage` builds each `DecisionOption` with `.strip() or key`, so a
    break-only label has already become the option key before it arrives."""
    if not LINE_BREAK_RE.search(value):
        return value
    return LINE_BREAK_RE.sub(" ", value).strip()


def _require_iso_date(value: str) -> None:
    """Raise unless `value` is a strict ISO `YYYY-MM-DD` calendar date.

    Raising is right here and wrong for free text: `date` is orchestrator-owned
    (`Engine._today()`), never model-authored, so a bad value is a programmer
    bug. Letting it through writes a `status:` line that reads as neither open
    nor done, which `classify` reports as malformed and `open_ids` drops — the
    entry silently leaves the sweep's world.

    The regex is not redundant with `date.fromisoformat`: since 3.11 that also
    accepts `20260611` and ISO week dates, neither of which the ledger's own
    readers recognize, and it is the regex — via `[0-9]` — that pins the digits
    to ASCII. `fromisoformat` in turn rejects the well-shaped impossible day
    (`2026-02-30`) that no pattern can catch."""
    if not _ISO_DATE_RE.fullmatch(value):
        raise ValueError(f"date must be YYYY-MM-DD: {value!r}")
    try:
        calendar_date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"date must be YYYY-MM-DD: {value!r}") from exc


def _require_canonical_status(status: str) -> None:
    """Raise unless `status` is exactly `open` or `done YYYY-MM-DD`.

    Two halves with two different dependents. The *first word* is what
    :attr:`DWEntry.open` and :func:`classify` branch on, so anything but `open`
    or `done` makes an entry unreadable to both. The *date* is invisible to them
    — they read `status.split()[0]` and cannot tell `done 2026-02-30` from a real
    day — but it is not invisible downstream: the whole status value is carried
    verbatim to readers (the TUI's deferred pane, the `--json` projections), so a
    malformed date is rendered to a human as though it were one."""
    if status == "open":
        return
    if status.startswith("done "):
        _require_iso_date(status.removeprefix("done "))
        return
    raise ValueError(f"status must be 'open' or 'done YYYY-MM-DD': {status!r}")


def _operation_digest(operation_id: str) -> str:
    """Encode a stable close-operation id as one ledger-safe token."""
    if not operation_id:
        raise ValueError("operation_id must not be empty")
    return hashlib.sha256(operation_id.encode("utf-8")).hexdigest()


def _apply_done(
    text: str,
    dw_id: str,
    date: str,
    note: str,
    *,
    undo_owner: str | None = None,
) -> str | None:
    """Flip one entry to `status: done <date>` + a resolution note *within* `text`.
    None when the entry is missing or not open. The entry is re-located after the
    status rewrite because that edit shifts every later span offset.

    The note is sanitized here, at the point of interpolation, rather than on
    :func:`mark_done`: that is a one-id wrapper over :func:`mark_done_many`, which
    `Engine._apply_deferred_closes` calls directly, so a wrapper-side guard would
    never see a story close (#305). `date` is validated by the sole caller, at its
    entry, so the check does not depend on a ledger existing."""
    note = _one_line(note)
    entry = _find_entry(text, dw_id)
    if entry is None or not entry.open:
        return None
    status_m = STATUS_RE.search(entry.body)
    assert status_m is not None  # open implies a status line
    start = entry.span[0] + status_m.start()
    end = entry.span[0] + status_m.end()
    previous_status_line = status_m.group(0)
    if undo_owner is not None and LINE_BREAK_RE.search(previous_status_line):
        # An undo marker must never preserve a value that becomes more than one line
        # under the ledger readers' shared splitlines semantics. Standard closes
        # retain their existing behavior; the undo-capable path refuses the mark.
        return None
    done_status_line = f"status: done {date}"
    text = text[:start] + done_status_line + text[end:]
    entry = _find_entry(text, dw_id)
    assert entry is not None
    tail = f"resolution: {note}"
    if undo_owner is not None:
        # The owner digest makes this close distinguishable from an earlier run
        # that reused its human-readable note. The encoded prior line makes the
        # undo lossless for parser-accepted spacing and annotations. Hex keeps
        # every payload on one ASCII line, including Unicode annotations.
        previous_status_hex = previous_status_line.encode("utf-8").hex()
        tail += f"\nresolution-undo: {undo_owner} {date} {previous_status_hex}"
    return _insert_after_status(text, entry, tail)


def _mark_done_many(
    path: Path,
    dw_ids: Sequence[str],
    date: str,
    note: str,
    *,
    operation_id: str | None = None,
) -> list[str]:
    """Shared atomic implementation for the public close operations."""
    _require_iso_date(date)
    undo_owner = _operation_digest(operation_id) if operation_id is not None else None
    if not path.is_file():
        return []
    text = path.read_text(encoding="utf-8")
    marked: list[str] = []
    for dw_id in dw_ids:
        updated = _apply_done(text, dw_id, date, note, undo_owner=undo_owner)
        if updated is None:
            continue
        text = updated
        marked.append(dw_id)
    if not marked:
        return []
    atomic_write_text(path, text)
    return marked


def mark_done_many(path: Path, dw_ids: Sequence[str], date: str, note: str) -> list[str]:
    """Flip every entry in `dw_ids` to `status: done <date>` + a resolution note,
    in ONE read and ONE atomic write. Returns the ids actually flipped (missing
    and already-done ids are skipped), in the order given.

    All-or-nothing on purpose. A per-id read-modify-write loop leaves marks on
    disk when it raises partway through several ids — a half-applied closure the
    caller never gets to journal, so the ledger claims resolutions the run has no
    record of. Here a failure writes nothing, and the returned list is exactly
    what landed.

    The write goes through :func:`~bmad_loop.platform_util.atomic_write_text`
    rather than a bare tmp+replace: swapping a fresh inode over the ledger
    otherwise resets its mode (a ``0600`` ledger silently becoming world-readable)
    and turns a symlinked ledger into a regular file.

    ``date`` is validated before the ``is_file`` short-circuit so a programmer bug
    fails the same way whether or not a ledger happens to exist — a guard that
    only fires when the file is present is one an absent fixture hides."""
    return _mark_done_many(path, dw_ids, date, note)


def mark_done_many_reopenable(
    path: Path,
    dw_ids: Sequence[str],
    date: str,
    note: str,
    operation_id: str,
) -> list[str]:
    """Close entries atomically with a durable, operation-specific undo marker.

    ``operation_id`` must be stable and recomputable across crash/replay from
    already-persisted identity (for example ``run_id`` + ``story_key``), never an
    ephemeral random value. Only entries actually flipped receive its marker;
    skipped, already-done ids therefore cannot be reopened by this operation.

    The ordinary :func:`mark_done_many` deliberately emits no marker and retains
    its existing ledger format. Use this variant only for a transaction with a
    later rollback leg.
    """
    return _mark_done_many(path, dw_ids, date, note, operation_id=operation_id)


def mark_done(path: Path, dw_id: str, date: str, note: str) -> bool:
    """Flip one entry to `status: done <date>` and record a resolution note.
    Returns False (no write) when the entry is missing or already done."""
    return bool(mark_done_many(path, [dw_id], date, note))


_MARK_DONE_TAIL_RE = re.compile(
    r"\nresolution:[ \t]*(.*)"
    r"\nresolution-undo:[ \t]*([0-9a-f]{64})[ \t]+"
    r"([0-9]{4}-[0-9]{2}-[0-9]{2})[ \t]+([0-9a-f]+)$",
    re.MULTILINE,
)


def mark_open(path: Path, dw_id: str, note: str, operation_id: str) -> bool:
    """Undo one close written by :func:`mark_done_many_reopenable`.

    The entry must still carry the operation's adjacent resolution and undo-marker
    lines. A standard or earlier close has no matching marker and cannot be
    reopened merely because it reused the same human-readable note.
    """
    undo_owner = _operation_digest(operation_id)
    if not path.is_file():
        return False
    text = path.read_text(encoding="utf-8")
    entry = _find_entry(text, dw_id)
    if entry is None or entry.open:
        return False
    status_m = STATUS_RE.search(entry.body)
    if status_m is None:
        # parse_ledger deliberately tolerates status-less entries. This primitive
        # is later called from _defer, where an AttributeError would crash the run
        # instead of completing the deferral.
        return False
    try:
        _require_canonical_status(entry.status)
    except ValueError:
        # Only a canonical status written by mark_done is eligible for undo.
        # Preserve malformed or human-authored statuses for validation/reporting.
        return False
    res_m = _MARK_DONE_TAIL_RE.match(entry.body, status_m.end())
    if res_m is None:
        return False
    if res_m.group(1).strip() != _one_line(note).strip() or res_m.group(2) != undo_owner:
        return False
    if status_m.group(0) != f"status: done {res_m.group(3)}":
        return False
    try:
        previous_status_line = bytes.fromhex(res_m.group(4)).decode("utf-8")
    except (UnicodeDecodeError, ValueError):
        return False
    if LINE_BREAK_RE.search(previous_status_line):
        return False
    previous_status_m = STATUS_RE.fullmatch(previous_status_line)
    previous_status = previous_status_m.group(1).strip() if previous_status_m else ""
    if not previous_status or previous_status.split()[0] != "open":
        return False
    start = entry.span[0] + status_m.start()
    end = entry.span[0] + res_m.end()
    atomic_write_text(path, text[:start] + previous_status_line + text[end:])
    return True


def append_decision(path: Path, dw_id: str, date: str, label: str, detail: str) -> bool:
    """Record a human decision on an entry without changing its status.

    `label` and `detail` come from a triage session's `DecisionOption`, so they
    are sanitized to one line rather than refused — see :func:`_one_line`. This
    is also where a build option's `intent` gets flattened, since it reaches the
    ledger only as `detail = option.resolution or option.intent`.

    Precondition: `date` is ISO `YYYY-MM-DD`; anything else raises `ValueError`,
    checked before the ``is_file`` short-circuit so an absent ledger cannot hide
    the bug."""
    _require_iso_date(date)
    if not path.is_file():
        return False
    text = path.read_text(encoding="utf-8")
    entry = _find_entry(text, dw_id)
    if entry is None:
        return False
    label = _one_line(label)
    # Sanitize before the emptiness test, never after: a break-only detail
    # collapses to "" and must then drop the separator with it, or the entry
    # carries a dangling `— ` promising a detail that is not there.
    detail = _one_line(detail)
    detail_part = f" — {detail}" if detail else ""
    text = _insert_after_status(text, entry, f"decision: {date} {label}{detail_part}")
    path.write_text(text, encoding="utf-8")
    return True


DW_ID_RE = re.compile(r"\bDW-(\d+)\b")


def next_seq(text: str) -> int:
    """The next free DW sequence number — one past the highest DW-<n> anywhere
    in the ledger (malformed entries included, so a number is never reused and
    the sweep numbering check stays satisfied)."""
    nums = [int(m.group(1)) for m in DW_ID_RE.finditer(text)]
    return (max(nums) + 1) if nums else 1


def field_line_present(body: str, field: str, value: str) -> bool:
    """True when `body` has a `field:` line whose value is exactly `value`,
    matching the shapes append_entry writes (plain, or backtick-wrapped as for
    `source_spec:`). Anchored per-line so an incidental substring elsewhere in
    the body (e.g. inside `reason:`) never counts as a match."""
    v = re.escape(value)
    return re.search(rf"(?m)^{re.escape(field)}:[ \t]*`?{v}`?[ \t]*$", body) is not None


def append_entry(
    path: Path,
    *,
    title: str,
    origin: str,
    source_spec: str,
    reason: str,
    location: str = "n/a",
    status: str = "open",
    severity: str | None = None,
) -> str | None:
    """Append a new canonical `### DW-<seq>` entry numbered past the highest
    existing DW id, returning the new id (e.g. "DW-42").

    Idempotent: returns None without writing when an open entry already carries
    the same `origin:` marker and `source_spec:` — so re-running the same defer
    (e.g. a second sweep of the same story) never duplicates the entry. Creates
    the ledger (and parent dir) if it does not yet exist.

    Free text is sanitized (:func:`_one_line`) **before** the idempotence scan,
    which compares the caller's value against the stored one via
    :func:`field_line_present`: sanitizing afterwards would compare a raw value
    against a sanitized line, so every replay of the same multiline defer would
    miss its own entry and append another. `status` and `severity` are
    orchestrator-owned enumerations and raise instead."""
    _require_canonical_status(status)
    # The whitelist is derived from the legacy parser's alias table (defined
    # below; resolved at call time) so what this writer emits and what
    # `field_severity` normalizes to cannot drift apart.
    if severity and severity not in _CANONICAL_SEVERITIES:
        raise ValueError(f"severity must be one of {sorted(_CANONICAL_SEVERITIES)}: {severity!r}")
    given_title = bool(title)
    title = _one_line(title)
    origin = _one_line(origin)
    source_spec = _one_line(source_spec)
    reason = _one_line(reason)
    location = _one_line(location)
    text = path.read_text(encoding="utf-8") if path.is_file() else ""
    for entry in parse_ledger(text):
        if (
            entry.open
            and field_line_present(entry.body, "origin", origin)
            and field_line_present(entry.body, "source_spec", source_spec)
        ):
            return None
    dw_id = f"DW-{next_seq(text)}"
    if given_title and not title.strip():
        # A break-only title sanitizes to nothing, and `### DW-<n>: ` is a
        # heading `HEADING_RE`'s `(.+?)` does not match: the caller is handed an
        # id no reader can find while `next_seq` has already burned it.
        #
        # Tested with `.strip()`, not `not title`: a title of `"  "` carries no
        # break at all, so `_one_line` returns it unchanged by the byte-identity
        # fast path and it stays truthy. It parses, but renders blank in
        # `status`, `--json` and the TUI — the unidentifiable half of the same
        # problem, reached without ever touching the sanitizer.
        #
        # Scoped to a title that *had* content: an already-empty one keeps its
        # long-standing behavior, and the invariant is about non-empty values.
        title = f"(untitled {dw_id})"
    lines = [
        f"### {dw_id}: {title}",
        f"origin: {origin}",
        f"location: {location}",
        f"source_spec: `{source_spec}`",
    ]
    if severity:
        lines.append(f"severity: {severity}")
    lines.append(f"reason: {reason}")
    lines.append(f"status: {status}")
    block = "\n".join(lines) + "\n"
    # exactly one blank line between the previous content and the new entry
    if text == "" or text.endswith("\n\n"):
        sep = ""
    elif text.endswith("\n"):
        sep = "\n"
    else:
        sep = "\n\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text + sep + block, encoding="utf-8")
    return dw_id


# ------------------------------------------------------------------- legacy
#
# Ledgers written before the DW format (older BMAD-method projects) are
# freeform markdown: "## Deferred from: ..." sections holding id'd or
# strikethrough bullets, "### D-1.2-003: title — RESOLVED" entry headings,
# topic sections closed with "(... — DONE)". parse_legacy() reads them
# tolerantly so the TUI can display them and a sweep can migrate them; the
# strict DW contract above is untouched — legacy items have no status line
# to flip, so mark_done/open_ids never see them.

# Severity is extracted forgivingly (the ledger is LLM-written): a
# `severity:`/`priority:` field line in any case, plain or bold-bulleted
# ("- **Severity:** high"), common synonyms accepted.
SEVERITY_ALIASES = {
    "critical": "critical",
    "blocker": "critical",
    "high": "high",
    "major": "high",
    "medium": "medium",
    "med": "medium",
    "moderate": "medium",
    "low": "low",
    "minor": "low",
    "trivial": "low",
}
# What every alias above normalizes to, and so the only values `append_entry` may
# write. Derived rather than restated: a hand-copied whitelist drifts the moment
# an alias is added for a new canonical level.
_CANONICAL_SEVERITIES = frozenset(SEVERITY_ALIASES.values())

SEVERITY_FIELD_RE = re.compile(
    r"^[ \t]*(?:[-*][ \t]+)?(?:\*\*)?(?:severity|priority)[ \t]*:[ \t]*(?:\*\*)?[ \t]*"
    r"([A-Za-z][\w-]*)",
    re.IGNORECASE | re.MULTILINE,
)


def field_severity(body: str) -> str | None:
    m = SEVERITY_FIELD_RE.search(body)
    return SEVERITY_ALIASES.get(m.group(1).lower()) if m else None


@dataclass(frozen=True)
class LegacyEntry:
    key: str  # stable content-derived identity, unique within the file
    id: str  # native id ("W2", "D-CAP-001", "0-1"), "" when the item has none
    title: str  # cleaned one-line title (markers/strikethrough stripped)
    done: bool
    severity: str | None  # normalized critical/high/medium/low, None unknown
    body: str  # the bullet/heading block verbatim
    section: str  # enclosing ##/### heading text, "" at top level
    span: tuple[int, int]  # char offsets in the ledger text


_DONE_WORDS = r"(?:DONE|RESOLVED|CLOSED|VERIFIED|DOCUMENTED|FIXED)"
_LINE_HEADING_RE = re.compile(r"^(#{1,6})[ \t]+(.*?)[ \t]*$")
# a single whitespace-free digit-bearing token before ":" or "—" makes a
# heading an entry ("### D-CAP-001: title", "## D-8.6-001 — title");
# "## Epic 0: ..." has a space and "## 2026-06-09 — ..." is a date, so: section
_ENTRY_HEADING_RE = re.compile(r"^(~~)?([^\s:*~]*\d[^\s:*~]*)(?::[ \t]+|[ \t]+[—–][ \t]+)(.+)$")
_DATE_TOKEN_RE = re.compile(r"\d{4}-\d{2}-\d{2}")
_SECTION_DONE_RE = re.compile(rf"(?:—|–|-|\()[ \t]*{_DONE_WORDS}\b[^)]*\)?[ \t]*$")
_TITLE_DONE_SUFFIX_RE = re.compile(rf"[ \t]*(?:—|–|-)[ \t]*{_DONE_WORDS}[ \t]*$")
_BARE_DONE_SUFFIX_RE = re.compile(rf"[ \t]*{_DONE_WORDS}\b.*$")
_BOLD_DONE_RE = re.compile(rf"\*\*{_DONE_WORDS}\b[^*]*\*\*")
_BRACKET_DONE_RE = re.compile(rf"\[{_DONE_WORDS}\]")
_DONE_PREFIX_RE = re.compile(rf"^\*\*{_DONE_WORDS}\b[^*]*\*\*:?[ \t]*")
# "- W-1.2-c — CLOSED: ..." / "CLOSED 2026-06-11 (story 1.11). ..."
_LEAD_DONE_RE = re.compile(rf"^{_DONE_WORDS}\b")
_LEAD_DONE_STRIP_RE = re.compile(
    rf"^{_DONE_WORDS}\b(?:[ \t]+\d{{4}}-\d{{2}}-\d{{2}})?(?:[ \t]*\([^)]*\))?[ \t]*[:.—–-]?[ \t]*"
)
# The generic bmad-dev-auto review appender writes a flat block per finding:
#   - source_spec: `spec-foo.md`
#     summary: <one sentence>
#     evidence: <why this is real>
# We recognize it so the `summary` becomes the title (not the source_spec path)
# and the entry migrates cleanly into the canonical `### DW-<seq>` shape. The
# opening line comes from the same `_FLAT_SOURCE_BODY` as FLAT_ENTRY_RE, which
# bounds canonical spans on it — see that constant for why they must not drift.
_FLAT_SOURCE_RE = re.compile(rf"^{_FLAT_SOURCE_BODY}", re.IGNORECASE)
_FLAT_SUMMARY_RE = re.compile(r"^[ \t]*summary:[ \t]*(.*)$", re.IGNORECASE | re.MULTILINE)
_BULLET_RE = re.compile(r"^[-*][ \t]+(.*)$")
_ITEM_ID_RE = re.compile(
    r"^(?:\*\*)?([^\s:*~]*\d[^\s:*~]*)(?:\*\*)?(?:[ \t]*[—–][ \t]+|:[ \t]+|[ \t]+-[ \t]+)"
)
_BRACKET_TOKEN_RE = re.compile(r"\[([A-Za-z]+)[^\]]*\]")
_LEAD_BOLD_RE = re.compile(r"^\*\*(.+?)\*\*")
_STRUCK_LINE_RE = re.compile(r"^~~(.*)~~")
_TRAIL_BRACKET_RE = re.compile(r"[ \t]*\[[^\]]+\][ \t.]*$")


def _bracket_severity(s: str) -> str | None:
    for m in _BRACKET_TOKEN_RE.finditer(s):
        sev = SEVERITY_ALIASES.get(m.group(1).lower())
        if sev:
            return sev
    return None


def _clean_title(s: str) -> str:
    return " ".join(s.replace("**", "").split())


def _item_entry(first: str, body: str, section: str, section_done: bool) -> dict:
    """Interpret one bullet item; returns the pre-key entry fields."""
    if _FLAT_SOURCE_RE.match(first):
        # generic bmad-dev-auto flat appender block: title is the `summary`
        sm = _FLAT_SUMMARY_RE.search(body)
        summary = sm.group(1).strip() if sm else ""
        return {
            "id": "",
            "title": _clean_title(summary) if summary else _clean_title(first),
            "done": section_done,
            "severity": field_severity(body),
            "section": section,
        }
    content = first
    struck = False
    m = _STRUCK_LINE_RE.match(content)
    if m:  # "~~text~~ DONE" / "~~text~~ → resolution" on the first line
        struck = True
        content = m.group(1)
    elif content.startswith("~~") and "~~" in body[2:]:
        struck = True  # strikethrough closes on a later line
        content = content[2:]
    item_id = ""
    m = _ITEM_ID_RE.match(content)
    if m:
        item_id = m.group(1)
        content = content[m.end() :]
    done = (
        struck
        or section_done
        or bool(_LEAD_DONE_RE.match(content))
        or bool(_BOLD_DONE_RE.search(body))
        or bool(_BRACKET_DONE_RE.search(body))
    )
    content = _DONE_PREFIX_RE.sub("", content)
    content = _LEAD_DONE_STRIP_RE.sub("", content)
    while True:  # trailing "[MINOR]" / "[CLOSED]" tokens are not title text
        trimmed = _TRAIL_BRACKET_RE.sub("", content)
        if trimmed == content:
            break
        content = trimmed
    bold = _LEAD_BOLD_RE.match(content)
    if bold and len(bold.group(1).split()) >= 3:
        title = bold.group(1)  # notey: the bold phrase is the title
    else:
        title = content
    return {
        "id": item_id,
        "title": _clean_title(title),
        "done": done,
        "severity": _bracket_severity(body) or field_severity(body),
        "section": section,
    }


def _heading_entry(struck: bool, hid: str, rest: str, body: str, section: str) -> dict:
    """Interpret one '### D-1: title' entry heading (story-maker shape)."""
    title = rest
    done = struck
    m = _TITLE_DONE_SUFFIX_RE.search(title)
    if m:
        done = True
        title = title[: m.start()]
    if struck:
        title = _BARE_DONE_SUFFIX_RE.sub("", title.replace("~~", ""))
    return {
        "id": hid,
        "title": _clean_title(title),
        "done": done,
        "severity": field_severity(body) or _bracket_severity(body),
        "section": section,
    }


def parse_legacy(text: str) -> list[LegacyEntry]:
    """Extract legacy (non-DW) deferred items. Canonical DW entries are
    masked out first, so mixed ledgers parse both ways without overlap."""
    masked = text
    for e in parse_ledger(text):
        s, t = e.span
        masked = masked[:s] + re.sub(r"[^\n]", " ", masked[s:t]) + masked[t:]

    found: list[tuple[dict, tuple[int, int]]] = []
    section = ""
    section_done = False
    # a done section with no items yet: emitted as its own done entry unless
    # bullets, an entry heading, or a deeper child heading claim it first
    pending: dict | None = None  # {"level", "fields", "span"}
    item: dict | None = None  # accumulating bullet or entry heading

    def close_item(end: int) -> None:
        nonlocal item
        if item is None:
            return
        body = text[item["start"] : end].rstrip()
        span = (item["start"], item["start"] + len(body))
        if item["kind"] == "item":
            fields = _item_entry(item["first"], body, item["section"], item["section_done"])
        else:
            fields = _heading_entry(
                item["struck"], item["hid"], item["rest"], body, item["section"]
            )
        found.append((fields, span))
        item = None

    def emit_pending() -> None:
        nonlocal pending
        if pending is not None:
            found.append((pending["fields"], pending["span"]))
            pending = None

    offset = 0
    for line in text.splitlines(keepends=True):
        masked_line = masked[offset : offset + len(line)].rstrip("\n")
        hm = _LINE_HEADING_RE.match(masked_line)
        if hm:
            level = len(hm.group(1))
            close_item(offset)
            if level == 1:
                emit_pending()
                section, section_done = "", False
            elif level in (2, 3):
                if pending is not None and level > pending["level"]:
                    pending = None  # a child heading: the parent is structure
                else:
                    emit_pending()
                em = _ENTRY_HEADING_RE.match(hm.group(2))
                if em and _DATE_TOKEN_RE.fullmatch(em.group(2)):
                    em = None  # "## 2026-06-09 — ..." is a dated section
                if em:
                    pending = None
                    item = {
                        "kind": "heading",
                        "start": offset,
                        "struck": bool(em.group(1)),
                        "hid": em.group(2),
                        "rest": em.group(3),
                        "section": section,
                        "section_done": section_done,
                    }
                else:
                    htext = hm.group(2)
                    struck = htext.startswith("~~") and "~~" in htext[2:]
                    section = _clean_title(htext.replace("~~", ""))
                    section_done = struck or bool(_SECTION_DONE_RE.search(htext))
                    if section_done:
                        pending = {
                            "level": level,
                            "span": (offset, offset + len(line.rstrip("\n"))),
                            "fields": {
                                "id": "",
                                "title": section,
                                "done": True,
                                "severity": None,
                                "section": "",
                            },
                        }
            offset += len(line)
            continue
        if item is not None and item["kind"] == "heading":
            if masked_line.strip() == "---" or (masked_line.strip() == "" and line.strip() != ""):
                close_item(offset)  # rule, or a masked canonical entry
            offset += len(line)
            continue
        bm = _BULLET_RE.match(masked_line)
        if bm:
            close_item(offset)
            pending = None
            item = {
                "kind": "item",
                "start": offset,
                "first": bm.group(1),
                "section": section,
                "section_done": section_done,
            }
        elif masked_line.strip() in ("", "---"):
            # a masked canonical entry reads as blank: it still bounds the item
            if masked_line.strip() == "---" or line.strip() != masked_line.strip():
                close_item(offset)
        elif masked_line[0] in " \t":
            pass  # indented continuation of the current item
        else:
            close_item(offset)  # column-0 prose ends an item, emits nothing
        offset += len(line)
    close_item(len(text))
    emit_pending()

    entries: list[LegacyEntry] = []
    counts: dict[str, int] = {}
    for fields, span in found:
        base = hashlib.sha1(
            f"{fields['section']}\0{fields['id'] or fields['title']}".encode(),
            usedforsecurity=False,  # display/identity key, not a credential
        ).hexdigest()[:10]
        n = counts.get(base, 0) + 1
        counts[base] = n
        entries.append(
            LegacyEntry(
                key=base if n == 1 else f"{base}-{n}",
                id=fields["id"],
                title=fields["title"],
                done=fields["done"],
                severity=fields["severity"],
                body=text[span[0] : span[1]],
                section=fields["section"],
                span=span,
            )
        )
    return entries


def has_legacy(text: str) -> bool:
    return bool(parse_legacy(text))
