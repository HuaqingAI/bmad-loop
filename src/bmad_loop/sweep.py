"""Deferred-work sweep: triage the ledger, decide, execute bundles.

A sweep is its own run type. One LLM triage session classifies every open
deferred-work entry (verified against actual code — ledger statuses are
unreliable); the orchestrator validates the result deterministically, asks
the human about decision items (interactive runs only), then drives each
work bundle through the inherited dev -> review -> verify -> commit pipeline.
The orchestrator performs all ledger edits it can do deterministically and
gates on the ones it delegates (verify.verify_review_bundle).
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Iterable, Literal, assert_never

from . import deferredwork, gates, verify
from .engine import Engine, RunPaused, _ArmedClose, _LedgerAnchor
from .escalation import critical_session_reason, env_fault_pause_reason, session_failure_reason
from .model import PAUSE_STORY_GATE, Phase, StoryTask, result_mapping
from .platform_util import (
    atomic_write_text,
    atomic_write_text_confined,
    neutralize_surrogates,
    safe_segment,
)
from .runs import StateRootError, _project_of_run_dir
from .statemachine import advance


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


TRIAGE_KEY = "sweep-triage"
TRIAGE_WORKFLOW = "deferred-sweep-triage"
MIGRATE_KEY = "sweep-migrate"
MIGRATE_WORKFLOW = "deferred-sweep-migrate"
BUNDLE_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9-]{1,39}\Z")
_BUNDLE_NAME_MAX_LENGTH = 40
_BUNDLE_NAME_INITIAL_CHARS = frozenset("abcdefghijklmnopqrstuvwxyz0123456789")
_BUNDLE_NAME_SAFE_CHARS = frozenset("abcdefghijklmnopqrstuvwxyz0123456789-")
# the inverse of SweepEngine._bundle_key: "dw-<name>" (cycle 1) / "dw<N>-<name>".
# A cycle-1 key always has "-" straight after "dw", so the cycle group matches
# empty and the split stays unambiguous even for a bundle named "2fix".
BUNDLE_KEY_RE = re.compile(r"^dw(\d*)-(.+)\Z")
# The token `_write_intent` emits and `_bundle_intent_reason` parses back out of
# a persisted intent.md. One definition so the writer and the grader cannot drift.
_INTENT_DW_IDS_PREFIX = "dw_ids: "
DECISION_EFFECTS = ("build", "close", "keep-open")
SEVERITY_ORDER = {"low": 0, "medium": 1, "high": 2, "critical": 3}
DW_ID_RE = re.compile(r"DW-\d+\Z")


def decimal_digits_key(value: str) -> tuple[int, str]:
    """Order arbitrary-length Unicode decimal text without converting to ``int``."""
    normalized = "".join(str(unicodedata.decimal(char)) for char in value)
    normalized = normalized.lstrip("0") or "0"
    return len(normalized), normalized


def increment_decimal_digits(value: str) -> str:
    """Normalize and increment arbitrary-length Unicode decimal text."""
    ascii_value = "".join(str(unicodedata.decimal(char)) for char in value)
    digits = list(ascii_value.lstrip("0") or "0")
    carry = 1
    for index in range(len(digits) - 1, -1, -1):
        if not carry:
            break
        if digits[index] == "9":
            digits[index] = "0"
        else:
            digits[index] = chr(ord(digits[index]) + 1)
            carry = 0
    if carry:
        digits.insert(0, "1")
    return "".join(digits)


# The two lines `_return_after_decisions` shows a human, as whole named constants
# joined explicitly. Assembling either at the call site out of adjacent string
# literals is what this avoids: a reflow can silently re-bind a trailing literal
# to one arm of a conditional, and these are the only strings that phase prints.
# The failure line does not lead with the success glyph — a `✓` in front of a
# miss reads as success at a glance — and it names the journal kind an operator
# greps for, since the answers really are saved and only the ledger is short.
_HANDBACK_TAIL = "sweep continues in the background"
_HANDBACK_RECORDED = " ".join(["✓ decisions recorded —", _HANDBACK_TAIL])
_HANDBACK_LEDGER_MISS = " ".join(
    [
        "! answers saved, but not every decision reached the deferred-work ledger",
        "(see sweep-decision-effect-unavailable in the journal) —",
        _HANDBACK_TAIL,
    ]
)
# The scalars a stored answer's consumers read as strings, split by WHO reads them.
# `_agreeing_option` runs on both `_materialize_bundles` lanes, so `key`/`label` are
# consumed whatever the effect; `intent`/`bundle_name` are read by the BUILD lane
# alone. Order within each tuple is the order a defect is reported in.
_ANSWER_STR_FIELDS = ("key", "label")
_BUILD_ANSWER_STR_FIELDS = ("intent", "bundle_name")


def unusable_answer_reason(value: Any, *, allow_close: bool) -> str | None:
    """Why `value` is not a usable persisted decision answer, or None when it is.

    ONE schema for the two readers of a stored answer — `SweepEngine._decisions_phase`'s
    read loops and `decisions.pending_missed_decisions` — because they have to agree
    (DW-142): a value one accepted and the other rejected was an id that either got
    silently ignored by every sweep while this command counted it answered, or the
    reverse. The returned string is the `malformed` row's reason, so it names ids,
    fields and TYPE names only, never a stored answer's prose.

    The checks are exactly what the consumers assume, no more — because rejecting is
    not free: an answer refused here stops being seeded, and for `keep-open` that
    means it stops SUPPRESSING bundles, so a later cycle can bundle and build work
    the human explicitly asked to leave alone. A field is therefore screened only
    where a reader of THIS effect actually consumes it.

    `effect` must be a recognized `DECISION_EFFECTS` member because
    `_materialize_bundles` routes on it and an unrecognized one matches no lane — the
    answer counts as given while nothing acts on it. `key` and `label`, when present,
    must be strings for every effect: `_agreeing_option` reads both and runs on both
    lanes. `intent` and `bundle_name` are screened for `build` ONLY, the sole lane
    that reads them — a list `intent` used to reach `Bundle.intent` as its truthy
    Python repr and ship into a dev session (DW-141), while the same corrupt field on
    a keep-open answer is inert prose no reader touches. Fields no reader consumes
    (`resolution`, `answered_at`) are not validated for any effect.

    `close` is usable for the RUN store (`allow_close=True`) even though
    `record_pre_answer` never stores it: the interactive writer in
    `_decisions_phase` records `effect: "close"` for a decision answered `close`
    this run, so rejecting it would re-ask a decision the human already answered
    inside the same run. That justification does NOT transfer to the project store
    (DW-147): `apply_pre_answer` applies a `close` to the LEDGER and deliberately
    skips `record_pre_answer`, so no legitimate producer writes one there — a
    `close` in `.bmad-loop/decisions.json` is hand-seeded or corrupt, and it used
    to be counted answered by every reader while matching NO `_materialize_bundles`
    lane (which needs `build` or `keep-open`), leaving the id never built, never
    closed and never re-offered. Hence the store, not the reader, selects the gate:
    `allow_close` is keyword-only and has no default, so a future reader has to
    state which store it is reading rather than inherit the wrong answer silently.

    Rejecting is never a repair — the caller keeps the value and re-publishes it
    unchanged; see `_decisions_phase`'s `unusable` map and `load_pre_answers`."""
    if not isinstance(value, dict):
        return f"not a JSON object: {type(value).__name__}"
    if "effect" not in value:
        return "effect missing"
    effect = value["effect"]
    if not isinstance(effect, str) or effect not in DECISION_EFFECTS:
        return "effect not recognized"
    if effect == "close" and not allow_close:
        return "effect close not accepted from this store"
    fields = _ANSWER_STR_FIELDS
    if effect == "build":
        fields += _BUILD_ANSWER_STR_FIELDS
    for field in fields:
        if field in value and not isinstance(value[field], str):
            return f"{field} not a string: {type(value[field]).__name__}"
    return None


def _answer_str(answer: dict[str, Any], field: str) -> str:
    """A stored answer's scalar read as a string, or "" when it is anything else.

    `str(answer.get(field, ""))` was the old spelling and it never failed: a list
    became "['a', 'b']" and a dict "{...}" — truthy prose no human authored, which
    the build lane then shipped as a `Bundle.intent` (DW-141). "" instead, so each
    site's EXISTING fallback chain handles it: an agreeing option's value, then the
    site's own default or its drop cause. No new branch, no new drop cause.

    Defense-in-depth, not the production path: `_decisions_phase` already rejects
    at the read site (`unusable_answer_reason`) every field a reader of that effect
    consumes, so in production each site here only ever sees strings. It stays
    because the test suite hands `_materialize_bundles` a map directly, past that
    read site — the same reason each lane holds its own `isinstance` guard."""
    value = answer.get(field)
    return value if isinstance(value, str) else ""


@dataclass(frozen=True)
class _BundleNameRepair:
    field: str
    original: str
    normalized: str


def _normalize_bundle_names(rj: dict[str, Any] | None) -> tuple[_BundleNameRepair, ...]:
    """Truncate overlong bundle-name fields only when their shape is already safe.

    Total on any input (DW-181): a non-mapping document answers `()` with no
    repairs rather than raising out of `.get`. Same totality as the
    `escalation._escalation_list` twin and for the same reason -- callers'
    totality over parseable JSON. DW-181 wrote both guards while
    `Engine._run_session` still dereferenced `result.result_json.get(...)`
    behind an `is not None` check alone, so a truthy non-mapping raised there
    before any sweep lane reached here; DW-206 routed that frame through
    `model.result_mapping`, and the triage lane now runs to this guard. It
    calls this one line ahead of `validate_triage`, which names the wrong shape
    on the existing `errors` channel.

    Kept as its own `isinstance` rather than delegated to `result_mapping`, so
    the ablation still proves this function total on its own.
    """
    if not isinstance(rj, dict):
        return ()

    repairs: list[_BundleNameRepair] = []

    def normalize(container: dict[str, Any], key: str, field: str) -> None:
        raw = container.get(key)
        if not isinstance(raw, str) or len(raw) <= _BUNDLE_NAME_MAX_LENGTH:
            return
        if raw[0] not in _BUNDLE_NAME_INITIAL_CHARS or any(
            char not in _BUNDLE_NAME_SAFE_CHARS for char in raw[1:]
        ):
            return
        normalized = raw[:_BUNDLE_NAME_MAX_LENGTH]
        container[key] = normalized
        repairs.append(_BundleNameRepair(field, raw, normalized))

    bundles = rj.get("bundles", [])
    if isinstance(bundles, list):
        for bundle_index, bundle in enumerate(bundles):
            if isinstance(bundle, dict):
                normalize(bundle, "name", f"bundles[{bundle_index}].name")

    decisions = rj.get("decisions", [])
    if isinstance(decisions, list):
        for decision_index, decision in enumerate(decisions):
            if not isinstance(decision, dict):
                continue
            options = decision.get("options", [])
            if not isinstance(options, list):
                continue
            for option_index, option in enumerate(options):
                if isinstance(option, dict):
                    normalize(
                        option,
                        "bundle_name",
                        f"decisions[{decision_index}].options[{option_index}].bundle_name",
                    )
    return tuple(repairs)


# ------------------------------------------------------------- triage plan


@dataclass(frozen=True)
class ResolvedEntry:
    id: str
    evidence: str


@dataclass(frozen=True)
class Bundle:
    name: str
    dw_ids: tuple[str, ...]
    intent: str
    decision_note: str = ""  # human-decision context appended to the intent file


@dataclass(frozen=True)
class DecisionOption:
    key: str
    label: str
    effect: str  # build | close | keep-open
    intent: str = ""  # required when effect == "build"
    resolution: str = ""  # optional when effect == "close"
    bundle_name: str = ""  # optional name override for the built bundle


@dataclass(frozen=True)
class Decision:
    id: str
    question: str
    context: str
    options: tuple[DecisionOption, ...]
    recommendation: str

    def option(self, key: str) -> DecisionOption | None:
        for opt in self.options:
            if opt.key == key:
                return opt
        return None


@dataclass(frozen=True)
class TriagePlan:
    open_ids: frozenset[str]
    already_resolved: tuple[ResolvedEntry, ...] = ()
    bundles: tuple[Bundle, ...] = ()
    blocked: tuple[tuple[str, str], ...] = ()  # (id, blocker)
    skip: tuple[tuple[str, str], ...] = ()  # (id, reason)
    decisions: tuple[Decision, ...] = ()


@dataclass(frozen=True)
class SweepSelection:
    selected: tuple[deferredwork.DWEntry, ...]
    excluded: tuple[deferredwork.DWEntry, ...]
    missing_severity: tuple[deferredwork.DWEntry, ...] = ()


def select_entries(
    entries: Iterable[deferredwork.DWEntry],
    *,
    only_ids: tuple[str, ...] | None = None,
    min_severity: str | None = None,
    validate_only: bool = False,
) -> SweepSelection:
    """Select from canonical open entries without changing the ledger parser's universe."""
    if only_ids is not None and min_severity is not None:
        raise ValueError("--only cannot combine with --min-severity")
    if only_ids is not None:
        if not only_ids:
            raise ValueError("--only requires at least one DW-<n> id")
        malformed = [dw_id for dw_id in only_ids if not DW_ID_RE.fullmatch(dw_id)]
        if malformed:
            raise ValueError("--only contains malformed ids: " + ", ".join(malformed))
    if min_severity is not None and min_severity not in SEVERITY_ORDER:
        raise ValueError("--min-severity must be one of: " + ", ".join(SEVERITY_ORDER))
    open_entries = tuple(entry for entry in entries if entry.open)
    if only_ids is not None:
        open_ids = {entry.id for entry in open_entries}
        unavailable = [dw_id for dw_id in only_ids if dw_id not in open_ids]
        if validate_only and unavailable:
            raise ValueError("--only ids must exist and be open: " + ", ".join(unavailable))
        requested = set(only_ids)
        return SweepSelection(
            selected=tuple(entry for entry in open_entries if entry.id in requested),
            excluded=tuple(entry for entry in open_entries if entry.id not in requested),
        )
    if min_severity is not None:
        floor = SEVERITY_ORDER[min_severity]
        missing = tuple(entry for entry in open_entries if entry.severity is None)
        selected = tuple(
            entry
            for entry in open_entries
            if entry.severity is not None and SEVERITY_ORDER[entry.severity] >= floor
        )
        selected_ids = {entry.id for entry in selected}
        return SweepSelection(
            selected=selected,
            excluded=tuple(entry for entry in open_entries if entry.id not in selected_ids),
            missing_severity=missing,
        )
    return SweepSelection(selected=open_entries, excluded=())


def _plan_str(container: dict[str, Any], field: str, where: str, errors: list[str]) -> str | None:
    """One LLM-authored free-text scalar off a triage plan, or None when it is
    not a string — the plan-input twin of `unusable_answer_reason` (DW-148).

    `str(value)` was the old spelling and it never failed: a list `intent`
    became the truthy repr "['do', 'x']", which satisfied the
    `effect == "build" and not intent` gate, landed in `DecisionOption.intent`
    and rode into `Bundle.intent`, `intent.md` and a dev session — the DW-141
    harm, on the plan surface instead of the persisted-answer store. A malformed
    plan is REFUSED and re-driven instead, through the `errors` channel that
    already exists; there is no repair path and no new drop cause.

    The message names a POSITION, or the decision id when that id is itself a
    string, plus the field and the type name only — never the offending value's
    prose, the same rule and the same wording as `unusable_answer_reason`. Since
    DW-157 a decision-level message can name the decision's own position too,
    when its `id` is not a string.

    Callers thread the `None` through rather than falling back to "": "" would
    re-enter the field's own empty/invalid-value branch and double-report, and
    one error per fault is the convention here (see
    `test_validate_triage_reports_one_error_when_a_name_fails_both_gates`). The
    dataclasses are constructed with `value or ""` at the end.

    The fields screened here: bundle `name` and `intent`; option `key` (DW-156),
    `effect`, `intent`, `label`, `resolution` and `bundle_name`; decision
    `question` (DW-156), `recommendation` and `context`. Everything else off a
    triage plan — the decision, section and `dw_ids` identifiers,
    `already_resolved.evidence`, `blocked.blocker`, `skip.reason` — keeps its
    `str(...)` treatment deliberately.

    `key` and `question` joined the screened set under DW-156 for their LIVE
    unscreened sinks, including operator-facing and journaled consumers:
    `question` is printed by `DecisionPrompter.ask`, announced by `gates.notify`,
    written to the `decision-pending` journal record and listed
    by `bmad-loop decisions --list`; `key` is printed among the options there,
    persisted into the answer store and written to `decision-answered`. Both also
    reach `Bundle.decision_note` and so `intent.md`. That path already screened
    `key`: `_materialize_bundles` takes the option key only when `_agreeing_option`
    matched it against the answer-store-screened `answer_key`. The agreement
    check does not screen `question`, whose type check here closes that path."""
    value = container.get(field, "")
    if isinstance(value, str):
        return value
    errors.append(f"{where}: {field} not a string: {type(value).__name__}")
    return None


def _plan_list(
    container: dict[str, Any], field: str, where: str, errors: list[str]
) -> list[Any] | None:
    """One list-shaped container off a triage plan, or None when it is not a
    list — the container twin of :func:`_plan_str` (DW-155/DW-158).

    Every one of these sites used to iterate `container.get(field, [])`
    unscreened, so a JSON `null` (the likeliest wrong shape an LLM emits) raised
    `TypeError: 'NoneType' object is not iterable` straight out of
    `validate_triage` — past `_ensure_triage`'s live-session call site, which has
    no guard of its own, and past `decisions.pending_missed_decisions`, whose
    read loop catches only decode faults. A malformed plan is REFUSED through the
    `errors` channel that already exists; there is no repair path, and the caller
    never sees a `[]` fallback it could mistake for an empty section.

    `None` (not `[]`) is threaded for the reason `_plan_str` threads it rather
    than "": `[]` would re-enter the field's own emptiness/arity branch and
    double-report one fault (`bundle ... has no dw_ids`, `decision ... needs at
    least 2 options`). One error per fault is the convention here.

    The message names a POSITION (or the decision id, when that id is itself a
    string — since DW-157 a decision-level message can name the decision's own
    position instead) and the type name only, never the offending value's prose —
    these strings reach a journal. `where` is "" for a top-level section, whose
    field name already locates it.
    """
    value = container.get(field, [])
    if isinstance(value, list):
        return value
    prefix = f"{where}: " if where else ""
    errors.append(f"{prefix}{field} not a list: {type(value).__name__}")
    return None


def _plan_mapping(item: Any, where: str, errors: list[str]) -> dict[str, Any] | None:
    """One object-shaped member of a triage plan's list section, or None when it
    is not an object (DW-155/DW-158).

    Each member loop called `.get` on whatever the list held, so a `null` or a
    bare string member raised `AttributeError` out of `validate_triage` and every
    caller of it. `_normalize_bundle_names`, which runs first, already carries
    exactly this guard on the same members — this is the check the validator was
    missing, not a new policy.

    Callers `continue` past a `None` while enumerating the RAW list, so a dropped
    member does not renumber the positions its siblings report. As with
    :func:`_plan_str` the message carries the position and the type name only.
    """
    if isinstance(item, dict):
        return item
    errors.append(f"{where} not an object: {type(item).__name__}")
    return None


def _plan_identifier(raw: Any, where: str, label_prefix: str) -> tuple[str, str, str]:
    """The three shapes one plan identifier takes: `(dw_id, shown, label)` —
    the IDENTITY, the bare subject a message interpolates, and the same subject
    behind its section prefix (DW-157/DW-171).

    The identity is `str(raw)` and stays that way: `id` members are deliberately
    NOT type-checked (the DW-145/148 Never clause), so what validates and what is
    refused is unchanged by this helper. What is screened is what gets PRINTED. An
    object-valued `id` would otherwise interpolate its own stringified contents
    into every message its loop emits, and this module promises those carry a
    POSITION and the type name only, never the offending value's prose — these
    strings reach a journal.

    A `str` id (the empty string included, since the fallback is keyed on the TYPE
    and not on truthiness) keeps today's wording byte for byte: `shown` is the id
    itself and `label` is `f"{label_prefix} {raw}"`. A non-string one is named by
    its position in both.

    `shown` and `label` are separate values because the two display shapes are:
    `claim`'s `appears in both` prints the subject BARE, while the loops'
    `has no evidence` / `names no blocker` / `gives no reason` / `has no question`
    messages print it behind a section prefix. For a string id the two collapse to
    the strings each site emitted before.

    The two BARE-only callers — a bundle's `dw_ids` members (DW-178) and
    `validate_migration`'s `mapping[i].dw_id` (DW-180) — have no section-prefixed
    message at all, so they pass `label_prefix=""` and discard `label`. Sites that
    print the subject `repr`-QUOTED are not this helper's: `mapping invents unknown
    key {k!r}` and `mapping repeats key {k!r}` go through `_shown_value`. Those stay
    byte-identical for a STRING key only, which is the whole of the parity this
    module promises: the old spelling was `repr(str(raw))`, so a non-string SCALAR
    key that printed `'5'` or `'None'` now prints `5` or `None` unquoted. That is
    accepted — `_shown_value(str(raw))` would restore the quoting only by putting an
    object key's stringified prose back into the message, which is the leak.
    `validate_migration`'s `manifest says ..., ledger disagrees` needs neither
    helper: it is reachable only once `source` AND `target` are both non-`None`,
    which proves its key is a genuine manifest key and its id a genuine ledger id —
    both strings by construction.
    """
    if isinstance(raw, str):
        return raw, raw, f"{label_prefix} {raw}"
    shown = f"{where} (id not a string: {type(raw).__name__})"
    return str(raw), shown, shown


def _shown_value(value: Any) -> str:
    """One LLM-authored value as a diagnostic prints it (DW-171).

    `repr` is kept for the flat scalars — `got None` and `got 'wrong'` are pinned
    wording. What that buys is NOT a length bound: an LLM-authored string is
    printed verbatim and can be arbitrarily long, which the byte-for-byte rule
    freezes here deliberately. What it buys is that a flat scalar has no NESTED
    structure to expose, and nesting is exactly the harm this screens: an object-
    or list-valued field used to print its whole contents — its keys included —
    into a message that reaches the journal. So anything non-scalar is named by
    its type alone.
    """
    if value is None or isinstance(value, (str, int, float)):
        return repr(value)
    return f"a {type(value).__name__}"


def validate_triage(
    rj: dict[str, Any] | None, expected_open_ids: set[str] | None
) -> tuple[TriagePlan | None, list[str]]:
    """Deterministic validation of the triage session's result.json. Returns
    (plan, []) or (None, errors). expected_open_ids=None skips the ledger
    equality check (used when reloading a previously validated plan)."""
    errors: list[str] = []
    if rj is None:
        rj = {}
    if not isinstance(rj, dict):
        # `rj = rj or {}` substituted only on a FALSY document, so every other
        # wrong-shape top level -- a list, a string, a number -- reached `.get`
        # and raised `AttributeError` out of every caller (DW-155). Refused
        # through the same channel as any other malformed plan, and BEFORE
        # `_normalize_bundle_names`, which also assumes a mapping.
        return None, [f"triage result not a JSON object: {type(rj).__name__}"]
    _normalize_bundle_names(rj)
    if rj.get("workflow") != TRIAGE_WORKFLOW:
        return None, [
            f"workflow must be {TRIAGE_WORKFLOW!r}: got {_shown_value(rj.get('workflow'))}"
        ]

    raw_open_ids = _plan_list(rj, "open_ids", "", errors)
    if raw_open_ids is None:
        # Early return, like the `workflow` and open-set-mismatch refusals around
        # it: the ledger-equality check below has nothing left to compare.
        return None, errors
    # Identity is `str(i)`, unchanged — the comparison below is byte for byte the
    # one it always was. What is derived alongside it is the DISPLAY name for each
    # claimed id, from the RAW member: `invented` is the half of the mismatch that
    # comes from the PLAN (`missed` comes from the ledger and is strings by
    # construction), so an object-valued `open_ids` member used to print its own
    # contents into a journaled message. First occurrence wins, so a duplicate
    # cannot rename the position its first spelling reported (DW-171).
    shown_open: dict[str, str] = {}
    for open_index, raw_open in enumerate(raw_open_ids):
        claimed_id = str(raw_open)
        if claimed_id in shown_open:
            continue
        shown_open[claimed_id] = (
            raw_open
            if isinstance(raw_open, str)
            else f"open_ids[{open_index}] (not a string: {type(raw_open).__name__})"
        )
    claimed_open = set(shown_open)
    if expected_open_ids is not None and claimed_open != expected_open_ids:
        missed = sorted(expected_open_ids - claimed_open)
        invented = sorted(shown_open[i] for i in claimed_open - expected_open_ids)
        return None, [
            "open_ids do not match the ledger's open entries"
            + (f"; missing: {', '.join(missed)}" if missed else "")
            + (f"; not open in the ledger: {', '.join(invented)}" if invented else "")
        ]
    universe = expected_open_ids if expected_open_ids is not None else claimed_open

    seen: dict[str, str] = {}  # id -> category that claimed it

    def claim(dw_id: str, category: str, subject: str) -> None:
        """`dw_id` is the plan's identity and keys `seen`; `subject` is what the
        error PRINTS. All five id-bearing loops pass an explicit subject derived
        by `_plan_identifier` — `decisions` (DW-157), `already_resolved`,
        `blocked`, `skip` (DW-171) and `bundles` (DW-178, which claims each
        member of `dw_ids` and names it `bundles[i] dw_ids[j]`) — so an
        object-valued `id` is named by its position instead of by its own
        stringified contents. Every one of them keeps `str(...)` as the IDENTITY
        that keys `seen` (the DW-145/148 Never clause); only the display is
        screened. `subject` is REQUIRED rather than defaulting to `dw_id`: that
        default is exactly how DW-178 happened — the bundles loop silently took
        it, and neither DW-157 nor DW-171 noticed the omission — so a sixth
        id-bearing loop must now name its subject or fail to typecheck rather
        than fail quietly. Byte-identical
        wording for a STRING id comes from `_plan_identifier`, which returns the
        id itself as the subject, not from any fallback here."""
        if dw_id not in universe:
            errors.append(f"{category} references unknown/closed id {subject}")
        elif dw_id in seen:
            errors.append(f"{subject} appears in both {seen[dw_id]} and {category}")
        else:
            seen[dw_id] = category

    resolved = []
    for resolved_index, raw_resolved in enumerate(
        _plan_list(rj, "already_resolved", "", errors) or []
    ):
        item = _plan_mapping(raw_resolved, f"already_resolved[{resolved_index}]", errors)
        if item is None:
            continue
        dw_id, id_shown, resolved_label = _plan_identifier(
            item.get("id", ""), f"already_resolved[{resolved_index}]", "already_resolved"
        )
        evidence = str(item.get("evidence", "")).strip()
        claim(dw_id, "already_resolved", id_shown)
        if not evidence:
            errors.append(f"{resolved_label} has no evidence")
        resolved.append(ResolvedEntry(dw_id, evidence))

    bundles = []
    names: set[str] = set()
    for bundle_index, raw_bundle in enumerate(_plan_list(rj, "bundles", "", errors) or []):
        # Positional, not by name: the name itself may be the non-string field,
        # so it cannot be the thing that identifies the bundle in an error. Same
        # label shape as `_normalize_bundle_names`, which ran above. Enumerating
        # the RAW list is what keeps a dropped member from renumbering its
        # siblings' positions.
        where = f"bundles[{bundle_index}]"
        item = _plan_mapping(raw_bundle, where, errors)
        if item is None:
            continue
        name = _plan_str(item, "name", where, errors)
        # `repr(name)` for every message that already named the bundle by name;
        # the position stands in when there is no name to print.
        label = repr(name) if name is not None else where
        if name is not None:
            if not BUNDLE_NAME_RE.match(name):
                errors.append(f"bundle name {name!r} invalid (want {BUNDLE_NAME_RE.pattern})")
            # The one rule BUNDLE_NAME_RE cannot express. A cycle-1 bundle's name IS its
            # directory (`_write_intent`), and the reserved Windows device basenames --
            # CON, NUL, AUX, PRN, COM<N>, LPT<N> -- are `[a-z0-9-]`-legal names that no
            # Windows filesystem will accept as one (matched case-insensitively, so
            # lowercase is no reprieve). Testing `safe_segment` identity rather than a
            # hand-written device list keeps this gate in lockstep with the sanitizer
            # that defines the set: the identical idiom, for the identical reason, as
            # `runs.is_valid_run_id`. Guarded on the match above so one bad name yields
            # one error and not two.
            if BUNDLE_NAME_RE.match(name) and safe_segment(name) != name:
                errors.append(f"bundle name {name!r} is not a legal path segment")
            if name in names:
                errors.append(f"duplicate bundle name {name!r}")
            # Only a string name is registered, so a type-failed bundle is invisible
            # to the option loop's `bundle_name in names` duplicate check. That gap
            # is covered by the refusal: its type error is already in `errors`, and a
            # non-empty `errors` returns `(None, errors)` before any duplicate could
            # matter.
            names.add(name)
        raw_dw_ids = _plan_list(item, "dw_ids", where, errors)
        # MEMBERS keep their `str(...)` treatment (DW-148 drew that line); only
        # the container is shape-checked. What `_plan_identifier` adds on top of
        # that identity is the DISPLAY name (DW-178): `dw_ids` below is still the
        # `str(...)` list, and still what feeds `Bundle` and the "has no dw_ids"
        # guard, while `claim` now prints an object-valued member by its POSITION
        # instead of its own stringified contents -- these messages reach the
        # journal. Enumerating the RAW list keeps positions stable, and the
        # `label_prefix` return is unused here because neither message `claim`
        # emits is section-prefixed. Guarded on the check having passed so a
        # `null` list does not also report "has no dw_ids".
        members = [
            _plan_identifier(raw_member, f"{where} dw_ids[{member_index}]", "")
            for member_index, raw_member in enumerate(raw_dw_ids or [])
        ]
        dw_ids = [identity for identity, _shown, _member_label in members]
        if raw_dw_ids is not None and not dw_ids:
            errors.append(f"bundle {label} has no dw_ids")
        for dw_id, id_shown, _member_label in members:
            claim(dw_id, f"bundle {label}", id_shown)
        intent = _plan_str(item, "intent", where, errors)
        if intent is not None:
            intent = intent.strip()
            if not intent:
                errors.append(f"bundle {label} has no intent")
        bundles.append(Bundle(name or "", tuple(dw_ids), intent or ""))

    blocked = []
    for blocked_index, raw_blocked in enumerate(_plan_list(rj, "blocked", "", errors) or []):
        item = _plan_mapping(raw_blocked, f"blocked[{blocked_index}]", errors)
        if item is None:
            continue
        dw_id, id_shown, blocked_label = _plan_identifier(
            item.get("id", ""), f"blocked[{blocked_index}]", "blocked"
        )
        blocker = str(item.get("blocker", "")).strip()
        claim(dw_id, "blocked", id_shown)
        if not blocker:
            errors.append(f"{blocked_label} names no blocker")
        blocked.append((dw_id, blocker))

    skip = []
    for skip_index, raw_skip in enumerate(_plan_list(rj, "skip", "", errors) or []):
        item = _plan_mapping(raw_skip, f"skip[{skip_index}]", errors)
        if item is None:
            continue
        dw_id, id_shown, skip_label = _plan_identifier(
            item.get("id", ""), f"skip[{skip_index}]", "skip"
        )
        reason = str(item.get("reason", "")).strip()
        claim(dw_id, "skip", id_shown)
        if not reason:
            errors.append(f"{skip_label} gives no reason")
        skip.append((dw_id, reason))

    decisions = []
    for decision_index, raw_decision in enumerate(_plan_list(rj, "decisions", "", errors) or []):
        item = _plan_mapping(raw_decision, f"decisions[{decision_index}]", errors)
        if item is None:
            continue
        raw_id = item.get("id", "")
        # Still the plan's identity, and still NOT type-checked: it keys `seen`
        # and becomes `Decision.id` (the DW-145/148 Never clause stands). What is
        # screened is what gets PRINTED. An object-valued `id` would otherwise
        # interpolate its own stringified contents into every message this loop
        # emits, and this module promises they carry type names only — these
        # reach a journal. So the two display shapes are derived ONCE, from the
        # raw value: a string id (the empty string included) keeps today's
        # wording byte for byte, a non-string one is named by its position.
        # `id_shown` is the bare subject `claim` interpolates; `decision_label`
        # is the prefix every other message in the loop carries (DW-157). The
        # derivation itself lives in `_plan_identifier`, shared with the three
        # section loops above since DW-171 — one definition of the idiom, not two.
        dw_id, id_shown, decision_label = _plan_identifier(
            raw_id, f"decisions[{decision_index}]", "decision"
        )
        claim(dw_id, "decisions", id_shown)
        # Type-checked rather than `str(...)`-ed (DW-156) for its live unscreened
        # sinks, all of them operator-facing or journaled: `DecisionPrompter.ask`
        # prints it, `gates.notify` announces it, the `decision-pending` journal
        # record carries it and `bmad-loop decisions --list` lists it. It reaches
        # `Bundle.decision_note` and `intent.md` too: `_agreeing_option` checks
        # option semantics, not the question, so this check closes that path.
        # Guarded on `is not None` so a non-string never also trips the emptiness
        # error — one error per fault.
        question = _plan_str(item, "question", decision_label, errors)
        if question is not None:
            question = question.strip()
            if not question:
                errors.append(f"{decision_label} has no question")
        options = []
        keys: set[str] = set()
        decision_bundle_names: set[str] = set()
        raw_options = _plan_list(item, "options", decision_label, errors)
        # Whether `keys` below is a faithful census of the options this decision
        # OFFERED. Only an object contributes a key, so a `null` container or a
        # dropped member leaves `keys` short and a perfectly good
        # `recommendation` would report `not an option` on top of the shape
        # error it is merely downstream of -- one fault, two errors, and a
        # re-driven triage session told to fix a field that was never wrong.
        # This is the `bundles`/`names` gap documented above, but not its
        # frequency: that one needs a second bundle to collide, while this one
        # fires on every shape-failed option a recommendation names.
        options_well_shaped = raw_options is not None
        for option_index, raw_option in enumerate(raw_options or []):
            raw = _plan_mapping(raw_option, f"{decision_label} options[{option_index}]", errors)
            if raw is None:
                options_well_shaped = False
                continue
            raw_key = raw.get("key", "")
            # Positional unless the key is a string, for the reason the `bundles`
            # loop is positional. `key` IS type-checked now (DW-156 — it is printed
            # among the options by `DecisionPrompter.ask` and `decisions --list`,
            # persisted into the answer store and written to the
            # `decision-answered` journal record), but the label still has to be
            # derived from the RAW value BEFORE that check runs: a failed check
            # leaves nothing to name the option with, and interpolating the raw
            # value would print an object key's own prose into a message this file
            # promises carries type names only — and these reach a journal. A
            # string key keeps today's wording byte for byte, empty ones included.
            where = (
                f"{decision_label} option {raw_key}"
                if isinstance(raw_key, str)
                else f"{decision_label} options[{option_index}]"
            )
            key = _plan_str(raw, "key", where, errors)
            if key is None:
                # `keys` is now short by one, exactly as a dropped member leaves
                # it short: a sound `recommendation` must not report `not an
                # option` on top of the fault it is merely downstream of.
                options_well_shaped = False
            # Every free-text scalar this option contributes downstream, screened
            # before any of them is read. A field that failed the type check is
            # None from here on, and each value check below is guarded on that --
            # one error per fault, never a type error plus the empty-value error
            # a "" fallback would also have tripped.
            effect = _plan_str(raw, "effect", where, errors)
            intent = _plan_str(raw, "intent", where, errors)
            if intent is not None:
                intent = intent.strip()
            option_label = _plan_str(raw, "label", where, errors)
            if option_label is not None:
                option_label = option_label.strip()
            resolution = _plan_str(raw, "resolution", where, errors)
            if resolution is not None:
                resolution = resolution.strip()
            bundle_name = _plan_str(raw, "bundle_name", where, errors)
            if key is not None:
                # The one site in this loop that still interpolates an identifier's
                # VALUE rather than a positional label. It is leak-free only because
                # `key` is `_plan_str`-screened above and so is known to be a string
                # here — not because the label is positional (DW-156 carries DW-157
                # at this site).
                if not key or key in keys:
                    errors.append(f"{decision_label}: missing/duplicate option key {key!r}")
                keys.add(key)
            if effect is not None and effect not in DECISION_EFFECTS:
                errors.append(f"{where}: bad effect {effect!r}")
            if effect == "build" and intent is not None and not intent:
                errors.append(f"{where}: effect 'build' needs intent")
            if bundle_name is not None:
                if bundle_name and not BUNDLE_NAME_RE.match(bundle_name):
                    errors.append(f"{where}: bad bundle_name {bundle_name!r}")
                # The second site that mints a bundle directory, gated for the reason
                # stated at the `bundles` loop above. A build-effect option's
                # `bundle_name` becomes `Bundle.name` in `_materialize_bundles`, so it
                # reaches `_write_intent`'s cycle-1 directory by the identical path --
                # `BUNDLE_NAME_RE` is no more able to express the rule here than there.
                # Guarded on the match above so one bad name yields one error, and on
                # nothing else: an absent `bundle_name` fails that match already.
                if BUNDLE_NAME_RE.match(bundle_name) and safe_segment(bundle_name) != bundle_name:
                    errors.append(
                        f"{where}: bundle_name {bundle_name!r} is not a legal path segment"
                    )
                if effect == "build" and bundle_name:
                    if bundle_name in names:
                        errors.append(f"duplicate bundle name {bundle_name!r}")
                    decision_bundle_names.add(bundle_name)
            options.append(
                DecisionOption(
                    key=key or "",
                    label=option_label or key or "",
                    effect=effect or "",
                    intent=intent or "",
                    resolution=resolution or "",
                    bundle_name=bundle_name or "",
                )
            )
        names.update(decision_bundle_names)
        # The RAW length, not the surviving one: a dropped member already
        # reported its own fault and must not also trip the arity error.
        if raw_options is not None and len(raw_options) < 2:
            errors.append(f"{decision_label} needs at least 2 options")
        recommendation = _plan_str(item, "recommendation", decision_label, errors)
        # Guarded on the option shapes for the reason stated at the loop above:
        # against a short `keys` this check reports a fault the plan does not
        # have. A recommendation that really is bogus is still refused on the
        # next pass, once the options are objects.
        if recommendation is not None and options_well_shaped and recommendation not in keys:
            errors.append(f"{decision_label}: recommendation {recommendation!r} not an option")
        context = _plan_str(item, "context", decision_label, errors)
        decisions.append(
            Decision(
                dw_id,
                question or "",
                (context or "").strip(),
                tuple(options),
                recommendation or "",
            )
        )

    unclaimed = sorted(universe - set(seen))
    if unclaimed:
        # Sorted on the IDENTITY as ever, joined on the DISPLAY name (DW-179).
        # `universe` is a subset of `shown_open`'s keys by construction — with
        # `expected_open_ids is None` universe IS `set(shown_open)`, and otherwise
        # the equality check above already returned on any mismatch — so the
        # direct index is total, the same invariant `invented` relies on.
        errors.append(f"open entries not triaged: {', '.join(shown_open[i] for i in unclaimed)}")

    if errors:
        return None, errors
    return (
        TriagePlan(
            open_ids=frozenset(universe),
            already_resolved=tuple(resolved),
            bundles=tuple(bundles),
            blocked=tuple(blocked),
            skip=tuple(skip),
            decisions=tuple(decisions),
        ),
        [],
    )


# ---------------------------------------------------------- migration plan


@dataclass(frozen=True)
class PreCanonical:
    """What a pre-existing canonical entry is held to across a migration.

    Status alone was the whole snapshot until #519, and that is what let a
    rewrite drop a ``gate:`` line and pass: the gate is the one field whose loss
    is both silent and unsafe. ``deferred-work-format.md`` calls removing it
    "the exact failure this field exists to prevent", and
    ``Engine._refuse_gated_story`` then dispatches the story the entry was
    holding back.

    ``gate_tokens`` unions :attr:`~bmad_loop.deferredwork.EntryGates.tokens`
    with ``malformed`` because the question is "did a token the entry declared
    survive", not "was it enforceable". A malformed ``gate: 3.2`` gates nothing,
    but it reads to anyone scanning the entry as a gate in force and ``validate``
    reports it (``deferred.hard-gate-unstructured``); dropping it retires that
    report silently, which is the same failure one level down.

    The counts ``EntryGates`` also carries — ``lines``, ``empty``, ``near_miss``
    — are deliberately NOT snapshotted. None of them names a story, so losing one
    cannot change which story is gated, and they are exactly what a legitimate
    reflow of a multi-line declaration moves.
    """

    status: str
    gate_tokens: tuple[str, ...]
    severity: str | None


def snapshot_canonical(text: str) -> dict[str, PreCanonical]:
    """The pre-migration state ``validate_migration`` holds the rewrite to.

    A named function rather than a comprehension inlined at its one call site so
    that the tests grade the snapshot production actually builds: a hand-written
    ``{"DW-1": PreCanonical("open", ("3-2",), "high")}`` would pass whatever the parser
    really produces for that entry, and the bug being fixed here lived in the
    snapshot, not in the comparison.

    Keying by id is safe only because ``_ensure_migration`` refuses a ledger
    that carries duplicate canonical ids before any rewrite is attempted. Do
    not soften that refusal into per-id collapse-hardening here: tokens and
    status snapshotted independently describe an entry that never existed, and
    each half patched on its own opens the next cross-product (a token
    harvested from a ``done`` twin paired with an ``open`` twin's status both
    refuses a faithful rewrite and newly gates a story that was not gated).
    """
    snapshot: dict[str, PreCanonical] = {}
    for e in deferredwork.parse_ledger(text):
        g = deferredwork.gates(e)
        snapshot[e.id] = PreCanonical(e.status, g.tokens + g.malformed, e.severity)
    return snapshot


def duplicate_ids(entries: Iterable[deferredwork.DWEntry]) -> list[str]:
    """The DW ids naming more than one entry, sorted.

    One function for both sides of a migration on purpose: the rewrite is
    refused when the ledger it STARTED from carries duplicates and when the
    ledger it produced does, and two detectors that disagreed about what counts
    as a duplicate would leave exactly the gap between them open.
    """
    seen: set[str] = set()
    dupes: set[str] = set()
    for e in entries:
        (dupes if e.id in seen else seen).add(e.id)
    return sorted(dupes)


def validate_migration(
    rj: dict[str, Any] | None,
    manifest: list[dict[str, Any]],
    pre_canonical: dict[str, PreCanonical],
    new_text: str,
) -> list[str]:
    """Deterministic validation of a legacy-ledger migration session: the
    rewritten ledger must contain zero legacy items, preserve every
    pre-existing canonical entry's status and every ``gate:`` token it
    declared, continue DW numbering, and the result.json mapping must cover
    the manifest exactly. Returns errors, empty on success."""
    if rj is None:
        rj = {}
    if not isinstance(rj, dict):
        # The DW-155 guard `validate_triage` carries one function over, which this
        # twin was left without (DW-170): `rj = rj or {}` substituted only on a
        # FALSY document, so every other wrong-shape top level -- a list, a string,
        # a number -- reached `.get` and raised `AttributeError` out of THIS
        # function. When DW-170 wrote this guard it bought totality for this
        # function's own callers only, NOT a live crash fix: `_ensure_migration`,
        # the only production caller, could not deliver a non-dict here, because
        # `Engine._run_session` dereferenced `result.result_json.get(...)` behind
        # an `is not None` check alone -- a truthy non-dict raised THERE, inside
        # `_run_session`, before this guard was reached. DW-206 routed that frame
        # through `model.result_mapping` while leaving the document itself
        # untouched, so the migration lane now runs to this guard and it answers
        # a real shape rather than an unreachable one -- the same reachability
        # the `validate_triage` twin gained. Refused through the existing
        # `errors` channel; never raised, never repaired.
        return [f"migration result not a JSON object: {type(rj).__name__}"]
    if rj.get("workflow") != MIGRATE_WORKFLOW:
        return [f"workflow must be {MIGRATE_WORKFLOW!r}: got {_shown_value(rj.get('workflow'))}"]
    errors: list[str] = []

    leftovers = deferredwork.parse_legacy(new_text)
    if leftovers:
        listed = "; ".join(f"{e.section or 'top level'}: {e.title[:60]}" for e in leftovers[:10])
        errors.append(f"{len(leftovers)} legacy item(s) still parse as legacy: {listed}")

    parsed = deferredwork.parse_ledger(new_text)
    entries: dict[str, deferredwork.DWEntry] = {e.id: e for e in parsed}
    dupes = duplicate_ids(parsed)
    if dupes:
        errors.append("duplicate DW ids: " + ", ".join(dupes))

    def first_word(status: str) -> str:
        return status.split()[0] if status.split() else ""

    pre_max = max(
        (dw_id.removeprefix("DW-") for dw_id in pre_canonical),
        key=decimal_digits_key,
        default="0",
    )
    pre_max = decimal_digits_key(pre_max)[1]
    for dw_id, pre in pre_canonical.items():
        e = entries.get(dw_id)
        if e is None:
            errors.append(f"pre-existing {dw_id} disappeared")
            continue
        if first_word(e.status) != first_word(pre.status):
            errors.append(f"pre-existing {dw_id} status changed: {pre.status!r} -> {e.status!r}")
        if e.severity != pre.severity:
            errors.append(
                f"pre-existing {dw_id} severity changed: {pre.severity!r} -> {e.severity!r}"
            )
        # Drops and edits only; an ADDED token is deliberately accepted. The two
        # directions are not the same failure: a dropped token un-gates a story
        # silently, which is what #519 is about, while an added one over-blocks
        # loudly and in the safe direction — the operator meets a refusal naming
        # the entry. Refusing an addition would spend one of the two migration
        # attempts on the only direction that cannot cause the failure this
        # guard exists to stop. An EDITED token is caught here anyway: an edit
        # is a drop plus an add, and the drop half is what this reads.
        post = deferredwork.gates(e)
        kept = set(post.tokens) | set(post.malformed)
        lost = [t for t in pre.gate_tokens if t not in kept]
        if lost:
            errors.append(f"pre-existing {dw_id} lost gate token(s): {', '.join(lost)}")
    for dw_id, e in entries.items():
        if dw_id in pre_canonical:
            continue
        if decimal_digits_key(dw_id.removeprefix("DW-")) <= decimal_digits_key(pre_max):
            errors.append(f"new entry {dw_id} does not continue numbering past DW-{pre_max}")
        if first_word(e.status) not in ("open", "done"):
            errors.append(f"new entry {dw_id} has status {e.status!r}; want open or done")

    manifest_by_key = {str(m["key"]): m for m in manifest}
    mapping = rj.get("mapping", [])
    if not isinstance(mapping, list):
        return errors + ["mapping must be a list of {key, dw_id}"]
    seen_keys: set[str] = set()
    target_by_key: dict[str, str] = {}
    sources_by_target: dict[str, list[dict[str, Any]]] = {}
    # Enumerated for the POSITION only: `key` and `dw_id` keep their `str(...)`
    # identities, so what maps, what is refused and what `seen_keys` records are
    # unchanged (DW-180). Screened is what gets PRINTED, because these errors
    # reach the migrate-decision journal record. The two display shapes split by
    # the wording each message already had: `invents unknown key` / `repeats key`
    # print the key `repr`-quoted, which `_shown_value` reproduces for a string,
    # while `no such entry` prints the id bare, which is `_plan_identifier`'s
    # `shown`. Both collapse to today's bytes for a STRING value and only for one:
    # the key half was `repr(str(raw))`, so a non-string SCALAR key that printed
    # `'5'` now prints `5` unquoted (see `_plan_identifier` for why that trade is
    # taken). Two messages here need no positional treatment at all, for the same
    # reachability reason: `repeats key` is past the `source is None` `continue`
    # and `manifest_by_key`'s keys are `str()`-forced, so its key is provably a
    # genuine manifest key — it is converted for UNIFORMITY with its sibling, not
    # from need — and `manifest says ..., ledger disagrees` is left alone outright
    # because `source` AND `target` are both non-`None` by the time it is
    # reachable, so its key and its id are both provably genuine.
    for item_index, item in enumerate(mapping):
        raw_key = item.get("key", "") if isinstance(item, dict) else ""
        raw_dw_id = item.get("dw_id", "") if isinstance(item, dict) else ""
        key = str(raw_key)
        shown_key = _shown_value(raw_key)
        dw_id, shown_dw_id, _dw_id_label = _plan_identifier(
            raw_dw_id, f"mapping[{item_index}].dw_id", ""
        )
        source = manifest_by_key.get(key)
        if source is None:
            errors.append(f"mapping invents unknown key {shown_key}")
            continue
        if key in seen_keys:
            errors.append(f"mapping repeats key {shown_key}")
        seen_keys.add(key)
        target = entries.get(dw_id)
        if target is None:
            errors.append(f"mapping {key} -> {shown_dw_id}: no such entry in the ledger")
            continue
        # Recorded for the manifest-order pass below only once the id resolved
        # to a ledger entry: past this point `dw_id` is a key of `entries`, so
        # the bare `{target}` that pass prints is provably genuine (the same
        # reasoning `manifest says ..., ledger disagrees` relies on), and an id
        # `no such entry` already refused is not reported a second time as an
        # ordering fault.
        target_by_key.setdefault(key, dw_id)
        if dw_id in pre_canonical:
            errors.append(
                f"mapping {key} -> {dw_id}: legacy items must map to newly created entries"
            )
        else:
            sources_by_target.setdefault(dw_id, []).append(source)
            if (first_word(target.status) == "done") != bool(source["done"]):
                want = "done" if source["done"] else "open"
                errors.append(f"mapping {key} -> {dw_id}: manifest says {want}, ledger disagrees")
    for dw_id, sources in sources_by_target.items():
        target = entries[dw_id]
        source_severities = [source.get("severity") for source in sources]
        present = [severity for severity in source_severities if severity is not None]
        expected = max(present, key=SEVERITY_ORDER.__getitem__) if present else None
        if target.severity != expected:
            if len(sources) == 1:
                key = str(sources[0]["key"])
                errors.append(
                    f"mapping {key} -> {dw_id}: manifest severity "
                    f"{expected!r}, ledger has {target.severity!r}"
                )
            else:
                errors.append(
                    f"merged mapping -> {dw_id}: highest manifest severity "
                    f"{expected!r}, ledger has {target.severity!r}"
                )
    missing = sorted(set(manifest_by_key) - seen_keys)
    if missing:
        errors.append("manifest keys not mapped: " + ", ".join(missing))

    # Dry-run projects legacy ids in manifest/file order.  Hold the rewrite to
    # that same contiguous allocation so a selected provisional id cannot name
    # a different issue after migration.  Equal adjacent targets are the one
    # permitted exception: migration mode may merge duplicate legacy items,
    # including nonadjacent items, onto any target allocated earlier.
    expected_suffix = increment_decimal_digits(pre_max)
    allocated_targets: set[str] = set()
    for manifest_item in manifest:
        key = str(manifest_item["key"])
        target = target_by_key.get(key)
        if target is None:
            continue
        if target in allocated_targets:
            continue
        expected_target = f"DW-{expected_suffix}"
        if target != expected_target:
            errors.append(
                f"mapping {key} -> {target}: migration ids must follow manifest order; "
                f"expected {expected_target}"
            )
        allocated_targets.add(target)
        expected_suffix = increment_decimal_digits(expected_suffix)
    return errors


# --------------------------------------------------------------- prompting


class DecisionPrompter:
    """Walks the human through pending decisions on the terminal. Injection
    points exist so tests can script answers.

    The interactive terminal prompt is the v1 protocol: observers (the TUI
    dashboard, ATTENTION watchers) learn a sweep is blocked from the
    decision-pending journal event written just before ask() and attach to
    the sweep's tmux window to answer. A decisions-file protocol — engine
    writes the pending question to a file and polls for an answer the TUI
    could write in-app — is deliberately deferred to v2; it needs timeout +
    ownership semantics this run-blocking prompt avoids."""

    def __init__(
        self,
        input_fn: Callable[[str], str] = input,
        print_fn: Callable[[str], None] = print,
    ):
        self.input_fn = input_fn
        self.print_fn = print_fn

    def ask(self, decision: Decision) -> DecisionOption:
        p = self.print_fn
        p("")
        p(f"── decision needed: {decision.id} " + "─" * 30)
        p(decision.question)
        if decision.context:
            p("")
            p(decision.context)
        p("")
        for opt in decision.options:
            marker = "  (recommended)" if opt.key == decision.recommendation else ""
            p(f"  [{opt.key}] {opt.label} — {opt.effect}{marker}")
            if opt.intent:
                p(f"      {opt.intent}")
        keys = [o.key for o in decision.options]
        while True:
            raw = self.input_fn(
                f"choice [{'/'.join(keys)}] (enter = {decision.recommendation}): "
            ).strip()
            if not raw:
                raw = decision.recommendation
            chosen = decision.option(raw)
            if chosen is not None:
                return chosen
            p(f"  invalid choice {raw!r}")


# ------------------------------------------------------------ sweep engine


def _rearm_generation(task: StoryTask) -> None:
    """Open a new session-id generation for a sweep task restarting from ESCALATED.

    The restart resets ``attempt`` to 0 for a fresh budget, and that reset is exactly
    what makes the next dispatch re-mint ``attempt == 1`` — an id byte-equal to the
    abandoned attempt's, since ``engine._session_task_id`` emits its discriminator only
    above zero. The artifact a shared id corrupts is ``tasks/<id>/escalation.json``: the
    sweep skill writes it, and two records carrying one id both name that one mutable
    file, so the abandoned cycle's escalation is the fresh session's too.
    ``resolve._gather_escalations`` now opens each distinct ``task_id`` once and
    de-duplicates entries by content, so it no longer reports the same aliased file
    twice. Both adapters also unlink cycle outputs in ``start_session``, which stops a
    healthy restart from inheriting stale contents — but cleanup still leaves the two
    historical records naming one mutable directory: a healthy restart erases the
    abandoned cycle's artifact, while a re-escalation replaces it for both records.
    Minting a fresh id is what preserves one artifact namespace per recorded cycle.

    Same pattern as ``runs.rearm_escalation``, DIFFERENT reason: #705's harm is
    ``_resumable_session`` verdict replay, which runs only on the dev/review phases and
    never reaches ``TRIAGE_RUNNING``/``TRIAGE_VERIFY``. ``cmd_resolve`` *can* reach a
    sweep task (``_escalate`` raises with ``PAUSE_ESCALATION`` and a story key, which
    the engine persists), and its own bump there is harmless: the re-arm leaves the task
    PENDING, so this restart arm does not fire on top of it.

    Call ONLY where the task is taking a genuinely fresh attempt budget: the
    ``Phase.ESCALATED`` restart arms, and ``Sweep._reset_superseded_bundle_state``
    (a reset bundle task adopting a DIFFERENT bundle's ids never attempted that
    bundle at all). An ordinary non-escalated restart keeps its attempt counter, so
    ``attempt += 1`` already yields a fresh id; bumping there would move the
    namespace for nothing and break the "every id already on disk stays
    byte-identical" property the suffix rule exists to hold.
    """
    task.generation += 1


class SweepEngine(Engine):
    """Engine variant whose loop processes the deferred-work ledger instead
    of sprint-status. Bundles reuse the inherited story pipeline through the
    override seams; the triage session has its own phase pair."""

    def __init__(
        self,
        *args: Any,
        triage_adapter: Any = None,
        prompting: bool = False,
        decisions_only: bool = False,
        max_bundles: int | None = None,
        repeat: bool | None = None,
        max_cycles: int | None = None,
        only_ids: tuple[str, ...] | None = None,
        min_severity: str | None = None,
        prompter: DecisionPrompter | None = None,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self.adapters["triage"] = (
            triage_adapter if triage_adapter is not None else self.adapters["dev"]
        )
        self.prompting = prompting
        self.decisions_only = decisions_only
        self.max_bundles = max_bundles if max_bundles is not None else self.policy.sweep.max_bundles
        self.repeat = repeat if repeat is not None else self.policy.sweep.repeat
        self.max_cycles = max_cycles if max_cycles is not None else self.policy.sweep.max_cycles
        self.only_ids = only_ids
        self.min_severity = min_severity
        self._selection_started = self.state.sweep_cycle > 1 or any(
            key == TRIAGE_KEY or key.startswith(f"{TRIAGE_KEY}-") or BUNDLE_KEY_RE.match(key)
            for key in self.state.tasks
        )
        self.prompter = prompter or DecisionPrompter()
        # The two decision quarantines — ids already journaled as skipped, and
        # ids whose recorded answer was already journaled as DROPPED (and
        # notified) — live on `state` (`sweep_skipped_decisions` /
        # `sweep_dropped_decisions`), not here. Without them a persistent
        # decision item notifies once per repeat cycle, and a cycle whose
        # re-triage happens to mint an AGREEING option revives a decision the
        # operator was already told had been dropped: `_materialize_bundles`
        # leaves the run-level `answers` entry alone (it is the human's recorded
        # answer and stays auditable on disk), so `_decisions_phase` re-reads it
        # every cycle. They are persisted run state (DW-124), deliberately:
        # the disposition is the RUN's — so a pause/resume of the same run must
        # not re-announce it, while a NEW run re-evaluates from scratch — and
        # the answer is the human's.
        #
        # The undecodable-prune carry (DW-182/186) is the exact opposite call, and
        # the contrast is the argument for both. It lives HERE, on the instance,
        # never on `state`: its only consumer is the `_loop` frame that just called
        # `_cycle`, and it has to reach that frame because the repeat boundary
        # below it COMMITS the ledger — a refusal that stays inside the prune
        # publishes bytes nobody could decode and crashes cycle N+1 on them
        # anyway. Persisting it would encode a decision no resume can reach:
        # `_loop`'s own `read_for_write` raises on the identical bytes at the top
        # of the cycle body, so a resume of this run never gets far enough to read
        # the flag. The quarantines above are persisted because their disposition
        # outlives the frame that made it; this one cannot outlive it at all.
        self._prune_ledger_unreadable = False
        self.state.run_type = "sweep"

    def _quarantine(self, ids: list[str], dw_id: str) -> None:
        """Add `dw_id` to one of `state`'s decision quarantines if absent, and
        persist immediately — mirroring `Engine._run_auto_sweep`'s
        mutate-then-`_save()` latch, since the whole point of the list is that a
        resume of this run sees it.

        Every call site runs this AFTER its journal row and its notify, so the
        residual crash window (announced, not yet persisted) resumes into a
        re-announcement rather than into a silent quarantine — the safe
        direction for a record an operator reads."""
        if dw_id not in ids:
            ids.append(dw_id)
        self._save()

    def _remaining_estimate(self) -> int | None:
        """Sweep override of the graceful-stop hint: how many deferred-work
        entries are still open in the ledger — the work a resume would pick up.
        Like the base, a hint only: the whole body is guarded so an
        unreadable/invalid ledger returns None rather than derailing the stop."""
        try:
            ledger = self.workspace.paths.deferred_work
            # OBSERVATION arm (DW-146): a graceful-stop hint, nothing written from
            # it. The outer guard stays — it also covers `open_ids` — but routing
            # the read through the named arm is what records the classification.
            #
            # The fault is CHECKED rather than discarded, because this helper's
            # `None` and its `0` mean opposite things to the stop: `None` is "no
            # estimate", while `0` is a positive claim that a resume would pick up
            # nothing — and that number is published, in the `run-stop` journal row
            # and the graceful-stop notice. Degrading an unreadable ledger to the
            # empty text would report "0 remaining" for a file nobody could read,
            # the same fabricated answer `cli._sweep_dry_run` refuses to print.
            #
            # And the READ's fault is JOURNALED before the `None`, not merely
            # checked: the observation arm's rule is "degrade, and journal the
            # fault where a journal is in hand" — one is in hand here, and an
            # unreadable ledger is by far the likeliest way this hint goes away.
            # Scoped to the read leg, deliberately. The outer guard still answers
            # `None` silently for anything raised AFTER it (`open_ids`, the append
            # below), so `remaining: null` is not in general self-explaining; what
            # the row buys is that the one fault class the arm hands back as a
            # value gets attributed instead of collapsing into that same silence.
            text, fault = deferredwork.read_for_observation(ledger)
            if fault is not None:
                self.journal.append(
                    "sweep-remaining-estimate-unreadable",
                    ledger=str(ledger),
                    error=fault,
                )
                return None
            selection = select_entries(
                deferredwork.parse_ledger(text),
                only_ids=self.only_ids,
                min_severity=self.min_severity,
            )
            return len(selection.selected)
        except Exception:  # a hint must never break the stop
            return None

    # ------------------------------------------------------------ main loop

    def _loop(self) -> None:
        ledger = self.workspace.paths.deferred_work
        cycle = max(1, self.state.sweep_cycle)
        if self._finish_inflight_bundles():
            # a recovered bundle's ledger restore can leave the LEDGER dirty, and
            # triage plus the first bundle baseline read it, so it is published
            # here. Only it: unrelated dirt in the same repository is left for
            # whoever owns it, so this no longer ends on a clean TREE and nothing
            # downstream may assume one. Guarded on a non-empty recovery pass, so
            # a fresh sweep spawns no git at all (see `_close_resolved` for the
            # guard inventory across all seven sites).
            # The LEDGER FILE (`_commit_ledger`): this publisher wrote the ledger,
            # so it names the file it published and the commit is narrowed to it.
            # Spelled off `self.workspace.paths` rather than a `ledger` local, at
            # every one of the five publishers: `self.paths.deferred_work` is a
            # DIFFERENT file under worktree isolation, and only the workspace's
            # copy is the one a publisher just wrote.
            self._commit_ledger(
                "chore(sweep): commit ledger after recovering in-flight bundles",
                path=self.workspace.paths.deferred_work,
                family="ledger",
            )
        while True:
            # First statement of the loop body: covers the boundary right after
            # _finish_inflight_bundles on resume and between repeat cycles. A
            # request during a cycle is caught before the next _run_bundle (see
            # _cycle); one landing between cycles stops here before cycle N+1
            # re-triages.
            self._check_stop_request()
            self.state.sweep_cycle = cycle
            self._save()
            # REPAIR/WRITE (DW-146): this text drives migration and the whole
            # write-bearing cycle below it.
            text = deferredwork.read_for_write(ledger) or ""
            if deferredwork.has_legacy(text):
                if cycle > 1:
                    # freeform text appeared mid-run; _ensure_migration assumes
                    # one migration per run, so hand off to a fresh sweep
                    # `stop_cause` beside `reason`, on all five stop sites (DW-201):
                    # `diagnostics._JOURNAL_DROP_FIELDS` holds `reason` and renders it
                    # as a presence boolean, so a scrubbed dump could not tell the five
                    # stops apart at all. The same closed-slug convention `regen_cause`
                    # (DW-164) and `drop_cause` use, and for the same reason — the two
                    # carry the SAME token, so `reason` is unchanged for every reader
                    # of the raw journal.
                    self.journal.append(
                        "sweep-repeat-done",
                        cycles=cycle - 1,
                        reason="legacy-appeared",
                        stop_cause="legacy-appeared",
                    )
                    gates.notify(
                        self.policy,
                        self.run_dir,
                        "legacy ledger entries appeared mid-sweep",
                        "run a fresh `bmad-loop sweep` to migrate them",
                    )
                    return
                self._ensure_migration(text)
                # REPAIR/WRITE (DW-146): same cycle, re-read after migration.
                text = deferredwork.read_for_write(ledger) or ""
            entries = deferredwork.parse_ledger(text)
            selection = select_entries(
                entries,
                only_ids=self.only_ids,
                min_severity=self.min_severity,
                validate_only=not self._selection_started,
            )
            self._selection_started = True
            open_now = {entry.id for entry in entries if entry.open}
            if not open_now:
                if cycle == 1:
                    self.journal.append("sweep-nothing-open", ledger=str(ledger))
                else:
                    self.journal.append(
                        "sweep-repeat-done",
                        cycles=cycle - 1,
                        reason="no-open",
                        stop_cause="no-open",
                    )
                return
            selected_ids = {entry.id for entry in selection.selected}
            selector = "only" if self.only_ids is not None else f"min-severity:{self.min_severity}"
            if selection.excluded:
                self.journal.append(
                    "sweep-selection-excluded",
                    cycle=cycle,
                    reason=selector,
                    dw_ids=[entry.id for entry in selection.excluded],
                )
            if selection.missing_severity:
                self.journal.append(
                    "sweep-selection-missing-severity",
                    cycle=cycle,
                    dw_ids=[entry.id for entry in selection.missing_severity],
                )
            if not selected_ids:
                if cycle == 1:
                    self.journal.append("sweep-selection-empty", reason=selector)
                else:
                    self.journal.append(
                        "sweep-repeat-done",
                        cycles=cycle - 1,
                        reason="no-selected",
                        stop_cause="no-selected",
                    )
                return
            if cycle > 1:
                self.journal.append("sweep-cycle", cycle=cycle, open=len(open_now))
            progressed = self._cycle(cycle, selected_ids)
            if self.decisions_only or not self.repeat:
                return
            if self._prune_ledger_unreadable:
                # DW-182/186. `_prune_pre_answers` refused to read the ledger
                # because nothing could decode it, and that refusal has to END a
                # repeating run rather than stay inside the cycle: the boundary
                # `_commit_ledger` below PUBLISHES the ledger, so falling through
                # would commit the undecodable bytes and cycle N+1 would then
                # crash on them at this loop's own `read_for_write` anyway. Cycle
                # `cycle` COMPLETED — that is what the prune's degrade bought —
                # so `cycles=cycle`, unlike the `legacy-appeared` arm above, which
                # fires before its cycle does any work and reports `cycle - 1`.
                # Placed above `not progressed` and `max_cycles` so it is the
                # reported reason whenever it fires; below the early return so a
                # non-repeating or `--decisions-only` run is untouched. Not a
                # pause and not recovery: the repair is a human editing the file,
                # and re-running `bmad-loop sweep` is the resume. Deliberately NOT
                # extended to the DW-176 absence refusal — an absent ledger ends
                # the next cycle cleanly on `no-open`.
                self.journal.append(
                    "sweep-repeat-done",
                    cycles=cycle,
                    reason="ledger-unreadable",
                    stop_cause="ledger-unreadable",
                )
                # The message NAMES the file and the re-run's precondition. Neither
                # is guessable: `implementation_artifacts` is configurable to any
                # absolute path and the ledger may be symlinked out of the project,
                # so "the ledger" names nothing an operator can open; and this stop
                # deliberately leaves the file DIRTY, which is exactly what
                # `cmd_sweep`'s `worktree_clean` refusal rejects in the code repo
                # (an external ledger repo is not checked), so an unqualified
                # "re-run `bmad-loop sweep`" sends the
                # human into an exit-1 they were told not to expect.
                gates.notify(
                    self.policy,
                    self.run_dir,
                    "the deferred-work ledger could not be decoded mid-sweep",
                    f"repair {ledger} by hand, then commit or stash any changes in "
                    f"{self.paths.repo_root} and re-run `bmad-loop sweep` "
                    "(which requires that worktree to be clean)",
                )
                return
            if not progressed:
                self.journal.append(
                    "sweep-repeat-done",
                    cycles=cycle,
                    reason="no-progress",
                    stop_cause="no-progress",
                )
                return
            if cycle >= self.max_cycles:
                self.journal.append(
                    "sweep-repeat-done",
                    cycles=cycle,
                    reason="max-cycles",
                    stop_cause="max-cycles",
                )
                return
            # a deferred bundle's ledger restore can leave the LEDGER dirty, and
            # the next cycle's triage and bundle baselines read it, so it is
            # published here. Only it — unrelated dirt stays with its owner and the
            # next cycle does not start on a clean TREE.
            # This site carries NO non-empty-write guard: `progressed` can be true
            # from a dropped answer that wrote no ledger at all (DW-135), so
            # `path_clean` inside `_commit_ledger` is what makes such a cycle a
            # no-op. See `_close_resolved` for the full inventory.
            # the ledger file, as above
            self._commit_ledger(
                "chore(sweep): commit ledger before next sweep cycle",
                path=self.workspace.paths.deferred_work,
                family="ledger",
            )
            cycle += 1

    def _finish_inflight_bundles(self) -> int:
        """Re-drive every bundle this run left in flight, keyed on the task's own
        persisted story_key. Returns how many were recovered.

        The base Engine._loop opens with _finish_inflight for exactly this reason;
        the sweep loop used to recover a bundle only from inside _run_bundle, which
        a cycle reaches only after re-deriving the bundle's key from the *current*
        triage plan. A re-armed bundle therefore survived a resume solely because
        the cached triage.json reloaded and re-emitted the same bundle name — lose
        that cache and a fresh triage partitions the ids under new names, silently
        orphaning the human's resolution (#94).

        Runs before the ledger is read, so a bundle it closes leaves the open set
        and no fresh triage can re-bundle those ids (validate_triage rejects a plan
        whose open_ids disagree with the ledger). A recovered bundle that defers or
        escalates keeps its ids open. A dev-leg discard never closes them; an
        in-place post-acceptance defer reopens this run's close, while an isolated
        unit's close dies with its unmerged worktree. The existing failed_ids filter
        then drops the fresh plan's overlapping bundle."""
        recovered = 0
        for task in list(self.state.tasks.values()):
            if task.terminal or not BUNDLE_KEY_RE.match(task.story_key):
                continue
            recovered += 1
            self.journal.append(
                "sweep-inflight-redrive",
                story_key=task.story_key,
                phase=str(task.phase),
                rearmed=task.rearmed,  # read before the recovery clears the latch
            )
            if self._recover_inflight_bundle(task):
                continue
            self._ensure_bundle_intent(task)
            self._save()
            self._emit("pre_bundle", task)
            self._run_story(task)
            self._emit("post_bundle", task)
        return recovered

    def _warn_stranded_bundles(self) -> None:
        """Invariant: _finish_inflight_bundles has driven every persisted bundle to
        a terminal phase before a cycle picks new work. A survivor means a bundle
        would be silently dropped — say so loudly rather than sweep past it."""
        stranded = [
            t.story_key
            for t in self.state.tasks.values()
            if BUNDLE_KEY_RE.match(t.story_key) and not t.terminal
        ]
        if not stranded:
            return
        self.journal.append("sweep-inflight-stranded", story_keys=stranded)
        gates.notify(
            self.policy,
            self.run_dir,
            f"{len(stranded)} sweep bundle(s) left in flight",
            "not re-driven by this cycle: " + ", ".join(stranded),
        )

    def _cycle(self, cycle: int, open_now: set[str]) -> bool:
        """One triage -> close -> decide -> bundle pass. Returns whether the
        cycle completed any addressable work — the repeat loop's progress
        predicate. Dropping a recorded decision answer counts (DW-123, widened
        from the keep-open lane to all three drop lanes by DW-135): the drop
        releases its id from a stored answer nothing can act on, so a later
        cycle's fresh triage can address it. It cannot spin the loop —
        `_materialize_bundles` bounds each id to one drop per run, and since
        DW-124 that bound is persisted on `state`, so it holds across a
        pause/resume too and the signal fires at most once per id. Caveat: on
        crash-resume of a cycle whose only progress was already-resolved closes,
        the replayed (idempotent) closes report 0 and the run stops with
        no-progress; the same now goes for a cycle whose only would-be event is a
        drop the pre-crash run already announced and persisted, which the
        quarantine skips rather than re-signalling. Errs toward stopping, never
        loops."""
        self._emit("pre_sweep_cycle", phase=str(cycle))
        self._warn_stranded_bundles()
        plan = self._ensure_triage(open_now, cycle)
        closed = self._close_resolved(plan)
        answers, decisions_closed = self._decisions_phase(plan)
        bundles, answer_dropped = self._materialize_bundles(plan, answers)
        if self.decisions_only:
            self.journal.append("sweep-decisions-only", bundles_not_run=len(bundles))
            self._prune_pre_answers()
            self._emit("post_sweep_cycle", phase=str(cycle))
            return False
        graded_keys: list[str] = []
        for bundle in bundles:
            # Item boundary: a request during bundle N lets N finish through
            # commit; bundle N+1 never starts. A request landing during triage
            # reaches the first iteration here, so triage completes but zero
            # bundles run. Mid-cycle stop is resume-safe: sweep_cycle is
            # persisted, triage.json is cached, closes are idempotent, and
            # terminal tasks are skipped on re-drive.
            self._check_stop_request()
            key = self._run_bundle(bundle, cycle)
            if key is not None:
                graded_keys.append(key)
        # Grade the key each bundle was actually resolved to — the one it ran
        # under, or the terminal one it was skipped as already-finished at,
        # which counts here exactly as it always has. What is never used is a
        # key re-derived from `bundle.name`: since DW-125 a bundle whose own key
        # is held by a terminal task carrying different dw_ids runs under a
        # DEDUPED name, so the re-derived key named the wrong task — the
        # finished one, whose DONE phase counted a bundle this cycle never ran,
        # in both the deduped case and the name-collision drop that runs nothing
        # at all. Reading the reported keys also keeps the lookup total: every
        # key returned here has a task by construction, where a re-derived one
        # need not.
        bundles_done = sum(1 for key in graded_keys if self.state.tasks[key].phase == Phase.DONE)
        self._prune_pre_answers()
        self._emit("post_sweep_cycle", phase=str(cycle))
        return closed > 0 or decisions_closed > 0 or bundles_done > 0 or answer_dropped

    def _prune_pre_answers(self) -> None:
        """Drop consumed pre-answers — entries built or closed this cycle have
        left the open set. Keeps the store from re-applying a stale answer (and a
        keep-open answer's audit line) on the next sweep.

        BOTH ledger-read faults degrade here rather than propagating: absence
        (DW-176) and undecodable bytes (DW-182). The case for staying loud is that
        this read decides a store WRITE, so refusing to guess is right — but the
        refusal IS the refusal to guess. It keeps every answer and prunes nothing,
        so the choice is not "guess vs. crash", it is "keep the store and say so
        vs. crash the sweep". And this call is the LAST in `_cycle`, after every
        bundle has run: a raise here reports a fully completed cycle as crashed
        over bookkeeping, where the refusal costs only consumed answers re-offered
        on the next sweep. Bytes nobody could decode are unknown open work for
        exactly the reason absence is, so they take the same journal row under a
        second fixed `reason` token rather than a kind of their own.

        The undecodable refusal is not only journaled, it is CARRIED: it sets
        `_prune_ledger_unreadable`, which `_loop` reads at the repeat boundary and
        which ends a repeating run there. Without the carry the degrade is a
        half-measure — the boundary `_commit_ledger`'s pathspec IS the ledger, so
        the very next thing a repeating run does is COMMIT the bytes this method
        just refused to read, and cycle N+1 crashes on them at `_loop`'s own bare
        read regardless. Absence (DW-176) sets nothing, deliberately: a ledger that
        is gone ends the next cycle cleanly on `no-open` rather than crashing it.

        `OSError` deliberately still propagates, as it does at every other DIRECT
        caller of `read_for_write` — `_loop`'s two reads take it bare as well — and
        it says nothing about what the ledger holds. `_close_resolved` and
        `_decisions_phase` DO name `OSError` in their catch tuples, but around
        `mark_done_many` and `record_decision`, which read and take the
        cross-process lock internally: what they are catching there is the lock's
        own failure, not this reader's.
        """
        from . import decisions as decisions_store  # lazy: decisions imports sweep

        ledger = self.workspace.paths.deferred_work
        # REPAIR/WRITE (DW-146): the open set derived here decides a store write,
        # and pruning from bytes nobody could read would drop live answers.
        try:
            text = deferredwork.read_for_write(ledger)
        except deferredwork.LedgerReadError as e:
            # UNDECODABLE is refused for the same reason absence is (DW-182), and
            # under the same kind: the open set is the KEEP list for a store write,
            # so a ledger nobody can decode is unknown open work, not zero of it.
            # `reason` stays a FIXED token and the decode fault goes in `error`,
            # already a `diagnostics._JOURNAL_DROP_FIELDS` field. `LedgerReadError`
            # is a plain `Exception` on purpose (DW-146), so it must be named: no
            # `except OSError` upstream would ever see it.
            self.journal.append(
                "sweep-preanswer-prune-refused",
                ledger=str(ledger),
                reason="ledger-unreadable",
                error=str(e),
            )
            # ...and the refusal is CARRIED to `_loop`, beside the row rather than
            # in place of it. The repeat boundary commits the ledger, so a refusal
            # that stayed local would publish bytes nobody could decode; `_loop`
            # reads this flag right after `_cycle` and ends a repeating run on
            # `reason="ledger-unreadable"` without taking that commit. Instance
            # state, not `state` — see the declaration in `__init__`. The absence
            # arm below sets nothing: an absent ledger ends the next cycle cleanly.
            self._prune_ledger_unreadable = True
            return
        # ABSENCE is refused, not collapsed to `""` (DW-176). The `or ""` spelling
        # every observation-shaped caller uses is exact for them because
        # `open_ids("")` and `open_ids(<absent>)` say the same thing about a ledger
        # nobody is writing — but here the open set is the KEEP list for a store
        # write, so an empty one means "nothing is open, drop every answer" and a
        # ledger that vanished mid-cycle would wipe the human's whole pre-answer
        # store and (since DW-160) commit the wipe. An absent ledger is unknown
        # open work, not zero of it. The test is `is None`, never falsiness: an
        # empty-but-PRESENT ledger genuinely has zero open ids and must keep
        # pruning exactly as it does today.
        if text is None:
            self.journal.append(
                "sweep-preanswer-prune-refused", ledger=str(ledger), reason="ledger-absent"
            )
            return
        # The store lives under the project that owns `run_dir`, never
        # `self.workspace.root`: where `repo_root` names a tree DISJOINT from the
        # project the two diverge and a workspace-rooted prune trimmed a store
        # that does not exist, leaving consumed entries behind (the comment in
        # `_decisions_phase` says why the run dir is the stable anchor). Scoped
        # to the disjoint shape on purpose — in the NESTED/monorepo shape
        # (`conftest.nested_repo_root_paths`) `repo_root` is an ANCESTOR of the
        # project, so the store sits inside it and a workspace-rooted spelling
        # found the same file. The ledger read above is unaffected either way —
        # `deferred_work` hangs off `implementation_artifacts`, which stays
        # project-rooted under the override.
        project = _project_of_run_dir(self.run_dir)
        dropped = decisions_store.prune_pre_answers(project, deferredwork.open_ids(text))
        if dropped:
            self.journal.append("decision-preanswers-pruned", dw_ids=dropped)
            # The STORE FILE, not the workspace root: the same divergence that
            # made the prune miss its file made the commit miss its tree (DW-160).
            # Name the file you published, the rule `_commit_ledger` states — this
            # prune writes the pre-answer store and nothing else, and the store is
            # a bare join off the project root that no config knob can move. So
            # the commit carries that one file, and the ledger this cycle's
            # decision phase may have withheld is not published by it (DW-187).
            # The ledger PUBLISHERS name the ledger for the same rule and a
            # different answer. Guarded on `dropped` above: a prune that consumed
            # nothing wrote nothing and spawns no git.
            self._commit_ledger(
                "chore(sweep): drop consumed deferred-work pre-answers",
                path=decisions_store.store_path(project),
                family="store",
            )

    def _prune_dropped_pre_answer(
        self, dw_id: str, drop_cause: str, answer: dict[str, Any]
    ) -> None:
        """Retire the PROJECT-level pre-answer a just-dropped stale answer came
        from (DW-143). Part of the drop itself, not a later cleanup. `answer` is
        the value this run just dropped, and the store entry goes ONLY while it
        still equals it — see the provenance guard below.

        Why it exists: DW-124's quarantine is RUN-scoped by design, so it bounds
        the drop to one announcement per run and a NEW run re-evaluates from
        scratch. But the answer that feeds a stale drop lives in the project store,
        `pending_missed_decisions` filters out any id already usably answered
        there, and `_prune_pre_answers` retires an entry only once a later cycle
        bundles the id and closes it. While triage keeps re-asking the id as a
        DECISION instead, that never happens: every new run re-read the same stale
        answer, re-dropped it and re-notified, and no surface re-offered the id.
        Removing the entry at the drop breaks that loop from both ends — the next
        run reads no stale answer, and `bmad-loop decisions` offers the id again.

        Keep-open-only, deliberately. A dropped `build` answer (`no-intent`,
        `name-collision`) leaves its entry open to be re-asked with the stored
        answer still meaningful, where a dropped keep-open answer has no payload
        left beyond the option it named — there is nothing to preserve.

        Called AFTER `_quarantine`, which is the announce-then-persist order that
        method's docstring promises: the residual crash window (announced,
        quarantined, store not yet pruned) resumes into the DW-124 skip and the
        entry is pruned by the next run that re-drops it. The reverse order would
        leave a window in which a human's answer is already gone while the run has
        no record of having dropped it.

        Reaches the project store ONLY. `<run>/decisions.json` keeps the answer
        (the run-local audit trail is untouched by design) and so does whatever
        `decision:` line `_apply_decision_effect` landed — since DW-186 that call
        can report it wrote none, and this drop is unchanged either way. The journal
        row carries the id and the drop cause alone — no answer prose, no store
        path.

        Retires the entry ONLY while it still holds the value that was dropped.
        The dropped `answer` is this run's RUN-LOCAL copy, and `_decisions_phase`
        lets that copy win over the project store for the rest of the run — so a
        human who re-answers the id out of band while the run is paused
        (`pending_missed_decisions` screens against the store alone, never against
        a run's `decisions.json`) leaves a NEWER store entry this run has never
        evaluated. Keyed on the id alone, the removal deleted that replacement, and
        committed the deletion, on the strength of a stale copy the human had
        already superseded. `drop_pre_answer` compares before it deletes: a seeded
        copy round-trips through JSON unchanged and so equals the entry it came
        from, while a re-answer differs in at least `answered_at`, and an
        interactive in-run answer never equals a store entry at all. The surviving
        replacement is left for the NEXT run to evaluate from scratch, exactly as
        a fresh answer would be; this run stays on its own record."""
        from . import decisions as decisions_store  # lazy: decisions imports sweep

        # `_project_of_run_dir`, never `self.workspace.root`: where `repo_root`
        # names a tree DISJOINT from the project the two diverge and only the run
        # dir stays anchored to the project that owns the store (see
        # `_decisions_phase` and `_prune_pre_answers`, which resolve it the same
        # way). The nested/monorepo shape is unaffected — `repo_root` is an
        # ancestor there, so the store sits inside it.
        project = _project_of_run_dir(self.run_dir)
        if not decisions_store.drop_pre_answer(project, dw_id, answer=answer):
            # Either no store entry (a run-local-only answer) or an entry that is
            # no longer the value dropped (a human's later replacement): no write,
            # no row — the store's bytes are untouched either way.
            return
        self.journal.append(
            "sweep-decision-preanswer-pruned", decision=dw_id, drop_cause=drop_cause
        )
        # Committed like `_prune_pre_answers`': `_materialize_bundles` runs ahead of
        # this cycle's bundles, and bundles need a clean baseline. The STORE FILE for
        # the same reason the removal used it — name the file you published
        # (`_commit_ledger`), and the store is a bare join off the project root.
        # Where `repo_root` names a DISJOINT tree, `workspace.root` is a separate
        # repo and a clean check there says nothing about the tree this write
        # dirtied (DW-160). Narrowed to the store, this site is also the one DW-187
        # is about: it runs LATER in the same cycle as the decision phase, so a wide
        # commit here republished the very ledger bytes that phase withheld.
        # Guarded on `drop_pre_answer` above: no store entry, no write, no git.
        self._commit_ledger(
            "chore(sweep): drop stale deferred-work pre-answer",
            path=decisions_store.store_path(project),
            family="store",
        )

    def _drive_story(self, task: StoryTask) -> None:
        # no spec-approval gate for bundles: the bundle intent came from the
        # validated triage plan (and, for decision bundles, from the human).
        # The base _run_story wraps this in a worktree when isolation=worktree.
        if self._dev_phase(task):
            self._review_and_commit(task)

    # cycle 1 keeps the legacy key so pre-repeat paused runs resume unchanged;
    # "dw{N}-" (not "dw-c{N}-") so a cycle-1 bundle named "c2-foo" can never
    # collide with a cycle-2 bundle named "foo"
    def _bundle_key(self, name: str, cycle: int) -> str:
        return f"dw-{name}" if cycle == 1 else f"dw{cycle}-{name}"

    def _bundle_name_for(self, bundle: Bundle, cycle: int) -> tuple[str, int] | None:
        """The name this bundle runs under and the attempt that found it, or
        None when no key is available. Pure apart from the exhaustion record:
        the dedupe record belongs to `_run_bundle`, which is the only caller
        that knows whether the bundle went on to USE the deduped name.

        The terminal-task early return below is what makes a resume cheap: a
        bundle already finished this run is skipped rather than re-driven. It
        used to compare the KEY alone (DW-125), and the key is a pure function of
        `(name, cycle)` — so when a resume loses `<run>/triage.json`,
        `_ensure_triage` regenerates a plan whose names are re-authored freely,
        and a fresh bundle that happens to reuse a finished bundle's name was
        silently swallowed with its ids never run. `_materialize_bundles`'
        uniqueness pass cannot see this: it compares names against THIS cycle's
        list, never against persisted state.

        The bundle's identity is its `dw_ids`, so agreement is tested on those,
        as SET equality — a regenerated triage may emit the same ids in a
        different order, and treating that as a new bundle would re-run finished
        work on every cache-loss resume, a worse regression than the bug. A
        persisted EMPTY list agrees with anything: it is the pre-`dw_ids`
        `state.json` shape (`model.py` loads a missing key as `[]`), and reading
        it as divergence would re-run every bundle of every legacy paused run.

        On divergence the name gains the same bounded `-2` … `-9` suffix
        `_materialize_bundles` applies to a colliding stored name — deduping the
        NAME rather than the key alone is what keeps `_bundle_key`, the intent
        dirname and `_ensure_bundle_intent`'s key→name round-trip consistent.
        This is the third collision remedy in this file and must not be confused
        with the other two: `_materialize_bundles` DISCARDS a colliding stored
        `bundle_name` (it has `decision-<id>` beneath it) and SUFFIXES that
        fallback (which has nothing beneath it). Here a validated plan name
        collides with PERSISTED state, and suffixing is the only repair — there
        is no fallback name to reach for.

        Scoped to TERMINAL tasks deliberately: an in-flight task at the key still
        goes through `_recover_inflight_bundle` exactly as before
        (`_finish_inflight_bundles` drives persisted bundles terminal before a
        cycle picks new work, and `_warn_stranded_bundles` says so loudly when
        one survives)."""
        wanted = set(bundle.dw_ids)
        for attempt in range(1, 10):
            name = bundle.name if attempt == 1 else f"{bundle.name}-{attempt}"
            task = self.state.tasks.get(self._bundle_key(name, cycle))
            if task is not None and task.terminal and task.dw_ids and set(task.dw_ids) != wanted:
                continue
            return name, attempt
        # Bounded, so the search is provably finite — and loud on both surfaces,
        # because the alternative is the swallowed bundle this guard exists to
        # prevent. The ids stay open for the next sweep.
        self.journal.append(
            "sweep-bundle-key-collision", name=bundle.name, dw_ids=list(bundle.dw_ids)
        )
        # Spell the KEYS, not the bare name: from cycle 2 they are `dw<N>-...`,
        # so a name-only message names nothing the operator can grep state.json
        # for.
        first = self._bundle_key(bundle.name, cycle)
        gates.notify(
            self.policy,
            self.run_dir,
            f"sweep bundle {bundle.name!r} could not be named",
            f"every key from {first} through {first}-9 (cycle {cycle}) is held by a "
            "finished bundle carrying different deferred-work ids; not run: "
            + ", ".join(bundle.dw_ids),
        )
        return None

    def _reset_superseded_bundle_state(self, task: StoryTask) -> None:
        """Drop the per-bundle state a reset task still carries from the bundle it
        was minted for, once `_run_bundle` adopts a DIFFERENT bundle's ids onto it
        (DW-162, DW-163, DW-165). The four ids are one behavior: on divergent
        adoption the task stops owning the superseded bundle's state, so the clears
        live together and this docstring is the record of what is deliberately
        absent from them.

        Every field cleared here NAMES or BUDGETS the superseded bundle:

        - ``bundle_closes_intended`` -- the previous bundle's intended ledger
          closes. ``_carry_isolated_ledger_writes`` and the engine's post-rollback
          replay predicate both key on it, so a stale list closes ids this task
          never ran.
        - ``spec_file`` -- the superseded bundle's amended contract.
          ``_generic_bundle_prompt`` selects the restore-review prompt naming it
          (paired with ``restore_patch``), and ``Engine._record_dev_spec`` is a
          no-op once set, so a survivor would also refuse the replacement bundle's
          own spec on escalation.
        - ``restore_patch`` -- the diff of the superseded bundle's attempt.
        - ``attempt`` + ``review_cycle`` + ``followup_reviews_spent`` -- reset the
          retry and review counters; clear the associated ``defer_reason`` and
          advance ``generation`` for fresh session ids. These operations follow
          ``runs._rearm_escalation_locked``. A sweep bundle runs the base engine's
          review loop, so a
          replacement inheriting an exhausted review budget would force-converge or
          defer on its first round. ``attempt`` and ``generation`` in particular are
          inseparable: zeroing ``attempt`` alone re-mints a byte-equal session id
          (see ``_rearm_generation``).

        ``resolved_redrive`` is deliberately NOT cleared (DW-165's recorded
        2026-09-07 decision): it records that a HUMAN resolved this task, which is
        a fact about the task, not a statement about which spec the task owns.
        Nor is any of the following, each for its own reason:

        - ``sessions`` -- an append-only audit trail; ``_rearm_generation``'s fresh
          id namespace is what keeps the replacement's records distinct.
        - ``bundle_file`` -- the single most bundle-naming field here, and the one
          exception: the caller OVERWRITES it two lines later with this bundle's
          freshly written ``intent.md``, so clearing it would be dead code. Nothing
          reads it in between.
        - ``isolated_ledger_carried`` / ``harvest_carry_commit_pending`` -- the
          ledger-carry replay latches (``Engine._replay_unlatched_ledger_carries``
          skips a task already latched, which WOULD strand the replacement
          bundle's own close). Safe because both are set only on legs that have
          already reached a TERMINAL phase -- the ``_defer`` leg (DEFERRED) and
          past a unit merge (DONE / AWAITING_OPERATOR) -- and ``_run_bundle``
          returns on a terminal task before ever reaching this branch.
        - ``commit_sha`` -- names a commit that really happened; a later commit
          overwrites it.
        - ``rearmed`` -- ``_recover_inflight_bundle`` already cleared it on the
          reset that got us here.
        - ``preserve_ref`` / ``preserve_partial`` -- a ref to a rolled-back
          worktree that still exists on disk; clearing the name would orphan it
          rather than release it.
        - the ``baseline_*`` pair and ``worktree_path`` / ``branch`` -- mount and
          rollback anchors owned by the reset, not by either bundle.
        - ``dispatched_spec_file`` / ``dispatched_spec_snapshot`` -- ``Sweep``
          overrides ``_dispatched_spec_for_attempt`` to ``None`` and
          ``_requires_dispatched_spec_snapshot`` to ``False``, so a sweep task never
          binds them and a clear would be unablatable dead code.

        These clears are DEFENSIVE. No reachable sequence was demonstrated that
        strands a task with a superseded ``spec_file`` / ``restore_patch``
        (DW-162, DW-165) -- but ``_warn_stranded_bundles`` concedes an in-flight
        survivor is possible at all, so the guard makes the hazard structurally
        impossible instead of argued unreachable.

        An EMPTY persisted ``task.dw_ids`` reaches here and takes the FULL reset:
        it satisfies the divergence gate, which is exactly the reading
        ``_run_bundle`` already takes of an empty list for id adoption (that task
        genuinely has no ids and must take the bundle's). Nothing it holds is
        exempt on account of having no ids.

        Why the rearm is gated on DIVERGENCE here rather than added to
        ``_recover_inflight_bundle``'s reset tail: that tail mirrors
        ``Engine._finish_inflight``'s restart arm, which deliberately KEEPS a plain
        crash-restart's budget -- zeroing it there would let a crash-looping run
        never exhaust ``limits.max_attempts``. The two siblings that do zero
        (``_ensure_migration``'s reset, the triage reset) each gate on
        ``Phase.ESCALATED``, i.e. "the human resumed deliberately"; ESCALATED is
        terminal and so never reaches ``_recover_inflight_bundle`` at all.
        Divergent adoption is this seam's equivalent gate. Do not widen the scope
        to the agreeing (or merely reordered) re-dispatch: that is the same bundle
        the task already attempted, and its budget, spec ownership and intended
        closes are rightfully its own.
        """
        task.bundle_closes_intended = []
        task.spec_file = None
        task.restore_patch = None
        task.attempt = 0
        task.review_cycle = 0
        task.followup_reviews_spent = 0
        task.defer_reason = None
        _rearm_generation(task)

    def _run_bundle(self, bundle: Bundle, cycle: int) -> str | None:
        """Run one bundle; returns the task key it ran under, or None when no key
        was available (see `_bundle_name_for`). `_cycle` grades progress on the
        returned keys, so it must never be re-derived from `bundle.name`."""
        resolved = self._bundle_name_for(bundle, cycle)
        if resolved is None:
            return None
        name, attempt = resolved
        key = self._bundle_key(name, cycle)
        task = self.state.tasks.get(key)
        if task is not None and task.terminal:
            return key  # finished (or adjudicated) in a previous resume cycle
        if attempt > 1:
            # Below the skip deliberately: the ordinary second-resume shape has
            # the deduped key ALREADY terminal and agreeing, and journaling the
            # rename up in the resolver re-announced it once per resume for a
            # bundle nothing then renamed. `original=` + `name=` so the record
            # stands on its own, the way its sibling `sweep-bundle-name-deduped`
            # does; `dw_ids` say which work the new key carries.
            self.journal.append(
                "sweep-bundle-key-deduped",
                original=bundle.name,
                name=name,
                attempt=attempt,
                dw_ids=list(bundle.dw_ids),
            )
        if task is None:
            task = StoryTask(story_key=key, epic=0, dw_ids=list(bundle.dw_ids))
            self.state.tasks[key] = task
            self.journal.append("bundle-start", story_key=key, dw_ids=list(bundle.dw_ids))
        elif self._recover_inflight_bundle(task):
            return key
        else:
            # DW-144 (+DW-162, DW-163, DW-165). Recovery reset the task to
            # PENDING and handed the dispatch back to us — and the intent written
            # below is THIS bundle's, not the one the persisted task was minted
            # for. `_bundle_name_for`'s dedupe is scoped to TERMINAL tasks, so a
            # non-terminal task at the key keeps the key whatever its ids are.
            # Stale task ids can reject a dev result for this bundle or make
            # `_close_bundle_ledger_when_spec_status` derive
            # `bundle_closes_intended` from the previous bundle's ids.
            #
            # Journal only on divergence but assign unconditionally: a bundle's
            # identity is its ids under SET equality (a regenerated triage may
            # reorder them, per `_bundle_name_for`), so a pure reorder is not
            # worth announcing once per resume. A persisted EMPTY list is the
            # pre-`dw_ids` `state.json` shape and reads as divergence here, which
            # is right — that task genuinely has no ids and must take these.
            if set(task.dw_ids) != set(bundle.dw_ids):
                self.journal.append(
                    "sweep-bundle-dwids-adopted",
                    story_key=key,
                    previous_dw_ids=list(task.dw_ids),
                    dw_ids=list(bundle.dw_ids),
                )
                # DW-162/163/165. Ids are not the only per-bundle field the reset
                # task carries: everything else naming or budgeting the superseded
                # bundle goes with them. It rides THIS gate — the same divergence
                # test the append above rides — and nothing else about its
                # placement is load-bearing: it never touches `task.dw_ids`.
                self._reset_superseded_bundle_state(task)
            task.dw_ids = list(bundle.dw_ids)
        dirname = name if cycle == 1 else f"c{cycle}-{name}"
        # The document has to agree with the directory it lands in and with the
        # name `_ensure_bundle_intent` recovers back out of the story key.
        written = bundle if name == bundle.name else replace(bundle, name=name)
        task.bundle_file = str(self._write_intent(written, dirname))
        self._save()
        self._emit("pre_bundle", task)
        self._run_story(task)
        self._emit("post_bundle", task)
        return key

    def _recover_inflight_bundle(self, task: StoryTask) -> bool:
        """Recover a bundle task interrupted mid-flight (or re-armed after a
        human resolved its escalation via `bmad-loop resolve`): the same recovery
        as Engine._finish_inflight, including the re-drive latch so a
        human-resolved escalation is protected through every reset (mirrors
        engine.py:1163-1169).

        Returns True when the persisted PROCEED receipt carried the bundle all
        the way through accepted sync, review, and commit — the caller is done.
        Returns False once the task has been reset to PENDING, leaving the caller
        to dispatch it. A bare DEV_VERIFY + spec_file shape is insufficient: the
        pre-action decision save has the same shape for rejected decisions.

        Deliberately narrower than the base _finish_inflight: no
        `_resumable_session` arm, so a bundle whose host died in the
        post-session window still restarts rather than replaying its recorded
        result. Lifting that is a resume-fidelity change of its own. The
        COMMITTING window IS recovered, though — same as the base engine's
        resume-commit arm (#115).

        The reset tail below deliberately does NOT zero `attempt` or re-arm the
        session-id generation: like the base restart arm it mirrors, a plain
        crash-restart keeps its budget, so zeroing here would let a crash-looping
        run never exhaust `limits.max_attempts`. The fresh budget belongs to the
        ADOPTION site instead — `_run_bundle`'s divergence branch, via
        `_reset_superseded_bundle_state`, which is reached only when the caller
        hands this task a different bundle's ids. Those superseded-state clears
        are defensive: no reachable sequence was demonstrated for DW-162 or
        DW-165; the adoption-site guard makes that hazard structurally impossible.
        """
        if task.worktree_path:
            # Sweep replaces Engine._loop, so it performs Engine._finish_inflight's
            # mount-relative re-anchor itself. Accepted receipts reopen this mount
            # regardless of live policy; restart is the only path allowed to release
            # or discard its ownership before future work begins.
            task.rebase_spec_paths_on(Path(task.worktree_path))
        mounted = bool(task.worktree_path)
        restart_isolated = self._isolated and mounted
        if task.phase == Phase.COMMITTING:
            # the gate+advance save landed pre-death; finish the commit
            # instead of rolling verified bundle work back (see
            # Engine._finalize_commit_phase for the re-drive contract).
            self.journal.append("resume-commit", story_key=task.story_key)
            if mounted:
                unit = self._reopen_unit(task)
                prev = self.workspace
                self.workspace = unit.workspace
                try:
                    self._finalize_commit_phase(task)
                finally:
                    self.workspace = prev
                self._integrate_unit(task, unit)
            else:
                self._finalize_commit_phase(task)
            return True
        self.journal.append("resume-restart", story_key=task.story_key, phase=str(task.phase))
        if (
            task.phase == Phase.DEV_VERIFY
            and task.spec_file
            and self._accepted_dev_session_matches(task)
        ):
            self._save()
            if mounted:
                unit = self._reopen_unit(task)
                prev = self.workspace
                self.workspace = unit.workspace
                try:
                    self._resume_after_dev_verify(task)
                finally:
                    self.workspace = prev
                self._integrate_unit(task, unit)
            else:
                self._resume_after_dev_verify(task)
            return True
        if restart_isolated:
            # drop the half-built worktree; _run_story mounts a fresh one
            self._discard_unit_for_restart(task)
        elif mounted:
            # Live in-place policy applies to the replacement attempt, not to an
            # incomplete attempt's mount-owned baselines, paths, and claims.
            self._release_orphaned_mount(task)
        if not restart_isolated and task.baseline_commit:
            # latch resolved_redrive so the corrected spec + restored diff stay
            # protected through every reset of this re-drive, not just this
            # first one; cause="resolved" keeps a human-initiated re-arm
            # pause-free regardless of scm.rollback_on_failure
            task.resolved_redrive = task.resolved_redrive or task.rearmed
            self._rollback_or_pause(task, cause="resolved" if task.rearmed else "stopped")
        task.rearmed = False  # past rollback (only reached when not paused)
        task.phase = Phase.PENDING  # deliberate reset, not a normal transition
        return False

    # ------------------------------------------------------------ migration

    def _ensure_migration(self, text: str) -> None:
        """Pre-DW-format ledger content (older BMAD-method projects) blocks a
        sweep: open_ids() cannot see it and mark_done() cannot flip it. One
        LLM session rewrites the legacy items into canonical DW entries; the
        orchestrator pins exactly what to convert (a manifest from
        parse_legacy), validates the rewrite deterministically, and restores
        the original ledger before any retry."""
        ledger = self.workspace.paths.deferred_work
        task = self.state.tasks.get(MIGRATE_KEY)
        if task is None:
            task = StoryTask(story_key=MIGRATE_KEY, epic=0)
            self.state.tasks[MIGRATE_KEY] = task
        elif task.phase != Phase.PENDING:
            # resumed mid-migration or retrying after an escalation: restart
            self.journal.append("resume-restart", story_key=MIGRATE_KEY, phase=str(task.phase))
            if task.phase == Phase.ESCALATED:
                task.attempt = 0  # the human resumed deliberately; fresh budget
                _rearm_generation(task)  # ...and into a fresh session-id namespace
            if task.baseline_commit and not verify.worktree_clean(self.workspace.root):
                self._safe_reset(task)  # a session died mid-rewrite; restore our ledger
                # REPAIR/WRITE (DW-146): the restored text this migration grades.
                text = deferredwork.read_for_write(ledger) or ""
            task.phase = Phase.PENDING  # deliberate reset, not a normal transition
        # **The invariant: a refusal that dispatches nothing leaves this task
        # owning NO baseline.** It takes both halves below. Sitting above the
        # stamp keeps a fresh entry from taking one; clearing handles the entry
        # that arrives already holding one, which the resume-after-escalation
        # branch above does. Either way the next resume re-stamps the repaired
        # HEAD. Leave a baseline behind and it names the PRE-repair tree: the
        # operator renumbers and resumes, and when that migration session
        # env-faults, the next resume's `_safe_reset` rewinds to that stale
        # baseline — destroying the repair and any commits beside it, and
        # landing back on this same pause. Safe to clear because nothing of
        # ours is outstanding here: no session has run, and the branch above
        # has already restored the tree if a previous one died mid-rewrite.
        # This still sits BELOW that branch, because the ledger it reads must
        # be the restored one.
        #
        # Refused BEFORE a rewrite is dispatched, not after one is graded (#519).
        # A ledger where one id names two entries is corrupt in a way the format
        # cannot express, and there is no automatic outcome that is right: this
        # mode is required to keep pre-existing entries byte-identical, so the
        # only rewrite that preserves the pair trips `duplicate DW ids` in
        # validate_migration, and the only rewrite that passes is a collapse
        # that drops one twin's `gate:` silently. Grading the collapse instead
        # cannot be made safe — a snapshot keyed by id describes a merged entry
        # that never existed, and each half hardened on its own opens the next
        # cross-product. So a human renumbers, which is the call
        # `_apply_deferred_closes` already makes on a duplicate id (#286).
        # Paused rather than ESCALATED, and the task stays PENDING on purpose:
        # that is what makes the refusal re-askable the way `_refuse_gated_story`
        # is. The operator renumbers the ids and resumes, and this re-reads the
        # ledger and lets the migration run — an ESCALATED task would also spend
        # the migration attempt budget on a rewrite that never happened.
        #
        # PAUSE_STORY_GATE, NOT PAUSE_ESCALATION, and the pairing is the point:
        # the stage picks the recovery UI, and every escalation action
        # (`runs.rearm_story`, the TUI's Resolve) requires the task to be
        # Phase.ESCALATED, which this one deliberately is not — so an escalation
        # stage here would offer the operator only actions that must fail. The
        # gate stage routes to a viewer whose single action is "resume", which
        # is the whole remedy once the ids are renumbered. `_refuse_gated_story`
        # already pauses this way with a task that is not escalated: same
        # contract — the deferred-work ledger blocks the run, a human edits it,
        # the resume re-reads it.
        dupes = duplicate_ids(deferredwork.parse_ledger(text))
        if dupes:
            reason = (
                f"{ledger.name} carries duplicate DW ids: {', '.join(dupes)} — one id "
                "names more than one entry, so no migration of it can both preserve "
                "the entries and produce a valid ledger; renumber or merge them by "
                "hand and COMMIT the fix, then resume"
            )
            self.journal.append("migrate-duplicate-ids", story_key=MIGRATE_KEY, dw_ids=list(dupes))
            gates.notify(
                self.policy,
                self.run_dir,
                f"migration refused: {ledger.name}",
                f"{reason} — then `bmad-loop resume {self.state.run_id}`",
            )
            task.baseline_commit = None
            task.baseline_untracked = None
            self._save()
            raise RunPaused(reason, PAUSE_STORY_GATE, MIGRATE_KEY)

        if not task.baseline_commit:
            # `self.workspace.root`, which is `paths.repo_root` — the same anchor the
            # dev writer (`Engine._dev_phase`), the re-arm writer
            # (`runs.rearm_escalation`) and every proof-of-work probe in
            # `verify._verify_shared_gates` use. Under the `repo_root` override it is
            # NOT `paths.project`, and a baseline stamped in one tree and measured in
            # the other names a commit the measuring repo has never heard of (#716).
            task.baseline_commit = verify.rev_parse_head(self.workspace.root)
            task.baseline_untracked = sorted(verify.untracked_files(self.workspace.root))

        legacy = deferredwork.parse_legacy(text)
        pre_canonical = snapshot_canonical(text)
        manifest = [
            {
                "key": e.key,
                "id": e.id,
                "title": e.title,
                "section": e.section,
                "done": e.done,
                "severity": e.severity,
            }
            for e in legacy
        ]
        manifest_path = self.run_dir / "migrate-manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        feedback: Path | None = None
        while True:
            task.attempt += 1
            advance(task, Phase.TRIAGE_RUNNING)
            self._save()
            result = self._run_session(
                task,
                role="triage",
                prompt=self._migrate_prompt(manifest_path, feedback),
                seq=task.attempt,
                session_stage="pre_migrate_session",
            )
            advance(task, Phase.TRIAGE_VERIFY)
            self._save()
            critical_reason = critical_session_reason("migration", result.result_json)
            if critical_reason is not None:
                self._escalate(task, critical_reason)
            # Split so ABSENCE survives: `new_text` stays `str` for
            # `validate_migration`, while `rewrite` keeps the difference between
            # "the session emptied the ledger" and "the session deleted it". On
            # an untracked ledger the restore below has no blob to anchor on and
            # this rejected rewrite — the exact text this attempt graded — is the
            # anchor instead, so flattening `None` to `""` here would make the
            # deleted-ledger case indistinguishable from a rival's empty write.
            # REPAIR/WRITE (DW-146), absence preserved: `None` is the restore
            # anchor's "the session deleted it", distinct from an empty write.
            rewrite = deferredwork.read_for_write(ledger)
            new_text = rewrite if rewrite is not None else ""
            if result.status != "completed":
                errors = [session_failure_reason("migration", result)]
            else:
                errors = validate_migration(result.result_json, manifest, pre_canonical, new_text)
            self.journal.append(
                "migrate-decision",
                attempt=task.attempt,
                session_status=result.status,
                ok=not errors,
                errors=errors,
                env_fault=result.env_fault,
            )
            if result.status != "completed" and result.env_fault:
                # The migration session's CLI lost its API connection (#194): it did
                # no rewrite work, so pause (the ESCALATED-resume above resets
                # task.attempt to 0 — fresh budget) instead of charging a migration
                # attempt. Escalate BEFORE the _safe_reset/attempt-cap path; that
                # resume also restores the ledger if the worktree is dirty.
                self._escalate(
                    task,
                    env_fault_pause_reason("migration", result),
                )
            if not errors:
                advance(task, Phase.DONE)
                self._save()
                (self.run_dir / "migrate-result.json").write_text(
                    json.dumps(result.result_json, indent=2), encoding="utf-8"
                )
                # the ledger file: the migration rewrote the ledger
                self._commit_ledger(
                    "chore(sweep): migrate legacy deferred-work entries to DW format",
                    path=self.workspace.paths.deferred_work,
                    family="ledger",
                )
                post = deferredwork.parse_ledger(new_text)
                self.journal.append(
                    "sweep-migrated",
                    converted=len(manifest),
                    entries_now=len(post),
                    open_now=sum(1 for e in post if e.open),
                )
                self._emit("post_migrate", task)
                return
            # never re-prompt over a half-broken rewrite; the baseline reset
            # covers tracked files, the explicit write covers an untracked
            # ledger that `git reset` cannot restore
            self._safe_reset(task)
            # The WRITE anchor derives from the committed blob, never from an
            # observation of the tree taken after the very reset it would attest
            # to: a rival writing a tracked ledger inside that window would BE
            # the observation, and this restore would overwrite it (#735). Probed
            # BEFORE the lock — it spawns git, and `ledger_lock` may cover file
            # I/O only, which no reset window can (#286). A ledger git does not
            # own has no blob to anchor on, and `reset --hard` cannot have
            # touched it either, so there the anchor is the rejected rewrite this
            # attempt actually graded — down to `None == None` when the session
            # deleted the ledger outright. No anchor at all withholds the write.
            anchor, committed = self._ledger_baseline_text(task)
            expected = committed if committed is not None else rewrite
            diverged = False
            with deferredwork.ledger_lock(ledger):
                # PURE TEXT ONLY under the hold — `ledger_lock` is not reentrant
                # and every mutator takes it.
                # REPAIR/WRITE (DW-146), absence preserved: this compare-and-set
                # authorizes the restore write below, and `None` is a real answer.
                current = deferredwork.read_for_write(ledger)
                if anchor is _LedgerAnchor.NO_RESET_CONTENT and current == text:
                    # ALREADY the text this restore exists to write, so it is
                    # done and there is nothing to escalate. Reachable without
                    # any rival: a session that atomic-SAVES the ledger replaces
                    # a tracked symlink with a regular file, `reset --hard` puts
                    # the link back, and the external target it cannot reach was
                    # never rewritten — so the ledger is correct while `rewrite`,
                    # read off the regular file, is not what is on disk. Demanding
                    # the anchor here would escalate a finished restore and spend
                    # the attempt budget on it.
                    #
                    # Scoped to NO_RESET_CONTENT deliberately. On a BASELINE
                    # anchor the reset republishes the committed text, so
                    # `current == text` is the ORDINARY post-reset state and
                    # accepting it there would retire the divergence check and
                    # the probe-fault escalation along with it. Only where the
                    # reset restored no text of its own is "already correct"
                    # information the anchor cannot supply.
                    pass
                # Either anchor will do below, unlike the engine's two restores:
                # this site supplies its own text for the no-reset-content case
                # (`rewrite`, which it graded), so `expected` is never the bare
                # `None` that would read a rival's deletion as the reset's work.
                elif anchor is not _LedgerAnchor.NONE and current == expected:
                    ledger.parent.mkdir(parents=True, exist_ok=True)
                    atomic_write_text(ledger, text)
                else:
                    diverged = True
            if diverged:
                # No merge and no silent skip. The comment above is the reason:
                # leaving the rejected rewrite standing IS re-prompting over a
                # half-broken ledger, and the migration input a human must fix is
                # no longer the one this attempt was graded against — the same
                # call `migrate-duplicate-ids` makes about a corrupt ledger.
                # A baseline probe that could not answer lands here too, and
                # deliberately: without an anchor there is no proof the text on
                # disk is the reset's own work rather than somebody's live write,
                # and an unprovable restore is exactly the overwrite this arm
                # exists to refuse. The escalation is the right recovery for both
                # — the resume above resets the attempt budget and re-reads the
                # ledger, which is what a rival-corrupted migration input needs.
                # Journaled outside the hold; `_escalate` raises.
                self.journal.append(
                    "sweep-migration-restore-diverged",
                    story_key=MIGRATE_KEY,
                    ledger=str(ledger),
                )
                self._escalate(
                    task,
                    "the ledger changed underneath the failed migration attempt — "
                    "re-run the sweep",
                )
            if task.attempt >= self.policy.sweep.max_migration_attempts:
                self._escalate(
                    task, "migration failed deterministic validation: " + "; ".join(errors)
                )
            feedback = self._write_feedback(
                task,
                "The legacy-ledger migration failed deterministic validation:\n- "
                + "\n- ".join(errors),
            )

    def _migrate_prompt(self, manifest: Path, feedback: Path | None) -> str:
        prompt = f"/bmad-loop-sweep --migrate {manifest}"
        if feedback is not None:
            prompt += f" --feedback {feedback}"
        return prompt

    # --------------------------------------------------------------- triage

    def _ensure_triage(self, open_now: set[str], cycle: int = 1) -> TriagePlan:
        suffix = "" if cycle == 1 else f"-{cycle}"
        triage_path = self.run_dir / f"triage{suffix}.json"
        triage_key = TRIAGE_KEY + suffix
        selector_cache_mismatch = False
        if triage_path.is_file():
            # already validated this run; the ledger has moved since (closes,
            # decisions), so skip the open-set equality re-check. A cache we
            # cannot read or that is not a JSON object degrades to a fresh
            # triage — a truncated file must not crash the whole run.
            try:
                cached = _read_json(triage_path)
            except (json.JSONDecodeError, OSError, UnicodeDecodeError) as exc:
                self.journal.append("sweep-triage-reload-failed", errors=[f"unreadable: {exc}"])
            else:
                if isinstance(cached, dict):
                    plan, errors = validate_triage(cached, None)
                else:
                    plan, errors = None, [f"not a JSON object: {type(cached).__name__}"]
                if (
                    plan is not None
                    and (self.only_ids is not None or self.min_severity is not None)
                    and plan.open_ids != frozenset(open_now)
                ):
                    selector_cache_mismatch = True
                    plan, errors = None, [
                        "cached selected open_ids no longer match the current selector universe"
                    ]
                if plan is not None:
                    return plan
                self.journal.append("sweep-triage-reload-failed", errors=errors)

        task = self.state.tasks.get(triage_key)
        if task is None:
            task = StoryTask(story_key=triage_key, epic=0)
            self.state.tasks[triage_key] = task
        elif task.phase != Phase.PENDING:
            # resumed mid-triage or retrying after an escalation: restart
            self.journal.append("resume-restart", story_key=triage_key, phase=str(task.phase))
            if selector_cache_mismatch:
                task.attempt = 0
                _rearm_generation(task)
            elif task.phase == Phase.ESCALATED:
                task.attempt = 0  # the human resumed deliberately; fresh budget
                _rearm_generation(task)  # ...and into a fresh session-id namespace
            task.phase = Phase.PENDING  # deliberate reset, not a normal transition

        feedback: Path | None = None
        while True:
            task.attempt += 1
            advance(task, Phase.TRIAGE_RUNNING)
            self._save()
            result = self._run_session(
                task,
                role="triage",
                prompt=self._triage_prompt(feedback, open_now),
                seq=task.attempt,
            )
            advance(task, Phase.TRIAGE_VERIFY)
            self._save()
            critical_reason = critical_session_reason("triage", result.result_json)
            if critical_reason is not None:
                self._escalate(task, critical_reason)
            if result.status != "completed":
                plan, errors = None, [session_failure_reason("triage", result)]
            else:
                repairs = _normalize_bundle_names(result.result_json)
                plan, errors = validate_triage(result.result_json, open_now)
                for repair in repairs:
                    self.journal.append(
                        "sweep-bundle-name-normalized",
                        field=repair.field,
                        original=repair.original,
                        normalized=repair.normalized,
                    )
            self.journal.append(
                "triage-decision",
                attempt=task.attempt,
                session_status=result.status,
                ok=plan is not None,
                errors=errors,
                env_fault=result.env_fault,
            )
            if result.status != "completed" and result.env_fault:
                # transport/API failure (#194): pause rather than charge a triage
                # attempt for a session that never reached the API. The
                # ESCALATED-resume above resets task.attempt to 0 (fresh budget).
                self._escalate(
                    task,
                    env_fault_pause_reason("triage", result),
                )
            if plan is not None:
                advance(task, Phase.DONE)
                self._save()
                triage_path.write_text(json.dumps(result.result_json, indent=2), encoding="utf-8")
                self.journal.append(
                    "sweep-triage-result",
                    bundles=len(plan.bundles),
                    decisions=len(plan.decisions),
                    already_resolved=len(plan.already_resolved),
                    blocked=len(plan.blocked),
                    skip=len(plan.skip),
                )
                self._emit("post_triage", task)
                return plan
            if task.attempt >= self.policy.sweep.max_triage_attempts:
                self._escalate(task, "triage output failed validation: " + "; ".join(errors))
            feedback = self._write_feedback(
                task,
                "The triage result.json failed deterministic validation:\n- " + "\n- ".join(errors),
            )

    def _triage_prompt(self, feedback: Path | None, open_now: set[str] | None = None) -> str:
        prompt = "/bmad-loop-sweep"
        if open_now is not None and (self.only_ids is not None or self.min_severity is not None):
            ordered = (
                [dw_id for dw_id in self.only_ids if dw_id in open_now]
                if self.only_ids is not None
                else sorted(
                    open_now,
                    key=lambda value: (
                        len(value.removeprefix("DW-").lstrip("0") or "0"),
                        value.removeprefix("DW-").lstrip("0") or "0",
                        value,
                    ),
                )
            )
            prompt += " --only " + ",".join(ordered)
        if feedback is not None:
            prompt += f" --feedback {feedback}"
        return prompt

    # ------------------------------------------------------ ledger phases

    def _close_resolved(self, plan: TriagePlan) -> int:
        self._emit("pre_close_resolved")
        ledger = self.workspace.paths.deferred_work
        # ONE locked read->edit->write for the whole batch (#286/#469). The
        # per-entry `mark_done` loop this replaces took the cross-process ledger
        # lock once per id, leaving a rival writer — a live run's harvest, the TUI
        # decision modal, `sweep --archive` — a window between every pair of
        # closures, so half this phase's closures could be lost while the other
        # half landed and the journal claimed all of them. `notes=` carries the
        # per-entry evidence the loop passed positionally, so the resulting ledger
        # text and the returned ids (order preserved, skips dropped) are unchanged.
        ids = [entry.id for entry in plan.already_resolved]
        # The write DEGRADES rather than propagating (DW-166). `mark_done_many`'s
        # locked `read_for_write` raises `LedgerReadError` on undecodable bytes,
        # and the lock itself can fail on `OSError` or `StateRootError`; bare, any
        # of them ended the whole sweep as crashed out of a bookkeeping phase that
        # runs before a single bundle. Entries staying `open` is the conservative
        # outcome — the next cycle re-triages them and closes them then — where a
        # crash loses the cycle. `ValueError` is the writers' `date` precondition,
        # unreachable from `_today()` and named for the same reason the DW-146
        # sibling handlers name it. `LedgerReadError` is a plain `Exception` and
        # `StateRootError` is not an `OSError`, so both must be spelled out.
        #
        # What does NOT degrade is the publish itself. `mark_done_many` ends in an
        # atomic write, and its `ENOSPC`/`EROFS`/failed-rename `OSError` reached
        # this tuple looking exactly like the lock's — the fault DW-166 never
        # named, swallowed along with the ones it did. `deferredwork._publish`
        # retypes it as `LedgerWriteError` (an `OSError` subclass, so the CLI and
        # TUI degrade arms are untouched) and this site re-raises it first: a
        # repair write that failed is not a phase that closed nothing, it is a
        # sweep that cannot keep its books, and the rule is AGENTS.md's —
        # observation may degrade, repair writes must raise.
        try:
            closed = deferredwork.mark_done_many(
                ledger,
                ids,
                self._today(),
                "already resolved",
                notes=[f"already resolved: {entry.evidence}" for entry in plan.already_resolved],
            )
        except deferredwork.LedgerWriteError:
            raise
        except (deferredwork.LedgerReadError, OSError, ValueError, StateRootError) as e:
            self.journal.append("sweep-resolved-close-unavailable", dw_ids=ids, error=str(e))
            # `post_close_resolved` still fires and 0 is still returned: the phase
            # RAN, it just closed nothing, and a plugin watching the phase boundary
            # must not silently lose its pairing with `pre_close_resolved`.
            self._emit("post_close_resolved")
            return 0
        if closed:
            self.journal.append("sweep-resolved-closed", dw_ids=closed)
            # ...and the commit rides the SAME guard: a NON-EMPTY pass. With zero
            # ids flipped `mark_done_many` wrote nothing, so there is nothing to
            # publish and no git is spawned at all (DW-183/DW-185).
            #
            # The guard inventory across all seven `_commit_ledger` sites, since
            # it is not uniform and reading it as uniform is the trap:
            #   * FOUR gate on a write result or a normally returned effect:
            #     both prunes (`dropped`, and `drop_pre_answer` answering True),
            #     this site (`closed`), and `_decisions_phase` (`any_effect_landed`).
            #   * THREE gate on something that does NOT prove a write: `_loop`'s
            #     post-recovery publisher counts recovered tasks (which may defer
            #     without editing the ledger), `_loop`'s
            #     cycle-boundary publisher gates on `progressed`, which a dropped
            #     answer can set without touching the ledger, and
            #     `_ensure_migration` gates on `if not errors:`, a verdict on the
            #     rewrite session rather than on bytes changing.
            # Beneath all seven are TWO uniform floors, in this order:
            #   * the TARGET VALIDATION (DW-199/203/205), which asks whether the
            #     declared `family`'s file is still there and still readable before
            #     any git runs. It is what the per-site guards above cannot cover:
            #     each of them grades the phase's own WRITE, and the ledger can
            #     vanish or go undecodable AFTER that write — at which point
            #     `commit_paths` would have staged the absence as a DELETION.
            #     Every site declares its family; none derives one.
            #   * `_commit_ledger`'s `path_clean`, which makes any of them a no-op
            #     when the published file already matches HEAD.
            # The per-site guards are the early-outs that keep a phase which wrote
            # nothing from reaching git at all.
            #
            # the ledger file: `mark_done_many` above wrote the ledger
            self._commit_ledger(
                "chore(sweep): close resolved deferred-work entries",
                path=self.workspace.paths.deferred_work,
                family="ledger",
            )
        self._emit("post_close_resolved")
        return len(closed)

    # `dict[str, Any]` per answer, not `dict[str, str]`: `unusable_answer_reason`
    # deliberately screens only the fields a reader consumes, so `resolution` and
    # `answered_at` can legitimately hold non-strings and a keep-open answer may
    # carry a corrupt `intent`. The narrower annotation read as a guarantee that
    # would justify deleting `_answer_str`; it never was one.
    def _decisions_phase(self, plan: TriagePlan) -> tuple[dict[str, dict[str, Any]], int]:
        from . import decisions as decisions_store  # lazy: decisions imports sweep

        decisions_path = self.run_dir / "decisions.json"
        # The project that OWNS `run_dir`, not `self.workspace.root`: under the
        # supported `repo_root` override (isolation = "none") the workspace root
        # is the separate code repo while the run dir — and the project-level
        # pre-answer store — stay under the PROJECT, so a workspace-rooted
        # confinement refused every write here and a workspace-rooted read
        # silently ignored the store. Derived from the run dir's own shape, which
        # no workspace swap moves.
        project_root = _project_of_run_dir(self.run_dir)
        # The orchestrator writes this store itself, but a crash mid-write, a hand
        # edit or an out-of-band writer can still leave it unreadable or wrongly
        # shaped — and every consumer below calls `.get(...)` on its values, so the
        # bare read let one malformed byte abort the whole sweep. Degrade exactly
        # the way `_ensure_triage`'s cache reload does (journal it, carry on with
        # what is usable): a decision left with no usable answer simply goes back
        # down the pending/skip path, which is where it was before anyone answered
        # it. Per-VALUE, not all-or-nothing, so one bad entry does not cost the
        # well-shaped rest their answers.
        #
        # `unusable` keeps the PER-VALUE drops so the two write-backs below
        # re-publish their parsed values unchanged: on that arm the degrade really is
        # in-memory and this method neither repairs nor trims the file. The two
        # WHOLE-FILE arms cannot offer that — an unreadable file and a non-object
        # top level leave nothing per-value to carry — so `unusable` stays empty
        # there and the next write this phase makes for its own reasons (a seeded
        # pre-answer, an in-run answer) replaces the corrupt file wholesale.
        answers: dict[str, dict[str, Any]] = {}
        unusable: dict[str, Any] = {}
        malformed: list[str] = []
        if decisions_path.is_file():
            try:
                stored = _read_json(decisions_path)
            except (json.JSONDecodeError, OSError, UnicodeDecodeError) as exc:
                self.journal.append("sweep-decisions-reload-failed", errors=[f"unreadable: {exc}"])
            else:
                if isinstance(stored, dict):
                    for stored_id, value in stored.items():
                        key = str(stored_id)
                        # `unusable_answer_reason`, not a local `isinstance`: the
                        # SAME schema `decisions.pending_missed_decisions` screens
                        # by (DW-142), so an id this loop refuses to answer is one
                        # that command re-offers instead of counting answered.
                        # The predicate is shared but store-PARAMETERIZED (DW-147):
                        # this is the run-local store, whose interactive writer
                        # legitimately records `effect: "close"`, so it alone
                        # passes `allow_close=True`. The pre-answer loop below
                        # reads a different file and passes False.
                        reason = unusable_answer_reason(value, allow_close=True)
                        if reason is None:
                            answers[key] = value
                        else:
                            unusable[key] = value
                            malformed.append(f"{key}: {reason} (<run>/decisions.json)")
                else:
                    self.journal.append(
                        "sweep-decisions-reload-failed",
                        errors=[f"not a JSON object: {type(stored).__name__}"],
                    )
        closed = 0
        # Adopt out-of-band pre-answers (a human answered decisions an earlier
        # unattended/abandoned sweep left). The ledger edits were already applied
        # when they answered, so here we only take the answer onboard — this run
        # won't re-prompt/re-skip and build answers materialize into bundles.
        pre = decisions_store.load_pre_answers(project_root)
        seeded = False
        for decision in plan.decisions:
            if decision.id in answers or decision.id not in pre:
                continue
            pre_answer = pre[decision.id]
            pre_reason = unusable_answer_reason(pre_answer, allow_close=False)
            if pre_reason is not None:
                # `load_pre_answers` validates only the TOP level (decisions.py),
                # so a value here can be any JSON at all. Same degrade as the
                # run-local store above — same predicate, too, but configured for
                # THIS store: `allow_close=False`, because `apply_pre_answer`
                # applies a close to the ledger and never records one here, so a
                # `close` in this file is hand-seeded or corrupt and matches no
                # bundling lane (DW-147). Journaled in the same record, which is
                # why each entry names the store it came from: the two files are
                # different, only one of them is the one to hand-fix, and the
                # reason names the effect alone so the suffix is not duplicated.
                # Nothing is written back to the project store:
                # this phase never repairs either file, and re-answering the
                # decision out of band is what overwrites the unusable value.
                malformed.append(f"{decision.id}: {pre_reason} (project .bmad-loop/decisions.json)")
                continue
            answers[decision.id] = pre_answer
            self.journal.append(
                "decision-preanswered",
                dw_id=decision.id,
                effect=pre_answer.get("effect"),
            )
            seeded = True
        if malformed:
            # The PER-VALUE record: one, however many values it covers, naming the
            # ids that lost their answer and the store each came from. It is not
            # the only one a read can write — a whole-file fault above journals its
            # own, so a run-local store that will not parse AND a malformed
            # pre-answer behind it produce two records, one per fault class. Ids,
            # store names and type names only: an answer's prose stays out of the
            # journal, the way `sweep-decision-option-mismatch` keeps it out.
            self.journal.append("sweep-decisions-reload-failed", errors=malformed)
        if seeded:
            # Same helper as `decisions._write_store` (#363), but NOT for #363's
            # reason: `decisions_path` here is the PER-RUN file under
            # `.bmad-loop/runs/<id>/`, which init gitignores, so a stranded temp
            # was never untracked and never held `worktree_clean` False. The
            # project-level `.bmad-loop/decisions.json` is a different file with a
            # near-identical temp name — that is the exposed one. Taken anyway for
            # the fsync and the unique temp name, which two writers of one key
            # would otherwise collide on. Confined to the project root (#593):
            # replacing the NAME, as the bare replace did, left `.bmad-loop/` and
            # `runs/` above it resolved by name, and a link planted at either
            # aimed this write out of the project. The root has to be the PROJECT
            # (`project_root` above; its comment says why not
            # `self.workspace.root`) — and not `self.run_dir` either: a file
            # confined against its own parent walks no components at all, which
            # would refuse nothing.
            atomic_write_text_confined(
                decisions_path,
                # `unusable` first so a well-shaped answer always wins the key:
                # the entries it holds are the ones the read above could not use,
                # re-published unchanged rather than dropped by a write this
                # method makes for an unrelated reason.
                json.dumps({**unusable, **answers}, indent=2),
                confine_root=project_root,
            )
        pending = [d for d in plan.decisions if d.id not in answers]
        answered_interactively = False
        # THREE flags, because the commit and the hand-back ask different
        # questions. `ledger_in_doubt` is the LAST attempt's verdict — set by the
        # degrade below, cleared by the next effect that succeeds — so it means
        # "the bytes now on disk are the ones an effect could not read".
        # `any_effect_faulted` is sticky and only shapes what the human is told. A
        # sticky flag on the commit would be wrong: a walk where DW-1 faults and
        # DW-2 then lands a human-authorized `decision:` line would leave that line
        # uncommitted immediately ahead of this cycle's bundles.
        # `any_effect_landed` is the NON-EMPTY PASS the commit needs beside the
        # withhold: `_apply_decision_effect` is this walk's only ledger write, so a
        # walk that ran none of them — every decision already answered, skipped
        # unattended, or dropped — published nothing and must spawn no git at all
        # (DW-183/DW-185). It is sticky in the other direction, and deliberately:
        # once an effect has written the ledger, a LATER fault is the withhold's
        # business, not this flag's.
        ledger_in_doubt = False
        any_effect_faulted = False
        any_effect_landed = False
        if not self.prompting:
            pending = [d for d in pending if d.id not in self.state.sweep_skipped_decisions]
            for decision in pending:
                self.journal.append("decision-skipped-unattended", dw_id=decision.id)
            if pending:
                gates.notify(
                    self.policy,
                    self.run_dir,
                    f"{len(pending)} deferred-work decisions pending",
                    "run `bmad-loop sweep` interactively to answer them",
                )
            # Quarantine LAST — after the journal rows AND the notify above, the
            # order `_quarantine`'s docstring promises. Persisting inside the loop
            # instead would leave a crash window between the last `_save()` and
            # the notify in which a resume finds every id already quarantined,
            # filters `pending` empty and never writes the ATTENTION line at all:
            # silently swallowing the announcement rather than repeating it.
            for decision in pending:
                self._quarantine(self.state.sweep_skipped_decisions, decision.id)
        else:
            for decision in pending:
                # announce before blocking on input so observers (TUI, ATTENTION
                # watchers) can tell a sweep is waiting on a human
                self.journal.append(
                    "decision-pending", dw_id=decision.id, question=decision.question
                )
                gates.notify(
                    self.policy,
                    self.run_dir,
                    f"decision needed: {decision.id}",
                    decision.question,
                )
                self._emit("pre_decision", story_key=decision.id)
                option = self.prompter.ask(decision)
                # True from the moment the human answers, which is what the flag
                # means: `_return_after_decisions` owes them a hand-back whether or
                # not the effect below lands.
                answered_interactively = True
                answers[decision.id] = {
                    "key": option.key,
                    "label": option.label,
                    "effect": option.effect,
                    "answered_at": self._today(),
                }
                atomic_write_text_confined(  # same file, same reasoning as above (#363, #593)
                    decisions_path,
                    json.dumps({**unusable, **answers}, indent=2),  # as above
                    confine_root=project_root,
                )
                self.journal.append(
                    "decision-answered",
                    dw_id=decision.id,
                    key=option.key,
                    effect=option.effect,
                )
                # The effect DEGRADES per decision and the walk carries on
                # (DW-166), the same shape `cli.cmd_decisions` and
                # `tui.app._record_decision` took in the DW-146 pass — and the
                # handler sits here rather than inside `_apply_decision_effect`
                # so the row can name the decision it lost, exactly as those two
                # wrap `apply_pre_answer` at the loop.
                #
                # This is the reachable shape, not a theoretical one: `prompter.ask`
                # above blocks on the human, so a ledger that goes undecodable
                # while the prompt is open raises out of `record_decision`'s locked
                # `read_for_write`. And by then the answer is already persisted to
                # `<run>/decisions.json` and journalled as `decision-answered` — so
                # bare, the run recorded an answer whose ledger line never landed
                # and then crashed. (That ordering is deliberate and stays: the
                # human's answer must survive a crash. It is the reason this
                # degrade matters, not a thing to fix by reordering.)
                #
                # The publish is the exception to the degrade, as in
                # `_close_resolved`: a `LedgerWriteError` out of `record_decision`
                # means the human's answer is in `<run>/decisions.json` and the
                # ledger's atomic write FAILED — not a lock we never got, not bytes
                # we could not read — and that raises. The ordering above is what
                # makes the raise safe: the answer already survives the crash, and
                # a `build` bundle must not be dispatched off an authorization the
                # ledger could not record.
                try:
                    recorded = self._apply_decision_effect(decision, option)
                except deferredwork.LedgerWriteError:
                    raise
                except (
                    deferredwork.LedgerReadError,
                    OSError,
                    ValueError,
                    StateRootError,
                ) as e:
                    self.journal.append(
                        "sweep-decision-effect-unavailable",
                        dw_id=decision.id,
                        effect=option.effect,
                        error=str(e),
                    )
                    # What is skipped is everything that would claim the effect
                    # landed: no `post_decision` emit, and no `closed` increment,
                    # so the cycle's progress signal does not count a closure the
                    # ledger never received.
                    #
                    # What the human is left with differs by effect, and neither
                    # is repaired THIS RUN — the answer is already in
                    # `<run>/decisions.json`, so the next cycle reloads it into
                    # `answers` and `pending` filters the id out. Only a NEW run
                    # re-offers it. For `close`, the entry simply stays open and a
                    # later run's triage can re-close it. For `build`, the bundle
                    # still materializes and runs off the stored answer — the work
                    # happens, but the entry carries no `decision:` audit line
                    # recording who authorized it. Both are recoverable; crashing
                    # the sweep mid-walk is not, which is the trade this arm makes.
                    #
                    # `ledger_in_doubt` withholds this phase's commit below. The
                    # commit's pathspec IS the ledger, so a commit taken while the
                    # ledger on disk is the text an effect could not read publishes
                    # exactly those bytes — narrowing the scope bounds what else
                    # rides along, it does not make the ledger itself safe to
                    # publish, so the withhold is unchanged. It is the LAST
                    # attempt's verdict, not the walk's: a later effect that
                    # succeeds proves the ledger reads again and clears it, and
                    # that commit then carries the earlier decisions' lines too.
                    # `_close_resolved` refuses its commit on the same fault by
                    # returning early, but its question is simpler — one
                    # all-or-nothing `mark_done_many` batch, so nothing there can
                    # have landed, where this walk writes one decision at a time.
                    ledger_in_doubt = True
                    any_effect_faulted = True
                    continue
                # A False RETURN is the same silent non-write, so it takes the same
                # arm (DW-186). `record_decision` answers False in exactly two
                # states — no ledger file, and no entry carrying this id — and both
                # mean no `decision:` line landed, which is the very claim the
                # `except` above refuses to let the phase make. Bare, the discarded
                # boolean let `closed` count a closure and `post_decision` announce
                # one for an entry the ledger never received. The SAME journal kind
                # on purpose: `_HANDBACK_LEDGER_MISS` prints exactly one kind for an
                # operator to grep, and a second kind here would make that pointer
                # incomplete.
                #
                # `ledger_in_doubt` is deliberately left ALONE — neither set nor
                # cleared. The latch means "the bytes on disk are ones nobody could
                # read", and a False return says nothing either way about that:
                # `record_decision` answers False from `if not path.is_file()`
                # BEFORE it reads anything, so a vanished ledger reaches here having
                # read nothing at all, while a missing entry reaches here off a
                # perfectly good read. So the latch keeps meaning what it meant —
                # the verdict of the last attempt that actually READ — and this arm
                # neither withholds a commit that may carry an earlier decision's
                # authorized line nor clears a doubt it cannot speak to.
                #
                # WHICH of the two states it was is named in `error`, because they
                # are not the same news: a missing entry is one retired id, where a
                # ledger that is gone means every earlier `decision:` line this walk
                # wrote went with it. `is_file()` is the same probe `record_decision`
                # made, re-taken rather than plumbed out of it — this is a journal
                # sentence, not a control decision, and a race between the two only
                # ever mislabels a row nothing acts on.
                if not recorded:
                    self.journal.append(
                        "sweep-decision-effect-unavailable",
                        dw_id=decision.id,
                        effect=option.effect,
                        error=(
                            "record_decision wrote no line: the ledger file is gone"
                            if not self.workspace.paths.deferred_work.is_file()
                            else "record_decision wrote no line: the ledger holds no entry for this id"
                        ),
                    )
                    any_effect_faulted = True
                    continue
                # the ledger read and wrote, so the doubt the last fault raised is
                # settled — whatever it left on disk is now committable, and this
                # walk now HAS something to publish
                ledger_in_doubt = False
                any_effect_landed = True
                self._emit("post_decision", story_key=decision.id, decision_action=option.effect)
                if option.effect == "close":
                    closed += 1
        if any_effect_landed and not ledger_in_doubt:
            # TWO conditions, and they are different questions. `ledger_in_doubt`
            # is the withhold: the last effect faulted, so whatever is on disk is
            # bytes nobody could read. `any_effect_landed` is the non-empty pass:
            # `_apply_decision_effect` is this walk's only ledger write, so a walk
            # that answered nothing (every decision pre-answered, skipped
            # unattended, or dropped) wrote nothing, and no git is spawned
            # (DW-183/DW-185). This is one of the FOUR sites gating on a write
            # result or returned effect; three gate on something weaker, and `path_clean`
            # is the uniform floor beneath all seven — the inventory is spelled out
            # at `_close_resolved`.
            #
            # The LEDGER FILE, not the project and not `self.workspace.root`: this
            # phase's write went to the ledger, and `implementation_artifacts` is
            # configurable to any absolute path (see `_commit_ledger`).
            self._commit_ledger(
                "chore(sweep): record deferred-work decisions",
                path=self.workspace.paths.deferred_work,
                family="ledger",
            )
        if answered_interactively:
            self._return_after_decisions(every_effect_landed=not any_effect_faulted)
        return answers, closed

    def _return_after_decisions(self, *, every_effect_landed: bool) -> None:
        """Once the human has answered this cycle's decisions over an attached
        terminal, hand it back so the sweep runs its bundles in the background —
        detach a plain-shell client, switch a tmux client back to its origin. A
        plain foreground sweep (nobody attached, no return target) is untouched.

        We then go unattended for the rest of the run: a later --repeat cycle's
        input() would otherwise block forever in a window no one is viewing. New
        decisions defer via the unattended path instead, recorded for
        `bmad-loop decisions` or the next attended sweep.

        The trigger for that is "nobody can be relied on to answer here any
        more", NOT "the hand-back succeeded" — the two come apart on a failed
        return, in opposite directions. A *refused* switch is evidence the
        client is still in this window with a human in front of it (ATTENDED:
        keep prompting, which is the whole point of #227). Everything else
        reports only that no hand-back was verified — a detach that found
        nothing attached, an effect the backend cannot observe, no detach verb
        at all, or a switch the backend cannot vouch for (a timed-out verb, an
        unreadable client count, nothing attached to move) — and under that
        uncertainty going unattended is the outcome that does not strand a
        --repeat cycle on input(); the decisions it defers stay reachable via
        `bmad-loop decisions`. The `sweep-return-no-client` record keeps its
        name across that widening: it has always meant "no hand-back verified",
        which is what an unvouched switch reports too. Only a real return is
        announced: UNREACHABLE prints nothing, since there may be no one to
        read it.

        `every_effect_landed` says only what the printed line may CLAIM, never
        whether to hand back: the trigger above is unchanged, so a walk in which
        every effect faulted still detaches and still goes unattended. It is
        REQUIRED and keyword-only — there is exactly one caller, and a default
        would make the optimistic claim the thing a new caller inherits by
        forgetting. Sticky over the whole walk, unlike the flag that gates the
        phase's commit: a PARTIAL miss is still a miss to the human, and the line
        says "not every decision" rather than claiming a total one either way. The
        answers themselves are on disk in `<run>/decisions.json`, which the next
        cycle reloads, so what is short is the ledger alone. `bmad-loop decisions`
        reconstructs unanswered questions from triage files and the project-level
        pre-answer store; it does not read these run-local answers.
        `sweep-returned-after-decisions` and every other branch are byte-identical
        either way: the journal records the hand-back, and the
        misses are already attributed by `sweep-decision-effect-unavailable`."""
        from .tui import launch  # import-light: launch.py has no textual imports

        outcome = launch.return_attached_client()
        if outcome is launch.ReturnOutcome.ATTENDED:
            return
        self.prompting = False
        if outcome is launch.ReturnOutcome.RETURNED:
            self.journal.append("sweep-returned-after-decisions")
            self.prompter.print_fn(
                _HANDBACK_RECORDED if every_effect_landed else _HANDBACK_LEDGER_MISS
            )
        else:
            self.journal.append("sweep-return-no-client")

    def _apply_decision_effect(self, decision: Decision, option: DecisionOption) -> bool:
        """Record the human's decision on its ledger entry, answering whether a
        `decision:` line actually landed.

        The boolean is the CALLER's non-write signal, not decoration (DW-186).
        `record_decision` answers False in exactly the two states that mean no line
        was written — no ledger file at all, and no entry carrying this `dw_id` —
        and True only when it wrote one. Discarded, those two states are
        indistinguishable from a successful write at the call site, which then
        counts a closure and announces a `post_decision` for an entry the ledger
        never received.

        What False does NOT say is that the ledger was readable: the missing-file
        arm answers before any read. So the caller treats it as a non-write and
        nothing more — see `_decisions_phase`, which leaves `ledger_in_doubt`
        untouched on it for exactly that reason.
        """
        ledger = self.workspace.paths.deferred_work
        detail = option.resolution or option.intent
        close_note = None
        if option.effect == "close":
            close_note = "closed by human decision" + (
                f": {option.resolution}" if option.resolution else ""
            )
        # ONE locked read->edit->write (#286/#469). As the `append_decision` +
        # `mark_done` pair it was two acquisitions with a window between them, and
        # a rival writer landing there saw an entry whose decision line says "close
        # it" and whose status still says open — a human answer half-recorded. The
        # bytes are identical to the pair's: `record_decision` inserts the decision
        # line before it applies the close, which is the order the pair produced.
        return deferredwork.record_decision(
            ledger, decision.id, self._today(), option.label, detail, close_note=close_note
        )

    def _commit_ledger(
        self, message: str, *, path: Path, family: Literal["ledger", "store"]
    ) -> None:
        """Publish the orchestrator bookkeeping FILE a phase just wrote: that one
        file reaches HEAD, and everything else the enclosing repository is
        carrying is left dirty for whoever owns it. No-op when the file already
        matches HEAD.

        The rule is NAME THE FILE YOU PUBLISHED. `path` is the file the caller
        just wrote, and everything else follows from it: it is resolved, both git
        calls run in the resolved parent, and both are pathspec'd to the resolved
        basename. Nothing is derived from a role ("the project owns sweep
        bookkeeping") — the two families name different files because they write
        different files:

        * the five ledger PUBLISHERS pass `self.workspace.paths.deferred_work`.
          The ledger hangs off `implementation_artifacts`, which
          `bmadconfig._resolve` accepts as any absolute path and
          `ProjectPaths.rebased` leaves unmoved when it sits outside the project
          — so it may be under the project, inside a disjoint `repo_root`, or in
          no repository at all, and git resolves the enclosing repository in all
          three. The WORKSPACE's copy, never `self.paths.deferred_work`, which is
          a different file in a different tree under worktree isolation.
        * the two pre-answer PRUNES pass `decisions.store_path(project)`. The
          pre-answer store is a bare join off the project root
          (`decisions.STORE_REL`) that no config knob can move, and the run dir is
          the anchor no workspace swap relocates.

        `self.workspace.root` is refused at every site. Where `repo_root` names a
        tree DISJOINT from the project it is the separate CODE repo, so a
        workspace-rooted commit interrogated a tree the write never touched
        (DW-160 for the prunes, DW-175 for the publishers): the clean check
        passed, nothing was committed, and the worktree carrying the edit stayed
        dirty ahead of this cycle's bundles. A hardcoded project root fails the
        mirror-image way for the publishers, which is why they do not use one.

        RESOLVED, following symlinks — and that rationale belongs to the LEDGER
        publishers alone (DW-188). Their writer is `platform_util
        .atomic_write_text`, whose default `follow_symlinks=True` resolves the
        target, so a ledger symlinked into the project has its TARGET rewritten.
        Against the lexical parent the two disagreed: the clean check interrogated
        the link's own directory while the bytes landed in the target's
        repository, which then received no commit at all. Resolving here is what
        makes the check, the commit and that write name one file, so
        `follow_symlinks=False` semantics would be exactly wrong for them —
        agreement with the writer is the property, not link-hardening.

        The two PRUNES are a different case and the resolve is not doing that job
        for them. Their writer is `decisions._write_store` via
        `atomic_write_text_confined`, which takes the OPPOSITE symlink policy:
        `follow_symlinks=False` refuses to write through a link planted at the
        store's own name, and a lexical parent walk refuses a redirected directory
        above it. So a store reached through a link is a shape that writer REFUSES
        rather than one it follows, and the resolve here can only ever agree with
        the plain path it did write. It stays uniform because a per-family
        spelling would claim a distinction the callers cannot act on — not because
        the two writers agree about links. Nothing here should be read as a
        promise that a symlinked pre-answer store works; its own writer says it
        does not.

        Both git calls see ONE scope: `verify.path_clean` checks the resolved
        basename and `verify.commit_paths` commits that same single path. That is
        what bounds the blast radius to the published file (DW-183/DW-185) — an
        operator's unrelated in-flight edits in the enclosing repository stay
        dirty rather than riding into a `chore(sweep):` commit — and what
        quarantines a withheld ledger from a LATER commit in the same cycle
        (DW-187): a prune commits the pre-answer store alone, so the undecodable
        bytes the decision phase refused to publish are still unpublished.
        `worktree_clean`'s `:(exclude)policy.toml` wart is gone with it: a single
        pathspec naming one published file cannot reach `policy.toml`.

        `commit_paths` rather than a narrow twin of `commit_story`: it already
        commits an exact path list, and it is stronger than a hand-rolled pair —
        it forces `:(literal)` pathspecs (an `implementation_artifacts` carrying a
        `[`, `*` or `?` reaches here verbatim from the operator's config, where a
        bare operand is a wildmatch glob), keeps a missing-but-TRACKED path as a
        deletion to stage, wraps its own root resolve in `GitError`, and answers
        `None` for "these paths held no change". `_commit_ledger` already knows
        that from `path_clean`, so a `None` here has TWO causes and not one: the
        pathspec went clean between the two calls, or the single operand survived
        neither the working tree nor the index (`commit_paths`' `if not rels:`
        arm, which drops a path git has never seen so one optional operand cannot
        hard-fail a whole commit) — a ledger deleted and left untracked after the
        check reported it dirty. Both mean nothing was published, which is what
        the row below announces.

        `path_clean` still runs FIRST, and is load-bearing rather than an
        optimization: `commit_paths` opens with `git add`, so without the check an
        already-clean publish would stage and re-interrogate a file it had nothing
        to say about — reaching git, and the index, for a non-event. Idempotent
        replays make that the ordinary case, not the rare one: a resumed cycle
        re-closing ids already `done` reproduces the committed bytes exactly.

        A `verify.GitError` degrades to a journal row naming the resolved
        directory and the error instead of propagating, and under this rule the
        degrade is REQUIRED rather than a kindness. `cli`'s sweep precondition only
        requires `paths.repo_root` to be a git repository, so neither the project
        nor a freestanding artifacts directory need be one, and `git status` there
        answers `fatal: not a git repository`. Letting the raise through would
        abort the whole sweep over bookkeeping that was always best effort —
        strictly worse than the missed commit it replaces.
        `decisions.apply_pre_answer` already degrades on `GitError` for this very
        file ("best effort, so a non-git or dirty tree never blocks the on-disk
        record") and this keeps them agreeing. The RESOLVE degrades to the same row
        for the same reason: `path.resolve()` can raise `OSError` (a broken link
        chain, a permission-denied component) or `RuntimeError` (a symlink loop),
        so both are caught alongside Git failures. `verify.commit_paths` and
        `verify.last_commit_for` guard their own resolves against the same pair.
        Best effort applies to Git publication and resolution only: journal I/O
        failures propagate, as they do for other journal writes.

        `path` is a REQUIRED keyword argument with no default, replacing the old
        `root=None` arm and its runtime raise. That is strictly louder, not
        laxer: a caller that forgets to name the file it dirtied now fails at call
        time and under pyright, before any run, instead of on whichever branch
        first reached the raise.

        `family` is REQUIRED and keyword-only for exactly that reason, and it is
        DECLARED rather than derived (DW-199/203/205). Before it, this method
        published whatever `path` named without ever asking whether that file was
        still there or still readable, and `verify.commit_paths` deliberately keeps
        a missing-but-TRACKED path as a DELETION to stage — so a ledger removed
        after the phase wrote it was committed as a deletion under a
        `chore(sweep):` message, and a resume whose ledger held undecodable bytes
        published them and only then raised on them. `_unpublishable` below is the
        guard, and it runs between the resolve and `path_clean`: after, because git
        is asked about the RESOLVED target (DW-188) and those are the bytes that
        would be published; before, because a refused publish must spawn no git at
        all — the same property the per-site guards buy. The two families need
        different validation (the five ledger publishers read through
        `deferredwork.read_for_write`, the two prunes check existence only), and
        the family is a caller's declaration because deriving it from the path
        would be precisely the "chosen by role" test the rule above refuses; a
        required keyword-only argument also makes a NEW call site fail under
        pyright rather than silently inherit a validation it does not want.

        A refusal journals `sweep-ledger-commit-refused` and returns, exactly as
        the other two no-op arms do — never a raise, because the read this guard
        takes is bookkeeping and not the sweep's own read. `refuse_cause` is one of
        TWO fixed tokens (`target-absent`, `target-unreadable`) and is minted for
        the reason `stop_cause` (DW-201), `drop_cause` and `regen_cause` were: the
        natural spelling is `reason`, which sits in
        `diagnostics._JOURNAL_DROP_FIELDS` and renders as a presence boolean, so a
        scrubbed dump could not tell the two causes apart. The decode or OS fault
        rides in `error` beside it, which is dropped, and `file` carries the same
        lexical basename the sibling rows do. What this guard does NOT do is
        rescue the run: `_loop`'s own `read_for_write` still raises on the same
        undecodable bytes right after the refusal, which is pre-existing behavior.
        What is gone is the publish that used to precede it.

        The guard NARROWS a window it does not close, and the residual is worth
        naming the way `_prune_dropped_pre_answer` names its own: a TRACKED target
        removed between `_unpublishable`'s probe and `commit_paths`' `git add` is
        still staged as a deletion. Closing it would mean changing
        `verify.commit_paths`, whose missing-but-tracked deletion contract other
        callers rely on, so it stays out of bounds here — and the residual is a
        genuine race (a file removed inside a few milliseconds by something that is
        not this sweep), where the shapes this guard exists for are steady states
        the publisher walked into deliberately.

        Both no-op outcomes journal `sweep-ledger-commit-clean` (DW-191).
        `path_clean` also answers True for an ignored path, so a ledger under a
        gitignored `implementation_artifacts` was previously skipped with no row
        at all. The shared row states only that nothing was published; no extra
        git call distinguishes ignored, unchanged or disappeared operands.
        Appends stay outside the guarded Git operations so a journal write fault
        cannot be misreported as a publication failure.

        `file` is the LEXICAL basename (`path.name`), never `target.name`, and
        that distinction is what makes it declarable (DW-192). The degrade row's
        other identifying field is `repo`, which — like `message` and `error`
        beside it — sits in `diagnostics._JOURNAL_DROP_FIELDS`, so a scrubbed dump
        retained nothing saying WHICH of the two published files went
        uncommitted. `file` is declared benign in
        `tests/test_portability_guard.py` and survives the scrub verbatim, and it
        can only be declared benign because every caller passes a code constant —
        `deferred-work.md` (`ProjectPaths.deferred_work`) or `decisions.json`
        (`decisions.STORE_REL`) — so the lexical tail is invariant by
        construction. The RESOLVED tail is not: the DW-188 resolve above follows
        a symlink to a target the OPERATOR named, so `target.name` can be
        arbitrary operator text of exactly the identifier shape `scrub_json`
        ships verbatim, and a benign row for it would be pre-approving that text.
        Bound beside `root` and before the `try` for the same reason `root` is,
        so the degrade row still names the file when the resolve is what failed.
        `repo` still carries the resolved directory for anyone reading the raw
        journal."""
        # `root` is bound inside the `try` because the resolve that derives it can
        # itself fail; until it succeeds the only directory known is the LEXICAL
        # parent, which is what the degrade row then names.
        root = path.parent
        # The LEXICAL tail, bound here rather than off `target` below: see the
        # docstring — it is a code constant at every caller, which is what lets it
        # be a benign (undropped) journal field, and the resolved tail is not.
        name = path.name
        # Bound ahead of the `try` so neither is possibly-unbound below it: the
        # refusal short-circuits past the two git calls, and `sha`'s `None` is the
        # same "nothing was published" the clean arm reads.
        sha: str | None = None
        refusal: tuple[str, str | None] | None = None
        try:
            target = path.resolve()
            root = target.parent
            # THE TARGET VALIDATION (DW-199/203/205), between the resolve and
            # `path_clean` for two reasons the docstring states: git is asked about
            # the RESOLVED target, and a refused publish must spawn no git at all.
            refusal = self._unpublishable(target, family)
            if refusal is None:
                # Preserve the clean short-circuit without catching journal write faults.
                clean = verify.path_clean(root, target.name)
                sha = None if clean else verify.commit_paths(root, message, [target])
        except (verify.GitError, OSError, RuntimeError) as e:
            # `repo` (not `root`): an absolute host path naming a git tree,
            # already routed out of diagnostics dumps, exactly as
            # `rearm-baseline-advance-failed` spells the same value. The RESOLVED
            # directory, which is the one git was actually asked about — or the
            # lexical parent when the resolve is what failed. `file` is what
            # SURVIVES a dump: `repo`, `message` and `error` are all dropped.
            self.journal.append(
                "sweep-ledger-commit-unavailable",
                message=message,
                repo=str(root),
                error=str(e),
                file=name,
            )
            return
        if refusal is not None:
            cause, error = refusal
            # Outside the guarded git block, like every other row here, so a journal
            # write fault is never misreported as a publication failure. `error`
            # only where the refusal HAS a fault to attribute — an absent target has
            # no exception text, and an empty string would read as one.
            extra = {} if error is None else {"error": error}
            self.journal.append(
                "sweep-ledger-commit-refused",
                message=message,
                file=name,
                refuse_cause=cause,
                **extra,
            )
            return
        if sha is None:
            # Already clean/ignored, or raced clean between the two calls. Absence
            # reaches here only as that RACE — a target removed after the guard
            # above read it and left untracked, which is `commit_paths`' `if not
            # rels:` arm. A plainly-absent target never gets this far; it took the
            # refusal arm before `path_clean` ran.
            self.journal.append("sweep-ledger-commit-clean", message=message, file=name)
            return
        self.journal.append("sweep-ledger-commit", message=message, commit=sha, file=name)

    def _unpublishable(
        self, target: Path, family: Literal["ledger", "store"]
    ) -> tuple[Literal["target-absent", "target-unreadable"], str | None] | None:
        """Why `target` must not be published, or `None` when it may be. Returns
        `(refuse_cause, error)` — the two fields the refusal row carries beyond
        `message` and `file`.

        Split out of `_commit_ledger` only so the two families read as the two
        different questions they are; it is not a seam anything else may call.

        The FAMILY is declared by the caller, never derived here. `path ==
        self.workspace.paths.deferred_work` would be exactly the "chosen by role"
        test `_commit_ledger`'s own rule refuses, and it would answer wrongly for a
        publisher whose ledger is symlinked (the argument is the RESOLVED target)
        or for any file a later caller publishes.

        LEDGER: `deferredwork.read_for_write`, because the ledger's own read
        contract (DW-146) already answers both questions in the two shapes this
        guard asks them — `None` for absence, `LedgerReadError` for bytes nobody
        can decode. Its `OSError` normally propagates; here it does not, because
        `_commit_ledger` is best-effort bookkeeping whose whole degrade discipline
        exists so a publication fault never aborts a sweep, so it joins the
        undecodable cause rather than escaping. No lock is taken: this is a read
        the writer above already took. A later disappearance or replacement can
        still change what git publishes, as `_commit_ledger` documents above.

        STORE: existence only, preserving the publisher's existing content
        policy. The writer emits valid UTF-8 JSON, but this guard does not check
        whether those bytes were replaced after the write. `_prune_pre_answers`'
        own DW-176 absence refusal is about the LEDGER it reads, not the store.

        Both probes are taken on the RESOLVED argument, which is what decides what
        the `is_symlink()` disjunct actually buys — and it is not what the spelling
        suggests. A DANGLING link does not survive the resolve as a link: non-strict
        `Path.resolve` collapses it to the plain non-existent path it points at, so
        both probes answer False and the store is refused `target-absent`. That is
        the right answer for it (the prune's writer,
        `atomic_write_text_confined`, REFUSES to write through a link at the
        store's own name, so a dangling one holds no write of ours to publish), but
        it means the disjunct is doing a different job: on Python 3.13+, a symlink
        LOOP resolves to the link ITSELF, which `exists()` calls False and
        `is_symlink()` calls True. The disjunct preserves publication of that link
        entry. Python 3.11–3.12 instead raise during resolve, taking the existing
        `sweep-ledger-commit-unavailable` arm before this helper runs.

        Returns the `refuse_cause` token as a `Literal` rather than a bare `str`,
        which is what makes the closed two-value claim
        `tests/test_portability_guard.py` declares `refuse_cause` benign on a
        typechecked property rather than a comment."""
        if family == "ledger":
            try:
                if deferredwork.read_for_write(target) is None:
                    return ("target-absent", None)
            except (deferredwork.LedgerReadError, OSError) as e:
                return ("target-unreadable", str(e))
            return None
        if family == "store":
            if not (target.exists() or target.is_symlink()):
                return ("target-absent", None)
            return None
        # Spelled as an exhaustive dispatch, not `if ledger / else store`: a THIRD
        # family added to the `Literal` would otherwise typecheck at every call site
        # and fall silently through to existence-only validation — precisely the
        # "inherit a validation it does not want" failure the required keyword-only
        # argument exists to prevent. This reds under pyright the moment the union
        # grows, before any run.
        assert_never(family)

    # ---------------------------------------------------------- bundles

    def _agreeing_option(
        self, decision: Decision, answer: dict[str, Any], answer_key: str
    ) -> DecisionOption | None:
        """The stored answer's key resolved against THIS cycle's decision, but only
        when the option it lands on is still the one the human answered.

        `Decision.option` matches on KEY ALONE, and a key is a position in a list a
        later triage re-authors freely: `_ensure_triage` mints a fresh
        `triage-<n>.json` per repeat cycle while `answers` persists for the whole run
        in `<run>/decisions.json`, and a pre-answer is resolved against a triage
        minted after it was recorded. Either provenance can hand a caller ONE
        question's answer beside a DIFFERENT question's option (DW-118: a stored
        `build` answer keyed "1" met a fresh option "1" spelled "Close as decayed",
        and the bundle shipped the stale intent under the close label). `label` +
        `effect` is the whole agreement test — the only two fields BOTH provenances
        always carry (`record_pre_answer` stores the chosen option's full semantics;
        an in-run answer is written with key/label/effect/answered_at) — and a
        disagreeing option is discarded outright, its mismatch journaled the way
        `sweep-bundle-name-discarded` is.

        ONE agreement discipline for both lanes of `_materialize_bundles` (DW-123):
        the build lane had this test inline while the keep-open lane trusted the
        stored `effect` with no resolution at all, so a renumbered option let a stale
        keep-open answer suppress a bundle under a `human-chose-keep-open` skip that
        reads as the human's decision. What the two lanes still differ on is the
        DISPOSITION of a `None` — see each call site.
        """
        option = decision.option(answer_key)
        if option is None:
            return None  # nothing resolved, so there is nothing to describe
        label_matched = option.label == _answer_str(answer, "label")
        if label_matched and option.effect == _answer_str(answer, "effect"):
            return option
        # No triage prose in the record (labels, questions): the fields are closed
        # effect enums and a bare boolean. `answer_effect` says which LANE wrote the
        # record — it is the stored answer's own effect, invariant per lane but no
        # longer invariant across the two that reach here, and it is what separates a
        # discarded build option from a discarded keep-open one in a journal both
        # write with the same kind.
        self.journal.append(
            "sweep-decision-option-mismatch",
            decision=decision.id,
            key=answer_key,
            option_effect=option.effect,
            label_matched=label_matched,
            answer_effect=_answer_str(answer, "effect"),
        )
        return None

    def _materialize_bundles(
        self, plan: TriagePlan, answers: dict[str, dict[str, Any]]
    ) -> tuple[list[Bundle], bool]:
        """This cycle's bundles, and whether ANY recorded answer was dropped by one
        of the three drop lanes below — `_cycle`'s progress signal.

        Every drop is progress for the same reason (DW-123, widened to the build
        lanes by DW-135): it quarantines the id in `state.sweep_dropped_decisions`,
        so the id stops being bound to a stored answer nothing can act on and a
        later cycle's fresh triage is free to address it. The signal stays finite
        because that same list bounds each id to ONE drop per run — persisted, so
        the bound holds across a pause/resume too (DW-124) — and a given id can
        raise it at most once however many repeat cycles run.
        """
        self._emit("pre_materialize_bundles")
        bundles = list(plan.bundles)
        answer_dropped = False
        for decision in plan.decisions:
            answer = answers.get(decision.id)
            # `isinstance` rather than truthiness: `answers`' annotation is a
            # contract this method cannot enforce, and the test suite is the caller
            # that hands it a map directly rather than through `_cycle`. Inside
            # `src/` the only caller IS `_cycle` (a resume re-enters there too), so
            # `_decisions_phase`'s read-site guard covers the production path — but
            # a lane that trusts the annotation aborts materialization on a
            # `.get(...)` the moment anything else supplies the map. A silent skip
            # either way: an unusable answer is journaled where it is read, not
            # once per lane that declines to use it.
            if not isinstance(answer, dict) or answer.get("effect") != "build":
                continue
            if decision.id in self.state.sweep_dropped_decisions:
                continue  # announced dropped earlier this run (see __init__)
            # ONE spelling of the key for the whole loop body: the lookup, the
            # mismatch record and the note below must name the same string, and
            # `str(answer.get("key"))` stringified a missing key to the literal
            # "None" while the record spelled it "" — and a non-string key to its
            # repr, which `_answer_str` reads as "" instead (DW-141).
            answer_key = _answer_str(answer, "key")
            # `matched` is exactly "an agreeing option was resolved": the helper
            # collapses the two ways that can fail (no such key / a re-authored one)
            # because this lane treats them alike. It tolerates BOTH — a build answer
            # carries its own `intent` payload and can still build from it — and
            # drops only when that payload is missing, a few lines below. The
            # keep-open lane has no payload to fall back on, so it cannot.
            option = self._agreeing_option(decision, answer, answer_key)
            matched = option is not None
            # The stored answer is the PAYLOAD; an agreeing option fills only what
            # the answer omits (`answer or option`, not the reverse). That single
            # expression routes both provenances without a provenance flag:
            # `record_pre_answer` stores the chosen option's full semantics and
            # `validate_triage` requires `intent` on every build option, so a build
            # PRE-answer always carries its own intent and never picks up prose
            # freshly re-authored by a triage the human never read; an IN-RUN
            # answer is written with only key/label/effect/answered_at, so it draws
            # intent and bundle_name from the option — but only an agreeing one.
            intent = _answer_str(answer, "intent") or (option.intent if option else "")
            if not intent:
                # A stale in-run answer: nothing to build from. Dropping it is the
                # only safe action here — where `_apply_decision_effect` landed this
                # decision's ledger line in the cycle that answered it, re-asking or
                # re-applying would double-apply. (Since DW-186 that call can report
                # it wrote no line at all, in which case there is nothing to
                # double-apply and dropping is still what this lane does — the
                # entry is left open and the next sweep re-asks it, which is the
                # same outcome.) But a recorded human `build` decision must not
                # vanish on a journal line alone.
                # The ledger entry is untouched, so the next sweep re-triages and
                # re-asks it through `_decisions_phase`.
                # `drop_cause` is a closed three-value enum (`no-intent` here,
                # `name-collision` below, `stale-option` in the keep-open lane) so
                # the drop lanes are discriminated by an enum rather than by free
                # text or by a second journal kind (`reason` is deliberately not a
                # benign journal field).
                self.journal.append(
                    "sweep-decision-answer-dropped",
                    decision=decision.id,
                    drop_cause="no-intent",
                )
                gates.notify(
                    self.policy,
                    self.run_dir,
                    f"decision {decision.id}: recorded build decision discarded",
                    "its triage option changed and the stored answer carries no "
                    "intent of its own — the entry stays open for the next sweep",
                )
                self._quarantine(self.state.sweep_dropped_decisions, decision.id)
                answer_dropped = True  # progress: see this method's docstring
                continue
            label = _answer_str(answer, "label") or (option.label if option else "") or "build"
            bundle_name = _answer_str(answer, "bundle_name") or (
                option.bundle_name if option else ""
            )
            # A stored answer's bundle_name never passed `validate_triage` — it was
            # answered out of band against an earlier triage, and a fresh one can
            # renumber or drop the option it named — so this lane was the one route
            # by which a name failing the two option-site gates (#637) still reached
            # `_write_intent` as a directory. Gate it with the same two rules, plus
            # the THIRD rule that site enforces as `duplicate bundle name`: a stored
            # name equal to one already on this list makes both bundles hash to one
            # `_bundle_key` and share one intent directory, so one of them is
            # silently lost. All three by DISCARD rather than by error: the human's
            # build decision is the payload and `decision-<id>` below is the
            # always-legal name it falls back to anyway, so the discard is journaled
            # the way `_normalize_bundle_names`'s repairs are and the sweep proceeds.
            # Why a colliding STORED name is discarded here while the fallback below
            # is SUFFIXED, two remedies for one collision condition: a stored name
            # has somewhere to fall back TO, and falling back is the better repair —
            # it is unvalidated prose carried by an answer whose option may be gone,
            # so a `widen-x-2` variant of it claims a name nothing authored. The
            # fallback has nothing below it, so suffixing is the only repair left.
            if bundle_name and (
                not BUNDLE_NAME_RE.match(bundle_name)
                or safe_segment(bundle_name) != bundle_name
                or any(b.name == bundle_name for b in bundles)
            ):
                self.journal.append(
                    "sweep-bundle-name-discarded",
                    decision=decision.id,
                    original=bundle_name,
                )
                bundle_name = ""
            key = (option.key if option else "") or answer_key or "?"
            name = bundle_name or "decision-" + decision.id.lower()
            # `decision-<id>` READS like a reserved namespace and is not one:
            # `validate_triage` builds its duplicate-name set from plan bundle
            # names and build-option `bundle_name`s only, so a triage plan may
            # legally author a bundle literally named `decision-dw-118` and
            # nothing ever compares this fallback against it. Downstream,
            # `_bundle_key` is a pure function of the name, so two same-named
            # `Bundle`s become ONE task: `_run_bundle` returns early on a
            # terminal task, or writes the second's `intent.md` over the first's
            # under the same dirname, and the human's decision bundle disappears
            # without a record. Reserving the prefix upstream was rejected (it
            # changes the triage-plan contract, escalates one unlucky
            # LLM-authored name into a whole-plan rejection, and still misses a
            # STORED name shaped `decision-<other-id>`, which never passes
            # `validate_triage` at all), so uniqueness is re-established here —
            # the one site where validated plan names, validated option names,
            # unvalidated stored-answer names and the fallback all meet. The
            # taken set is recomputed per decision, never snapshotted before the
            # loop: it must cover the decision bundles appended by earlier
            # iterations, which collide with each other the same way.
            taken = {b.name for b in bundles}
            if name in taken:
                for attempt in range(2, 10):
                    candidate = f"{name}-{attempt}"
                    if candidate not in taken:
                        # `name=` so the record stands on its own, the way its
                        # sibling `sweep-bundle-name-discarded` carries `original=`:
                        # without it the resulting name has to be re-derived by hand
                        # from the id and the suffix.
                        self.journal.append(
                            "sweep-bundle-name-deduped",
                            decision=decision.id,
                            attempt=attempt,
                            name=candidate,
                        )
                        name = candidate
                        break
                else:
                    # The one point in NAME ASSIGNMENT at which a buildable stored
                    # answer yields no bundle — a naming impossibility, not a
                    # mismatch disposition, and bounded so the loop is provably
                    # finite. (Scoped to this step deliberately: an already-named
                    # decision bundle can still be removed further down by the
                    # failed/keep-open skip or by the max_bundles truncation.) Loud
                    # on both surfaces, like the no-intent drop it shares a kind
                    # with.
                    self.journal.append(
                        "sweep-decision-answer-dropped",
                        decision=decision.id,
                        drop_cause="name-collision",
                    )
                    gates.notify(
                        self.policy,
                        self.run_dir,
                        f"decision {decision.id}: recorded build decision discarded",
                        f"its bundle could not be given a name unique among this "
                        f"cycle's bundles ({name} and every -2..-9 suffix are "
                        "taken) — the entry stays open for the next sweep",
                    )
                    self._quarantine(self.state.sweep_dropped_decisions, decision.id)
                    answer_dropped = True  # progress: see this method's docstring
                    continue
            bundles.append(
                Bundle(
                    name=name,
                    dw_ids=(decision.id,),
                    intent=intent,
                    decision_note=(
                        f"The human chose option {key} ({label}) for the "
                        f"question: {decision.question}"
                        if matched
                        # Never quote `decision.question` here: the option this
                        # answer names has since been re-authored, so the question
                        # now on file is not the one the human answered.
                        else f"The human chose option {key} ({label}) against an "
                        f"earlier triage of {decision.id}, whose options have "
                        "since changed. The stored answer's own intent above is "
                        "the contract; the question now on file is not the one "
                        "it answered."
                    ),
                )
            )
        # ids a prior bundle already failed on: re-triaging them would rebuild
        # the same hopeless bundle every repeat cycle (and a cached build-effect
        # decision answer would re-materialize its bundle each cycle)
        failed_ids = {
            i
            for t in self.state.tasks.values()
            if t.story_key.startswith("dw") and t.phase in (Phase.DEFERRED, Phase.ESCALATED)
            for i in t.dw_ids
        }
        # ids a human explicitly chose to keep open: a later triage must not
        # override that answer (bundle dev sessions mark their dw_ids done). Held to
        # the SAME agreement test the build lane above runs (DW-123): this set is
        # read straight off `answers`, whose entries outlive the triage they were
        # answered against, so an unresolved `effect == "keep-open"` let a stale
        # answer suppress an overlapping bundle — journaled only as a
        # `human-chose-keep-open` skip, which reads as the human's decision on a
        # question this cycle never asked.
        #
        # DW-133 proposed gating the stale-option drop below on overlap with THIS
        # cycle's bundles. REFUTED (2026-09-06, human-resolved) — do not
        # re-propose. Two placements are possible and both are wrong:
        #
        # At the drop itself the gate is UNREACHABLE, so it buys nothing.
        # `validate_triage`'s `claim()` (see :178) records every id in one `seen`
        # map and errors on "appears in both", so `plan.bundles` and
        # `plan.decisions` are disjoint by validation; the drop is reached only
        # when `by_id.get(dw_id)` is not None — the id IS in `decisions`, hence in
        # no plan bundle — and a keep-open answer mints no decision bundle of its
        # own (that needs `effect == "build"`, which this lane's own guard
        # excludes). Measured: with the condition replaced by a `raise`, the whole
        # of tests/test_sweep.py passes — it never once fires.
        #
        # Hoisted ABOVE the `decision is None` arm it stops being a no-op and
        # starts doing harm, since that arm is exactly where a kept answer DOES
        # overlap a bundle: it suppresses that bundle, which is what keep-open
        # means. Measured: three tests red, `test_repeat_keep_open_answer_blocks_rebundle`
        # among them — the gate breaks legitimate suppression rather than the drop.
        #
        # Underneath both: the drop's forward-looking timing is load-bearing BY
        # DESIGN. It must fire in the cycle that PROVES the answer stale — where
        # the id is in `decisions` and so in no bundle — so that a LATER cycle's
        # bundle is not silently suppressed. `tests/test_sweep.py`'s
        # `test_keep_open_answer_whose_option_was_re_authored_stops_suppressing_bundles`
        # is the shape to keep in view: its cycle 2 holds DW-1 in `decisions` with
        # no bundles (where the drop must fire) and only cycle 3 bundles DW-1, so
        # any rule keyed on this cycle's bundles can never see them together.
        by_id = {d.id: d for d in plan.decisions}
        keep_open_ids: set[str] = set()
        for dw_id, answer in answers.items():
            # Shape-guarded for the same reason the build lane above is.
            if not isinstance(answer, dict) or answer.get("effect") != "keep-open":
                continue
            if dw_id in self.state.sweep_dropped_decisions:
                continue  # announced dropped earlier this run (see __init__)
            decision = by_id.get(dw_id)
            if decision is None:
                # No decision for this id THIS cycle — the fresh triage bundled or
                # closed it directly instead of re-asking. There is no option to
                # disagree with, so the answer is the only record of the human's
                # choice and it stands: suppressing the bundle is exactly what
                # keep-open means.
                keep_open_ids.add(dw_id)
                continue
            answer_key = _answer_str(answer, "key")
            if self._agreeing_option(decision, answer, answer_key) is not None:
                keep_open_ids.add(dw_id)
                continue
            # Unlike the build lane, a keep-open answer has no payload beyond
            # "keep-open" itself, so without a currently-resolvable, agreeing option
            # there is nothing left to trust and the answer is dropped. Hence a THIRD
            # `drop_cause` covering both failures — a renumbered option (which wrote
            # a mismatch record just now) and a vanished one (which could not) —
            # rather than one named for the mismatch alone. Dropping is deliberately
            # the loud direction: honouring a stale keep-open answer silently skips
            # work the human never protected, while dropping it is journaled and
            # notified. What this drop does that the build lanes' do not is UNBLOCK
            # the id: a keep-open answer actively suppresses bundles, so removing it
            # makes the id eligible for a bundle a later valid triage cycle can run
            # and close the entry with, where a dropped build answer simply leaves
            # the entry open to be re-asked. Both count as repeat progress (DW-135;
            # this method's docstring says why). The
            # RUN-LOCAL record is what survives — the answer stays auditable in
            # `<run>/decisions.json` and the ledger line `_apply_decision_effect`
            # wrote is unchanged — while an out-of-band pre-answer in the PROJECT
            # store is pruned by the drop itself, below (DW-143): waiting for
            # `_prune_pre_answers` to retire it once a later bundle closed the entry
            # never came due while triage kept re-asking the id as a decision, so
            # every new run re-read the same stale answer and re-dropped it.
            self.journal.append(
                "sweep-decision-answer-dropped",
                decision=dw_id,
                drop_cause="stale-option",
            )
            # Which of the two failures fired, named rather than left to the
            # journal: the notify is the surface an operator actually reads, and
            # "changed" is wrong for a key this triage simply does not offer.
            fate = (
                "is gone from this cycle's triage"
                if decision.option(answer_key) is None
                else "has been re-authored since"
            )
            gates.notify(
                self.policy,
                self.run_dir,
                f"decision {dw_id}: recorded keep-open decision discarded",
                f"the option it answered ({answer_key}) {fate}, so the keep-open "
                f"protection is discarded and {dw_id} is eligible for bundling again",
            )
            self._quarantine(self.state.sweep_dropped_decisions, dw_id)
            # After the row, the notify and the quarantine — announce-then-persist,
            # so a crash mid-drop resumes into the DW-124 skip rather than into a
            # silent removal (the helper's docstring has the full argument). Keep-
            # open only: the build lanes' `no-intent`/`name-collision` drops leave
            # their stored answer alone, since it still carries a payload to re-ask
            # against. `<run>/decisions.json` and the ledger are untouched either
            # way — only the PROJECT store entry goes, and only while it is still
            # THIS answer: a replacement a human recorded out of band since this
            # run last read the store is not the value being dropped, and survives.
            self._prune_dropped_pre_answer(dw_id, "stale-option", answer)
            answer_dropped = True
        kept = []
        for b in bundles:
            overlap = sorted(set(b.dw_ids) & (failed_ids | keep_open_ids))
            if overlap:
                self.journal.append(
                    "sweep-bundle-skipped",
                    name=b.name,
                    dw_ids=overlap,
                    reason=(
                        "failed-or-escalated-earlier"
                        if set(b.dw_ids) & failed_ids
                        else "human-chose-keep-open"
                    ),
                )
                continue
            kept.append(b)
        bundles = kept
        if len(bundles) > self.max_bundles:
            dropped = [b.name for b in bundles[self.max_bundles :]]
            self.journal.append("sweep-bundles-truncated", dropped=dropped)
            bundles = bundles[: self.max_bundles]
        self._emit("post_materialize_bundles")
        return bundles, answer_dropped

    def _write_intent(self, bundle: Bundle, dirname: str) -> Path:
        ledger = self.workspace.paths.deferred_work
        # REPAIR/WRITE (DW-146): these bytes become the bundle intent file a
        # session is dispatched on — an empty one would brief the session on
        # nothing at all.
        text = deferredwork.read_for_write(ledger) or ""
        entries = {e.id: e for e in deferredwork.parse_ledger(text)}
        blocks = [entries[i].body.rstrip() for i in bundle.dw_ids if i in entries]
        lines = [
            f"# Deferred-work bundle: {bundle.name}",
            "",
            f"bundle_name: {bundle.name}",
            _INTENT_DW_IDS_PREFIX + ", ".join(bundle.dw_ids),
            "",
            "## Intent",
            "",
            bundle.intent,
        ]
        if bundle.decision_note:
            lines += ["", "## Human decision", "", bundle.decision_note]
        lines += ["", "## Ledger entries (verbatim)", "", "\n\n".join(blocks), ""]
        path = self.run_dir / "bundles" / dirname / "intent.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        # Surrogates are neutralized over the whole document, not per field: the
        # triage-authored `intent`/`decision_note` are the ones that can revive one
        # (#329), but a document-wide pass covers whatever prose is added here
        # later. Line breaks are deliberately *kept* — this file is markdown, so
        # `_one_line`'s collapse would be damage, and the ledger blocks are read
        # back from a strict-UTF-8 file and so pass through byte-unchanged.
        atomic_write_text(path, neutralize_surrogates("\n".join(lines)))
        return path

    def _bundle_intent_reason(self, task: StoryTask) -> str | None:
        """Grade the persisted intent document against the task that owns it.
        Returns ``None`` to reuse it untouched, or the reason
        `_ensure_bundle_intent` must regenerate: ``"missing"`` (no `bundle_file`,
        or it is not a file), ``"dw-ids-mismatch"`` (the document's ``dw_ids:``
        line names a different SET than `task.dw_ids`, or carries no such line at
        all), ``"unreadable"`` (the read faulted or the bytes would not decode).

        DW-164: `_run_bundle` writes `task.bundle_file` and only then `_save()`s
        the adopted ids, so a crash between them leaves the persisted ids OLD and
        the document NEW. Re-ordering the two writes does not close that hole, it
        only inverts it — persisted ids NEW, document OLD — and both shapes pair a
        task with a document naming other ids. Grading the document against
        `task.dw_ids` (the field the ledger close, the key dedupe and the dev
        prompt all key on) is TOTAL over both, and the degraded rebuild it triggers
        is exactly the recovery `_ensure_bundle_intent` already exists to perform.

        An EMPTY `task.dw_ids` is deliberately NOT an authority: that is the
        pre-`dw_ids` `state.json` shape, and grading a real document against it
        would trade the bundle's actual brief for a degraded one naming nothing.
        Such a task keeps whatever document it has.

        Bundle identity is SET equality here, as `_bundle_name_for` and
        `_run_bundle` already define it, so a `_write_intent` line whose ids are
        merely reordered still agrees."""
        if not task.bundle_file:
            return "missing"
        path = Path(task.bundle_file)
        if not path.is_file():
            return "missing"
        if not task.dw_ids:
            return None
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return "unreadable"
        for line in text.splitlines():
            if not line.startswith(_INTENT_DW_IDS_PREFIX):
                continue
            rest = line[len(_INTENT_DW_IDS_PREFIX) :]
            found = {part.strip() for part in rest.split(",") if part.strip()}
            return None if found == set(task.dw_ids) else "dw-ids-mismatch"
        return "dw-ids-mismatch"

    def _ensure_bundle_intent(self, task: StoryTask) -> None:
        """Guarantee a recovered bundle has the intent file its dev prompt points
        at, and that the file it points at is the one for THIS task's ids. The
        rendered intent.md persists in the run dir and the prompt consumes nothing
        else from the plan, so the normal case is to reuse it untouched.

        Only when `_bundle_intent_reason` rejects it — gone, unreadable, or naming
        other ids — do we rebuild a degraded one from the task itself. The triage
        session's authored intent prose is the single unrecoverable piece; the
        verbatim ledger entries _write_intent re-attaches carry the actual work, so
        say plainly that they are now the contract."""
        reason = self._bundle_intent_reason(task)
        if reason is None:
            return
        match = BUNDLE_KEY_RE.match(task.story_key)
        if match is None:  # pragma: no cover - callers filter on BUNDLE_KEY_RE
            return
        cycle = int(match.group(1)) if match.group(1) else 1
        name = match.group(2)
        bundle = Bundle(
            name=name,
            dw_ids=tuple(task.dw_ids),
            intent=(
                "Resolve the deferred-work entries reproduced below. This bundle's "
                "original triage intent did not survive the run it was written in, "
                "so the verbatim ledger entries are the authoritative statement of "
                "the work."
            ),
        )
        dirname = name if cycle == 1 else f"c{cycle}-{name}"
        task.bundle_file = str(self._write_intent(bundle, dirname))
        self.journal.append(
            "sweep-intent-regenerated",
            story_key=task.story_key,
            dw_ids=list(task.dw_ids),
            path=task.bundle_file,
            # `regen_cause`, not `reason`: `diagnostics._JOURNAL_DROP_FIELDS` holds
            # `reason` as free text and renders it as a presence boolean, which
            # would defeat this field's whole purpose. Closed-slug siblings in
            # this file (`drop_cause`) use the same convention for the same reason.
            regen_cause=reason,
        )

    # ------------------------------------------------------ override seams

    def _dispatched_spec_for_attempt(self, task: StoryTask) -> str | None:
        """Sweep dispatch owns intent.md, never an accepted bundle spec."""
        return None

    def _requires_dispatched_spec_snapshot(self, task: StoryTask, prompt: str) -> bool:
        """Keep explicit bundle-spec routing separate from recovery ownership.

        Repair and patch-restore prompts name the accepted spec so deterministic
        read-back follows it, but Sweep still owns ``intent.md`` as its dispatched
        input and must never promote that result artifact into attempt ownership.
        """
        return False

    def _retains_dispatched_spec_snapshot_on_repair(self) -> bool:
        """Sweep repairs remain owned by intent.md, not the accepted spec."""
        return False

    def _dev_prompt(self, task: StoryTask, feedback: Path | None) -> str:
        return self._generic_bundle_prompt(task, feedback)

    def _generic_bundle_prompt(self, task: StoryTask, feedback: Path | None) -> str:
        """Bundle invocation for the generic dev primitive (disk-resolved, see
        ``Engine._dev_skill``): the self-contained
        intent.md (intent + verbatim ledger entries) is handed over as freeform
        intent. The orchestrator owns the deferred-work ledger — the skill is told
        not to edit it — and records resolution itself in `_post_dev_accepted_sync`.
        On a repair the bundle spec is re-opened first (B6) so step-01 resumes.

        A patch-restore re-drive (#2564, #75) must point at the bundle spec
        explicitly: only step-01's spec-pointer intent check EARLY EXITs on the
        `in-review` status the re-arm set — before step-01's version-control
        sanity check, which would otherwise HALT `blocked` on the diff
        `_restore_patch` just laid onto the tree. The freeform intent.md pointer
        takes the path where that dirty-tree check runs first."""
        bundle_ref = task.bundle_file or task.story_key
        if feedback is None:
            if task.restore_patch and task.spec_file:
                return (
                    f"/{self._dev_skill()} Resume review of the in-review spec at "
                    f"`{task.spec_file}` for the deferred-work bundle `{bundle_ref}`. "
                    f"The attempted change was restored onto the working tree after "
                    f"an intent-gap resolution; review it against the amended spec. "
                    f"Do NOT edit the deferred-work ledger; the orchestrator records "
                    f"resolution."
                )
            return (
                f"/{self._dev_skill()} Implement the deferred-work bundle described in "
                f"`{bundle_ref}` — it carries the intent and the verbatim ledger "
                f"entries to resolve. Do NOT edit the deferred-work ledger; the "
                f"orchestrator records resolution."
            )
        self._reset_spec_for_repair(task)
        spec_ref = task.spec_file or bundle_ref
        return (
            f"/{self._dev_skill()} Resume the autonomous dev session on the in-progress "
            f"spec at `{spec_ref}` for the deferred-work bundle `{bundle_ref}`. The "
            f"previous session's work failed deterministic verification; repair the "
            f"working tree so verification passes without changing the frozen intent "
            f"contract or editing the deferred-work ledger. Verification evidence is "
            f"in `{feedback}`."
        )

    def _post_dev_state_sync(self, task: StoryTask, result_json: dict | None) -> None:
        """No-op: bundles have no sprint-status row for the pre-gate sync.

        This override and the accepted-only override below are one behavior
        change. Leaving the former close here as well would run bundle closure
        twice at two different gate positions.
        """
        return

    def _post_dev_accepted_sync(self, task: StoryTask, result_json: dict | None) -> None:
        """Generic-path ledger single-writer for bundles. The decoupled
        bmad-build-auto skill does not touch the ledger, so the orchestrator marks
        each dw id the bundle owns ``done`` once the bundle's spec reaches the
        terminal status for the current stage. No-op on the legacy path.

        This runs only after the artifact gate, verify commands, and ``decide_dev``
        have accepted the attempt. In particular, ``outcome.ok`` is insufficient:
        a CRITICAL escalation in the session result preempts that outcome. The
        review gate later requires these entries closed; ``_verify_review`` retains
        its separate reclose because a review session can rewrite the ledger.
        """
        if not self._generic_dev():
            return
        spec_file = result_mapping(result_json).get("spec_file")
        if not spec_file:
            return
        success_status = "in-review" if self._dev_review_enabled() else "done"
        self._close_bundle_ledger_when_spec_status(task, str(spec_file), success_status)

    def _bundle_close_operation_id(self, task: StoryTask) -> str:
        """Stable identity for a close and its possible defer-time undo."""
        return f"{self.state.run_id}/{task.story_key}"

    def _bundle_close_note(self, task: StoryTask) -> str:
        """Resolution note shared by a bundle close and its possible undo."""
        return f"resolved by sweep bundle {task.story_key}"

    def _close_declared_deferred(
        self, task: StoryTask, snapshot: list[_ArmedClose] | None = None
    ) -> None:
        """No-op: a bundle's ledger closure is owned by
        ``_close_bundle_ledger_when_spec_status``, which runs after accepted dev
        because ``verify_review_bundle`` *requires* those entries closed before the
        later commit boundary. Letting the base class's commit-boundary hook (#234)
        also fire here would re-derive closure for a task whose ids come from
        ``task.dw_ids``, not from a ``closes_deferred:`` declaration."""

    def _close_bundle_ledger_when_spec_status(
        self,
        task: StoryTask,
        spec_file: str,
        success_status: str,
        kind: str = "sweep-bundle-closed",
    ) -> None:
        spec_path = verify.resolve_spec_path(spec_file, self.workspace.paths)
        if not spec_path.is_file():
            return
        fm = self._observed_frontmatter(spec_path, task.story_key, "bundle-ledger-close")
        if fm is None:
            return
        if verify.status_of(fm) != success_status:
            return
        ledger = self.workspace.paths.deferred_work
        note = self._bundle_close_note(task)
        # Record the intended ids, never only `marked`. This method is called once
        # after accepted dev and again by the review-leg reclose. The second call
        # normally finds every entry already done, so `marked` is empty; deriving
        # the record from it would erase exactly the state a landing bundle needs.
        task.bundle_closes_intended = list(task.dw_ids)
        marked = deferredwork.mark_done_many_reopenable(
            ledger,
            task.dw_ids,
            self._today(),
            note,
            self._bundle_close_operation_id(task),
        )
        if marked:
            self.journal.append(kind, story_key=task.story_key, dw_ids=marked)

    def _reopen_ledger_after_defer(self, task: StoryTask) -> None:
        """Reopen only this run's bundle closes after its code was discarded.

        ``Engine._defer`` deliberately restores the whole post-review ledger after
        rollback so harvested findings survive. That restore can also replay a
        bundle close whose code no longer exists. The operation-specific undo marker
        makes this "undo my close", not "open these ids": human, legacy, and earlier
        run closures remain untouched. Replaying this method is idempotent.
        """
        ledger = self.workspace.paths.deferred_work
        note = self._bundle_close_note(task)
        operation_id = self._bundle_close_operation_id(task)
        # ONE locked read->edit->write (#286/#469): the per-id `mark_open`
        # comprehension this replaces took the lock once per id, and a rollback
        # that leaves some closes undone and others standing is the one shape this
        # method exists to prevent. Order and skip semantics are `mark_open_many`'s
        # own, which are the comprehension's.
        reopened = deferredwork.mark_open_many(ledger, list(task.dw_ids), note, operation_id)
        if reopened:
            self.journal.append("sweep-bundle-reopened", story_key=task.story_key, dw_ids=reopened)

    def _carry_isolated_ledger_writes(self, task: StoryTask) -> None:
        """The base hook's sweep half: re-apply this bundle's ledger CLOSES to the
        MAIN checkout after an isolated unit lands, the base having first carried
        the harvest.

        ``super()`` runs FIRST, and that is a contract rather than a style choice.
        ``deferredwork.append_entry``'s idempotence scan is OPEN-ONLY, so a close
        applied ahead of the harvest hides an already-filed row from it and mints a
        duplicate under a fresh id. The base hook now ends with a CLOSE of its own
        (``_carry_story_deferred_closes``, #458) and still satisfies the rule, since
        both of its appends precede it; the two closes never coexist on one task,
        because ``_close_declared_deferred`` is a no-op here and a story run has no
        bundle. ``_carry_harvested_deferrals`` defends the same
        hazard a second time with its own status-agnostic pre-scan, which is exactly
        why the order is pinned by a test: with that second line of defence in place
        a reversal is silent today and would only surface if the pre-scan were ever
        narrowed back to the writer's semantics.

        ``_close_bundle_ledger_when_spec_status`` writes
        ``self.workspace.paths.deferred_work`` — under ``scm.isolation = "worktree"``
        that is the unit worktree's copy. The shape this rescues is a GITIGNORED
        ledger named in ``scm.worktree_seed``: the flip lands in the worktree, then
        ``finalize_commit``'s ``git add -A`` skips the ignored path in silence, so it
        never rides the unit branch and the merge brings nothing over. The bundle's
        entries stay ``open``, ``deferredwork.open_ids`` re-bundles them, and every
        later sweep re-triages work that is already done — an unbounded loop rather
        than a one-time drop, which is what makes this worth a carry.

        An UNSEEDED gitignored ledger is a different, still-open hole this cannot
        reach: a worktree checks out tracked files only, so the ledger is absent
        there entirely, ``verify_review_bundle`` (which reads the WORKTREE's copy)
        never sees the ids ``done``, and the unit DEFERS on a fixable retry instead
        of landing. That shape is loud where this one is silent, and no DONE-leg
        carry helps a unit that never reaches DONE (#426).

        DONE leg only, deliberately: ``Engine._defer``'s isolated arm calls
        ``_carry_harvested_deferrals`` directly and never this hook, and
        ``_replay_unlatched_ledger_carries``'s DEFERRED leg matches it for the same
        reason. A defer discarded the code the close claims to have RESOLVED, and a
        close is the most expensive engine-side write to leave behind — ``open_ids``
        re-bundles only ``open`` entries, so a wrong ``done`` is invisible to every
        future sweep.

        Keyed on ``task.bundle_closes_intended``, never on the ids the in-worktree
        close managed to flip: those are exactly empty in the broken case above.
        The close is written reopenable, with the same note and operation id as the
        in-worktree close, so a carried row is indistinguishable from one that
        arrived through the merge instead of a second row shape for one event.

        No ``_generic_dev()`` guard, unlike the two ledger WRITERS: the record is
        the guard. ``bundle_closes_intended`` is assigned only by
        ``_close_bundle_ledger_when_spec_status``, which ``_post_dev_accepted_sync``
        reaches on the generic path alone, so on the legacy path — where the session
        owns the ledger — it is empty and this returns before touching anything. A
        second predicate saying the same thing would be a branch no test can redden.

        The commit is best effort, where ``_carry_harvested_deferrals`` re-raises on
        a ledger git can own. The asymmetry is deliberate: that method's raise is
        backed by ``harvest_carry_commit_pending``, so a replay still owes the commit
        after dedup empties its carried list. This carry has no such latch and its
        flips are idempotent, so a replay finds nothing left to commit — raising here
        would cost the run its ``integrate_unit`` over bookkeeping whose real work is
        already done. The flips themselves are unguarded: losing them is the hazard
        this exists to prevent.
        """
        super()._carry_isolated_ledger_writes(task)
        if not task.bundle_closes_intended:
            return
        ledger = self.paths.deferred_work
        carried = deferredwork.mark_done_many_reopenable(
            ledger,
            task.bundle_closes_intended,
            self._today(),
            self._bundle_close_note(task),
            self._bundle_close_operation_id(task),
        )
        if carried:
            try:
                verify.commit_paths(
                    self.paths.repo_root,
                    f"chore(deferred-work): close {task.story_key}'s bundle ids",
                    [ledger],
                )
            except verify.GitError as e:
                self.journal.append(
                    "sweep-bundle-close-carry-uncommitted",
                    story_key=task.story_key,
                    dw_ids=carried,
                    error=str(e),
                )
        self.journal.append("sweep-bundle-close-carried", story_key=task.story_key, dw_ids=carried)

    def _verify_dev_artifacts(self, task: StoryTask, result_json: dict | None):
        return verify.verify_dev_bundle(
            task,
            self.workspace.paths,
            result_json,
            review_enabled=self._dev_review_enabled(),
            engine_written=self._harvest_gate_exclude(task),
        )

    def _verify_review(self, task: StoryTask):
        # Generic bundle dev sessions are told not to edit deferred-work.md; the
        # orchestrator is the ledger writer. A follow-up review can rewrite the
        # ledger from its own snapshot and re-open entries that were already
        # closed after dev. Re-apply that idempotent closure immediately before
        # verify_review_bundle requires those entries. The distinct journal kind
        # makes "a review rewrote the ledger" greppable when diagnosing runs.
        if self._generic_dev() and task.spec_file:
            self._close_bundle_ledger_when_spec_status(
                task, task.spec_file, "done", kind="sweep-bundle-reclosed"
            )
        return verify.verify_review_bundle(
            task,
            self.workspace.paths,
            self.policy,
            on_results=self._review_command_sink(task),
        )

    def _operator_park_enabled(self) -> bool:
        # A bundle carries no sprint-status entry, so the pair a park is verified
        # against does not exist, and `verify_review_bundle` gates on closed dw
        # ids instead. Whether a deferred-work bundle can owe a human action is a
        # separate question from whether a story can; not answered here.
        return False

    def _commit_message(self, task: StoryTask) -> str:
        rendered = self._render_commit_template(task)
        if rendered is not None:
            return rendered
        return f"sweep {task.story_key}: {', '.join(task.dw_ids)} via bmad-loop"
