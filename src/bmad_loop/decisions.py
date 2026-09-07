"""Cross-run pre-answers for deferred-work decisions.

A sweep's triage can surface decisions only a human can make. An unattended
sweep (`--no-prompt`) skips them; an interactive sweep can be abandoned before
every prompt is answered. Either way the answer is otherwise lost: triage
re-derives the `decisions` partition from the open ledger on every run, and the
only record of an answer — the run-scoped `{run_dir}/decisions.json` — does not
carry across runs, so the next sweep re-surfaces (and re-skips) the same
decision.

This module is the durable carrier. A human answers missed decisions out of band
(`bmad-loop decisions`, or the TUI), the answer is recorded both as a ledger
`decision:` line and — for build/keep-open — in a project-level
`.bmad-loop/decisions.json` keyed by DW id, and the next sweep consumes it
instead of asking again (see SweepEngine._decisions_phase). `close` answers need
no store entry: they are applied to the ledger immediately (status -> done), so
the entry simply leaves the open set.

Layering note: this module sits above sweep.py (it reuses Decision/validate_triage
and the deterministic ledger helpers). sweep.py imports it lazily to avoid a cycle.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from . import bmadconfig, deferredwork, runs, verify
from .platform_util import atomic_write_text_confined
from .sweep import Decision, DecisionOption, unusable_answer_reason, validate_triage

STORE_REL = Path(".bmad-loop") / "decisions.json"
_TRIAGE_RE = re.compile(r"^triage(?:-(\d+))?\.json$")


def store_path(project: Path) -> Path:
    return project / STORE_REL


# --------------------------------------------------------------- store I/O


def load_pre_answers(project: Path) -> dict[str, dict]:
    """The project-level pre-answer store, {DW-id: {effect,label,intent,...}}.
    Tolerant of a missing or malformed file (returns {}): an absent file, an
    unreadable one, JSON that will not parse, bytes that are not UTF-8 at all
    (DW-140) and a non-object top level all degrade to an empty store rather than
    aborting the caller — every caller here is either a sweep or the `decisions`
    command, and neither has anything to gain from dying on one bad byte.

    Only the TOP level is validated. Values stay exactly as stored, however
    shaped: `record_pre_answer` and `prune_pre_answers` both read-modify-write the
    whole store through this function, so filtering here would make an unrelated
    write silently DELETE a human's corrupt entries — the file repair this
    codebase refuses. Readers that consume a value screen it themselves with
    `sweep.unusable_answer_reason`."""
    path = store_path(project)
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _write_store(project: Path, data: dict) -> None:
    """#363: via the helper, not a hand-rolled tmp+replace. The store lives at a
    path nothing gitignores, so a failed replace used to strand
    `.bmad-loop/decisions.tmp` as an untracked file and hold `worktree_clean`
    False until a human deleted it; the helper removes its temp on any raise.

    Confined to ``project`` (#593). Refusing to follow a link planted at
    `decisions.json` itself was the behaviour-preserving choice — `os.replace`
    never dereferenced this destination either — and the security one: a driven
    session can write under `.bmad-loop/`, so honouring a link planted here would
    hand it a host-side write to any operator-writable path. But that refusal
    stopped at the final component, and the `mkdir` on the line below accepts a
    symlink-to-a-directory, so a link planted at `.bmad-loop/` survived the setup
    and redirected both the temp and the publish. The confined writer walks the
    components below `project` `O_NOFOLLOW` and writes through the descriptor
    that walk produced, so the same escalation now costs a refusal instead of a
    host-side write. Permission-neutral: no-follow never inherited a mode either,
    so the store still lands at `0600`.

    ``require_writable_target=True`` (#597) restores the `PermissionError` a bare
    `Path.write_text` raised here before #363 made the write atomic. The store is
    operator-curated — an operator who marks it read-only is answered rather than
    quietly overwritten and left with the `0444` still showing."""
    path = store_path(project)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text_confined(
        path,
        json.dumps(data, indent=2, sort_keys=True),
        confine_root=project,
        require_writable_target=True,
    )


def record_pre_answer(project: Path, dw_id: str, option: DecisionOption, *, date: str) -> None:
    """Persist a chosen option so a future sweep applies it without asking. The
    option's full semantics are stored (not just its key): a later triage may
    renumber options, so the sweep reads effect/intent from here directly."""
    data = load_pre_answers(project)
    data[dw_id] = {
        "key": option.key,
        "label": option.label,
        "effect": option.effect,
        "intent": option.intent,
        "resolution": option.resolution,
        "bundle_name": option.bundle_name,
        "answered_at": date,
    }
    _write_store(project, data)


def prune_pre_answers(project: Path, open_ids: set[str]) -> list[str]:
    """Drop store entries whose DW id is no longer open (built or closed). No-op
    write when nothing is dropped. Returns the dropped ids."""
    data = load_pre_answers(project)
    dropped = [k for k in data if k not in open_ids]
    if dropped:
        for k in dropped:
            del data[k]
        _write_store(project, data)
    return dropped


def drop_pre_answer(project: Path, dw_id: str) -> bool:
    """Remove ONE id's entry, returning whether an entry was actually there. The
    single-id sibling of `prune_pre_answers` above, and public for the same reason
    that one is: a sweep that has just dropped a stored answer as stale (DW-143)
    must be able to retire the entry that fed it without reaching into
    `_write_store`, which is this module's private writer.

    Same read-modify-write shape, same no-op-when-nothing-changes discipline: an
    absent id writes nothing at all, so a drop whose answer only ever lived in
    `<run>/decisions.json` leaves the project store's bytes (and mtime) untouched.
    A removal goes through `_write_store`, so an operator-locked store still raises
    `PermissionError` rather than silently skipping — deleting a human-authored
    answer is a store write, never a repair."""
    data = load_pre_answers(project)
    if dw_id not in data:
        return False
    del data[dw_id]
    _write_store(project, data)
    return True


# ------------------------------------------------------- discovery + apply


def pending_missed_decisions(project: Path) -> list[Decision]:
    """Decisions earlier sweeps surfaced but no one answered: reconstructed from
    every run's persisted triage*.json, kept only when the DW id is still open
    and not already usably answered in the store — the value has to be one a sweep
    would actually consume, not merely a key that is present (see `answered`
    below). The most recent triage's wording of each id wins. Sorted by DW
    number."""
    paths = bmadconfig.load_paths(project)
    ledger = paths.deferred_work
    # OBSERVATION arm of the ledger-read contract (DW-146). This helper writes
    # nothing: every caller is a read-only surface (`cmd_decisions`, `cmd_status`,
    # the TUI), so an undecodable ledger must not take the whole listing down —
    # `UnicodeDecodeError` is a `ValueError` and escaped every `except OSError`
    # above it, exactly as it did for the triage-cache read below (DW-145).
    # The degradation is SILENT here, unlike the engine's observation sites: no
    # journal is reachable from a module-level function handed only a project
    # path, and the same is true of the triage read below. An empty ledger means
    # no open ids, which returns [] — the honest answer for a file nobody could
    # read, and the one the surfaces above already render.
    text, _fault = deferredwork.read_for_observation(ledger)
    open_now = deferredwork.open_ids(text)
    if not open_now:
        return []
    # By usable VALUE, not by key presence: `load_pre_answers` validates only the
    # top level, and a sweep drops a non-dict value and re-files the decision as
    # unanswered (`sweep-decisions-reload-failed`). Counting the bare key as
    # answered hid exactly those ids from this command, so the id was skipped by
    # every sweep and re-offered by nothing — unanswerable until a human found the
    # file. Re-answering overwrites the unusable value, which is the repair.
    #
    # Usable is `sweep.unusable_answer_reason` — the SAME predicate the sweep read
    # site applies (DW-142), not a local restatement of it. A value the sweep will
    # not consume must be re-offered here, so a shape either reader alone screened
    # out was an id no reader ever surfaced: a missing or unrecognized `effect` and
    # a non-string `key`/`label`/`intent`/`bundle_name` are unusable here for
    # exactly the reason they are unusable there.
    answered = {
        k for k, v in load_pre_answers(project).items() if unusable_answer_reason(v) is None
    }

    # (run-id, cycle) descending == most recent first; run ids sort chronologically
    triage_files: list[tuple[str, int, Path]] = []
    for run_dir in runs.list_run_dirs(project):
        for tp in run_dir.glob("triage*.json"):
            m = _TRIAGE_RE.match(tp.name)
            if m:
                triage_files.append((run_dir.name, int(m.group(1) or 1), tp))
    triage_files.sort(reverse=True)

    by_id: dict[str, Decision] = {}
    for _run, _cycle, tp in triage_files:
        try:
            rj = json.loads(tp.read_text(encoding="utf-8"))
        # `UnicodeDecodeError` is a `ValueError`, NOT an `OSError`, so bytes that
        # are not UTF-8 at all escaped this arm (DW-145). Every caller here is a
        # read-only surface with nothing to gain from dying on one bad byte in one
        # run's cache: `cmd_decisions` and `cmd_status` catch `BmadConfigError`
        # alone, so `main`'s broad backstop turned the whole command into exit 1
        # (and `decisions --json` into no document at all), while the TUI's
        # `(BmadConfigError, OSError)` catch let it escape outright. Same widening,
        # same reason, as `load_pre_answers` and the two sweep siblings
        # (`_ensure_triage`, `_decisions_phase`).
        except (json.JSONDecodeError, OSError, UnicodeDecodeError):
            continue
        plan, _errors = validate_triage(rj, None)
        if plan is None:
            continue
        for decision in plan.decisions:
            by_id.setdefault(decision.id, decision)  # first (most recent) wins

    pending = [by_id[i] for i in by_id if i in open_now and i not in answered]
    return sorted(pending, key=lambda d: int(d.id.split("-")[1]))


def apply_pre_answer(
    project: Path, decision: Decision, option: DecisionOption, *, date: str, commit: bool = True
) -> None:
    """Record a human's out-of-band answer durably. Always writes a ledger
    `decision:` audit line; `close` also flips the entry to done (so it leaves
    the open set now), while `build`/`keep-open` are saved to the pre-answer
    store for the next sweep to consume. When `commit`, the ledger and store are
    committed on their own (only those paths) — best effort, so a non-git or
    dirty tree never blocks the on-disk record.

    Precondition: `date` is ISO `YYYY-MM-DD`. The ledger writers raise
    `ValueError` on anything else (it would otherwise land a `status:` line that
    reads as neither open nor done), so a caller building the date itself must
    either guarantee the format or catch it. The option's own free text carries
    no such precondition — it is sanitized, never refused."""
    paths = bmadconfig.load_paths(project)
    ledger = paths.deferred_work
    detail = option.resolution or option.intent
    close_note = None
    if option.effect == "close":
        close_note = "closed by human decision" + (
            f": {option.resolution}" if option.resolution else ""
        )
    # ONE locked read->edit->write (#286/#469). As the `append_decision` +
    # `mark_done` pair it was two acquisitions with a window between them, and a
    # rival writer landing there left the entry carrying a decision that says
    # "close it" over a status that still says open. The bytes are identical to
    # the pair's. The commit below stays OUTSIDE any lock — locks are held only
    # around file I/O, never across a subprocess (#286).
    deferredwork.record_decision(
        ledger, decision.id, date, option.label, detail, close_note=close_note
    )
    if option.effect != "close":
        record_pre_answer(project, decision.id, option, date=date)
    if commit:
        try:
            verify.commit_paths(
                project,
                f"chore(decisions): pre-answer {decision.id}",
                [ledger, store_path(project)],
            )
        except verify.GitError:
            pass  # files are written; git history is best effort
