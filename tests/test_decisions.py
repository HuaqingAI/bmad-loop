"""Pre-answer store, discovery of missed decisions, and out-of-band apply."""

import contextlib
import json
import sys

import pytest
from conftest import fault_read_text, install_bmad_config, write_ledger

from bmad_loop import decisions, deferredwork, platform_util, runs
from bmad_loop.sweep import DecisionOption


def _decision(dw_id, *, question="q", options=None, recommendation="1"):
    options = options or [
        {"key": "1", "label": "Build it", "effect": "build", "intent": "do it"},
        {"key": "2", "label": "Keep as is", "effect": "keep-open"},
    ]
    return {
        "id": dw_id,
        "question": question,
        "context": "ctx",
        "options": options,
        "recommendation": recommendation,
    }


def _triage(open_ids, decisions_):
    return {
        "workflow": "deferred-sweep-triage",
        "open_ids": list(open_ids),
        "already_resolved": [],
        "bundles": [],
        "blocked": [],
        "skip": [],
        "decisions": decisions_,
        "escalations": [],
    }


def _make_run(project, run_id, triage_rj, cycle=1):
    run_dir = project.project / ".bmad-loop" / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "state.json").write_text("{}", encoding="utf-8")  # so list_run_dirs sees it
    name = "triage.json" if cycle == 1 else f"triage-{cycle}.json"
    (run_dir / name).write_text(json.dumps(triage_rj), encoding="utf-8")
    return run_dir


# ------------------------------------------------------------- store I/O


def test_store_round_trip_and_prune(project):
    opt = DecisionOption(key="1", label="Build it", effect="build", intent="do it")
    decisions.record_pre_answer(project.project, "DW-7", opt, date="2026-06-13")
    loaded = decisions.load_pre_answers(project.project)
    assert loaded["DW-7"]["effect"] == "build"
    assert loaded["DW-7"]["intent"] == "do it"
    assert loaded["DW-7"]["answered_at"] == "2026-06-13"

    # only entries whose id is still open survive a prune
    dropped = decisions.prune_pre_answers(project.project, {"DW-9"})
    assert dropped == ["DW-7"]
    assert decisions.load_pre_answers(project.project) == {}


# The pre-answer store's cross-process writers. Same doctrine as the ledger-lock
# section of tests/test_deferredwork.py, whose comment says why exclusion is
# probed with `blocking=False` only: `file_lock` is per open fd, so a blocking
# probe from this process would wait forever on POSIX against a lock this very
# thread holds. Every row is seeded to WRITE — a no-op row would grade nothing.
_OPT = DecisionOption(key="1", label="Build it", effect="build", intent="do it")
STORE_WRITERS = {
    "record_pre_answer": lambda root: decisions.record_pre_answer(
        root, "DW-2", _OPT, date="2026-06-13"
    ),
    "prune_pre_answers": lambda root: decisions.prune_pre_answers(root, set()),
    "drop_pre_answer": lambda root: decisions.drop_pre_answer(
        root, "DW-1", answer=decisions.load_pre_answers(root)["DW-1"]
    ),
}


def _store_lock_is_held(root) -> bool:
    """True when the store's sidecar lock cannot be taken right now."""
    try:
        with platform_util.file_lock(
            runs.lock_path_for(decisions.store_path(root)), blocking=False
        ):
            return False
    except OSError:
        return True


@pytest.mark.parametrize("name", sorted(STORE_WRITERS))
def test_every_store_writer_holds_the_store_lock_once(project, monkeypatch, name):
    """Each store writer takes `store_lock` exactly once, at depth zero, and the
    hold really excludes.

    The store has two writers in different processes — a sweep's prunes and the
    human's `bmad-loop decisions` — and each is a whole-file read->edit->write that
    `_write_store`'s atomic replace protects only the last step of. For the DW-143
    prune the unlocked interval was the whole defect: a replacement recorded
    between its read and its write is exactly the value its compare exists to
    spare, and it would overwrite it with the stale snapshot. That the spy fired
    says the writer routes through the lock; ONCE says the read, the edit and the
    write sit inside one acquisition rather than a per-step hold another writer
    can slip between; the probe inside says it is a real OS lock and not a no-op.

    Ablation: dedent any one writer's body out of its `with store_lock(project):`
    — that row's spy never fires and it reds. Wrap only `drop_pre_answer` and the
    other two rows red, which is the point: a lock one writer takes excludes
    nobody."""
    decisions.record_pre_answer(project.project, "DW-1", _OPT, date="2026-06-13")
    real_lock = decisions.store_lock
    probed = []
    depth = 0

    @contextlib.contextmanager
    def spy_lock(root):
        nonlocal depth
        assert depth == 0, f"{name} nested a store_lock acquisition"
        depth += 1
        try:
            with real_lock(root):
                probed.append(_store_lock_is_held(root))
                yield
        finally:
            depth -= 1

    monkeypatch.setattr(decisions, "store_lock", spy_lock)

    STORE_WRITERS[name](project.project)

    assert probed == [True]


def test_store_lock_refuses_to_nest(project):
    """A nested acquisition raises rather than self-deadlocking on POSIX `flock`
    (per open fd, no timeout, no traceback) — the same guard `ledger_lock` has.

    Ablation: delete the `_LOCK_STATE.held` check and this row hangs instead of
    raising, which is why it is the one row here that MUST stay a raise."""
    with decisions.store_lock(project.project):
        with pytest.raises(RuntimeError, match="not reentrant"):
            with decisions.store_lock(project.project):
                pass  # pragma: no cover — never entered
    # and the guard is released on exit, so a later acquisition is fine
    with decisions.store_lock(project.project):
        pass


def test_a_store_writer_that_cannot_lock_writes_nothing(project, monkeypatch):
    """A failed acquisition propagates without touching the store: a write that
    could not be serialized fails loudly rather than proceeding unlocked. The
    error shape is the one `msvcrt.locking` raises when its ~10 s retry runs out
    — a routine outcome on the Windows legs, not a contrived one."""
    decisions.record_pre_answer(project.project, "DW-1", _OPT, date="2026-06-13")
    store = decisions.store_path(project.project)
    before = store.read_bytes()

    @contextlib.contextmanager
    def unavailable(path, **kwargs):
        raise OSError(11, "Resource deadlock avoided")
        yield  # pragma: no cover — unreachable

    monkeypatch.setattr(decisions, "file_lock", unavailable)

    with pytest.raises(OSError):
        decisions.record_pre_answer(project.project, "DW-2", _OPT, date="2026-06-13")
    with pytest.raises(OSError):
        decisions.drop_pre_answer(project.project, "DW-1", answer=json.loads(before)["DW-1"])
    assert store.read_bytes() == before


def test_drop_pre_answer_removes_one_entry_and_leaves_the_rest(project):
    """DW-143's store primitive at its own layer. `prune_pre_answers` above is
    covered directly; its single-id sibling was reachable only through
    `SweepEngine._materialize_bundles`, which cannot see the returned bool at all
    and pins the no-op branch only indirectly.

    Four claims: the bool reports whether an entry was actually removed (all
    branches), an absent id writes NOTHING — the file's bytes are untouched, so the
    keep-open drop of an answer that only ever lived in `<run>/decisions.json`
    cannot re-serialize a store it has no business rewriting — an id whose entry is
    no longer the `answer` being retired is left alone the same way (the caller's
    copy is a stale read; what the store holds now is a human's LATER answer, and
    deleting it on the strength of the stale copy is exactly the loss the keyword
    exists to refuse), and a real removal carries every sibling through, the
    unusable one included, since `load_pre_answers` validates only the top level
    and an unrelated write must not delete a human's corrupt entry.

    Ablation: drop the `dw_id not in data` half of the early return and the bool
    and the byte-equality both redden; drop the `data[dw_id] != answer` half and
    the replaced-entry branch reddens; write `_write_store(project, {})` and the
    siblings redden."""
    opt = DecisionOption(key="1", label="Build it", effect="build", intent="do it")
    decisions.record_pre_answer(project.project, "DW-7", opt, date="2026-06-13")
    decisions.record_pre_answer(project.project, "DW-8", opt, date="2026-06-13")
    store = decisions.store_path(project.project)
    # planted past the writer, the way a hand edit would: unusable, and not ours to repair
    data = json.loads(store.read_text(encoding="utf-8"))
    data["DW-9"] = ["not a decision answer at all"]
    store.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    before = store.read_bytes()
    dw7 = decisions.load_pre_answers(project.project)["DW-7"]

    # an id the store never held: False, and not one byte written
    assert decisions.drop_pre_answer(project.project, "DW-404", answer=dw7) is False
    assert store.read_bytes() == before

    # an id the store holds under a DIFFERENT value than the one being retired — a
    # human re-answered it since the caller read its copy: False, not one byte written
    superseded = {**dw7, "answered_at": "2026-06-01"}
    assert decisions.drop_pre_answer(project.project, "DW-7", answer=superseded) is False
    assert store.read_bytes() == before

    assert decisions.drop_pre_answer(project.project, "DW-7", answer=dw7) is True
    remaining = decisions.load_pre_answers(project.project)
    assert set(remaining) == {"DW-8", "DW-9"}
    assert remaining["DW-8"]["intent"] == "do it"
    assert remaining["DW-9"] == ["not a decision answer at all"]
    # and removing the same id twice is False the second time
    assert decisions.drop_pre_answer(project.project, "DW-7", answer=dw7) is False


def test_load_pre_answers_tolerates_garbage(project):
    decisions.store_path(project.project).parent.mkdir(parents=True, exist_ok=True)
    decisions.store_path(project.project).write_text("not json", encoding="utf-8")
    assert decisions.load_pre_answers(project.project) == {}


def test_load_pre_answers_tolerates_undecodable_bytes(project):
    """DW-140. The sibling fault `json.JSONDecodeError` never covered: bytes that
    are not UTF-8 at all raise `UnicodeDecodeError` out of `read_text` BEFORE any
    JSON parsing, so one bad byte in the project store aborted the whole sweep
    that read it (and `bmad-loop decisions` with it).
    Ablation: drop `UnicodeDecodeError` from the except tuple in
    `load_pre_answers` and this reddens with that exception rather than {}."""
    store = decisions.store_path(project.project)
    store.parent.mkdir(parents=True, exist_ok=True)
    store.write_bytes(b'{"DW-1": {"effect": "\xff"}}')

    assert decisions.load_pre_answers(project.project) == {}


def test_record_pre_answer_write_failure_raises_and_keeps_the_store(project, monkeypatch):
    """#363. `_write_store` is a read-modify-rewrite of a file nothing gitignores,
    so its temp must not outlive a failed write: a stranded
    `.bmad-loop/decisions.tmp` is an untracked file that holds `worktree_clean`
    False until a human deletes it. Routing through the helper is what closes that
    — it unlinks its own temp on any raise — and the raise still reaches the caller.

    Patched at decisions' OWN binding, never `Path.write_text`: the helper writes
    through an `mkstemp` fd via `os.fdopen`, so a `Path` patch never fires and the
    test would pass having exercised nothing.

    Ablation A3: revert `_write_store` to the hand-rolled `tmp.write_text(...)` +
    `atomic_replace` and this reddens alone — loudly, as an AttributeError from
    `monkeypatch.setattr`, because the module binding disappears with the revert."""
    path = decisions.store_path(project.project)
    decisions.record_pre_answer(
        project.project,
        "DW-7",
        DecisionOption(key="1", label="Build it", effect="build", intent="do it"),
        date="2026-06-13",
    )
    before = path.read_bytes()

    def boom(path, text, *, confine_root, require_writable_target=False):
        raise OSError("disk full")

    monkeypatch.setattr(decisions, "atomic_write_text_confined", boom)
    with pytest.raises(OSError, match="disk full"):
        decisions.record_pre_answer(
            project.project,
            "DW-9",
            DecisionOption(key="2", label="Keep as is", effect="keep-open"),
            date="2026-06-14",
        )

    assert path.read_bytes() == before
    assert b"DW-9" not in path.read_bytes()  # the specific mutation that must not land


# ------------------------------------------------------- discovery


def test_pending_missed_decisions_most_recent_wins_and_filters(project):
    install_bmad_config(project)
    write_ledger(project, {"DW-1": "open", "DW-2": "open", "DW-3": "done 2026-06-01"})
    # older run: DW-1 with stale wording; newer run: DW-1 (fresh wording) + DW-2;
    # DW-3 surfaces too but is closed in the ledger
    _make_run(
        project, "20260101-000000-aaaa", _triage(["DW-1"], [_decision("DW-1", question="old")])
    )
    _make_run(
        project,
        "20260102-000000-bbbb",
        _triage(
            ["DW-1", "DW-2", "DW-3"],
            [
                _decision("DW-1", question="new"),
                _decision("DW-2"),
                _decision("DW-3"),
            ],
        ),
    )
    # DW-2 already pre-answered out of band -> excluded
    decisions.record_pre_answer(
        project.project,
        "DW-2",
        DecisionOption(key="2", label="x", effect="keep-open"),
        date="2026-06-13",
    )

    pending = decisions.pending_missed_decisions(project.project)
    ids = [d.id for d in pending]
    assert ids == ["DW-1"]  # DW-2 answered, DW-3 closed
    assert pending[0].question == "new"  # newest run's wording


def test_pending_missed_decisions_re_offers_an_id_whose_stored_value_is_unusable(project):
    """DW-134's loose end. `load_pre_answers` validates only the top level, so a
    store VALUE can be any JSON; a sweep drops such a value and re-files the id as
    unanswered. Keying `answered` off presence alone therefore hid the id from
    this command while every sweep skipped it — unanswerable until a human opened
    the file. DW-2 (well-shaped) still counts as answered, so the exclusion is not
    simply gone. Ablation: restore `answered = set(load_pre_answers(project))` and
    this reddens — DW-1 drops out of the list."""
    install_bmad_config(project)
    write_ledger(project, {"DW-1": "open", "DW-2": "open"})
    _make_run(
        project,
        "20260101-000000-aaaa",
        _triage(["DW-1", "DW-2"], [_decision("DW-1"), _decision("DW-2")]),
    )
    store = decisions.store_path(project.project)
    store.parent.mkdir(parents=True, exist_ok=True)
    store.write_text(
        json.dumps(
            {
                "DW-1": "keep-open",  # a bare effect string, not the answer object
                "DW-2": {"key": "2", "label": "Keep as is", "effect": "keep-open"},
            }
        ),
        encoding="utf-8",
    )

    assert [d.id for d in decisions.pending_missed_decisions(project.project)] == ["DW-1"]


@pytest.mark.parametrize(
    "unusable",
    [
        pytest.param({}, id="empty-object"),
        pytest.param({"effect": "frobnicate"}, id="unknown-effect"),
        pytest.param({"effect": ["build"]}, id="non-string-effect"),
        pytest.param(
            {"key": "1", "label": "Widen", "effect": "build", "intent": ["a", "b"]},
            id="non-string-intent",
        ),
        pytest.param({"key": 1, "effect": "keep-open"}, id="non-string-key"),
        pytest.param(
            {"key": "1", "label": "Widen", "effect": "build", "bundle_name": 7},
            id="non-string-bundle-name",
        ),
    ],
)
def test_pending_missed_decisions_re_offers_ids_whose_stored_shape_is_unusable(project, unusable):
    """DW-142. DW-134 widened `answered` from key presence to "the value is a
    dict", which still counted every shape a sweep now refuses: no `effect`, an
    `effect` outside `DECISION_EFFECTS`, or a scalar the bundle lanes read as a
    string that is not one. Each is an id `_materialize_bundles` silently ignores
    while this command called it answered — so no reader ever surfaced it. Both
    readers share `sweep.unusable_answer_reason`, so the two sets agree by
    construction. DW-2 (well-shaped) still counts as answered, so the exclusion is
    per-value rather than "the store has a bad entry, offer everything".
    Ablation: restore `if isinstance(v, dict)` in `pending_missed_decisions` and
    every parametrization but the non-dict cases reddens — DW-1 drops out."""
    install_bmad_config(project)
    write_ledger(project, {"DW-1": "open", "DW-2": "open"})
    _make_run(
        project,
        "20260101-000000-aaaa",
        _triage(["DW-1", "DW-2"], [_decision("DW-1"), _decision("DW-2")]),
    )
    store = decisions.store_path(project.project)
    store.parent.mkdir(parents=True, exist_ok=True)
    store.write_text(
        json.dumps(
            {
                "DW-1": unusable,
                "DW-2": {"key": "2", "label": "Keep as is", "effect": "keep-open"},
            }
        ),
        encoding="utf-8",
    )

    assert [d.id for d in decisions.pending_missed_decisions(project.project)] == ["DW-1"]


def test_pending_missed_decisions_keeps_a_close_answer_answered(project):
    """`close` never reaches the PROJECT store (`apply_pre_answer` applies it to
    the ledger instead), but `unusable_answer_reason` accepts it for the run-local
    store's sake — so this command must not start re-offering an id a hand-seeded
    `close` covers. Ablation: drop "close" from `DECISION_EFFECTS` and this
    reddens with DW-1 offered."""
    install_bmad_config(project)
    write_ledger(project, {"DW-1": "open"})
    _make_run(project, "20260101-000000-aaaa", _triage(["DW-1"], [_decision("DW-1")]))
    store = decisions.store_path(project.project)
    store.parent.mkdir(parents=True, exist_ok=True)
    store.write_text(json.dumps({"DW-1": {"key": "3", "effect": "close"}}), encoding="utf-8")

    assert decisions.pending_missed_decisions(project.project) == []


def test_pending_missed_decisions_skips_an_undecodable_triage_cache(project):
    """DW-145. The read loop caught `(json.JSONDecodeError, OSError)`, but
    `UnicodeDecodeError` is a `ValueError`: bytes that are not UTF-8 at all raise
    out of `read_text` BEFORE any JSON parsing, so one bad byte in one run's
    cached triage escaped out of this helper, past every caller: `cmd_decisions`
    and `cmd_status` catch `BmadConfigError` alone (so `main`'s backstop turned
    the command into exit 1) and the TUI catches `(BmadConfigError, OSError)` (so
    it escaped outright). Degradation is per FILE: the newest run is skipped and the
    older run's DW-1 still surfaces, so the widening is not "return nothing".
    Ablation: revert the except tuple to `(json.JSONDecodeError, OSError)` and
    this reddens with `UnicodeDecodeError` rather than ["DW-1"]."""
    install_bmad_config(project)
    write_ledger(project, {"DW-1": "open"})
    _make_run(project, "20260101-000000-aaaa", _triage(["DW-1"], [_decision("DW-1")]))
    bad = _make_run(project, "20260102-000000-bbbb", _triage([], []))
    (bad / "triage.json").write_bytes(b'{"workflow": "deferred-sweep-triage", "x": "\xff"}')

    assert [d.id for d in decisions.pending_missed_decisions(project.project)] == ["DW-1"]


def test_pending_missed_decisions_skips_an_unparseable_triage_cache(project):
    """The sibling arm the DW-145 widening sits beside: truncated JSON is skipped
    the same way, and the other run still contributes. Guards against a widening
    that accidentally narrows — both faults share one `continue`."""
    install_bmad_config(project)
    write_ledger(project, {"DW-1": "open"})
    _make_run(project, "20260101-000000-aaaa", _triage(["DW-1"], [_decision("DW-1")]))
    bad = _make_run(project, "20260102-000000-bbbb", _triage([], []))
    (bad / "triage.json").write_text('{"workflow": "deferred-sw', encoding="utf-8")

    assert [d.id for d in decisions.pending_missed_decisions(project.project)] == ["DW-1"]


def test_pending_missed_decisions_skips_an_unreadable_triage_cache(project, monkeypatch):
    """The third arm sharing that `continue`: a filesystem refusal on one run's
    cache. Distinct from JSON and UTF-8 decoding — it raises before either — and
    the same degrade applies, so the older run's DW-1 still surfaces. Ablation:
    drop `OSError` from the except tuple and this raises `PermissionError`."""
    install_bmad_config(project)
    write_ledger(project, {"DW-1": "open"})
    _make_run(project, "20260101-000000-aaaa", _triage(["DW-1"], [_decision("DW-1")]))
    bad = _make_run(project, "20260102-000000-bbbb", _triage([], []))
    fault_read_text(monkeypatch, bad / "triage.json")

    assert [d.id for d in decisions.pending_missed_decisions(project.project)] == ["DW-1"]


def test_pending_missed_decisions_skips_a_triage_the_stricter_validation_refuses(project):
    """The DW-148 blast radius on THIS reader, pinned rather than discovered. A
    cached triage that decodes and parses cleanly can still fail the new
    type check, and a plan of `None` hits the same `continue` as a truncated file:
    the run contributes nothing, `_errors` is discarded unread at this call site
    (there is no journal here — this is a read-only command surface), and the id
    drops out of `decisions --list`, `status` and the TUI. It is not lost: the
    ledger entry is still open, so the NEXT sweep re-triages it and the fresh
    triage restores it. A pre-existing cache written before this release can hold
    such a value — `resolution: 5` was accepted as `str(5)` until now — so this is
    a real transition, not a hypothetical.

    The string control is what makes the refusal attributable: the two runs differ
    in that one byte, so `[]` cannot be passing for an unrelated reason."""
    install_bmad_config(project)
    write_ledger(project, {"DW-1": "open"})
    refused = _decision("DW-1")
    refused["options"][0]["resolution"] = 5  # not a string: refused from now on
    _make_run(project, "20260101-000000-aaaa", _triage(["DW-1"], [refused]))

    assert decisions.pending_missed_decisions(project.project) == []

    # Control: the identical triage with a string resolution still surfaces DW-1.
    accepted = _decision("DW-1")
    accepted["options"][0]["resolution"] = "5"
    _make_run(project, "20260102-000000-bbbb", _triage(["DW-1"], [accepted]))

    assert [d.id for d in decisions.pending_missed_decisions(project.project)] == ["DW-1"]


def test_pending_missed_decisions_empty_when_nothing_open(project):
    install_bmad_config(project)
    write_ledger(project, {"DW-1": "done 2026-06-01"})
    _make_run(project, "20260101-000000-aaaa", _triage([], []))
    assert decisions.pending_missed_decisions(project.project) == []


# ------------------------------------------------------- apply


def test_apply_pre_answer_build_records_store_and_ledger(project):
    install_bmad_config(project)
    write_ledger(project, {"DW-1": "open"})
    from bmad_loop.sweep import Decision

    opt = DecisionOption(key="1", label="Build", effect="build", intent="widen field")
    d = Decision(id="DW-1", question="build it?", context="", options=(opt,), recommendation="1")
    decisions.apply_pre_answer(project.project, d, opt, date="2026-06-13")

    entries = {
        e.id: e
        for e in deferredwork.parse_ledger(project.deferred_work.read_text(encoding="utf-8"))
    }
    assert "decision: 2026-06-13 Build — widen field" in entries["DW-1"].body
    assert entries["DW-1"].open  # build stays open until a sweep builds it
    assert decisions.load_pre_answers(project.project)["DW-1"]["effect"] == "build"
    assert "chore(decisions): pre-answer DW-1" in _git_log(project)


def test_apply_pre_answer_sanitizes_a_multiline_detail(project):
    """The human-decision writer path (`decisions.py:146-150`), called bare: it
    must sanitize rather than raise. Pins `detail = option.resolution or
    option.intent` on its fallback branch — the resolution is empty here, so an
    option `intent` is what reaches the ledger."""
    install_bmad_config(project)
    write_ledger(project, {"DW-1": "open"})
    from bmad_loop.sweep import Decision

    opt = DecisionOption(
        key="1", label="Build\ncap", effect="build", intent="widen the field.\nThen backfill."
    )
    d = Decision(id="DW-1", question="build it?", context="", options=(opt,), recommendation="1")

    decisions.apply_pre_answer(project.project, d, opt, date="2026-06-13")

    text = project.deferred_work.read_text(encoding="utf-8")
    entries = {e.id: e for e in deferredwork.parse_ledger(text)}
    assert set(entries) == {"DW-1"}  # no phantom entry minted
    assert (
        "decision: 2026-06-13 Build cap — widen the field. Then backfill." in entries["DW-1"].body
    )
    assert len([line for line in text.splitlines() if line.startswith("decision:")]) == 1
    assert entries["DW-1"].open  # build stays open until a sweep builds it


def test_apply_pre_answer_raises_on_a_bad_date_leaving_nothing_written(project):
    """The documented precondition, and that it fires before *any* of the four
    side effects `apply_pre_answer` chains: `append_decision`, `mark_done`,
    `record_pre_answer` and `commit_paths`. A raise partway through would leave a
    ledger annotation with no store entry, or either with no commit.

    Both callers catch it — the TUI degrades to a per-decision notification and
    `bmad-loop decisions` to an error line — so the failure a human sees must
    correspond to nothing having happened."""
    install_bmad_config(project)
    write_ledger(project, {"DW-1": "open"})
    from bmad_loop.sweep import Decision

    opt = DecisionOption(key="1", label="Build", effect="build", intent="do it")
    d = Decision(id="DW-1", question="?", context="", options=(opt,), recommendation="1")
    ledger_before = project.deferred_work.read_text(encoding="utf-8")
    store_before = decisions.load_pre_answers(project.project)

    with pytest.raises(ValueError, match="date must be YYYY-MM-DD"):
        decisions.apply_pre_answer(project.project, d, opt, date="13/06/2026")

    assert project.deferred_work.read_text(encoding="utf-8") == ledger_before
    assert decisions.load_pre_answers(project.project) == store_before
    assert not decisions.store_path(project.project).exists()


def test_apply_pre_answer_close_marks_done_no_store(project):
    install_bmad_config(project)
    write_ledger(project, {"DW-1": "open"})
    from bmad_loop.sweep import Decision

    opt = DecisionOption(key="1", label="Close", effect="close", resolution="superseded")
    d = Decision(id="DW-1", question="close?", context="", options=(opt,), recommendation="1")
    decisions.apply_pre_answer(project.project, d, opt, date="2026-06-13")

    entries = {
        e.id: e
        for e in deferredwork.parse_ledger(project.deferred_work.read_text(encoding="utf-8"))
    }
    assert entries["DW-1"].status.startswith("done")
    assert "closed by human decision: superseded" in entries["DW-1"].body
    assert decisions.load_pre_answers(project.project) == {}  # close needs no carry-forward


def test_apply_pre_answer_commit_leaves_unrelated_changes(project):
    install_bmad_config(project)
    write_ledger(project, {"DW-1": "open"})
    (project.project / "src.txt").write_text("user edit, uncommitted\n")  # unrelated work
    from bmad_loop.sweep import Decision

    opt = DecisionOption(key="1", label="Close", effect="close", resolution="x")
    d = Decision(id="DW-1", question="?", context="", options=(opt,), recommendation="1")
    decisions.apply_pre_answer(project.project, d, opt, date="2026-06-13")
    # the unrelated change is still uncommitted (commit_paths staged only the ledger)
    assert "src.txt" in _git_status(project)


def _git_log(project):
    import subprocess

    return subprocess.run(
        ["git", "-C", str(project.project), "log", "--oneline"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout


def _git_status(project):
    import subprocess

    return subprocess.run(
        ["git", "-C", str(project.project), "status", "--porcelain"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout


# ------------------------------------------ the store's confined write (#593, #597)


def _answer(project, dw_id="DW-7", date="2026-06-13"):
    decisions.record_pre_answer(
        project.project,
        dw_id,
        DecisionOption(key="1", label="Build it", effect="build", intent="do it"),
        date=date,
    )


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX symlinks")
def test_the_store_write_refuses_a_symlinked_bmad_loop(project, tmp_path):
    """The escape #593 names, at this site. `follow_symlinks=False` refused a link
    planted at `decisions.json`; it never refused one at `.bmad-loop/`, and
    `_write_store`'s own `mkdir(parents=True, exist_ok=True)` ACCEPTS a
    symlink-to-a-directory, so the planted parent survives the setup step and both
    the temp and the published store land wherever the link points.

    A driven session can write under `.bmad-loop/`, which is what makes this a
    real writer rather than a hypothetical one: the escalation the no-follow was
    added to close costs a directory swap instead of a file swap.

    The second assertion is the one that pins the fix — refusing loudly is worth
    nothing if the write already landed outside the project.

    Ablation: revert `_write_store` to
    `atomic_write_text(path, ..., follow_symlinks=False)` and this fails
    `DID NOT RAISE`, with `decisions.json` sitting in `outside/`."""
    outside = tmp_path / "outside"
    outside.mkdir()
    (project.project / ".bmad-loop").symlink_to(outside, target_is_directory=True)

    with pytest.raises(platform_util.UnconfinedWriteError):
        _answer(project)

    assert list(outside.iterdir()) == []  # nothing escaped the project


def test_the_store_write_lands_on_a_clean_tree(project):
    """The positive control for the refusal above. Without it that test passes for
    a `_write_store` wired to refuse everything, which is every reason a file
    could be absent from `outside/`."""
    _answer(project)

    assert decisions.load_pre_answers(project.project)["DW-7"]["effect"] == "build"
    assert decisions.store_path(project.project).is_file()


def test_the_store_write_refuses_a_readonly_store(project):
    """#597 at this site: the store is operator-curated — a human answers these
    decisions out of band — so a read-only one is answered with the
    `PermissionError` a bare `Path.write_text` raised, not routed around by a
    replace that only needs the DIRECTORY writable.

    chmod is on the per-test copytree copy the `project` fixture makes, never the
    session template (a read-only template would be inherited by every later
    copy), and it is restored in a `finally` because Windows rmtree refuses a
    READONLY file at cleanup.

    Ablation: drop `require_writable_target=True` from `_write_store` and this
    fails `DID NOT RAISE`, with the store rewritten and still reading 0444."""
    _answer(project, "DW-7")
    store = decisions.store_path(project.project)
    before = store.read_bytes()
    store.chmod(0o444)
    try:
        with pytest.raises(PermissionError):
            _answer(project, "DW-9", date="2026-06-14")
    finally:
        store.chmod(0o644)

    assert store.read_bytes() == before  # the second answer never landed


@pytest.mark.parametrize(
    ("effect", "label", "extra", "close_note"),
    [
        ("build", "Build", {"intent": "widen field"}, None),
        ("close", "Close", {"resolution": "superseded"}, "closed by human decision: superseded"),
    ],
)
def test_apply_pre_answer_is_one_ledger_transaction(
    project, tmp_path, monkeypatch, effect, label, extra, close_note
):
    """The decision record and the closure it asks for land in ONE locked
    read->edit->write, byte-identical to the released `append_decision` +
    `mark_done` pair (#286/#469).

    Two claims, and both are needed. The golden text says the collapse moved no
    bytes — these ledgers are committed and read by humans, so `record_decision`
    inserting the decision line before it applies the close is a contract, not a
    detail (`_MARK_DONE_TAIL_RE` anchors an undo marker on the status/resolution
    adjacency the other order would break). The acquisition count is what says
    the pair actually collapsed: as two calls it was two acquisitions with a
    window between them, and a rival writer landing there left the entry
    carrying a decision that says "close it" over a status that still says open.
    Byte equality alone passes just as well for the released pair.

    Ablation: restore the pair in `decisions.apply_pre_answer`. The golden assert
    still passes — that is the point — and `acquisitions` goes to 2 on the CLOSE
    variant, which is the one that grades the collapse. The build variant stays
    green under that ablation and is known to: with `close_note=None` there is no
    second call to make, and `append_decision` is itself a one-acquisition
    delegate to `record_decision`, so the two spellings are the same transaction.
    It is kept for the claim it does decide — that the no-close path still writes
    the pair's bytes and leaves the entry open — not as a second count oracle.
    """

    from bmad_loop.sweep import Decision

    install_bmad_config(project)
    write_ledger(project, {"DW-1": "open"}, commit=False)
    pristine = project.deferred_work.read_text(encoding="utf-8")
    opt = DecisionOption(key="1", label=label, effect=effect, **extra)
    d = Decision(id="DW-1", question="?", context="", options=(opt,), recommendation="1")

    # The released serial pair, run against a twin of the same pristine ledger in
    # its own directory so it contends on its own lock and is never counted.
    golden = tmp_path / f"golden-{effect}" / "deferred-work.md"
    golden.parent.mkdir(parents=True)
    golden.write_text(pristine, encoding="utf-8")
    deferredwork.append_decision(golden, "DW-1", "2026-06-13", label, opt.resolution or opt.intent)
    if close_note is not None:
        deferredwork.mark_done(golden, "DW-1", "2026-06-13", close_note)

    acquisitions = []
    real_lock = deferredwork.ledger_lock

    @contextlib.contextmanager
    def spy_lock(p):
        acquisitions.append(p)
        with real_lock(p):
            yield

    monkeypatch.setattr(deferredwork, "ledger_lock", spy_lock)
    decisions.apply_pre_answer(project.project, d, opt, date="2026-06-13", commit=False)

    assert project.deferred_work.read_text(encoding="utf-8") == golden.read_text(encoding="utf-8")
    assert acquisitions == [project.deferred_work]  # ONE, and on the project's ledger
