"""Sweep engine scenario tests against the mock adapter — no tmux, no LLM."""

import contextlib
import json
import re
import sys
from dataclasses import replace
from pathlib import Path

import pytest
from conftest import (
    _OK,
    _file_exists_cmd,
    _spec_baseline,
    attach_profile,
    bundle_dev_effect,
    bundle_dev_escalates,
    bundle_review_effect,
    crash_at_merge_back,
    fault_read_text,
    git,
    ignore_before_commit,
    install_bmad_config,
    install_build_auto_skill,
    migrate_effect,
    passes_once,
    triage_effect,
    write_ledger,
    write_legacy_ledger,
    write_spec,
)

from bmad_loop import deferredwork, platform_util, runs
from bmad_loop import sweep as sweep_mod
from bmad_loop import verify
from bmad_loop.adapters.base import SessionResult
from bmad_loop.adapters.mock import MockAdapter
from bmad_loop.bmadconfig import ProjectPaths
from bmad_loop.engine import RunPaused
from bmad_loop.journal import Journal, load_state, save_state
from bmad_loop.model import PAUSE_STORY_GATE, Phase, RunState, StoryTask, TokenUsage
from bmad_loop.policy import (
    AdapterPolicy,
    DevPolicy,
    GatesPolicy,
    LimitsPolicy,
    NotifyPolicy,
    Policy,
    ReviewPolicy,
    ScmPolicy,
    StageAdapterPolicy,
    SweepPolicy,
    VerifyPolicy,
)
from bmad_loop.sweep import (
    BUNDLE_KEY_RE,
    Bundle,
    Decision,
    DecisionOption,
    DecisionPrompter,
    PreCanonical,
    ResolvedEntry,
    SweepEngine,
    TriagePlan,
    duplicate_ids,
    snapshot_canonical,
    validate_migration,
    validate_triage,
)
from bmad_loop.tui import launch
from bmad_loop.verify import worktree_clean

QUIET = NotifyPolicy(desktop=False, file=True)


def triage_result(open_ids, **sections):
    return {
        "workflow": "deferred-sweep-triage",
        "open_ids": list(open_ids),
        "already_resolved": sections.get("already_resolved", []),
        "bundles": sections.get("bundles", []),
        "blocked": sections.get("blocked", []),
        "skip": sections.get("skip", []),
        "decisions": sections.get("decisions", []),
        "escalations": [],
    }


def make_sweep(
    project, script, policy=None, answers=(), prompting=False, run_id="sweep-run", **kwargs
):
    # `run_id` names a DISTINCT run: its own run_dir, state.json and journal.
    # Overridden only where a test needs a second run rather than a resume of the
    # first (run-scoped state, e.g. the DW-124 quarantine).
    run_dir = project.project / ".bmad-loop" / "runs" / run_id
    adapter = MockAdapter(script, usage_per_session=TokenUsage(input_tokens=10, output_tokens=5))
    state = RunState(run_id=run_id, project=str(project.project), started_at="now")
    inputs = iter(answers)
    prompter = DecisionPrompter(input_fn=lambda _: next(inputs), print_fn=lambda _line: None)
    engine = SweepEngine(
        paths=project,
        policy=policy
        or Policy(
            gates=GatesPolicy(mode="none"),
            notify=QUIET,
            scm=ScmPolicy(rollback_on_failure=True),
        ),
        adapter=adapter,
        run_dir=run_dir,
        journal=Journal(run_dir),
        state=state,
        prompting=prompting,
        prompter=prompter,
        **kwargs,
    )
    return engine, adapter


def test_remaining_estimate_is_none_for_an_undecodable_ledger(project):
    """`None` and `0` mean opposite things to the graceful stop, and both are
    PUBLISHED — in the `run-stop` journal row and the stop notice. `None` is "no
    estimate"; `0` is a positive claim that a resume would pick up nothing. Routing
    this read through `read_for_observation` (DW-146) put a fault one `if` away from
    being reported as `0 remaining` for a ledger nobody could read — the fabricated
    answer `cli._sweep_dry_run` refuses to print. The fault is checked, not
    discarded.
    The READ's fault is also JOURNALED, not merely checked. The observation arm
    journals its fault wherever a journal is in hand, and one is in hand here. Note
    the scope: `run-stop` publishes `remaining: null` either way, and the outer
    guard still answers `None` silently for anything raised AFTER the read, so the
    row does not make every null self-explaining — what it buys is that the one
    fault class the arm hands back as a VALUE gets attributed instead of collapsing
    into that same silence.
    Ablation: drop the `if fault is not None:` arm and this reddens twice — with `0`
    for the estimate, because the degraded empty text has no open ids, and with no
    `sweep-remaining-estimate-unreadable` row."""
    install_bmad_config(project)
    engine, _ = make_sweep(project, [])
    project.deferred_work.write_bytes(b"# Deferred Work\n\n### DW-1: bad \xff byte\n")

    assert engine._remaining_estimate() is None

    faults = [
        e for e in engine.journal.entries() if e["kind"] == "sweep-remaining-estimate-unreadable"
    ]
    assert len(faults) == 1
    assert faults[0]["ledger"] == str(project.deferred_work)
    assert faults[0]["error"].startswith("UnicodeDecodeError: ")


def resume_sweep(project, engine, script, answers=(), prompting=False, **kwargs):
    state = load_state(engine.run_dir)
    state.clear_pause()
    adapter = MockAdapter(script)
    inputs = iter(answers)
    prompter = DecisionPrompter(input_fn=lambda _: next(inputs), print_fn=lambda _line: None)
    new_engine = SweepEngine(
        paths=project,
        policy=engine.policy,
        adapter=adapter,
        run_dir=engine.run_dir,
        journal=engine.journal,
        state=state,
        prompting=prompting,
        prompter=prompter,
        **kwargs,
    )
    return new_engine, adapter


def journal_text(engine) -> str:
    return (engine.run_dir / "journal.jsonl").read_text()


def ledger_entries(project) -> dict:
    return {
        e.id: e
        for e in deferredwork.parse_ledger(project.deferred_work.read_text(encoding="utf-8"))
    }


def _records(engine, kind: str) -> list[dict]:
    """Every journal record of `kind` the run wrote, in order."""
    records = [json.loads(line) for line in journal_text(engine).splitlines() if line.strip()]
    return [r for r in records if r.get("kind") == kind]


def _mismatches(engine) -> list[dict]:
    """Every `sweep-decision-option-mismatch` record the run wrote, in order."""
    return _records(engine, "sweep-decision-option-mismatch")


# ------------------------------------------------------- validate_triage

_OVERLONG_BUNDLE_NAME = "integration-double-checkout-shared-client"
_NORMALIZED_BUNDLE_NAME = _OVERLONG_BUNDLE_NAME[:40]


def test_validate_triage_happy():
    rj = triage_result(
        ["DW-1", "DW-2", "DW-3", "DW-4", "DW-5"],
        already_resolved=[{"id": "DW-1", "evidence": "fixed in abc123"}],
        bundles=[{"name": "fix-strings", "dw_ids": ["DW-2", "DW-3"], "intent": "harden it"}],
        blocked=[{"id": "DW-4", "blocker": "story 5-2"}],
        decisions=[
            {
                "id": "DW-5",
                "question": "renegotiate?",
                "context": "ctx",
                "options": [
                    {
                        "key": "1",
                        "label": "build it",
                        "effect": "build",
                        "intent": "do x",
                    },
                    {"key": "2", "label": "keep", "effect": "keep-open"},
                ],
                "recommendation": "1",
            }
        ],
    )
    plan, errors = validate_triage(rj, {"DW-1", "DW-2", "DW-3", "DW-4", "DW-5"})
    assert errors == []
    assert plan.bundles[0].dw_ids == ("DW-2", "DW-3")
    assert plan.decisions[0].option("1").effect == "build"


def test_validate_triage_open_ids_mismatch():
    rj = triage_result(["DW-1", "DW-9"], bundles=[])
    plan, errors = validate_triage(rj, {"DW-1", "DW-2"})
    assert plan is None
    assert "DW-2" in errors[0] and "DW-9" in errors[0]


def test_validate_triage_partition_errors():
    rj = triage_result(
        ["DW-1", "DW-2"],
        already_resolved=[{"id": "DW-1", "evidence": "x"}],
        bundles=[{"name": "b", "dw_ids": ["DW-1"], "intent": "dup claim"}],
    )
    plan, errors = validate_triage(rj, {"DW-1", "DW-2"})
    assert plan is None
    joined = "; ".join(errors)
    assert "DW-1 appears in both" in joined  # double-counted
    assert "not triaged: DW-2" in joined  # missed


def test_validate_triage_bad_fields():
    rj = triage_result(
        ["DW-1", "DW-2"],
        bundles=[{"name": "Bad_Name", "dw_ids": ["DW-1"], "intent": ""}],
        decisions=[
            {
                "id": "DW-2",
                "question": "q",
                "options": [
                    {"key": "1", "label": "a", "effect": "build"},  # build w/o intent
                ],
                "recommendation": "7",
            }
        ],
    )
    plan, errors = validate_triage(rj, {"DW-1", "DW-2"})
    assert plan is None
    joined = "; ".join(errors)
    assert "Bad_Name" in joined
    assert "no intent" in joined
    assert "needs intent" in joined
    assert "at least 2 options" in joined
    assert "recommendation" in joined


# --------------------------------- trailing newline in an identifier (#330)
#
# The other axis from the free-text block below. A bundle name and a bundle key
# are identifiers: the name becomes a path segment (run_dir/bundles/<name>/) and
# a story key, with no sanitizer on either path, so unlike free text they are
# gated. `$` matches before a trailing newline and every call site uses
# `.match()`, so one bare LF slipped through both patterns until `\Z` closed it.


def test_validate_triage_rejects_a_trailing_newline_in_bundle_name():
    rj = triage_result(
        ["DW-1"],
        bundles=[{"name": "fix-it\n", "dw_ids": ["DW-1"], "intent": "do x"}],
    )

    plan, errors = validate_triage(rj, {"DW-1"})

    assert plan is None
    # The error text assertion is load-bearing, not decoration: validate_triage
    # returns None whenever *any* error is present, so a bare `plan is None`
    # would pass for reasons unrelated to the name. Do not simplify it away.
    joined = "; ".join(errors)
    assert repr("fix-it\n") in joined
    assert "invalid" in joined


def test_validate_triage_rejects_a_trailing_newline_in_option_bundle_name():
    rj = triage_result(
        ["DW-1"],
        decisions=[
            {
                "id": "DW-1",
                "question": "renegotiate?",
                "options": [
                    {
                        "key": "1",
                        "label": "build it",
                        "effect": "build",
                        "intent": "do x",
                        "bundle_name": "fix-it\n",
                    },
                    {"key": "2", "label": "keep", "effect": "keep-open"},
                ],
                "recommendation": "1",
            }
        ],
    )

    plan, errors = validate_triage(rj, {"DW-1"})

    assert plan is None
    # Same reason as above: `plan is None` alone would pass on any unrelated
    # error, so the option's own message is what pins the second call site.
    assert "bad bundle_name" in "; ".join(errors)


def test_bundle_key_re_refuses_a_trailing_newline():
    # The hazard this records: before the `\Z` anchor, "dw-foo\n" *matched* with
    # group(2) == "foo", because `.` does not match a newline. _ensure_bundle_intent
    # rebuilds the intent path from that reconstructed name, so a degraded bundle
    # keyed "dw-foo\n" silently regenerated its intent under the different
    # directory "foo", without a word in the journal. That is why this test exists.
    assert BUNDLE_KEY_RE.match("dw-foo\n") is None
    assert BUNDLE_KEY_RE.match("dw2-foo\n") is None

    # Well-formed keys still round-trip through the inverse of _bundle_key.
    assert BUNDLE_KEY_RE.match("dw-foo").groups() == ("", "foo")
    assert BUNDLE_KEY_RE.match("dw2-foo").groups() == ("2", "foo")
    assert BUNDLE_KEY_RE.match("dw-c2-foo").group(2) == "c2-foo"


# ------------------------------ reserved Windows device basenames (#637)
#
# The third axis on the same "a bundle name IS a path segment" surface. A
# reserved device basename is `[a-z0-9-]`-legal and at least 2 characters, so
# BUNDLE_NAME_RE accepts every one of them, and a cycle-1 bundle turns the name
# into run_dir/bundles/<name>/ verbatim -- a directory native Windows will not
# create. The gate is `safe_segment` identity, so the accepted set stays in
# lockstep with the sanitizer instead of a second hand-written device list.


@pytest.mark.parametrize(
    "name",
    ["con", "nul", "aux", "prn", "com1", "com9", "lpt1", "lpt9"],
    ids=["con", "nul", "aux", "prn", "com1", "com9", "lpt1", "lpt9"],
)
def test_validate_triage_rejects_reserved_device_bundle_names(name):
    """ABLATION: delete the safe_segment identity gate and every row here accepts."""
    rj = triage_result(["DW-1"], bundles=[{"name": name, "dw_ids": ["DW-1"], "intent": "do x"}])

    plan, errors = validate_triage(rj, {"DW-1"})

    assert plan is None
    # Exactly one error, not merely one that matches: these names pass
    # BUNDLE_NAME_RE, so a bare `plan is None` (or an `any(...)` over errors)
    # would pass for reasons unrelated to the device name. The count is what
    # says the two name gates report a given defect once between them.
    assert len(errors) == 1
    assert repr(name) in errors[0]
    assert "not a legal path segment" in errors[0]


@pytest.mark.parametrize(
    "name",
    ["CON", "nul.", "aux.txt", "lpt9 "],
    ids=["uppercase", "trailing-dot", "extension", "trailing-space"],
)
def test_validate_triage_reports_one_error_when_a_name_fails_both_gates(name):
    """ABLATION: drop the `BUNDLE_NAME_RE.match(name) and` guard and each row
    double-reports -- two errors for one name. Unlike the reserved-name rows
    above, these fail BUNDLE_NAME_RE *and* safe_segment identity, so they are
    the only inputs on which that guard can be observed at all."""
    rj = triage_result(["DW-1"], bundles=[{"name": name, "dw_ids": ["DW-1"], "intent": "do x"}])

    plan, errors = validate_triage(rj, {"DW-1"})

    assert plan is None
    assert len(errors) == 1
    assert repr(name) in errors[0]
    assert "invalid" in errors[0]


@pytest.mark.parametrize(
    "name",
    ["com10", "console", "com", "lpt", "aux2", "nul-fix", "a" * 40],
    ids=["com10", "console", "com", "lpt", "aux2", "nul-fix", "max-length"],
)
def test_validate_triage_accepts_ordinary_bundle_names(name):
    """The over-refusal guard: `com10` and `console` merely start with a device
    name, and a 40-character name is BUNDLE_NAME_RE's maximum -- well under
    platform_util.MAX_SEGMENT (120), so length never reaches the new gate."""
    rj = triage_result(["DW-1"], bundles=[{"name": name, "dw_ids": ["DW-1"], "intent": "do x"}])

    plan, errors = validate_triage(rj, {"DW-1"})

    assert errors == []
    assert plan is not None
    assert plan.bundles[0].name == name


# ------------------------- the same rule at the decision-option site (#637)
#
# `validate_triage` gates bundle names TWICE. A build-effect option's
# `bundle_name` becomes `Bundle.name` in `_materialize_bundles`, so it reaches
# `_write_intent`'s cycle-1 directory by the same path the `bundles` loop above
# does -- and it was validated against BUNDLE_NAME_RE alone.


def _option_bundle_decision(bundle_name):
    """One otherwise-clean build decision whose only possible defect is its
    option's `bundle_name`. That cleanliness is what lets the tests below assert
    an error COUNT: a decision also carries question / >=2 options /
    recommendation rules, any of which would add errors of their own."""
    return triage_result(
        ["DW-1"],
        decisions=[
            {
                "id": "DW-1",
                "question": "build it?",
                "options": [
                    {
                        "key": "1",
                        "label": "build",
                        "effect": "build",
                        "intent": "fix it",
                        "bundle_name": bundle_name,
                    },
                    {"key": "2", "label": "keep", "effect": "keep-open"},
                ],
                "recommendation": "1",
            }
        ],
    )


@pytest.mark.parametrize(
    "bundle_name",
    ["con", "nul", "aux", "prn", "com1", "com9", "lpt1", "lpt9"],
    ids=["con", "nul", "aux", "prn", "com1", "com9", "lpt1", "lpt9"],
)
def test_validate_triage_rejects_reserved_device_option_bundle_names(bundle_name):
    """ABLATION: delete the safe_segment identity gate at the decision-option
    site and every row here accepts. The `bundles` loop's gate does not reach
    this value -- it is a different loop over a different key."""
    rj = _option_bundle_decision(bundle_name)

    plan, errors = validate_triage(rj, {"DW-1"})

    assert plan is None
    # Exactly one error, not merely one that matches: these names pass
    # BUNDLE_NAME_RE, so a bare `plan is None` would pass for reasons unrelated
    # to the device name.
    assert len(errors) == 1
    assert repr(bundle_name) in errors[0]
    assert "not a legal path segment" in errors[0]
    assert "option 1" in errors[0]


@pytest.mark.parametrize(
    "bundle_name",
    ["CON", "nul.", "aux.txt", "lpt9 "],
    ids=["uppercase", "trailing-dot", "extension", "trailing-space"],
)
def test_validate_triage_reports_one_option_error_when_a_name_fails_both_gates(bundle_name):
    """ABLATION: drop the `BUNDLE_NAME_RE.match(bundle_name) and` guard and each
    row double-reports -- two errors for one name. The rows above cannot show
    this: a lowercase device name PASSES BUNDLE_NAME_RE, so it raises exactly one
    error with or without the guard. Only an input failing both gates can
    observe it at all."""
    rj = _option_bundle_decision(bundle_name)

    plan, errors = validate_triage(rj, {"DW-1"})

    assert plan is None
    assert len(errors) == 1
    assert repr(bundle_name) in errors[0]
    assert "bad bundle_name" in errors[0]


@pytest.mark.parametrize(
    "bundle_name",
    ["com10", "console", "nul-fix"],
    ids=["com10", "console", "nul-fix"],
)
def test_validate_triage_accepts_ordinary_option_bundle_names(bundle_name):
    """The over-refusal guard at this site: `com10` and `console` merely start
    with a device name, and the gate must not reach them."""
    rj = _option_bundle_decision(bundle_name)

    plan, errors = validate_triage(rj, {"DW-1"})

    assert errors == []
    assert plan is not None
    assert plan.decisions[0].option("1").bundle_name == bundle_name


# ---------------------- non-string LLM-authored plan scalars (DW-148)
#
# The DW-141 harm on the plan-INPUT surface. `validate_triage` used to
# `str(...)` every free-text scalar, so a list `intent` became the truthy repr
# "['do', 'x']", satisfied `effect == "build" and not intent`, landed in
# `DecisionOption.intent` and rode into `Bundle.intent`, `intent.md` and a dev
# session. The plan is refused instead, through the existing `errors` channel;
# the message names the id or the bundle position, the field and the TYPE name
# only -- never the offending value's prose -- mirroring
# `unusable_answer_reason`.


def _clean_option_decision(**option_overrides):
    """One otherwise-clean build decision, so the tests below can assert an error
    COUNT: the decision satisfies question / >=2 options / recommendation, and the
    build option carries a valid intent, so the only possible defect is the
    override under test."""
    option = {"key": "1", "label": "build", "effect": "build", "intent": "fix it"}
    option.update(option_overrides)
    return triage_result(
        ["DW-1"],
        decisions=[
            {
                "id": "DW-1",
                "question": "build it?",
                "context": "ctx",
                "options": [option, {"key": "2", "label": "keep", "effect": "keep-open"}],
                "recommendation": "1",
            }
        ],
    )


def test_validate_triage_refuses_a_non_string_option_intent():
    """The headline DW-148 case. ABLATION: restore
    `intent = str(raw.get("intent", "")).strip()` and this reddens -- the plan is
    ACCEPTED with `option("1").intent == "['do', 'x']"`, a Python repr no human
    authored, bound for `Bundle.intent` and an `intent.md`."""
    rj = _clean_option_decision(intent=["do", "x"])

    plan, errors = validate_triage(rj, {"DW-1"})

    assert plan is None
    assert len(errors) == 1
    assert "DW-1" in errors[0]
    assert "intent not a string: list" in errors[0]
    # The value's prose never appears -- the reason `unusable_answer_reason`
    # reports type names, restated here because this text reaches a journal.
    assert not any("['do', 'x']" in e for e in errors)


@pytest.mark.parametrize(
    ("field", "value", "type_name"),
    [
        pytest.param("effect", ["build"], "list", id="effect"),
        # JSON `null` -- the likeliest non-string an LLM emits, and the one shape
        # `str(...)` turned into the literal "None" rather than an obvious repr.
        pytest.param("intent", None, "NoneType", id="intent-null"),
        pytest.param("intent", {"a": 1}, "dict", id="intent"),
        pytest.param("label", 5, "int", id="label"),
        pytest.param("resolution", 5.5, "float", id="resolution"),
        pytest.param("bundle_name", 7, "int", id="bundle_name"),
    ],
)
def test_validate_triage_reports_one_error_per_non_string_option_scalar(field, value, type_name):
    """Every option scalar the intent enumerates, each reporting its own field and
    type name -- and EXACTLY one error apiece. The count is the one-error-per-fault
    convention: threading `None` (not "") is what keeps a non-string `effect` out of
    the "bad effect" branch and a non-string `intent` out of "effect 'build' needs
    intent".
    ABLATION: return "" instead of None from `_plan_str` and the effect and intent
    rows double-report. The label / resolution / bundle_name rows cannot show it --
    "" is a legal value for all three (an absent `bundle_name` already fails
    BUNDLE_NAME_RE without an error) -- they pin the message text and the fact that
    each field is screened at all."""
    rj = _clean_option_decision(**{field: value})

    plan, errors = validate_triage(rj, {"DW-1"})

    assert plan is None
    assert len(errors) == 1
    assert errors[0] == f"decision DW-1 option 1: {field} not a string: {type_name}"


def test_validate_triage_names_an_option_positionally_when_its_key_is_not_a_string():
    """`key` is deliberately NOT type-checked (it keeps `str(...)`), so it cannot
    be trusted to name the option in an error -- the same reason the `bundles` loop
    is positional. Interpolating it would print the offending value's own prose
    into a message that carries type names only, and these reach a journal.
    ABLATION: build `where` as `f"decision {dw_id} option {key}"` unconditionally
    and this reddens with the dict's contents in the message."""
    rj = _clean_option_decision(key={"leaky": "secret prose"}, intent=["do", "x"])
    rj["decisions"][0]["recommendation"] = "{'leaky': 'secret prose'}"

    plan, errors = validate_triage(rj, {"DW-1"})

    assert plan is None
    assert errors == ["decision DW-1 options[0]: intent not a string: list"]
    assert not any("secret prose" in e for e in errors)


def test_validate_triage_keeps_todays_wording_for_an_empty_option_key():
    """The other side of that switch: an empty key is still a STRING key, so the
    existing wording stays byte-identical (`option ` with nothing after it) rather
    than silently moving to the positional form. Pins that the positional fallback
    is keyed on the TYPE, not on truthiness."""
    rj = _clean_option_decision(key="", effect="frobnicate")

    plan, errors = validate_triage(rj, {"DW-1"})

    assert plan is None
    assert "decision DW-1 option : bad effect 'frobnicate'" in errors


def test_validate_triage_reports_one_error_for_a_non_string_recommendation():
    """ABLATION: drop the `recommendation is not None` guard and this double-reports
    -- the type error plus `recommendation '' not an option`."""
    rj = _clean_option_decision()
    rj["decisions"][0]["recommendation"] = 1

    plan, errors = validate_triage(rj, {"DW-1"})

    assert plan is None
    assert errors == ["decision DW-1: recommendation not a string: int"]


def test_validate_triage_reports_one_error_for_a_non_string_decision_context():
    """`Decision.context` is free text a human reads when answering; a repr there
    is the same defect, one field further out."""
    rj = _clean_option_decision()
    rj["decisions"][0]["context"] = ["ctx"]

    plan, errors = validate_triage(rj, {"DW-1"})

    assert plan is None
    assert errors == ["decision DW-1: context not a string: list"]


@pytest.mark.parametrize(
    ("field", "value", "type_name"),
    [
        pytest.param("name", 5, "int", id="name"),
        pytest.param("intent", ["harden"], "list", id="intent"),
    ],
)
def test_validate_triage_reports_one_error_per_non_string_bundle_scalar(field, value, type_name):
    """The bundle is identified POSITIONALLY, not by name: the name itself may be
    the non-string field, so it cannot be what names the bundle in an error. Same
    `bundles[{i}]` label shape `_normalize_bundle_names` already uses.
    ABLATION: fall back to "" and the `name` row double-reports (the type error
    plus `bundle name '' invalid (want ...)`) and the `intent` row double-reports
    (plus `has no intent`)."""
    bundle = {"name": "fix-it", "dw_ids": ["DW-1"], "intent": "harden it"}
    bundle[field] = value
    rj = triage_result(["DW-1"], bundles=[bundle])

    plan, errors = validate_triage(rj, {"DW-1"})

    assert plan is None
    assert len(errors) == 1
    assert errors[0] == f"bundles[0]: {field} not a string: {type_name}"


def test_validate_triage_names_the_right_bundle_position():
    """Two bundles, the SECOND one malformed: the index has to be the offender's,
    not a constant. ABLATION: hard-code `bundles[0]` and this reddens."""
    rj = triage_result(
        ["DW-1", "DW-2"],
        bundles=[
            {"name": "fix-it", "dw_ids": ["DW-1"], "intent": "harden it"},
            {"name": "widen-it", "dw_ids": ["DW-2"], "intent": 7},
        ],
    )

    plan, errors = validate_triage(rj, {"DW-1", "DW-2"})

    assert plan is None
    assert errors == ["bundles[1]: intent not a string: int"]


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(
            lambda rj: rj["decisions"][0]["options"][0].__setitem__("key", 1), id="option-key"
        ),
        pytest.param(lambda rj: rj["decisions"][0].__setitem__("question", 5), id="question"),
    ],
)
def test_validate_triage_leaves_out_of_scope_decision_scalars_stringified(mutate):
    """The scope boundary, asserted rather than assumed -- and it is a SCOPE
    boundary, not a safety argument. `key` and `question` do reach an intent file:
    `_materialize_bundles` interpolates both into `Bundle.decision_note`, which
    `_write_intent` writes under `## Human decision` in
    `run_dir/bundles/<name>/intent.md`, so a list `question` is accepted today and
    lands there as `['a', 'b']`. They keep `str(...)` only because the bundle
    intent enumerates the fields to type-check and these are not among them.
    Widening the check to them would start refusing plans this release accepts, so
    that is a separate decision with its own evidence -- this row exists to make
    the current boundary explicit and to fail loudly if it moves by accident."""
    rj = _clean_option_decision()
    mutate(rj)
    rj["decisions"][0]["recommendation"] = "1"

    plan, errors = validate_triage(rj, {"DW-1"})

    assert errors == []
    assert plan is not None


@pytest.mark.parametrize(
    ("section", "item"),
    [
        pytest.param("already_resolved", {"id": "DW-1", "evidence": 5}, id="evidence"),
        pytest.param("blocked", {"id": "DW-1", "blocker": 5}, id="blocker"),
        pytest.param("skip", {"id": "DW-1", "reason": 5}, id="reason"),
    ],
)
def test_validate_triage_leaves_out_of_scope_section_scalars_stringified(section, item):
    """The other half of the scope boundary. `evidence`, `blocker` and `reason`
    are ledger-bound notes, not `Bundle.intent` input, so they were deliberately
    left on `str(...)`; a non-string one is still accepted."""
    rj = triage_result(["DW-1"], **{section: [item]})

    plan, errors = validate_triage(rj, {"DW-1"})

    assert errors == []
    assert plan is not None


def test_validate_triage_truncates_overlong_bundle_name():
    """ABLATION A1: delete direct-bundle normalization and this fails on validation."""
    rj = triage_result(
        ["DW-1"],
        bundles=[{"name": _OVERLONG_BUNDLE_NAME, "dw_ids": ["DW-1"], "intent": "fix it"}],
    )

    plan, errors = validate_triage(rj, {"DW-1"})

    assert errors == []
    assert plan is not None
    assert plan.bundles[0].name == _NORMALIZED_BUNDLE_NAME
    assert rj["bundles"][0]["name"] == _NORMALIZED_BUNDLE_NAME


def test_validate_triage_truncates_overlong_decision_bundle_name():
    """ABLATION A2: delete decision-option normalization and this fails validation."""
    rj = triage_result(
        ["DW-1"],
        decisions=[
            {
                "id": "DW-1",
                "question": "build it?",
                "options": [
                    {
                        "key": "1",
                        "label": "build",
                        "effect": "build",
                        "intent": "fix it",
                        "bundle_name": _OVERLONG_BUNDLE_NAME,
                    },
                    {"key": "2", "label": "keep", "effect": "keep-open"},
                ],
                "recommendation": "1",
            }
        ],
    )

    plan, errors = validate_triage(rj, {"DW-1"})

    assert errors == []
    assert plan is not None
    assert plan.decisions[0].option("1").bundle_name == _NORMALIZED_BUNDLE_NAME
    assert rj["decisions"][0]["options"][0]["bundle_name"] == _NORMALIZED_BUNDLE_NAME


def test_validate_triage_rejects_post_truncation_bundle_name_collision():
    """ABLATION A1: delete direct normalization and the duplicate error disappears."""
    shared_prefix = "a" * 40
    rj = triage_result(
        ["DW-1", "DW-2"],
        bundles=[
            {"name": shared_prefix + "x", "dw_ids": ["DW-1"], "intent": "fix one"},
            {"name": shared_prefix + "y", "dw_ids": ["DW-2"], "intent": "fix two"},
        ],
    )

    plan, errors = validate_triage(rj, {"DW-1", "DW-2"})

    assert plan is None
    assert f"duplicate bundle name {shared_prefix!r}" in errors


@pytest.mark.parametrize("with_direct_bundle", [True, False], ids=["direct-decision", "decisions"])
def test_validate_triage_rejects_post_truncation_decision_bundle_name_collision(
    with_direct_bundle,
):
    """Selected build options cannot alias a direct or another decision bundle task."""
    shared_prefix = "a" * 40

    def decision(dw_id, suffix):
        return {
            "id": dw_id,
            "question": "build it?",
            "options": [
                {
                    "key": "1",
                    "label": "build",
                    "effect": "build",
                    "intent": "fix it",
                    "bundle_name": shared_prefix + suffix,
                },
                {"key": "2", "label": "keep", "effect": "keep-open"},
            ],
            "recommendation": "1",
        }

    bundles = (
        [{"name": shared_prefix + "x", "dw_ids": ["DW-1"], "intent": "fix one"}]
        if with_direct_bundle
        else []
    )
    decisions = [decision("DW-2", "y")]
    if not with_direct_bundle:
        decisions.insert(0, decision("DW-1", "x"))
    rj = triage_result(["DW-1", "DW-2"], bundles=bundles, decisions=decisions)

    plan, errors = validate_triage(rj, {"DW-1", "DW-2"})

    assert plan is None
    assert errors.count(f"duplicate bundle name {shared_prefix!r}") == 1


def test_validate_triage_allows_post_truncation_collision_between_sibling_options():
    """Mutually exclusive options in one decision cannot materialize together."""
    shared_prefix = "a" * 40
    rj = triage_result(
        ["DW-1"],
        decisions=[
            {
                "id": "DW-1",
                "question": "which fix?",
                "options": [
                    {
                        "key": "1",
                        "label": "first",
                        "effect": "build",
                        "intent": "fix it first",
                        "bundle_name": shared_prefix + "x",
                    },
                    {
                        "key": "2",
                        "label": "second",
                        "effect": "build",
                        "intent": "fix it second",
                        "bundle_name": shared_prefix + "y",
                    },
                ],
                "recommendation": "1",
            }
        ],
    )

    plan, errors = validate_triage(rj, {"DW-1"})

    assert errors == []
    assert plan is not None
    assert [option.bundle_name for option in plan.decisions[0].options] == [
        shared_prefix,
        shared_prefix,
    ]


@pytest.mark.parametrize(
    "value",
    [
        "A" + "a" * 40,
        "a" * 40 + "/",
        "a" * 40 + " ",
        "a" * 40 + "_",
    ],
    ids=["uppercase", "slash", "whitespace", "underscore"],
)
def test_validate_triage_does_not_repair_overlong_malformed_bundle_name(value):
    """Ablation target: delete the direct-name regex gate and this accepts each value."""
    rj = triage_result(["DW-1"], bundles=[{"name": value, "dw_ids": ["DW-1"], "intent": "fix it"}])

    plan, errors = validate_triage(rj, {"DW-1"})

    assert plan is None
    assert any(f"bundle name {value!r} invalid" in error for error in errors)
    assert rj["bundles"][0]["name"] == value


@pytest.mark.parametrize(
    ("updates", "expected_error"),
    [
        ({"workflow": "wrong", "bundles": None}, "workflow must be"),
        ({"open_ids": ["DW-2"], "bundles": [None]}, "open_ids do not match"),
        ({"open_ids": ["DW-2"], "decisions": [None]}, "open_ids do not match"),
        (
            {"open_ids": ["DW-2"], "decisions": [{"options": [None]}]},
            "open_ids do not match",
        ),
    ],
    ids=["null-container", "non-object-bundle", "non-object-decision", "non-object-option"],
)
def test_validate_triage_malformed_name_containers_do_not_preempt_early_feedback(
    updates, expected_error
):
    """Normalization must not crash before the validator's established early feedback."""
    rj = triage_result(["DW-1"])
    rj.update(updates)

    plan, errors = validate_triage(rj, {"DW-1"})

    assert plan is None
    assert expected_error in errors[0]


# --------------------------------- line breaks in ledger-bound text (#305)
#
# validate_triage deliberately does NOT gate on line breaks. The sanitizer in
# deferredwork already neutralizes them losslessly, so a rejection here would buy
# no integrity — it would only spend a triage attempt, and the second failure
# escalates the run to a human who has nothing to resolve. Acceptance is the
# behavior under test; the two live writer paths below are where it lands.


@pytest.mark.parametrize("field", ["evidence", "label", "resolution"])
def test_validate_triage_accepts_multiline_free_text(field):
    """A formatting-only defect must never cost a triage attempt or pause a run.
    Deleting the plan over one line break trains nothing — the feedback file dies
    with the session — and the escalation pages a human whose only remedy is to
    re-roll the same dice."""
    injected = "fixed in abc123\n### DW-99: fake\nstatus: open"
    if field == "evidence":
        rj = triage_result(["DW-1"], already_resolved=[{"id": "DW-1", "evidence": injected}])
    else:
        option = {"key": "1", "label": "build it", "effect": "build", "intent": "do x"}
        option[field] = injected
        rj = triage_result(
            ["DW-1"],
            decisions=[
                {
                    "id": "DW-1",
                    "question": "renegotiate?",
                    "context": "ctx",
                    "options": [option, {"key": "2", "label": "keep", "effect": "keep-open"}],
                    "recommendation": "1",
                }
            ],
        )

    plan, errors = validate_triage(rj, {"DW-1"})

    assert errors == []
    assert plan is not None


def test_close_resolved_sanitizes_an_injected_evidence(project):
    """First live writer path (`evidence` -> the `resolution:` note). The run
    completes and the ledger keeps its shape — no raise into `_cycle`'s bare
    call, which would end the sweep as crashed."""
    write_ledger(project, {"DW-1": "open", "DW-2": "open"})
    engine, _adapter = make_sweep(project, [])
    before = len(ledger_entries(project))
    plan = TriagePlan(
        open_ids=frozenset({"DW-1", "DW-2"}),
        already_resolved=(ResolvedEntry("DW-1", "fixed\n### DW-99: fake\nstatus: open"),),
    )

    assert engine._close_resolved(plan) == 1

    entries = ledger_entries(project)
    assert len(entries) == before  # no phantom entry minted
    assert set(entries) == {"DW-1", "DW-2"}
    assert entries["DW-1"].status.startswith("done ")
    assert entries["DW-2"].open
    assert "already resolved: fixed ### DW-99: fake status: open" in entries["DW-1"].body


@pytest.mark.parametrize(
    ("resolution", "intent", "expected_detail"),
    [
        ("close it\n### DW-99: fake", "unused\nintent", "close it ### DW-99: fake"),
        ("", "widen the field.\nThen backfill.", "widen the field. Then backfill."),
    ],
    ids=["resolution-wins", "intent-when-no-resolution"],
)
def test_apply_decision_effect_sanitizes_the_ledger_detail(
    project, resolution, intent, expected_detail
):
    """Second live writer path, and the one that carries an option's `intent` to
    the ledger. Pins `detail = option.resolution or option.intent`: the
    resolution wins when present, the intent is the fallback, and either way
    `append_decision` flattens it — which is why gating `intent` upstream would
    prevent nothing while flattening the brief that drives a dev bundle."""
    write_ledger(project, {"DW-1": "open"})
    engine, _adapter = make_sweep(project, [])
    option = DecisionOption(
        key="1", label="Keep\ncap", effect="keep-open", intent=intent, resolution=resolution
    )
    decision = Decision(id="DW-1", question="?", context="", options=(option,), recommendation="1")

    engine._apply_decision_effect(decision, option)

    text = project.deferred_work.read_text(encoding="utf-8")
    decision_lines = [line for line in text.splitlines() if line.startswith("decision:")]
    assert len(decision_lines) == 1
    assert decision_lines[0].endswith(f"Keep cap — {expected_detail}")
    assert ledger_entries(project)["DW-1"].open  # keep-open does not close


def test_apply_decision_effect_close_sanitizes_the_close_note(project):
    write_ledger(project, {"DW-1": "open"})
    engine, _adapter = make_sweep(project, [])
    option = DecisionOption(
        key="1", label="Close", effect="close", resolution="superseded\n### DW-99: fake"
    )
    decision = Decision(id="DW-1", question="?", context="", options=(option,), recommendation="1")

    engine._apply_decision_effect(decision, option)

    entries = ledger_entries(project)
    assert set(entries) == {"DW-1"}  # no phantom entry from the close note
    assert entries["DW-1"].status.startswith("done ")
    assert (
        "resolution: closed by human decision: superseded ### DW-99: fake" in entries["DW-1"].body
    )


def test_validate_triage_unknown_id():
    rj = triage_result(
        ["DW-1"],
        skip=[{"id": "DW-1", "reason": "moot"}, {"id": "DW-42", "reason": "ghost"}],
    )
    plan, errors = validate_triage(rj, {"DW-1"})
    assert plan is None
    assert any("DW-42" in e for e in errors)


# ---------------------------------------------------- validate_migration

LEGACY_LEDGER = (
    "# Deferred Work\n\n"
    "## Deferred from: epic 1 review (2026-04-06)\n\n"
    "- ~~**Old fixed thing** — was broken, then repaired~~ → fixed in 1.3\n"
    "- **Open legacy thing here** — `src.txt` mishandles em-dashes\n"
)


def legacy_manifest(text: str = LEGACY_LEDGER) -> list[dict]:
    return [
        {
            "key": e.key,
            "id": e.id,
            "title": e.title,
            "section": e.section,
            "done": e.done,
            "severity": e.severity,
        }
        for e in deferredwork.parse_legacy(text)
    ]


def migrated_ledger(first_id: int = 1) -> str:
    return (
        "# Deferred Work\n\n"
        f"### DW-{first_id}: Old fixed thing\n\n"
        "origin: migrated from legacy ledger, 2026-06-12\nlocation: n/a\n"
        "reason: was broken, then repaired.\nstatus: done 2026-04-06\n\n"
        f"### DW-{first_id + 1}: Open legacy thing here\n\n"
        "origin: migrated from legacy ledger, 2026-06-12\nlocation: src.txt\n"
        "reason: mishandles em-dashes.\nstatus: open\n"
    )


def migrate_result(mapping) -> dict:
    return {"workflow": "deferred-sweep-migrate", "mapping": list(mapping), "escalations": []}


def _gated_dw1(*gate_lines: str) -> str:
    """The pre-existing canonical DW-1 both migration halves below carry, so the
    two texts differ ONLY in their second half by construction rather than by
    inspection — an accepted rewrite really did leave the entry untouched."""
    return (
        "### DW-1: item DW-1\n\norigin: test, 2026-06-01\nlocation: src.txt:1\n"
        "reason: test entry.\nstatus: open\n" + "".join(f"{line}\n" for line in gate_lines)
    )


def pre_gated_ledger(*gate_lines: str) -> str:
    """The ledger a migration is handed: one pre-existing canonical DW-1
    carrying the given `gate:` lines verbatim, plus the one legacy item to
    convert. Mirrors `test_mixed_ledger_migration_preserves_canonical_open_set`,
    which is the shape a real pre-DW-format project reaches migration in."""
    return (
        "# Deferred Work\n\n" + _gated_dw1(*gate_lines) + "\n"
        "## Deferred from: epic 1 review (2026-04-06)\n\n"
        "- **Open legacy thing here** — `src.txt` mishandles em-dashes\n"
    )


def rewritten_gated_ledger(*gate_lines: str) -> str:
    """The rewrite the migration session hands back: DW-1 preserved with the
    given `gate:` lines (pass none for the drop this guards against), the legacy
    item converted to DW-2."""
    return (
        "# Deferred Work\n\n" + _gated_dw1(*gate_lines) + "\n"
        "### DW-2: Open legacy thing here\n\n"
        "origin: migrated from legacy ledger, 2026-06-12\nlocation: src.txt\n"
        "reason: mishandles em-dashes.\nstatus: open\n"
    )


def _gated_migration_case(*gate_lines: str):
    """(manifest, result.json, snapshot) for a migration handed
    `pre_gated_ledger(*gate_lines)`. The `len(manifest) == 1` precondition rides
    here so no caller can omit it: without it a fixture whose canonical entry
    accidentally parsed as legacy would make the whole test mean something
    else."""
    before = pre_gated_ledger(*gate_lines)
    manifest = legacy_manifest(before)
    assert len(manifest) == 1  # the canonical entry is not a legacy item
    rj = migrate_result([{"key": manifest[0]["key"], "dw_id": "DW-2"}])
    return manifest, rj, snapshot_canonical(before)


def test_validate_migration_refuses_a_dropped_gate_token():
    """#519. A rewrite that drops a `gate:` line a pre-existing entry declared is
    refused, so a migration cannot silently un-gate the story that entry exists
    to hold back.

    Ablation: delete the `post`/`kept`/`lost` block in `validate_migration` and
    this test fails alone on the final assertion — the positive control above it
    stays green, which is what separates "the guard fired" from "the fixture is
    broken"."""
    manifest, rj, pre = _gated_migration_case("gate: 3-2")
    # the SNAPSHOT carries the token at all, so a failure below cannot be
    # blamed on the fixture
    assert pre["DW-1"].gate_tokens == ("3-2",)
    # the paired positive control: the same rewrite with the gate KEPT is fully
    # accepted. Without it the refusal could be produced by any unrelated defect
    # in the fixture and this test would prove nothing.
    assert validate_migration(rj, manifest, pre, rewritten_gated_ledger("gate: 3-2")) == []

    errors = validate_migration(rj, manifest, pre, rewritten_gated_ledger())
    assert any("DW-1 lost gate token" in e and "3-2" in e for e in errors)


def test_validate_migration_refuses_an_altered_gate_token():
    """#519. Editing a token is a drop plus an add, and the drop half is what the
    guard reads — so a rewrite that renames `3-2` to a different story is refused
    exactly like one that deleted the line.

    Ablation: delete the `post`/`kept`/`lost` block in `validate_migration` and
    this test fails."""
    manifest, rj, pre = _gated_migration_case("gate: 3-2")
    # `3-20` is the control: a token that merely shares a prefix must not keep
    # `3-2` alive, which is what stops the comparison ever being written as a
    # `startswith` (`3-20-later` is a different story).
    errors = validate_migration(rj, manifest, pre, rewritten_gated_ledger("gate: 3-20"))
    assert any("DW-1 lost gate token" in e and "3-2" in e for e in errors)


def test_validate_migration_refuses_a_dropped_malformed_gate_token():
    """#519. `gate: 3.2` is `malformed` — no story key spells its numbers with a
    `.` — and dropping it is refused all the same. It gated nothing, but it reads
    to anyone scanning the entry as a gate in force and `bmad-loop validate`
    reports it (`deferred.hard-gate-unstructured`); a migration that deletes it
    silently retires that report, which is the same failure one level down.

    Ablation: change `snapshot_canonical` to snapshot `g.tokens` alone and this
    test fails ALONE — the enforceable-token tests never touch `malformed`."""
    manifest, rj, pre = _gated_migration_case("gate: 3.2")
    assert pre["DW-1"].gate_tokens == ("3.2",)  # the malformed half really is snapshotted

    errors = validate_migration(rj, manifest, pre, rewritten_gated_ledger())
    assert any("DW-1 lost gate token" in e and "3.2" in e for e in errors)


def test_validate_migration_allows_a_reflowed_gate_declaration():
    """#519. Two `gate:` lines folded onto one, reordered, lose nothing — and a
    migration legitimately reformats. This is the control that keeps the three
    refusal tests above from being satisfiable by a byte comparison or an
    ordered-tuple comparison, either of which would refuse this rewrite.

    Ablation: compare `pre.gate_tokens` against `post.tokens + post.malformed`
    as a sequence instead of by membership and this test fails while the refusal
    tests stay green."""
    manifest, rj, pre = _gated_migration_case("gate: 3-2", "gate: 3-3")
    assert pre["DW-1"].gate_tokens == ("3-2", "3-3")

    assert validate_migration(rj, manifest, pre, rewritten_gated_ledger("gate: 3-3, 3-2")) == []


def test_validate_migration_allows_an_added_gate_token():
    """#519. The asymmetry is a deliberate bound, not an oversight: a dropped
    token un-gates a story silently, while an added one over-blocks loudly and in
    the safe direction — the operator meets a refusal naming the entry. Only the
    drop is refused, so a migration attempt is never spent on the one direction
    that cannot cause the failure the guard exists to stop.

    Green-ablation record: no mutation of the #519 guard reddens this test,
    because it pins the guard's BOUND rather than the guard. The drop direction
    is pinned by `test_validate_migration_refuses_a_dropped_gate_token`."""
    manifest, rj, pre = _gated_migration_case("gate: 3-2")

    assert validate_migration(rj, manifest, pre, rewritten_gated_ledger("gate: 3-2, 3-4")) == []


def test_duplicate_ids_names_every_repeated_id_once():
    """#519 (user-approved scope addition). The predicate both sides of a
    migration are refused on — the ledger it starts from and the ledger it
    produces — so it is graded once here rather than twice through its callers.

    A repeated id is named ONCE however many times it repeats: the message tells
    an operator which id to go renumber, and `DW-1, DW-1` would read as two
    separate problems.

    Ablation: this pins a pure predicate, so its ablation is its callers' —
    `test_migration_refuses_a_ledger_carrying_duplicate_dw_ids` deletes the
    `_ensure_migration` guard, and the pre-existing `duplicate DW ids` unit
    coverage holds the rewrite side."""
    entries = deferredwork.parse_ledger(
        "# Deferred Work\n\n"
        + _gated_dw1("gate: 3-2")
        + "\n"
        + _gated_dw1("gate: 3-3")
        + "\n"
        + _gated_dw1("gate: 3-4")
        + "\n"
    )
    assert len(entries) == 3  # the fixture really does carry three parsed entries
    assert duplicate_ids(entries) == ["DW-1"]
    # and a ledger with no repeat is not refused
    assert duplicate_ids(deferredwork.parse_ledger(pre_gated_ledger("gate: 3-2"))) == []


def test_validate_migration_happy():
    manifest = legacy_manifest()
    done_key, open_key = manifest[0]["key"], manifest[1]["key"]
    rj = migrate_result([{"key": done_key, "dw_id": "DW-1"}, {"key": open_key, "dw_id": "DW-2"}])
    assert validate_migration(rj, manifest, {}, migrated_ledger()) == []


def test_validate_migration_rejects_leftover_legacy():
    manifest = legacy_manifest()
    half_done = migrated_ledger() + "\n## Deferred from: leftovers\n\n- still freeform item\n"
    rj = migrate_result(
        [{"key": manifest[0]["key"], "dw_id": "DW-1"}, {"key": manifest[1]["key"], "dw_id": "DW-2"}]
    )
    errors = validate_migration(rj, manifest, {}, half_done)
    assert any("still parse as legacy" in e and "still freeform item" in e for e in errors)


def test_validate_migration_guards_pre_existing_canonical():
    manifest = legacy_manifest()
    # DW-1 regressed to done; DW-9 vanished
    pre = {
        "DW-1": PreCanonical("open", (), None),
        "DW-9": PreCanonical("open", (), None),
    }
    rj = migrate_result(
        [{"key": manifest[0]["key"], "dw_id": "DW-1"}, {"key": manifest[1]["key"], "dw_id": "DW-2"}]
    )
    errors = validate_migration(rj, manifest, pre, migrated_ledger())
    joined = "; ".join(errors)
    assert "DW-1 status changed" in joined
    assert "DW-9 disappeared" in joined
    # and the new DW-2 does not continue numbering past DW-9
    assert "does not continue numbering past DW-9" in joined


def test_validate_migration_refuses_changed_pre_existing_canonical_severity():
    before = pre_gated_ledger().replace("status: open", "severity: high\nstatus: open", 1)
    manifest = legacy_manifest(before)
    assert len(manifest) == 1
    rj = migrate_result([{"key": manifest[0]["key"], "dw_id": "DW-2"}])
    pre = snapshot_canonical(before)
    assert pre["DW-1"].severity == "high"
    kept = rewritten_gated_ledger().replace("status: open", "severity: high\nstatus: open", 1)
    changed = rewritten_gated_ledger().replace("status: open", "severity: low\nstatus: open", 1)

    assert validate_migration(rj, manifest, pre, kept) == []
    errors = validate_migration(rj, manifest, pre, changed)

    assert errors == ["pre-existing DW-1 severity changed: 'high' -> 'low'"]


def test_validate_migration_mapping_errors():
    manifest = legacy_manifest()
    done_key = manifest[0]["key"]
    rj = migrate_result(
        [
            {"key": done_key, "dw_id": "DW-2"},  # done-ness mismatch (DW-2 is open)
            {"key": "no-such-key", "dw_id": "DW-1"},  # invented
            {"key": done_key, "dw_id": "DW-77"},  # repeated key + missing entry
        ]
    )
    errors = validate_migration(rj, manifest, {}, migrated_ledger())
    joined = "; ".join(errors)
    assert "manifest says done, ledger disagrees" in joined
    assert "invents unknown key" in joined
    assert "repeats key" in joined
    assert "DW-77: no such entry" in joined
    assert "not mapped" in joined  # the open item's key never appeared


def test_validate_migration_refuses_mapping_legacy_to_a_pre_existing_entry():
    before = pre_gated_ledger()
    manifest = legacy_manifest(before)
    assert len(manifest) == 1
    mapping = migrate_result([{"key": manifest[0]["key"], "dw_id": "DW-1"}])

    errors = validate_migration(mapping, manifest, snapshot_canonical(before), before)

    assert any("legacy items must map to newly created entries" in error for error in errors)


def test_validate_migration_compares_arbitrarily_large_dw_ids_without_int_conversion():
    huge_suffix = "9" * 5_000
    next_suffix = "1" + "0" * 5_000
    before = "# Deferred Work\n\n" f"### DW-{huge_suffix}: existing\n\norigin: test\nstatus: open\n"
    rewritten = (
        before + "\n" + f"### DW-{next_suffix}: migrated\n\norigin: migrated\nstatus: open\n"
    )
    manifest = [{"key": "legacy-1", "done": False, "severity": None}]
    result = migrate_result([{"key": "legacy-1", "dw_id": f"DW-{next_suffix}"}])

    assert validate_migration(result, manifest, snapshot_canonical(before), rewritten) == []


def test_validate_migration_normalizes_unicode_decimal_ids_for_numbering():
    before = "# Deferred Work\n\n### DW-９: existing\n\norigin: test\nstatus: open\n"
    rewritten = before + "\n### DW-10: migrated\n\norigin: migrated\nstatus: open\n"
    manifest = [{"key": "legacy-1", "done": False, "severity": None}]
    result = migrate_result([{"key": "legacy-1", "dw_id": "DW-10"}])

    assert validate_migration(result, manifest, snapshot_canonical(before), rewritten) == []


def test_validate_migration_requires_contiguous_ids_in_manifest_order():
    legacy = (
        "# Deferred Work\n\n"
        "### D-1: first legacy\n\nreason: first\n\n"
        "### D-2: second legacy\n\nreason: second\n"
    )
    manifest = legacy_manifest(legacy)
    rewritten = (
        "# Deferred Work\n\n"
        "### DW-1: second legacy\n\norigin: migrated\nstatus: open\n\n"
        "### DW-2: first legacy\n\norigin: migrated\nstatus: open\n"
    )
    result = migrate_result(
        [
            {"key": manifest[0]["key"], "dw_id": "DW-2"},
            {"key": manifest[1]["key"], "dw_id": "DW-1"},
        ]
    )

    errors = validate_migration(result, manifest, {}, rewritten)

    assert any("must follow manifest order; expected DW-1" in error for error in errors)


def test_validate_migration_allows_nonadjacent_dedupe_merge():
    manifest = [
        {"key": "duplicate-a", "done": False, "severity": None},
        {"key": "other", "done": False, "severity": None},
        {"key": "duplicate-c", "done": False, "severity": None},
    ]
    rewritten = (
        "# Deferred Work\n\n"
        "### DW-1: merged duplicate\n\norigin: migrated\nstatus: open\n\n"
        "### DW-2: other\n\norigin: migrated\nstatus: open\n"
    )
    result = migrate_result(
        [
            {"key": "duplicate-a", "dw_id": "DW-1"},
            {"key": "other", "dw_id": "DW-2"},
            {"key": "duplicate-c", "dw_id": "DW-1"},
        ]
    )

    assert validate_migration(result, manifest, {}, rewritten) == []


def test_validate_migration_allows_dedupe_merge():
    # two legacy items of equal done-ness may merge into one DW entry
    text = (
        "## Deferred from: review A (2026-04-06)\n\n- same thing, worded one way\n"
        "## Deferred from: review B (2026-04-07)\n\n- same thing, worded another way\n"
    )
    manifest = legacy_manifest(text)
    merged = (
        "# Deferred Work\n\n### DW-1: same thing\n\n"
        "origin: migrated from legacy ledger, 2026-06-12\nlocation: n/a\n"
        "reason: seen in review A and review B.\nstatus: open\n"
    )
    rj = migrate_result([{"key": m["key"], "dw_id": "DW-1"} for m in manifest])
    assert validate_migration(rj, manifest, {}, merged) == []


def test_validate_migration_wrong_workflow():
    errors = validate_migration({"workflow": "quick-dev"}, [], {}, "")
    assert errors and "workflow" in errors[0]


def test_validate_migration_rejects_changed_manifest_severity():
    manifest = [
        {
            "key": "legacy-1",
            "id": "D-1",
            "title": "legacy",
            "section": "",
            "done": False,
            "severity": "high",
        }
    ]
    migrated = (
        "# Deferred Work\n\n### DW-1: legacy\n\norigin: migrated\n" "severity: low\nstatus: open\n"
    )
    rj = migrate_result([{"key": "legacy-1", "dw_id": "DW-1"}])

    errors = validate_migration(rj, manifest, {}, migrated)

    assert errors == ["mapping legacy-1 -> DW-1: manifest severity 'high', ledger has 'low'"]


def test_validate_migration_uses_highest_severity_for_a_merged_target():
    manifest = [
        {"key": "low", "done": False, "severity": "low"},
        {"key": "critical", "done": False, "severity": "critical"},
        {"key": "missing", "done": False, "severity": None},
    ]
    rj = migrate_result(
        [
            {"key": "low", "dw_id": "DW-1"},
            {"key": "critical", "dw_id": "DW-1"},
            {"key": "missing", "dw_id": "DW-1"},
        ]
    )
    correct = (
        "# Deferred Work\n\n### DW-1: merged\n\norigin: migrated\n"
        "severity: critical\nstatus: open\n"
    )
    weakened = correct.replace("severity: critical", "severity: high")

    assert validate_migration(rj, manifest, {}, correct) == []
    errors = validate_migration(rj, manifest, {}, weakened)

    assert errors == [
        "merged mapping -> DW-1: highest manifest severity 'critical', ledger has 'high'"
    ]


# ------------------------------------------------------------ engine flow


def test_sweep_nothing_open(project):
    write_ledger(project, {"DW-1": "done 2026-06-01"})
    engine, adapter = make_sweep(project, [])
    summary = engine.run()
    assert summary.done == 0 and not summary.paused
    assert adapter.sessions == []
    assert "sweep-nothing-open" in journal_text(engine)


def test_only_scopes_triage_and_audits_excluded_open_entries(project):
    write_ledger(project, {"DW-1": "open", "DW-2": "open", "DW-3": "open"})
    plan = triage_result(
        ["DW-3", "DW-1"],
        skip=[
            {"id": "DW-1", "reason": "leave it"},
            {"id": "DW-3", "reason": "leave it too"},
        ],
    )
    engine, adapter = make_sweep(
        project,
        [triage_effect(plan)],
        only_ids=("DW-3", "DW-1"),
    )

    summary = engine.run()

    assert not summary.crashed and len(adapter.sessions) == 1
    assert adapter.sessions[0].prompt == "/bmad-loop-sweep --only DW-3,DW-1"
    excluded = [
        entry for entry in engine.journal.entries() if entry["kind"] == "sweep-selection-excluded"
    ]
    assert excluded[0]["dw_ids"] == ["DW-2"]
    assert all(entry.open for entry in ledger_entries(project).values())


def test_only_selects_parser_supported_unicode_decimal_id(project):
    project.deferred_work.write_text(
        "# Deferred Work\n\n### DW-９: unicode id\n\norigin: test\nstatus: open\n",
        encoding="utf-8",
    )
    plan = triage_result(["DW-９"], skip=[{"id": "DW-９", "reason": "leave it"}])
    engine, adapter = make_sweep(project, [triage_effect(plan)], only_ids=("DW-９",))

    summary = engine.run()

    assert not summary.crashed
    assert adapter.sessions[0].prompt == "/bmad-loop-sweep --only DW-９"


@pytest.mark.parametrize("only_id", ["DW-9", "DW-2"], ids=["unknown", "not-open"])
def test_only_refuses_an_id_outside_the_initial_open_universe(project, only_id):
    """Ablation: remove select_entries' validate_only refusal and this finishes silently."""
    write_ledger(project, {"DW-1": "open", "DW-2": "done 2026-06-01"})
    engine, adapter = make_sweep(project, [], only_ids=(only_id,))

    summary = engine.run()

    assert summary.crashed
    assert adapter.sessions == []
    assert "must exist and be open" in (engine.run_dir / "crash.txt").read_text()


def test_only_refuses_when_the_initial_ledger_has_no_open_entries(project):
    write_ledger(project, {"DW-1": "done 2026-06-01"})
    engine, adapter = make_sweep(project, [], only_ids=("DW-1",))

    summary = engine.run()

    assert summary.crashed and adapter.sessions == []
    assert "must exist and be open: DW-1" in (engine.run_dir / "crash.txt").read_text()


def test_resumed_only_scope_contracts_after_its_entry_closed(project):
    write_ledger(project, {"DW-1": "done 2026-06-01", "DW-2": "open"})
    original, _ = make_sweep(project, [], only_ids=("DW-1",))
    triage = StoryTask(story_key="sweep-triage", epic=0)
    triage.phase = Phase.DONE
    original.state.tasks[triage.story_key] = triage
    save_state(original.run_dir, original.state)
    resumed, adapter = resume_sweep(project, original, [], only_ids=("DW-1",))

    summary = resumed.run()

    assert not summary.crashed and not summary.paused
    assert adapter.sessions == []
    assert "sweep-selection-empty" in journal_kinds(resumed)


def test_repeat_only_scope_contracts_after_selected_entry_closes(project):
    write_ledger(project, {"DW-1": "open", "DW-2": "open"})
    plan = triage_result(
        ["DW-1"],
        already_resolved=[{"id": "DW-1", "evidence": "fixed at src.txt:1"}],
    )
    engine, adapter = make_sweep(
        project,
        [triage_effect(plan)],
        only_ids=("DW-1",),
        repeat=True,
    )

    summary = engine.run()

    assert not summary.crashed and not summary.paused
    assert len(adapter.sessions) == 1
    done = [entry for entry in engine.journal.entries() if entry["kind"] == "sweep-repeat-done"]
    assert done[-1]["reason"] == "no-selected"


def test_min_severity_is_fence_safe_and_reports_every_exclusion(project):
    project.deferred_work.write_text(
        "# Deferred Work\n\n"
        "### DW-1: high\n\nseverity: high\nstatus: open\n\n"
        "### DW-2: low\n\npriority: minor\nstatus: open\n\n"
        "### DW-3: quoted only\n\n```markdown\nseverity: critical\n```\nstatus: open\n",
        encoding="utf-8",
    )
    git(project.project, "add", "-A")
    git(project.project, "commit", "-q", "-m", "ledger")
    plan = triage_result(["DW-1"], skip=[{"id": "DW-1", "reason": "leave it"}])
    engine, adapter = make_sweep(
        project,
        [triage_effect(plan)],
        min_severity="high",
    )

    summary = engine.run()

    assert not summary.crashed
    assert adapter.sessions[0].prompt == "/bmad-loop-sweep --only DW-1"
    records = engine.journal.entries()
    excluded = [e for e in records if e["kind"] == "sweep-selection-excluded"]
    missing = [e for e in records if e["kind"] == "sweep-selection-missing-severity"]
    assert excluded[0]["dw_ids"] == ["DW-2", "DW-3"]
    assert missing[0]["dw_ids"] == ["DW-3"]


def test_repeat_re_evaluates_severity_and_admits_a_new_matching_entry(project):
    project.deferred_work.write_text(
        "# Deferred Work\n\n### DW-1: first\n\nseverity: high\nstatus: open\n",
        encoding="utf-8",
    )
    git(project.project, "add", "-A")
    git(project.project, "commit", "-q", "-m", "ledger")
    first_plan = triage_result(
        ["DW-1"], already_resolved=[{"id": "DW-1", "evidence": "fixed at src.txt:1"}]
    )
    second_plan = triage_result(["DW-2"], skip=[{"id": "DW-2", "reason": "new but not actionable"}])

    def first_triage(_spec):
        assert (
            deferredwork.append_entry(
                project.deferred_work,
                title="second",
                origin="repeat test",
                source_spec="spec-repeat.md",
                reason="appeared during cycle one",
                severity="critical",
            )
            == "DW-2"
        )
        return SessionResult(status="completed", result_json=first_plan)

    engine, adapter = make_sweep(
        project,
        [first_triage, triage_effect(second_plan)],
        min_severity="high",
        repeat=True,
    )

    summary = engine.run()

    assert not summary.crashed
    assert [session.prompt for session in adapter.sessions] == [
        "/bmad-loop-sweep --only DW-1",
        "/bmad-loop-sweep --only DW-2",
    ]


def test_selection_precedes_max_bundle_truncation(project):
    write_ledger(project, {"DW-1": "open", "DW-2": "open", "DW-3": "open"})
    plan = triage_result(
        ["DW-2", "DW-3"],
        bundles=[
            {"name": "two", "dw_ids": ["DW-2"], "intent": "fix two"},
            {"name": "three", "dw_ids": ["DW-3"], "intent": "fix three"},
        ],
    )
    engine, adapter = make_sweep(
        project,
        [triage_effect(plan)],
        only_ids=("DW-2", "DW-3"),
        max_bundles=1,
        decisions_only=True,
    )

    summary = engine.run()

    assert not summary.crashed and len(adapter.sessions) == 1
    truncated = [e for e in engine.journal.entries() if e["kind"] == "sweep-bundles-truncated"]
    assert truncated[0]["dropped"] == ["three"]
    excluded = [e for e in engine.journal.entries() if e["kind"] == "sweep-selection-excluded"]
    assert excluded[0]["dw_ids"] == ["DW-1"]


def test_excluded_entry_still_gates_story_launch(project):
    project.deferred_work.write_text(
        "# Deferred Work\n\n"
        "### DW-1: selected\n\nstatus: open\n\n"
        "### DW-2: excluded gate\n\nstatus: open\ngate: 1-1\n",
        encoding="utf-8",
    )
    engine, _ = make_sweep(project, [], only_ids=("DW-1",))

    with pytest.raises(RunPaused, match="DW-2"):
        engine._refuse_gated_story("1-1-story")


def test_selector_limits_graceful_stop_remaining_estimate(project):
    write_ledger(project, {"DW-1": "open", "DW-2": "open"})
    engine, _ = make_sweep(project, [], only_ids=("DW-2",))

    assert engine._remaining_estimate() == 1


def test_selector_scope_change_declines_cached_triage(project):
    write_ledger(project, {"DW-1": "open", "DW-2": "open"})
    fresh = triage_result(
        ["DW-2"],
        skip=[{"id": "DW-2", "reason": "fresh selector universe"}],
    )
    engine, adapter = make_sweep(
        project,
        [triage_effect(fresh)],
        min_severity="high",
    )
    engine.run_dir.mkdir(parents=True, exist_ok=True)
    cached = triage_result(
        ["DW-1"],
        skip=[{"id": "DW-1", "reason": "stale selector universe"}],
    )
    (engine.run_dir / "triage.json").write_text(json.dumps(cached), encoding="utf-8")

    plan = engine._ensure_triage({"DW-2"})

    assert plan.open_ids == frozenset({"DW-2"})
    assert len(adapter.sessions) == 1
    assert "cached selected open_ids no longer match" in journal_text(engine)


def test_selector_scope_change_gets_a_fresh_triage_retry_budget(project):
    write_ledger(project, {"DW-1": "open", "DW-2": "open"})
    stale = triage_result(["DW-1"], skip=[{"id": "DW-1", "reason": "stale"}])
    fresh = triage_result(["DW-2"], skip=[{"id": "DW-2", "reason": "fresh"}])
    engine, adapter = make_sweep(
        project,
        [triage_effect(stale), triage_effect(fresh)],
        min_severity="high",
    )
    engine.run_dir.mkdir(parents=True, exist_ok=True)
    (engine.run_dir / "triage.json").write_text(json.dumps(stale), encoding="utf-8")
    task = StoryTask(story_key="sweep-triage", epic=0)
    task.phase = Phase.DONE
    task.attempt = engine.policy.sweep.max_triage_attempts
    engine.state.tasks[task.story_key] = task

    plan = engine._ensure_triage({"DW-2"})

    assert plan.open_ids == frozenset({"DW-2"})
    assert len(adapter.sessions) == 2
    assert engine.state.tasks[task.story_key].generation == 1


def test_sweep_worktree_bundle_merges_to_target(project):
    """A sweep bundle runs in its own worktree and merges back: the ledger
    closes land on the target branch and the worktree is cleaned up."""
    from conftest import _spec_baseline, write_spec

    from bmad_loop.verify import branch_exists, rev_parse_head, worktree_list

    write_ledger(project, {"DW-1": "open"})  # committed → visible in the worktree
    plan = triage_result(
        ["DW-1"],
        bundles=[{"name": "fix", "dw_ids": ["DW-1"], "intent": "fix it"}],
    )

    def wt_bundle_dev(spec):
        cwd = spec.cwd
        wt = project.rebased(cwd)
        baseline = rev_parse_head(cwd)
        src = cwd / "src.txt"
        src.write_text(src.read_text() + "change for dw-fix\n")
        sp = wt.implementation_artifacts / "spec-dw-fix.md"
        # mirror bmad-dev-auto: self-finalize the bundle spec to done, leave the
        # ledger to the orchestrator (single writer, marks inside the worktree)
        write_spec(sp, "done", baseline)
        return SessionResult(
            status="completed",
            result_json={
                "workflow": "auto-dev",
                "story_key": "dw-fix",
                "spec_file": str(sp),
                "baseline_commit": baseline,
                "tasks_total": 1,
                "tasks_done": 1,
                "verification": [],
                "escalations": [],
                "dw_ids": ["DW-1"],
            },
        )

    def wt_bundle_review(spec):
        wt = project.rebased(spec.cwd)
        sp = wt.implementation_artifacts / "spec-dw-fix.md"
        write_spec(sp, "done", _spec_baseline(sp))
        return SessionResult(
            status="completed",
            result_json={
                "workflow": "code-review",
                "clean": True,
                "patched": 0,
                "deferred": 0,
                "dismissed": 0,
                "escalations": [],
            },
        )

    pol = Policy(gates=GatesPolicy(mode="none"), notify=QUIET, scm=ScmPolicy(isolation="worktree"))
    engine, _ = make_sweep(
        project, [triage_effect(plan), wt_bundle_dev, wt_bundle_review], policy=pol
    )
    summary = engine.run()

    assert not summary.paused
    assert engine.state.tasks["dw-fix"].phase == Phase.DONE
    # the ledger close landed on the target branch (main, in the main repo)
    assert ledger_entries(project)["DW-1"].status.startswith("done")
    assert "change for dw-fix" in (project.project / "src.txt").read_text()
    # worktree cleaned up, branch deleted
    assert [p.resolve() for p in worktree_list(project.project)] == [project.project.resolve()]
    assert not branch_exists(project.project, "bmad_loop/sweep-run/dw-fix")
    assert worktree_clean(project.project)


# ------------------------------------- isolated bundle ledger carry + replay
#
# The shape every test below shares: the ledger is GITIGNORED and named in
# `scm.worktree_seed`, so the unit worktree carries a copy the orchestrator
# closes, `finalize_commit`'s `git add -A` skips it, and the merge brings
# nothing back. Without `SweepEngine._carry_isolated_ledger_writes` the main
# checkout's entries stay `open` and every later sweep re-triages resolved work.

_BUNDLE_HARVEST = {
    "summary": "Bundle left the retry ceiling unbounded",
    "evidence": "the backoff still doubles forever",
    "location": "src/retry.py:88",
    "severity": "medium",
}


def journal_kinds(engine):
    return [entry["kind"] for entry in engine.journal.entries()]


def ledger_rel(project) -> str:
    return project.deferred_work.relative_to(project.project).as_posix()


def ignored_ledger(project, statuses: dict[str, str]) -> str:
    """Gitignore the ledger, then write and commit it in that order.

    Order matters both ways: committing the rule first leaves `write_ledger`'s
    `git add -A` with nothing to stage (empty-index commit failure), and writing
    the ledger first would track it, which is the shape that needs no carry at
    all. `check-ignore` is the oracle — a pattern that is present is not
    necessarily effective."""
    ignore_before_commit(project, "deferred-work.md")
    write_ledger(project, statuses)
    rel = ledger_rel(project)
    assert git(project.project, "check-ignore", rel).strip() == rel
    assert not verify.path_tracked(project.project, rel)
    return rel


def isolated_seeded_policy(project, **scm) -> Policy:
    return Policy(
        gates=GatesPolicy(mode="none"),
        notify=QUIET,
        scm=ScmPolicy(isolation="worktree", worktree_seed=(ledger_rel(project),), **scm),
    )


def wt_bundle_dev(project, name="fix", dw_ids=("DW-1",), deferred=None):
    """Bundle dev session running inside the unit worktree (spec.cwd)."""

    def effect(spec):
        cwd = spec.cwd
        wt = project.rebased(cwd)
        baseline = verify.rev_parse_head(cwd)
        src = cwd / "src.txt"
        src.write_text(src.read_text() + f"change for dw-{name}\n")
        sp = wt.implementation_artifacts / f"spec-dw-{name}.md"
        # A real dev session creates its artifacts dir. A worktree that seeds
        # nothing into it has none — git does not track the template's empty one —
        # so without this the session dies on the spec write and no test of the
        # unseeded shape could reach the gate it is about.
        sp.parent.mkdir(parents=True, exist_ok=True)
        # mirror bmad-dev-auto: self-finalize the spec, leave the ledger to the
        # orchestrator (single writer, marking inside the worktree)
        write_spec(sp, "done", baseline, deferred=deferred)
        return SessionResult(
            status="completed",
            result_json={
                "workflow": "auto-dev",
                "story_key": f"dw-{name}",
                "spec_file": str(sp),
                "baseline_commit": baseline,
                "tasks_total": 1,
                "tasks_done": 1,
                "verification": [],
                "escalations": [],
                "dw_ids": list(dw_ids),
                "followup_review_recommended": False,
            },
        )

    return effect


def wt_bundle_review(project, name="fix"):
    """Follow-up review effect that resolves the accepted spec from ``spec.cwd``."""

    def effect(spec):
        wt = project.rebased(spec.cwd)
        sp = wt.implementation_artifacts / f"spec-dw-{name}.md"
        baseline = _spec_baseline(sp)
        write_spec(sp, "done", baseline)
        return SessionResult(
            status="completed",
            result_json={
                "workflow": "auto-dev",
                "story_key": f"dw-{name}",
                "spec_file": str(sp),
                "baseline_commit": baseline,
                "status": "done",
                "followup_review_recommended": False,
                "escalations": [],
            },
        )

    return effect


def bundle_plan(dw_ids=("DW-1",), name="fix"):
    return triage_result(
        list(dw_ids),
        bundles=[{"name": name, "dw_ids": list(dw_ids), "intent": "fix it"}],
    )


def isolated_policy(**scm) -> Policy:
    """`isolated_seeded_policy` without the seed — the SHIPPED default, since
    `scm.worktree_seed` defaults to `()`."""
    return Policy(
        gates=GatesPolicy(mode="none"),
        notify=QUIET,
        scm=ScmPolicy(isolation="worktree", **scm),
    )


def test_isolated_bundle_lands_when_the_gitignored_ledger_was_never_seeded(project):
    """#426, the default configuration: `worktree_seed` defaults to `()` and the
    ledger's home is the artifacts dir, so a project that gitignores it gets a unit
    worktree with NO ledger. The orchestrator writes the close through
    `self.workspace.paths` — that absent file — `mark_done` returns False, and
    `verify_review_bundle` reads the same absent file and fails `fixable=True`.
    Without the auto-seed that leaves `phase=deferred`, `DW-1` still `open`, a
    `review-verify-failed` naming the worktree's path, and `open_ids` re-bundling
    the same work on every later sweep.

    No DONE-leg carry can rescue that — the unit never reaches DONE — so the fix
    is upstream of the carry: seed the ledger the checkout cannot deliver, which
    moves the failure onto the leg `_carry_isolated_ledger_writes` already covers.

    The extra dev effects are the repair round the gate's `fixable=True` buys.
    They go unused on the fixed path; without them a regression would exhaust the
    script and fail as a crash rather than as the defer it actually is."""
    ignored_ledger(project, {"DW-1": "open", "DW-2": "open"})
    plan = triage_result(
        ["DW-1", "DW-2"],
        bundles=[{"name": "fix", "dw_ids": ["DW-1"], "intent": "fix it"}],
        skip=[{"id": "DW-2", "reason": "moot"}],
    )
    engine, _ = make_sweep(
        project,
        [triage_effect(plan)] + [wt_bundle_dev(project)] * 6,
        policy=isolated_policy(),
    )

    summary = engine.run()

    assert not summary.paused and not summary.crashed and summary.deferred == 0
    task = engine.state.tasks["dw-fix"]
    assert task.phase == Phase.DONE and task.isolated_ledger_carried
    entries = ledger_entries(project)
    assert entries["DW-1"].status.startswith("done")
    assert entries["DW-2"].open
    assert "review-verify-failed" not in journal_kinds(engine)
    # The seeded copy is shielded from the unit's `git add -A` like every other
    # seed, so the close still does not ride the merge: the carry is what lands
    # it, and `git add -- <ignored path>` still refuses with rc 1.
    carried = [e for e in engine.journal.entries() if e["kind"] == "sweep-bundle-close-carried"]
    assert [e["dw_ids"] for e in carried] == [["DW-1"]]
    uncommitted = [
        e for e in engine.journal.entries() if e["kind"] == "sweep-bundle-close-carry-uncommitted"
    ]
    assert len(uncommitted) == 1 and uncommitted[0]["dw_ids"] == ["DW-1"]
    assert worktree_clean(project.project)


def test_isolated_bundle_with_a_tracked_ledger_seeds_nothing(project):
    """A tracked ledger is delivered by the checkout, so auto-seeding it would
    report `worktree-seed-skipped` — "a seed you asked for did nothing" — on every
    isolated unit of every ordinary project, burying the real ones."""
    write_ledger(project, {"DW-1": "open"})
    assert verify.path_tracked(project.project, ledger_rel(project))
    engine, _ = make_sweep(
        project,
        [triage_effect(bundle_plan(["DW-1"]))] + [wt_bundle_dev(project)] * 6,
        policy=isolated_policy(),
    )

    summary = engine.run()

    assert not summary.paused and not summary.crashed
    assert engine.state.tasks["dw-fix"].phase == Phase.DONE
    assert ledger_entries(project)["DW-1"].status.startswith("done")
    assert "worktree-seed-skipped" not in journal_kinds(engine)


def test_isolated_bundle_carries_its_gitignored_ledger_close_to_the_main_checkout(project):
    ignored_ledger(project, {"DW-1": "open", "DW-2": "open"})
    plan = triage_result(
        ["DW-1", "DW-2"],
        bundles=[{"name": "fix", "dw_ids": ["DW-1"], "intent": "fix it"}],
        skip=[{"id": "DW-2", "reason": "moot"}],
    )
    engine, _ = make_sweep(
        project,
        [triage_effect(plan), wt_bundle_dev(project)],
        policy=isolated_seeded_policy(project),
    )

    summary = engine.run()

    task = engine.state.tasks["dw-fix"]
    assert not summary.paused and not summary.crashed
    assert task.phase == Phase.DONE and task.isolated_ledger_carried
    assert task.bundle_closes_intended == ["DW-1"]
    entries = ledger_entries(project)
    assert entries["DW-1"].status.startswith("done")
    assert entries["DW-2"].open  # an untouched id is never swept up by the carry
    carried = [e for e in engine.journal.entries() if e["kind"] == "sweep-bundle-close-carried"]
    assert [e["dw_ids"] for e in carried] == [["DW-1"]]
    # `git add -- <ignored path>` refuses with rc 1, so the flips land but the
    # commit cannot: best effort, and recorded rather than raised.
    uncommitted = [
        e for e in engine.journal.entries() if e["kind"] == "sweep-bundle-close-carry-uncommitted"
    ]
    assert len(uncommitted) == 1 and uncommitted[0]["dw_ids"] == ["DW-1"]
    assert [p.resolve() for p in verify.worktree_list(project.project)] == [
        project.project.resolve()
    ]
    assert worktree_clean(project.project)


def test_isolated_bundle_carry_never_touches_the_sprint_board(project):
    """The board carry rides the shared dispatcher, so a sweep runs it too — and it
    must return before doing anything (#350).

    `SweepEngine._post_dev_state_sync` is a no-op, so `board_advance_intended` is
    never set and the record IS the guard: no `_isolated`, `_generic_dev` or
    run-type predicate is added anywhere for this. The journal is the oracle rather
    than the board's content, which for a bundle has no row to move in the first
    place and so would agree with a carry that ran on nothing.
    """
    ignored_ledger(project, {"DW-1": "open"})
    engine, _ = make_sweep(
        project,
        [triage_effect(bundle_plan(["DW-1"])), wt_bundle_dev(project)],
        policy=isolated_seeded_policy(project),
    )

    summary = engine.run()

    task = engine.state.tasks["dw-fix"]
    assert not summary.paused and not summary.crashed
    assert task.phase == Phase.DONE and task.isolated_ledger_carried
    assert task.board_advance_intended is None
    kinds = journal_kinds(engine)
    assert "sweep-bundle-close-carried" in kinds  # the dispatcher DID run
    assert "board-advance-carried" not in kinds
    assert "board-advance-carry-uncommitted" not in kinds


def test_isolated_bundle_carry_files_the_harvest_before_closing_the_bundle(project):
    """`super()` first: `append_entry`'s idempotence scan is open-only, so a close
    applied ahead of the harvest would hide an already-filed row from it.

    Pinned on the observable sequence rather than on a reproduced duplicate:
    `_carry_harvested_deferrals` re-scans status-agnostically, which defends the
    same hazard a second time, so reversing the two halves is silent on the ledger
    today. A reviewer reasoning from "no duplicate appears" would ship the bug."""
    ignored_ledger(project, {"DW-1": "open"})
    engine, _ = make_sweep(
        project,
        [
            triage_effect(bundle_plan(["DW-1"])),
            wt_bundle_dev(project, deferred=[_BUNDLE_HARVEST]),
        ],
        policy=isolated_seeded_policy(project),
    )

    summary = engine.run()

    assert not summary.paused and not summary.crashed
    kinds = journal_kinds(engine)
    assert kinds.index("harvest-carried") < kinds.index("sweep-bundle-close-carried")
    entries = ledger_entries(project)
    assert entries["DW-1"].status.startswith("done")
    # the harvest reached the MAIN checkout too, exactly once
    titles = [e.title for e in deferredwork.parse_ledger(project.deferred_work.read_text())]
    assert titles.count(_BUNDLE_HARVEST["summary"]) == 1


def test_isolated_bundle_close_replays_after_a_crash_between_the_carry_and_its_latch(project):
    """`isolated_ledger_carried` is latched by the CALL SITE, never by the base
    hook. A hook-side latch would be durable the moment the base half returned, so
    a subclass half that had not run yet would look carried and never replay."""
    ignored_ledger(project, {"DW-1": "open"})
    engine, _ = make_sweep(
        project,
        [triage_effect(bundle_plan(["DW-1"])), wt_bundle_dev(project)],
        policy=isolated_seeded_policy(project),
    )
    crash_at_merge_back(engine, after="carry")

    assert engine.run().crashed

    crashed = load_state(engine.run_dir).tasks["dw-fix"]
    assert crashed.phase == Phase.DONE
    assert not crashed.isolated_ledger_carried
    assert crashed.bundle_closes_intended == ["DW-1"]

    resumed, adapter = resume_sweep(project, engine, [])
    summary = resumed.run()

    assert not summary.crashed and not summary.paused
    assert adapter.sessions == []  # no re-triage: the close left the open set first
    assert "resume-ledger-carry" in journal_kinds(resumed)
    assert ledger_entries(project)["DW-1"].status.startswith("done")
    assert load_state(resumed.run_dir).tasks["dw-fix"].isolated_ledger_carried
    # the replay re-ran an already-applied carry: idempotent, no second row
    assert len(ledger_entries(project)) == 1


def test_resumed_sweep_replays_a_close_only_carry_before_it_re_triages(project):
    """A bundle whose only ledger payload is closures still replays.

    Two things fail together if either half regresses: gating the replay on
    `harvested_deferrals` alone strands the close, and running the replay after
    `_loop` lets `deferredwork.open_ids` re-bundle an id that already landed."""
    ignored_ledger(project, {"DW-1": "open"})
    engine, _ = make_sweep(
        project,
        [triage_effect(bundle_plan(["DW-1"])), wt_bundle_dev(project)],
        policy=isolated_seeded_policy(project),
    )
    crash_at_merge_back(engine, after="merge")

    assert engine.run().crashed

    crashed = load_state(engine.run_dir).tasks["dw-fix"]
    assert crashed.phase == Phase.DONE and not crashed.isolated_ledger_carried
    assert not crashed.harvested_deferrals  # close-only payload
    assert crashed.bundle_closes_intended == ["DW-1"]
    assert ledger_entries(project)["DW-1"].open  # the merge brought nothing over

    resumed, adapter = resume_sweep(project, engine, [])
    summary = resumed.run()

    assert not summary.crashed and not summary.paused
    assert adapter.sessions == []
    kinds = journal_kinds(resumed)
    assert "resume-ledger-carry" in kinds and "sweep-bundle-close-carried" in kinds
    assert ledger_entries(project)["DW-1"].status.startswith("done")
    assert load_state(resumed.run_dir).tasks["dw-fix"].isolated_ledger_carried


def test_replayed_bundle_close_leaves_the_open_set_before_the_sweep_loop_reads_it(project):
    """The replay pre-pass sits in `run()`, above `_loop`, and must stay there.

    `SweepEngine` replaces `_loop` wholesale and does not override `run()`, and
    `_loop` reads `deferredwork.open_ids` at the head of every cycle — so a close
    replayed after it re-enters triage as open work. Pinned on the invariant (the
    ledger state `_loop` is handed) rather than on a reproduced re-triage: a
    resumed sweep finishes its persisted cycle's bundles and returns without
    opening a fresh triage, so the consequence is latent on main today. Moving the
    call below `_loop` reddens this and nothing else in the suite."""
    ignored_ledger(project, {"DW-1": "open", "DW-2": "open"})
    plan = triage_result(
        ["DW-1", "DW-2"],
        bundles=[
            {"name": "one", "dw_ids": ["DW-1"], "intent": "fix one"},
            {"name": "two", "dw_ids": ["DW-2"], "intent": "fix two"},
        ],
    )
    engine, _ = make_sweep(
        project,
        [triage_effect(plan), wt_bundle_dev(project, name="one", dw_ids=["DW-1"])],
        policy=isolated_seeded_policy(project),
    )
    crash_at_merge_back(engine, after="merge")
    assert engine.run().crashed
    assert ledger_entries(project)["DW-1"].open  # the close never rode the merge

    resumed, _ = resume_sweep(
        project, engine, [wt_bundle_dev(project, name="two", dw_ids=["DW-2"])]
    )
    at_loop_entry: list[list[str]] = []
    real_loop = resumed._loop

    def recording_loop() -> None:
        text = project.deferred_work.read_text(encoding="utf-8")
        at_loop_entry.append(sorted(deferredwork.open_ids(text)))
        real_loop()

    resumed._loop = recording_loop
    summary = resumed.run()

    assert not summary.crashed and not summary.paused
    assert at_loop_entry == [["DW-2"]], "the landed bundle's id must already be closed"
    assert ledger_entries(project)["DW-1"].status.startswith("done")


def test_bundle_close_carry_is_a_no_op_when_no_close_was_recorded(project):
    """`bundle_closes_intended` IS the guard, and it is keyed on the record rather
    than on `task.dw_ids`: only `_close_bundle_ledger_when_spec_status` writes it,
    so an empty record means this bundle closed nothing and the carry must leave no
    trace — including no journal line claiming a carry happened, which would read as
    "the carry ran" when diagnosing a run."""
    write_ledger(project, {"DW-1": "open"})
    before = project.deferred_work.read_text(encoding="utf-8")
    engine, _ = make_sweep(project, [], policy=isolated_seeded_policy(project))
    task = StoryTask(story_key="dw-fix", epic=0)
    task.dw_ids = ["DW-1"]  # ids the bundle owns, but no close was ever recorded

    engine._carry_isolated_ledger_writes(task)

    assert project.deferred_work.read_text(encoding="utf-8") == before
    assert "sweep-bundle-close-carried" not in journal_kinds(engine)


def test_deferred_bundle_replay_carries_the_harvest_without_closing_its_ids(project):
    """The DEFERRED replay leg mirrors `_defer`: harvest only, never the hook.

    A defer discarded the code the close claims to have resolved, and `open_ids`
    re-bundles only `open` entries — so a close carried here would hide real work
    from every later sweep."""
    write_ledger(project, {"DW-1": "open"})
    engine, _ = make_sweep(project, [], policy=isolated_seeded_policy(project))
    task = StoryTask(story_key="dw-fix", epic=0)
    task.phase = Phase.DEFERRED
    task.worktree_path = str(project.project / ".bmad-loop" / "worktrees" / "dw-fix")
    task.dw_ids = ["DW-1"]
    task.bundle_closes_intended = ["DW-1"]
    task.harvest_carry_commit_pending = True
    task.harvested_deferrals = [
        {
            "origin": "spec-deferred abc123",
            "title": _BUNDLE_HARVEST["summary"],
            "reason": _BUNDLE_HARVEST["evidence"],
            "location": _BUNDLE_HARVEST["location"],
            "severity": _BUNDLE_HARVEST["severity"],
            "source_spec": "spec-dw-fix.md",
        }
    ]
    engine.state.tasks["dw-fix"] = task

    engine._replay_unlatched_ledger_carries()

    entries = ledger_entries(project)
    assert entries["DW-1"].open, "a discarded bundle's close must not be carried"
    assert [e.title for e in entries.values()] == ["item DW-1", _BUNDLE_HARVEST["summary"]]
    kinds = journal_kinds(engine)
    assert "resume-ledger-carry" in kinds and "harvest-carried" in kinds
    assert "sweep-bundle-close-carried" not in kinds


def test_sweep_happy_path(project):
    write_ledger(project, {"DW-1": "open", "DW-2": "open", "DW-3": "open"})
    plan = triage_result(
        ["DW-1", "DW-2", "DW-3"],
        already_resolved=[{"id": "DW-1", "evidence": "already guarded at src.txt:1"}],
        bundles=[{"name": "fix-things", "dw_ids": ["DW-2", "DW-3"], "intent": "fix both"}],
    )
    engine, adapter = make_sweep(
        project,
        [
            triage_effect(plan),
            bundle_dev_effect(project, "fix-things", ["DW-2", "DW-3"]),
            bundle_review_effect(project, "fix-things"),
        ],
    )
    summary = engine.run()

    assert not summary.paused
    tasks = engine.state.tasks
    assert tasks["sweep-triage"].phase == Phase.DONE
    assert tasks["dw-fix-things"].phase == Phase.DONE
    assert tasks["dw-fix-things"].commit_sha

    entries = ledger_entries(project)
    assert entries["DW-1"].status.startswith("done")
    assert "already resolved: already guarded" in entries["DW-1"].body
    assert entries["DW-2"].status.startswith("done")
    assert entries["DW-3"].status.startswith("done")
    assert worktree_clean(project.project)

    log = git(project.project, "log", "--oneline")
    assert "chore(sweep): close resolved deferred-work entries" in log
    assert "sweep dw-fix-things: DW-2, DW-3 via bmad-loop" in log

    # dev session was invoked in bundle mode with the rendered intent file
    dev_spec = adapter.sessions[1]
    assert "Implement the deferred-work bundle" in dev_spec.prompt
    intent_path = re.findall(r"`([^`]*)`", dev_spec.prompt)[0]
    intent = open(intent_path).read()
    assert "fix both" in intent and "DW-2" in intent and "### DW-3" in intent


def test_write_intent_neutralizes_surrogates_and_keeps_the_markdown(project):
    """intent.md is written with `atomic_write_text` too, so a surrogate carried
    by the triage session's authored prose crashes the strict encode exactly as
    it does in the ledger — same revival chain, different file. What differs is
    the remedy: line breaks are LEGITIMATE markdown here, so `_one_line`'s
    collapse would be damage and only the surrogates are neutralized."""
    engine, _ = make_sweep(project, [])
    bundle = Bundle(
        name="fix-things",
        dw_ids=(),
        intent="keep\ud800this\n\nsecond para",
        decision_note="note\ud800",
    )

    path = engine._write_intent(bundle, "fix-things")

    assert path.is_file()
    text = path.read_text(encoding="utf-8")  # strict read; the write did not raise
    assert "note�" in text
    # the surrogate is gone AND the paragraph break either side of it is verbatim
    assert "keep�this\n\nsecond para" in text


def test_sweep_is_exempt_from_the_dispatch_hard_gate(project):
    """The sweep must never be gated by the ledger it exists to drain.

    `Engine._refuse_gated_story` refuses a picked story named by an unlanded
    `gate:` entry. `SweepEngine` overrides `_loop` and so never reaches that call —
    exemption by omission, which is exactly the kind of thing a later refactor
    "unifies" away. Gating the sweep would deadlock the gate against its own
    remedy: closing DW-1 is what the pause tells the operator to run a sweep for,
    and here DW-1 gates the sweep's own unit keys.

    Written behaviorally rather than as "the method was not called" so it also
    fails if the refusal arrives by some other route.
    """
    paths = project
    paths.deferred_work.write_text(
        "# Deferred Work\n\n"
        "### DW-1: item DW-1\n\norigin: test, 2026-06-01\nlocation: src.txt:1\n"
        "reason: test entry.\nstatus: open\ngate: sweep-triage, dw-fix-things\n",
        encoding="utf-8",
    )
    git(paths.project, "add", "-A")
    git(paths.project, "commit", "-q", "-m", "ledger")
    plan = triage_result(
        ["DW-1"],
        bundles=[{"name": "fix-things", "dw_ids": ["DW-1"], "intent": "fix it"}],
    )
    engine, _ = make_sweep(
        project,
        [
            triage_effect(plan),
            bundle_dev_effect(project, "fix-things", ["DW-1"]),
            bundle_review_effect(project, "fix-things"),
        ],
    )

    summary = engine.run()

    assert not summary.paused, "the sweep must not be gated by the ledger it drains"
    assert engine.state.tasks["sweep-triage"].phase == Phase.DONE
    assert engine.state.tasks["dw-fix-things"].phase == Phase.DONE
    # and the gating entry is closed — the remedy the story-gate pause points at
    assert ledger_entries(project)["DW-1"].status.startswith("done")


def test_generic_skill_bundle_orchestrator_closes_ledger(project):
    """B4: on the generic bmad-dev-auto path the bundle session never edits the
    ledger; the orchestrator marks each owned dw id done only after the dev attempt
    is accepted, and verify_review_bundle confirms its own write. The invocation is
    freeform."""
    write_ledger(project, {"DW-1": "open", "DW-2": "open"})
    plan = triage_result(
        ["DW-1", "DW-2"],
        bundles=[{"name": "fix-things", "dw_ids": ["DW-1", "DW-2"], "intent": "fix both"}],
    )
    pol = Policy(
        gates=GatesPolicy(mode="none"),
        notify=QUIET,
        review=ReviewPolicy(enabled=False),
        dev=DevPolicy(skill="bmad-dev-auto"),
        scm=ScmPolicy(rollback_on_failure=True),
    )
    engine, adapter = make_sweep(
        project,
        # mark_ledger=False: the decoupled skill does NOT touch the ledger
        [
            triage_effect(plan),
            bundle_dev_effect(project, "fix-things", ["DW-1", "DW-2"], mark_ledger=False),
        ],
        policy=pol,
    )
    summary = engine.run()

    assert not summary.paused
    assert engine.state.tasks["dw-fix-things"].phase == Phase.DONE
    entries = ledger_entries(project)
    assert entries["DW-1"].status.startswith("done")
    assert entries["DW-2"].status.startswith("done")
    assert "resolved by sweep bundle dw-fix-things" in entries["DW-1"].body
    # freeform generic invocation pointing at the rendered intent.md, no --dw-bundle flag
    dev_prompt = adapter.sessions[1].prompt
    assert dev_prompt.startswith("/bmad-dev-auto Implement the deferred-work bundle")
    assert "--dw-bundle" not in dev_prompt
    events = engine.journal.entries()
    close_at = next(i for i, e in enumerate(events) if e["kind"] == "sweep-bundle-closed")
    decision_at = next(
        i for i, e in enumerate(events) if e["kind"] == "dev-decision" and e["action"] == "proceed"
    )
    assert decision_at < close_at
    # Review-disabled still calls `_verify_review`, so this flow invokes the close
    # twice. The second call marks nothing and must not erase the durable carry
    # receipt recorded from task.dw_ids.
    assert "sweep-bundle-reclosed" not in {e["kind"] for e in events}
    saved = load_state(engine.run_dir).tasks["dw-fix-things"]
    assert saved.bundle_closes_intended == ["DW-1", "DW-2"]


def test_resume_dev_verify_bundle_replays_accepted_sync_before_review(project, monkeypatch):
    """A persisted PROCEED decision must finish accepted bookkeeping before review."""
    write_ledger(project, {"DW-1": "open"})
    plan = triage_result(
        ["DW-1"], bundles=[{"name": "fix", "dw_ids": ["DW-1"], "intent": "resolve DW-1"}]
    )
    pol = Policy(
        gates=GatesPolicy(mode="none"),
        notify=QUIET,
        review=ReviewPolicy(enabled=True, trigger="always"),
        dev=DevPolicy(skill="bmad-dev-auto"),
        scm=ScmPolicy(rollback_on_failure=True),
    )
    engine, _ = make_sweep(
        project,
        [
            triage_effect(plan),
            bundle_dev_effect(project, "fix", ["DW-1"], mark_ledger=False),
        ],
        policy=pol,
    )

    def crash_before_accepted_sync(task, result_json):
        persisted = load_state(engine.run_dir).tasks[task.story_key]
        assert persisted.accepted_dev_session_index == len(task.sessions) - 1
        raise RuntimeError("host died before accepted sync")

    monkeypatch.setattr(engine, "_post_dev_accepted_sync", crash_before_accepted_sync)
    assert engine.run().crashed

    crashed = load_state(engine.run_dir).tasks["dw-fix"]
    assert crashed.phase == Phase.DEV_VERIFY and crashed.spec_file
    assert crashed.accepted_dev_session_index == len(crashed.sessions) - 1
    assert crashed.pre_harvest_ledger_captured is True
    assert ledger_entries(project)["DW-1"].open

    seen_at_review = []
    review = bundle_review_effect(project, "fix")

    def review_after_sync(spec):
        seen_at_review.append(ledger_entries(project)["DW-1"].status)
        return review(spec)

    resumed, adapter = resume_sweep(project, engine, [review_after_sync])
    summary = resumed.run()

    assert not summary.crashed and not summary.paused
    assert [s.role for s in adapter.sessions] == ["review"]
    assert seen_at_review[0].startswith("done")
    saved = load_state(engine.run_dir).tasks["dw-fix"]
    assert saved.bundle_closes_intended == ["DW-1"]
    accepted = saved.accepted_dev_session_index
    assert accepted is not None and saved.sessions[accepted].role == "dev"
    assert saved.pre_harvest_ledger_captured is False


def test_isolation_flip_resumes_accepted_bundle_in_mount_and_integrates(project, monkeypatch):
    """A sweep receipt owns its recorded tree until sync, review, commit, and merge.

    Ablation: gate accepted DEV_VERIFY reopening on live isolation and review sees
    main's still-open ledger instead of the accepted close in the mounted workspace.
    """
    write_ledger(project, {"DW-1": "open"})
    plan = bundle_plan()
    isolated = Policy(
        gates=GatesPolicy(mode="none"),
        notify=QUIET,
        review=ReviewPolicy(enabled=True, trigger="always"),
        dev=DevPolicy(skill="bmad-dev-auto"),
        scm=ScmPolicy(isolation="worktree", rollback_on_failure=True),
    )
    engine, _ = make_sweep(
        project,
        [triage_effect(plan), wt_bundle_dev(project)],
        policy=isolated,
    )

    def crash_before_accepted_sync(task, result_json):
        raise RuntimeError("host died before accepted sync")

    monkeypatch.setattr(engine, "_post_dev_accepted_sync", crash_before_accepted_sync)
    assert engine.run().crashed
    crashed = load_state(engine.run_dir).tasks["dw-fix"]
    mount = Path(crashed.worktree_path)
    assert crashed.phase == Phase.DEV_VERIFY and mount.is_dir()

    engine.policy = Policy(
        gates=GatesPolicy(mode="none"),
        notify=QUIET,
        review=ReviewPolicy(enabled=True, trigger="always"),
        dev=DevPolicy(skill="bmad-dev-auto"),
        scm=ScmPolicy(isolation="none", rollback_on_failure=True),
    )
    seen: list[tuple[Path, str]] = []
    review = wt_bundle_review(project)

    def review_mounted_close(spec):
        wt = project.rebased(spec.cwd)
        seen.append((spec.cwd, ledger_entries(wt)["DW-1"].status))
        return review(spec)

    resumed, adapter = resume_sweep(project, engine, [review_mounted_close])
    summary = resumed.run()

    assert not summary.crashed and not summary.paused
    assert [session.role for session in adapter.sessions] == ["review"]
    assert seen and seen[0][0] == mount and seen[0][1].startswith("done")
    assert "change for dw-fix" in (project.project / "src.txt").read_text(encoding="utf-8")
    assert ledger_entries(project)["DW-1"].status.startswith("done")
    assert "unit-merged" in journal_kinds(resumed)
    assert "isolation-flip-orphaned-worktree" not in journal_kinds(resumed)
    assert not mount.exists()


def test_resume_dev_verify_bundle_after_repair_preserves_acceptance(project, monkeypatch):
    """A verify-green repair remains accepted across a DEV_VERIFY crash."""
    write_ledger(project, {"DW-1": "open"})
    plan = triage_result(
        ["DW-1"], bundles=[{"name": "fix", "dw_ids": ["DW-1"], "intent": "resolve DW-1"}]
    )
    marker = project.project / "verify-green"
    dev = bundle_dev_effect(project, "fix", ["DW-1"], mark_ledger=False)
    review = bundle_review_effect(project, "fix")

    def dev_with_marker(spec):
        result = dev(spec)
        marker.write_text("ok\n", encoding="utf-8")
        return result

    def breaking_review(spec):
        marker.unlink()
        return review(spec)

    def repair(spec):
        marker.write_text("repaired\n", encoding="utf-8")
        return SessionResult(
            status="completed", result_json={"workflow": "auto-dev", "escalations": []}
        )

    pol = Policy(
        gates=GatesPolicy(mode="none"),
        notify=QUIET,
        review=ReviewPolicy(enabled=True, trigger="always"),
        dev=DevPolicy(skill="bmad-dev-auto"),
        scm=ScmPolicy(rollback_on_failure=True),
        verify=VerifyPolicy(commands=(_file_exists_cmd(marker),)),
    )
    engine, _ = make_sweep(
        project,
        [triage_effect(plan), dev_with_marker, breaking_review, repair],
        policy=pol,
    )
    original_save = engine._save
    crashed = False

    def crash_after_repair_dev_verify_save():
        nonlocal crashed
        original_save()
        task = engine.state.tasks.get("dw-fix")
        if (
            not crashed
            and task is not None
            and task.phase == Phase.DEV_VERIFY
            and task.attempt == 2
            and task.sessions[-1].role == "dev"
        ):
            crashed = True
            raise RuntimeError("host died after repair DEV_VERIFY save")

    monkeypatch.setattr(engine, "_save", crash_after_repair_dev_verify_save)
    assert engine.run().crashed

    persisted = load_state(engine.run_dir).tasks["dw-fix"]
    assert persisted.phase == Phase.DEV_VERIFY
    assert persisted.attempt == 2 and persisted.review_cycle == 1
    assert [session.role for session in persisted.sessions] == ["dev", "review", "dev"]
    assert persisted.accepted_dev_session_index == len(persisted.sessions) - 1
    assert persisted.sessions[-1].task_id.endswith("-dev-2")
    assert ledger_entries(project)["DW-1"].status.startswith("done")
    fix_decisions = [e for e in engine.journal.entries() if e["kind"] == "fix-decision"]
    assert fix_decisions[-1]["ok"] is True

    seen_at_review = []
    final_review = bundle_review_effect(project, "fix")

    def review_repaired_tree(spec):
        seen_at_review.append(marker.read_text(encoding="utf-8"))
        return final_review(spec)

    resumed, adapter = resume_sweep(project, engine, [review_repaired_tree])
    summary = resumed.run()

    assert not summary.crashed and not summary.paused
    assert [session.role for session in adapter.sessions] == ["review"]
    assert seen_at_review == ["repaired\n"]
    saved = load_state(engine.run_dir).tasks["dw-fix"]
    assert saved.phase == Phase.DONE
    assert saved.accepted_dev_session_index == len(persisted.sessions) - 1


def test_resolved_bundle_review_repair_retains_intent_ownership(project):
    """A resolved Sweep repair never claims the accepted spec as input."""
    write_ledger(project, {"DW-1": "open"})
    plan = triage_result(
        ["DW-1"], bundles=[{"name": "fix", "dw_ids": ["DW-1"], "intent": "resolve DW-1"}]
    )
    marker = project.project / "resolved-sweep-verify"
    dev = bundle_dev_effect(project, "fix", ["DW-1"], mark_ledger=False)
    review = bundle_review_effect(project, "fix")
    observed: list[tuple[str | None, bytes | None]] = []

    def dev_with_marker(spec):
        result = dev(spec)
        marker.write_text("ok\n", encoding="utf-8")
        engine.state.tasks["dw-fix"].resolved_redrive = True
        return result

    def breaking_review(spec):
        marker.unlink()
        return review(spec)

    def repair(_spec):
        task = load_state(engine.run_dir).tasks["dw-fix"]
        observed.append((task.dispatched_spec_file, task.dispatched_spec_snapshot))
        marker.write_text("repaired\n", encoding="utf-8")
        return SessionResult(
            status="completed", result_json={"workflow": "auto-dev", "escalations": []}
        )

    policy = Policy(
        gates=GatesPolicy(mode="none"),
        notify=QUIET,
        review=ReviewPolicy(enabled=True, trigger="always"),
        dev=DevPolicy(skill="bmad-dev-auto"),
        verify=VerifyPolicy(commands=(_file_exists_cmd(marker),)),
        scm=ScmPolicy(rollback_on_failure=True),
    )
    engine, _ = make_sweep(
        project,
        [
            triage_effect(plan),
            dev_with_marker,
            breaking_review,
            repair,
            bundle_review_effect(project, "fix"),
        ],
        policy=policy,
    )

    summary = engine.run()

    assert summary.done >= 1 and not summary.crashed
    assert observed == [(None, None)]


def test_bundle_ledger_close_withheld_on_a_non_fixable_retry(project):
    """A rejected attempt must not close the ids whose code it rolls back."""
    write_ledger(project, {"DW-1": "open"})
    plan = triage_result(
        ["DW-1"], bundles=[{"name": "fix", "dw_ids": ["DW-1"], "intent": "resolve DW-1"}]
    )
    seen = []

    def capture_then_die(spec):
        seen.append(project.deferred_work.read_text(encoding="utf-8"))
        return SessionResult(status="died")

    pol = Policy(
        gates=GatesPolicy(mode="none"),
        notify=QUIET,
        review=ReviewPolicy(enabled=False),
        dev=DevPolicy(skill="bmad-dev-auto"),
        scm=ScmPolicy(rollback_on_failure=True),
        limits=LimitsPolicy(max_dev_attempts=2),
    )
    engine, _ = make_sweep(
        project,
        [
            triage_effect(plan),
            bundle_dev_effect(project, "fix", ["DW-99"], mark_ledger=False, followup_review=False),
            capture_then_die,
        ],
        policy=pol,
    )

    summary = engine.run()

    decisions = [e for e in engine.journal.entries() if e["kind"] == "dev-decision"]
    assert decisions[0]["action"] == "retry" and "do not match" in decisions[0]["reason"]
    assert "rollback-auto" in journal_text(engine)
    assert "sweep-bundle-closed" not in {e["kind"] for e in engine.journal.entries()}
    assert "status: open" in seen[0]
    assert "resolved by sweep bundle dw-fix" not in seen[0]
    assert summary.deferred == 1 and not summary.paused
    assert ledger_entries(project)["DW-1"].open


def test_bundle_ledger_close_withheld_when_a_completed_attempt_defers(project):
    """A completed but rejected final attempt reaches DEFER with the ids still open."""
    write_ledger(project, {"DW-1": "open"})
    plan = triage_result(
        ["DW-1"], bundles=[{"name": "fix", "dw_ids": ["DW-1"], "intent": "resolve DW-1"}]
    )

    def mismatching_attempt():
        return bundle_dev_effect(
            project, "fix", ["DW-99"], mark_ledger=False, followup_review=False
        )

    pol = Policy(
        gates=GatesPolicy(mode="none"),
        notify=QUIET,
        review=ReviewPolicy(enabled=False),
        dev=DevPolicy(skill="bmad-dev-auto"),
        scm=ScmPolicy(rollback_on_failure=True),
        limits=LimitsPolicy(max_dev_attempts=2),
    )
    engine, _ = make_sweep(
        project,
        [triage_effect(plan), mismatching_attempt(), mismatching_attempt()],
        policy=pol,
    )

    summary = engine.run()

    actions = [e["action"] for e in engine.journal.entries() if e["kind"] == "dev-decision"]
    assert actions == ["retry", "defer"]
    assert summary.deferred == 1 and not summary.paused
    text = project.deferred_work.read_text(encoding="utf-8")
    assert "resolved by sweep bundle dw-fix" not in text
    assert deferredwork.open_ids(text) == {"DW-1"}
    assert "change for dw-fix" not in (project.project / "src.txt").read_text()
    kinds = {e["kind"] for e in engine.journal.entries()}
    assert "sweep-bundle-closed" not in kinds
    assert "sweep-bundle-reopened" not in kinds


def test_bundle_ledger_close_withheld_when_critical_preempts_a_passing_gate(project):
    """PROCEED, not outcome.ok, authorizes the close."""
    write_ledger(project, {"DW-1": "open"})
    plan = triage_result(
        ["DW-1"], bundles=[{"name": "fix", "dw_ids": ["DW-1"], "intent": "resolve DW-1"}]
    )
    clean = bundle_dev_effect(project, "fix", ["DW-1"], mark_ledger=False, followup_review=False)

    def clean_but_critical(spec):
        result = clean(spec)
        assert result.result_json is not None
        result.result_json["escalations"] = [
            {"type": "bundle-item-blocked", "severity": "CRITICAL", "detail": "intent gap"}
        ]
        return result

    pol = Policy(
        gates=GatesPolicy(mode="none"),
        notify=QUIET,
        review=ReviewPolicy(enabled=False),
        dev=DevPolicy(skill="bmad-dev-auto"),
        scm=ScmPolicy(rollback_on_failure=True),
    )
    engine, _ = make_sweep(project, [triage_effect(plan), clean_but_critical], policy=pol)

    summary = engine.run()

    decisions = [e for e in engine.journal.entries() if e["kind"] == "dev-decision"]
    assert len(decisions) == 1
    assert decisions[0]["action"] == "pause" and "CRITICAL" in decisions[0]["reason"]
    assert summary.paused
    assert ledger_entries(project)["DW-1"].open
    assert "sweep-bundle-closed" not in {e["kind"] for e in engine.journal.entries()}


def test_bundle_ledger_close_reopened_when_the_review_leg_defers(project):
    """A later review failure undoes the accepted close after rollback."""
    write_ledger(project, {"DW-1": "open"})
    plan = triage_result(
        ["DW-1"], bundles=[{"name": "fix", "dw_ids": ["DW-1"], "intent": "resolve DW-1"}]
    )
    pol = Policy(
        gates=GatesPolicy(mode="none"),
        notify=QUIET,
        review=ReviewPolicy(enabled=True, trigger="always"),
        dev=DevPolicy(skill="bmad-dev-auto"),
        scm=ScmPolicy(rollback_on_failure=True),
        limits=LimitsPolicy(max_review_cycles=1),
    )
    engine, _ = make_sweep(
        project,
        [
            triage_effect(plan),
            bundle_dev_effect(project, "fix", ["DW-1"], mark_ledger=False),
            lambda spec: SessionResult(status="died"),
        ],
        policy=pol,
    )

    summary = engine.run()

    kinds = [e["kind"] for e in engine.journal.entries()]
    assert "sweep-bundle-closed" in kinds
    assert summary.deferred == 1 and not summary.paused
    assert "change for dw-fix" not in (project.project / "src.txt").read_text()
    text = project.deferred_work.read_text(encoding="utf-8")
    assert deferredwork.open_ids(text) == {"DW-1"}
    assert "resolved by sweep bundle dw-fix" not in text
    reopened = [e for e in engine.journal.entries() if e["kind"] == "sweep-bundle-reopened"]
    assert len(reopened) == 1 and reopened[0]["dw_ids"] == ["DW-1"]
    assert worktree_clean(project.project)


def test_bundle_ledger_close_reopened_when_review_disabled_defers_at_gate(project, tmp_path):
    """The no-review path also reopens when its post-accept verify gate defers."""
    write_ledger(project, {"DW-1": "open"})
    plan = triage_result(
        ["DW-1"], bundles=[{"name": "fix", "dw_ids": ["DW-1"], "intent": "resolve DW-1"}]
    )
    pol = Policy(
        gates=GatesPolicy(mode="none"),
        notify=QUIET,
        review=ReviewPolicy(enabled=False),
        dev=DevPolicy(skill="bmad-dev-auto"),
        scm=ScmPolicy(rollback_on_failure=True),
        verify=VerifyPolicy(commands=(passes_once(tmp_path / "verify-ran"),)),
        limits=LimitsPolicy(max_dev_attempts=1),
    )
    engine, _ = make_sweep(
        project,
        [
            triage_effect(plan),
            bundle_dev_effect(project, "fix", ["DW-1"], mark_ledger=False, followup_review=False),
        ],
        policy=pol,
    )

    summary = engine.run()

    kinds = [e["kind"] for e in engine.journal.entries()]
    assert "sweep-bundle-closed" in kinds
    assert summary.deferred == 1 and not summary.paused
    assert "change for dw-fix" not in (project.project / "src.txt").read_text()
    assert deferredwork.open_ids(project.deferred_work.read_text(encoding="utf-8")) == {"DW-1"}
    assert "sweep-bundle-reopened" in kinds


def test_bundle_pre_gate_state_sync_is_a_noop(project):
    """The sweep override must not retain the old pre-gate close."""
    write_ledger(project, {"DW-1": "open"})
    pol = Policy(
        gates=GatesPolicy(mode="none"),
        notify=QUIET,
        review=ReviewPolicy(enabled=False),
        dev=DevPolicy(skill="bmad-dev-auto"),
    )
    engine, _ = make_sweep(project, [], policy=pol)
    spec = project.implementation_artifacts / "spec-dw-fix.md"
    write_spec(spec, "done", git(project.project, "rev-parse", "HEAD"))
    task = StoryTask(story_key="dw-fix", epic=0, dw_ids=["DW-1"])

    engine._post_dev_state_sync(task, {"spec_file": str(spec)})

    assert ledger_entries(project)["DW-1"].open
    assert task.bundle_closes_intended == []
    assert "sweep-bundle-closed" not in {e["kind"] for e in engine.journal.entries()}
    # and no sprint-board intent either: a bundle has no board row, and this
    # override being a no-op is the whole of `_carry_board_advance`'s guard (#350)
    assert task.board_advance_intended is None


def test_bundle_review_gate_journals_its_verify_commands(project):
    """`SweepEngine._verify_review` threads the base engine's review sink, so a
    bundle's review-leg verifier pass lands the same `verify-command-result`
    records a story's does.

    Its own row rather than a claim carried by `test_engine.py`: the sink is
    passed at each override, so dropping it here would leave every sweep run
    silently unrecorded while the base engine's tests stayed green — which is the
    shape the #695 root bug already took across these same three gates.

    Ablation: remove `on_results=` from `SweepEngine._verify_review` and the
    record assertion fails at zero entries."""
    write_ledger(project, {"DW-1": "done 2026-06-11"})
    pol = Policy(
        gates=GatesPolicy(mode="none"),
        notify=QUIET,
        verify=VerifyPolicy(commands=(_OK,)),
    )
    engine, _ = make_sweep(project, [], policy=pol)
    spec = project.implementation_artifacts / "spec-dw-fix.md"
    spec.parent.mkdir(parents=True, exist_ok=True)
    write_spec(spec, "done", git(project.project, "rev-parse", "HEAD"))
    task = StoryTask(story_key="dw-fix", epic=0, dw_ids=["DW-1"])
    task.spec_file = str(spec)

    assert engine._verify_review(task).ok

    (entry,) = [e for e in engine.journal.entries() if e["kind"] == "verify-command-result"]
    assert entry["verification_stage"] == "review"
    assert entry["command"] == _OK and entry["story_key"] == "dw-fix"


def test_bundle_ledger_close_skips_on_unreadable_spec(project, monkeypatch):
    """The bundle counterpart of the sprint-board sync: an unreadable bundle spec
    must not close any dw id (the ledger write is a repair — it must never fire off
    an observation the orchestrator could not make) and must not crash the sweep."""
    write_ledger(project, {"DW-1": "open", "DW-2": "open"})
    pol = Policy(
        gates=GatesPolicy(mode="none"),
        notify=QUIET,
        review=ReviewPolicy(enabled=False),
        dev=DevPolicy(skill="bmad-dev-auto"),
        scm=ScmPolicy(rollback_on_failure=True),
    )
    engine, _ = make_sweep(project, [], policy=pol)
    sp = project.implementation_artifacts / "spec-dw-fix-things.md"
    sp.parent.mkdir(parents=True, exist_ok=True)
    write_spec(sp, "done", "abc123")
    task = StoryTask(story_key="dw-fix-things", epic=0, dw_ids=["DW-1", "DW-2"])
    fault_read_text(monkeypatch, sp)

    engine._post_dev_accepted_sync(task, {"spec_file": str(sp)})

    entries = ledger_entries(project)
    assert entries["DW-1"].status == "open" and entries["DW-2"].status == "open"
    kinds = [e["kind"] for e in engine.journal.entries()]
    assert "sweep-bundle-closed" not in kinds
    events = [e for e in engine.journal.entries() if e["kind"] == "spec-read-failed"]
    assert len(events) == 1 and events[0]["site"] == "bundle-ledger-close"
    assert events[0]["story_key"] == "dw-fix-things"


def test_generic_bundle_review_verify_recloses_ledger_after_review_rewrites_it(project):
    """A follow-up review can rewrite deferred-work.md from its own snapshot and
    re-open entries the orchestrator already closed after dev. The review gate
    should re-apply the orchestrator-owned ledger closure before verification,
    otherwise the sweep launches a spurious repair/dev pass for complete work."""
    write_ledger(project, {"DW-1": "open", "DW-2": "open"})
    baseline = git(project.project, "rev-parse", "HEAD")
    spec = project.implementation_artifacts / "spec-dw-fix-things.md"
    write_spec(spec, "done", baseline)
    pol = Policy(
        gates=GatesPolicy(mode="none"),
        notify=QUIET,
        review=ReviewPolicy(enabled=True, trigger="always"),
        dev=DevPolicy(skill="bmad-dev-auto"),
        scm=ScmPolicy(rollback_on_failure=True),
    )
    engine, _ = make_sweep(project, [], policy=pol)
    task = StoryTask(
        story_key="dw-fix-things",
        epic=0,
        dw_ids=["DW-1", "DW-2"],
        spec_file=str(spec),
    )

    out = engine._verify_review(task)

    assert out.ok
    entries = ledger_entries(project)
    assert entries["DW-1"].status.startswith("done")
    assert entries["DW-2"].status.startswith("done")
    assert "resolved by sweep bundle dw-fix-things" in entries["DW-1"].body
    assert "sweep-bundle-reclosed" in {e["kind"] for e in engine.journal.entries()}


def test_generic_bundle_reconcile_closes_ledger_on_stale_frontmatter(project):
    """Regression for the DW-159/160/162 false-defer: the bundle session finalized
    in prose (## Auto Run Result: Status done) but left the bundle spec frontmatter
    at the template default `draft`. The orchestrator reconciles the frontmatter
    before accepted bookkeeping, so the bundle CLOSES — its dw ids are marked done and
    not stranded in failed_ids — instead of falsely deferring completed work."""
    write_ledger(project, {"DW-1": "open", "DW-2": "open"})
    plan = triage_result(
        ["DW-1", "DW-2"],
        bundles=[{"name": "fix-things", "dw_ids": ["DW-1", "DW-2"], "intent": "fix both"}],
    )
    pol = Policy(
        gates=GatesPolicy(mode="none"),
        notify=QUIET,
        review=ReviewPolicy(enabled=False),
        dev=DevPolicy(skill="bmad-dev-auto"),
        scm=ScmPolicy(rollback_on_failure=True),
    )
    engine, _ = make_sweep(
        project,
        [
            triage_effect(plan),
            # the skill leaves frontmatter at draft but writes prose Status: done
            bundle_dev_effect(
                project,
                "fix-things",
                ["DW-1", "DW-2"],
                mark_ledger=False,
                final_status="draft",
                prose_status="done",
            ),
        ],
        policy=pol,
    )
    summary = engine.run()

    assert not summary.paused
    assert engine.state.tasks["dw-fix-things"].phase == Phase.DONE
    entries = ledger_entries(project)
    assert entries["DW-1"].status.startswith("done")
    assert entries["DW-2"].status.startswith("done")
    recon = [e for e in engine.journal.entries() if e["kind"] == "spec-status-reconciled"]
    assert len(recon) == 1
    assert recon[0]["frm"] == "draft" and recon[0]["to"] == "done"
    assert "sweep-bundle-closed" in {e["kind"] for e in engine.journal.entries()}


def test_generic_bundle_reconcile_closes_ledger_on_in_review_frontmatter(project):
    """The Lights-Out DW-153 symptom on the bundle path: the session finalized in
    prose (## Auto Run Result: Status done) but left the bundle spec frontmatter at
    the transient `in-review` marker. in-review is never a deliberate terminal on
    the generic path (the legacy review-handoff fork is retired), so the
    orchestrator reconciles it to done before accepted bookkeeping — the bundle CLOSES
    instead of false-deferring + rolling back into an endless re-sweep loop."""
    write_ledger(project, {"DW-1": "open", "DW-2": "open"})
    plan = triage_result(
        ["DW-1", "DW-2"],
        bundles=[{"name": "fix-things", "dw_ids": ["DW-1", "DW-2"], "intent": "fix both"}],
    )
    pol = Policy(
        gates=GatesPolicy(mode="none"),
        notify=QUIET,
        review=ReviewPolicy(enabled=False),
        dev=DevPolicy(skill="bmad-dev-auto"),
        scm=ScmPolicy(rollback_on_failure=True),
    )
    engine, _ = make_sweep(
        project,
        [
            triage_effect(plan),
            # the skill leaves frontmatter at the transient in-review marker but
            # writes prose Status: done
            bundle_dev_effect(
                project,
                "fix-things",
                ["DW-1", "DW-2"],
                mark_ledger=False,
                final_status="in-review",
                prose_status="done",
            ),
        ],
        policy=pol,
    )
    summary = engine.run()

    assert not summary.paused
    assert engine.state.tasks["dw-fix-things"].phase == Phase.DONE
    entries = ledger_entries(project)
    assert entries["DW-1"].status.startswith("done")
    assert entries["DW-2"].status.startswith("done")
    recon = [e for e in engine.journal.entries() if e["kind"] == "spec-status-reconciled"]
    assert len(recon) == 1
    assert recon[0]["frm"] == "in-review" and recon[0]["to"] == "done"
    assert "sweep-bundle-closed" in {e["kind"] for e in engine.journal.entries()}


def test_triage_validation_failure_retries_with_feedback_then_escalates(project):
    write_ledger(project, {"DW-1": "open"})
    bad = triage_result(["DW-1"])  # DW-1 not triaged anywhere
    engine, adapter = make_sweep(project, [triage_effect(bad), triage_effect(bad)])
    summary = engine.run()

    assert summary.paused
    assert engine.state.tasks["sweep-triage"].phase == Phase.ESCALATED
    prompts = [s.prompt for s in adapter.sessions]
    assert len(prompts) == 2
    assert "--feedback" not in prompts[0] and "--feedback" in prompts[1]
    feedback_path = prompts[1].split("--feedback ", 1)[1]
    assert "not triaged: DW-1" in open(feedback_path).read()


def test_overlong_bundle_name_is_normalized_without_triage_retry(project):
    """ABLATION A1: delete direct normalization and the first triage attempt fails."""
    write_ledger(project, {"DW-1": "open"})
    result = triage_result(
        ["DW-1"],
        bundles=[{"name": _OVERLONG_BUNDLE_NAME, "dw_ids": ["DW-1"], "intent": "fix it"}],
    )
    engine, adapter = make_sweep(
        project,
        [
            triage_effect(result),
            bundle_dev_effect(project, _NORMALIZED_BUNDLE_NAME, ["DW-1"]),
            bundle_review_effect(project, _NORMALIZED_BUNDLE_NAME),
        ],
    )

    summary = engine.run()

    assert not summary.paused
    triage_sessions = [session for session in adapter.sessions if session.role == "triage"]
    assert len(triage_sessions) == 1
    assert "--feedback" not in triage_sessions[0].prompt
    persisted = json.loads((engine.run_dir / "triage.json").read_text(encoding="utf-8"))
    assert persisted["bundles"][0]["name"] == _NORMALIZED_BUNDLE_NAME
    repairs = [
        entry
        for entry in engine.journal.entries()
        if entry["kind"] == "sweep-bundle-name-normalized"
    ]
    assert len(repairs) == 1
    assert repairs[0]["field"] == "bundles[0].name"
    assert repairs[0]["original"] == _OVERLONG_BUNDLE_NAME
    assert repairs[0]["normalized"] == _NORMALIZED_BUNDLE_NAME
    assert engine.state.tasks[f"dw-{_NORMALIZED_BUNDLE_NAME}"].phase == Phase.DONE
    intent_path = engine.run_dir / "bundles" / _NORMALIZED_BUNDLE_NAME / "intent.md"
    assert intent_path.is_file()
    assert _NORMALIZED_BUNDLE_NAME in intent_path.read_text(encoding="utf-8")
    assert not (engine.run_dir / "bundles" / _OVERLONG_BUNDLE_NAME).exists()


def test_triage_session_env_fault_escalates_then_resume_restores_budget(project):
    """A triage session whose CLI lost its API connection (#194) escalates on the
    first attempt instead of retrying up to max_triage_attempts; the decision
    carries env_fault, and the ESCALATED-resume restores the budget (attempt -> 0)
    so a resume after the outage re-drives triage cleanly."""
    write_ledger(project, {"DW-1": "open"})
    evidence = "API Error: Unable to connect (ECONNREFUSED)"
    engine, adapter = make_sweep(
        project,
        [SessionResult(status="timeout", env_fault=True, env_fault_evidence=evidence)],
    )
    summary = engine.run()

    assert summary.paused
    task = engine.state.tasks["sweep-triage"]
    assert task.phase == Phase.ESCALATED
    assert task.attempt == 1  # only the one session — no retry budget spent
    assert "environment fault: triage session timeout" in engine.state.paused_reason
    assert evidence in engine.state.paused_reason
    assert len(adapter.sessions) == 1  # no feedback-retry session
    dec = [e for e in engine.journal.entries() if e["kind"] == "triage-decision"][-1]
    assert dec["env_fault"] is True

    abandoned = [s.task_id for s in adapter.sessions]
    assert abandoned == ["sweep-triage-triage-1"]  # generation 0 emits no suffix

    # resume once the outage clears: the ESCALATED-resume resets attempt to 0
    # (fresh budget) and re-drives triage to completion
    good = triage_result(["DW-1"], skip=[{"id": "DW-1", "reason": "moot"}])
    resumed, radapter = resume_sweep(project, engine, [triage_effect(good)])
    assert not resumed.run().paused
    assert resumed.state.tasks["sweep-triage"].phase == Phase.DONE
    assert len(radapter.sessions) == 1
    # ...and it does so in a NEW generation. The attempt reset above is exactly what
    # would otherwise re-mint `attempt == 1` — an id byte-equal to the abandoned
    # attempt's, pointing the fresh record at the abandoned cycle's
    # tasks/<id>/escalation.json. Both adapters now clear cycle outputs at launch, and
    # `resolve._gather_escalations` opens each distinct task_id once, but neither makes
    # two historical records stop aliasing one mutable directory. The fresh id preserves
    # a separate artifact namespace for each cycle, independent of cleanup.
    assert resumed.state.tasks["sweep-triage"].generation == 1
    assert [s.task_id for s in radapter.sessions] == ["sweep-triage-triage-1-g1"]
    assert radapter.sessions[0].task_id not in abandoned


@pytest.mark.parametrize("role", ["triage", "migration"])
def test_spec_less_critical_session_keeps_full_reason_and_points_displays_to_journal(project, role):
    if role == "migration":
        write_legacy_ledger(project, LEGACY_LEDGER)
    else:
        write_ledger(project, {"DW-1": "open"})
    tail = "RECOVERY-TAIL"
    detail = "x" * 2500 + tail
    result = SessionResult(
        status="completed",
        result_json={
            "escalations": [{"severity": "CRITICAL", "detail": detail}],
        },
    )
    engine, _ = make_sweep(project, [result])

    summary = engine.run()

    assert engine.state.paused_reason.endswith(tail)
    assert summary.paused_reason.endswith("[… truncated; full detail in journal.jsonl]")
    assert tail not in summary.paused_reason
    (escalated,) = [
        entry for entry in engine.journal.entries() if entry["kind"] == "story-escalated"
    ]
    assert escalated["reason"] == f"CRITICAL escalation from {role} session: {detail}"


def test_repeated_triage_escalation_restarts_keep_advancing_generation(project):
    """Every ESCALATED restart opens a new namespace, not only the first one.

    Starting from generation zero alone would let ``generation += 1`` regress to
    ``generation = 1`` while every first-restart assertion stayed green. A second
    escalation proves the next reset advances to generation two and cannot re-mint
    either earlier session id.
    """
    write_ledger(project, {"DW-1": "open"})
    outage = SessionResult(
        status="timeout",
        env_fault=True,
        env_fault_evidence="API Error: Unable to connect (ECONNREFUSED)",
    )
    engine, first = make_sweep(project, [outage])
    assert engine.run().paused
    first_id = first.sessions[0].task_id

    resumed_once, second = resume_sweep(project, engine, [outage])
    assert resumed_once.run().paused
    assert resumed_once.state.tasks["sweep-triage"].generation == 1
    second_id = second.sessions[0].task_id
    assert second_id == "sweep-triage-triage-1-g1"

    good = triage_result(["DW-1"], skip=[{"id": "DW-1", "reason": "moot"}])
    resumed_twice, third = resume_sweep(project, resumed_once, [triage_effect(good)])
    assert not resumed_twice.run().paused
    assert resumed_twice.state.tasks["sweep-triage"].generation == 2
    third_id = third.sessions[0].task_id
    assert third_id == "sweep-triage-triage-1-g2"
    assert len({first_id, second_id, third_id}) == 3


def test_triage_plain_timeout_still_retries_to_cap(project):
    """Guard: a NON-env-fault triage timeout keeps today's behavior — retries up to
    max_triage_attempts (2) then escalates. The env-fault branch must not swallow
    ordinary transient failures on the first attempt."""
    write_ledger(project, {"DW-1": "open"})
    engine, adapter = make_sweep(
        project,
        [SessionResult(status="timeout"), SessionResult(status="timeout")],
    )
    summary = engine.run()

    assert summary.paused
    assert engine.state.tasks["sweep-triage"].phase == Phase.ESCALATED
    assert len(adapter.sessions) == 2  # both attempts spent, not escalated on the first
    dec = [e for e in engine.journal.entries() if e["kind"] == "triage-decision"]
    assert all(d["env_fault"] is False for d in dec)


def test_triage_lost_session_names_the_mux_in_its_errors(project):
    """#489 on the sweep triage path: a destroyed session must not be filed as a
    triage that ran and failed. The adopted reason builder is only verified on the
    dev/review paths otherwise, so reverting this site would go unnoticed."""
    write_ledger(project, {"DW-1": "open"})
    engine, adapter = make_sweep(
        project,
        [
            SessionResult(status="crashed", session_vanished=True),
            SessionResult(status="crashed", session_vanished=True),
        ],
    )
    engine.run()

    assert len(adapter.sessions) == 2  # both attempts spent: the retry actually ran
    dec = [e for e in engine.journal.entries() if e["kind"] == "triage-decision"][-1]
    assert any("triage session crashed" in e for e in dec["errors"])
    assert any("multiplexer no longer reports the session" in e for e in dec["errors"])


def test_migration_lost_session_names_the_mux_in_its_errors(project):
    """The migration twin of the triage case above — same builder, same blind spot."""
    write_legacy_ledger(project, LEGACY_LEDGER)
    engine, adapter = make_sweep(
        project,
        [
            SessionResult(status="crashed", session_vanished=True),
            SessionResult(status="crashed", session_vanished=True),
        ],
    )
    engine.run()

    assert len(adapter.sessions) == 2  # both attempts spent: the retry actually ran
    dec = [e for e in engine.journal.entries() if e["kind"] == "migrate-decision"][-1]
    assert any("migration session crashed" in e for e in dec["errors"])
    assert any("multiplexer no longer reports the session" in e for e in dec["errors"])


def test_migration_session_env_fault_escalates_without_consuming_attempts(project):
    """A migration session whose CLI lost its API connection (#194) escalates on the
    first attempt instead of charging a migration retry; migrate-decision carries
    env_fault, and the legacy ledger is left untouched (no half-rewrite lands)."""
    write_legacy_ledger(project, LEGACY_LEDGER)
    evidence = "API Error: Unable to connect (ECONNREFUSED)"
    engine, adapter = make_sweep(
        project,
        [SessionResult(status="timeout", env_fault=True, env_fault_evidence=evidence)],
    )
    summary = engine.run()

    assert summary.paused
    task = engine.state.tasks["sweep-migrate"]
    assert task.phase == Phase.ESCALATED
    assert task.attempt == 1  # only the one session — no migration retry spent
    assert "environment fault: migration session timeout" in engine.state.paused_reason
    assert evidence in engine.state.paused_reason
    assert len(adapter.sessions) == 1
    dec = [e for e in engine.journal.entries() if e["kind"] == "migrate-decision"][-1]
    assert dec["env_fault"] is True
    # the legacy ledger is intact and the tree clean: no half-rewrite escaped
    assert project.deferred_work.read_text(encoding="utf-8") == LEGACY_LEDGER
    assert worktree_clean(project.project)


def test_triage_escalation_resume_retries_triage(project):
    write_ledger(project, {"DW-1": "open"})
    bad = triage_result(["DW-1"])
    engine, _ = make_sweep(project, [triage_effect(bad), triage_effect(bad)])
    assert engine.run().paused

    good = triage_result(["DW-1"], skip=[{"id": "DW-1", "reason": "moot"}])
    resumed, adapter = resume_sweep(project, engine, [triage_effect(good)])
    summary = resumed.run()
    assert not summary.paused
    assert resumed.state.tasks["sweep-triage"].phase == Phase.DONE
    assert len(adapter.sessions) == 1


def test_non_escalated_triage_restart_keeps_its_generation(project):
    """Control for the ESCALATED-arm bump: a task restarted from a NON-escalated
    phase (the host died mid-triage) keeps its attempt counter, so `attempt += 1`
    already yields a fresh number and the namespace must not move. Bumping outside
    that arm would break the property `_session_task_id`'s suffix rule exists to
    hold — every id an existing run already wrote to disk stays byte-identical."""
    write_ledger(project, {"DW-1": "open"})
    good = triage_result(["DW-1"], skip=[{"id": "DW-1", "reason": "moot"}])
    engine, adapter = make_sweep(project, [triage_effect(good)])
    # a session that never reported: TRIAGE_RUNNING with one attempt already spent
    task = StoryTask(story_key="sweep-triage", epic=0)
    task.phase = Phase.TRIAGE_RUNNING
    task.attempt = 1
    engine.state.tasks["sweep-triage"] = task

    assert not engine.run().paused

    assert engine.state.tasks["sweep-triage"].generation == 0  # NOT bumped
    assert engine.state.tasks["sweep-triage"].attempt == 2  # the counter continued
    # attempt 2 is already a fresh id; no -g suffix rewrites the namespace
    assert [s.task_id for s in adapter.sessions] == ["sweep-triage-triage-2"]


def test_non_escalated_migrate_restart_keeps_its_generation(project):
    """The migrate twin of the row above. Both restart arms scope the bump to
    `Phase.ESCALATED` independently, so pinning only the triage one leaves
    `_ensure_migration`'s scoping free: dedenting its `_rearm_generation(task)` call
    a level passes the whole triage-side suite."""
    write_legacy_ledger(project, LEGACY_LEDGER)
    manifest = legacy_manifest()
    mapping = [
        {"key": manifest[0]["key"], "dw_id": "DW-1"},
        {"key": manifest[1]["key"], "dw_id": "DW-2"},
    ]
    plan = triage_result(["DW-2"], skip=[{"id": "DW-2", "reason": "moot"}])
    engine, adapter = make_sweep(
        project,
        [migrate_effect(project, migrated_ledger(), mapping), triage_effect(plan)],
    )
    # a migration session that never reported: TRIAGE_RUNNING, one attempt spent
    task = StoryTask(story_key="sweep-migrate", epic=0)
    task.phase = Phase.TRIAGE_RUNNING
    task.attempt = 1
    engine.state.tasks["sweep-migrate"] = task

    assert not engine.run().paused

    assert engine.state.tasks["sweep-migrate"].generation == 0  # NOT bumped
    assert engine.state.tasks["sweep-migrate"].attempt == 2  # the counter continued
    # attempt 2 is already a fresh id; no -g suffix rewrites the namespace
    assert adapter.sessions[0].task_id == "sweep-migrate-triage-2"


def test_interactive_decisions_build_and_close(project):
    write_ledger(project, {"DW-1": "open", "DW-2": "open"})
    plan = triage_result(
        ["DW-1", "DW-2"],
        decisions=[
            {
                "id": "DW-1",
                "question": "build the widening?",
                "context": "ctx",
                "options": [
                    {
                        "key": "1",
                        "label": "Widen it",
                        "effect": "build",
                        "intent": "widen the field",
                    },
                    {"key": "2", "label": "Keep as is", "effect": "keep-open"},
                ],
                "recommendation": "1",
            },
            {
                "id": "DW-2",
                "question": "close as moot?",
                "context": "",
                "options": [
                    {
                        "key": "1",
                        "label": "Close it",
                        "effect": "close",
                        "resolution": "superseded by v2",
                    },
                    {"key": "2", "label": "Keep open", "effect": "keep-open"},
                ],
                "recommendation": "1",
            },
        ],
    )
    engine, adapter = make_sweep(
        project,
        [
            triage_effect(plan),
            bundle_dev_effect(project, "decision-dw-1", ["DW-1"]),
            bundle_review_effect(project, "decision-dw-1"),
        ],
        # DW-1: invalid input, then empty (= recommendation "1" -> build);
        # DW-2: explicit "1" (close)
        answers=["9", "", "1"],
        prompting=True,
    )
    summary = engine.run()
    assert not summary.paused

    journal = journal_text(engine)
    assert journal.count('"decision-pending"') == 2  # announced before each prompt
    assert journal.index('"decision-pending"') < journal.index('"decision-answered"')
    attention = (engine.run_dir / "ATTENTION").read_text()
    assert "decision needed: DW-1" in attention

    answers = json.loads((engine.run_dir / "decisions.json").read_text())
    assert answers["DW-1"]["effect"] == "build"
    assert answers["DW-2"]["effect"] == "close"

    entries = ledger_entries(project)
    assert "decision:" in entries["DW-1"].body
    assert entries["DW-1"].status.startswith("done")  # closed by the built bundle
    assert entries["DW-2"].status.startswith("done")  # closed by the decision
    assert "closed by human decision: superseded by v2" in entries["DW-2"].body
    assert engine.state.tasks["dw-decision-dw-1"].phase == Phase.DONE
    assert "chore(sweep): record deferred-work decisions" in git(
        project.project, "log", "--oneline"
    )
    # DW-118 row 1 — the IN-RUN lane against the very triage it answered. An
    # in-run answer is written with only key/label/effect/answered_at, so the
    # agreeing option is what supplies `intent` at all; the note quotes the
    # question the human actually saw, and nothing is journaled as mismatched.
    assert _mismatches(engine) == []
    intent_md = (engine.run_dir / "bundles" / "decision-dw-1" / "intent.md").read_text(
        encoding="utf-8"
    )
    assert "widen the field" in intent_md  # only the option carries it
    assert "build the widening?" in intent_md  # the current question, quoted
    assert "against an earlier triage" not in intent_md


def _close_decision_plan():
    return triage_result(
        ["DW-1"],
        decisions=[
            {
                "id": "DW-1",
                "question": "close as moot?",
                "context": "",
                "options": [
                    {"key": "1", "label": "Close it", "effect": "close", "resolution": "moot"},
                    {"key": "2", "label": "Keep open", "effect": "keep-open"},
                ],
                "recommendation": "1",
            }
        ],
    )


def _stub_return(monkeypatch, outcome):
    """Pin return_attached_client's answer and record how often it was asked."""
    asked: list[object] = []
    monkeypatch.setattr(
        "bmad_loop.tui.launch.return_attached_client",
        lambda: (asked.append(outcome), outcome)[1],
    )
    return asked


def _run_one_decision_sweep(project):
    engine, _adapter = make_sweep(
        project,
        [triage_effect(_close_decision_plan())],
        answers=["1"],
        prompting=True,
    )
    summary = engine.run()
    assert not summary.paused
    return engine


def test_interactive_decisions_return_client_goes_unattended(project, monkeypatch):
    """When a client was attached to answer, the sweep hands the terminal back
    after the decisions and goes unattended so later cycles don't block on a
    detached window."""
    asked = _stub_return(monkeypatch, launch.ReturnOutcome.RETURNED)
    write_ledger(project, {"DW-1": "open"})
    engine = _run_one_decision_sweep(project)
    assert len(asked) == 1  # asked exactly once, after the decisions phase
    assert engine.prompting is False
    assert '"sweep-returned-after-decisions"' in journal_text(engine)


def test_interactive_decisions_no_attach_stays_attended(project, monkeypatch):
    """A plain foreground sweep (nobody attached, no return target) keeps
    prompting and never emits the return event."""
    _stub_return(monkeypatch, launch.ReturnOutcome.ATTENDED)
    write_ledger(project, {"DW-1": "open"})
    engine = _run_one_decision_sweep(project)
    assert engine.prompting is True
    assert '"sweep-returned-after-decisions"' not in journal_text(engine)


def test_interactive_decisions_unreachable_goes_unattended(project, monkeypatch):
    """A failed detach is not the same non-return as a failed switch: it means
    there was no client to hand back, so nobody can answer here any more. The
    sweep must go unattended anyway — keeping `prompting` would leave a
    --repeat cycle blocked on input() in a window no one is viewing — but it
    must not claim a hand-back that did not happen."""
    _stub_return(monkeypatch, launch.ReturnOutcome.UNREACHABLE)
    write_ledger(project, {"DW-1": "open"})
    engine = _run_one_decision_sweep(project)
    assert engine.prompting is False
    assert '"sweep-return-no-client"' in journal_text(engine)
    assert '"sweep-returned-after-decisions"' not in journal_text(engine)


def test_unattended_skips_decisions(project):
    write_ledger(project, {"DW-1": "open", "DW-2": "open"})
    plan = triage_result(
        ["DW-1", "DW-2"],
        bundles=[{"name": "safe-fix", "dw_ids": ["DW-2"], "intent": "fix it"}],
        decisions=[
            {
                "id": "DW-1",
                "question": "q",
                "context": "",
                "options": [
                    {"key": "1", "label": "a", "effect": "build", "intent": "x"},
                    {"key": "2", "label": "b", "effect": "keep-open"},
                ],
                "recommendation": "2",
            }
        ],
    )
    engine, adapter = make_sweep(
        project,
        [
            triage_effect(plan),
            bundle_dev_effect(project, "safe-fix", ["DW-2"]),
            bundle_review_effect(project, "safe-fix"),
        ],
        prompting=False,
    )
    summary = engine.run()
    assert not summary.paused
    assert "decision-skipped-unattended" in journal_text(engine)
    entries = ledger_entries(project)
    assert entries["DW-1"].open  # untouched, waits for an interactive sweep
    assert entries["DW-2"].status.startswith("done")
    assert not (engine.run_dir / "decisions.json").is_file()


def test_triage_session_stamps_resolved_adapter_identity(project):
    """#153 phase 1: the sweep triage session's session-start entry and persisted
    SessionRecord carry the resolved triage adapter profile + model. A triage-stage
    name override switches the profile and resets the model to the CLI default ""."""
    write_ledger(project, {"DW-1": "open"})
    plan = triage_result(
        ["DW-1"],
        already_resolved=[{"id": "DW-1", "evidence": "already guarded at src.txt:1"}],
    )
    policy = Policy(
        gates=GatesPolicy(mode="none"),
        notify=QUIET,
        scm=ScmPolicy(rollback_on_failure=True),
        adapter=AdapterPolicy(
            name="claude",
            model="opus",
            triage=StageAdapterPolicy(name="gemini"),
        ),
    )
    engine, adapter = make_sweep(project, [triage_effect(plan)], policy=policy)
    summary = engine.run()
    assert not summary.paused
    assert [s.role for s in adapter.sessions] == ["triage"]

    expected = policy.adapter.resolved("triage")
    assert (expected.name, expected.model) == ("gemini", "")  # client switch resets model

    entries = [json.loads(line) for line in journal_text(engine).splitlines()]
    starts = [e for e in entries if e["kind"] == "session-start" and e.get("role") == "triage"]
    assert len(starts) == 1
    assert starts[0]["adapter"] == "gemini"
    assert starts[0]["model"] == ""
    assert starts[0]["story_key"] == "sweep-triage"

    saved = load_state(engine.run_dir)
    rec = saved.tasks["sweep-triage"].sessions[-1]
    assert rec.role == "triage"
    assert (rec.adapter, rec.model) == ("gemini", "")


def test_bundle_review_disabled_skips_review_session(project):
    """review.enabled = false: a bundle's dev session finalizes to done and the
    sweep commits with no separate bundle-review session."""
    write_ledger(project, {"DW-1": "open"})
    plan = triage_result(
        ["DW-1"],
        bundles=[{"name": "some-fix", "dw_ids": ["DW-1"], "intent": "fix it"}],
    )
    engine, adapter = make_sweep(
        project,
        [triage_effect(plan), bundle_dev_effect(project, "some-fix", ["DW-1"])],
        policy=Policy(
            gates=GatesPolicy(mode="none"),
            notify=QUIET,
            review=ReviewPolicy(enabled=False),
        ),
    )
    summary = engine.run()
    assert not summary.paused
    assert [s.role for s in adapter.sessions] == ["triage", "dev"]  # no review
    assert engine.state.tasks["dw-some-fix"].phase == Phase.DONE
    assert ledger_entries(project)["DW-1"].status.startswith("done")
    assert "review-skipped" in journal_text(engine)


def test_decisions_only_runs_no_bundles(project):
    write_ledger(project, {"DW-1": "open", "DW-2": "open"})
    plan = triage_result(
        ["DW-1", "DW-2"],
        bundles=[{"name": "some-fix", "dw_ids": ["DW-2"], "intent": "fix it"}],
        decisions=[
            {
                "id": "DW-1",
                "question": "q",
                "context": "",
                "options": [
                    {"key": "1", "label": "Close", "effect": "close"},
                    {"key": "2", "label": "Keep", "effect": "keep-open"},
                ],
                "recommendation": "1",
            }
        ],
    )
    engine, adapter = make_sweep(
        project,
        [triage_effect(plan)],
        answers=["1"],
        prompting=True,
        decisions_only=True,
    )
    summary = engine.run()
    assert not summary.paused
    assert len(adapter.sessions) == 1  # triage only
    assert "sweep-decisions-only" in journal_text(engine)
    entries = ledger_entries(project)
    assert entries["DW-1"].status.startswith("done")
    assert entries["DW-2"].open  # bundle not run


def _decision(dw_id, options, recommendation="1", question="q"):
    return {
        "id": dw_id,
        "question": question,
        "context": "",
        "options": options,
        "recommendation": recommendation,
    }


def test_preanswered_build_materializes_bundle_unattended(project):
    """A build pre-answered out of band is consumed by a later unattended sweep
    even though triage re-surfaced it as a decision — and the stored intent is
    used when the triage option keys no longer match (option renumbered)."""
    from bmad_loop import decisions
    from bmad_loop.sweep import DecisionOption

    write_ledger(project, {"DW-1": "open"})
    # answered out of band against an earlier triage: stored key "9" is NOT one
    # of this triage's option keys, so the sweep must fall back to stored intent
    decisions.record_pre_answer(
        project.project,
        "DW-1",
        DecisionOption(key="9", label="Widen", effect="build", intent="widen the field"),
        date="2026-06-12",
    )
    plan = triage_result(
        ["DW-1"],
        decisions=[
            _decision(
                "DW-1",
                [
                    {"key": "1", "label": "Widen", "effect": "build", "intent": "fresh intent"},
                    {"key": "2", "label": "Keep", "effect": "keep-open"},
                ],
            )
        ],
    )
    engine, _ = make_sweep(
        project,
        [
            triage_effect(plan),
            bundle_dev_effect(project, "decision-dw-1", ["DW-1"]),
            bundle_review_effect(project, "decision-dw-1"),
        ],
        prompting=False,  # unattended: without the pre-answer this would be skipped
    )
    summary = engine.run()
    assert not summary.paused

    journal = journal_text(engine)
    assert '"decision-preanswered"' in journal
    assert "decision-skipped-unattended" not in journal
    assert engine.state.tasks["dw-decision-dw-1"].phase == Phase.DONE
    assert ledger_entries(project)["DW-1"].status.startswith("done")
    # consumed: the entry left the open set, so its pre-answer is pruned
    assert decisions.load_pre_answers(project.project) == {}
    assert '"decision-preanswers-pruned"' in journal
    # DW-118 row 8 — the key resolved to NO option, so there is no option to
    # describe and nothing is journaled as mismatched; the note still takes the
    # earlier-triage wording, since this answer was made against a triage whose
    # options are gone. Ablation: journal the record unconditionally and the
    # first assert reddens; keep the single matched-note wording and the last.
    assert _mismatches(engine) == []
    intent_md = (engine.run_dir / "bundles" / "decision-dw-1" / "intent.md").read_text(
        encoding="utf-8"
    )
    assert "widen the field" in intent_md
    assert "fresh intent" not in intent_md
    assert "against an earlier triage of DW-1" in intent_md


def test_preanswered_bundle_name_failing_the_segment_gate_is_discarded(project):
    """Round-2 review: a pre-answer's `bundle_name` never passes `validate_triage`
    — it was answered out of band against an earlier triage, and a fresh one can
    renumber or drop the option it named — so it was the one route by which a name
    failing the two option-site gates (#637) still reached `_write_intent` as a
    directory (`nul` passes BUNDLE_NAME_RE and fails `safe_segment` identity).
    `_materialize_bundles` now applies the same two rules to that lane, by
    journaled DISCARD rather than by error: the build decision is the payload and
    the always-legal `decision-<id>` fallback is what an unnamed answer gets
    anyway. Ablation: drop that gate and this reddens — the bundle materializes
    as `nul`, so the `decision-dw-1` effects below never match and the discard
    event never appears."""
    from bmad_loop import decisions
    from bmad_loop.sweep import DecisionOption

    write_ledger(project, {"DW-1": "open"})
    # stored key "9" is NOT one of this triage's option keys, so every field —
    # bundle_name included — comes from the stored answer, not a validated option
    decisions.record_pre_answer(
        project.project,
        "DW-1",
        DecisionOption(
            key="9", label="Widen", effect="build", intent="widen the field", bundle_name="nul"
        ),
        date="2026-06-12",
    )
    plan = triage_result(
        ["DW-1"],
        decisions=[
            _decision(
                "DW-1",
                [
                    {"key": "1", "label": "Widen", "effect": "build", "intent": "fresh intent"},
                    {"key": "2", "label": "Keep", "effect": "keep-open"},
                ],
            )
        ],
    )
    engine, _ = make_sweep(
        project,
        [
            triage_effect(plan),
            bundle_dev_effect(project, "decision-dw-1", ["DW-1"]),
            bundle_review_effect(project, "decision-dw-1"),
        ],
        prompting=False,
    )
    summary = engine.run()
    assert not summary.paused

    journal = journal_text(engine)
    assert '"sweep-bundle-name-discarded"' in journal
    assert '"nul"' in journal  # the discard names the spelling it dropped
    assert engine.state.tasks["dw-decision-dw-1"].phase == Phase.DONE
    assert "dw-nul" not in engine.state.tasks  # the raw name minted nothing


def test_stale_preanswer_meeting_a_close_option_keeps_its_own_semantics(project):
    """DW-118: `Decision.option` matches on KEY ALONE, and a key is a position in
    a list a later triage re-authors freely, so a renumbering re-triage hands this
    loop one question's stored answer beside another question's option. On the
    real DW-55 a stored `build` answer keyed "1" met a fresh option "1" spelled
    "Close as decayed": the bundle shipped the stale review intent under the close
    label and quoted a question the human never answered. The label/effect
    agreement check now discards the option outright — it contributes no field —
    and the note says the choice was made against an earlier triage. Ablation:
    delete the agreement check and this reddens twice over — `label` takes the
    fresh option's "Close as decayed" and the note interpolates `decision.question`
    back into `intent.md`."""
    from bmad_loop import decisions
    from bmad_loop.sweep import DecisionOption

    write_ledger(project, {"DW-1": "open"})
    decisions.record_pre_answer(
        project.project,
        "DW-1",
        DecisionOption(
            key="1",
            label="Run the follow-up review",
            effect="build",
            intent="re-run the follow-up review over the seam",
        ),
        date="2026-06-12",
    )
    plan = triage_result(
        ["DW-1"],
        decisions=[
            _decision(
                "DW-1",
                [
                    {"key": "1", "label": "Close as decayed", "effect": "close"},
                    {"key": "2", "label": "Keep", "effect": "keep-open"},
                ],
                question="has this entry decayed past usefulness?",
            )
        ],
    )
    engine, _ = make_sweep(
        project,
        [
            triage_effect(plan),
            bundle_dev_effect(project, "decision-dw-1", ["DW-1"]),
            bundle_review_effect(project, "decision-dw-1"),
        ],
        prompting=False,
    )
    summary = engine.run()
    assert not summary.paused

    intent_md = (engine.run_dir / "bundles" / "decision-dw-1" / "intent.md").read_text(
        encoding="utf-8"
    )
    assert "re-run the follow-up review over the seam" in intent_md  # the STORED intent
    assert "Run the follow-up review" in intent_md  # the STORED label
    assert "Close as decayed" not in intent_md  # never the non-matching option's
    assert "decayed past usefulness" not in intent_md  # nor the question it never answered
    assert "against an earlier triage of DW-1" in intent_md
    [record] = _mismatches(engine)
    assert record["decision"] == "DW-1"
    assert record["key"] == "1"
    assert record["option_effect"] == "close"
    assert record["label_matched"] is False
    # the lane discriminator, pinned from the BUILD side — the keep-open row that
    # pins "keep-open" cannot tell a real read of the answer from a constant
    assert record["answer_effect"] == "build"


def test_stale_preanswer_meeting_a_different_build_option_keeps_its_own_intent(project):
    """The harm class is not "a build answer met a close option" — it is a
    DIFFERENT option's fields reaching the bundle. Two build options renumbered
    against each other is the case a `effect`-only check would miss entirely: both
    sides say `build`, so only the label clause separates them, and the dev
    session would otherwise be dispatched to narrow a field the human asked to
    widen. Ablation: delete the agreement check and `intent` takes the fresh
    option's "narrow the field" (the old precedence preferred a non-empty option
    field), so both intent asserts flip."""
    from bmad_loop import decisions
    from bmad_loop.sweep import DecisionOption

    write_ledger(project, {"DW-1": "open"})
    decisions.record_pre_answer(
        project.project,
        "DW-1",
        DecisionOption(key="1", label="Widen", effect="build", intent="widen the field"),
        date="2026-06-12",
    )
    plan = triage_result(
        ["DW-1"],
        decisions=[
            _decision(
                "DW-1",
                [
                    {
                        "key": "1",
                        "label": "Narrow",
                        "effect": "build",
                        "intent": "narrow the field",
                    },
                    {"key": "2", "label": "Keep", "effect": "keep-open"},
                ],
            )
        ],
    )
    engine, _ = make_sweep(
        project,
        [
            triage_effect(plan),
            bundle_dev_effect(project, "decision-dw-1", ["DW-1"]),
            bundle_review_effect(project, "decision-dw-1"),
        ],
        prompting=False,
    )
    summary = engine.run()
    assert not summary.paused

    intent_md = (engine.run_dir / "bundles" / "decision-dw-1" / "intent.md").read_text(
        encoding="utf-8"
    )
    assert "widen the field" in intent_md
    assert "narrow the field" not in intent_md
    assert "Narrow" not in intent_md
    [record] = _mismatches(engine)
    assert record["option_effect"] == "build"
    assert record["label_matched"] is False


def test_stale_preanswer_mismatching_on_effect_alone_is_still_non_matching(project):
    """`validate_triage` normalizes an absent option label to its KEY, so when
    both triages omit the label the label clause degenerates to key equality and
    the `effect` clause is the only discriminator left. This row drives that
    clause on its own: identical labels, `build` re-authored to `close`. Ablation:
    delete `or option.effect != str(answer.get("effect", ""))` and this reddens
    while every other mismatch row here stays green — they all differ on `label`
    too. (The stored intent still reaches the bundle either way; what the ablation
    loses is the record and the earlier-triage note wording.)"""
    from bmad_loop import decisions
    from bmad_loop.sweep import DecisionOption

    write_ledger(project, {"DW-1": "open"})
    decisions.record_pre_answer(
        project.project,
        "DW-1",
        DecisionOption(key="1", label="Handle it", effect="build", intent="handle the seam"),
        date="2026-06-12",
    )
    plan = triage_result(
        ["DW-1"],
        decisions=[
            _decision(
                "DW-1",
                [
                    # SAME label, re-authored effect: only the effect clause can see it
                    {"key": "1", "label": "Handle it", "effect": "close"},
                    {"key": "2", "label": "Keep", "effect": "keep-open"},
                ],
                question="is this seam still worth handling?",
            )
        ],
    )
    engine, _ = make_sweep(
        project,
        [
            triage_effect(plan),
            bundle_dev_effect(project, "decision-dw-1", ["DW-1"]),
            bundle_review_effect(project, "decision-dw-1"),
        ],
        prompting=False,
    )
    summary = engine.run()
    assert not summary.paused

    [record] = _mismatches(engine)
    assert record["label_matched"] is True  # the label agreed; the effect did not
    assert record["option_effect"] == "close"
    intent_md = (engine.run_dir / "bundles" / "decision-dw-1" / "intent.md").read_text(
        encoding="utf-8"
    )
    assert "handle the seam" in intent_md  # a mismatch never suppresses a buildable bundle
    assert "against an earlier triage of DW-1" in intent_md
    assert "still worth handling" not in intent_md


def test_preanswer_agreeing_with_a_reworded_option_keeps_the_prose_the_human_chose(project):
    """Field-presence precedence: the stored answer is the payload, so an AGREEING
    option fills only the fields the answer omits. Every triage session authors
    fresh prose, so an option that still agrees on label and effect routinely
    carries a rewritten `intent`/`bundle_name` the human never read — and under
    the old `option_field or answer_field` order that rewrite is what the dev
    session was dispatched on. Agreement means no mismatch record; it does not
    mean the fresh prose wins. Ablation: restore `(option.intent if option else
    "") or str(answer.get("intent", ""))` and this reddens — `intent.md` takes
    "REWRITTEN" and the bundle is minted as `rewritten-x`, so the `widen-x`
    session effects below never match."""
    from bmad_loop import decisions
    from bmad_loop.sweep import DecisionOption

    write_ledger(project, {"DW-1": "open"})
    decisions.record_pre_answer(
        project.project,
        "DW-1",
        DecisionOption(
            key="1",
            label="Widen",
            effect="build",
            intent="widen the field",
            bundle_name="widen-x",
        ),
        date="2026-06-12",
    )
    plan = triage_result(
        ["DW-1"],
        decisions=[
            _decision(
                "DW-1",
                [
                    {
                        "key": "1",
                        "label": "Widen",  # agrees
                        "effect": "build",  # agrees
                        "intent": "REWRITTEN by a later triage",
                        "bundle_name": "rewritten-x",
                    },
                    {"key": "2", "label": "Keep", "effect": "keep-open"},
                ],
                question="how wide should the field be?",
            )
        ],
    )
    engine, _ = make_sweep(
        project,
        [
            triage_effect(plan),
            bundle_dev_effect(project, "widen-x", ["DW-1"]),
            bundle_review_effect(project, "widen-x"),
        ],
        prompting=False,
    )
    summary = engine.run()
    assert not summary.paused

    assert _mismatches(engine) == []  # the option agreed; nothing to journal
    assert not (engine.run_dir / "bundles" / "rewritten-x").exists()
    intent_md = (engine.run_dir / "bundles" / "widen-x" / "intent.md").read_text(encoding="utf-8")
    assert "widen the field" in intent_md
    assert "REWRITTEN" not in intent_md
    # an agreeing option keeps the current-question note wording
    assert "how wide should the field be?" in intent_md
    assert "against an earlier triage" not in intent_md


def test_preanswer_omitting_a_bundle_name_takes_it_from_the_agreeing_option(project):
    """The accepted seam of field-presence precedence: an agreeing option fills
    only what the stored answer OMITS, so a pre-answer carrying no `bundle_name`
    is minted under the option's. That is deliberate — the option agrees on both
    label and effect, so it is the same option under a re-authored name, and the
    alternative is the generic `decision-<id>` fallback. It is not the DW-118
    harm class, which is a DIFFERENT option's fields reaching the bundle.
    Ablation: make an agreeing option contribute nothing either and the bundle
    mints as `decision-dw-1`, so the `widen-x` session effects below never match
    and the run fails verification."""
    from bmad_loop import decisions
    from bmad_loop.sweep import DecisionOption

    write_ledger(project, {"DW-1": "open"})
    decisions.record_pre_answer(
        project.project,
        "DW-1",
        # no bundle_name of its own: the gap the agreeing option fills
        DecisionOption(key="1", label="Widen", effect="build", intent="widen the field"),
        date="2026-06-12",
    )
    plan = triage_result(
        ["DW-1"],
        decisions=[
            _decision(
                "DW-1",
                [
                    {
                        "key": "1",
                        "label": "Widen",  # agrees
                        "effect": "build",  # agrees
                        "intent": "fresh intent",
                        "bundle_name": "widen-x",
                    },
                    {"key": "2", "label": "Keep", "effect": "keep-open"},
                ],
            )
        ],
    )
    engine, _ = make_sweep(
        project,
        [
            triage_effect(plan),
            bundle_dev_effect(project, "widen-x", ["DW-1"]),
            bundle_review_effect(project, "widen-x"),
        ],
        prompting=False,
    )
    summary = engine.run()
    assert not summary.paused

    assert _mismatches(engine) == []  # the option agreed
    assert engine.state.tasks["dw-widen-x"].phase == Phase.DONE  # the OPTION's name
    assert not (engine.run_dir / "bundles" / "decision-dw-1").exists()
    intent_md = (engine.run_dir / "bundles" / "widen-x" / "intent.md").read_text(encoding="utf-8")
    assert "widen the field" in intent_md  # the intent the answer DID carry still wins
    assert "fresh intent" not in intent_md


def test_stale_in_run_answer_across_repeat_cycles_is_dropped_and_notified(project):
    """The cross-cycle IN-RUN lane, which no pre-answer test reaches: `_cycle`
    calls `_ensure_triage` per repeat cycle (a fresh, renumberable `triage-2.json`)
    while `answers` persists for the whole run in `<run>/decisions.json`, so a
    cycle-1 in-run answer is replayed against cycle-2's options. An in-run answer
    is written with only key/label/effect/answered_at, so once the disagreeing
    option is discarded there is nothing left to build from. Re-asking here would
    double-apply `_apply_decision_effect` (cycle 1 already wrote the ledger's
    decision line), so the answer is dropped — but a recorded human `build`
    decision must not vanish on a journal line alone, hence the dedicated record
    plus `gates.notify`, and the entry stays open for the next sweep to re-ask.
    `max_bundles=1` is what keeps DW-1 open into cycle 2: cycle 1's decision
    bundle is truncated behind the plan bundle. Ablation: delete the
    `sweep-decision-answer-dropped` + notify block and both the record and the
    ATTENTION assert redden."""
    write_ledger(project, {"DW-1": "open", "DW-2": "open"})
    plan1 = triage_result(
        ["DW-1", "DW-2"],
        bundles=[{"name": "safe-fix", "dw_ids": ["DW-2"], "intent": "a"}],
        decisions=[
            _decision(
                "DW-1",
                [
                    {"key": "1", "label": "Widen", "effect": "build", "intent": "widen the field"},
                    {"key": "2", "label": "Keep", "effect": "keep-open"},
                ],
            )
        ],
    )
    plan2 = triage_result(
        ["DW-1"],
        decisions=[
            _decision(
                "DW-1",
                [
                    # cycle 2 renumbered: key "1" is now a different question's answer
                    {"key": "1", "label": "Close as decayed", "effect": "close"},
                    {"key": "2", "label": "Keep", "effect": "keep-open"},
                ],
            )
        ],
    )
    engine, _ = make_sweep(
        project,
        [
            triage_effect(plan1),
            bundle_dev_effect(project, "safe-fix", ["DW-2"]),
            bundle_review_effect(project, "safe-fix"),
            triage_effect(plan2),
        ],
        policy=repeat_policy(max_bundles=1),
        answers=["1"],
        prompting=True,
    )
    summary = engine.run()
    assert not summary.paused

    journal = journal_text(engine)
    # PRECONDITION: the in-run answer was recorded, and cycle 1 truncated its bundle
    assert '"decision-answered"' in journal
    assert '"sweep-bundles-truncated"' in journal
    [record] = _mismatches(engine)
    assert record["decision"] == "DW-1" and record["option_effect"] == "close"
    assert record["label_matched"] is False
    assert '"sweep-decision-answer-dropped"' in journal
    assert not any(k.startswith("dw2-") for k in engine.state.tasks)  # cycle 2 built nothing
    attention = (engine.run_dir / "ATTENTION").read_text(encoding="utf-8")
    assert "recorded build decision discarded" in attention
    assert ledger_entries(project)["DW-1"].open  # re-asked by the next sweep


def test_non_matching_answers_bundle_name_still_faces_the_segment_gate(project):
    """The `bundle_name` discard gate (#637) runs over whatever name survives the
    agreement check, not only over the pre-existing key-not-found lane. This row
    drives it through the label/effect mismatch lane: the key DOES resolve, the
    option disagrees, so the stored `nul` is the surviving name and must still be
    discarded down to `decision-<id>`. Ablation: drop the two-rule gate and the
    bundle materializes as `nul`, so the `decision-dw-1` effects below never match
    and the run fails verification."""
    from bmad_loop import decisions
    from bmad_loop.sweep import DecisionOption

    write_ledger(project, {"DW-1": "open"})
    decisions.record_pre_answer(
        project.project,
        "DW-1",
        DecisionOption(
            key="1", label="Widen", effect="build", intent="widen the field", bundle_name="nul"
        ),
        date="2026-06-12",
    )
    plan = triage_result(
        ["DW-1"],
        decisions=[
            _decision(
                "DW-1",
                [
                    # same key, different label: resolves, then disagrees
                    {"key": "1", "label": "Narrow", "effect": "build", "intent": "narrow it"},
                    {"key": "2", "label": "Keep", "effect": "keep-open"},
                ],
            )
        ],
    )
    engine, _ = make_sweep(
        project,
        [
            triage_effect(plan),
            bundle_dev_effect(project, "decision-dw-1", ["DW-1"]),
            bundle_review_effect(project, "decision-dw-1"),
        ],
        prompting=False,
    )
    summary = engine.run()
    assert not summary.paused

    journal = journal_text(engine)
    assert len(_mismatches(engine)) == 1  # PRECONDITION: the mismatch lane, not key-not-found
    assert '"sweep-bundle-name-discarded"' in journal
    assert '"nul"' in journal
    assert engine.state.tasks["dw-decision-dw-1"].phase == Phase.DONE
    assert "dw-nul" not in engine.state.tasks


def test_non_matching_answer_that_builds_closes_its_entry_and_is_pruned(project):
    """A mismatch changes which fields the bundle is built from — nothing else. A
    non-matching answer that still carries its own intent runs the full pipeline,
    so the ledger entry closes and the consumed pre-answer is pruned exactly as a
    matching one's is; the human's `build` decision stands and only the current
    triage's option is discarded. Driven through the label/effect mismatch lane
    (the key resolves), not the pre-existing key-not-found lane. Ablation: make
    the mismatch `continue` instead of discarding the option and DW-1 stays open
    with its pre-answer still in the store."""
    from bmad_loop import decisions
    from bmad_loop.sweep import DecisionOption

    write_ledger(project, {"DW-1": "open"})
    decisions.record_pre_answer(
        project.project,
        "DW-1",
        DecisionOption(key="1", label="Widen", effect="build", intent="widen the field"),
        date="2026-06-12",
    )
    plan = triage_result(
        ["DW-1"],
        decisions=[
            _decision(
                "DW-1",
                [
                    {"key": "1", "label": "Close as decayed", "effect": "close"},
                    {"key": "2", "label": "Keep", "effect": "keep-open"},
                ],
            )
        ],
    )
    engine, _ = make_sweep(
        project,
        [
            triage_effect(plan),
            bundle_dev_effect(project, "decision-dw-1", ["DW-1"]),
            bundle_review_effect(project, "decision-dw-1"),
        ],
        prompting=False,
    )
    summary = engine.run()
    assert not summary.paused

    assert len(_mismatches(engine)) == 1  # PRECONDITION: the mismatch lane
    assert engine.state.tasks["dw-decision-dw-1"].phase == Phase.DONE
    assert ledger_entries(project)["DW-1"].status.startswith("done")
    assert decisions.load_pre_answers(project.project) == {}
    assert '"decision-preanswers-pruned"' in journal_text(engine)


def test_stale_in_run_answer_whose_option_vanished_is_dropped_without_a_mismatch(project):
    """The other cross-cycle shape: cycle 2 DROPS the option instead of renumbering
    it, so the key resolves to nothing. There is no option to describe, so no
    `sweep-decision-option-mismatch` is written — but the in-run answer still
    carries no `intent`, so the drop lane fires and the operator is told. This is
    a behavior change on a lane that used to `continue` in silence, and it is the
    one drop shape the renumbering row above cannot reach. Ablation: delete the
    `sweep-decision-answer-dropped` + notify block and both the record and the
    ATTENTION assert redden, while the empty-mismatch assert stays green."""
    write_ledger(project, {"DW-1": "open", "DW-2": "open"})
    plan1 = triage_result(
        ["DW-1", "DW-2"],
        bundles=[{"name": "safe-fix", "dw_ids": ["DW-2"], "intent": "a"}],
        decisions=[
            _decision(
                "DW-1",
                [
                    {"key": "1", "label": "Widen", "effect": "build", "intent": "widen the field"},
                    {"key": "2", "label": "Keep", "effect": "keep-open"},
                ],
            )
        ],
    )
    plan2 = triage_result(
        ["DW-1"],
        decisions=[
            # the answered option is GONE from cycle 2, not renumbered onto
            _decision(
                "DW-1",
                # a decision needs two options; the answered key "1" is not one
                [
                    {"key": "2", "label": "Keep", "effect": "keep-open"},
                    {"key": "3", "label": "Close as decayed", "effect": "close"},
                ],
                recommendation="2",  # key "1" is gone, so it cannot be recommended either
            ),
        ],
    )
    engine, _ = make_sweep(
        project,
        [
            triage_effect(plan1),
            bundle_dev_effect(project, "safe-fix", ["DW-2"]),
            bundle_review_effect(project, "safe-fix"),
            triage_effect(plan2),
        ],
        policy=repeat_policy(max_bundles=1),
        answers=["1"],
        prompting=True,
    )
    summary = engine.run()
    assert not summary.paused

    journal = journal_text(engine)
    assert '"sweep-bundles-truncated"' in journal  # PRECONDITION: DW-1 survived cycle 1
    assert _mismatches(engine) == []  # no option resolved, so nothing to describe
    assert '"sweep-decision-answer-dropped"' in journal
    assert not any(k.startswith("dw2-") for k in engine.state.tasks)
    attention = (engine.run_dir / "ATTENTION").read_text(encoding="utf-8")
    assert "recorded build decision discarded" in attention
    assert ledger_entries(project)["DW-1"].open


def test_stale_in_run_answer_never_inherits_a_disagreeing_options_intent(project):
    """The `option = None` discard is what makes "a non-matching option
    contributes NO field" true — `matched = False` alone only changes the note
    wording. Every other mismatch row here stores an answer that already carries
    the field the disagreeing option could leak, so none of them can see the
    discard. This one can: the cycle-1 IN-RUN answer carries no `intent` at all,
    and cycle 2 renumbers key "1" onto a *build* option that does. Ablation:
    delete `option = None` alone (leaving `matched = False`) and this reddens —
    `intent` falls through to "narrow the field", a cycle-2 bundle is minted from
    prose the human never chose, and the dropped record is never written."""
    write_ledger(project, {"DW-1": "open", "DW-2": "open"})
    plan1 = triage_result(
        ["DW-1", "DW-2"],
        bundles=[{"name": "safe-fix", "dw_ids": ["DW-2"], "intent": "a"}],
        decisions=[
            _decision(
                "DW-1",
                [
                    {"key": "1", "label": "Widen", "effect": "build", "intent": "widen the field"},
                    {"key": "2", "label": "Keep", "effect": "keep-open"},
                ],
            )
        ],
    )
    plan2 = triage_result(
        ["DW-1"],
        decisions=[
            _decision(
                "DW-1",
                [
                    # renumbered onto a DIFFERENT build option — one that carries the
                    # very field the in-run answer lacks
                    {
                        "key": "1",
                        "label": "Narrow",
                        "effect": "build",
                        "intent": "narrow the field",
                    },
                    {"key": "2", "label": "Keep", "effect": "keep-open"},
                ],
            )
        ],
    )
    engine, _ = make_sweep(
        project,
        [
            triage_effect(plan1),
            bundle_dev_effect(project, "safe-fix", ["DW-2"]),
            bundle_review_effect(project, "safe-fix"),
            triage_effect(plan2),
        ],
        policy=repeat_policy(max_bundles=1),
        answers=["1"],
        prompting=True,
    )
    summary = engine.run()
    assert not summary.paused

    journal = journal_text(engine)
    assert '"sweep-bundles-truncated"' in journal  # PRECONDITION: DW-1 survived cycle 1
    [record] = _mismatches(engine)
    assert record["option_effect"] == "build" and record["label_matched"] is False
    assert '"sweep-decision-answer-dropped"' in journal
    assert not any(k.startswith("dw2-") for k in engine.state.tasks)
    # the discarded option's intent reached no bundle anywhere in the run
    assert not any(
        "narrow the field" in f.read_text(encoding="utf-8")
        for f in engine.run_dir.glob("bundles/*/intent.md")
    )
    assert ledger_entries(project)["DW-1"].open


def test_disagreeing_options_bundle_name_never_names_the_bundle(project):
    """The other half of the `option = None` discard: a stored answer that omits
    `bundle_name` beside a DISAGREEING option that carries one. The omission is
    legitimate (the fallback is `decision-<id>`), so nothing but the discard stops
    a different option's directory name from being minted. Ablation: delete
    `option = None` alone and this reddens — the bundle mints as `narrow-x`, so
    `dw-decision-dw-1` is never created and the session effects below never
    match."""
    from bmad_loop import decisions
    from bmad_loop.sweep import DecisionOption

    write_ledger(project, {"DW-1": "open"})
    decisions.record_pre_answer(
        project.project,
        "DW-1",
        # no bundle_name of its own
        DecisionOption(key="1", label="Widen", effect="build", intent="widen the field"),
        date="2026-06-12",
    )
    plan = triage_result(
        ["DW-1"],
        decisions=[
            _decision(
                "DW-1",
                [
                    {
                        "key": "1",
                        "label": "Narrow",  # disagrees
                        "effect": "build",
                        "intent": "narrow it",
                        "bundle_name": "narrow-x",
                    },
                    {"key": "2", "label": "Keep", "effect": "keep-open"},
                ],
            )
        ],
    )
    engine, _ = make_sweep(
        project,
        [
            triage_effect(plan),
            bundle_dev_effect(project, "decision-dw-1", ["DW-1"]),
            bundle_review_effect(project, "decision-dw-1"),
        ],
        prompting=False,
    )
    summary = engine.run()
    assert not summary.paused

    assert len(_mismatches(engine)) == 1  # PRECONDITION: the mismatch lane
    assert engine.state.tasks["dw-decision-dw-1"].phase == Phase.DONE
    assert "dw-narrow-x" not in engine.state.tasks
    assert not (engine.run_dir / "bundles" / "narrow-x").exists()
    intent_md = (engine.run_dir / "bundles" / "decision-dw-1" / "intent.md").read_text(
        encoding="utf-8"
    )
    assert "widen the field" in intent_md and "narrow it" not in intent_md


def test_stored_bundle_name_colliding_with_a_plan_bundle_is_discarded(project):
    """A stored answer's `bundle_name` never faced `validate_triage`'s `duplicate
    bundle name` rule (sweep.py:280) that a validated option's name did — and once
    the stored name wins the blend it can equal a plan bundle's, at which point
    both `Bundle`s hash to one `_bundle_key` and share one intent directory, so
    one of them is silently lost. The existing discard gate now carries that third
    rule, falling back to the always-legal `decision-<id>`. Driven on the AGREEING
    lane, which is where the precedence flip introduced the hazard. Ablation: drop
    the collision clause and this reddens — `dw-decision-dw-1` is never minted and
    `widen-x`'s intent.md carries the decision bundle's intent instead of the plan
    bundle's."""
    from bmad_loop import decisions
    from bmad_loop.sweep import DecisionOption

    write_ledger(project, {"DW-1": "open", "DW-2": "open"})
    decisions.record_pre_answer(
        project.project,
        "DW-1",
        DecisionOption(
            key="1",
            label="Widen",
            effect="build",
            intent="widen the field",
            bundle_name="widen-x",  # collides with the plan bundle below
        ),
        date="2026-06-12",
    )
    plan = triage_result(
        ["DW-1", "DW-2"],
        bundles=[{"name": "widen-x", "dw_ids": ["DW-2"], "intent": "the plan bundle's intent"}],
        decisions=[
            _decision(
                "DW-1",
                [
                    # agrees on label and effect, and carries no name of its own,
                    # so the stored `widen-x` is what survives the blend
                    {"key": "1", "label": "Widen", "effect": "build", "intent": "fresh intent"},
                    {"key": "2", "label": "Keep", "effect": "keep-open"},
                ],
            )
        ],
    )
    engine, _ = make_sweep(
        project,
        [
            triage_effect(plan),
            bundle_dev_effect(project, "widen-x", ["DW-2"]),
            bundle_review_effect(project, "widen-x"),
            bundle_dev_effect(project, "decision-dw-1", ["DW-1"]),
            bundle_review_effect(project, "decision-dw-1"),
        ],
        prompting=False,
    )
    summary = engine.run()
    assert not summary.paused

    journal = journal_text(engine)
    assert _mismatches(engine) == []  # PRECONDITION: the AGREEING lane
    assert '"sweep-bundle-name-discarded"' in journal
    assert engine.state.tasks["dw-widen-x"].phase == Phase.DONE
    assert engine.state.tasks["dw-decision-dw-1"].phase == Phase.DONE  # both bundles survived
    plan_intent = (engine.run_dir / "bundles" / "widen-x" / "intent.md").read_text(encoding="utf-8")
    assert "the plan bundle's intent" in plan_intent
    assert "widen the field" not in plan_intent  # never overwritten by the decision bundle
    assert ledger_entries(project)["DW-2"].status.startswith("done")


def test_fallback_name_colliding_with_a_plan_bundle_is_suffixed(project):
    """DW-118 follow-up: `decision-<id>` READS like a reserved namespace and is
    not one. `validate_triage` builds its duplicate-name set from plan bundle
    names and build-option `bundle_name`s only, so a plan may legally author a
    bundle called `decision-dw-1` and nothing compares the fallback against it —
    at which point `_bundle_key` (a pure function of the name) aliases both
    `Bundle`s onto ONE task and one intent directory, and the human's decision
    bundle is silently skipped or overwritten. The final name is now made unique
    with a bounded numeric suffix. Ablation: delete the uniqueness loop and this
    reddens — no `sweep-bundle-name-deduped` record, `dw-decision-dw-1-2` is
    never minted, and the single surviving `decision-dw-1` directory holds one
    intent instead of two."""
    from bmad_loop import decisions
    from bmad_loop.sweep import DecisionOption

    write_ledger(project, {"DW-1": "open", "DW-2": "open"})
    # no stored `bundle_name`, and the agreeing option carries none either, so
    # the name falls back to `decision-dw-1` — the plan bundle's exact name
    decisions.record_pre_answer(
        project.project,
        "DW-1",
        DecisionOption(key="1", label="Widen", effect="build", intent="widen the field"),
        date="2026-06-12",
    )
    plan = triage_result(
        ["DW-1", "DW-2"],
        bundles=[
            {"name": "decision-dw-1", "dw_ids": ["DW-2"], "intent": "the plan bundle's intent"}
        ],
        decisions=[
            _decision(
                "DW-1",
                [
                    {"key": "1", "label": "Widen", "effect": "build", "intent": "fresh intent"},
                    {"key": "2", "label": "Keep", "effect": "keep-open"},
                ],
            )
        ],
    )
    engine, _ = make_sweep(
        project,
        [
            triage_effect(plan),
            bundle_dev_effect(project, "decision-dw-1", ["DW-2"]),
            bundle_review_effect(project, "decision-dw-1"),
            bundle_dev_effect(project, "decision-dw-1-2", ["DW-1"]),
            bundle_review_effect(project, "decision-dw-1-2"),
        ],
        prompting=False,
    )
    summary = engine.run()
    assert not summary.paused

    assert _mismatches(engine) == []  # PRECONDITION: the AGREEING lane
    [deduped] = _records(engine, "sweep-bundle-name-deduped")
    assert deduped["decision"] == "DW-1" and deduped["attempt"] == 2
    # the alias is what the rule closes: distinct task keys AND distinct dirs
    assert engine.state.tasks["dw-decision-dw-1"].phase == Phase.DONE
    assert engine.state.tasks["dw-decision-dw-1-2"].phase == Phase.DONE
    assert engine.state.tasks["dw-decision-dw-1"].dw_ids == ["DW-2"]
    assert engine.state.tasks["dw-decision-dw-1-2"].dw_ids == ["DW-1"]
    bundles_dir = engine.run_dir / "bundles"
    plan_intent = (bundles_dir / "decision-dw-1" / "intent.md").read_text(encoding="utf-8")
    decision_intent = (bundles_dir / "decision-dw-1-2" / "intent.md").read_text(encoding="utf-8")
    assert "the plan bundle's intent" in plan_intent
    assert "widen the field" not in plan_intent  # never overwritten by the decision bundle
    assert "widen the field" in decision_intent
    entries = ledger_entries(project)
    assert entries["DW-1"].status.startswith("done")
    assert entries["DW-2"].status.startswith("done")


def test_fallback_name_colliding_with_an_earlier_decision_bundle_is_suffixed(project):
    """Two rules in one row. (a) The taken set is recomputed per decision, never
    snapshotted before the loop: decision bundles collide with EACH OTHER exactly
    as they collide with plan bundles — DW-1's stored `bundle_name` is literally
    `decision-dw-2`, the name DW-2's own answer then falls back to. (b) FIRST FREE
    wins, not a fixed `-2` retry: `-2` is occupied by a plan bundle here, so the
    scan has to walk past it to `-3`. Ablations: hoist
    `taken = {b.name for b in bundles}` above the decision loop and this reddens
    (the second bundle takes `decision-dw-2` and aliases onto the first); replace
    the `range(2, 10)` scan with a single fixed `-2` candidate and it reddens too
    (nothing is free, so the answer is dropped instead of deduped)."""
    from bmad_loop import decisions
    from bmad_loop.sweep import DecisionOption

    write_ledger(project, {"DW-1": "open", "DW-2": "open", "DW-3": "open"})
    decisions.record_pre_answer(
        project.project,
        "DW-1",
        DecisionOption(
            key="1",
            label="Widen",
            effect="build",
            intent="widen the field",
            bundle_name="decision-dw-2",  # the name DW-2's fallback will want
        ),
        date="2026-06-12",
    )
    decisions.record_pre_answer(
        project.project,
        "DW-2",
        DecisionOption(key="1", label="Narrow", effect="build", intent="narrow the field"),
        date="2026-06-12",
    )
    plan = triage_result(
        ["DW-1", "DW-2", "DW-3"],
        # occupies the FIRST suffix, so `-2` is not free when DW-2 is deduped
        bundles=[
            {"name": "decision-dw-2-2", "dw_ids": ["DW-3"], "intent": "the plan bundle's intent"}
        ],
        decisions=[
            _decision(
                "DW-1",
                [
                    {"key": "1", "label": "Widen", "effect": "build", "intent": "fresh 1"},
                    {"key": "2", "label": "Keep", "effect": "keep-open"},
                ],
            ),
            _decision(
                "DW-2",
                [
                    {"key": "1", "label": "Narrow", "effect": "build", "intent": "fresh 2"},
                    {"key": "2", "label": "Keep", "effect": "keep-open"},
                ],
            ),
        ],
    )
    engine, _ = make_sweep(
        project,
        [
            triage_effect(plan),
            bundle_dev_effect(project, "decision-dw-2-2", ["DW-3"]),
            bundle_review_effect(project, "decision-dw-2-2"),
            bundle_dev_effect(project, "decision-dw-2", ["DW-1"]),
            bundle_review_effect(project, "decision-dw-2"),
            bundle_dev_effect(project, "decision-dw-2-3", ["DW-2"]),
            bundle_review_effect(project, "decision-dw-2-3"),
        ],
        prompting=False,
        max_bundles=3,
    )
    summary = engine.run()
    assert not summary.paused

    assert _mismatches(engine) == []  # PRECONDITION: both options agree
    [deduped] = _records(engine, "sweep-bundle-name-deduped")
    assert deduped["decision"] == "DW-2"
    assert deduped["attempt"] == 3  # `-2` was taken; the scan walked past it
    assert deduped["name"] == "decision-dw-2-3"
    assert engine.state.tasks["dw-decision-dw-2"].dw_ids == ["DW-1"]
    assert engine.state.tasks["dw-decision-dw-2-3"].dw_ids == ["DW-2"]
    assert engine.state.tasks["dw-decision-dw-2-2"].dw_ids == ["DW-3"]  # the plan bundle
    bundles_dir = engine.run_dir / "bundles"
    first = (bundles_dir / "decision-dw-2" / "intent.md").read_text(encoding="utf-8")
    second = (bundles_dir / "decision-dw-2-3" / "intent.md").read_text(encoding="utf-8")
    assert "widen the field" in first and "narrow the field" not in first
    assert "narrow the field" in second


def test_exhausting_the_name_suffix_bound_drops_and_notifies_the_decision(project):
    """The SOLE case in which a buildable stored answer produces no bundle: the
    fallback and every candidate in the bound are already taken, so there is no
    name left that would not alias the human's bundle onto another one. A naming
    impossibility, not a mismatch disposition, so it is loud on both surfaces —
    `drop_cause` (a closed three-value enum: this one, `no-intent`, and the
    keep-open lane's `stale-option`) is what tells it apart from the other drops,
    and the entry stays open for the next sweep. `max_bundles=1`
    keeps the nine occupying plan bundles from having to run. Ablation: delete
    the exhausted-bound `else` and this reddens — no drop record, no ATTENTION
    line, and the decision bundle silently re-takes `decision-dw-1`."""
    from bmad_loop import decisions
    from bmad_loop.sweep import DecisionOption

    ids = [f"DW-{n}" for n in range(1, 11)]
    write_ledger(project, {i: "open" for i in ids})
    decisions.record_pre_answer(
        project.project,
        "DW-1",
        DecisionOption(key="1", label="Widen", effect="build", intent="widen the field"),
        date="2026-06-12",
    )
    # `decision-dw-1` plus every one of `-2` ... `-9`: the whole bound, taken
    occupied = ["decision-dw-1"] + [f"decision-dw-1-{n}" for n in range(2, 10)]
    plan = triage_result(
        ids,
        bundles=[
            {"name": name, "dw_ids": [dw_id], "intent": f"plan intent {dw_id}"}
            for name, dw_id in zip(occupied, ids[1:], strict=True)
        ],
        decisions=[
            _decision(
                "DW-1",
                [
                    {"key": "1", "label": "Widen", "effect": "build", "intent": "fresh intent"},
                    {"key": "2", "label": "Keep", "effect": "keep-open"},
                ],
            )
        ],
    )
    engine, _ = make_sweep(
        project,
        [
            triage_effect(plan),
            bundle_dev_effect(project, "decision-dw-1", ["DW-2"]),
            bundle_review_effect(project, "decision-dw-1"),
        ],
        prompting=False,
        max_bundles=1,
    )
    summary = engine.run()
    assert not summary.paused

    assert _mismatches(engine) == []  # PRECONDITION: the AGREEING lane
    assert _records(engine, "sweep-bundle-name-deduped") == []  # the bound was exhausted
    [drop] = _records(engine, "sweep-decision-answer-dropped")
    assert drop["decision"] == "DW-1" and drop["drop_cause"] == "name-collision"
    attention = (engine.run_dir / "ATTENTION").read_text(encoding="utf-8")
    assert "recorded build decision discarded" in attention
    assert "could not be given a name unique" in attention
    # the one bundle that ran is the PLAN bundle; nothing aliased onto its task
    assert engine.state.tasks["dw-decision-dw-1"].dw_ids == ["DW-2"]
    assert ledger_entries(project)["DW-1"].open  # re-asked by the next sweep
    assert "DW-1" in decisions.load_pre_answers(project.project)  # not consumed


def test_a_name_collision_drop_is_not_revived_by_a_later_free_cycle(project):
    """The quarantine's OTHER drop lane. `test_a_dropped_decision_is_not_revived...`
    reaches it through `no-intent`; this row reaches it through `name-collision`,
    where the answer is perfectly buildable and only the names were taken. Cycle 2
    re-triages with a plan that authors NONE of the contested names, so the
    fallback is free again — the revival the quarantine has to refuse, since the
    operator has already been told this decision was discarded. `max_bundles=1`
    keeps the nine occupying plan bundles from having to run. Ablation: delete the
    `self._quarantine(self.state.sweep_dropped_decisions, decision.id)` on the
    name-collision lane alone
    and this reddens — cycle 2 mints a `dw2-decision-dw-1` task for an answer the
    run already gave up on."""
    from bmad_loop import decisions
    from bmad_loop.sweep import DecisionOption

    ids = [f"DW-{n}" for n in range(1, 11)]
    write_ledger(project, {i: "open" for i in ids})
    decisions.record_pre_answer(
        project.project,
        "DW-1",
        DecisionOption(key="1", label="Widen", effect="build", intent="widen the field"),
        date="2026-06-12",
    )
    decision = _decision(
        "DW-1",
        [
            {"key": "1", "label": "Widen", "effect": "build", "intent": "fresh intent"},
            {"key": "2", "label": "Keep", "effect": "keep-open"},
        ],
    )
    occupied = ["decision-dw-1"] + [f"decision-dw-1-{n}" for n in range(2, 10)]
    plan1 = triage_result(
        ids,
        bundles=[
            {"name": name, "dw_ids": [dw_id], "intent": f"plan intent {dw_id}"}
            for name, dw_id in zip(occupied, ids[1:], strict=True)
        ],
        decisions=[decision],
    )
    # cycle 2 authors no bundle at all, so every contested name is free again
    plan2 = triage_result(
        ["DW-1"] + ids[2:],
        skip=[{"id": i, "reason": "not this cycle"} for i in ids[2:]],
        decisions=[decision],
    )
    engine, _ = make_sweep(
        project,
        [
            triage_effect(plan1),
            bundle_dev_effect(project, "decision-dw-1", ["DW-2"]),
            bundle_review_effect(project, "decision-dw-1"),
            triage_effect(plan2),
        ],
        policy=repeat_policy(max_bundles=1),
        prompting=False,
    )
    summary = engine.run()
    assert not summary.paused

    # PRECONDITION: cycle 2 ran, and against an option that AGREES with the answer
    assert [r["cycle"] for r in _records(engine, "sweep-cycle")] == [2]
    assert _mismatches(engine) == []
    [drop] = _records(engine, "sweep-decision-answer-dropped")
    assert drop["decision"] == "DW-1" and drop["drop_cause"] == "name-collision"
    attention = (engine.run_dir / "ATTENTION").read_text(encoding="utf-8")
    assert attention.count("recorded build decision discarded") == 1
    assert "decision-dw-1 and every -2..-9 suffix are taken" in attention
    # cycle 2's free fallback revived nothing
    assert not any(k.startswith("dw2-") for k in engine.state.tasks)
    assert ledger_entries(project)["DW-1"].open


def test_a_dropped_decision_is_not_revived_by_a_later_agreeing_cycle(project):
    """Dropping an answer is an announcement to the operator, and
    `_materialize_bundles` deliberately leaves the run-level `answers` entry on
    disk (it is the human's recorded answer and stays auditable), so every later
    repeat cycle re-reads it. Cycle 3 here re-triages DW-1 back to an option that
    AGREES with the cycle-1 in-run answer cycle 2 dropped: without the
    run-scoped quarantine that answer revives and builds a bundle for a
    decision the operator was told had been dropped — and each cycle re-notifies.
    Ablation: delete the build lane's `state.sweep_dropped_decisions` guard and
    this reddens — a
    `dw3-decision-dw-1` task appears and a second drop record plus a second
    ATTENTION line are written."""
    write_ledger(project, {"DW-1": "open", "DW-2": "open", "DW-3": "open"})
    plan1 = triage_result(
        ["DW-1", "DW-2", "DW-3"],
        bundles=[{"name": "safe-fix", "dw_ids": ["DW-2"], "intent": "a"}],
        # DW-3 is held back for cycle 2 to bundle: that bundle is what makes
        # cycle 2 progress, so the repeat loop reaches cycle 3 at all
        skip=[{"id": "DW-3", "reason": "not this cycle"}],
        decisions=[
            _decision(
                "DW-1",
                [
                    {"key": "1", "label": "Widen", "effect": "build", "intent": "widen the field"},
                    {"key": "2", "label": "Keep", "effect": "keep-open"},
                ],
            )
        ],
    )
    plan2 = triage_result(
        ["DW-1", "DW-3"],
        bundles=[{"name": "later-fix", "dw_ids": ["DW-3"], "intent": "b"}],
        decisions=[
            _decision(
                "DW-1",
                [
                    # renumbered: key "1" now answers a different question
                    {"key": "1", "label": "Close as decayed", "effect": "close"},
                    {"key": "2", "label": "Keep", "effect": "keep-open"},
                ],
            )
        ],
    )
    plan3 = triage_result(
        ["DW-1"],
        decisions=[
            _decision(
                "DW-1",
                [
                    # cycle 3 re-authors key "1" back into agreement with the
                    # cycle-1 answer — the revival the quarantine has to refuse
                    {"key": "1", "label": "Widen", "effect": "build", "intent": "widen the field"},
                    {"key": "2", "label": "Keep", "effect": "keep-open"},
                ],
            )
        ],
    )
    engine, _ = make_sweep(
        project,
        [
            triage_effect(plan1),
            bundle_dev_effect(project, "safe-fix", ["DW-2"]),
            bundle_review_effect(project, "safe-fix"),
            triage_effect(plan2),
            bundle_dev_effect(project, "later-fix", ["DW-3"]),
            bundle_review_effect(project, "later-fix"),
            triage_effect(plan3),
        ],
        policy=repeat_policy(max_bundles=1),
        answers=["1"],
        prompting=True,
    )
    summary = engine.run()
    assert not summary.paused

    journal = journal_text(engine)
    # PRECONDITIONS: the in-run answer was recorded, cycle 1 truncated its bundle,
    # and all three cycles ran
    assert '"decision-answered"' in journal
    assert '"sweep-bundles-truncated"' in journal
    assert engine.state.tasks["dw2-later-fix"].phase == Phase.DONE
    assert [r["cycle"] for r in _records(engine, "sweep-cycle")] == [2, 3]
    # exactly one drop, one mismatch and one notification for the whole run
    assert len(_mismatches(engine)) == 1
    [drop] = _records(engine, "sweep-decision-answer-dropped")
    assert drop["decision"] == "DW-1" and drop["drop_cause"] == "no-intent"
    attention = (engine.run_dir / "ATTENTION").read_text(encoding="utf-8")
    assert attention.count("recorded build decision discarded") == 1
    # and cycle 3's agreeing option revived nothing
    assert not any(k.startswith("dw3-") for k in engine.state.tasks)
    assert ledger_entries(project)["DW-1"].open


def test_preanswered_keep_open_suppresses_prompt_and_persists(project):
    """A keep-open pre-answer is adopted (no skip, no re-prompt) and, since the
    entry stays open, the store keeps it for the next sweep too."""
    from bmad_loop import decisions
    from bmad_loop.sweep import DecisionOption

    write_ledger(project, {"DW-1": "open"})
    decisions.record_pre_answer(
        project.project,
        "DW-1",
        DecisionOption(key="2", label="Keep", effect="keep-open"),
        date="2026-06-12",
    )
    plan = triage_result(
        ["DW-1"],
        decisions=[
            _decision(
                "DW-1",
                [
                    {"key": "1", "label": "Build", "effect": "build", "intent": "x"},
                    {"key": "2", "label": "Keep", "effect": "keep-open"},
                ],
                recommendation="2",
            )
        ],
    )
    engine, adapter = make_sweep(project, [triage_effect(plan)], prompting=False)
    summary = engine.run()
    assert not summary.paused
    assert len(adapter.sessions) == 1  # triage only — no bundle, no prompt
    journal = journal_text(engine)
    assert '"decision-preanswered"' in journal
    assert "decision-skipped-unattended" not in journal
    assert ledger_entries(project)["DW-1"].open
    assert decisions.load_pre_answers(project.project)["DW-1"]["effect"] == "keep-open"


def test_preanswered_keep_open_whose_option_was_re_authored_stops_suppressing(project):
    """The keep-open drop's OUT-OF-BAND provenance, and the one the agreement check
    exists for: a pre-answer outlives the process that recorded it, so the triage it
    meets is always a later one — the row above is the same store meeting a triage
    that still agrees. `record_pre_answer` stores the chosen option's full semantics
    under fixed keys, so a lane that exempted answers carrying their own payload
    would wave exactly this provenance through unchecked. Cycle 1 re-authors the
    answered key "2" onto `close`, so the pre-answer is journaled, dropped and
    quarantined; cycle 2's bundle over DW-1 then runs instead of being skipped
    `human-chose-keep-open`, and closing the entry prunes the store.

    Ablation: exempt pre-answers from the agreement check — `if "intent" in answer:
    keep_open_ids.add(dw_id); continue` ahead of it, which selects this provenance
    exactly, since `record_pre_answer` is the only writer of that key — and this
    reddens: no mismatch, no drop, and `dw2-sneaky-fix` never becomes a task."""
    from bmad_loop import decisions
    from bmad_loop.sweep import DecisionOption

    write_ledger(project, {"DW-1": "open", "DW-2": "open"})
    decisions.record_pre_answer(
        project.project,
        "DW-1",
        DecisionOption(key="2", label="Keep", effect="keep-open"),
        date="2026-06-12",
    )
    plan1 = triage_result(
        ["DW-1", "DW-2"],
        bundles=[{"name": "safe-fix", "dw_ids": ["DW-2"], "intent": "a"}],
        decisions=[
            _decision(
                "DW-1",
                [
                    {"key": "1", "label": "Build", "effect": "build", "intent": "x"},
                    # key "2" answers a different question than the store's answer did
                    {"key": "2", "label": "Close as decayed", "effect": "close"},
                ],
            )
        ],
    )
    plan2 = triage_result(
        ["DW-1"], bundles=[{"name": "sneaky-fix", "dw_ids": ["DW-1"], "intent": "y"}]
    )
    engine, _ = make_sweep(
        project,
        [
            triage_effect(plan1),
            bundle_dev_effect(project, "safe-fix", ["DW-2"]),
            bundle_review_effect(project, "safe-fix"),
            triage_effect(plan2),
            bundle_dev_effect(project, "sneaky-fix", ["DW-1"]),
            bundle_review_effect(project, "sneaky-fix"),
        ],
        policy=repeat_policy(),
        prompting=False,
    )
    summary = engine.run()
    assert not summary.paused

    journal = journal_text(engine)
    # PRECONDITION: the answer came from the STORE, not from an in-run prompt
    assert '"decision-preanswered"' in journal
    assert '"decision-answered"' not in journal
    [record] = _mismatches(engine)
    assert record["option_effect"] == "close" and record["label_matched"] is False
    assert record["answer_effect"] == "keep-open"
    [drop] = _records(engine, "sweep-decision-answer-dropped")
    assert drop["decision"] == "DW-1" and drop["drop_cause"] == "stale-option"
    attention = (engine.run_dir / "ATTENTION").read_text(encoding="utf-8")
    assert attention.count("recorded keep-open decision discarded") == 1
    assert "human-chose-keep-open" not in journal
    assert engine.state.tasks["dw2-sneaky-fix"].phase == Phase.DONE
    # the unblocked bundle closed the entry, so the project store is pruned with it
    assert decisions.load_pre_answers(project.project) == {}


# ------------------------------- the two per-run `decisions.json` writes (cf. #363)
#
# `sweep.atomic_write_text_confined` (#593) is the binding the two `_decisions_phase`
# writes below share. The module's OTHER two writers — the `_ensure_migration` ledger
# restore and `_write_bundle_intent` — kept the plain `sweep.atomic_write_text`, so
# splitting the adoption narrowed what these patches can reach. The FILENAME filter
# and the delegate-otherwise arm stay anyway: a bare module-wide boom would make the
# "revert this site" ablation pass for the WRONG REASON — the run still crashes on
# "disk full", just raised from somewhere else — and the filter is what keeps that
# true if a third site in this module ever adopts the confined helper. Both plans
# carry ZERO bundles so the bundle-intent write is unreachable too.
#
# The filter is safe against the project-level pre-answer store, which is also named
# `decisions.json`: that goes through `decisions.atomic_write_text_confined`, a
# different module's binding, which these patches do not touch.


def test_seeded_decisions_write_failure_strands_nothing(project, monkeypatch):
    """The pre-answer-seeded write. Before the move to the helper this was a
    hand-rolled `decisions.json.tmp` + `atomic_replace`, and a failed replace left
    the temp behind.

    NOT #363's exposure, despite the shared helper and the shared filename. This
    `decisions.json` is `engine.run_dir / "decisions.json"` — under
    `.bmad-loop/runs/<id>/`, which `init` gitignores — so a stranded temp here was
    never untracked and never held `worktree_clean` False. #363's file is the
    project-level `.bmad-loop/decisions.json` (`decisions._write_store`), a
    different file one directory up. What this row pins is the helper's
    unlink-on-raise on a site that took it for the fsync and the unique temp name.

    The `decision-preanswered` journal line is a PRECONDITION, not decoration: it is
    appended before the write, so without it `assert not ... .exists()` would pass
    for the entirely different reason that the phase was never entered.

    Ablation A4: revert this site to `tmp.write_text(...)` + `atomic_replace` and
    this reddens on the assertions (not, as in the decisions/policy/tui-settings
    rows, on an AttributeError) — the `sweep.atomic_write_text` binding survives the
    revert because three other sites still use it. That is exactly why the stub
    filters by filename: a module-wide boom would keep this green through the
    revert, crashing on a write this test is not about."""
    from bmad_loop import decisions
    from bmad_loop.sweep import DecisionOption

    write_ledger(project, {"DW-1": "open"})
    decisions.record_pre_answer(
        project.project,
        "DW-1",
        DecisionOption(key="2", label="Keep", effect="keep-open"),
        date="2026-06-12",
    )
    plan = triage_result(  # keep-open only: no bundle, so no bundle-intent write
        ["DW-1"],
        decisions=[
            _decision(
                "DW-1",
                [
                    {"key": "1", "label": "Build", "effect": "build", "intent": "x"},
                    {"key": "2", "label": "Keep", "effect": "keep-open"},
                ],
                recommendation="2",
            )
        ],
    )
    engine, _ = make_sweep(project, [triage_effect(plan)], prompting=False)

    real = sweep_mod.atomic_write_text_confined

    def boom(path, text, *, confine_root, require_writable_target=False):
        if Path(path).name == "decisions.json":
            raise OSError("disk full")
        real(path, text, confine_root=confine_root, require_writable_target=require_writable_target)

    monkeypatch.setattr(sweep_mod, "atomic_write_text_confined", boom)
    summary = engine.run()

    assert summary.crashed and "disk full" in str(summary.crash_error)
    assert '"decision-preanswered"' in journal_text(engine)  # PRECONDITION: site reached
    assert not (engine.run_dir / "decisions.json").exists()
    assert list(engine.run_dir.glob("decisions*.tmp")) == []  # no stranded temp


def test_interactive_decision_write_failure_strands_nothing(project, monkeypatch):
    """The interactive write — the same hand-rolled shape as the seeded one above,
    on the same path, a few lines further down, and covered by the same correction
    in that docstring: this is the per-run `decisions.json`, not #363's
    project-level store.

    Two preconditions rather than one. `decision-pending` is appended before
    `prompter.ask`, so it proves the phase was entered; `decision-answered` is
    appended AFTER the write, so its ABSENCE places the raise between the ask and
    that append — i.e. exactly at the site under test, not somewhere upstream that
    would have skipped the write entirely.

    Ablation A7: revert this site and it reddens on the assertions, for the reason
    the seeded twin's docstring gives — the shared binding survives the revert, so
    only the filename filter makes the row grade this site."""
    write_ledger(project, {"DW-1": "open"})
    plan = triage_result(  # close-only: no bundle, so no bundle-intent write
        ["DW-1"],
        decisions=[
            _decision(
                "DW-1",
                [
                    {"key": "1", "label": "Close it", "effect": "close", "resolution": "moot"},
                    {"key": "2", "label": "Keep open", "effect": "keep-open"},
                ],
            )
        ],
    )
    engine, _ = make_sweep(project, [triage_effect(plan)], answers=["1"], prompting=True)

    real = sweep_mod.atomic_write_text_confined

    def boom(path, text, *, confine_root, require_writable_target=False):
        if Path(path).name == "decisions.json":
            raise OSError("disk full")
        real(path, text, confine_root=confine_root, require_writable_target=require_writable_target)

    monkeypatch.setattr(sweep_mod, "atomic_write_text_confined", boom)
    summary = engine.run()

    journal = journal_text(engine)
    assert summary.crashed and "disk full" in str(summary.crash_error)
    assert '"decision-pending"' in journal  # PRECONDITION: the prompt was reached
    assert '"decision-answered"' not in journal  # the raise landed AT the write
    assert not (engine.run_dir / "decisions.json").exists()
    assert list(engine.run_dir.glob("decisions*.tmp")) == []  # no stranded temp


def test_decisions_phase_keys_on_the_run_dir_project_not_workspace_root(project, tmp_path):
    """The supported `repo_root` override (`_bmad/bmm/config.yaml`, honoured only
    with `isolation = "none"` — worktree isolation refuses the divergence) puts
    `workspace.root` at the separate CODE repo while the per-run `decisions.json`
    and the project-level pre-answer store stay under the PROJECT. Keying this
    phase on `workspace.root` therefore made every pre-answer seed — and the
    first interactive answer — raise `UnconfinedWriteError`, and aimed the
    pre-answer READ at a store that does not exist. All three now key on the
    project that owns `run_dir` (`runs._project_of_run_dir`), which no workspace
    swap moves.

    The divergence is installed at the exact seam the override reaches:
    `workspace` is a plain attribute the bundle pipeline itself swaps, and
    `_decisions_phase` runs outside that pipeline by contract. The divergent
    root is a real empty git repo so the phase's ledger-commit tail stays a
    no-op rather than a spawn error.

    Ablations, each reddening its own assertion: point the pre-answer read back
    at `self.workspace.root` → the seeded answer is missing; point the seeded
    write's `confine_root` back → `UnconfinedWriteError` before any assertion;
    point the interactive write's back → the same raise at the DW-2 answer."""
    from bmad_loop import decisions
    from bmad_loop.workspace import Workspace

    write_ledger(project, {"DW-1": "open", "DW-2": "open"})
    decisions.record_pre_answer(
        project.project,
        "DW-1",
        DecisionOption(key="2", label="Keep", effect="keep-open"),
        date="2026-06-12",
    )
    options = [
        {"key": "1", "label": "Build", "effect": "build", "intent": "x"},
        {"key": "2", "label": "Keep", "effect": "keep-open"},
    ]
    rj = triage_result(
        ["DW-1", "DW-2"],
        decisions=[
            _decision("DW-1", options, recommendation="2"),
            _decision("DW-2", options, recommendation="2"),
        ],
    )
    plan, errors = validate_triage(rj, {"DW-1", "DW-2"})
    assert errors == []
    engine, _ = make_sweep(project, [], answers=["2"], prompting=True)
    elsewhere = tmp_path / "code-repo"
    elsewhere.mkdir()
    git(elsewhere, "init")
    engine.workspace = Workspace(root=elsewhere, paths=engine.workspace.paths)
    engine.run_dir.mkdir(parents=True, exist_ok=True)

    answers, closed = engine._decisions_phase(plan)

    assert answers["DW-1"]["effect"] == "keep-open"  # the READ found the project store
    assert answers["DW-2"]["key"] == "2"  # the interactive write landed too
    stored = json.loads((engine.run_dir / "decisions.json").read_text(encoding="utf-8"))
    assert set(stored) == {"DW-1", "DW-2"}
    assert closed == 0
    assert list(elsewhere.rglob("decisions*")) == []  # nothing leaked into the code repo


def test_prune_pre_answers_keys_on_the_run_dir_project_not_workspace_root(project, tmp_path):
    """`_prune_pre_answers` is the fourth workspace-rooted call in the family
    the test above pins (read, seeded write, interactive write): under the
    `repo_root` override it pruned against `workspace.root` — the separate CODE
    repo, whose store does not exist — so the prune silently dropped nothing
    and a consumed pre-answer survived in the PROJECT store, suppressing
    `pending_missed_decisions` for its id and standing ready to re-apply if the
    id ever returns to the open set. It now keys on `runs._project_of_run_dir`,
    the same root the load reads. The ledger read is deliberately untouched:
    `deferred_work` hangs off `implementation_artifacts`, which stays
    project-rooted under the override.

    Ablation: point the prune back at `self.workspace.root` → DW-1 survives in
    the project store and the pruned journal line never lands."""
    from bmad_loop import decisions
    from bmad_loop.workspace import Workspace

    write_ledger(project, {"DW-2": "open"})  # DW-1 consumed: absent from the open set
    for dw in ("DW-1", "DW-2"):
        decisions.record_pre_answer(
            project.project,
            dw,
            DecisionOption(key="2", label="Keep", effect="keep-open"),
            date="2026-06-12",
        )
    engine, _ = make_sweep(project, [])
    elsewhere = tmp_path / "code-repo"
    elsewhere.mkdir()
    git(elsewhere, "init")
    engine.workspace = Workspace(root=elsewhere, paths=engine.workspace.paths)
    engine.run_dir.mkdir(parents=True, exist_ok=True)

    engine._prune_pre_answers()

    # DW-1 dropped from the PROJECT store; the still-open keep-open answer kept
    assert set(decisions.load_pre_answers(project.project)) == {"DW-2"}
    assert '"decision-preanswers-pruned"' in journal_text(engine)
    assert list(elsewhere.rglob("decisions*")) == []  # the code repo was never the store


def test_max_bundles_truncation(project):
    write_ledger(project, {"DW-1": "open", "DW-2": "open", "DW-3": "open"})
    plan = triage_result(
        ["DW-1", "DW-2", "DW-3"],
        bundles=[
            {"name": "first-fix", "dw_ids": ["DW-1"], "intent": "a"},
            {"name": "second-fix", "dw_ids": ["DW-2"], "intent": "b"},
            {"name": "third-fix", "dw_ids": ["DW-3"], "intent": "c"},
        ],
    )
    policy = Policy(gates=GatesPolicy(mode="none"), notify=QUIET, sweep=SweepPolicy(max_bundles=1))
    engine, adapter = make_sweep(
        project,
        [
            triage_effect(plan),
            bundle_dev_effect(project, "first-fix", ["DW-1"]),
            bundle_review_effect(project, "first-fix"),
        ],
        policy=policy,
    )
    summary = engine.run()
    assert not summary.paused
    assert "sweep-bundles-truncated" in journal_text(engine)
    entries = ledger_entries(project)
    assert entries["DW-1"].status.startswith("done")
    assert entries["DW-2"].open and entries["DW-3"].open


def test_escalated_bundle_resume_skips_it_and_runs_rest(project):
    write_ledger(project, {"DW-1": "open", "DW-2": "open"})
    plan = triage_result(
        ["DW-1", "DW-2"],
        bundles=[
            {"name": "bad-fix", "dw_ids": ["DW-1"], "intent": "a"},
            {"name": "good-fix", "dw_ids": ["DW-2"], "intent": "b"},
        ],
    )

    def escalating_dev(spec):
        return SessionResult(
            status="completed",
            result_json={
                "workflow": "auto-dev",
                "escalations": [
                    {
                        "type": "bundle-item-blocked",
                        "severity": "CRITICAL",
                        "detail": "no",
                    }
                ],
            },
        )

    engine, _ = make_sweep(project, [triage_effect(plan), escalating_dev])
    summary = engine.run()
    assert summary.paused
    assert engine.state.tasks["dw-bad-fix"].phase == Phase.ESCALATED

    resumed, adapter = resume_sweep(
        project,
        engine,
        [
            bundle_dev_effect(project, "good-fix", ["DW-2"]),
            bundle_review_effect(project, "good-fix"),
        ],
    )
    summary = resumed.run()
    assert not summary.paused
    assert resumed.state.tasks["dw-good-fix"].phase == Phase.DONE
    # triage was NOT re-run: only the two bundle sessions
    assert len(adapter.sessions) == 2
    assert ledger_entries(project)["DW-1"].open  # escalated bundle untouched


# ------------------------------ intent-gap patch-restore re-drive (#75)


def _run_to_dev_escalation(project, policy=None):
    """Drive a one-bundle sweep until its dev session escalates on an intent gap.
    Returns the paused engine; the bundle task is ESCALATED with spec_file set and
    DW-1 still open (blocked spec is not synced done)."""
    write_ledger(project, {"DW-1": "open"})
    plan = triage_result(
        ["DW-1"], bundles=[{"name": "fix", "dw_ids": ["DW-1"], "intent": "resolve DW-1"}]
    )
    engine, _ = make_sweep(
        project,
        [triage_effect(plan), bundle_dev_escalates(project, "fix", ["DW-1"])],
        policy=policy,
    )
    summary = engine.run()
    assert summary.paused
    task = engine.state.tasks["dw-fix"]
    assert task.phase == Phase.ESCALATED
    assert task.spec_file  # latched by _record_dev_spec on the dev escalation
    assert ledger_entries(project)["DW-1"].open  # blocked spec not synced done
    return engine


def test_generic_bundle_prompt_restore_branch_points_at_spec(project):
    # T-A: the restore-aware prompt (Change A) — no run needed.
    engine, _ = make_sweep(project, [])
    spec = str(project.implementation_artifacts / "spec-dw-fix.md")
    task = StoryTask(
        story_key="dw-fix",
        epic=0,
        dw_ids=["DW-1"],
        bundle_file="/run/bundles/fix/intent.md",
        spec_file=spec,
        restore_patch="/run/artifacts/attempt-dw-fix.patch",
    )
    restore_prompt = engine._generic_bundle_prompt(task, None)
    assert "Resume review of the in-review spec" in restore_prompt
    assert spec in restore_prompt
    assert "Do NOT edit the deferred-work ledger" in restore_prompt
    # without a latched patch the fresh-implement prompt is unchanged
    task.restore_patch = None
    fresh_prompt = engine._generic_bundle_prompt(task, None)
    assert "Implement the deferred-work bundle" in fresh_prompt
    assert "Resume review of the in-review spec" not in fresh_prompt


def test_bundle_dev_prompt_has_no_board_clause_but_the_review_prompt_does(project):
    """A bundle has no sprint-status row, so the board-ownership clause the story dev
    prompt carries is not appended here — `SweepEngine` overrides `_dev_prompt`, and
    `_generic_bundle_prompt` never reaches the seam.

    The inherited REVIEW prompt does carry it, both halves, and that is a decision
    rather than an accident: a sweep runs inside a project whose sprint board exists
    and is just as revertible from a bundle session as from a story one, so unlike
    `StoriesEngine` — which has no board at all and empties the clause — `SweepEngine`
    never overrides `_review_prompt`. Note the deliberate asymmetry with
    `_operator_park_enabled`, which IS False here: the clause still names
    `awaiting-operator`, because the row it defends was written by an earlier *story*
    run, not by this bundle.

    Coverage gap, deliberate: this asserts on `_dev_prompt`'s return value, so moving
    the injection into `_run_session` would leave it green while bundle prompts
    silently gained the clause. The builder is the correct layer today."""
    engine, _ = make_sweep(project, [])
    task = StoryTask(
        story_key="dw-fix", epic=0, dw_ids=["DW-1"], bundle_file="/run/bundles/fix/intent.md"
    )
    assert engine._operator_park_enabled() is False

    dev = engine._dev_prompt(task, None)
    assert "sprint-status.yaml is owned by the orchestrator" not in dev
    assert "status: blocked and say why" not in dev

    task.spec_file = str(project.implementation_artifacts / "spec-dw-fix.md")
    review = engine._review_prompt(task)
    assert "sprint-status.yaml is owned by the orchestrator" in review
    assert "done or awaiting-operator" in review
    assert "status: blocked and say why" in review


def test_generic_bundle_prompt_spells_the_post_rename_primitive(project):
    """All three bundle legs (restore, fresh implement, repair) spell the dev
    primitive resolved off the dev adapter's skill tree, not a hardcoded name —
    upstream renamed it bmad-dev-auto -> bmad-build-auto (BMAD-METHOD #2651).

    Both setup lines are load-bearing: the `project` fixture installs no skills
    and `MockAdapter` carries no profile, so dropping either one resolves every
    leg through `dev_primitive_or_default`'s legacy fallback — which the sibling
    tests above already pin, and which would make this one pass for the wrong
    reason."""
    install_build_auto_skill(project.project, ".claude/skills")
    engine, adapter = make_sweep(project, [])
    attach_profile(adapter)  # claude -> .claude/skills, where the new name now lives
    spec = str(project.implementation_artifacts / "spec-dw-fix.md")
    task = StoryTask(
        story_key="dw-fix",
        epic=0,
        dw_ids=["DW-1"],
        bundle_file="/run/bundles/fix/intent.md",
        spec_file=spec,
        restore_patch="/run/artifacts/attempt-dw-fix.patch",
    )

    assert engine._generic_bundle_prompt(task, None).startswith(
        "/bmad-build-auto Resume review of the in-review spec"
    )
    task.restore_patch = None
    assert engine._generic_bundle_prompt(task, None).startswith(
        "/bmad-build-auto Implement the deferred-work bundle"
    )
    feedback = project.implementation_artifacts / "feedback.md"
    assert engine._generic_bundle_prompt(task, feedback).startswith(
        "/bmad-build-auto Resume the autonomous dev session"
    )


def test_sweep_bundle_restore_redrive_reaches_done_and_clears_latch(project, monkeypatch):
    # T-C + T-D: rearm with a restore patch, resume, land done; assert the dispatched
    # prompt pointed at the in-review spec, the patch apply seam fired, the dw id
    # closed, and both latches cleared on commit (inherited Engine._commit).
    monkeypatch.setattr(verify, "apply_patch", lambda repo, patch: None)
    engine = _run_to_dev_escalation(project)
    patch = project.implementation_artifacts / "attempt-dw-fix.patch"
    patch.parent.mkdir(parents=True, exist_ok=True)
    patch.write_text("dummy\n")

    runs.rearm_escalation(
        engine.run_dir,
        "dw-fix",
        restore_patch=str(patch),
        isolated_redrive=False,
        resolution_recorded=True,
    )

    resumed, adapter = resume_sweep(
        project,
        engine,
        [bundle_dev_effect(project, "fix", ["DW-1"]), bundle_review_effect(project, "fix")],
    )
    summary = resumed.run()

    assert not summary.paused
    task = resumed.state.tasks["dw-fix"]
    assert task.phase == Phase.DONE
    # Change A: the re-drive dev session was pointed at the in-review spec
    spec = str(project.implementation_artifacts / "spec-dw-fix.md")
    assert "Resume review of the in-review spec" in adapter.sessions[0].prompt
    assert spec in adapter.sessions[0].prompt
    # the restore apply seam fired
    assert "attempt-restored" in journal_text(resumed)
    # latches cleared on commit
    assert task.restore_patch is None
    assert task.resolved_redrive is False
    # the deferred-work id was closed
    assert ledger_entries(project)["DW-1"].status.startswith("done")


def test_sweep_restore_redrive_exhaustion_pauses_not_defers(project, monkeypatch):
    # T-B (restore): a non-critical exhaustion mid-restore-re-drive must PAUSE
    # (re-escalate for the human), not silently DEFER the resolved escalation.
    monkeypatch.setattr(verify, "apply_patch", lambda repo, patch: None)
    policy = Policy(
        gates=GatesPolicy(mode="none"),
        notify=QUIET,
        scm=ScmPolicy(rollback_on_failure=True),
        limits=LimitsPolicy(max_dev_attempts=1),
    )
    engine = _run_to_dev_escalation(project, policy=policy)
    patch = project.implementation_artifacts / "attempt-dw-fix.patch"
    patch.parent.mkdir(parents=True, exist_ok=True)
    patch.write_text("dummy\n")
    runs.rearm_escalation(
        engine.run_dir,
        "dw-fix",
        restore_patch=str(patch),
        isolated_redrive=False,
        resolution_recorded=True,
    )

    resumed, _ = resume_sweep(project, engine, [lambda spec: SessionResult(status="died")])
    summary = resumed.run()

    assert summary.paused
    assert resumed.state.tasks["dw-fix"].phase == Phase.ESCALATED


def test_sweep_from_scratch_redrive_exhaustion_pauses_not_defers(project):
    # T-B (from-scratch twin): the resolved_redrive latch also protects a plain
    # `resolve` re-drive (no --restore-patch) — the pre-existing sweep defer bug
    # Change B closes. Without the fix this exhaustion would DEFER, not PAUSE.
    policy = Policy(
        gates=GatesPolicy(mode="none"),
        notify=QUIET,
        scm=ScmPolicy(rollback_on_failure=True),
        limits=LimitsPolicy(max_dev_attempts=1),
    )
    engine = _run_to_dev_escalation(project, policy=policy)
    runs.rearm_escalation(
        engine.run_dir, "dw-fix", isolated_redrive=False, resolution_recorded=True
    )  # from-scratch, no restore

    resumed, _ = resume_sweep(project, engine, [lambda spec: SessionResult(status="died")])
    summary = resumed.run()

    assert summary.paused
    assert resumed.state.tasks["dw-fix"].phase == Phase.ESCALATED


# ----------------------------------------------------------- repeat cycles


def repeat_policy(**kw):
    return Policy(
        gates=GatesPolicy(mode="none"), notify=QUIET, sweep=SweepPolicy(repeat=True, **kw)
    )


def appending_dev(project, inner, dw_id):
    """Wrap a bundle dev effect so the session also appends a new open ledger
    entry — the 'sweep generated new deferred work' scenario."""

    def effect(spec):
        result = inner(spec)
        ledger = project.deferred_work
        ledger.write_text(
            ledger.read_text(encoding="utf-8")
            + f"\n### {dw_id}: item {dw_id}\n\norigin: test, 2026-06-11\n"
            f"location: src.txt:1\nreason: follow-up from bundle.\nstatus: open\n",
            encoding="utf-8",
        )
        return result

    return effect


def test_repeat_off_is_single_cycle(project):
    write_ledger(project, {"DW-1": "open"})
    plan = triage_result(
        ["DW-1"], bundles=[{"name": "one-fix", "dw_ids": ["DW-1"], "intent": "fix"}]
    )
    engine, adapter = make_sweep(
        project,
        [
            triage_effect(plan),
            appending_dev(project, bundle_dev_effect(project, "one-fix", ["DW-1"]), "DW-2"),
            bundle_review_effect(project, "one-fix"),
        ],
    )
    summary = engine.run()
    assert not summary.paused
    assert len(adapter.sessions) == 3  # no second triage
    journal = journal_text(engine)
    assert "sweep-cycle" not in journal and "sweep-repeat-done" not in journal
    assert ledger_entries(project)["DW-2"].open  # waits for the next sweep


def test_repeat_two_cycles_then_no_open(project):
    write_ledger(project, {"DW-1": "open"})
    plan1 = triage_result(
        ["DW-1"], bundles=[{"name": "first-fix", "dw_ids": ["DW-1"], "intent": "a"}]
    )
    plan2 = triage_result(
        ["DW-2"], bundles=[{"name": "follow-up", "dw_ids": ["DW-2"], "intent": "b"}]
    )
    engine, adapter = make_sweep(
        project,
        [
            triage_effect(plan1),
            appending_dev(project, bundle_dev_effect(project, "first-fix", ["DW-1"]), "DW-2"),
            bundle_review_effect(project, "first-fix"),
            triage_effect(plan2),
            bundle_dev_effect(project, "follow-up", ["DW-2"]),
            bundle_review_effect(project, "follow-up"),
        ],
        policy=repeat_policy(),
    )
    summary = engine.run()
    assert not summary.paused
    tasks = engine.state.tasks
    assert tasks["sweep-triage"].phase == Phase.DONE
    assert tasks["dw-first-fix"].phase == Phase.DONE
    assert tasks["sweep-triage-2"].phase == Phase.DONE
    assert tasks["dw2-follow-up"].phase == Phase.DONE
    journal = journal_text(engine)
    assert "sweep-cycle" in journal
    assert "sweep-repeat-done" in journal and "no-open" in journal
    entries = ledger_entries(project)
    assert entries["DW-1"].status.startswith("done")
    assert entries["DW-2"].status.startswith("done")
    assert worktree_clean(project.project)
    # cycle-2 dev got the cycle-scoped intent file
    intent_path = re.findall(r"`([^`]*)`", adapter.sessions[4].prompt)[0]
    assert "c2-follow-up" in intent_path


def test_repeat_stops_on_no_progress(project):
    write_ledger(project, {"DW-1": "open"})
    plan1 = triage_result(
        ["DW-1"], bundles=[{"name": "first-fix", "dw_ids": ["DW-1"], "intent": "a"}]
    )
    plan2 = triage_result(["DW-2"], blocked=[{"id": "DW-2", "blocker": "story 9-9"}])
    engine, adapter = make_sweep(
        project,
        [
            triage_effect(plan1),
            appending_dev(project, bundle_dev_effect(project, "first-fix", ["DW-1"]), "DW-2"),
            bundle_review_effect(project, "first-fix"),
            triage_effect(plan2),
        ],
        policy=repeat_policy(),
    )
    summary = engine.run()
    assert not summary.paused
    assert len(adapter.sessions) == 4  # the cycle-2 triage confirmed nothing addressable
    assert "no-progress" in journal_text(engine)
    assert ledger_entries(project)["DW-2"].open


def test_repeat_max_cycles_cap(project):
    write_ledger(project, {"DW-1": "open"})
    plan1 = triage_result(
        ["DW-1"], bundles=[{"name": "fix-one", "dw_ids": ["DW-1"], "intent": "a"}]
    )
    plan2 = triage_result(
        ["DW-2"], bundles=[{"name": "fix-two", "dw_ids": ["DW-2"], "intent": "b"}]
    )
    engine, adapter = make_sweep(
        project,
        [
            triage_effect(plan1),
            appending_dev(project, bundle_dev_effect(project, "fix-one", ["DW-1"]), "DW-2"),
            bundle_review_effect(project, "fix-one"),
            triage_effect(plan2),
            appending_dev(project, bundle_dev_effect(project, "fix-two", ["DW-2"]), "DW-3"),
            bundle_review_effect(project, "fix-two"),
        ],
        policy=repeat_policy(max_cycles=2),
    )
    summary = engine.run()
    assert not summary.paused
    assert len(adapter.sessions) == 6  # no cycle-3 triage despite DW-3 open
    assert "max-cycles" in journal_text(engine)
    assert ledger_entries(project)["DW-3"].open


def test_repeat_failed_bundle_not_rebuilt(project):
    """A bundle that deferred in cycle 1 must not be re-materialized when a
    later triage re-proposes its ids — that would loop until max_cycles."""
    write_ledger(project, {"DW-1": "open", "DW-2": "open"})
    plan1 = triage_result(
        ["DW-1", "DW-2"],
        bundles=[
            {"name": "bad-fix", "dw_ids": ["DW-1"], "intent": "a"},
            {"name": "good-fix", "dw_ids": ["DW-2"], "intent": "b"},
        ],
    )
    plan2 = triage_result(
        ["DW-1"], bundles=[{"name": "bad-fix-again", "dw_ids": ["DW-1"], "intent": "a2"}]
    )
    policy = Policy(
        gates=GatesPolicy(mode="none"),
        notify=QUIET,
        sweep=SweepPolicy(repeat=True),
        limits=LimitsPolicy(max_review_cycles=1, max_dev_attempts=1),
        scm=ScmPolicy(rollback_on_failure=True),  # exercise defer-and-continue
    )
    engine, adapter = make_sweep(
        project,
        [
            triage_effect(plan1),
            # bad-fix: spec never reaches in-review -> dev verify fails -> deferred
            lambda spec: SessionResult(
                status="completed", result_json={"workflow": "auto-dev", "escalations": []}
            ),
            bundle_dev_effect(project, "good-fix", ["DW-2"]),
            bundle_review_effect(project, "good-fix"),
            triage_effect(plan2),
        ],
        policy=policy,
    )
    summary = engine.run()
    assert not summary.paused
    assert engine.state.tasks["dw-bad-fix"].phase == Phase.DEFERRED
    assert "sweep-bundle-skipped" in journal_text(engine)
    assert not any(k.startswith("dw2-") for k in engine.state.tasks)
    assert "no-progress" in journal_text(engine)
    assert ledger_entries(project)["DW-1"].open


def test_repeat_keep_open_answer_blocks_rebundle(project):
    write_ledger(project, {"DW-1": "open", "DW-2": "open"})
    decision = {
        "id": "DW-1",
        "question": "build it?",
        "context": "",
        "options": [
            {"key": "1", "label": "Build", "effect": "build", "intent": "x"},
            {"key": "2", "label": "Keep open", "effect": "keep-open"},
        ],
        "recommendation": "2",
    }
    plan1 = triage_result(
        ["DW-1", "DW-2"],
        bundles=[{"name": "safe-fix", "dw_ids": ["DW-2"], "intent": "fix"}],
        decisions=[decision],
    )
    # cycle 2: triage tries to bundle the kept-open entry directly
    plan2 = triage_result(
        ["DW-1"], bundles=[{"name": "sneaky-fix", "dw_ids": ["DW-1"], "intent": "y"}]
    )
    engine, adapter = make_sweep(
        project,
        [
            triage_effect(plan1),
            bundle_dev_effect(project, "safe-fix", ["DW-2"]),
            bundle_review_effect(project, "safe-fix"),
            triage_effect(plan2),
        ],
        policy=repeat_policy(),
        answers=["2"],
        prompting=True,
    )
    summary = engine.run()
    assert not summary.paused
    journal = journal_text(engine)
    assert "sweep-bundle-skipped" in journal and "human-chose-keep-open" in journal
    assert not any("sneaky-fix" in k for k in engine.state.tasks)
    assert ledger_entries(project)["DW-1"].open


# ------------------- keep-open answers meet a re-authored option (DW-123)
#
# The row above is the shape where the id has NO decision in the cycle that
# bundles it — nothing to disagree with, so the stored answer is the only record
# of the human's choice and it stands. The four rows below are the shapes where a
# later cycle DOES re-ask the id, which is where a key resolving to a re-authored
# option can turn a stale answer into a `human-chose-keep-open` skip of work the
# human never protected. All four are cross-cycle by necessity: `validate_triage`
# lets one id be claimed by exactly one category, so no single plan can put an id
# in `decisions` and in a bundle at once — the disagreement is detected in the
# cycle that re-asks and its consequence lands in a later one.


def test_keep_open_answer_agreeing_with_a_re_authored_option_still_blocks_rebundle(project):
    """The direction the fix must NOT change. Cycle 2 re-asks DW-1 with the same
    option the cycle-1 answer chose, so the agreement test passes and the answer
    keeps its authority: cycle 3's bundle over DW-1 is still skipped
    `human-chose-keep-open`, with no mismatch and no drop anywhere in the run.
    Ablation: replace the keep arm's `self._agreeing_option(...) is not None` with
    `False` — "a decision exists, so distrust the answer" — and this reddens on
    both negative asserts: the drop fires in cycle 2 and `dw3-sneaky-fix` runs."""
    write_ledger(project, {"DW-1": "open", "DW-2": "open", "DW-3": "open"})
    keep_open = [
        {"key": "1", "label": "Build", "effect": "build", "intent": "x"},
        {"key": "2", "label": "Keep", "effect": "keep-open"},
    ]
    plan1 = triage_result(
        ["DW-1", "DW-2", "DW-3"],
        bundles=[{"name": "safe-fix", "dw_ids": ["DW-2"], "intent": "a"}],
        # held back so cycle 2 has work of its own and the repeat loop reaches 3
        skip=[{"id": "DW-3", "reason": "not this cycle"}],
        decisions=[_decision("DW-1", keep_open)],
    )
    plan2 = triage_result(
        ["DW-1", "DW-3"],
        bundles=[{"name": "later-fix", "dw_ids": ["DW-3"], "intent": "b"}],
        # re-asked, and the answered key still spells the answered option
        decisions=[_decision("DW-1", keep_open)],
    )
    plan3 = triage_result(
        ["DW-1"], bundles=[{"name": "sneaky-fix", "dw_ids": ["DW-1"], "intent": "y"}]
    )
    engine, _ = make_sweep(
        project,
        [
            triage_effect(plan1),
            bundle_dev_effect(project, "safe-fix", ["DW-2"]),
            bundle_review_effect(project, "safe-fix"),
            triage_effect(plan2),
            bundle_dev_effect(project, "later-fix", ["DW-3"]),
            bundle_review_effect(project, "later-fix"),
            triage_effect(plan3),
        ],
        policy=repeat_policy(),
        answers=["2"],
        prompting=True,
    )
    summary = engine.run()
    assert not summary.paused

    journal = journal_text(engine)
    # PRECONDITIONS: the answer was recorded and all three cycles ran
    assert '"decision-answered"' in journal
    assert [r["cycle"] for r in _records(engine, "sweep-cycle")] == [2, 3]
    assert _mismatches(engine) == []
    assert _records(engine, "sweep-decision-answer-dropped") == []
    assert "human-chose-keep-open" in journal
    assert not any("sneaky-fix" in k for k in engine.state.tasks)
    assert ledger_entries(project)["DW-1"].open


def test_keep_open_answer_whose_option_was_re_authored_stops_suppressing_bundles(project):
    """The bug itself. Cycle 2 renumbers key "2" off `Keep`/`keep-open` and onto
    `Close as decayed`/`close`, so the stored keep-open answer now names a question
    the human never answered. Cycle 2 contains only that drop: it must count as
    progress by itself so cycle 3 can bundle DW-1 instead of stopping at
    `no-progress`. The recorded answer is left alone on disk: it is the human's,
    and it stays auditable.
    Ablation: remove `answer_dropped` from `_cycle`'s progress predicate
    and this reddens because cycle 3 never runs and `dw3-sneaky-fix` is absent.
    Restoring the one-line keep-open comprehension also reddens the mismatch,
    drop and ATTENTION assertions."""
    write_ledger(project, {"DW-1": "open", "DW-2": "open"})
    plan1 = triage_result(
        ["DW-1", "DW-2"],
        bundles=[{"name": "safe-fix", "dw_ids": ["DW-2"], "intent": "a"}],
        decisions=[
            _decision(
                "DW-1",
                [
                    {"key": "1", "label": "Build", "effect": "build", "intent": "x"},
                    {"key": "2", "label": "Keep", "effect": "keep-open"},
                ],
            )
        ],
    )
    plan2 = triage_result(
        ["DW-1"],
        decisions=[
            _decision(
                "DW-1",
                [
                    {"key": "1", "label": "Build", "effect": "build", "intent": "x"},
                    # key "2" now answers a different question
                    {"key": "2", "label": "Close as decayed", "effect": "close"},
                ],
            )
        ],
    )
    plan3 = triage_result(
        ["DW-1"], bundles=[{"name": "sneaky-fix", "dw_ids": ["DW-1"], "intent": "y"}]
    )
    engine, _ = make_sweep(
        project,
        [
            triage_effect(plan1),
            bundle_dev_effect(project, "safe-fix", ["DW-2"]),
            bundle_review_effect(project, "safe-fix"),
            triage_effect(plan2),
            triage_effect(plan3),
            bundle_dev_effect(project, "sneaky-fix", ["DW-1"]),
            bundle_review_effect(project, "sneaky-fix"),
        ],
        policy=repeat_policy(),
        answers=["2"],
        prompting=True,
    )
    summary = engine.run()
    assert not summary.paused

    journal = journal_text(engine)
    assert '"decision-answered"' in journal  # PRECONDITION: the answer was recorded
    assert [r["cycle"] for r in _records(engine, "sweep-cycle")] == [2, 3]
    [record] = _mismatches(engine)
    assert record["option_effect"] == "close" and record["label_matched"] is False
    # the field that says WHICH lane wrote the record — the stored answer's effect
    assert record["answer_effect"] == "keep-open"
    [drop] = _records(engine, "sweep-decision-answer-dropped")
    assert drop["decision"] == "DW-1" and drop["drop_cause"] == "stale-option"
    attention = (engine.run_dir / "ATTENTION").read_text(encoding="utf-8")
    assert attention.count("recorded keep-open decision discarded") == 1
    assert "the option it answered (2) has been re-authored since" in attention
    assert "is gone from this cycle's triage" not in attention
    # the bundle the stale answer used to skip ran instead
    assert "human-chose-keep-open" not in journal
    assert engine.state.tasks["dw3-sneaky-fix"].phase == Phase.DONE
    # nothing on disk was rewritten: the human's answer stays as recorded
    stored = json.loads((engine.run_dir / "decisions.json").read_text(encoding="utf-8"))
    assert stored["DW-1"]["key"] == "2" and stored["DW-1"]["effect"] == "keep-open"


def test_keep_open_answer_whose_option_vanished_is_dropped_without_a_mismatch(project):
    """The other stale shape, and why the third `drop_cause` is `stale-option`
    rather than `option-mismatch`: cycle 2 DROPS key "2" instead of renumbering it,
    so nothing resolves and there is no option to describe. No
    `sweep-decision-option-mismatch` is written — but a keep-open answer carries
    nothing beyond the option it named, so the drop lane still fires and the
    operator is still told, and the drop alone advances repeat mode so a later
    valid triage cycle can bundle the id.
    Ablation A: move the mismatch append above the helper's `option is None` early
    return and the empty-mismatch assert reddens. Ablation B: restore the one-line
    `keep_open_ids` comprehension and both the drop record and the ATTENTION assert
    redden while the empty-mismatch assert stays green. Ablation C: count the drop
    as progress only when the answered key still resolves and cycle 3 never runs."""
    write_ledger(project, {"DW-1": "open", "DW-2": "open"})
    plan1 = triage_result(
        ["DW-1", "DW-2"],
        bundles=[{"name": "safe-fix", "dw_ids": ["DW-2"], "intent": "a"}],
        decisions=[
            _decision(
                "DW-1",
                [
                    {"key": "1", "label": "Build", "effect": "build", "intent": "x"},
                    {"key": "2", "label": "Keep", "effect": "keep-open"},
                ],
            )
        ],
    )
    plan2 = triage_result(
        ["DW-1"],
        decisions=[
            _decision(
                "DW-1",
                # a decision needs two options; the answered key "2" is not one
                [
                    {"key": "1", "label": "Build", "effect": "build", "intent": "x"},
                    {"key": "3", "label": "Keep", "effect": "keep-open"},
                ],
            )
        ],
    )
    plan3 = triage_result(
        ["DW-1"], bundles=[{"name": "sneaky-fix", "dw_ids": ["DW-1"], "intent": "y"}]
    )
    engine, _ = make_sweep(
        project,
        [
            triage_effect(plan1),
            bundle_dev_effect(project, "safe-fix", ["DW-2"]),
            bundle_review_effect(project, "safe-fix"),
            triage_effect(plan2),
            triage_effect(plan3),
            bundle_dev_effect(project, "sneaky-fix", ["DW-1"]),
            bundle_review_effect(project, "sneaky-fix"),
        ],
        policy=repeat_policy(),
        answers=["2"],
        prompting=True,
    )
    summary = engine.run()
    assert not summary.paused

    assert '"decision-answered"' in journal_text(engine)  # PRECONDITION
    assert _mismatches(engine) == []  # no option resolved, so nothing to describe
    [drop] = _records(engine, "sweep-decision-answer-dropped")
    assert drop["decision"] == "DW-1" and drop["drop_cause"] == "stale-option"
    attention = (engine.run_dir / "ATTENTION").read_text(encoding="utf-8")
    assert attention.count("recorded keep-open decision discarded") == 1
    assert "the option it answered (2) is gone from this cycle's triage" in attention
    assert "has been re-authored since" not in attention
    assert [r["cycle"] for r in _records(engine, "sweep-cycle")] == [2, 3]
    assert engine.state.tasks["dw3-sneaky-fix"].phase == Phase.DONE
    # the drop does not rewrite the recorded answer; the later bundle closes the id
    stored = json.loads((engine.run_dir / "decisions.json").read_text(encoding="utf-8"))
    assert stored["DW-1"]["key"] == "2" and stored["DW-1"]["effect"] == "keep-open"
    assert ledger_entries(project)["DW-1"].status.startswith("done")


def test_a_dropped_keep_open_answer_is_not_revived_by_a_later_agreeing_cycle(project):
    """The keep-open lane's half of the quarantine the build drops already had.
    Cycle 2 drops the answer (renumbered option); cycle 3 re-authors key "2" back
    into agreement — the shape that would otherwise hand a decision the operator
    was told had been dropped its authority back, silently. It stays dropped and
    silent: one mismatch, one drop record and one ATTENTION line for the whole
    run, and cycle 4's bundle over DW-1 runs rather than being skipped.
    Ablation: delete the keep-open pass's `if dw_id in
    self.state.sweep_dropped_decisions`
    guard and this reddens — cycle 3's agreeing option puts DW-1 back into
    `keep_open_ids` and `dw4-sneaky-fix` is skipped `human-chose-keep-open`."""
    write_ledger(project, {"DW-1": "open", "DW-2": "open", "DW-3": "open", "DW-4": "open"})
    keep_open = [
        {"key": "1", "label": "Build", "effect": "build", "intent": "x"},
        {"key": "2", "label": "Keep", "effect": "keep-open"},
    ]
    plan1 = triage_result(
        ["DW-1", "DW-2", "DW-3", "DW-4"],
        bundles=[{"name": "safe-fix", "dw_ids": ["DW-2"], "intent": "a"}],
        # DW-3 and DW-4 are held back so cycles 2 and 3 each make progress of
        # their own and the repeat loop reaches cycle 4
        skip=[
            {"id": "DW-3", "reason": "not this cycle"},
            {"id": "DW-4", "reason": "not this cycle"},
        ],
        decisions=[_decision("DW-1", keep_open)],
    )
    plan2 = triage_result(
        ["DW-1", "DW-3", "DW-4"],
        bundles=[{"name": "later-fix", "dw_ids": ["DW-3"], "intent": "b"}],
        skip=[{"id": "DW-4", "reason": "not this cycle"}],
        decisions=[
            _decision(
                "DW-1",
                [
                    {"key": "1", "label": "Build", "effect": "build", "intent": "x"},
                    {"key": "2", "label": "Close as decayed", "effect": "close"},
                ],
            )
        ],
    )
    plan3 = triage_result(
        ["DW-1", "DW-4"],
        bundles=[{"name": "third-fix", "dw_ids": ["DW-4"], "intent": "c"}],
        # back in agreement with the cycle-1 answer: the revival to refuse
        decisions=[_decision("DW-1", keep_open)],
    )
    plan4 = triage_result(
        ["DW-1"], bundles=[{"name": "sneaky-fix", "dw_ids": ["DW-1"], "intent": "y"}]
    )
    engine, _ = make_sweep(
        project,
        [
            triage_effect(plan1),
            bundle_dev_effect(project, "safe-fix", ["DW-2"]),
            bundle_review_effect(project, "safe-fix"),
            triage_effect(plan2),
            bundle_dev_effect(project, "later-fix", ["DW-3"]),
            bundle_review_effect(project, "later-fix"),
            triage_effect(plan3),
            bundle_dev_effect(project, "third-fix", ["DW-4"]),
            bundle_review_effect(project, "third-fix"),
            triage_effect(plan4),
            bundle_dev_effect(project, "sneaky-fix", ["DW-1"]),
            bundle_review_effect(project, "sneaky-fix"),
        ],
        policy=repeat_policy(),
        answers=["2"],
        prompting=True,
    )
    summary = engine.run()
    assert not summary.paused

    # PRECONDITIONS: the answer was recorded and all four cycles ran
    assert '"decision-answered"' in journal_text(engine)
    assert [r["cycle"] for r in _records(engine, "sweep-cycle")] == [2, 3, 4]
    # exactly one mismatch, one drop and one notification for the whole process
    assert len(_mismatches(engine)) == 1
    [drop] = _records(engine, "sweep-decision-answer-dropped")
    assert drop["decision"] == "DW-1" and drop["drop_cause"] == "stale-option"
    attention = (engine.run_dir / "ATTENTION").read_text(encoding="utf-8")
    assert attention.count("recorded keep-open decision discarded") == 1
    # and cycle 3's agreeing option did not restore the answer's authority
    assert "human-chose-keep-open" not in journal_text(engine)
    assert engine.state.tasks["dw4-sneaky-fix"].phase == Phase.DONE


_STALE_KEEP_OPEN_ANSWER = {"key": "2", "label": "Keep", "effect": "keep-open"}


def _stale_keep_open_plan():
    """A cycle that re-asks DW-1 with key "2" renumbered off `Keep`/`keep-open`
    and onto `Close as decayed`/`close` — the disagreement that drops a stored
    keep-open answer."""
    return TriagePlan(
        open_ids=frozenset({"DW-1"}),
        decisions=(
            Decision(
                id="DW-1",
                question="q",
                context="",
                options=(
                    DecisionOption(key="1", label="Build", effect="build", intent="x"),
                    DecisionOption(key="2", label="Close as decayed", effect="close"),
                ),
                recommendation="1",
            ),
        ),
    )


def _drop_keep_open_answer_and_persist(project):
    """Engine 1: drop DW-1's stale keep-open answer, leaving the disposition on
    the persisted `state.json` for a resume to read back."""
    write_ledger(project, {"DW-1": "open"})
    engine, _ = make_sweep(project, [])
    engine.run_dir.mkdir(parents=True, exist_ok=True)
    (engine.run_dir / "decisions.json").write_text(
        json.dumps({"DW-1": _STALE_KEEP_OPEN_ANSWER}, indent=2), encoding="utf-8"
    )
    _, dropped = engine._materialize_bundles(
        _stale_keep_open_plan(), {"DW-1": _STALE_KEEP_OPEN_ANSWER}
    )
    # PRECONDITION: the drop happened, and `_quarantine` persisted it ITSELF —
    # no `save_state` call here, because durability at the mutation is the point
    assert dropped and engine.state.sweep_dropped_decisions == ["DW-1"]
    assert load_state(engine.run_dir).sweep_dropped_decisions == ["DW-1"]
    return engine


def test_a_dropped_decision_stays_quarantined_across_a_resume(project):
    """DW-124. The quarantine is the RUN's disposition, not one engine process's:
    it rides `state.json`, so rebuilding through the resume helper reads it back
    and the stale option is not re-evaluated. Before this it was `__init__` set
    state, so a resume started empty, re-read the unchanged run-local answer and
    re-announced a drop the operator had already been told about — a second
    mismatch, a second drop row and a second ATTENTION line for one decision.
    One of each across BOTH engines is the whole claim.

    Ablation: drop `sweep_dropped_decisions` from `RunState.to_dict` (or make
    `_quarantine` skip its `_save()`) and this reddens with two of each."""
    engine = _drop_keep_open_answer_and_persist(project)

    resumed, _ = resume_sweep(project, engine, [])
    assert resumed.state.sweep_dropped_decisions == ["DW-1"]
    resumed_answers, _ = resumed._decisions_phase(_stale_keep_open_plan())
    _, dropped_again = resumed._materialize_bundles(_stale_keep_open_plan(), resumed_answers)

    assert not dropped_again  # already announced; not progress a second time
    assert len(_mismatches(resumed)) == 1
    assert len(_records(resumed, "sweep-decision-answer-dropped")) == 1
    attention = (resumed.run_dir / "ATTENTION").read_text(encoding="utf-8")
    assert attention.count("recorded keep-open decision discarded") == 1
    # the human's answer is untouched by any of it — only the disposition persisted
    stored = json.loads((resumed.run_dir / "decisions.json").read_text(encoding="utf-8"))
    assert stored == {"DW-1": _STALE_KEEP_OPEN_ANSWER}


@pytest.mark.parametrize("drop_cause", ["no-intent", "name-collision"])
def test_a_build_answer_drop_stays_quarantined_across_a_resume(project, drop_cause):
    """Both build-answer drop sites publish the same durable disposition as the
    keep-open site: resume neither announces the drop again nor lets an answer
    regain authority when a later plan makes it usable.

    Ablation: replace either build lane's `_quarantine(...)` with an in-memory
    append and that parameter reddens after `resume_sweep` reloads `state.json`.
    """
    write_ledger(project, {"DW-1": "open"})
    agreeing = Decision(
        id="DW-1",
        question="q",
        context="",
        options=(DecisionOption(key="1", label="Widen", effect="build", intent="x"),),
        recommendation="1",
    )
    answer = {"key": "1", "label": "Widen", "effect": "build"}
    if drop_cause == "no-intent":
        dropping = Decision(
            id="DW-1",
            question="q",
            context="",
            options=(DecisionOption(key="1", label="Close", effect="close"),),
            recommendation="1",
        )
        dropping_plan = TriagePlan(open_ids=frozenset({"DW-1"}), decisions=(dropping,))
    else:
        answer["intent"] = "x"
        occupied = ["decision-dw-1"] + [f"decision-dw-1-{n}" for n in range(2, 10)]
        dropping_plan = TriagePlan(
            open_ids=frozenset({"DW-1"}),
            bundles=tuple(
                Bundle(name, (f"DW-{n + 2}",), "occupied") for n, name in enumerate(occupied)
            ),
            decisions=(agreeing,),
        )

    engine, _ = make_sweep(project, [])
    engine.run_dir.mkdir(parents=True, exist_ok=True)
    (engine.run_dir / "decisions.json").write_text(
        json.dumps({"DW-1": answer}, indent=2), encoding="utf-8"
    )
    _, dropped = engine._materialize_bundles(dropping_plan, {"DW-1": answer})
    assert dropped and engine.state.sweep_dropped_decisions == ["DW-1"]
    assert load_state(engine.run_dir).sweep_dropped_decisions == ["DW-1"]

    resumed, _ = resume_sweep(project, engine, [])
    _, dropped_again = resumed._materialize_bundles(dropping_plan, {"DW-1": answer})
    revival, dropped_on_revival = resumed._materialize_bundles(
        TriagePlan(open_ids=frozenset({"DW-1"}), decisions=(agreeing,)),
        {"DW-1": answer},
    )

    assert not dropped_again and not dropped_on_revival
    assert revival == []
    [drop] = _records(resumed, "sweep-decision-answer-dropped")
    assert drop["decision"] == "DW-1" and drop["drop_cause"] == drop_cause
    attention = (resumed.run_dir / "ATTENTION").read_text(encoding="utf-8")
    assert attention.count("recorded build decision discarded") == 1
    stored = json.loads((resumed.run_dir / "decisions.json").read_text(encoding="utf-8"))
    assert stored == {"DW-1": answer}


def test_a_resumed_cycles_agreeing_option_does_not_revive_a_quarantined_answer(project):
    """The revival the persisted quarantine has to refuse. The resumed cycle
    re-authors key "2" back into agreement with the stored answer, which without
    the quarantine hands a decision the operator was told had been dropped its
    suppressing authority back — silently, since an agreeing option writes no
    record at all. `keep_open_ids` has no observable of its own, so the bundle
    over DW-1 is the probe: it must still run.

    The plan is hand-built rather than validated, since `validate_triage` refuses
    one id in both `bundles` and `decisions`; this is a unit probe of the lane.

    Ablation: delete the keep-open lane's `if dw_id in
    self.state.sweep_dropped_decisions` guard and this reddens — DW-1 re-enters
    `keep_open_ids` and `sneaky-fix` is skipped `human-chose-keep-open`."""
    engine = _drop_keep_open_answer_and_persist(project)

    agreeing_plan = TriagePlan(
        open_ids=frozenset({"DW-1"}),
        bundles=(Bundle(name="sneaky-fix", dw_ids=("DW-1",), intent="y"),),
        decisions=(
            Decision(
                id="DW-1",
                question="q",
                context="",
                options=(
                    DecisionOption(key="1", label="Build", effect="build", intent="x"),
                    DecisionOption(key="2", label="Keep", effect="keep-open"),
                ),
                recommendation="1",
            ),
        ),
    )
    resumed, _ = resume_sweep(project, engine, [])
    bundles, dropped_again = resumed._materialize_bundles(
        agreeing_plan, {"DW-1": _STALE_KEEP_OPEN_ANSWER}
    )

    assert not dropped_again
    assert [b.name for b in bundles] == ["sneaky-fix"]
    assert "human-chose-keep-open" not in journal_text(resumed)
    # and nothing was re-announced or re-recorded on the way there
    assert len(_records(resumed, "sweep-decision-answer-dropped")) == 1
    assert len(_mismatches(resumed)) == 1


def test_the_unattended_skip_quarantine_survives_a_resume(project):
    """DW-124's other half. `_decisions_phase`'s unattended arm journals
    `decision-skipped-unattended` once per id and notifies once per non-empty
    `pending` — a latch that was `__init__` state too, so a resumed run re-skipped
    and re-notified every decision it had already reported as pending.

    Ablation: drop `sweep_skipped_decisions` from `RunState.from_dict` and this
    reddens with two skip rows and two ATTENTION lines."""
    write_ledger(project, {"DW-1": "open"})
    plan = TriagePlan(open_ids=frozenset({"DW-1"}), decisions=(_decision_for("DW-1"),))
    engine, _ = make_sweep(project, [])
    engine.run_dir.mkdir(parents=True, exist_ok=True)

    engine._decisions_phase(plan)
    assert engine.state.sweep_skipped_decisions == ["DW-1"]
    assert load_state(engine.run_dir).sweep_skipped_decisions == ["DW-1"]

    resumed, _ = resume_sweep(project, engine, [])
    resumed._decisions_phase(plan)

    assert [r["dw_id"] for r in _records(resumed, "decision-skipped-unattended")] == ["DW-1"]
    attention = (resumed.run_dir / "ATTENTION").read_text(encoding="utf-8")
    assert attention.count("deferred-work decisions pending") == 1


def test_a_new_run_re_evaluates_a_dropped_keep_open_answer(project):
    """DW-124's quarantine is run-scoped BY DESIGN: persisted so the SAME run does
    not re-announce a disposition it already made, but never project-scoped, so a
    NEW run re-evaluates every decision from scratch. That second half is the claim
    the CHANGELOG, docs/FEATURES.md and both the `RunState` and `__init__` comments
    all make, and the resume rows above cannot catch it — they share one run_dir,
    so a quarantine that leaked across runs would leave them green.

    Run 2 is a genuinely separate run (its own run_dir, state.json and journal),
    handed the same stored answer a project-level pre-answer would give it, and it
    must announce the drop again: this is the loud direction on purpose, since the
    stale answer is still suppressing bundles for DW-1 in that run.

    Ablation: make the quarantine project-scoped — read it from a module-level set,
    or key it off the project rather than `state` — and this reddens, run 2 writing
    no drop row, no mismatch and no ATTENTION line."""
    engine = _drop_keep_open_answer_and_persist(project)

    fresh, _ = make_sweep(project, [], run_id="sweep-run-2")
    # PRECONDITION: a different run, and its quarantine starts empty
    assert fresh.run_dir != engine.run_dir
    assert fresh.state.sweep_dropped_decisions == []

    _, dropped = fresh._materialize_bundles(
        _stale_keep_open_plan(), {"DW-1": _STALE_KEEP_OPEN_ANSWER}
    )

    assert dropped  # progress again in the new run, not swallowed
    assert fresh.state.sweep_dropped_decisions == ["DW-1"]
    assert len(_records(fresh, "sweep-decision-answer-dropped")) == 1
    assert len(_mismatches(fresh)) == 1
    attention = (fresh.run_dir / "ATTENTION").read_text(encoding="utf-8")
    assert attention.count("recorded keep-open decision discarded") == 1
    # run 1's own records are untouched — one drop each, not two in either
    assert len(_records(engine, "sweep-decision-answer-dropped")) == 1


# ------------------------ DW-143: the stale drop retires its PROJECT pre-answer


def _seed_project_keep_open_answer(project, others=None):
    """Plant DW-1's keep-open answer in the PROJECT store — the durable carrier
    `bmad-loop decisions` writes and `_decisions_phase` seeds runs from, as opposed
    to the run-local `<run>/decisions.json` the rows above hand-write. Committed,
    so the only pending edit at prune time is the prune's own.

    `others` seeds the SIBLING entries a single-id removal has to leave alone."""
    from bmad_loop import decisions as decisions_store

    store = decisions_store.store_path(project.project)
    store.parent.mkdir(parents=True, exist_ok=True)
    store.write_text(
        json.dumps({"DW-1": _STALE_KEEP_OPEN_ANSWER, **(others or {})}), encoding="utf-8"
    )
    git(project.project, "add", "-A")
    git(project.project, "commit", "-q", "-m", "out-of-band pre-answer")
    return store


def _cache_keep_open_triage(engine):
    """`_stale_keep_open_plan()` in the persisted form a real cycle caches as
    `<run>/triage.json` — what `decisions.pending_missed_decisions` reconstructs
    the decision from when it re-offers an unanswered id."""
    (engine.run_dir / "triage.json").write_text(
        json.dumps(
            triage_result(
                ["DW-1"],
                decisions=[
                    {
                        "id": "DW-1",
                        "question": "q",
                        "context": "",
                        "options": [
                            {"key": "1", "label": "Build", "effect": "build", "intent": "x"},
                            {"key": "2", "label": "Close as decayed", "effect": "close"},
                        ],
                        "recommendation": "1",
                    }
                ],
            )
        ),
        encoding="utf-8",
    )


@pytest.mark.parametrize("option_vanished", [False, True], ids=["re-authored", "vanished"])
def test_the_keep_open_stale_drop_prunes_the_project_pre_answer_that_fed_it(
    project, option_vanished
):
    """DW-143. The drop's own surfaces are RUN-scoped (DW-124's quarantine bounds
    them to one per id per run, deliberately), but the answer that feeds it lives in
    the PROJECT store, so the drop now retires that entry as part of itself.

    The whole first half of this row is the byte-identical claim: the journal row,
    the ATTENTION text, the quarantine and the returned `answer_dropped` are exactly
    what they were before the prune existed — the prune is additive.

    The store is seeded with two SIBLINGS as well, because a removal is a whole-file
    read-modify-write and the entries it must carry through are as much of the
    contract as the one it deletes: a usable `build` answer for another id, and a
    value `sweep.unusable_answer_reason` rejects — `load_pre_answers`' docstring is
    explicit that an unrelated write must not silently delete a human's corrupt
    entry, which is the file repair this codebase refuses.

    Ablation: delete the `_prune_dropped_pre_answer(...)` call from the keep-open
    lane and this reddens — the store still holds DW-1 and no prune row is written.
    Ablation: write `_write_store(project, {})` in `drop_pre_answer` and this reddens
    on the siblings, which no other row would notice.
    Follow-up ablations: guard the prune on the answered key still existing and
    the vanished-key case fails; corrupt the sandbox ledger in the prune helper
    and both cases fail the preserved-history assertion.
    """
    from bmad_loop import decisions as decisions_store

    write_ledger(project, {"DW-1": "open"})
    assert deferredwork.record_decision(
        project.deferred_work, "DW-1", "2026-09-06", "Keep", "Preserve this decision history"
    )
    ledger_before = project.deferred_work.read_bytes()
    siblings = {
        "DW-2": {"key": "1", "label": "Widen", "effect": "build", "intent": "x"},
        "DW-3": ["not a decision answer at all"],  # unusable, and NOT ours to repair
    }
    store = _seed_project_keep_open_answer(project, siblings)
    engine, _ = make_sweep(project, [])
    engine.run_dir.mkdir(parents=True, exist_ok=True)
    plan = _stale_keep_open_plan()
    if option_vanished:
        decision = plan.decisions[0]
        plan = replace(
            plan,
            decisions=(
                replace(
                    decision, options=(decision.options[0], replace(decision.options[1], key="3"))
                ),
            ),
        )

    answers, _closed = engine._decisions_phase(plan)
    # PRECONDITION: the answer reached this run through the PROJECT store, not a
    # hand-written run-local file — the carrier every earlier row bypasses
    assert answers == {"DW-1": _STALE_KEEP_OPEN_ANSWER}
    assert [r["dw_id"] for r in _records(engine, "decision-preanswered")] == ["DW-1"]

    _, dropped = engine._materialize_bundles(plan, answers)

    assert dropped is True
    [drop] = _records(engine, "sweep-decision-answer-dropped")
    assert {k: v for k, v in drop.items() if k != "ts"} == {
        "kind": "sweep-decision-answer-dropped",
        "decision": "DW-1",
        "drop_cause": "stale-option",
    }
    assert engine.state.sweep_dropped_decisions == ["DW-1"]
    attention = (engine.run_dir / "ATTENTION").read_text(encoding="utf-8")
    fate = "is gone from this cycle's triage" if option_vanished else "has been re-authored since"
    assert re.sub(r"^\[[^\]]+\] ", "", attention) == (
        "decision DW-1: recorded keep-open decision discarded: "
        f"the option it answered (2) {fate}, so the keep-open "
        "protection is discarded and DW-1 is eligible for bundling again\n"
    )
    assert len(_mismatches(engine)) == (0 if option_vanished else 1)
    assert project.deferred_work.read_bytes() == ledger_before
    # and the entry that fed it is gone from the project store — but ONLY it: both
    # siblings ride through the read-modify-write unchanged, the unusable one included
    assert json.loads(store.read_text(encoding="utf-8")) == siblings
    assert decisions_store.load_pre_answers(project.project) == siblings
    [pruned] = _records(engine, "sweep-decision-preanswer-pruned")
    assert pruned["decision"] == "DW-1" and pruned["drop_cause"] == "stale-option"
    # id and cause ONLY: deleting a human's answer stays auditable without the
    # answer's prose (or the store's path) entering the journal
    assert set(pruned) == {"ts", "kind", "decision", "drop_cause"}
    # committed, because this cycle's bundles run next and need a clean baseline
    assert worktree_clean(project.project)
    assert len(_records(engine, "sweep-ledger-commit")) == 1
    # the RUN-LOCAL record still stands — only the project entry went
    run_local = json.loads((engine.run_dir / "decisions.json").read_text(encoding="utf-8"))
    assert run_local == {"DW-1": _STALE_KEEP_OPEN_ANSWER}


def test_a_second_run_reads_no_stale_answer_because_the_first_pruned_it(project):
    """DW-143's actual regression: the notification loop the prune breaks. Before
    it, run 1 dropped the answer while the project store kept it, and
    `pending_missed_decisions` counts a usably-answered id as answered — so triage
    kept re-asking DW-1 as a decision, nothing ever bundled it, `_prune_pre_answers`
    never came due, and every NEW run re-seeded the same stale answer, re-dropped it
    and re-notified, with no surface offering the human a way to re-answer.

    Run 2 is a genuinely separate run (its own run_dir, state.json and journal), so
    DW-124's run-scoped quarantine gives it no cover: silence here is the store
    being empty, not the quarantine.

    Ablation: delete the `_prune_dropped_pre_answer(...)` call and this reddens on
    every assertion below — run 2 re-seeds, re-drops, re-notifies, and
    `pending_missed_decisions` returns [] because the stale entry still counts as
    answered."""
    from bmad_loop import decisions as decisions_store

    write_ledger(project, {"DW-1": "open"})
    install_bmad_config(project)  # `pending_missed_decisions` resolves paths from it
    _seed_project_keep_open_answer(project)
    plan = _stale_keep_open_plan()

    first, _ = make_sweep(project, [])
    first.run_dir.mkdir(parents=True, exist_ok=True)
    _cache_keep_open_triage(first)
    first_answers, _closed = first._decisions_phase(plan)
    _, dropped_first = first._materialize_bundles(plan, first_answers)
    assert dropped_first  # PRECONDITION: run 1 really did drop (and so prune)

    second, _ = make_sweep(project, [], run_id="sweep-run-2")
    second.run_dir.mkdir(parents=True, exist_ok=True)
    assert second.run_dir != first.run_dir
    second_answers, _closed = second._decisions_phase(plan)
    _, dropped_second = second._materialize_bundles(plan, second_answers)

    assert second_answers == {}  # nothing left in the store to seed
    assert not dropped_second
    assert _records(second, "sweep-decision-answer-dropped") == []
    assert _records(second, "sweep-decision-preanswer-pruned") == []
    assert _mismatches(second) == []
    attention = (second.run_dir / "ATTENTION").read_text(encoding="utf-8")
    assert "recorded keep-open decision discarded" not in attention
    # and the id is re-offered out of band instead of being invisible to everyone
    assert [d.id for d in decisions_store.pending_missed_decisions(project.project)] == ["DW-1"]


def test_prune_dropped_pre_answer_keys_on_the_run_dir_project_not_workspace_root(project, tmp_path):
    """The fifth member of the workspace-rooted family the two rows above pin (the
    `_decisions_phase` read, its seeded and interactive writes, and
    `_prune_pre_answers`), and it fails the same way: under the `repo_root` override
    `workspace.root` is the separate CODE repo, whose `.bmad-loop/decisions.json`
    does not exist, so a workspace-rooted removal would find nothing to delete, write
    nothing, journal nothing — and leave the stale answer in the PROJECT store to be
    re-seeded and re-dropped by every later run, which is the exact defect DW-143
    exists to close. `runs._project_of_run_dir` is the stable anchor, the same root
    the seeding read uses.

    Ablation: point `_prune_dropped_pre_answer` at `self.workspace.root` and this
    reddens — DW-1 survives in the project store and no prune row lands."""
    from bmad_loop import decisions as decisions_store
    from bmad_loop.workspace import Workspace

    write_ledger(project, {"DW-1": "open"})
    _seed_project_keep_open_answer(project)
    engine, _ = make_sweep(project, [])
    elsewhere = tmp_path / "code-repo"
    elsewhere.mkdir()
    git(elsewhere, "init")
    engine.workspace = Workspace(root=elsewhere, paths=engine.workspace.paths)
    engine.run_dir.mkdir(parents=True, exist_ok=True)

    _, dropped = engine._materialize_bundles(
        _stale_keep_open_plan(), {"DW-1": _STALE_KEEP_OPEN_ANSWER}
    )

    assert dropped
    assert decisions_store.load_pre_answers(project.project) == {}
    [pruned] = _records(engine, "sweep-decision-preanswer-pruned")
    assert pruned["decision"] == "DW-1"
    assert list(elsewhere.rglob("decisions*")) == []  # the code repo was never the store


def test_a_run_local_only_stale_answer_writes_nothing_to_the_project_store(project):
    """The prune retires a PROJECT store entry and there is none here — this stale
    answer only ever lived in `<run>/decisions.json`. A store holding OTHER ids must
    therefore keep its exact bytes: the removal is a read-modify-write, and an
    unconditional one would re-serialize (indent + sort) a file this drop has no
    business rewriting.

    Ablation: delete `drop_pre_answer`'s `if dw_id not in data: return False` and
    this reddens — the compact bytes below come back re-indented, and the prune row
    appears for a removal that never happened."""
    from bmad_loop import decisions as decisions_store

    store = decisions_store.store_path(project.project)
    store.parent.mkdir(parents=True, exist_ok=True)
    other = {"DW-2": {"key": "1", "label": "Widen", "effect": "build", "intent": "x"}}
    store.write_text(json.dumps(other), encoding="utf-8")  # compact, unlike _write_store's
    before = store.read_bytes()

    engine = _drop_keep_open_answer_and_persist(project)

    assert store.read_bytes() == before
    assert _records(engine, "sweep-decision-preanswer-pruned") == []
    assert _records(engine, "sweep-ledger-commit") == []
    # the drop itself is unaffected by having nothing to prune
    assert len(_records(engine, "sweep-decision-answer-dropped")) == 1


@pytest.mark.parametrize("drop_cause", ["no-intent", "name-collision"])
def test_a_build_answer_drop_leaves_the_project_pre_answer_in_place(project, drop_cause):
    """The prune is scoped to the keep-open lane. A dropped BUILD answer leaves its
    deferred-work entry open to be re-asked with the stored answer still carrying a
    payload — an intent, a label, a bundle name — so retiring it would delete a
    usable answer; a dropped keep-open answer has nothing left beyond the option it
    named, which is why only that lane prunes.

    Ablation: call `_prune_dropped_pre_answer` from either build lane's drop and
    this reddens with an empty store."""
    from bmad_loop import decisions as decisions_store

    write_ledger(project, {"DW-1": "open"})
    agreeing = Decision(
        id="DW-1",
        question="q",
        context="",
        options=(DecisionOption(key="1", label="Widen", effect="build", intent="x"),),
        recommendation="1",
    )
    answer = {"key": "1", "label": "Widen", "effect": "build"}
    if drop_cause == "no-intent":
        dropping_plan = TriagePlan(
            open_ids=frozenset({"DW-1"}),
            decisions=(
                Decision(
                    id="DW-1",
                    question="q",
                    context="",
                    options=(DecisionOption(key="1", label="Close", effect="close"),),
                    recommendation="1",
                ),
            ),
        )
    else:
        answer["intent"] = "x"
        occupied = ["decision-dw-1"] + [f"decision-dw-1-{n}" for n in range(2, 10)]
        dropping_plan = TriagePlan(
            open_ids=frozenset({"DW-1"}),
            bundles=tuple(
                Bundle(name, (f"DW-{n + 2}",), "occupied") for n, name in enumerate(occupied)
            ),
            decisions=(agreeing,),
        )

    store = decisions_store.store_path(project.project)
    store.parent.mkdir(parents=True, exist_ok=True)
    store.write_text(json.dumps({"DW-1": answer}), encoding="utf-8")
    engine, _ = make_sweep(project, [])
    engine.run_dir.mkdir(parents=True, exist_ok=True)

    _, dropped = engine._materialize_bundles(dropping_plan, {"DW-1": answer})

    assert dropped
    [drop] = _records(engine, "sweep-decision-answer-dropped")
    assert drop["drop_cause"] == drop_cause
    assert decisions_store.load_pre_answers(project.project) == {"DW-1": answer}
    assert _records(engine, "sweep-decision-preanswer-pruned") == []


def test_the_stale_drop_refuses_a_readonly_project_store(project):
    """Retiring a human-authored answer is a store WRITE, never a repair, so it
    inherits `_write_store`'s `require_writable_target=True` (#597): an operator who
    marks the store read-only is answered with a `PermissionError` rather than
    quietly overwritten, and the prune is not special-cased into swallowing it.

    The drop's own announcement is already durable when this raises — the row, the
    notify and the quarantine all precede the prune — so the run reports what it did
    before the write failed, and the entry is retired by the next run that re-drops
    it once the store is writable again.

    chmod is on the per-test copytree copy the `project` fixture makes, and restored
    in a `finally` because Windows rmtree refuses a READONLY file at cleanup.

    Ablation: wrap `_prune_dropped_pre_answer`'s `drop_pre_answer` call in a
    `contextlib.suppress(OSError)` and this fails `DID NOT RAISE`."""
    from bmad_loop import decisions as decisions_store

    write_ledger(project, {"DW-1": "open"})
    store = _seed_project_keep_open_answer(project)
    before = store.read_bytes()
    engine, _ = make_sweep(project, [])
    engine.run_dir.mkdir(parents=True, exist_ok=True)

    store.chmod(0o444)
    try:
        with pytest.raises(PermissionError):
            engine._materialize_bundles(_stale_keep_open_plan(), {"DW-1": _STALE_KEEP_OPEN_ANSWER})
    finally:
        store.chmod(0o644)

    assert store.read_bytes() == before  # the answer is still there to retire later
    assert decisions_store.load_pre_answers(project.project) == {"DW-1": _STALE_KEEP_OPEN_ANSWER}
    # the drop was announced and latched before the write was attempted
    assert len(_records(engine, "sweep-decision-answer-dropped")) == 1
    assert engine.state.sweep_dropped_decisions == ["DW-1"]
    assert _records(engine, "sweep-decision-preanswer-pruned") == []

    # THE RECOVERY the helper's docstring promises, driven rather than asserted in
    # prose: the surviving answer is not stranded. A NEW run (DW-124's quarantine is
    # run-scoped, so it gives run 2 no cover) re-seeds it from the store the write
    # failed on, re-drops it — the loud direction, since it is still suppressing
    # bundles for DW-1 — and this time the prune lands.
    fresh, _ = make_sweep(project, [], run_id="sweep-run-2")
    fresh.run_dir.mkdir(parents=True, exist_ok=True)
    plan = _stale_keep_open_plan()
    reseeded, _closed = fresh._decisions_phase(plan)
    assert reseeded == {"DW-1": _STALE_KEEP_OPEN_ANSWER}  # the locked store still fed it

    _, dropped_again = fresh._materialize_bundles(plan, reseeded)

    assert dropped_again
    assert len(_records(fresh, "sweep-decision-answer-dropped")) == 1
    [pruned] = _records(fresh, "sweep-decision-preanswer-pruned")
    assert pruned["decision"] == "DW-1" and pruned["drop_cause"] == "stale-option"
    assert decisions_store.load_pre_answers(project.project) == {}


# --------------------------------------- DW-134: a malformed stored-answer store


_widen_or_keep = [
    {"key": "1", "label": "Widen", "effect": "build", "intent": "x"},
    {"key": "2", "label": "Keep", "effect": "keep-open"},
]


def _decision_for(dw_id):
    """One-option `Decision` for the DW-134 rows, which care about the STORE's
    shape rather than about option agreement."""
    return Decision(
        id=dw_id,
        question="q",
        context="",
        options=(DecisionOption(key="1", label="Widen", effect="build", intent="widen the field"),),
        recommendation="1",
    )


def test_non_dict_stored_answer_is_dropped_while_the_rest_still_answer(project):
    """DW-134. `<run>/decisions.json` is the orchestrator's own file, but a crash
    mid-write or a hand edit can leave a value that is not an object — and every
    consumer calls `.get(...)` on it. The bad entry loses its answer and is
    journaled; the well-shaped one beside it builds its bundle exactly as before.
    Ablation: restore the bare `_read_json` assignment in `_decisions_phase` and
    this reddens with an AttributeError out of `_materialize_bundles`."""
    write_ledger(project, {"DW-1": "open", "DW-2": "open"})
    plan = triage_result(
        ["DW-1", "DW-2"],
        decisions=[
            _decision("DW-1", _widen_or_keep),
            _decision("DW-2", _widen_or_keep),
        ],
    )
    good = {"key": "1", "label": "Widen", "effect": "build", "answered_at": "2026-09-05"}
    engine, _ = make_sweep(
        project,
        [
            triage_effect(plan),
            bundle_dev_effect(project, "decision-dw-2", ["DW-2"]),
            bundle_review_effect(project, "decision-dw-2"),
        ],
    )
    engine.run_dir.mkdir(parents=True, exist_ok=True)
    store = engine.run_dir / "decisions.json"
    store.write_text(json.dumps({"DW-1": "keep-open", "DW-2": good}, indent=2), encoding="utf-8")

    summary = engine.run()
    assert not summary.paused

    [reload_failed] = _records(engine, "sweep-decisions-reload-failed")
    assert reload_failed["errors"] == [
        "DW-1: not a JSON object: str (<run>/decisions.json)"  # names the file to hand-fix
    ]
    # DW-2's answer was honoured: its decision bundle ran and closed the entry
    assert engine.state.tasks["dw-decision-dw-2"].phase == Phase.DONE
    assert not ledger_entries(project)["DW-2"].open
    # DW-1 kept no answer, so it went back down the unattended pending path
    assert [r["dw_id"] for r in _records(engine, "decision-skipped-unattended")] == ["DW-1"]
    assert ledger_entries(project)["DW-1"].open
    # the degrade is IN-MEMORY: the human's file is neither repaired nor trimmed
    assert json.loads(store.read_text(encoding="utf-8")) == {"DW-1": "keep-open", "DW-2": good}


def test_stored_answers_with_a_non_dict_top_level_degrade_to_none(project):
    """DW-134. A store that is a JSON array answers nothing; the cycle proceeds
    through its normal pending path instead of raising."""
    write_ledger(project, {"DW-1": "open"})
    plan = TriagePlan(open_ids=frozenset({"DW-1"}), decisions=(_decision_for("DW-1"),))
    engine, _ = make_sweep(project, [])
    engine.run_dir.mkdir(parents=True, exist_ok=True)
    (engine.run_dir / "decisions.json").write_text('["DW-1"]', encoding="utf-8")

    answers, closed = engine._decisions_phase(plan)

    assert answers == {} and closed == 0
    [reload_failed] = _records(engine, "sweep-decisions-reload-failed")
    assert reload_failed["errors"] == ["not a JSON object: list"]
    assert [r["dw_id"] for r in _records(engine, "decision-skipped-unattended")] == ["DW-1"]


def test_unreadable_stored_answers_degrade_to_none(project):
    """DW-134. Truncated JSON (a crash mid-write) — caught, journaled with the
    exception text, and the sweep carries on."""
    write_ledger(project, {"DW-1": "open"})
    plan = TriagePlan(open_ids=frozenset({"DW-1"}), decisions=(_decision_for("DW-1"),))
    engine, _ = make_sweep(project, [])
    engine.run_dir.mkdir(parents=True, exist_ok=True)
    (engine.run_dir / "decisions.json").write_text('{"DW-1": {"key": "1"', encoding="utf-8")

    answers, closed = engine._decisions_phase(plan)

    assert answers == {} and closed == 0
    [reload_failed] = _records(engine, "sweep-decisions-reload-failed")
    assert reload_failed["errors"][0].startswith("unreadable: ")
    assert [r["dw_id"] for r in _records(engine, "decision-skipped-unattended")] == ["DW-1"]


def test_undecodable_stored_answers_degrade_to_none(project):
    """DW-134. The third caught fault: bytes that are not UTF-8 at all."""
    write_ledger(project, {"DW-1": "open"})
    plan = TriagePlan(open_ids=frozenset({"DW-1"}), decisions=(_decision_for("DW-1"),))
    engine, _ = make_sweep(project, [])
    engine.run_dir.mkdir(parents=True, exist_ok=True)
    (engine.run_dir / "decisions.json").write_bytes(b'{"DW-1": {"key": "\xff"}}')

    answers, closed = engine._decisions_phase(plan)

    assert answers == {}
    [reload_failed] = _records(engine, "sweep-decisions-reload-failed")
    assert reload_failed["errors"][0].startswith("unreadable: ")


def test_oserror_reading_stored_answers_degrades_to_none(project, monkeypatch):
    """DW-134. The read-side I/O arm is distinct from JSON and UTF-8 decoding:
    a filesystem refusal must journal and take the same pending path. Ablation:
    remove `OSError` from `_decisions_phase`'s catch tuple and this raises."""
    write_ledger(project, {"DW-1": "open"})
    plan = TriagePlan(open_ids=frozenset({"DW-1"}), decisions=(_decision_for("DW-1"),))
    engine, _ = make_sweep(project, [])
    engine.run_dir.mkdir(parents=True, exist_ok=True)
    store = engine.run_dir / "decisions.json"
    store.write_text(json.dumps({"DW-1": {"key": "1"}}), encoding="utf-8")
    fault_read_text(monkeypatch, store)

    answers, closed = engine._decisions_phase(plan)

    assert answers == {} and closed == 0
    [reload_failed] = _records(engine, "sweep-decisions-reload-failed")
    assert reload_failed["errors"][0].startswith("unreadable: ")
    assert [r["dw_id"] for r in _records(engine, "decision-skipped-unattended")] == ["DW-1"]


def test_whole_file_fault_is_replaced_by_seeded_decision_write_back(project):
    """A project pre-answer is a legitimate phase write, so it replaces a
    non-object run store that has no per-value content the phase can preserve."""
    from bmad_loop import decisions as decisions_store

    write_ledger(project, {"DW-1": "open"})
    good = {"key": "1", "label": "Widen", "effect": "build", "intent": "widen the field"}
    pre_store = decisions_store.store_path(project.project)
    pre_store.parent.mkdir(parents=True, exist_ok=True)
    pre_store.write_text(json.dumps({"DW-1": good}), encoding="utf-8")
    plan = TriagePlan(open_ids=frozenset({"DW-1"}), decisions=(_decision_for("DW-1"),))
    engine, _ = make_sweep(project, [])
    engine.run_dir.mkdir(parents=True, exist_ok=True)
    run_store = engine.run_dir / "decisions.json"
    run_store.write_text('["DW-1"]', encoding="utf-8")

    answers, _closed = engine._decisions_phase(plan)

    assert answers == {"DW-1": good}
    assert json.loads(run_store.read_text(encoding="utf-8")) == {"DW-1": good}


def test_whole_file_fault_is_replaced_by_interactive_decision_write_back(project):
    """The interactive writer also replaces an unreadable whole-file fault so
    the answer it just accepted survives a resume."""
    write_ledger(project, {"DW-1": "open"})
    plan = TriagePlan(open_ids=frozenset({"DW-1"}), decisions=(_decision_for("DW-1"),))
    engine, _ = make_sweep(project, [], answers=["1"], prompting=True)
    engine.run_dir.mkdir(parents=True, exist_ok=True)
    run_store = engine.run_dir / "decisions.json"
    run_store.write_text('{"DW-1": {"key": "1"', encoding="utf-8")

    answers, _closed = engine._decisions_phase(plan)

    rewritten = json.loads(run_store.read_text(encoding="utf-8"))
    assert answers["DW-1"]["effect"] == "build"
    assert rewritten["DW-1"]["effect"] == "build"


def test_malformed_pre_answer_is_dropped_and_the_write_back_keeps_the_unusable_value(project):
    """DW-134, both halves in one row. `decisions.load_pre_answers` validates only
    the TOP level, so a project-store VALUE can be any JSON and used to reach
    `.get("effect")` in the seeding loop; it is dropped and named in the same
    single record as the run-local store's bad value. And the write-back the
    seeding triggers re-publishes the run-local value it could not use, rather
    than deleting it — the degrade is in-memory, not a repair of the file."""
    from bmad_loop import decisions as decisions_store

    write_ledger(project, {"DW-1": "open", "DW-2": "open", "DW-3": "open"})
    good = {"key": "1", "label": "Widen", "effect": "build", "intent": "widen the field"}
    store = decisions_store.store_path(project.project)
    store.parent.mkdir(parents=True, exist_ok=True)
    store.write_text(json.dumps({"DW-2": ["nope"], "DW-3": good}), encoding="utf-8")

    plan = TriagePlan(
        open_ids=frozenset({"DW-1", "DW-2", "DW-3"}),
        decisions=(_decision_for("DW-1"), _decision_for("DW-2"), _decision_for("DW-3")),
    )
    engine, _ = make_sweep(project, [])
    engine.run_dir.mkdir(parents=True, exist_ok=True)
    run_store = engine.run_dir / "decisions.json"
    run_store.write_text(json.dumps({"DW-1": 7}), encoding="utf-8")

    answers, _closed = engine._decisions_phase(plan)

    assert answers == {"DW-3": good}
    [reload_failed] = _records(engine, "sweep-decisions-reload-failed")
    assert reload_failed["errors"] == [
        "DW-1: not a JSON object: int (<run>/decisions.json)",
        "DW-2: not a JSON object: list (project .bmad-loop/decisions.json)",
    ]
    assert json.loads(run_store.read_text(encoding="utf-8")) == {"DW-1": 7, "DW-3": good}
    assert sorted(r["dw_id"] for r in _records(engine, "decision-skipped-unattended")) == [
        "DW-1",
        "DW-2",
    ]


def test_materialize_bundles_skips_a_non_dict_answer(project):
    """DW-134's other site. The test suite hands `_materialize_bundles` a map
    directly, past `_decisions_phase`'s read-site guard, so each lane has to hold
    the shape itself. A non-dict answer is a silent skip on BOTH lanes: no bundle
    is built from it and it suppresses nothing.
    Ablation: restore either lane's `answer.get(...)` without the isinstance and
    this reddens with `AttributeError: 'int' object has no attribute 'get'`."""
    write_ledger(project, {"DW-1": "open"})
    plan = TriagePlan(
        open_ids=frozenset({"DW-1"}),
        bundles=(Bundle(name="safe-fix", dw_ids=("DW-1",), intent="a"),),
        decisions=(_decision_for("DW-1"),),
    )
    engine, _ = make_sweep(project, [])
    engine.run_dir.mkdir(parents=True, exist_ok=True)
    (engine.run_dir / "journal.jsonl").write_text("", encoding="utf-8")

    bundles, dropped = engine._materialize_bundles(plan, {"DW-1": 7})

    assert not dropped
    # built nothing of its own, and did not suppress the plan bundle over its id
    assert [b.name for b in bundles] == ["safe-fix"]
    assert engine.state.sweep_dropped_decisions == []
    # the two records a lane that ACTED on the answer would have written
    assert _records(engine, "sweep-decision-answer-dropped") == []
    assert _records(engine, "sweep-bundle-skipped") == []


def test_interactive_answer_write_back_keeps_an_unusable_stored_value(project):
    """DW-134. The per-answer write-back inside the prompting loop republishes the
    whole map, so without the `unusable` merge the first answer a human gives
    deletes every value the read could not use — a silent repair of the human's
    file on a write made for an unrelated reason. DW-9 is not a decision this
    cycle, so nothing re-answers it; it must still be in the file afterwards.
    Ablation: revert that call to `json.dumps(answers, indent=2)` and this reddens
    — DW-9 is gone from the rewritten store."""
    write_ledger(project, {"DW-1": "open", "DW-9": "open"})
    plan = TriagePlan(open_ids=frozenset({"DW-1", "DW-9"}), decisions=(_decision_for("DW-1"),))
    engine, _ = make_sweep(project, [], answers=["1"], prompting=True)
    engine.run_dir.mkdir(parents=True, exist_ok=True)
    store = engine.run_dir / "decisions.json"
    store.write_text(json.dumps({"DW-9": 7}), encoding="utf-8")

    answers, _closed = engine._decisions_phase(plan)

    assert answers["DW-1"]["effect"] == "build"  # PRECONDITION: the prompt ran
    assert "DW-9" not in answers  # unusable in memory...
    rewritten = json.loads(store.read_text(encoding="utf-8"))
    assert rewritten["DW-9"] == 7  # ...but still on disk, unchanged
    assert rewritten["DW-1"]["effect"] == "build"
    [reload_failed] = _records(engine, "sweep-decisions-reload-failed")
    assert reload_failed["errors"] == ["DW-9: not a JSON object: int (<run>/decisions.json)"]


# ------- DW-141/DW-142: the stored-answer SHAPE, not just its container's type


# (value, the reason `unusable_answer_reason` gives for it). The reason string is
# half the contract: it is what the `malformed` row names, and it must stay ids /
# field names / type names only — never the answer's prose.
_UNUSABLE_SHAPES = [
    pytest.param({}, "effect missing", id="empty-object"),
    pytest.param({"key": "1", "label": "Widen"}, "effect missing", id="scalars-but-no-effect"),
    pytest.param({"effect": "frobnicate"}, "effect not recognized", id="unknown-effect"),
    pytest.param({"effect": ["build"]}, "effect not recognized", id="non-string-effect"),
    pytest.param(
        {"key": "1", "label": "Widen", "effect": "build", "intent": ["a", "b"]},
        "intent not a string: list",
        id="non-string-intent",
    ),
    pytest.param({"key": 1, "effect": "keep-open"}, "key not a string: int", id="non-string-key"),
    pytest.param(
        {"label": None, "effect": "build", "intent": "x"},
        "label not a string: NoneType",
        id="null-label",
    ),
    pytest.param(
        {"effect": "build", "intent": "x", "bundle_name": 7},
        "bundle_name not a string: int",
        id="non-string-bundle-name",
    ),
]

# The other half of the contract: shapes that must stay USABLE. Rejecting is not a
# free "be strict" — an answer refused at the read site stops being seeded, and a
# refused KEEP-OPEN answer stops suppressing its bundle, so a later cycle builds
# work the human said to leave alone. Each row is a field this predicate must not
# reach for.
_USABLE_SHAPES = [
    pytest.param({"effect": "build", "intent": "x"}, id="minimal-build"),
    pytest.param({"key": "1", "label": "Keep", "effect": "keep-open"}, id="minimal-keep-open"),
    pytest.param({"key": "1", "label": "W", "effect": "close"}, id="close-from-the-in-run-writer"),
    pytest.param(
        {"key": "1", "effect": "keep-open", "intent": ["a", "b"], "bundle_name": 7},
        id="keep-open-with-build-only-fields-corrupt",
    ),
    pytest.param(
        {"effect": "build", "intent": "x", "resolution": ["a"], "answered_at": 20260906},
        id="non-string-fields-no-reader-consumes",
    ),
]


@pytest.mark.parametrize("value,reason", _UNUSABLE_SHAPES)
def test_unusable_answer_reason_names_the_defect(value, reason):
    """The predicate itself, at the lowest layer that can catch a regression — its
    reason strings are the `malformed` rows' wording AND the boundary both readers
    agree on, so pinning them only through journal assertions left the pure
    function untested. Ids, field names and type names only: never the answer's
    prose (`journal.append` neither truncates nor sanitizes what it is handed)."""
    assert sweep_mod.unusable_answer_reason(value) == reason


@pytest.mark.parametrize("value", _USABLE_SHAPES)
def test_unusable_answer_reason_accepts_what_readers_can_consume(value):
    """The non-validation boundary, pinned so a later "tighten everything" pass
    reddens here instead of silently rejecting answers in the field. `resolution`
    and `answered_at` are unvalidated because no reader consumes them; `intent` and
    `bundle_name` are unvalidated for keep-open because only the BUILD lane reads
    them, and refusing a keep-open answer un-suppresses its bundle.
    Ablation: validate `intent`/`bundle_name` for every effect (or add `resolution`
    to the screened fields) and the matching row reddens."""
    assert sweep_mod.unusable_answer_reason(value) is None


@pytest.mark.parametrize("value,reason", _UNUSABLE_SHAPES)
def test_unusable_stored_answer_shape_is_dropped_from_the_run_store(project, value, reason):
    """DW-142. DW-134 stopped at "is it a dict"; a dict with no usable `effect` —
    or with a scalar the bundle lanes read as a string that is not one — still
    counted as an answer. `_materialize_bundles` routes on `effect`, so such a
    value matched NO lane: the id was answered as far as every reader was
    concerned and acted on by none. Rejected at the read site instead, it goes
    back down the pending path, and the file is neither repaired nor trimmed.
    Ablation: restore `if isinstance(value, dict)` in `_decisions_phase`'s
    run-local loop and every row here reddens — the value lands in `answers`."""
    write_ledger(project, {"DW-1": "open"})
    plan = TriagePlan(open_ids=frozenset({"DW-1"}), decisions=(_decision_for("DW-1"),))
    engine, _ = make_sweep(project, [])
    engine.run_dir.mkdir(parents=True, exist_ok=True)
    store = engine.run_dir / "decisions.json"
    store.write_text(json.dumps({"DW-1": value}), encoding="utf-8")

    answers, closed = engine._decisions_phase(plan)

    assert answers == {} and closed == 0
    [reload_failed] = _records(engine, "sweep-decisions-reload-failed")
    assert reload_failed["errors"] == [f"DW-1: {reason} (<run>/decisions.json)"]
    assert [r["dw_id"] for r in _records(engine, "decision-skipped-unattended")] == ["DW-1"]
    # in-memory degrade: no write this phase makes touches the human's file
    assert json.loads(store.read_text(encoding="utf-8")) == {"DW-1": value}


def test_close_effect_stays_a_usable_stored_answer(project):
    """The interactive writer records `effect: "close"` for a decision answered
    `close` this run, so the shared predicate has to accept it: rejecting it would
    re-ask, inside one run, a decision the human already answered. DW-1 keeps its
    answer and is NOT re-offered down the unattended pending path.
    Ablation: drop "close" from `DECISION_EFFECTS` and this reddens."""
    write_ledger(project, {"DW-1": "open"})
    plan = TriagePlan(open_ids=frozenset({"DW-1"}), decisions=(_decision_for("DW-1"),))
    engine, _ = make_sweep(project, [])
    engine.run_dir.mkdir(parents=True, exist_ok=True)
    stored = {"key": "1", "label": "Widen", "effect": "close", "answered_at": "2026-09-06"}
    (engine.run_dir / "decisions.json").write_text(json.dumps({"DW-1": stored}), encoding="utf-8")
    # this phase journals nothing at all on the happy path, so `_records` needs the file
    (engine.run_dir / "journal.jsonl").write_text("", encoding="utf-8")

    answers, _closed = engine._decisions_phase(plan)

    assert answers == {"DW-1": stored}
    assert _records(engine, "sweep-decisions-reload-failed") == []
    assert _records(engine, "decision-skipped-unattended") == []


def test_keep_open_answer_with_a_corrupt_build_only_field_still_suppresses_its_bundle(project):
    """The cost of over-rejecting, pinned end to end. `intent` and `bundle_name` are
    read by the BUILD lane alone, so screening them for every effect would refuse
    this keep-open answer at the read site — and a refused keep-open answer is not
    a safe no-op: it stops being seeded, so it stops SUPPRESSING its bundle, and
    this cycle builds work the human explicitly said to leave open. The corrupt
    fields are inert prose no reader touches, so the answer stays usable and
    `safe-fix` is skipped as `human-chose-keep-open`.
    Ablation: screen `intent`/`bundle_name` for every effect (drop the
    `effect == "build"` guard in `unusable_answer_reason`) and this reddens — the
    answer lands in `unusable`, `answers` is empty and the bundle survives."""
    write_ledger(project, {"DW-1": "open"})
    decision = Decision(
        id="DW-1",
        question="q",
        context="",
        options=(DecisionOption(key="1", label="Keep", effect="keep-open"),),
        recommendation="1",
    )
    plan = TriagePlan(
        open_ids=frozenset({"DW-1"}),
        bundles=(Bundle(name="safe-fix", dw_ids=("DW-1",), intent="a"),),
        decisions=(decision,),
    )
    stored = {
        "key": "1",
        "label": "Keep",
        "effect": "keep-open",
        "intent": ["a", "b"],  # build-only fields, corrupt...
        "bundle_name": 7,
    }
    engine, _ = make_sweep(project, [])
    engine.run_dir.mkdir(parents=True, exist_ok=True)
    (engine.run_dir / "decisions.json").write_text(json.dumps({"DW-1": stored}), encoding="utf-8")
    (engine.run_dir / "journal.jsonl").write_text("", encoding="utf-8")

    answers, _closed = engine._decisions_phase(plan)

    assert answers == {"DW-1": stored}  # ...and the answer is still usable
    assert _records(engine, "sweep-decisions-reload-failed") == []

    bundles, dropped = engine._materialize_bundles(plan, answers)

    assert bundles == []  # the protection the human bought is still in force
    assert not dropped
    [skipped] = _records(engine, "sweep-bundle-skipped")
    assert skipped["name"] == "safe-fix" and skipped["reason"] == "human-chose-keep-open"


def test_unusable_pre_answer_shape_is_dropped_and_the_write_back_republishes_it(project):
    """DW-142, the project store's half plus the write-back promise. A pre-answer
    whose `intent` is a list is rejected by the SAME predicate the run-local loop
    uses, named in the same single record with its own store spelled out (the two
    files are different and only one is the one to hand-fix), and the write-back
    the well-shaped seeding triggers re-publishes the run-local value it could not
    use rather than deleting it. Ablation: restore `isinstance(pre_answer, dict)`
    in the seeding loop and DW-2 is seeded as an answer, reddening `answers`."""
    from bmad_loop import decisions as decisions_store

    write_ledger(project, {"DW-1": "open", "DW-2": "open", "DW-3": "open"})
    good = {"key": "1", "label": "Widen", "effect": "build", "intent": "widen the field"}
    bad_pre = {"key": "1", "label": "Widen", "effect": "build", "intent": ["a", "b"]}
    pre_store = decisions_store.store_path(project.project)
    pre_store.parent.mkdir(parents=True, exist_ok=True)
    pre_store.write_text(json.dumps({"DW-2": bad_pre, "DW-3": good}), encoding="utf-8")

    plan = TriagePlan(
        open_ids=frozenset({"DW-1", "DW-2", "DW-3"}),
        decisions=(_decision_for("DW-1"), _decision_for("DW-2"), _decision_for("DW-3")),
    )
    engine, _ = make_sweep(project, [])
    engine.run_dir.mkdir(parents=True, exist_ok=True)
    run_store = engine.run_dir / "decisions.json"
    run_store.write_text(json.dumps({"DW-1": {"effect": "frobnicate"}}), encoding="utf-8")

    answers, _closed = engine._decisions_phase(plan)

    assert answers == {"DW-3": good}
    [reload_failed] = _records(engine, "sweep-decisions-reload-failed")
    assert reload_failed["errors"] == [
        "DW-1: effect not recognized (<run>/decisions.json)",
        "DW-2: intent not a string: list (project .bmad-loop/decisions.json)",
    ]
    # the seeded write-back re-published DW-1 verbatim beside the answer it took on
    assert json.loads(run_store.read_text(encoding="utf-8")) == {
        "DW-1": {"effect": "frobnicate"},
        "DW-3": good,
    }
    # and the PROJECT store is untouched — this phase repairs neither file
    assert json.loads(pre_store.read_text(encoding="utf-8")) == {"DW-2": bad_pre, "DW-3": good}
    assert sorted(r["dw_id"] for r in _records(engine, "decision-skipped-unattended")) == [
        "DW-1",
        "DW-2",
    ]


def test_materialize_bundles_never_derives_an_intent_from_a_non_string(project):
    """DW-141. `str(answer.get("intent", ""))` never failed on a list — it produced
    the truthy repr `"['a', 'b']"`, which passed the `if not intent` gate and
    shipped into `Bundle.intent`, `intent.md` and a dev session as prose no human
    wrote. Read as `""` instead, it falls through the existing chain: no agreeing
    option here (stored key "9" is not offered), so the existing `no-intent` drop
    fires — no new branch and no new drop cause. The lane guard stays because the
    suite hands `_materialize_bundles` a map directly, past the read site above.
    Ablation: restore `str(answer.get("intent", ""))` and this reddens with a
    bundle whose intent is the repr."""
    write_ledger(project, {"DW-1": "open"})
    plan = TriagePlan(open_ids=frozenset({"DW-1"}), decisions=(_decision_for("DW-1"),))
    engine, _ = make_sweep(project, [])
    engine.run_dir.mkdir(parents=True, exist_ok=True)
    (engine.run_dir / "journal.jsonl").write_text("", encoding="utf-8")

    bundles, dropped = engine._materialize_bundles(
        plan, {"DW-1": {"key": "9", "effect": "build", "intent": ["a", "b"]}}
    )

    assert bundles == []
    assert dropped  # the drop is repeat progress (DW-135)
    assert [r["drop_cause"] for r in _records(engine, "sweep-decision-answer-dropped")] == [
        "no-intent"
    ]


def test_materialize_bundles_falls_back_to_the_agreeing_option_for_non_string_scalars(project):
    """DW-141's other half: a non-string scalar must not defeat the fallback the
    site already has. The answer agrees with option "1" on label+effect, so the
    list `intent` and the int `bundle_name` both read `""` and the OPTION supplies
    each — the bundle carries the option's authored intent and its validated name,
    with no `sweep-bundle-name-discarded` record (the repr "7" fails
    `BUNDLE_NAME_RE`, so the old spelling discarded the name and fell back to
    `decision-dw-1`). Ablation: restore either `str(answer.get(...))` and this
    reddens on `intent` or on `name`."""
    write_ledger(project, {"DW-1": "open"})
    option = DecisionOption(
        key="1", label="Widen", effect="build", intent="widen the field", bundle_name="widen-x"
    )
    decision = Decision(id="DW-1", question="q", context="", options=(option,), recommendation="1")
    plan = TriagePlan(open_ids=frozenset({"DW-1"}), decisions=(decision,))
    engine, _ = make_sweep(project, [])
    engine.run_dir.mkdir(parents=True, exist_ok=True)
    (engine.run_dir / "journal.jsonl").write_text("", encoding="utf-8")

    answer = {
        "key": "1",
        "label": "Widen",
        "effect": "build",
        "intent": ["a", "b"],
        "bundle_name": 7,
    }
    bundles, dropped = engine._materialize_bundles(plan, {"DW-1": answer})

    assert not dropped
    assert [(b.name, b.intent) for b in bundles] == [("widen-x", "widen the field")]
    assert _records(engine, "sweep-bundle-name-discarded") == []
    assert _records(engine, "sweep-decision-answer-dropped") == []


def test_keep_open_lane_does_not_resolve_an_option_from_a_non_string_key(project):
    """DW-141, the keep-open lane. `str(answer.get("key", ""))` turned the JSON
    number 1 into "1" and resolved option "1" by coincidence of repr, letting a
    malformed answer suppress a bundle under a `human-chose-keep-open` skip that
    reads as the human's decision. Read as `""` it resolves nothing, so the lane's
    existing `stale-option` drop fires and the bundle survives.
    Ablation: restore `str(answer.get("key", ""))` and this reddens — the bundle
    is suppressed and nothing is dropped. The decision's sole option is written to
    AGREE with the answer on label and effect, so the key read is the only thing
    deciding the outcome; `_decision_for`'s build option would have failed the
    effect comparison instead and hidden the ablation."""
    write_ledger(project, {"DW-1": "open"})
    decision = Decision(
        id="DW-1",
        question="q",
        context="",
        options=(DecisionOption(key="1", label="Keep", effect="keep-open"),),
        recommendation="1",
    )
    plan = TriagePlan(
        open_ids=frozenset({"DW-1"}),
        bundles=(Bundle(name="safe-fix", dw_ids=("DW-1",), intent="a"),),
        decisions=(decision,),
    )
    engine, _ = make_sweep(project, [])
    engine.run_dir.mkdir(parents=True, exist_ok=True)
    (engine.run_dir / "journal.jsonl").write_text("", encoding="utf-8")

    bundles, dropped = engine._materialize_bundles(
        plan, {"DW-1": {"key": 1, "label": "Keep", "effect": "keep-open"}}
    )

    assert [b.name for b in bundles] == ["safe-fix"]  # not suppressed
    assert dropped
    assert [r["drop_cause"] for r in _records(engine, "sweep-decision-answer-dropped")] == [
        "stale-option"
    ]


# ------------------------------- DW-135: every drop lane is repeat progress


def test_dropped_build_answer_is_repeat_progress_exactly_once(project):
    """DW-135. Cycle 2 renumbers key "1" off `Widen`/`build` and onto
    `Close as decayed`/`close`, so the in-run build answer no longer resolves to an
    agreeing option and carries no intent of its own — the `no-intent` drop. That
    drop is cycle 2's ONLY event: it must count as progress by itself, or the run
    stops at `no-progress` before cycle 3's fresh triage ever gets to look at DW-1.
    Cycle 3 then proves the bound: DW-1 is already quarantined, so its re-agreeing
    option produces no second drop, no second notification and no second progress
    signal, and the repeat loop terminates.
    Ablation: drop the `answer_dropped = True` from the `no-intent` lane and this
    reddens — only one `sweep-cycle` record is written and `sweep-repeat-done`
    reports 1 cycle."""
    write_ledger(project, {"DW-1": "open", "DW-2": "open"})
    plan1 = triage_result(
        ["DW-1", "DW-2"],
        bundles=[{"name": "safe-fix", "dw_ids": ["DW-2"], "intent": "a"}],
        decisions=[
            _decision(
                "DW-1",
                [
                    {"key": "1", "label": "Widen", "effect": "build", "intent": "widen the field"},
                    {"key": "2", "label": "Keep", "effect": "keep-open"},
                ],
            )
        ],
    )
    # cycle 2: no bundles, no closes, no new decisions to answer — the drop is
    # the whole of its addressable work
    plan2 = triage_result(
        ["DW-1"],
        decisions=[
            _decision(
                "DW-1",
                [
                    {"key": "1", "label": "Close as decayed", "effect": "close"},
                    {"key": "2", "label": "Keep", "effect": "keep-open"},
                ],
            )
        ],
    )
    # cycle 3 re-authors key "1" back into agreement: the quarantine must refuse
    # it, leaving the cycle with nothing addressable
    plan3 = triage_result(
        ["DW-1"],
        decisions=[
            _decision(
                "DW-1",
                [
                    {"key": "1", "label": "Widen", "effect": "build", "intent": "widen the field"},
                    {"key": "2", "label": "Keep", "effect": "keep-open"},
                ],
            )
        ],
    )
    engine, _ = make_sweep(
        project,
        [
            triage_effect(plan1),
            bundle_dev_effect(project, "safe-fix", ["DW-2"]),
            bundle_review_effect(project, "safe-fix"),
            triage_effect(plan2),
            triage_effect(plan3),
        ],
        # max_bundles=1 holds cycle 1's `decision-dw-1` bundle back (truncated), so
        # the answer is still unspent when cycle 2 renumbers its option
        policy=repeat_policy(max_bundles=1),
        answers=["1"],
        prompting=True,
    )
    summary = engine.run()
    assert not summary.paused

    journal = journal_text(engine)
    # PRECONDITIONS: the in-run answer was recorded and cycle 1 truncated its bundle
    assert '"decision-answered"' in journal
    assert '"sweep-bundles-truncated"' in journal
    assert engine.state.tasks["dw-safe-fix"].phase == Phase.DONE
    # THE BUG: cycle 2's only event was the drop, and cycle 3 still ran
    assert [r["cycle"] for r in _records(engine, "sweep-cycle")] == [2, 3]
    [drop] = _records(engine, "sweep-decision-answer-dropped")
    assert drop["decision"] == "DW-1" and drop["drop_cause"] == "no-intent"
    # THE BOUND: one drop, one mismatch, one notification, and cycle 3 found
    # nothing addressable, so the loop stopped instead of spinning
    assert len(_mismatches(engine)) == 1
    attention = (engine.run_dir / "ATTENTION").read_text(encoding="utf-8")
    assert attention.count("recorded build decision discarded") == 1
    [done] = _records(engine, "sweep-repeat-done")
    assert done["reason"] == "no-progress" and done["cycles"] == 3
    assert not any(k.startswith("dw3-") for k in engine.state.tasks)
    assert ledger_entries(project)["DW-1"].open


def test_dropped_name_collision_answer_is_repeat_progress(project):
    """DW-135's OTHER build lane. `test_dropped_build_answer_is_repeat_progress...`
    reaches the flag through `no-intent`; this row reaches it through
    `name-collision`, where the answer is perfectly buildable and only the names
    are taken. Cycle 2 authors `decision-dw-1` and every `-2`..`-9` suffix, so the
    bound is exhausted and the answer is dropped — and those nine plan bundles are
    all suppressed by keep-open answers (the taken-name set is built during the
    decision loop, the keep-open skip runs after it), so the drop is the ONLY
    addressable thing cycle 2 does. Cycle 3 then finds DW-1 quarantined and stops.
    Ablation: delete `answer_dropped = True` from the name-collision lane and this
    reddens — cycle 3 never runs and `sweep-repeat-done` reports 2 cycles."""
    ids = [f"DW-{n}" for n in range(1, 12)]  # DW-1 decides; DW-2..10 occupy; DW-11 runs
    write_ledger(project, {i: "open" for i in ids})
    decision = _decision(
        "DW-1",
        [
            {"key": "1", "label": "Widen", "effect": "build", "intent": "fresh intent"},
            {"key": "2", "label": "Keep", "effect": "keep-open"},
        ],
    )
    # `decision-dw-1` plus every one of `-2` ... `-9`: the whole bound, taken
    occupied = ["decision-dw-1"] + [f"decision-dw-1-{n}" for n in range(2, 10)]
    plan1 = triage_result(
        ids,
        bundles=[{"name": "safe-fix", "dw_ids": ["DW-11"], "intent": "a"}],
        skip=[{"id": i, "reason": "not this cycle"} for i in ids[1:10]],
        decisions=[decision],
    )
    plan2 = triage_result(
        ids[:10],
        bundles=[
            {"name": name, "dw_ids": [dw_id], "intent": f"plan intent {dw_id}"}
            for name, dw_id in zip(occupied, ids[1:10], strict=True)
        ],
        decisions=[decision],
    )
    plan3 = triage_result(
        ids[:10],
        skip=[{"id": i, "reason": "not this cycle"} for i in ids[1:10]],
        decisions=[decision],
    )
    engine, _ = make_sweep(
        project,
        [
            triage_effect(plan1),
            bundle_dev_effect(project, "safe-fix", ["DW-11"]),
            bundle_review_effect(project, "safe-fix"),
            triage_effect(plan2),
            triage_effect(plan3),
        ],
        # max_bundles=1 holds cycle 1's free `decision-dw-1` bundle back (truncated),
        # so the answer is still unspent when cycle 2 takes every name it could use
        policy=repeat_policy(max_bundles=1),
    )
    engine.run_dir.mkdir(parents=True, exist_ok=True)
    (engine.run_dir / "decisions.json").write_text(
        json.dumps(
            {
                "DW-1": {"key": "1", "label": "Widen", "effect": "build"},
                # no decision re-asks these, so they suppress their bundles outright
                **{i: {"key": "2", "label": "Keep", "effect": "keep-open"} for i in ids[1:10]},
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    summary = engine.run()
    assert not summary.paused

    # PRECONDITIONS: the AGREEING lane, the bound exhausted rather than suffixed,
    # and cycle 2 ran no bundle of any kind
    assert _mismatches(engine) == []
    assert _records(engine, "sweep-bundle-name-deduped") == []
    assert engine.state.tasks["dw-safe-fix"].phase == Phase.DONE
    assert not any(k.startswith("dw2-") for k in engine.state.tasks)
    # THE BUG: cycle 2's only addressable event was the name-collision drop
    [drop] = _records(engine, "sweep-decision-answer-dropped")
    assert drop["decision"] == "DW-1" and drop["drop_cause"] == "name-collision"
    assert [r["cycle"] for r in _records(engine, "sweep-cycle")] == [2, 3]
    # THE BOUND: cycle 3 found DW-1 quarantined, so the loop stopped
    [done] = _records(engine, "sweep-repeat-done")
    assert done["reason"] == "no-progress" and done["cycles"] == 3
    attention = (engine.run_dir / "ATTENTION").read_text(encoding="utf-8")
    assert attention.count("recorded build decision discarded") == 1
    assert ledger_entries(project)["DW-1"].open


def test_repeat_resume_mid_cycle_two(project):
    write_ledger(project, {"DW-1": "open"})
    plan1 = triage_result(
        ["DW-1"], bundles=[{"name": "first-fix", "dw_ids": ["DW-1"], "intent": "a"}]
    )
    plan2 = triage_result(
        ["DW-2"], bundles=[{"name": "follow-up", "dw_ids": ["DW-2"], "intent": "b"}]
    )

    def escalating_dev(spec):
        return SessionResult(
            status="completed",
            result_json={
                "workflow": "auto-dev",
                "escalations": [
                    {"type": "bundle-item-blocked", "severity": "CRITICAL", "detail": "no"}
                ],
            },
        )

    engine, _ = make_sweep(
        project,
        [
            triage_effect(plan1),
            appending_dev(project, bundle_dev_effect(project, "first-fix", ["DW-1"]), "DW-2"),
            bundle_review_effect(project, "first-fix"),
            triage_effect(plan2),
            escalating_dev,
        ],
        policy=repeat_policy(),
    )
    summary = engine.run()
    assert summary.paused
    assert load_state(engine.run_dir).sweep_cycle == 2
    assert engine.state.tasks["dw2-follow-up"].phase == Phase.ESCALATED

    resumed, adapter = resume_sweep(project, engine, [])
    summary = resumed.run()
    assert not summary.paused
    # resume re-enters cycle 2 directly: triage-2.json reloads (no session),
    # the escalated bundle is dropped by the failed-ids filter, and the cycle
    # reports no progress
    assert adapter.sessions == []
    journal = journal_text(resumed)
    assert "sweep-bundle-skipped" in journal and "no-progress" in journal
    assert ledger_entries(project)["DW-2"].open  # escalated bundle untouched


def test_repeat_decisions_only_single_cycle(project):
    write_ledger(project, {"DW-1": "open"})
    plan = triage_result(
        ["DW-1"], bundles=[{"name": "one-fix", "dw_ids": ["DW-1"], "intent": "fix"}]
    )
    engine, adapter = make_sweep(
        project, [triage_effect(plan)], policy=repeat_policy(), decisions_only=True
    )
    summary = engine.run()
    assert not summary.paused
    assert len(adapter.sessions) == 1
    journal = journal_text(engine)
    assert "sweep-decisions-only" in journal and "sweep-cycle" not in journal


def test_repeat_unattended_decision_notifies_once(project):
    write_ledger(project, {"DW-1": "open", "DW-2": "open"})
    decision = {
        "id": "DW-1",
        "question": "q",
        "context": "",
        "options": [
            {"key": "1", "label": "a", "effect": "build", "intent": "x"},
            {"key": "2", "label": "b", "effect": "keep-open"},
        ],
        "recommendation": "2",
    }
    plan1 = triage_result(
        ["DW-1", "DW-2"],
        bundles=[{"name": "safe-fix", "dw_ids": ["DW-2"], "intent": "fix"}],
        decisions=[decision],
    )
    plan2 = triage_result(["DW-1"], decisions=[decision])
    engine, _ = make_sweep(
        project,
        [
            triage_effect(plan1),
            bundle_dev_effect(project, "safe-fix", ["DW-2"]),
            bundle_review_effect(project, "safe-fix"),
            triage_effect(plan2),
        ],
        policy=repeat_policy(),
        prompting=False,
    )
    summary = engine.run()
    assert not summary.paused
    assert journal_text(engine).count("decision-skipped-unattended") == 1
    assert "no-progress" in journal_text(engine)


def test_repeat_truncated_bundles_picked_up_next_cycle(project):
    write_ledger(project, {"DW-1": "open", "DW-2": "open"})
    plan1 = triage_result(
        ["DW-1", "DW-2"],
        bundles=[
            {"name": "first-fix", "dw_ids": ["DW-1"], "intent": "a"},
            {"name": "second-fix", "dw_ids": ["DW-2"], "intent": "b"},
        ],
    )
    plan2 = triage_result(
        ["DW-2"], bundles=[{"name": "second-fix", "dw_ids": ["DW-2"], "intent": "b"}]
    )
    engine, _ = make_sweep(
        project,
        [
            triage_effect(plan1),
            bundle_dev_effect(project, "first-fix", ["DW-1"]),
            bundle_review_effect(project, "first-fix"),
            triage_effect(plan2),
            bundle_dev_effect(project, "second-fix", ["DW-2"]),
            bundle_review_effect(project, "second-fix"),
        ],
        policy=repeat_policy(max_bundles=1),
    )
    summary = engine.run()
    assert not summary.paused
    assert "sweep-bundles-truncated" in journal_text(engine)
    # same bundle name across cycles lands under distinct task keys
    assert engine.state.tasks["dw-first-fix"].phase == Phase.DONE
    assert engine.state.tasks["dw2-second-fix"].phase == Phase.DONE
    entries = ledger_entries(project)
    assert entries["DW-1"].status.startswith("done")
    assert entries["DW-2"].status.startswith("done")


# ------------------------------------------------------- legacy migration


def test_sweep_migrates_legacy_then_triages_and_runs_bundle(project):
    write_legacy_ledger(project, LEGACY_LEDGER)
    manifest = legacy_manifest()
    mapping = [
        {"key": manifest[0]["key"], "dw_id": "DW-1"},
        {"key": manifest[1]["key"], "dw_id": "DW-2"},
    ]
    plan = triage_result(
        ["DW-2"],
        bundles=[{"name": "fix-emdash", "dw_ids": ["DW-2"], "intent": "guard em-dashes"}],
    )
    engine, adapter = make_sweep(
        project,
        [
            migrate_effect(project, migrated_ledger(), mapping),
            triage_effect(plan),
            bundle_dev_effect(project, "fix-emdash", ["DW-2"]),
            bundle_review_effect(project, "fix-emdash"),
        ],
    )
    summary = engine.run()

    assert not summary.paused
    tasks = engine.state.tasks
    assert tasks["sweep-migrate"].phase == Phase.DONE
    assert tasks["sweep-triage"].phase == Phase.DONE
    assert tasks["dw-fix-emdash"].phase == Phase.DONE

    text = project.deferred_work.read_text(encoding="utf-8")
    assert not deferredwork.has_legacy(text)
    entries = ledger_entries(project)
    assert entries["DW-1"].status.startswith("done")
    assert entries["DW-2"].status.startswith("done")  # bundle closed it

    log = git(project.project, "log", "--oneline")
    assert "chore(sweep): migrate legacy deferred-work entries to DW format" in log
    journal = journal_text(engine)
    assert "sweep-migrated" in journal and "sweep-nothing-open" not in journal

    # the migration session was prompted with the manifest path
    assert "--migrate" in adapter.sessions[0].prompt
    manifest_path = adapter.sessions[0].prompt.split("--migrate ", 1)[1].split()[0]
    written = json.loads(open(manifest_path).read())
    assert [m["key"] for m in written] == [m["key"] for m in manifest]
    # triage ran against the post-migration open set, strict check intact
    assert "--migrate" not in adapter.sessions[1].prompt


def test_severity_selector_applies_to_the_post_migration_ledger(project):
    legacy = (
        "# Deferred Work\n\n"
        "### D-1: Low legacy\n\nseverity: low\nreason: low item\n\n"
        "### D-2: High legacy\n\nseverity: high\nreason: high item\n"
    )
    write_legacy_ledger(project, legacy)
    manifest = legacy_manifest(legacy)
    assert [item["severity"] for item in manifest] == ["low", "high"]
    mapping = [
        {"key": manifest[0]["key"], "dw_id": "DW-1"},
        {"key": manifest[1]["key"], "dw_id": "DW-2"},
    ]
    migrated = (
        "# Deferred Work\n\n"
        "### DW-1: Low legacy\n\norigin: migrated\nseverity: low\nstatus: open\n\n"
        "### DW-2: High legacy\n\norigin: migrated\nseverity: high\nstatus: open\n"
    )
    plan = triage_result(
        ["DW-2"],
        skip=[{"id": "DW-2", "reason": "selected high entry"}],
    )
    engine, adapter = make_sweep(
        project,
        [migrate_effect(project, migrated, mapping), triage_effect(plan)],
        min_severity="high",
    )

    summary = engine.run()

    assert not summary.crashed and not summary.paused
    assert adapter.sessions[1].prompt == "/bmad-loop-sweep --only DW-2"
    excluded = [
        entry for entry in engine.journal.entries() if entry["kind"] == "sweep-selection-excluded"
    ]
    assert excluded[0]["dw_ids"] == ["DW-1"]


def test_only_revalidates_against_compacted_post_migration_ids(project):
    legacy = (
        "# Deferred Work\n\n"
        "### D-1: Duplicate wording one\n\nreason: same issue\n\n"
        "### D-2: Duplicate wording two\n\nreason: same issue\n"
    )
    write_legacy_ledger(project, legacy)
    manifest = legacy_manifest(legacy)
    mapping = [{"key": item["key"], "dw_id": "DW-1"} for item in manifest]
    migrated = (
        "# Deferred Work\n\n"
        "### DW-1: merged duplicate\n\norigin: migrated\nreason: same issue\nstatus: open\n"
    )
    engine, adapter = make_sweep(
        project,
        [migrate_effect(project, migrated, mapping)],
        only_ids=("DW-2",),
    )

    summary = engine.run()

    assert summary.crashed and not summary.paused
    assert len(adapter.sessions) == 1
    assert adapter.sessions[0].role == "triage" and "--migrate" in adapter.sessions[0].prompt
    assert "must exist and be open: DW-2" in (engine.run_dir / "crash.txt").read_text()


def test_migration_validation_failure_restores_ledger_then_escalates(project):
    write_legacy_ledger(project, LEGACY_LEDGER)
    manifest = legacy_manifest()
    # converts only the done item; the open one remains legacy -> invalid
    half = (
        "# Deferred Work\n\n"
        "### DW-1: Old fixed thing\n\norigin: migrated, 2026-06-12\nlocation: n/a\n"
        "reason: repaired.\nstatus: done 2026-04-06\n\n"
        "## Deferred from: epic 1 review (2026-04-06)\n\n"
        "- **Open legacy thing here** — `src.txt` mishandles em-dashes\n"
    )
    bad = migrate_effect(project, half, [{"key": manifest[0]["key"], "dw_id": "DW-1"}])
    engine, adapter = make_sweep(project, [bad, bad])
    summary = engine.run()

    assert summary.paused
    assert engine.state.tasks["sweep-migrate"].phase == Phase.ESCALATED
    # the broken rewrite never sticks: original ledger text restored
    assert project.deferred_work.read_text(encoding="utf-8") == LEGACY_LEDGER
    assert worktree_clean(project.project)
    prompts = [s.prompt for s in adapter.sessions]
    assert len(prompts) == 2
    assert "--feedback" not in prompts[0] and "--feedback" in prompts[1]
    feedback = open(prompts[1].split("--feedback ", 1)[1]).read()
    assert "still parse as legacy" in feedback and "not mapped" in feedback


def test_migration_restore_write_failure_propagates_and_keeps_the_ledger(project, monkeypatch):
    """#328. The restore after a failed migration is a repair write: it raises
    rather than degrades, and it must never be the thing that empties the ledger
    it exists to put back. Under a bare `Path.write_text` the truncate landed
    before the failure did — the run crashed with the ledger at zero bytes, and
    the `_safe_reset` that would have restored it was already spent.

    The patch is module-wide but reaches exactly one call — re-probed on this
    harness by COUNTING calls through the stub, not by reading line order. The
    raising stub fires once, on the ledger restore.

    `sweep.atomic_write_text` has four call sites and the other three are all out
    of reach here: the two `_decisions_phase` writes (#363) and the bundle-intent
    write each need a triage plan carrying decisions or bundles, and this run
    crashes in `_ensure_migration` before the decisions phase is entered.
    Delegating instead of raising confirms it — the run then reaches two writes,
    both the ledger, one per migration attempt.

    The stub takes `**_kw` because those decisions writes pass
    `follow_symlinks=False`. Two positional parameters are enough for the restore
    (it passes neither keyword), but should a keyword-passing site ever become
    reachable here, a narrow stub would meet it as a TypeError — the run would
    still crash, just on the wrong fault, reddening the "disk full" assertion in a
    way that points nowhere near the cause."""
    write_legacy_ledger(project, LEGACY_LEDGER)
    manifest = legacy_manifest()
    half = (
        "# Deferred Work\n\n"
        "### DW-1: Old fixed thing\n\norigin: migrated, 2026-06-12\nlocation: n/a\n"
        "reason: repaired.\nstatus: done 2026-04-06\n\n"
        "## Deferred from: epic 1 review (2026-04-06)\n\n"
        "- **Open legacy thing here** — `src.txt` mishandles em-dashes\n"
    )
    bad = migrate_effect(project, half, [{"key": manifest[0]["key"], "dw_id": "DW-1"}])
    engine, _ = make_sweep(project, [bad, bad])

    def boom(path, text, **_kw):
        raise OSError("disk full")

    monkeypatch.setattr("bmad_loop.sweep.atomic_write_text", boom)

    summary = engine.run()

    assert summary.crashed and "disk full" in str(summary.crash_error)
    # compared as TEXT, not bytes: the fixture reached disk through `write_text`,
    # so a byte assertion would read CRLF on Windows and redden there only
    assert project.deferred_work.read_text(encoding="utf-8") == LEGACY_LEDGER
    assert project.deferred_work.read_bytes()  # and emphatically not zero bytes


@contextlib.contextmanager
def _rival_appending_ledger_lock(monkeypatch, ledger, addition):
    """A `ledger_lock` spy that lands one foreign append BEFORE acquiring, and
    still really locks.

    Ahead of the acquisition on purpose: that is the window the migration
    restore's compare-and-set covers — between the post-reset observation and the
    hold. Still really locking for `_counting_ledger_lock`'s reason: a spy that
    only staged the rival would let a nested acquisition through. One-shot, so a
    later acquisition cannot file the rival twice.
    """
    real_lock = deferredwork.ledger_lock
    landed: list[bool] = []

    @contextlib.contextmanager
    def spy_lock(path):
        if path == ledger and not landed:
            landed.append(True)
            with ledger.open("a", encoding="utf-8") as f:
                f.write(addition)
        with real_lock(path):
            yield

    monkeypatch.setattr(deferredwork, "ledger_lock", spy_lock)
    yield landed


def _half_migrated_effect(project):
    """A migration rewrite that converts only the done item, so the open one
    stays legacy and `validate_migration` rejects it — the restore's trigger."""
    manifest = legacy_manifest()
    half = (
        "# Deferred Work\n\n"
        "### DW-1: Old fixed thing\n\norigin: migrated, 2026-06-12\nlocation: n/a\n"
        "reason: repaired.\nstatus: done 2026-04-06\n\n"
        "## Deferred from: epic 1 review (2026-04-06)\n\n"
        "- **Open legacy thing here** — `src.txt` mishandles em-dashes\n"
    )
    return migrate_effect(project, half, [{"key": manifest[0]["key"], "dw_id": "DW-1"}])


def test_migration_restore_escalates_on_divergence(project, monkeypatch):
    """#286. The failed migration's restore is compare-and-set: when the ledger
    moved between the reset and the lock, the pre-migration text is NOT written
    back over it and a human is asked to re-run the sweep.

    No merge and no silent skip, and the site's own rule is why. Leaving the
    rejected rewrite standing would be re-prompting over a half-broken ledger,
    and a migration input that changed underneath the attempt it was graded
    against is a human problem — the same call `migrate-duplicate-ids` makes.

    Ablation: drop the `anchored and current == expected` arm so the restore
    always writes, and the rival's line is clobbered while no escalation is
    raised.
    """
    write_legacy_ledger(project, LEGACY_LEDGER)
    engine, adapter = make_sweep(project, [_half_migrated_effect(project)] * 2)
    rival = "- **Filed by another process** — `other.txt` needs a look\n"
    with _rival_appending_ledger_lock(monkeypatch, project.deferred_work, rival) as landed:
        summary = engine.run()

    assert summary.paused and landed == [True]
    assert engine.state.tasks["sweep-migrate"].phase == Phase.ESCALATED
    assert "changed underneath the failed migration attempt" in engine.state.paused_reason
    assert "sweep-migration-restore-diverged" in journal_kinds(engine)
    # the rival's line stands, and the refusal did not paper the rewrite over
    text = project.deferred_work.read_text(encoding="utf-8")
    assert "Filed by another process" in text and text != LEGACY_LEDGER
    # the refusal is terminal for this run: the second attempt never dispatches
    assert len(adapter.sessions) == 1


@contextlib.contextmanager
def _rival_appending_safe_reset(monkeypatch, engine, ledger, addition):
    """A `_safe_reset` spy that lands one foreign append the instant the real
    reset returns, and still really resets.

    That instant is where the migration restore's window opens: `_safe_reset`
    returning is the last event before the restore decides what the ledger is
    supposed to contain. A rival there is precisely what #735 describes, and it
    is the half the old `current == observed` anchor could not see — the rival
    became the observation. One-shot, because this wrapper is re-entered on every
    migration attempt and the retry must not file it twice.
    """
    real_reset = engine._safe_reset
    landed: list[bool] = []

    def reset_then_rival(task, **kwargs):
        real_reset(task, **kwargs)
        if not landed:
            landed.append(True)
            with ledger.open("a", encoding="utf-8") as f:
                f.write(addition)

    monkeypatch.setattr(engine, "_safe_reset", reset_then_rival)
    yield landed


def test_migration_restore_escalates_when_a_rival_writes_inside_the_reset_window(
    project, monkeypatch
):
    """#735, tracked. The twin above lands its rival at the lock, where the old
    anchor already refused. This one lands it in the window the old anchor was
    blind to: between `_safe_reset` returning and the read that graded it.

    A rival writing a TRACKED ledger there BECOMES `observed`, so the compare
    holds under the lock and the pre-migration text is written straight over it —
    and the run then RETRIES over a ledger no human has looked at. The anchor is
    the committed blob at `task.baseline_commit` instead, which is what the reset
    republished and which no rival can author.

    Ablation: restore the `observed` read after `_safe_reset` and compare
    `current == observed` — the rival's line is clobbered, the escalation never
    raises, and the second attempt dispatches. Every row below reds.
    """
    write_legacy_ledger(project, LEGACY_LEDGER)
    engine, adapter = make_sweep(project, [_half_migrated_effect(project)] * 2)
    rival = "- **Filed by another process** — `other.txt` needs a look\n"
    with _rival_appending_safe_reset(monkeypatch, engine, project.deferred_work, rival) as landed:
        summary = engine.run()

    assert summary.paused and landed == [True]
    assert engine.state.tasks["sweep-migrate"].phase == Phase.ESCALATED
    assert "changed underneath the failed migration attempt" in engine.state.paused_reason
    assert "sweep-migration-restore-diverged" in journal_kinds(engine)
    # the rival's line stands, and the refusal did not paper the rewrite over
    text = project.deferred_work.read_text(encoding="utf-8")
    assert "Filed by another process" in text and text != LEGACY_LEDGER
    # the refusal is terminal for this run: the second attempt never dispatches
    assert len(adapter.sessions) == 1


def test_migration_restore_escalates_for_an_untracked_rival_inside_the_reset_window(
    project, monkeypatch
):
    """#735, untracked. An untracked ledger has no blob to anchor on, and
    `reset --hard` cannot have put anything back into it either — so the anchor
    is the rejected rewrite this attempt actually graded, which is the one text
    here that predates the window.

    Same rival, same window, same refusal: the point is that losing the blob does
    NOT send this site back to trusting the observation. `_ledger_baseline_text`
    answers determinate absence (`(True, None)`) rather than "no anchor", and the
    rewrite fills the slot.

    Ablation: restore the `observed` read after `_safe_reset` and compare
    `current == observed` — the rival is clobbered and the run retries.
    """
    write_legacy_ledger(project, LEGACY_LEDGER, commit=False)
    engine, adapter = make_sweep(project, [_half_migrated_effect(project)] * 2)
    rival = "- **Filed by another process** — `other.txt` needs a look\n"
    with _rival_appending_safe_reset(monkeypatch, engine, project.deferred_work, rival) as landed:
        summary = engine.run()

    assert summary.paused and landed == [True]
    assert engine.state.tasks["sweep-migrate"].phase == Phase.ESCALATED
    assert "changed underneath the failed migration attempt" in engine.state.paused_reason
    assert "sweep-migration-restore-diverged" in journal_kinds(engine)
    text = project.deferred_work.read_text(encoding="utf-8")
    assert "Filed by another process" in text and text != LEGACY_LEDGER
    assert len(adapter.sessions) == 1


def test_migration_restore_escalates_when_the_baseline_probe_fails(project, monkeypatch):
    """DIRECTION PIN, #735: an unprovable baseline escalates, it never falls back
    to the observation.

    No rival anywhere — the quiet fixture of the positive control below, with
    only the probe faulted. Two failed probes' worth of uncertainty is the most
    this site can have, and the answer to it is the same one a rival gets: refuse
    the write and put it in front of a human. Unlike the defer restore there is
    no append-only merge to degrade into — republishing the pre-migration text is
    an overwrite or it is nothing — so the escalation IS the degrade, and the
    resume it routes to resets the attempt budget and re-reads the ledger.

    Ablation: invert the fault direction — have `_ledger_baseline_text`'s except
    arm return `(True, self._ledger_text())` — and the restore writes, the run
    retries, and every row below reds.
    """
    write_legacy_ledger(project, LEGACY_LEDGER)
    engine, adapter = make_sweep(project, [_half_migrated_effect(project)] * 2)

    def fail_probe(*args, **kwargs):
        raise verify.GitError("injected baseline probe failure")

    monkeypatch.setattr(verify, "worktree_file_bytes_at_revision", fail_probe)

    summary = engine.run()

    assert summary.paused
    assert engine.state.tasks["sweep-migrate"].phase == Phase.ESCALATED
    assert "changed underneath the failed migration attempt" in engine.state.paused_reason
    kinds = journal_kinds(engine)
    assert "ledger-baseline-probe-failed" in kinds
    assert "sweep-migration-restore-diverged" in kinds
    # escalated on the FIRST failed attempt: an unprovable restore does not get
    # to spend the retry budget over a ledger nobody has graded
    assert len(adapter.sessions) == 1


def test_migration_restore_quiet_path_unchanged(project):
    """#286. With no interleaving writer the restore still puts the
    pre-migration text back byte for byte, and journals no divergence — the CAS
    is invisible on the path every migration failure actually takes.

    The ledger is deliberately left UNTRACKED. `git reset --hard` puts a tracked
    one back on its own, so the explicit write is the only thing standing between
    a rejected rewrite and disk here — over a committed ledger this assertion
    passes with the write deleted, which is what the site's own comment predicts
    ("the baseline reset covers tracked files, the explicit write covers an
    untracked ledger that `git reset` cannot restore").

    Ablation: delete the restore write and the rejected rewrite stays on disk;
    force the diverged arm and the journal row appears.
    """
    write_legacy_ledger(project, LEGACY_LEDGER, commit=False)
    engine, _ = make_sweep(project, [_half_migrated_effect(project)] * 2)
    summary = engine.run()

    assert summary.paused
    assert project.deferred_work.read_text(encoding="utf-8") == LEGACY_LEDGER
    assert "sweep-migration-restore-diverged" not in journal_kinds(engine)


def test_migration_escalation_resume_retries(project):
    write_legacy_ledger(project, LEGACY_LEDGER)
    manifest = legacy_manifest()
    bad = migrate_effect(project, LEGACY_LEDGER, [])  # no conversion at all
    engine, first = make_sweep(project, [bad, bad])
    assert engine.run().paused
    abandoned = [s.task_id for s in first.sessions]
    assert abandoned == ["sweep-migrate-triage-1", "sweep-migrate-triage-2"]

    mapping = [
        {"key": manifest[0]["key"], "dw_id": "DW-1"},
        {"key": manifest[1]["key"], "dw_id": "DW-2"},
    ]
    plan = triage_result(["DW-2"], skip=[{"id": "DW-2", "reason": "moot"}])
    resumed, adapter = resume_sweep(
        project,
        engine,
        [migrate_effect(project, migrated_ledger(), mapping), triage_effect(plan)],
    )
    summary = resumed.run()
    assert not summary.paused
    assert resumed.state.tasks["sweep-migrate"].phase == Phase.DONE
    assert resumed.state.tasks["sweep-triage"].phase == Phase.DONE
    assert len(adapter.sessions) == 2
    # the ESCALATED-resume opened a new generation of the migrate task, so its
    # restarted attempt 1 does not re-mint the abandoned attempt 1's id
    assert resumed.state.tasks["sweep-migrate"].generation == 1
    migrate_id = adapter.sessions[0].task_id
    assert migrate_id == "sweep-migrate-triage-1-g1"
    assert migrate_id not in abandoned


def test_no_legacy_skips_migration(project):
    write_ledger(project, {"DW-1": "open"})
    plan = triage_result(["DW-1"], skip=[{"id": "DW-1", "reason": "moot"}])
    engine, adapter = make_sweep(project, [triage_effect(plan)])
    assert not engine.run().paused
    assert "sweep-migrate" not in engine.state.tasks
    assert "--migrate" not in adapter.sessions[0].prompt


def test_mixed_ledger_migration_preserves_canonical_open_set(project):
    mixed = (
        "# Deferred Work\n\n"
        "### DW-1: item DW-1\n\norigin: test, 2026-06-01\nlocation: src.txt:1\n"
        "reason: test entry.\nstatus: open\n\n"
        "## Deferred from: epic 1 review (2026-04-06)\n\n"
        "- **Open legacy thing here** — `src.txt` mishandles em-dashes\n"
    )
    write_legacy_ledger(project, mixed)
    manifest = legacy_manifest(mixed)
    assert len(manifest) == 1  # the canonical entry is not a legacy item
    migrated = (
        "# Deferred Work\n\n"
        "### DW-1: item DW-1\n\norigin: test, 2026-06-01\nlocation: src.txt:1\n"
        "reason: test entry.\nstatus: open\n\n"
        "### DW-2: Open legacy thing here\n\n"
        "origin: migrated from legacy ledger, 2026-06-12\nlocation: src.txt\n"
        "reason: mishandles em-dashes.\nstatus: open\n"
    )
    plan = triage_result(
        ["DW-1", "DW-2"],
        skip=[{"id": "DW-1", "reason": "moot"}, {"id": "DW-2", "reason": "moot"}],
    )
    engine, _ = make_sweep(
        project,
        [
            migrate_effect(project, migrated, [{"key": manifest[0]["key"], "dw_id": "DW-2"}]),
            triage_effect(plan),
        ],
    )
    summary = engine.run()
    assert not summary.paused
    assert engine.state.tasks["sweep-migrate"].phase == Phase.DONE
    assert engine.state.tasks["sweep-triage"].phase == Phase.DONE
    assert ledger_entries(project)["DW-1"].open  # skipped, untouched


def test_migration_dropping_a_gate_restores_the_ledger_then_escalates(project):
    """#519 at the seam that runs: the widened snapshot has to be the one
    `_ensure_migration` actually builds.

    The six `validate_migration` unit tests each construct their own snapshot, so
    every one of them stays green if the call site is left on the old inline
    `{e.id: e.status for e in ...}` form — the exact shape the bug lived in. This
    test is the only one that can see that, which is what earns it its place
    beside the cheaper unit tests.

    Ablation: revert the `snapshot_canonical(text)` call site in
    `_ensure_migration` to a gateless snapshot of the same type
    (`{e.id: PreCanonical(e.status, ()) for e in deferredwork.parse_ledger(text)}`)
    and this test fails ALONE."""
    before = pre_gated_ledger("gate: 3-2")
    write_legacy_ledger(project, before)
    manifest = legacy_manifest(before)
    assert len(manifest) == 1  # the canonical entry is not a legacy item
    # converts the legacy item correctly, but drops DW-1's `gate:` line with it
    bad = migrate_effect(
        project, rewritten_gated_ledger(), [{"key": manifest[0]["key"], "dw_id": "DW-2"}]
    )
    # two scripted sessions: SweepPolicy.max_migration_attempts defaults to 2
    engine, adapter = make_sweep(project, [bad, bad])
    summary = engine.run()

    assert summary.paused
    assert engine.state.tasks["sweep-migrate"].phase == Phase.ESCALATED
    # the un-gating rewrite never sticks: original ledger text restored. Compared
    # as TEXT, not bytes — the fixture reached disk through `write_text`, so a
    # byte assertion would read CRLF on Windows and redden there only.
    assert project.deferred_work.read_text(encoding="utf-8") == before
    assert worktree_clean(project.project)
    # and the refusal reached the session, naming the token it dropped
    prompts = [s.prompt for s in adapter.sessions]
    assert len(prompts) == 2
    # explicit encoding, for the same reason the ledger assertion above pins it:
    # the platform default is not UTF-8 on Windows, so a feedback file carrying a
    # non-ASCII byte would redden there and nowhere else
    feedback = Path(prompts[1].split("--feedback ", 1)[1]).read_text(encoding="utf-8")
    assert "lost gate token" in feedback and "3-2" in feedback


def test_migration_refuses_a_ledger_carrying_duplicate_dw_ids(project):
    """#519, user-approved scope addition: a migration of a ledger where one id
    names two entries is refused BEFORE a rewrite is dispatched, rather than
    graded after one comes back.

    There is no automatic outcome that is right. Migration mode requires
    pre-existing entries to survive byte-identical, so the only rewrite that
    keeps both twins trips `duplicate DW ids` in `validate_migration`, and the
    only rewrite that passes is a collapse that drops one twin's `gate:`
    silently — #519's own failure. Grading the collapse cannot be made safe: a
    snapshot keyed by id describes a merged entry that never existed, so
    hardening the token half alone lets a `done` twin retire the gate, and
    hardening both halves independently pairs one twin's status with the
    other's token. A human renumbers instead, which is the call
    `_apply_deferred_closes` already makes on a duplicate id (#286).

    `adapter.sessions == []` is the load-bearing assertion — it is what
    separates "refused up front" from "refused after burning a session", and
    the pause alone would pass either way.

    The task stays PENDING rather than ESCALATED, like `_refuse_gated_story`'s
    pause: the operator renumbers and resumes, and the migration then runs with
    its full attempt budget intact.

    Ablation, two independent axes:
    - delete the `dupes`/`RunPaused` guard in `_ensure_migration` and this test
      fails; the `validate_migration` gate-token unit tests stay green, since
      they build their snapshots from single-entry fixtures.
    - move the guard back BELOW the `if not task.baseline_commit:` stamp and
      only the `baseline_commit is None` assertion fails — the pause itself is
      unaffected, which is why that assertion has to be here rather than being
      read off the refusal."""
    before = (
        "# Deferred Work\n\n" + _gated_dw1("gate: 3-2") + "\n" + _gated_dw1("gate: 3-3") + "\n"
        "## Deferred from: epic 1 review (2026-04-06)\n\n"
        "- **Open legacy thing here** — `src.txt` mishandles em-dashes\n"
    )
    write_legacy_ledger(project, before)
    # no scripted sessions at all: reaching one would raise rather than pass
    engine, adapter = make_sweep(project, [])
    summary = engine.run()

    assert summary.paused
    # PENDING, not ESCALATED: the refusal is re-askable after a renumber
    assert engine.state.tasks["sweep-migrate"].phase == Phase.PENDING
    # the GATE stage, not the escalation one: every escalation recovery action
    # requires Phase.ESCALATED, which this task deliberately is not, so an
    # escalation stage would offer the operator only actions that must fail
    assert engine.state.paused_stage == PAUSE_STORY_GATE
    # refused BEFORE the rewrite — no migration session was ever dispatched
    assert adapter.sessions == []
    # ...and before the baseline stamp, so the pause strands nothing for a later
    # `_safe_reset` to rewind to. Stamped first, an operator who renumbers and
    # resumes loses that repair the moment the migration session env-faults: the
    # next resume resets the tree to this pre-repair HEAD and lands back here.
    assert engine.state.tasks["sweep-migrate"].baseline_commit is None
    # the ledger is untouched (text, not bytes: the fixture reached disk through
    # `write_text`, so a byte assertion would read CRLF on Windows only)
    assert project.deferred_work.read_text(encoding="utf-8") == before
    assert worktree_clean(project.project)
    # and the operator is told which id to go renumber
    assert "duplicate DW ids" in journal_text(engine)
    assert "DW-1" in journal_text(engine)


def test_migration_duplicate_refusal_clears_a_baseline_it_arrived_holding(project):
    """#519. The refusal's invariant is that it leaves the task owning NO
    baseline, and placing it above the stamp only covers half of that: a task
    that arrives ALREADY holding one keeps it. That is the
    resume-after-escalation path — a first migration attempt stamped a baseline
    and escalated, and the operator's hand-edit between attempts is what puts
    duplicate ids in front of this guard.

    A kept baseline names the PRE-repair tree. The operator renumbers, commits,
    resumes; the migration session then fails dirty, and the next resume's
    `_safe_reset` rewinds to that stale baseline — destroying the repair and any
    commits beside it, and landing back on this same pause.

    Ablation: drop the two `task.baseline_commit`/`baseline_untracked` clears
    from the guard and this test fails ALONE. Its sibling
    `test_migration_refuses_a_ledger_carrying_duplicate_dw_ids` enters with a
    FRESH task, where placement above the stamp already prevents a baseline —
    which is exactly why that case cannot see this one."""
    before = (
        "# Deferred Work\n\n" + _gated_dw1("gate: 3-2") + "\n" + _gated_dw1("gate: 3-3") + "\n"
        "## Deferred from: epic 1 review (2026-04-06)\n\n"
        "- **Open legacy thing here** — `src.txt` mishandles em-dashes\n"
    )
    write_legacy_ledger(project, before)
    engine, adapter = make_sweep(project, [])
    # a first attempt already stamped a baseline and escalated. The sha is never
    # resolved: the tree is clean, so the recovery branch's `_safe_reset` is not
    # reached — this fixture pins the CLEAR, not the reset.
    engine.state.tasks["sweep-migrate"] = StoryTask(
        story_key="sweep-migrate",
        epic=0,
        phase=Phase.ESCALATED,
        baseline_commit="0" * 40,
        baseline_untracked=[],
    )
    summary = engine.run()

    assert summary.paused
    assert adapter.sessions == []
    task = engine.state.tasks["sweep-migrate"]
    # the stale baseline is gone, so the next resume re-stamps the repaired HEAD
    assert task.baseline_commit is None
    assert task.baseline_untracked is None


# ------------------------------------------ review-budget commit-instead-of-rollback


def test_sweep_bundle_budget_exhausted_commits_and_journals(project):
    """A bundle whose review keeps recommending a follow-up but is finalized
    (spec done, owned dw ids closed, verify green) is COMMITTED when the review
    budget is exhausted — not rolled back. The spent budget is journaled; no
    deferred-work entry is filed."""
    write_ledger(project, {"DW-1": "open"})
    plan = triage_result(
        ["DW-1"],
        bundles=[{"name": "fix-it", "dw_ids": ["DW-1"], "intent": "fix it"}],
    )
    # pin the damping cap high so this exercises the max_review_cycles exhaustion
    # path (the damped force-converge has its own sweep test below).
    engine, _ = make_sweep(
        project,
        [triage_effect(plan), bundle_dev_effect(project, "fix-it", ["DW-1"])]
        + [bundle_review_effect(project, "fix-it", clean=False) for _ in range(3)],
        policy=Policy(
            gates=GatesPolicy(mode="none"),
            notify=QUIET,
            scm=ScmPolicy(rollback_on_failure=True),
            limits=LimitsPolicy(max_followup_reviews=99),
        ),
    )
    summary = engine.run()

    assert summary.deferred == 0 and not summary.paused
    assert engine.state.tasks["dw-fix-it"].phase == Phase.DONE
    entries = ledger_entries(project)
    assert entries["DW-1"].status.startswith("done")  # the worked item closed
    refiled = [e for e in entries.values() if e.open and "origin: review-budget-followup" in e.body]
    assert refiled == []  # journal-only: no follow-up entry filed
    kinds = {e["kind"] for e in engine.journal.entries()}
    assert "review-budget-committed" in kinds and "story-deferred" not in kinds


def test_sweep_bundle_budget_followup_not_refiled_twice(project):
    """Re-review cap: when a bundle itself closes a `review-budget-followup` entry
    (legacy and hand-filed rows still exist) and still won't converge, the work is
    committed and the journal flags the repeat — a second non-convergence should
    reach a human, not loop across sweeps."""
    ledger = (
        "# Deferred Work\n\n"
        "### DW-1: follow-up still recommended for dw-prior\n"
        "origin: review-budget-followup\nsource_spec: `spec-dw-fix-it.md`\n"
        "severity: low\nreason: a prior budget exhaustion.\nstatus: open\n"
    )
    project.deferred_work.write_text(ledger, encoding="utf-8")
    git(project.project, "add", "-A")
    git(project.project, "commit", "-q", "-m", "ledger")
    plan = triage_result(
        ["DW-1"],
        bundles=[{"name": "fix-it", "dw_ids": ["DW-1"], "intent": "fix it"}],
    )
    # pin the damping cap high so this exercises the max_review_cycles exhaustion
    # re-review cap (the damped re-review cap has its own sweep test below).
    engine, _ = make_sweep(
        project,
        [triage_effect(plan), bundle_dev_effect(project, "fix-it", ["DW-1"])]
        + [bundle_review_effect(project, "fix-it", clean=False) for _ in range(3)],
        policy=Policy(
            gates=GatesPolicy(mode="none"),
            notify=QUIET,
            scm=ScmPolicy(rollback_on_failure=True),
            limits=LimitsPolicy(max_followup_reviews=99),
        ),
    )
    summary = engine.run()

    assert summary.deferred == 0 and not summary.paused
    assert engine.state.tasks["dw-fix-it"].phase == Phase.DONE
    entries = ledger_entries(project)
    assert entries["DW-1"].status.startswith("done")  # the worked entry still closes
    open_followups = [
        e for e in entries.values() if e.open and "origin: review-budget-followup" in e.body
    ]
    assert open_followups == []  # no second follow-up entry created
    capped = [e for e in engine.journal.entries() if e["kind"] == "review-budget-committed"]
    assert len(capped) == 1 and capped[0]["re_review_capped"] is True


def test_sweep_bundle_followup_damped_commits_and_journals(project):
    """Default damping cap (1): a bundle whose review keeps recommending a follow-up
    converges after ONE honored round instead of burning the whole review budget.
    The spent budget is journaled (no ledger entry), the work is committed, and —
    the steady state — the damped converge stays quiet (no review-budget ATTENTION)."""
    write_ledger(project, {"DW-1": "open"})
    plan = triage_result(
        ["DW-1"],
        bundles=[{"name": "fix-it", "dw_ids": ["DW-1"], "intent": "fix it"}],
    )
    engine, _ = make_sweep(
        project,
        [triage_effect(plan), bundle_dev_effect(project, "fix-it", ["DW-1"])]
        + [bundle_review_effect(project, "fix-it", clean=False) for _ in range(3)],
    )
    summary = engine.run()

    assert summary.deferred == 0 and not summary.paused
    task = engine.state.tasks["dw-fix-it"]
    assert task.phase == Phase.DONE
    assert task.review_cycle == 2  # converged after one honored follow-up (3rd review unused)
    assert task.followup_reviews_spent == 1
    entries = ledger_entries(project)
    assert entries["DW-1"].status.startswith("done")  # the worked item closed
    refiled = [e for e in entries.values() if e.open and "origin: review-budget-followup" in e.body]
    assert refiled == []  # journal-only: no follow-up entry filed
    kinds = {e["kind"] for e in engine.journal.entries()}
    assert "review-followup-damped" in kinds
    assert "review-budget-committed" not in kinds and "story-deferred" not in kinds
    attention = engine.run_dir / "ATTENTION"
    assert not attention.exists() or "review budget reached" not in attention.read_text()


def test_sweep_bundle_damped_re_review_capped_notifies_not_refiles(project):
    """Re-review cap survives damping: when a bundle itself closes a
    `review-budget-followup` entry (legacy and hand-filed rows still exist) and
    still won't converge, the damped force-converge commits — and, unlike an
    ordinary quiet damped converge, it raises an ATTENTION notice so a human sees
    the repeat."""
    ledger = (
        "# Deferred Work\n\n"
        "### DW-1: follow-up still recommended for dw-prior\n"
        "origin: review-budget-followup\nsource_spec: `spec-dw-fix-it.md`\n"
        "severity: low\nreason: a prior budget exhaustion.\nstatus: open\n"
    )
    project.deferred_work.write_text(ledger, encoding="utf-8")
    git(project.project, "add", "-A")
    git(project.project, "commit", "-q", "-m", "ledger")
    plan = triage_result(
        ["DW-1"],
        bundles=[{"name": "fix-it", "dw_ids": ["DW-1"], "intent": "fix it"}],
    )
    engine, _ = make_sweep(
        project,
        [triage_effect(plan), bundle_dev_effect(project, "fix-it", ["DW-1"])]
        + [bundle_review_effect(project, "fix-it", clean=False) for _ in range(3)],
    )
    summary = engine.run()

    assert summary.deferred == 0 and not summary.paused
    assert engine.state.tasks["dw-fix-it"].phase == Phase.DONE
    entries = ledger_entries(project)
    assert entries["DW-1"].status.startswith("done")  # the worked entry still closes
    open_followups = [
        e for e in entries.values() if e.open and "origin: review-budget-followup" in e.body
    ]
    assert open_followups == []  # no second follow-up entry created
    capped = [e for e in engine.journal.entries() if e["kind"] == "review-followup-damped"]
    assert len(capped) == 1 and capped[0]["re_review_capped"] is True
    # the loud re-review path fires even under damping — a human must see it
    assert "review budget reached" in (engine.run_dir / "ATTENTION").read_text(encoding="utf-8")


# ------------------------------ in-flight bundle recovery on resume (#94)


def _lose_triage(run_dir, corruption="missing"):
    """Make the cached triage plan unusable the three ways a real run can: the
    file vanished, it was truncated mid-write, or it holds something that is not
    a triage result."""
    path = run_dir / "triage.json"
    if corruption == "missing":
        path.unlink()
    elif corruption == "invalid-json":
        path.write_text("{{{", encoding="utf-8")
    else:
        path.write_text("{}", encoding="utf-8")


def _run_two_bundle_dev_escalation(project):
    """Drive a two-bundle sweep until the first bundle's dev session escalates.
    Returns the paused engine; `dw-fix` is ESCALATED, `dw-other` never started,
    both dw ids still open."""
    write_ledger(project, {"DW-1": "open", "DW-2": "open"})
    plan = triage_result(
        ["DW-1", "DW-2"],
        bundles=[
            {"name": "fix", "dw_ids": ["DW-1"], "intent": "resolve DW-1"},
            {"name": "other", "dw_ids": ["DW-2"], "intent": "resolve DW-2"},
        ],
    )
    engine, _ = make_sweep(
        project, [triage_effect(plan), bundle_dev_escalates(project, "fix", ["DW-1"])]
    )
    assert engine.run().paused
    assert engine.state.tasks["dw-fix"].phase == Phase.ESCALATED
    assert "dw-other" not in engine.state.tasks
    return engine


def _redrive_script(project):
    return [bundle_dev_effect(project, "fix", ["DW-1"]), bundle_review_effect(project, "fix")]


def test_rearmed_bundle_redrives_when_triage_json_lost(project):
    # The regression: a human-resolved bundle used to re-drive only because the
    # cached triage plan reloaded and re-emitted its name. Recovery now keys on
    # the persisted task, so losing the cache changes nothing.
    engine = _run_to_dev_escalation(project)
    runs.rearm_escalation(
        engine.run_dir, "dw-fix", isolated_redrive=False, resolution_recorded=True
    )
    _lose_triage(engine.run_dir)

    resumed, adapter = resume_sweep(project, engine, _redrive_script(project))
    summary = resumed.run()

    assert not summary.paused
    assert resumed.state.tasks["dw-fix"].phase == Phase.DONE
    # no triage session: the re-drive never consulted a plan
    assert [s.role for s in adapter.sessions] == ["dev", "review"]
    journal = journal_text(resumed)
    assert "sweep-inflight-redrive" in journal
    assert "sweep-nothing-open" in journal  # recovery closed the only open id
    assert ledger_entries(project)["DW-1"].status.startswith("done")


@pytest.mark.parametrize("corruption", ["missing", "invalid-json", "wrong-shape"])
def test_fresh_triage_different_bundle_name_no_double_drive(project, corruption):
    # The fresh triage renames the surviving bundle, so a name-matched recovery
    # would orphan the re-armed one. It must re-drive by identity, and its ids
    # must have left the open set before the fresh triage sees them.
    engine = _run_two_bundle_dev_escalation(project)
    runs.rearm_escalation(
        engine.run_dir, "dw-fix", isolated_redrive=False, resolution_recorded=True
    )
    _lose_triage(engine.run_dir, corruption)

    fresh = triage_result(
        ["DW-2"], bundles=[{"name": "renamed-fix", "dw_ids": ["DW-2"], "intent": "resolve DW-2"}]
    )
    resumed, adapter = resume_sweep(
        project,
        engine,
        [
            *_redrive_script(project),
            triage_effect(fresh),
            bundle_dev_effect(project, "renamed-fix", ["DW-2"]),
            bundle_review_effect(project, "renamed-fix"),
        ],
    )
    summary = resumed.run()

    assert not summary.paused
    assert [s.role for s in adapter.sessions] == ["dev", "review", "triage", "dev", "review"]
    assert resumed.state.tasks["dw-fix"].phase == Phase.DONE
    assert resumed.state.tasks["dw-renamed-fix"].phase == Phase.DONE
    # each id was closed exactly once, by the bundle that owned it
    closed = [e for e in resumed.journal.entries() if e["kind"] == "sweep-bundle-closed"]
    owners = [(i, e["story_key"]) for e in closed for i in e["dw_ids"]]
    assert sorted(owners) == [("DW-1", "dw-fix"), ("DW-2", "dw-renamed-fix")]
    if corruption != "missing":
        # a truncated / wrong-shape cache degrades to a fresh triage, never a crash
        assert "sweep-triage-reload-failed" in journal_text(resumed)


def test_restore_patch_latch_honored_when_triage_json_lost(project, monkeypatch):
    monkeypatch.setattr(verify, "apply_patch", lambda repo, patch: None)
    engine = _run_to_dev_escalation(project)
    patch = project.implementation_artifacts / "attempt-dw-fix.patch"
    patch.parent.mkdir(parents=True, exist_ok=True)
    patch.write_text("dummy\n")
    runs.rearm_escalation(
        engine.run_dir,
        "dw-fix",
        restore_patch=str(patch),
        isolated_redrive=False,
        resolution_recorded=True,
    )
    _lose_triage(engine.run_dir)

    resumed, adapter = resume_sweep(project, engine, _redrive_script(project))
    summary = resumed.run()

    assert not summary.paused
    task = resumed.state.tasks["dw-fix"]
    assert task.phase == Phase.DONE
    # the recovery pass preserved the restore semantics _run_bundle used to own
    assert "Resume review of the in-review spec" in adapter.sessions[0].prompt
    assert "attempt-restored" in journal_text(resumed)
    assert task.restore_patch is None and task.resolved_redrive is False
    assert ledger_entries(project)["DW-1"].status.startswith("done")


def test_escalated_unresolved_still_skipped_when_triage_json_lost(project):
    # An escalation nobody resolved is terminal: recovery must not touch it, and
    # the fresh triage's overlapping bundle is still dropped by the failed-ids filter.
    engine = _run_two_bundle_dev_escalation(project)
    _lose_triage(engine.run_dir)

    fresh = triage_result(
        ["DW-1", "DW-2"],
        bundles=[
            {"name": "retry-fix", "dw_ids": ["DW-1"], "intent": "a"},
            {"name": "other", "dw_ids": ["DW-2"], "intent": "b"},
        ],
    )
    resumed, adapter = resume_sweep(
        project,
        engine,
        [
            triage_effect(fresh),
            bundle_dev_effect(project, "other", ["DW-2"]),
            bundle_review_effect(project, "other"),
        ],
    )
    summary = resumed.run()

    assert not summary.paused
    assert "sweep-inflight-redrive" not in journal_text(resumed)
    assert [s.role for s in adapter.sessions] == ["triage", "dev", "review"]
    assert resumed.state.tasks["dw-fix"].phase == Phase.ESCALATED
    assert "dw-retry-fix" not in resumed.state.tasks
    assert "sweep-bundle-skipped" in journal_text(resumed)
    entries = ledger_entries(project)
    assert entries["DW-1"].open  # escalated bundle untouched
    assert entries["DW-2"].status.startswith("done")


def test_interrupted_bundle_redrives_by_identity_after_triage_loss(project):
    # Not a re-arm: the host just died mid-dev. The restart arm rolls the attempt
    # back against its own baseline (cause="stopped") and re-runs the bundle.
    engine = _run_to_dev_escalation(project)
    state = load_state(engine.run_dir)
    task = state.tasks["dw-fix"]
    assert task.baseline_commit
    task.phase = Phase.DEV_RUNNING
    save_state(engine.run_dir, state)
    _lose_triage(engine.run_dir)

    resumed, adapter = resume_sweep(project, engine, _redrive_script(project))
    summary = resumed.run()

    assert not summary.paused
    assert resumed.state.tasks["dw-fix"].phase == Phase.DONE
    assert [s.role for s in adapter.sessions] == ["dev", "review"]
    redrive = [e for e in resumed.journal.entries() if e["kind"] == "sweep-inflight-redrive"]
    assert len(redrive) == 1
    assert redrive[0]["phase"] == "dev-running" and redrive[0]["rearmed"] is False
    journal = journal_text(resumed)
    assert "rollback-auto" in journal  # a stopped attempt, rolled back to baseline
    assert "attempt-restored" not in journal  # not a resolved re-drive
    assert ledger_entries(project)["DW-1"].status.startswith("done")


def test_resume_committing_bundle_finishes_commit(project):
    """#115, sweep flavor: a bundle whose host died in the commit window
    (COMMITTING persisted, DONE save never landed) is finished on resume —
    the recovery arm mirrors the base engine's resume-commit, not the
    rollback+restart the recovery used to apply to every non-DEV_VERIFY phase."""
    write_ledger(project, {"DW-1": "open"})
    plan = triage_result(
        ["DW-1"], bundles=[{"name": "fix", "dw_ids": ["DW-1"], "intent": "resolve DW-1"}]
    )
    isolated = Policy(
        gates=GatesPolicy(mode="none"),
        notify=QUIET,
        review=ReviewPolicy(enabled=True, trigger="always"),
        dev=DevPolicy(skill="bmad-dev-auto"),
        scm=ScmPolicy(isolation="worktree", rollback_on_failure=True),
    )
    engine, _ = make_sweep(
        project,
        [
            triage_effect(plan),
            wt_bundle_dev(project),
            wt_bundle_review(project),
        ],
        policy=isolated,
    )

    def crashing_emit(stage, *args, **kwargs):
        if stage == "pre_commit":
            raise RuntimeError("host died in the commit window")
        return original_emit(stage, *args, **kwargs)

    original_emit = engine._emit
    engine._emit = crashing_emit
    assert engine.run().crashed
    crashed = load_state(engine.run_dir).tasks["dw-fix"]
    assert crashed.phase == Phase.COMMITTING
    assert not crashed.commit_sha  # stamped only by the DONE save that never ran
    mount = Path(crashed.worktree_path)
    assert mount.is_dir()

    engine.policy = Policy(
        gates=GatesPolicy(mode="none"),
        notify=QUIET,
        review=ReviewPolicy(enabled=True, trigger="always"),
        dev=DevPolicy(skill="bmad-dev-auto"),
        scm=ScmPolicy(isolation="none", rollback_on_failure=True),
    )
    resumed, adapter = resume_sweep(project, engine, [])
    summary = resumed.run()

    assert not summary.paused and not summary.crashed
    task = resumed.state.tasks["dw-fix"]
    assert task.phase == Phase.DONE and task.commit_sha
    assert adapter.sessions == []  # no triage, dev, review, or gate session re-run
    journal = journal_text(resumed)
    assert "resume-commit" in journal
    assert "resume-restart" not in journal
    assert ledger_entries(project)["DW-1"].status.startswith("done")
    assert "change for dw-fix" in (project.project / "src.txt").read_text(encoding="utf-8")
    assert "unit-merged" in journal_kinds(resumed)
    assert not mount.exists()


def test_sweep_isolation_flip_restart_releases_mount_state_without_main_rollback(
    project, monkeypatch
):
    """Rejected/incomplete work restarts in main without carrying mount operands.

    Ablation: omit the mounted restart release and the rollback spy receives the
    unit baseline while the mount claim remains attached.
    """
    in_place = Policy(
        gates=GatesPolicy(mode="none"),
        notify=QUIET,
        scm=ScmPolicy(isolation="none", rollback_on_failure=True),
    )
    engine, _ = make_sweep(project, [], policy=in_place)
    mount = engine.run_dir / "worktrees" / "dw-fix"
    mount.mkdir(parents=True)
    task = StoryTask(
        "dw-fix",
        0,
        phase=Phase.DEV_RUNNING,
        worktree_path=str(mount),
        branch="bmad-loop/sweep-run/dw-fix",
        baseline_commit=verify.rev_parse_head(project.project),
        baseline_untracked=[],
        spec_file="_bmad-output/implementation-artifacts/spec-dw-fix.md",
        dispatched_spec_file="_bmad-output/implementation-artifacts/spec-dw-fix.md",
        dispatched_spec_snapshot=b"bound",
    )
    engine.state.tasks[task.story_key] = task
    rolled: list[str] = []
    monkeypatch.setattr(engine, "_rollback_or_pause", lambda _task, cause: rolled.append(cause))

    assert engine._recover_inflight_bundle(task) is False

    assert rolled == []
    assert task.phase == Phase.PENDING
    assert task.worktree_path == "" and task.branch == ""
    assert task.baseline_commit is None and task.baseline_untracked is None
    assert task.dispatched_spec_file is None and task.dispatched_spec_snapshot is None
    assert task.spec_file == "_bmad-output/implementation-artifacts/spec-dw-fix.md"
    assert mount.is_dir()  # released and journaled, not destroyed
    assert "isolation-flip-orphaned-worktree" in journal_kinds(engine)


def test_sweep_missing_recorded_mount_escalates_without_finalizing_in_main(project, monkeypatch):
    """A mounted COMMITTING receipt has no in-place fallback when its tree is gone.

    Ablation: gate the reopen on live isolation and the finalizer spy runs against
    main rather than the bundle escalating.
    """
    from bmad_loop.engine import RunPaused

    in_place = Policy(
        gates=GatesPolicy(mode="none"),
        notify=QUIET,
        scm=ScmPolicy(isolation="none"),
    )
    engine, _ = make_sweep(project, [], policy=in_place)
    task = StoryTask(
        "dw-fix",
        0,
        phase=Phase.COMMITTING,
        worktree_path=str(engine.run_dir / "worktrees" / "gone"),
        branch="bmad-loop/sweep-run/dw-fix",
    )
    engine.state.tasks[task.story_key] = task
    finalized: list[Path] = []
    monkeypatch.setattr(
        engine, "_finalize_commit_phase", lambda _task: finalized.append(engine.workspace.root)
    )

    with pytest.raises(RunPaused, match="is gone"):
        engine._recover_inflight_bundle(task)

    assert finalized == []
    assert task.phase == Phase.ESCALATED
    assert engine.workspace.root == project.project


def test_regenerated_intent_when_bundle_file_missing(project):
    # The triage session's authored prose is the one unrecoverable piece; the
    # verbatim ledger entries are re-attached and become the contract.
    engine = _run_to_dev_escalation(project)
    runs.rearm_escalation(
        engine.run_dir, "dw-fix", isolated_redrive=False, resolution_recorded=True
    )
    _lose_triage(engine.run_dir)
    intent = Path(engine.state.tasks["dw-fix"].bundle_file)
    intent.unlink()

    resumed, adapter = resume_sweep(project, engine, _redrive_script(project))
    summary = resumed.run()

    assert not summary.paused
    assert resumed.state.tasks["dw-fix"].phase == Phase.DONE
    regen = [e for e in resumed.journal.entries() if e["kind"] == "sweep-intent-regenerated"]
    assert len(regen) == 1 and regen[0]["dw_ids"] == ["DW-1"]
    assert regen[0]["path"] == str(intent)
    text = intent.read_text(encoding="utf-8")
    assert "bundle_name: fix" in text
    assert "### DW-1" in text and "reason: test entry." in text  # verbatim ledger entry
    assert "authoritative" in text
    assert str(intent) in adapter.sessions[0].prompt  # the dev session got the rebuilt file


def test_stranded_bundle_task_warns_loudly(project):
    write_ledger(project, {"DW-1": "open"})
    engine, _ = make_sweep(project, [])
    engine.state.tasks["dw-ghost"] = StoryTask(
        story_key="dw-ghost", epic=0, dw_ids=["DW-1"], phase=Phase.DEV_RUNNING
    )

    engine._warn_stranded_bundles()

    stranded = [e for e in engine.journal.entries() if e["kind"] == "sweep-inflight-stranded"]
    assert len(stranded) == 1 and stranded[0]["story_keys"] == ["dw-ghost"]
    assert "dw-ghost" in (engine.run_dir / "ATTENTION").read_text(encoding="utf-8")

    # a terminal bundle and a non-bundle task are not stranded
    engine.state.tasks["dw-ghost"].phase = Phase.DONE
    engine.state.tasks["sweep-triage"] = StoryTask(
        story_key="sweep-triage", epic=0, phase=Phase.TRIAGE_RUNNING
    )
    engine._warn_stranded_bundles()
    assert len([e for e in engine.journal.entries() if e["kind"] == "sweep-inflight-stranded"]) == 1


# ------------------------------------------------------------- graceful stop
#
# A graceful stop is delivered out-of-band via the stop-request.json control
# file (Phase 1's runs helpers). SweepEngine overrides _loop, so it carries its
# own boundary checks: the first statement of the while body, and before every
# _run_bundle. These tests lodge the control file directly (an existence read is
# all the engine checks) so the request lands at a chosen boundary, then prove
# the in-flight item still runs to completion and the next never starts.


def _lodge_stop_request(run_dir: Path) -> None:
    (run_dir / runs.STOP_REQUEST_FILE).write_text(
        '{"requested_at": "2026-07-20T00:00:00", "mode": "graceful"}', encoding="utf-8"
    )


def _lodge_after(inner, run_dir: Path):
    """Wrap a scripted effect so the graceful-stop request lands as ``inner``
    returns — proving the in-flight item still runs to completion, because the
    boundary check only fires at the next loop head / next bundle."""

    def effect(spec):
        result = inner(spec)
        _lodge_stop_request(run_dir)
        return result

    return effect


def test_graceful_stop_between_bundles_finishes_current_then_stops(project):
    """A request lodged while bundle 1 runs lets it finish through commit; the
    boundary check before bundle 2 raises, so bundle 2 never starts and the run
    ends `stopped` + resumable. A re-run completes bundle 2."""
    write_ledger(project, {"DW-1": "open", "DW-2": "open"})
    run_dir = project.project / ".bmad-loop" / "runs" / "sweep-run"
    plan = triage_result(
        ["DW-1", "DW-2"],
        bundles=[
            {"name": "first-fix", "dw_ids": ["DW-1"], "intent": "a"},
            {"name": "second-fix", "dw_ids": ["DW-2"], "intent": "b"},
        ],
    )
    engine, _ = make_sweep(
        project,
        [
            triage_effect(plan),
            bundle_dev_effect(project, "first-fix", ["DW-1"]),
            _lodge_after(bundle_review_effect(project, "first-fix"), run_dir),
        ],
    )
    summary = engine.run()

    assert not summary.paused
    tasks = engine.state.tasks
    assert tasks["dw-first-fix"].phase == Phase.DONE and tasks["dw-first-fix"].commit_sha
    assert "dw-second-fix" not in tasks  # bundle 2 never dispatched
    saved = load_state(engine.run_dir)
    assert saved.stopped is True and saved.finished is False
    assert not runs.graceful_stop_requested(engine.run_dir)  # control file consumed
    kinds = [e["kind"] for e in engine.journal.entries()]
    assert "run-complete" not in kinds
    stops = [e for e in engine.journal.entries() if e["kind"] == "run-stop"]
    assert stops and stops[-1]["graceful"] is True
    assert stops[-1]["remaining"] == 1  # DW-2 still open, never bundled

    entries = ledger_entries(project)
    assert entries["DW-1"].status.startswith("done") and entries["DW-2"].open

    # resume: bundle 2 runs to completion and the run finishes
    resumed, adapter = resume_sweep(
        project,
        engine,
        [
            bundle_dev_effect(project, "second-fix", ["DW-2"]),
            bundle_review_effect(project, "second-fix"),
        ],
    )
    summary = resumed.run()
    assert not summary.paused
    assert resumed.state.tasks["dw-second-fix"].phase == Phase.DONE
    assert len(adapter.sessions) == 2  # triage reloaded from cache, bundle 1 skipped
    assert ledger_entries(project)["DW-2"].status.startswith("done")
    assert worktree_clean(project.project)


def test_graceful_stop_during_triage_runs_zero_bundles(project):
    """A request landing during the triage session completes triage (artifacts
    written, resolved entries closed) but the very first bundle-loop check raises,
    so zero bundles run."""
    write_ledger(project, {"DW-1": "open", "DW-2": "open"})
    run_dir = project.project / ".bmad-loop" / "runs" / "sweep-run"
    plan = triage_result(
        ["DW-1", "DW-2"],
        already_resolved=[{"id": "DW-1", "evidence": "already guarded at src.txt:1"}],
        bundles=[{"name": "fix-it", "dw_ids": ["DW-2"], "intent": "fix"}],
    )
    engine, _ = make_sweep(project, [_lodge_after(triage_effect(plan), run_dir)])
    summary = engine.run()

    assert not summary.paused
    tasks = engine.state.tasks
    assert tasks["sweep-triage"].phase == Phase.DONE
    assert (engine.run_dir / "triage.json").is_file()  # triage completed
    assert "dw-fix-it" not in tasks  # zero bundles started
    saved = load_state(engine.run_dir)
    assert saved.stopped is True and saved.finished is False
    assert not runs.graceful_stop_requested(engine.run_dir)
    stops = [e for e in engine.journal.entries() if e["kind"] == "run-stop"]
    assert stops and stops[-1]["graceful"] is True
    entries = ledger_entries(project)
    assert entries["DW-1"].status.startswith("done")  # resolved-close still ran
    assert entries["DW-2"].open
    assert stops[-1]["remaining"] == 1  # only DW-2 left open


def test_graceful_stop_between_cycles_skips_next_triage(project):
    """In repeat mode a request lodged during cycle 1 lets that cycle finish, then
    the while-head check fires before cycle 2 re-triages — cycle 2 never runs."""
    write_ledger(project, {"DW-1": "open"})
    run_dir = project.project / ".bmad-loop" / "runs" / "sweep-run"
    plan1 = triage_result(
        ["DW-1"], bundles=[{"name": "first-fix", "dw_ids": ["DW-1"], "intent": "a"}]
    )
    engine, _ = make_sweep(
        project,
        [
            triage_effect(plan1),
            appending_dev(project, bundle_dev_effect(project, "first-fix", ["DW-1"]), "DW-2"),
            _lodge_after(bundle_review_effect(project, "first-fix"), run_dir),
        ],
        policy=repeat_policy(),
    )
    summary = engine.run()

    assert not summary.paused
    tasks = engine.state.tasks
    assert tasks["dw-first-fix"].phase == Phase.DONE
    assert "sweep-triage-2" not in tasks  # cycle 2 never triaged
    saved = load_state(engine.run_dir)
    assert saved.stopped is True and saved.finished is False
    assert not runs.graceful_stop_requested(engine.run_dir)
    kinds = [e["kind"] for e in engine.journal.entries()]
    assert "sweep-cycle" not in kinds  # the cycle-2 header never printed
    stops = [e for e in engine.journal.entries() if e["kind"] == "run-stop"]
    assert stops and stops[-1]["graceful"] is True
    entries = ledger_entries(project)
    assert entries["DW-1"].status.startswith("done") and entries["DW-2"].open
    assert stops[-1]["remaining"] == 1  # DW-2, generated in cycle 1, still open


def test_hard_stop_between_bundles_takes_hard_arm(project, monkeypatch):
    """Sibling of the graceful bundle-boundary test for the HARD mode (#319): the
    same control file carrying `mode: "hard"` is honored at the same boundary, but
    takes the hard arm — unconditional teardown, no clean-finish subset, `run-stop`
    with `via="stop-request"` and no `graceful` flag. Lodged at `post_bundle`, so
    it lands after bundle 1 is fully done and past `_run_session`'s own hard-file
    check; the boundary before bundle 2 is what sees it."""
    killed = []
    monkeypatch.setattr("bmad_loop.engine.kill_session", lambda rid: killed.append(rid))
    write_ledger(project, {"DW-1": "open", "DW-2": "open"})
    run_dir = project.project / ".bmad-loop" / "runs" / "sweep-run"
    plan = triage_result(
        ["DW-1", "DW-2"],
        bundles=[
            {"name": "first-fix", "dw_ids": ["DW-1"], "intent": "a"},
            {"name": "second-fix", "dw_ids": ["DW-2"], "intent": "b"},
        ],
    )
    engine, adapter = make_sweep(
        project,
        [
            triage_effect(plan),
            bundle_dev_effect(project, "first-fix", ["DW-1"]),
            bundle_review_effect(project, "first-fix"),
        ],
    )

    post_run_stages = []
    original_emit = engine._emit

    def spy_emit(stage, *args, **kwargs):
        if stage == "post_run":
            post_run_stages.append(stage)
        if stage == "post_bundle":
            # the operator's `bmad-loop stop` lands between bundle 1 and bundle 2
            (run_dir / runs.STOP_REQUEST_FILE).write_text(
                '{"requested_at": "2026-07-20T00:00:00", "mode": "hard"}', encoding="utf-8"
            )
        return original_emit(stage, *args, **kwargs)

    engine._emit = spy_emit

    summary = engine.run()

    assert not summary.paused
    tasks = engine.state.tasks
    assert tasks["dw-first-fix"].phase == Phase.DONE and tasks["dw-first-fix"].commit_sha
    assert "dw-second-fix" not in tasks  # bundle 2 never dispatched
    assert len(adapter.sessions) == 3  # triage + bundle-1 dev + bundle-1 review
    saved = load_state(engine.run_dir)
    assert saved.stopped is True and saved.finished is False
    assert not runs.graceful_stop_requested(engine.run_dir)  # consumed at the boundary
    assert killed == ["sweep-run"]  # hard arm's unconditional teardown
    assert post_run_stages == []  # hard path skips the clean-finish subset
    kinds = [e["kind"] for e in engine.journal.entries()]
    assert "run-complete" not in kinds
    assert "stop-request-discarded" not in kinds  # honored, not discarded as stale
    stops = [e for e in engine.journal.entries() if e["kind"] == "run-stop"]
    assert stops and stops[-1]["via"] == "stop-request"
    assert "graceful" not in stops[-1]
    entries = ledger_entries(project)
    assert entries["DW-1"].status.startswith("done") and entries["DW-2"].open


def test_bundle_dispatch_does_not_pin_expected_spec(project, tmp_path):
    """A sweep bundle's fresh dispatch points at `intent.md`, never at a spec — the
    session is free to CREATE one, and #161 has it legitimately adopting a
    pre-existing story spec under a different name. So the #261 read-back must stay
    on the scan here even when a prior attempt recorded `task.spec_file`; pinning
    would poll a path this dispatch never promised to rewrite.

    Falls out of the naming rule rather than a sweep-specific carve-out, which is
    exactly why it needs pinning down: nothing in sweep.py mentions expected_spec.

    Ablation: route ``SweepEngine._dev_prompt`` through its superclass and this
    test fails because the inherited known-spec arm names ``task.spec_file`` and
    pins read-back to it instead of dispatching ``intent.md``.
    """
    engine, adapter = make_sweep(project, [SessionResult(status="crashed")])
    intent = tmp_path / "bundles" / "fix" / "intent.md"
    intent.parent.mkdir(parents=True)
    intent.write_text("# bundle intent\n")
    task = StoryTask(
        story_key="dw-fix",
        epic=0,
        bundle_file=str(intent),
        spec_file=str(project.implementation_artifacts / "spec-adopted-elsewhere.md"),
    )

    prompt = engine._dev_prompt(task, None)
    assert str(intent) in prompt and task.spec_file not in prompt
    engine._run_session(task, role="dev", prompt=prompt, seq=1)
    assert adapter.sessions[-1].expected_spec is None


def test_bundle_dispatch_does_not_bind_accepted_spec_as_attempt_ownership(project):
    """A bundle carries accepted result history, but dispatch still owns intent.md.

    Ablation: delete the SweepEngine override and the inherited sprint seam binds
    the existing ``task.spec_file``, failing this absence assertion alone.
    """
    engine, _ = make_sweep(project, [])
    accepted = project.implementation_artifacts / "spec-adopted-elsewhere.md"
    accepted.parent.mkdir(parents=True, exist_ok=True)
    write_spec(accepted, "done", "abc")
    task = StoryTask(
        story_key="dw-fix",
        epic=0,
        spec_file=str(accepted),
        dispatched_spec_file="stale/from/earlier-attempt.md",
    )

    task.dispatched_spec_file = engine._dispatched_spec_for_attempt(task)

    assert task.dispatched_spec_file is None


def test_bundle_explicit_spec_route_pins_readback_without_claiming_ownership(project):
    """Sweep's explicit restore route names output without owning it as input.

    Ablation: delete the Sweep snapshot-requirement override and the inherited
    pre-session guard refuses this launch because no dispatched spec is bound.
    """
    engine, adapter = make_sweep(project, [SessionResult(status="crashed")])
    accepted = project.implementation_artifacts / "spec-dw-fix.md"
    accepted.parent.mkdir(parents=True, exist_ok=True)
    write_spec(accepted, "in-review", "abc")
    task = StoryTask(
        story_key="dw-fix",
        epic=0,
        phase=Phase.DEV_RUNNING,
        spec_file=str(accepted),
        restore_patch="/run/artifacts/attempt-dw-fix.patch",
    )
    prompt = engine._generic_bundle_prompt(task, None)

    engine._run_session(task, role="dev", prompt=prompt, seq=1)

    assert adapter.sessions[-1].expected_spec == str(accepted)
    assert task.dispatched_spec_file is None
    assert task.dispatched_spec_snapshot is None


# ---------------- frontmatter `deferred:` harvest parity (BMAD-METHOD #2640)

SWEEP_FINDING = {
    "summary": "The retry cap should be configurable",
    "evidence": "hardcoded while fixing DW-1",
    "location": "src/retry.py:88",
    "severity": "low",
}


def _harvest_bundle_policy(*, attempts: int = 3) -> Policy:
    return Policy(
        gates=GatesPolicy(mode="none"),
        notify=QUIET,
        review=ReviewPolicy(enabled=False),
        dev=DevPolicy(skill="bmad-dev-auto"),
        limits=LimitsPolicy(max_dev_attempts=attempts),
        scm=ScmPolicy(rollback_on_failure=True),
    )


def test_generic_bundle_harvests_new_spec_deferrals_alongside_its_ids(project):
    write_ledger(project, {"DW-1": "open", "DW-2": "open"})
    plan = triage_result(
        ["DW-1", "DW-2"],
        bundles=[{"name": "fix-things", "dw_ids": ["DW-1", "DW-2"], "intent": "fix both"}],
    )
    engine, _ = make_sweep(
        project,
        [
            triage_effect(plan),
            bundle_dev_effect(
                project,
                "fix-things",
                ["DW-1", "DW-2"],
                mark_ledger=False,
                followup_review=False,
                deferred=[SWEEP_FINDING],
            ),
        ],
        policy=_harvest_bundle_policy(),
    )

    summary = engine.run()

    assert not summary.paused
    assert engine.state.tasks["dw-fix-things"].phase == Phase.DONE
    entries = ledger_entries(project)
    assert entries["DW-1"].status.startswith("done")
    assert entries["DW-2"].status.startswith("done")
    assert entries["DW-3"].open
    assert entries["DW-3"].title == SWEEP_FINDING["summary"]
    assert "origin: spec-deferred " in entries["DW-3"].body
    assert "location: src/retry.py:88" in entries["DW-3"].body
    assert "source_spec: `spec-dw-fix-things.md`" in entries["DW-3"].body
    assert engine.state.tasks["dw-fix-things"].dw_ids == ["DW-1", "DW-2"]
    harvested = [e for e in engine.journal.entries() if e["kind"] == "spec-deferrals-harvested"]
    assert len(harvested) == 1 and harvested[0]["dw_ids"] == ["DW-3"]


def test_bundle_harvest_alone_is_not_proof_of_work(project):
    """A bundle session that files a spec deferral but changes nothing still reads
    as "no changes": the harvest's own ledger write is not the session's proof of
    work. The harvest itself still lands, so what this pins is the gate's action,
    not a lost harvest."""
    write_ledger(project, {"DW-1": "open"})
    plan = triage_result(
        ["DW-1"],
        bundles=[{"name": "fix-things", "dw_ids": ["DW-1"], "intent": "fix it"}],
    )
    engine, _ = make_sweep(
        project,
        [
            triage_effect(plan),
            bundle_dev_effect(
                project,
                "fix-things",
                ["DW-1"],
                mark_ledger=False,
                followup_review=False,
                deferred=[SWEEP_FINDING],
                write_src=False,
            ),
        ],
        policy=_harvest_bundle_policy(attempts=1),
    )

    summary = engine.run()

    decisions = [e for e in engine.journal.entries() if e["kind"] == "dev-decision"]
    assert [decision["action"] for decision in decisions] == ["defer"]
    assert "no changes" in decisions[0]["reason"]
    assert summary.deferred == 1
    assert engine.state.tasks["dw-fix-things"].phase == Phase.DEFERRED
    harvested = [e for e in engine.journal.entries() if e["kind"] == "spec-deferrals-harvested"]
    assert len(harvested) == 1 and harvested[0]["dw_ids"] == ["DW-2"]


# ------------------------- the per-run decisions writes, confined to the project (#593)


def _redirect_the_run_dir(project, tmp_path):
    """Plant a link at the run directory `make_sweep` will use, pointing OUT of the
    project, and hand back where it points.

    The link goes at the run dir rather than at `.bmad-loop/` deliberately: the
    engine writes its journal, state and triage records into this same directory
    through the ordinary (unconfined) writers, and those must keep working so the
    run reaches the decisions phase at all. Only the two confined writes walk the
    components from the project root, so only they refuse — which makes the
    refusal attributable to the site under test rather than to a broken run."""
    run_dir = project.project / ".bmad-loop" / "runs" / "sweep-run"
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    run_dir.symlink_to(outside, target_is_directory=True)
    return outside


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX symlinks")
def test_the_seeded_decisions_write_refuses_a_redirected_run_dir(project, tmp_path):
    """The escape #593 names, on the pre-answer-seeded write. `follow_symlinks=False`
    refused a link planted at `decisions.json`; it never refused one planted at any
    directory ABOVE it, and this file sits three deep under the project root
    (`.bmad-loop/runs/<id>/`), so a link at any of those three redirected both the
    temp and the published file out of the project.

    The `decision-preanswered` journal line is the PRECONDITION, exactly as in the
    disk-full twin above: it is appended before the write, so without it the
    "nothing escaped" assertion would pass for the entirely different reason that
    the phase was never entered.

    Ablation: revert this site to `atomic_write_text(..., follow_symlinks=False)`
    and this fails — the run stops crashing and `decisions.json` appears in
    `outside/`."""
    from bmad_loop import decisions
    from bmad_loop.sweep import DecisionOption

    outside = _redirect_the_run_dir(project, tmp_path)
    write_ledger(project, {"DW-1": "open"})
    decisions.record_pre_answer(
        project.project,
        "DW-1",
        DecisionOption(key="2", label="Keep", effect="keep-open"),
        date="2026-06-12",
    )
    plan = triage_result(  # keep-open only: no bundle, so no bundle-intent write
        ["DW-1"],
        decisions=[
            _decision(
                "DW-1",
                [
                    {"key": "1", "label": "Build", "effect": "build", "intent": "x"},
                    {"key": "2", "label": "Keep", "effect": "keep-open"},
                ],
                recommendation="2",
            )
        ],
    )
    engine, _ = make_sweep(project, [triage_effect(plan)], prompting=False)

    summary = engine.run()

    assert summary.crashed
    assert platform_util.UnconfinedWriteError.__name__ in str(summary.crash_error)
    assert '"decision-preanswered"' in journal_text(engine)  # PRECONDITION: site reached
    assert not (outside / "decisions.json").exists()  # nothing escaped the project


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX symlinks")
def test_the_interactive_decisions_write_refuses_a_redirected_run_dir(project, tmp_path):
    """The interactive write's own refusal — the same file, a few lines further
    down, and a separate row for the reason its disk-full twin is separate: the two
    writes are reached by different paths, and one adopted without the other would
    leave half the file's writers unconfined.

    Two preconditions, as in that twin. `decision-pending` is appended before
    `prompter.ask`, proving the phase was entered; `decision-answered` is appended
    AFTER the write, so its ABSENCE places the refusal at the site under test
    rather than upstream of it.

    Ablation: revert this site and this fails — the run completes and
    `decisions.json` appears in `outside/`."""
    outside = _redirect_the_run_dir(project, tmp_path)
    write_ledger(project, {"DW-1": "open"})
    plan = triage_result(  # close-only: no bundle, so no bundle-intent write
        ["DW-1"],
        decisions=[
            _decision(
                "DW-1",
                [
                    {"key": "1", "label": "Close it", "effect": "close", "resolution": "moot"},
                    {"key": "2", "label": "Keep open", "effect": "keep-open"},
                ],
            )
        ],
    )
    engine, _ = make_sweep(project, [triage_effect(plan)], answers=["1"], prompting=True)

    summary = engine.run()

    journal = journal_text(engine)
    assert summary.crashed
    assert platform_util.UnconfinedWriteError.__name__ in str(summary.crash_error)
    assert '"decision-pending"' in journal  # PRECONDITION: the prompt was reached
    assert '"decision-answered"' not in journal  # the refusal landed AT the write
    assert not (outside / "decisions.json").exists()  # nothing escaped the project


def test_the_decisions_writes_land_on_a_clean_tree(project):
    """The positive control for both refusals above. Without it they pass for a
    `_decisions_phase` wired to crash on anything, which is every reason
    `decisions.json` could be absent from `outside/` — and the run dir under test
    there is a symlink, so this row also pins that an ORDINARY run dir still gets
    its file."""
    write_ledger(project, {"DW-1": "open"})
    plan = triage_result(
        ["DW-1"],
        decisions=[
            _decision(
                "DW-1",
                [
                    {"key": "1", "label": "Close it", "effect": "close", "resolution": "moot"},
                    {"key": "2", "label": "Keep open", "effect": "keep-open"},
                ],
            )
        ],
    )
    engine, _ = make_sweep(project, [triage_effect(plan)], answers=["1"], prompting=True)

    summary = engine.run()

    assert not summary.crashed
    landed = json.loads((engine.run_dir / "decisions.json").read_text(encoding="utf-8"))
    assert landed["DW-1"]["effect"] == "close"


# ------------------------------------------ batched locked ledger adoption (#286/#469)


@contextlib.contextmanager
def _counting_ledger_lock(monkeypatch, acquisitions):
    """A `ledger_lock` spy that records every acquisition and still really locks.

    Still locking matters: a spy that only counted would let a nesting bug
    through, and `ledger_lock` raising on nested entry is the guard that keeps
    these callers honest.
    """
    real_lock = deferredwork.ledger_lock

    @contextlib.contextmanager
    def spy_lock(p):
        acquisitions.append(p)
        with real_lock(p):
            yield

    monkeypatch.setattr(deferredwork, "ledger_lock", spy_lock)
    yield


def test_close_resolved_batches_with_per_entry_evidence(project, monkeypatch):
    """The whole already-resolved set closes in ONE locked read->edit->write, and
    each entry keeps its own evidence (#286/#469).

    Two claims, and both are needed. The per-entry evidence says the collapse to
    `mark_done_many` did not flatten three distinct notes into one shared string
    — `notes=` pairs positionally with the ids, and a batch that dropped it would
    still close all three entries and still journal all three ids. The
    acquisition count is what says this is one transaction: as a per-entry
    `mark_done` loop it took the cross-process ledger lock once per id, leaving a
    rival writer — a live run's harvest, the TUI decision modal, `sweep
    --archive` — a window between every pair of closures, so the sweep could
    journal three closures with only some of them on disk.

    Ablation: restore the per-entry `mark_done` loop. The evidence asserts still
    pass, which is exactly why the count is asserted too; `acquisitions` goes to
    3 and this reddens.
    """
    write_ledger(project, {"DW-1": "open", "DW-2": "open", "DW-3": "open"})
    engine, _adapter = make_sweep(project, [])
    plan = TriagePlan(
        open_ids=frozenset({"DW-1", "DW-2", "DW-3"}),
        already_resolved=(
            ResolvedEntry("DW-1", "fixed by a1b2c3d"),
            ResolvedEntry("DW-2", "superseded by DW-9"),
            ResolvedEntry("DW-3", "never reproduced"),
        ),
    )

    acquisitions = []
    with _counting_ledger_lock(monkeypatch, acquisitions):
        assert engine._close_resolved(plan) == 3

    assert acquisitions == [project.deferred_work]  # ONE, and on this ledger
    text = project.deferred_work.read_text(encoding="utf-8")
    assert "already resolved: fixed by a1b2c3d" in text
    assert "already resolved: superseded by DW-9" in text
    assert "already resolved: never reproduced" in text
    entries = ledger_entries(project)
    assert all(entries[i].status.startswith("done ") for i in ("DW-1", "DW-2", "DW-3"))
    closed = [e for e in engine.journal.entries() if e["kind"] == "sweep-resolved-closed"]
    assert len(closed) == 1 and closed[0]["dw_ids"] == ["DW-1", "DW-2", "DW-3"]


def test_reopen_after_defer_uses_one_lock(project, monkeypatch):
    """A deferred bundle's closes are all undone in ONE locked read->edit->write
    (#286/#469), and the journal reports the same ids in the same order.

    A rollback that leaves some of this run's closes undone and others standing
    is the one outcome this method exists to prevent, and the per-id `mark_open`
    comprehension it replaces took the lock once per id — three separate windows
    for a rival writer, in a step that has to be atomic to mean anything. The
    journal assert is what says the collapse preserved the contract the caller
    reads: `mark_open_many` returns the ids actually reopened, in the order
    given, exactly as the comprehension did.

    Ablation: restore the per-id comprehension — `acquisitions` goes to 3.
    """
    write_ledger(project, {"DW-1": "open", "DW-2": "open", "DW-3": "open"})
    engine, _adapter = make_sweep(project, [])
    task = StoryTask(story_key="dw-fix", epic=0, dw_ids=["DW-1", "DW-2", "DW-3"])
    # Seed the reopenable closes BEFORE the spy: this write is the setup, not
    # the behavior under test, and counting it would hide the real number.
    deferredwork.mark_done_many_reopenable(
        project.deferred_work,
        task.dw_ids,
        engine._today(),
        engine._bundle_close_note(task),
        engine._bundle_close_operation_id(task),
    )

    acquisitions = []
    with _counting_ledger_lock(monkeypatch, acquisitions):
        engine._reopen_ledger_after_defer(task)

    assert acquisitions == [project.deferred_work]  # ONE, and on this ledger
    entries = ledger_entries(project)
    assert all(entries[i].open for i in ("DW-1", "DW-2", "DW-3"))
    reopened = [e for e in engine.journal.entries() if e["kind"] == "sweep-bundle-reopened"]
    assert len(reopened) == 1
    assert reopened[0]["dw_ids"] == ["DW-1", "DW-2", "DW-3"]
    assert reopened[0]["story_key"] == "dw-fix"


def test_migration_restore_writes_back_an_external_ledger(project, tmp_path):
    """#735 follow-up. An `implementation_artifacts` dir configured OUTSIDE the
    repo tree is a supported shape — `ProjectPaths.rebased` deliberately leaves
    such dirs put, because they are shared rather than per-checkout — so the
    ledger can resolve outside `workspace.root`.

    `git reset --hard` provably cannot have touched a path no revision of this
    repo can even name, which is the same proof the untracked-inside-the-tree
    case relies on. So the anchor is the rejected rewrite this attempt graded,
    the restore puts the pre-migration text back, and no divergence is journaled.
    Escalating here would strand the half-migrated rewrite on disk for a
    configuration that is not ambiguous at all.
    """
    external = ProjectPaths(
        project=project.project,
        implementation_artifacts=tmp_path / "shared-artifacts" / "implementation-artifacts",
        planning_artifacts=project.planning_artifacts,
    )
    external.implementation_artifacts.mkdir(parents=True)
    write_legacy_ledger(external, LEGACY_LEDGER, commit=False)
    engine, _ = make_sweep(external, [_half_migrated_effect(external)] * 2)
    summary = engine.run()

    assert summary.paused
    assert external.deferred_work.read_text(encoding="utf-8") == LEGACY_LEDGER
    assert "sweep-migration-restore-diverged" not in journal_kinds(engine)


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX symlinks")
def test_migration_restore_writes_back_a_symlinked_ledger(project, tmp_path):
    """#735 follow-up. A ledger symlinked into the repo is a supported shape —
    `atomic_write_text` follows symlinks BY DEFAULT precisely so such a ledger
    "keeps being a symlink and the real file is what gets rewritten".

    Git stores that tracked symlink as a blob holding the TARGET PATHNAME, so a
    baseline anchor taken from the blob would compare a pathname against ledger
    text and never be true — escalating every failed migration over a shape that
    is not ambiguous at all. `reset --hard` restores the link, never what it
    points at, so the reset republished no ledger text and the anchor is the
    rejected rewrite this attempt graded, as for an untracked or external ledger.

    Ablation: drop the `path_is_non_regular_at_revision` arm from
    `_ledger_baseline_text` and this reddens on the restored-text assertion, with
    `sweep-migration-restore-diverged` journaled and the half-migrated rewrite
    left standing.
    """
    target = tmp_path / "shared" / "deferred-work.md"
    target.parent.mkdir(parents=True)
    target.write_text(LEGACY_LEDGER, encoding="utf-8")
    if project.deferred_work.is_symlink() or project.deferred_work.exists():
        project.deferred_work.unlink()
    project.deferred_work.symlink_to(target)
    git(project.project, "add", "-A")
    git(project.project, "commit", "-q", "-m", "symlinked ledger")
    engine, _ = make_sweep(project, [_half_migrated_effect(project)] * 2)
    summary = engine.run()

    assert summary.paused
    # the link survived the round trip, and the real file holds the restored text
    assert project.deferred_work.is_symlink()
    assert target.read_text(encoding="utf-8") == LEGACY_LEDGER
    assert "sweep-migration-restore-diverged" not in journal_kinds(engine)


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX symlinks")
def test_migration_restore_accepts_an_already_restored_symlink_ledger(project, tmp_path):
    """#735/#736. A restore that finds the ledger already holding the text it
    would write is DONE, not divergent.

    Reachable with no rival at all. A migration session that atomic-saves —
    write-temp-then-rename, which is how most editors and many CLIs write —
    replaces the tracked symlink with a regular file. `reset --hard` puts the
    link back, and the external target it can never reach was therefore never
    rewritten, so the ledger is already correct. But `rewrite`, read off that
    regular file, is the rejected migration text, so demanding the anchor here
    reports a divergence that did not happen, escalates, and spends the attempt
    budget: the second attempt never dispatches.

    This is the #736 principle at the restore: an operation with nothing to write
    must not fail.

    Ablation: drop the `current == text` arm and this reddens on all three —
    `sweep-migration-restore-diverged` is journaled, the paused reason becomes
    the "changed underneath" accusation, and only one session runs.
    """
    target = tmp_path / "shared" / "deferred-work.md"
    target.parent.mkdir(parents=True)
    target.write_text(LEGACY_LEDGER, encoding="utf-8")
    if project.deferred_work.is_symlink() or project.deferred_work.exists():
        project.deferred_work.unlink()
    project.deferred_work.symlink_to(target)
    git(project.project, "add", "-A")
    git(project.project, "commit", "-q", "-m", "symlinked ledger")

    manifest = legacy_manifest()
    half = (
        "# Deferred Work\n\n"
        "### DW-1: Old fixed thing\n\norigin: migrated, 2026-06-12\nlocation: n/a\n"
        "reason: repaired.\nstatus: done 2026-04-06\n\n"
        "## Deferred from: epic 1 review (2026-04-06)\n\n"
        "- **Open legacy thing here** — `src.txt` mishandles em-dashes\n"
    )

    def atomic_save_effect(spec):
        # the rename an atomic save performs: the symlink is REPLACED, so the
        # external target keeps the pre-migration text throughout.
        project.deferred_work.unlink()
        project.deferred_work.write_text(half, encoding="utf-8")
        return SessionResult(
            status="completed",
            result_json={
                "workflow": "deferred-sweep-migrate",
                "mapping": [{"key": manifest[0]["key"], "dw_id": "DW-1"}],
                "escalations": [],
            },
        )

    engine, adapter = make_sweep(project, [atomic_save_effect] * 2)
    summary = engine.run()

    assert summary.paused  # on the attempt cap, having actually retried
    assert "sweep-migration-restore-diverged" not in journal_kinds(engine)
    assert "changed underneath the failed migration attempt" not in engine.state.paused_reason
    assert target.read_text(encoding="utf-8") == LEGACY_LEDGER
    assert len(adapter.sessions) == 2


def test_bundle_restart_arm_anchors_spec_ownership_before_it_discards_the_mount(
    project, monkeypatch
):
    """Sweep's restart arm is the engine's, and it needed the same re-anchor.

    `SweepEngine` replaces `_loop` wholesale and `Engine._loop` is the ONLY caller of
    `_finish_inflight`, so the re-anchor that method makes never runs here — while
    `_recover_inflight_bundle` reaches the very same shared
    `Engine._discard_unit_for_restart`. The baseline half of that helper was therefore
    inherited by sweep and the spec-ownership half was not.

    Both spec paths are persisted RELATIVE to the mount
    (`model._serialized_worktree_path`), and the restart arm discards the worktree and
    clears `task.worktree_path` before the caller saves — so without the re-anchor the
    save strands a worktree-relative spelling beside an EMPTY `worktree_path`, and the
    next resume resolves it against the main checkout, which carries the same layout.
    `recovery_flow._attempt_owned_spec` then finds exactly one candidate,
    `spec_within_roots` accepts it, and the snapshot restore rewrites the operator's own
    copy.

    Graded at the discard, like its engine sibling: the ordering is the property, and a
    later rebind would let a post-hoc assertion pass with the re-anchor deleted.

    Ablation: drop `task.rebase_spec_paths_on(...)` from `_recover_inflight_bundle` and
    both assertions fail with the bare relative spellings.
    """
    from bmad_loop.workspace import open_unit_workspace

    engine, _ = make_sweep(project, [], policy=isolated_policy())
    unit = open_unit_workspace(
        project.project, project, "sweep-run", "dw-fix", "main", "bundle", engine.run_dir
    )
    task = StoryTask("dw-fix", 1, phase=Phase.DEV_RUNNING)
    task.worktree_path = str(unit.path)
    task.branch = unit.branch
    task.spec_file = "_bmad-output/accepted.md"
    task.dispatched_spec_file = "_bmad-output/dispatched.md"
    engine.state.tasks["dw-fix"] = task

    seen: dict[str, str | None] = {}

    class _StopAtDiscard(Exception):
        pass

    def _spy(*_args, **_kwargs):
        seen["spec_file"] = task.spec_file
        seen["dispatched_spec_file"] = task.dispatched_spec_file
        raise _StopAtDiscard

    monkeypatch.setattr("bmad_loop.engine.discard_worktree", _spy)

    with pytest.raises(_StopAtDiscard):
        engine._recover_inflight_bundle(task)

    assert seen["spec_file"] == str(unit.path / "_bmad-output/accepted.md")
    assert seen["dispatched_spec_file"] == str(unit.path / "_bmad-output/dispatched.md")


# ------------- terminal-key / dw_ids agreement on a resume (DW-125)
#
# `_run_bundle`'s terminal-task early return used to compare the KEY alone, and
# the key is a pure function of `(name, cycle)`. Lose `<run>/triage.json` on a
# resume and `_ensure_triage` regenerates a plan whose bundle names are
# re-authored freely — so a fresh bundle that happened to reuse a finished
# bundle's name was silently swallowed with its ids never run.
# `_materialize_bundles`' uniqueness pass cannot see this: it compares names
# against THIS cycle's list, never against persisted state. The guard now tests
# the persisted task's `dw_ids` and suffixes the NAME on divergence.


def _bundle_task(engine, key: str, dw_ids, phase=Phase.DONE) -> StoryTask:
    task = StoryTask(story_key=key, epic=0, dw_ids=list(dw_ids), phase=phase)
    engine.state.tasks[key] = task
    return task


def _stub_run_story(engine, monkeypatch) -> list[StoryTask]:
    """Record dispatches without spending a session — these rows are about
    naming, not about what the dev leg then does."""
    dispatched: list[StoryTask] = []
    monkeypatch.setattr(engine, "_run_story", dispatched.append)
    return dispatched


@pytest.mark.parametrize(
    "persisted",
    [["DW-1", "DW-2"], ["DW-2", "DW-1"], []],
    ids=["same-ids", "reordered-ids", "legacy-empty"],
)
def test_run_bundle_skips_terminal_task_whose_dw_ids_agree(project, monkeypatch, persisted):
    """Agreement is SET equality, so a regenerated plan that reorders the same
    ids is still the same bundle and is still skipped — treating it as new would
    re-run finished work on every cache-loss resume. An EMPTY persisted list is
    the pre-`dw_ids` `state.json` shape (`model.py` loads a missing key as `[]`)
    and agrees with anything, so no paused legacy run needs migrating."""
    write_ledger(project, {"DW-1": "done 2026-06-11", "DW-2": "done 2026-06-11"})
    engine, adapter = make_sweep(project, [])
    _bundle_task(engine, "dw-fix", persisted)
    dispatched = _stub_run_story(engine, monkeypatch)

    key = engine._run_bundle(Bundle(name="fix", dw_ids=("DW-1", "DW-2"), intent="i"), 1)

    assert key == "dw-fix"
    assert list(engine.state.tasks) == ["dw-fix"]  # no second task minted
    assert not dispatched and not adapter.sessions
    kinds = {e["kind"] for e in engine.journal.entries()}
    assert "bundle-start" not in kinds and "sweep-bundle-key-deduped" not in kinds


def test_run_bundle_dedupes_when_a_terminal_task_holds_the_key_for_other_ids(project, monkeypatch):
    """Ablation: restore the key-only early return and this fails at `key` —
    the bundle is skipped, `dw-fix-2` never exists and DW-2 never runs."""
    write_ledger(project, {"DW-1": "done 2026-06-11", "DW-2": "open"})
    engine, _ = make_sweep(project, [])
    _bundle_task(engine, "dw-fix", ["DW-1"])
    dispatched = _stub_run_story(engine, monkeypatch)

    key = engine._run_bundle(Bundle(name="fix", dw_ids=("DW-2",), intent="second"), 1)

    assert key == "dw-fix-2"
    assert dispatched == [engine.state.tasks["dw-fix-2"]]
    assert engine.state.tasks["dw-fix-2"].dw_ids == ["DW-2"]
    assert engine.state.tasks["dw-fix"].dw_ids == ["DW-1"]  # the finished one untouched
    # the intent DOCUMENT has to agree with the directory and with the name
    # `_ensure_bundle_intent` recovers back out of the story key
    intent = engine.run_dir / "bundles" / "fix-2" / "intent.md"
    text = intent.read_text(encoding="utf-8")
    assert "# Deferred-work bundle: fix-2" in text and "bundle_name: fix-2" in text
    assert engine.state.tasks["dw-fix-2"].bundle_file == str(intent)
    (rec,) = _records(engine, "sweep-bundle-key-deduped")
    assert rec["original"] == "fix" and rec["name"] == "fix-2"
    assert rec["attempt"] == 2 and rec["dw_ids"] == ["DW-2"]
    (start,) = _records(engine, "bundle-start")
    assert start["story_key"] == "dw-fix-2"
    # a freshly minted task is born with the bundle's ids: nothing to adopt (DW-144)
    assert not _records(engine, "sweep-bundle-dwids-adopted")


def test_run_bundle_skips_a_deduped_key_without_re_announcing_the_rename(project, monkeypatch):
    """The ordinary SECOND resume: `dw-fix` is terminal with divergent ids and
    the deduped `dw-fix-2` is terminal with agreeing ids, so the bundle is
    skipped as already-finished. The rename record belongs to the run that
    actually took the name — journaling it in the resolver re-announced it once
    per resume for a bundle nothing renamed.

    Ablation: move the `sweep-bundle-key-deduped` append back above
    `_run_bundle`'s terminal early return and the no-record assertion fails."""
    write_ledger(project, {"DW-1": "done 2026-06-11", "DW-2": "done 2026-06-11"})
    engine, adapter = make_sweep(project, [])
    _bundle_task(engine, "dw-fix", ["DW-1"])
    _bundle_task(engine, "dw-fix-2", ["DW-2"])
    dispatched = _stub_run_story(engine, monkeypatch)

    key = engine._run_bundle(Bundle(name="fix", dw_ids=("DW-2",), intent="second"), 1)

    assert key == "dw-fix-2"
    assert not dispatched and not adapter.sessions
    # `entries()` rather than `_records`: a skip this total journals NOTHING, so
    # journal.jsonl does not exist and the raw-file helper would fault on the read
    kinds = {e["kind"] for e in engine.journal.entries()}
    assert "sweep-bundle-key-deduped" not in kinds and "bundle-start" not in kinds


def test_run_bundle_dedupe_keeps_the_cycle_n_key_and_dirname_shape(project, monkeypatch):
    write_ledger(project, {"DW-1": "done 2026-06-11", "DW-2": "open"})
    engine, _ = make_sweep(project, [])
    _bundle_task(engine, "dw3-fix", ["DW-1"])
    _stub_run_story(engine, monkeypatch)

    key = engine._run_bundle(Bundle(name="fix", dw_ids=("DW-2",), intent="second"), 3)

    assert key == "dw3-fix-2"
    intent = engine.run_dir / "bundles" / "c3-fix-2" / "intent.md"
    assert "bundle_name: fix-2" in intent.read_text(encoding="utf-8")
    # deduping the NAME (not the key alone) is what keeps this round-trip correct
    match = BUNDLE_KEY_RE.match(key)
    assert match is not None and match.group(1) == "3" and match.group(2) == "fix-2"


def test_run_bundle_leaves_a_non_terminal_task_to_the_inflight_recovery(project, monkeypatch):
    """The guard is deliberately scoped to TERMINAL tasks: an in-flight task at
    the key still goes down `_recover_inflight_bundle` exactly as before, ids
    agreeing or not."""
    write_ledger(project, {"DW-1": "open", "DW-2": "open"})
    engine, _ = make_sweep(project, [])
    _bundle_task(engine, "dw-fix", ["DW-1"], phase=Phase.DEV_RUNNING)
    dispatched = _stub_run_story(engine, monkeypatch)

    key = engine._run_bundle(Bundle(name="fix", dw_ids=("DW-2",), intent="second"), 1)

    assert key == "dw-fix" and "dw-fix-2" not in engine.state.tasks
    assert dispatched == [engine.state.tasks["dw-fix"]]
    assert engine.state.tasks["dw-fix"].phase == Phase.PENDING  # reset by the recovery
    assert not _records(engine, "sweep-bundle-key-deduped")
    assert (engine.run_dir / "bundles" / "fix" / "intent.md").is_file()


def test_run_bundle_adopts_the_bundle_dw_ids_onto_a_reset_inflight_task(project, monkeypatch):
    """DW-144: adopt before publishing the new intent, then close only its ids.

    Ablation: delete `task.dw_ids = list(bundle.dw_ids)` from `_run_bundle`.
    The write-boundary assertion fails with ["DW-1"] instead of ["DW-2"],
    before dispatch can reject mismatched dev results and exhaust the script.
    Deleting only the adoption journal append fails the record assertions.
    """
    write_ledger(project, {"DW-1": "open", "DW-2": "open"})
    engine, _ = make_sweep(
        project,
        [bundle_dev_effect(project, "fix", ["DW-2"], mark_ledger=False, followup_review=False)],
    )
    task = _bundle_task(engine, "dw-fix", ["DW-1"], phase=Phase.DEV_RUNNING)
    write_intent = engine._write_intent

    def assert_adopted_before_write(bundle, dirname):
        assert task.dw_ids == list(bundle.dw_ids) == ["DW-2"]
        return write_intent(bundle, dirname)

    monkeypatch.setattr(engine, "_write_intent", assert_adopted_before_write)

    key = engine._run_bundle(Bundle(name="fix", dw_ids=("DW-2",), intent="second"), 1)

    task = engine.state.tasks["dw-fix"]
    assert key == "dw-fix" and list(engine.state.tasks) == ["dw-fix"]
    assert task.dw_ids == ["DW-2"]
    # the document, the task and the close all name ONE set of ids
    intent = (engine.run_dir / "bundles" / "fix" / "intent.md").read_text(encoding="utf-8")
    assert "dw_ids: DW-2" in intent and "DW-1" not in intent
    assert task.bundle_closes_intended == ["DW-2"]
    entries = ledger_entries(project)
    assert not entries["DW-2"].open and entries["DW-1"].open
    (rec,) = _records(engine, "sweep-bundle-dwids-adopted")
    assert rec["story_key"] == "dw-fix"
    assert rec["previous_dw_ids"] == ["DW-1"] and rec["dw_ids"] == ["DW-2"]


@pytest.mark.parametrize(
    "persisted", [["DW-1", "DW-2"], ["DW-2", "DW-1"]], ids=["same-order", "reordered"]
)
def test_run_bundle_adopts_silently_when_the_reset_task_already_agrees(
    project, monkeypatch, persisted
):
    """Assign unconditionally, announce only on divergence: a bundle's identity is
    its ids under SET equality (`_bundle_name_for` says so), so a regenerated
    triage that merely reorders them is not a divergence worth a record on every
    resume.

    The name says "silently", but the `reordered` case is carrying a second and
    less obvious claim: it is the row that pins the assignment as UNCONDITIONAL.
    `same-order` cannot — with the persisted ids already equal to the bundle's,
    adoption and no-adoption are indistinguishable, so that case grades only the
    silence. Under `reordered` the two differ, and the assert on the bundle's own
    ORDER is what fails if someone folds the assignment inside the divergence
    check. So the pair is not collapsible to assign-only-on-divergence, and
    neither case is redundant."""
    write_ledger(project, {"DW-1": "open", "DW-2": "open"})
    engine, _ = make_sweep(project, [])
    _bundle_task(engine, "dw-fix", persisted, phase=Phase.DEV_RUNNING)
    dispatched = _stub_run_story(engine, monkeypatch)

    key = engine._run_bundle(Bundle(name="fix", dw_ids=("DW-1", "DW-2"), intent="again"), 1)

    assert key == "dw-fix" and dispatched == [engine.state.tasks["dw-fix"]]
    # the bundle's OWN order, so the task agrees with the document written beside it
    assert engine.state.tasks["dw-fix"].dw_ids == ["DW-1", "DW-2"]
    assert not _records(engine, "sweep-bundle-dwids-adopted")


def test_run_bundle_adopts_onto_a_reset_task_with_no_persisted_dw_ids(project, monkeypatch):
    """The pre-`dw_ids` `state.json` shape (`model.py` loads a missing key as `[]`)
    on the reset path, where it means the OPPOSITE of what it means to DW-125 —
    which is why it gets a row of its own rather than riding along on the sibling
    above.

    `_bundle_name_for` reads an empty persisted list as agreeing with anything, so
    a paused legacy run is not re-driven bundle by bundle. Here the same emptiness
    is a divergence: the task genuinely has no ids, nothing else will supply them,
    and dispatching it without adopting the bundle's would hand
    `_close_bundle_ledger_when_spec_status` an empty `bundle_closes_intended` and
    close nothing at all. Both readings are correct for their own question, and
    the `_run_bundle` comment claims exactly this — so it is pinned here.

    Ablation: guard the assignment with `if task.dw_ids:` (the shape someone
    copying DW-125's empty-agrees-with-anything rule would write) and the adopted
    assert reddens at `[]`."""
    write_ledger(project, {"DW-1": "open", "DW-2": "open"})
    engine, _ = make_sweep(project, [])
    _bundle_task(engine, "dw-fix", [], phase=Phase.DEV_RUNNING)
    dispatched = _stub_run_story(engine, monkeypatch)

    key = engine._run_bundle(Bundle(name="fix", dw_ids=("DW-2",), intent="second"), 1)

    assert key == "dw-fix" and dispatched == [engine.state.tasks["dw-fix"]]
    assert engine.state.tasks["dw-fix"].dw_ids == ["DW-2"]
    # emptiness is announced like any other divergence — the record is what tells an
    # operator a legacy task took ids it was never minted with
    (rec,) = _records(engine, "sweep-bundle-dwids-adopted")
    assert rec["story_key"] == "dw-fix"
    assert rec["previous_dw_ids"] == [] and rec["dw_ids"] == ["DW-2"]


def test_run_bundle_adopts_nothing_when_recovery_finished_the_bundle(project, monkeypatch):
    """The adoption sits on the recovery-returned-FALSE arm alone. A persisted
    receipt that carries the bundle through commit returns True, and the caller is
    done: no intent is written over it, and the finished task keeps the ids it
    actually ran — writing this bundle's ids onto it would misname work already
    closed under the previous ones.

    `_finalize_commit_phase` is stubbed the way its sibling recovery rows stub it:
    this is about what `_run_bundle` does with a True, not about the commit."""
    write_ledger(project, {"DW-1": "open", "DW-2": "open"})
    engine, _ = make_sweep(project, [])
    task = _bundle_task(engine, "dw-fix", ["DW-1"], phase=Phase.COMMITTING)
    dispatched = _stub_run_story(engine, monkeypatch)
    finalized: list[str] = []
    monkeypatch.setattr(engine, "_finalize_commit_phase", lambda t: finalized.append(t.story_key))

    key = engine._run_bundle(Bundle(name="fix", dw_ids=("DW-2",), intent="second"), 1)

    assert key == "dw-fix" and finalized == ["dw-fix"] and not dispatched
    assert task.dw_ids == ["DW-1"]  # the ids it ran under, untouched
    assert task.bundle_file is None  # never pointed at this bundle's document
    assert not (engine.run_dir / "bundles" / "fix" / "intent.md").exists()
    assert not _records(engine, "sweep-bundle-dwids-adopted")


def test_run_bundle_refuses_when_every_candidate_key_is_taken(project, monkeypatch):
    """The suffix search is bounded at 2-9 so it is provably finite. Exhausting
    it is loud on both surfaces and mints nothing — the ids stay open for the
    next sweep rather than being swallowed by a key that is not theirs.

    Ablation: drop the `sweep-bundle-key-collision` append and the record
    assertion fails at zero entries; drop the `gates.notify` and ATTENTION is
    never written."""
    write_ledger(project, {"DW-9": "open"})
    engine, _ = make_sweep(project, [])
    for name in ["fix"] + [f"fix-{n}" for n in range(2, 10)]:
        _bundle_task(engine, f"dw-{name}", ["DW-1"])
    dispatched = _stub_run_story(engine, monkeypatch)
    before = set(engine.state.tasks)

    key = engine._run_bundle(Bundle(name="fix", dw_ids=("DW-9",), intent="ninth"), 1)

    assert key is None and not dispatched
    assert set(engine.state.tasks) == before  # no task minted
    (rec,) = _records(engine, "sweep-bundle-key-collision")
    assert rec["name"] == "fix" and rec["dw_ids"] == ["DW-9"]
    attention = (engine.run_dir / "ATTENTION").read_text(encoding="utf-8")
    # the KEYS, not the bare name: from cycle 2 they carry a `dw<N>-` prefix, so
    # a name-only message names nothing the operator can find in state.json
    assert "'fix'" in attention and "DW-9" in attention
    assert "dw-fix through dw-fix-9" in attention and "cycle 1" in attention
    assert not _records(engine, "sweep-bundle-key-deduped")


def test_cycle_survives_a_bundle_that_could_not_be_named(project):
    """`bundles_done` grades the keys `_run_bundle` actually reports, so a bundle
    that yielded no key is neither counted nor looked up — the old re-derivation
    from `bundle.name` would have read the finished `dw-fix` task instead."""
    write_ledger(project, {"DW-9": "open"})
    plan = triage_result(["DW-9"], bundles=[{"name": "fix", "dw_ids": ["DW-9"], "intent": "i"}])
    engine, adapter = make_sweep(project, [triage_effect(plan)])
    for name in ["fix"] + [f"fix-{n}" for n in range(2, 10)]:
        _bundle_task(engine, f"dw-{name}", ["DW-1"])

    assert engine._cycle(1, {"DW-9"}) is False  # no addressable work

    assert len(adapter.sessions) == 1  # triage only; nothing dispatched for the bundle
    assert _records(engine, "sweep-bundle-key-collision")
    assert ledger_entries(project)["DW-9"].open


def test_resume_with_a_lost_triage_cache_runs_a_regenerated_same_named_bundle(project):
    """The end-to-end shape the guard exists for: a cycle-1 bundle `fix` (DW-1)
    finishes, the run is stopped mid-cycle, and the resume loses `triage.json`.
    The fresh triage re-authors the remaining id under the SAME name — which used
    to hit the finished task's key and vanish. It now runs as `dw-fix-2`.

    Ablation: restore the key-only early return and `dw-fix-2` is never created,
    DW-2 stays open, and the run finishes having done nothing."""
    write_ledger(project, {"DW-1": "open", "DW-2": "open"})
    run_dir = project.project / ".bmad-loop" / "runs" / "sweep-run"
    plan = triage_result(
        ["DW-1", "DW-2"],
        bundles=[
            {"name": "fix", "dw_ids": ["DW-1"], "intent": "first"},
            {"name": "later", "dw_ids": ["DW-2"], "intent": "second"},
        ],
    )
    engine, _ = make_sweep(
        project,
        [
            triage_effect(plan),
            bundle_dev_effect(project, "fix", ["DW-1"]),
            _lodge_after(bundle_review_effect(project, "fix"), run_dir),
        ],
    )
    engine.run()

    assert engine.state.tasks["dw-fix"].phase == Phase.DONE
    assert "dw-later" not in engine.state.tasks  # the stop landed between bundles
    assert ledger_entries(project)["DW-1"].status.startswith("done")

    (run_dir / "triage.json").unlink()  # the cache the resume would have reloaded
    regenerated = triage_result(
        ["DW-2"], bundles=[{"name": "fix", "dw_ids": ["DW-2"], "intent": "second"}]
    )
    resumed, _ = resume_sweep(
        project,
        engine,
        [
            triage_effect(regenerated),
            bundle_dev_effect(project, "fix-2", ["DW-2"]),
            bundle_review_effect(project, "fix-2"),
        ],
    )
    summary = resumed.run()

    assert not summary.paused
    assert resumed.state.tasks["dw-fix-2"].phase == Phase.DONE
    assert resumed.state.tasks["dw-fix-2"].dw_ids == ["DW-2"]
    assert ledger_entries(project)["DW-2"].status.startswith("done")
    # the already-finished bundle keeps its key, its ids and its phase
    assert resumed.state.tasks["dw-fix"].phase == Phase.DONE
    assert resumed.state.tasks["dw-fix"].dw_ids == ["DW-1"]
    (rec,) = _records(resumed, "sweep-bundle-key-deduped")
    assert rec["original"] == "fix" and rec["name"] == "fix-2"
    assert "bundle_name: fix-2" in (run_dir / "bundles" / "fix-2" / "intent.md").read_text(
        encoding="utf-8"
    )
    assert worktree_clean(project.project)
