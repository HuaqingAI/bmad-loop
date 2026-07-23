"""Unit tests for the dev/review retry-budget decisions — specifically the
resolved-escalation guard that re-escalates instead of silently deferring."""

from bmad_loop.adapters.base import SessionResult
from bmad_loop.escalation import Action, decide_dev, decide_review_session
from bmad_loop.model import StoryTask
from bmad_loop.policy import LimitsPolicy, NotifyPolicy, Policy
from bmad_loop.verify import VerifyOutcome

POLICY = Policy(
    limits=LimitsPolicy(max_dev_attempts=2, max_review_cycles=2),
    notify=NotifyPolicy(desktop=False, file=True),
)
COMPLETED = SessionResult(status="completed", result_json={"escalations": []})
FAILING = VerifyOutcome.retry("spec status is 'in-progress', expected 'done'")


def _task(**kw) -> StoryTask:
    return StoryTask(story_key="9-0-x", epic=9, **kw)


def test_exhausted_budget_defers_normal_story():
    task = _task(attempt=2)  # 2 == max_dev_attempts -> budget spent
    assert decide_dev(task, COMPLETED, FAILING, POLICY).action == Action.DEFER


def test_exhausted_budget_reescalates_resolved_redrive():
    task = _task(attempt=2, resolved_redrive=True)
    decision = decide_dev(task, COMPLETED, FAILING, POLICY)
    assert decision.action == Action.PAUSE
    assert "re-escalating instead of deferring" in decision.reason


def test_budget_left_still_retries_even_when_resolved_redrive():
    task = _task(attempt=1, resolved_redrive=True)  # 1 < 2 -> budget remains
    decision = decide_dev(task, COMPLETED, FAILING, POLICY)
    assert decision.action == Action.RETRY
    # a plain retry must NOT carry the exhausted re-escalation wording — that reason
    # is journaled/fed back, and "re-escalating instead of deferring" is a lie here.
    assert "re-escalating" not in decision.reason


def test_noncompleted_session_reescalates_resolved_redrive():
    task = _task(attempt=2, resolved_redrive=True)
    crashed = SessionResult(status="crashed")
    assert decide_dev(task, crashed, None, POLICY).action == Action.PAUSE


def test_env_fault_outcome_pauses_even_with_budget_left():
    """An environment-fault verify outcome (rc 126/127 → CRITICAL escalate)
    pauses immediately — the attempt budget is never consulted for it."""
    task = _task(attempt=1)  # 1 < 2 -> budget remains
    env_fault = VerifyOutcome.escalate("verify environment fault (rc=127): pint", env_fault=True)
    decision = decide_dev(task, COMPLETED, env_fault, POLICY)
    assert decision.action == Action.PAUSE
    assert "rc=127" in decision.reason


def test_dev_env_fault_session_pauses_even_with_budget_left():
    """A dev session classified as a transport/API environment fault (#194) PAUSEs
    immediately — like the verify env-fault above, the attempt budget is never
    consulted, and the reason carries the evidence line."""
    task = _task(attempt=1)  # 1 < 2 -> budget remains
    env_fault = SessionResult(
        status="timeout", env_fault=True, env_fault_evidence="API Error: Connection refused"
    )
    decision = decide_dev(task, env_fault, None, POLICY)
    assert decision.action == Action.PAUSE
    assert "environment fault: dev session timeout" in decision.reason
    assert "API Error: Connection refused" in decision.reason


def test_dev_env_fault_session_pauses_even_when_budget_exhausted():
    """The env-fault pause outranks budget exhaustion too — a spent budget must not
    downgrade it to a defer (that would file a transport failure as deferred work)."""
    task = _task(attempt=2)  # 2 == max_dev_attempts -> budget spent
    env_fault = SessionResult(status="crashed", env_fault=True)
    decision = decide_dev(task, env_fault, None, POLICY)
    assert decision.action == Action.PAUSE
    assert "environment fault" in decision.reason


def test_dev_plain_noncompleted_still_retries_with_budget():
    """Guard pin: a NON-env-fault timeout with budget left still RETRYs — the
    env-fault branch must not swallow ordinary transient failures."""
    task = _task(attempt=1)
    plain = SessionResult(status="timeout")  # env_fault defaults False
    decision = decide_dev(task, plain, None, POLICY)
    assert decision.action == Action.RETRY
    assert "environment fault" not in decision.reason


def test_review_env_fault_session_pauses():
    """The same classification pauses a review session (evidence in the reason),
    where a plain non-completed review would RETRY/DEFER."""
    task = _task(review_cycle=1)  # budget remains
    env_fault = SessionResult(
        status="stalled", env_fault=True, env_fault_evidence="API Error: ETIMEDOUT"
    )
    decision = decide_review_session(task, env_fault, POLICY)
    assert decision.action == Action.PAUSE
    assert "environment fault: review session stalled" in decision.reason
    assert "ETIMEDOUT" in decision.reason


def test_review_plain_noncompleted_still_retries_with_budget():
    task = _task(review_cycle=1)
    plain = SessionResult(status="crashed")
    assert decide_review_session(task, plain, POLICY).action == Action.RETRY


def test_review_exhausted_defers_normal_story():
    task = _task(review_cycle=2)  # 2 == max_review_cycles
    crashed = SessionResult(status="crashed")
    assert decide_review_session(task, crashed, POLICY).action == Action.DEFER


def test_review_exhausted_reescalates_resolved_redrive():
    task = _task(review_cycle=2, resolved_redrive=True)
    crashed = SessionResult(status="crashed")
    decision = decide_review_session(task, crashed, POLICY)
    assert decision.action == Action.PAUSE
    assert "re-escalating instead of deferring" in decision.reason
