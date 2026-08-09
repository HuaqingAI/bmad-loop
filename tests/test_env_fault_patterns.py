"""The shipped profiles' env_fault_patterns, held to both halves of the contract.

A pattern that misses a real provider outage burns a story's retry budget on an
environment fault (the bug that motivated this file). A pattern that fires on
ordinary model output pauses a *healthy* run — the inverse bug, and the worse of
the two, because the miss merely reproduces today's behaviour while the false
positive halts work that was fine. Both halves are cheap to assert and expensive
to rediscover, so the corpora below are the gate: add a line here before
loosening a pattern.

Only two profiles seed patterns, and only because captured output exists for
them. The other four stay inert deliberately — see test_unseeded_profiles_stay_inert.

WHAT BAIT IS FOR. The tmux adapters match against a tmux pane capture, which
contains the MODEL'S OWN OUTPUT: a story about quota handling prints this
vocabulary constantly, in fixtures, banners, acceptance criteria, docs and
assertions. Every BAIT line is something a healthy session plausibly emits.
The opencode adapter is structurally different — the file it scans is
`<task_id>.server.out`, the `opencode serve` process's own stdout/stderr, which
the model cannot write to at all. (NOT `<task_id>.log`, which is that adapter's
curated `[bmad]` conversation transcript and does carry the model's words.) It is
held to the same corpus anyway, because a pattern loose enough to be wrong on a
pane log should not be trusted just because today's log happens to be clean —
and because which file is scanned has already changed once underneath these
patterns.

KNOWN LIMIT, so nobody mistakes a green run here for proof. This gate matches
line-by-line against text, whereas the real pane log is a raw `tmux pipe-pane`
byte stream carrying cursor-addressed repaints (the TUI renders the same file
through pyte — see tui/data.py). Line and column position in that stream do not
correspond to what was on screen. So patterns must not depend on POSITION,
only on content; a positional anchor would pass here and be meaningless in
production.
"""

from __future__ import annotations

import pytest
import regex

from bmad_loop.adapters.profile import get_profile

# The one profile whose patterns are backed by captured output from a real outage
# AND whose log the model cannot write to. Held to the full BAIT corpus.
SEEDED_PROFILES = ("opencode",)
# Profiles that deliberately seed NOTHING, so classification stays inert for them.
# Patterns were drafted for these and withdrawn: they could only be written from
# error strings scraped off public issue trackers, for CLIs nobody here has run.
# An unverified pattern is not a neutral bet — one that fires on a healthy session
# pauses the whole run, which is worse than the fault it was meant to catch.
# Seeding nothing costs only the status quo. See test_unseeded_profiles_stay_inert.
UNSEEDED_PROFILES = ("codex", "gemini", "copilot", "antigravity")
# claude ships #194's original connection pattern, which predates this module and
# is NOT held to BAIT — it cannot pass. See test_claude_pattern_is_known_to_false_positive.
LEGACY_PROFILES = ("claude",)
ALL_PROFILES = SEEDED_PROFILES + UNSEEDED_PROFILES + LEGACY_PROFILES

# The two logfmt shapes `opencode serve` emitted during a real 5-hour provider
# quota outage on the opencode-http adapter (zai-coding-plan/glm-5.2). The line
# SHAPE and the AI_APICallError text are reproduced exactly — those are what the
# patterns key on — but the correlation ids are FAKES: the run this was captured
# from was a private client project, and a real `run=`/`session.id=` pair is a
# usable handle on it. Keep them fake; ids carry no test signal.
OPENCODE_REAL = [
    'timestamp=2026-07-26T13:12:53.262Z level=ERROR run=fake0001 message="stream error" '
    "providerID=zai-coding-plan modelID=glm-5.2 session.id=ses_fake0000000000000000000001 "
    'small=false agent=general mode=subagent error.error="AI_APICallError: Usage limit '
    'reached for 5 hour. Your limit will reset at 2026-07-26 22:49:46"',
    'timestamp=2026-07-26T22:52:57.193Z level=ERROR run=fake0002 message="stream error" '
    "providerID=zai-coding-plan modelID=glm-5.2 session.id=ses_fake0000000000000000000002 "
    'small=false agent=build mode=primary error.error="AI_APICallError: Cannot connect to '
    'API: The socket connection was closed unexpectedly."',
]

# Claude Code surfaces provider failures behind its own "API Error" prefix. Only
# the CONNECTION class is covered — claude.toml ships #194's original pattern and
# nothing more. A quota/rate-limit pattern of the same shape was drafted for this
# profile and withdrawn: no captured Claude Code quota line exists here, and the
# "API Error" anchor is too weak to add a second cause class safely (see
# test_claude_pattern_is_known_to_false_positive). So an Anthropic plan's usage
# limit is still unclassified on this adapter — a real, deliberately-left gap.
CLAUDE_REAL = [
    "API Error: Connection error (ECONNREFUSED)",
]

# Every profile with patterns, paired with the lines its own CLI actually emits.
REAL_BY_PROFILE = [
    (name, line)
    for name, corpus in (("opencode", OPENCODE_REAL), ("claude", CLAUDE_REAL))
    for line in corpus
]

# Bait that actually REACHES a seeded profile's anchor. Kept separate because it
# is the half that is easiest to get wrong: a corpus can be large and still prove
# nothing if every line bounces off the anchor before the rest of the pattern is
# ever exercised. This list did exactly that once — it was written while claude
# was still seeded, and after claude was withdrawn not one line could reach the
# only remaining anchor, so the test guarded nothing while looking thorough.
#
# So the bar for a line here is mechanical, and test_anchor_reaching_bait_is_not_
# vacuous enforces it: every seeded profile must have at least one line carrying
# its literal anchor. Lines below reproduce `error.error="AI_APICallError: ` (the
# opencode anchor) inside ordinary healthy-session output.
ANCHOR_REACHING_BAIT = [
    # The cause OUTSIDE the quoted error value: a real but UNRELATED provider
    # error whose trailing logfmt fields merely happen to contain a cause word.
    # These are why the gap in the pattern is [^"] and not `.` — with `.` the lazy
    # gap walks past the closing quote and finds "quota" in a later field, so an
    # invalid-API-key error reads as a quota outage.
    'timestamp=2026-07-26T13:12:53.262Z level=ERROR message="stream error" '
    'error.error="AI_APICallError: Invalid API key" request.path=/v1/quota',
    'timestamp=2026-07-26T13:12:53.262Z level=ERROR message="stream error" '
    'error.error="AI_APICallError: Model not found" endpoint=https://example/rate-limits',
    'error.error="AI_APICallError: Model not found" note=see-the-rate-limit-docs',
    # The anchor named but not completed — prose about the field, not the field.
    "docs: the server writes error.error= with an AI_APICallError when the quota trips",
    # claude's framing, kept for the day a claude pattern is seeded again.
    '// TODO: surface "API Error: 429" to the user with a Retry-After hint',
    'expect(banner).toBe("API Error: 429 rate_limit_error")',
    "docs: document the API Error 429 quota path per AC-3",
    '  it("renders API Error: overloaded", () => {',
    "- [ ] AC-2: map API Error: 429 to a friendly rate limit message",
]

# The limit of content-based matching, pinned rather than papered over.
#
# A story that quotes a provider's error VERBATIM — the whole framing token and a
# real cause, inside the quotes — produces a line byte-identical to the emitted
# one. No content-based pattern can separate those, because there is nothing left
# to separate them BY. It is a property of the approach, not a gap in these
# patterns, and it is the reason four profiles ship nothing at all: their real
# refusals are bare sentences, so every citation of them is indistinguishable.
#
# It is acceptable for opencode ONLY because of where that profile's log comes
# from: the file the classifier scans is logs/<task_id>.server.out — the
# `opencode serve` process's own stdout/stderr, which the model cannot write to,
# so a citation cannot physically reach the scanned bytes. NOT logs/<task_id>.log,
# which is that adapter's curated `[bmad]` transcript and does carry the model's
# words (the file is chosen by EnvFaultMixin.ENV_FAULT_LOG_SUFFIX). If the
# classifier is ever repointed at the transcript, or the adapter tees model output
# into the server log, these lines become live false positives and the patterns
# must be withdrawn.
#
# Characterisation test — asserts they DO match, so the day that changes it shows.
INSEPARABLE_VERBATIM_CITATIONS = [
    'docs: the server logs error.error="AI_APICallError: Usage limit reached" on quota exhaustion',
    "  assert line == 'error.error=\"AI_APICallError: Usage limit reached for 5 hour\"'",
    '# fixture: error.error="AI_APICallError: rate limit" — see tests/fixtures/quota.log',
    "expect(parse(line)).toEqual({error: 'error.error=\"AI_APICallError: quota exceeded\"'})",
]

# The literal each seeded profile's patterns key on. Used to prove the corpus
# above actually reaches them, rather than trusting that it does.
PROFILE_ANCHOR_LITERAL = {"opencode": 'error.error="AI_APICallError: '}

# Ordinary output from a healthy session working on a story that involves rate
# limiting. Every one of these is reachable in a tmux pane capture. None may
# classify — a hit here means the loop pauses a run that was working fine.
BAIT = [
    "  def test_rate_limit_retries_with_backoff(self):  # story 4 AC-2",
    "PASS  tests/rate_limiter.test.ts (12 tests)",
    "Implementing the quota exceeded banner per AC-3.",
    "wrote src/middleware/rate_limit.py",
    "- [x] AC-4: requests over quota return 429 with a Retry-After header",
    "I'll add a 429 handler that surfaces a friendly message to the user.",
    "429 lines changed across 12 files",
    "grep -c 'rate limit' logs/app.log  # 429",
    "throw new Error('Rate limit exceeded');  // added in src/api/client.ts",
    "docs: explain the quota model and the 429 response contract",
    "INFO cleanup prune=7.days",
    "timestamp=2026-07-26T15:45:39.732Z level=INFO message=stream providerID=zai-coding-plan",
    # A story's own passing test whose NAME mentions the very error class the
    # anchors key on — the nastiest realistic collision.
    "PASS tests/test_api_error_handling.py::test_rate_limit_error_is_retried",
    # …and the same collision on a FAILING test, which puts an error-shaped word
    # ahead of the cause on the line. A pattern anchored on a bare "error|failed"
    # rather than on a CLI-specific token trips on exactly this.
    "FAILED tests/test_retry.py::test_econnrefused_is_retried",
    "FAILED tests/network.spec.ts > retries on ETIMEDOUT",
    "  ✗ Error: handles RESOURCE_EXHAUSTED from the quota middleware",
    "AssertionError: expected 429, got 200 — see src/quota/limiter.ts",
    # ---------------------------------------------------------------------
    # THE CLASS THAT WAS MISSED: the model CITING the provider's error, verbatim.
    #
    # Everything above is a story *paraphrasing* rate limiting, so a
    # CLI-specific token was enough to separate it. These lines instead contain
    # a CLI's exact error string — because the story's job IS that error:
    # fixture literals, golden JSON, assertion text, test names, banner copy,
    # ACs, doc bullets, commit subjects, diff hunks, stack frames.
    #
    # This class is why four profiles ship NO patterns. Their real refusals are
    # bare provider sentences ("Resource has been exhausted (e.g. check quota).")
    # with no framing to anchor on, so no amount of vocabulary separates the
    # emitted line from the quoted one. A positional scheme was tried — anchor on
    # the CLI "owning" column 0 — and abandoned as unsound: the pane log is a raw
    # pipe-pane byte stream, so position in it is not position on screen (see this
    # module's header). Lines below are retained as the standing bar any future
    # pattern for those CLIs must clear before it is seeded.
    'ERROR in test setup: expected "code": "insufficient_quota"',
    'assert msg == "Resource has been exhausted (e.g. check quota)."',
    '"You have exhausted your daily quota on this model." // banner copy, AC-2',
    'it("renders TerminalQuotaError", () => {',
    'docs: quota copy - "You exceeded your current quota, please check your plan..."',
    'expected = {"code": "quota_exceeded"}',
    '  mock.return_value = {"code":"rate_limited"}',
    'expected = {"error": {"code": 429, "message": "Resource has been exhausted '
    '(e.g. check quota).", "status": "RESOURCE_EXHAUSTED"}}',
    '  Received: {"error":{"code":429,"message":"Resource has been exhausted '
    '(e.g. check quota).","status":"RESOURCE_EXHAUSTED"}}',
    '+   {"error": {"code": 429, "message": "Resource has been exhausted '
    '(e.g. check quota).", "status": "RESOURCE_EXHAUSTED"}}',
    '  "status": "RESOURCE_EXHAUSTED",',
    'fixtures/gemini_429.json: {"error": {"code": 429, "status": "RESOURCE_EXHAUSTED"}}',
    "class TerminalQuotaError extends Error { constructor() { super('quota exhausted'); } }",
    "export { TerminalQuotaError, GaxiosError } from './errors';",
    "fix(quota): raise TerminalQuotaError: quota exhausted when the token bucket empties",
    "  47 |   throw new TerminalQuotaError('quota exhausted');",
    '# The upstream copy is "You have exhausted your daily quota on this model."',
    'TODO: handle "Quota exceeded for metric: generativelanguage.googleapis.com/'
    'generate_content_free_tier_requests"',
    "docs(errors): document Resource has been exhausted (e.g. check quota). handling",
    "* Copy per AC-6: You have exhausted your daily quota on this model.",
    '#   ERROR codex_api::endpoint::responses: error=http 429 Too Many Requests: Some("{',
    '    "ERROR: exceeded retry limit, last status: 429 Too Many Requests, '
    'request id: 9bd33f31fd269bb1-HNL",',
    '    "You have exhausted your daily quota on this model.",',
    '+  "Sorry, you\'ve hit a rate limit that restricts the number of Copilot model requests "',
    "2026-07-27T11:02:03.114Z ERROR app::api::client: error=http 429 Too Many Requests",
    "2026-07-27T11:02:03.114Z ERROR codex_client_stub: error=http 429 Too Many Requests",
    "Error sending request for url (https://api.openai.com/v1/chat/completions): "
    "error trying to connect: ECONNREFUSED",
    "Error sending request for url (http://localhost:8080/v1/responses): "
    "error trying to connect: ECONNREFUSED",
    "GaxiosError: request to https://sheets.googleapis.com/v4/spreadsheets failed, "
    "reason: connect ECONNREFUSED 127.0.0.1:443",
    "✗ Model call failed: surfaces quota_exceeded as a 402 (12 ms)",
    "  ✓ Model call failed: retries rate_limited twice then gives up",
    "│ Sorry, you've hit a rate limit — that's the banner string we render, per AC-2.",
    '  assert body["message"] == "Sorry, you have exceeded your Copilot token usage."',
    '- Model call failed: {"message":"You have no quota","code":"quota_exceeded"}  '
    "<- tests/fixtures/copilot_402.json",
    'it.each([["rate_limited", 429], ["quota_exceeded", 402]])("maps %s to %d", (code, status) => {',
    '  File "src/retry.py", line 41, in _raise_exhausted: '
    'RuntimeError("exceeded retry limit, last status: 429 Too Many Requests")',
    'raise UpstreamError("exceeded retry limit, last status: 503 Service Unavailable")',
    "> exceeded retry limit, last status: 429 Too Many Requests   "
    "(expected output, docs/retry.md)",
    '+    RESOURCE_EXHAUSTED_COPY = "Resource has been exhausted (e.g. check quota)."',
    '-        code: "rate_limit_exceeded",',
    'FIXTURE = {"error": {"code": 429, "message": "Resource has been exhausted '
    '(e.g. check quota).", "status": "RESOURCE_EXHAUSTED"}}',
    'expect(body).toEqual({"error":{"code":429,"message":"Resource has been exhausted '
    '(e.g. check quota).","status":"RESOURCE_EXHAUSTED"}})',
    'assert_eq!(err.to_string(), "exceeded retry limit, last status: 429 Too Many Requests");',
    'assert res.json()["error"]["status"] == "RESOURCE_EXHAUSTED"  # 429 path',
    'it("gives up once it has exceeded retry limit, last status: 503", async () => {',
    "  1) rate limiter › Model call failed: rate_limited (expected, retries once)",
    "stdout | src/quota.test.ts > Model call failed: quota_exceeded",
    '✗ should render "Quota exceeded for metric: generativelanguage.googleapis.com/x" '
    "in the toast",
    "  ✓ TerminalQuotaError is raised after the third 429 (18ms)",
    "# You have exhausted your daily quota on this model.  <- exact upstream copy, "
    "do not reword",
    "* You exceeded your current quota, please check your plan and billing details.",
    "> Resource has been exhausted (e.g. check quota).",
    "  GaxiosError: request to https://generativelanguage.googleapis.com/v1beta/models/x "
    "failed, reason: connect ECONNREFUSED  # expected in test_offline",
    "  ERROR codex_api::endpoint::responses: error=http 429 Too Many Requests   "
    "# pasted from openai/codex#9135",
    "Quota exceeded for metric: {metric}, limit: {limit}",
    'You have exhausted your daily quota on this model. Retry at {reset_at}."',
    "TerminalQuotaError = class extends Error {}",
    "2026-07-27T04:11:02.001Z ERROR quota_api::endpoint::responses: "
    "error=http 429 Too Many Requests",
    "2026-07-27T04:12:00.113Z ERROR app_api::responses: error=http 503 Service Unavailable",
    '[ERROR] billing: Model call failed shim returned {"code":"rate_limited"} (mocked)',
    "feat(retry): give up after exceeded retry limit, last status: 429",
    "fix(api): map RESOURCE_EXHAUSTED to a TerminalQuotaError subclass",
    "chore(copy): use \"Sorry, you've hit a rate limit that restricts the number of "
    'Copilot model requests" verbatim',
    "printf 'ERROR: exceeded retry limit, last status: %d\\n' \"$status\"",
    'console.log("Model call failed: " + JSON.stringify({ code: "rate_limited" }))',
    'throw new GaxiosError("connect ETIMEDOUT", { code: "ETIMEDOUT" });',
    '> Model call failed: {"message":"You have no quota","code":"quota_exceeded"}',
    'banner_text = "Sorry, you have exceeded your Copilot token usage. '
    'Please review our Terms of Service."',
    "- [ ] AC-7: surface the \"Sorry, you've hit a rate limit that restricts the number "
    'of Copilot model requests" toast verbatim',
    'RESPONSE_FIXTURE = {"error": {"code": 429, "message": "Resource has been exhausted '
    '(e.g. check quota).", "status": "RESOURCE_EXHAUSTED"}}',
    "expect(err).toBeInstanceOf(GaxiosError);  // ECONNREFUSED path, AC-5",
    'test("GaxiosError: request to https://generativelanguage.googleapis.com/v1beta failed, '
    'reason: connect ECONNREFUSED is retried", async () => {',
    'throw new Error("exceeded retry limit, last status: " + res.status);',
    'docs: document the "ERROR: exceeded retry limit, last status: 429" line codex prints',
    "fix(api): map codex_api error=http 429 onto our RetryableError type",
    '  ✓ Model call failed: {"message":"You have no quota","code":"quota_exceeded"} '
    "→ renders upgrade CTA (3 ms)",
    "      \"Sorry, you've hit a rate limit that restricts the number of Copilot model "
    'requests you can make within a specific time period."',
    "[app] Error sending request for url (http://localhost:8080/v1/chat): "
    "error trying to connect: ECONNREFUSED",
    "  Quota exceeded for metric: generativelanguage.googleapis.com/"
    "generate_content_free_tier_input_token_count, limit: 0  # golden fixture",
    "2026-07-27T09:14:02.117Z ERROR app::quota: error=http 429 Too Many Requests "
    "(raised by our own middleware)",
    '  ✗ quota banner shows "You have exhausted your daily quota on this model."',
    'chore: add fixture for the {"status": "RESOURCE_EXHAUSTED"} envelope',
    "You have exhausted your daily quota on this model.  <- expected copy for AC-2",
    'console.log("Model call failed: " + JSON.stringify({code: "rate_limited"}));',
    "TerminalQuotaError = class extends Error {}  // src/errors.ts",
    "  at TerminalQuotaError.handle (src/quota/errors.ts:42:11)",
    'PASS tests/quota.spec.ts > "You exceeded your current quota, please check your plan '
    'and billing details."',
    "Resource has been exhausted (e.g. check quota).  # copied from the Google docs",
    'banner.text = "Sorry, you have exceeded your Copilot token usage"',
    "2026-07-27T09:15:44.001Z ERROR codex_api_shim::mock: replaying error=http 429 "
    "fixture for the retry test",
    "- Sorry, you've hit a rate limit that restricts the number of Copilot model requests",
]

# A duplicated bait line silently weakens the corpus (it reads as coverage it is
# not), and pytest's parametrize ids collide on it. Guard the list itself.
assert len(BAIT) == len(set(BAIT)), "BAIT contains duplicate lines"


def _patterns(name: str) -> tuple[regex.Pattern[str], ...]:
    return tuple(regex.compile(p) for p in get_profile(name).env_fault_patterns)


def _matches(name: str, line: str) -> bool:
    return any(p.search(line) for p in _patterns(name))


@pytest.mark.parametrize("name", SEEDED_PROFILES)
@pytest.mark.parametrize("line", BAIT)
def test_seeded_profiles_never_classify_healthy_model_output(name: str, line: str) -> None:
    """The false-positive half. A hit pauses a run that was working."""
    assert not _matches(name, line), f"{name}.toml false-positives on: {line}"


@pytest.mark.parametrize("name", SEEDED_PROFILES)
@pytest.mark.parametrize("line", ANCHOR_REACHING_BAIT)
def test_seeded_profiles_survive_bait_that_reaches_their_anchor(name: str, line: str) -> None:
    """The half that a large corpus can silently fail to test. Lines here carry the
    CLI's own framing token, so they get past the anchor and put the rest of the
    pattern under real pressure."""
    assert not _matches(name, line), f"{name}.toml false-positives on: {line}"


@pytest.mark.parametrize("line", INSEPARABLE_VERBATIM_CITATIONS)
def test_verbatim_citation_of_the_real_error_is_inseparable(line: str) -> None:
    """Characterises the limit of content-based matching: a verbatim citation of
    the provider's error IS the provider's error, byte for byte, so it matches.

    Asserted positively on purpose. It is safe only because opencode's scanned log
    is the serve process's own stdout — ``<task_id>.server.out``, which the model
    cannot write to. So if this ever starts failing, either the patterns gained a
    citation-awareness they cannot soundly have, or the scanned file gained model
    output and the patterns must go. Either way it should not change silently.

    That second case is not hypothetical: ``<task_id>.log`` USED to be the server's
    stdout and these patterns were seeded on that basis, then it was repurposed as
    a curated `[bmad]` conversation transcript carrying the model's own words. The
    classifier kept scanning it. Every test here still passed, because they write
    their fixture to whatever path the code resolves — only the end-to-end test
    caught it. If the scanned file is ever repointed again, re-derive this safety
    argument before trusting it: see EnvFaultMixin.ENV_FAULT_LOG_SUFFIX."""
    assert _matches("opencode", line)


@pytest.mark.parametrize("name", SEEDED_PROFILES)
def test_anchor_reaching_bait_is_not_vacuous(name: str) -> None:
    """Guards the guard. The test above is worthless unless its corpus actually
    contains the anchor under test — and it silently stopped doing so once, when
    the profile the corpus had been written for was withdrawn. Assert reachability
    mechanically instead of trusting it."""
    anchor = PROFILE_ANCHOR_LITERAL[name]
    reaching = [line for line in ANCHOR_REACHING_BAIT if anchor in line]
    assert reaching, f"no ANCHOR_REACHING_BAIT line contains {name}'s anchor {anchor!r}"


@pytest.mark.parametrize(("name", "line"), REAL_BY_PROFILE)
def test_each_seeded_profile_catches_its_own_cli_error_lines(name: str, line: str) -> None:
    """The false-negative half, per CLI. Each seeded profile is anchored on a
    framing token its CLI emits, so a miss here means a real provider outage would
    burn the story's retry budget instead of pausing the run."""
    assert _matches(name, line), f"{name}.toml misses a real error line: {line[:120]}"


@pytest.mark.parametrize("name", SEEDED_PROFILES)
def test_patterns_require_an_anchor_not_a_bare_cause(name: str) -> None:
    """The doctrine itself: a cause word alone must never be sufficient. Guards
    against a future edit that drops an anchor for 'better coverage'."""
    for bare in ("quota", "rate limit", "429", "usage limit reached", "too many requests"):
        assert not _matches(name, bare), f"{name}.toml classifies the bare cause {bare!r}"


CLAUDE_KNOWN_FALSE_POSITIVES = [
    '  assert log == "API Error: Connection refused"',
    'docs: explain the "API Error: Unable to connect" retry path',
    "- [ ] AC-5: show a banner on API Error: Connection timed out",
    'expect(msg).toBe("API Error: Connection error")',
    "// handle API Error: ECONNREFUSED by retrying with backoff",
]


@pytest.mark.parametrize("line", CLAUDE_KNOWN_FALSE_POSITIVES)
@pytest.mark.xfail(strict=True, reason="pre-existing #194 defect: claude's pattern matches prose")
def test_claude_pattern_is_known_to_false_positive(line: str) -> None:
    """Documents debt rather than hiding it. claude.toml's `API Error.*<cause>`
    predates this module and cannot pass BAIT: its pane log is a tmux capture of
    the model's own output, so a story that merely writes ABOUT provider errors —
    a TODO, an assertion, an acceptance criterion — reads as an environment fault
    and pauses a healthy run.

    It also fires on this repo's own tracked files (CHANGELOG.md, docs/FEATURES.md
    and the profile comment describing the pattern), so a bmad-loop session
    working on bmad-loop is exposed.

    Left in place here, not widened: a quota pattern of the same shape was drafted
    for this profile and withdrawn for exactly this reason, so the gap is real —
    an Anthropic plan's usage limit is still unclassified. Fixing it needs a
    captured Claude Code quota line and a stronger anchor than "API Error", which
    is a separate change from the one this branch makes.

    strict=True on purpose: the day the pattern is fixed, this test fails and
    forces the debt note to be removed with it."""
    assert not _matches("claude", line)


@pytest.mark.parametrize("name", UNSEEDED_PROFILES)
def test_unseeded_profiles_stay_inert(name: str) -> None:
    """These four ship no patterns ON PURPOSE, and this pins that.

    Patterns for them were drafted and withdrawn. They could only be written
    against error strings scraped from public issue trackers — no captured output
    from a real outage on any of these CLIs exists here — and an unverified
    pattern is not a neutral bet: one that fires on a healthy session pauses the
    entire run, which is worse than the fault it is meant to catch. Seeding
    nothing costs only the status quo.

    Do not seed these from a plausible-looking string. Seed them from a captured
    log line, with the run it came from cited, and add it to REAL_BY_PROFILE and
    this module's BAIT in the same change."""
    assert get_profile(name).env_fault_patterns == ()
