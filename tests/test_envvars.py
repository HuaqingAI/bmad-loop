"""Registry tests for the three core `BMAD_LOOP_*` runtime overrides.

`envvars` is the one place each core var is named, typed, and given a reader, so
what is pinned here is the *contract* the call sites (`engine`,
`adapters.multiplexer`, `cli`, `process_host`) and the README's "Environment
variables" table both depend on: the literal names, and each reader's parse and
fallback. The two name readers pass their value through verbatim on purpose —
validation lives downstream in the registry that resolves the name — so a test
asserting rejection here would be asserting the wrong module's job.

Contract parity: `test_engine.py::test_session_timeout_s_env_override` pins the
same rejection set one layer up, through `Engine._session_timeout_s` (does the
policy default survive a bad override?). This file grades the reader itself
(what does the parse return?), which is why it carries the rows that only a
direct read can distinguish — `nan`, which parses and is rejected by the
comparison — and the two name readers the engine never touches. Deliberately
layered, not duplicated: a behavior change lands in both or records the
divergence.
"""

import pytest

from bmad_loop import envvars


def test_constants_are_the_literal_env_var_names():
    """The constants ARE the public contract: call sites import the name rather
    than spelling the string, and the README table documents these literals for
    operators. Renaming one silently retires an override — the variable an
    operator exports simply stops being read, with no error anywhere — so the
    strings are pinned, not merely the readers that consume them."""
    assert envvars.SESSION_TIMEOUT_S == "BMAD_LOOP_SESSION_TIMEOUT_S"
    assert envvars.MUX_BACKEND == "BMAD_LOOP_MUX_BACKEND"
    assert envvars.PROCESS_HOST == "BMAD_LOOP_PROCESS_HOST"


def test_session_timeout_s_is_none_when_unset(monkeypatch):
    """Unset is the ordinary case — `None` is what makes the engine keep its
    policy budget (`limits.session_timeout_min x 60`) instead of an override.

    Ablation target (verified 2026-08-11): delete the `if raw is None: return
    None` early-out and this test fails ALONE, on `TypeError: float() argument
    must be a string or a real number, not 'NoneType'` — a TypeError the
    `except ValueError` arm below deliberately does not catch, so the early-out
    is a separate gate rather than a shortcut through the parse."""
    monkeypatch.delenv(envvars.SESSION_TIMEOUT_S, raising=False)
    assert envvars.session_timeout_s() is None


# ABLATION (each gate run singly, 2026-08-11). The rejection is three independent
# gates, and ablating them together would grade none of them — it would redden
# every row at once while telling you nothing about which gate holds which:
#   (a) drop the `try/except ValueError` (parse straight into `float(raw)`) and
#       exactly the two unparseable rows fail — [not-a-number] and [] raise
#       `ValueError: could not convert string to float` out of the reader
#       instead of reading as None. Every other row keeps passing.
#   (b) drop the `value <= 0` half and exactly the four zero/negative rows fail
#       — [0], [0.0], [-1], [-0.5]. Nothing else moves.
#   (c) drop the `not math.isfinite(value)` half and FIVE rows fail: the four
#       non-finite spellings [inf], [1e999], [Infinity], [INF] — and [nan].
#       That last one is the measured surprise, and it is load-bearing. Under
#       the old `return value if value > 0 else None`, nan was rejected by the
#       comparison (every comparison against nan is False). The guard is now
#       spelled `value <= 0`, and `nan <= 0` is ALSO False — so nan no longer
#       falls out of the comparison and is held by the finiteness check alone.
#       Same result, different gate: do not "simplify" (c) away on the theory
#       that `> 0` already covers nan, because this form does not.
# So every row names the gate that holds it. Rows [inf]/[1e999]/[Infinity]/[INF]
# are why the finiteness half exists: `float()` accepts all four and each passes
# a bare `> 0`, so before it they read as a real budget, and the deadline the
# adapters computed from it (`time.monotonic() + timeout_s`) could never expire.
@pytest.mark.parametrize(
    "raw",
    [
        "not-a-number",  # unparseable: ValueError out of float()
        "",  # empty: also a ValueError, and the spelling an unset-looking export leaves
        "0",  # parses, but zero would expire every session instantly
        "0.0",  # the float spelling of the same
        "-1",  # negative: already-elapsed budget
        "-0.5",  # negative float
        "nan",  # parses to nan; held by the finiteness check AND by `<= 0` being False
        "inf",  # parses to inf and passes `> 0` — a deadline that never arrives
        "1e999",  # overflows to inf: the same hole reachable without typing "inf"
        "Infinity",  # float()'s other accepted spelling
        "INF",  # float() is case-insensitive here, so the guard must be too
    ],
)
def test_session_timeout_s_ignores_a_value_that_cannot_be_a_budget(monkeypatch, raw):
    """Anything that is not a finite positive number of seconds reads as `None`
    (ignored) rather than as a budget, and the guard is two-sided because the
    failure mode is silent in BOTH directions. A fat-fingered `0` or `-1` would
    not error — it would shorten every session to nothing and read as a run of
    instant timeouts. `inf` (or `1e999`, which overflows to it) is the opposite
    and worse: both adapters build their deadlines as `time.monotonic() +
    timeout_s`, so a non-finite budget yields a deadline that never arrives, and
    this is the outer bound every stall-grace and wake-nudge window defers to.
    An unattended run would wedge with nothing left to stop it."""
    monkeypatch.setenv(envvars.SESSION_TIMEOUT_S, raw)
    assert envvars.session_timeout_s() is None


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("3", 3.0),
        ("0.5", 0.5),
        # The deliberate boundary of the finiteness guard, pinned so it is a
        # reviewable decision rather than an accident: a huge but FINITE value is
        # still honoured. It is a duration, however unwise, and an operator asking
        # for one is expressing intent — where `inf` is not a duration at all.
        # Drawing the line anywhere else would mean inventing a ceiling, which is
        # a policy number this module has no business choosing.
        ("1e308", 1e308),
    ],
)
def test_session_timeout_s_reads_a_positive_override_as_seconds(monkeypatch, raw, expected):
    """Both spellings land as a float: the int-looking one an operator types and
    the sub-second one the E2E gates rely on (`test_stories_e2e` drives a
    3-second budget through this seam). The value is seconds, not minutes — the
    policy key it overrides is in minutes, which is exactly the confusion the
    reader's name and this assertion pin down."""
    monkeypatch.setenv(envvars.SESSION_TIMEOUT_S, raw)
    assert envvars.session_timeout_s() == expected


def test_mux_backend_is_a_verbatim_passthrough(monkeypatch):
    """The reader forces nothing and validates nothing — it hands back the raw
    string so the forced-selection semantics match the raw env read exactly.
    An unregistered name must survive the read *intact*: the multiplexer
    registry is what raises on it, and it can only do that if the bad name
    reaches it. A reader that swallowed the unknown name would turn a loud
    misconfiguration into a silent auto-select.

    Ablation target (verified 2026-08-11): make the reader "helpful" —
    `return raw if raw in {"tmux", "psmux"} else None` — and this test fails
    alone (of this file's 13), on `assert None == 'no-such-backend'` — the
    unset and `tmux` rows above keep passing under it, so they do not carry the
    no-validation claim on their own."""
    monkeypatch.delenv(envvars.MUX_BACKEND, raising=False)
    assert envvars.mux_backend() is None

    monkeypatch.setenv(envvars.MUX_BACKEND, "tmux")
    assert envvars.mux_backend() == "tmux"

    monkeypatch.setenv(envvars.MUX_BACKEND, "no-such-backend")
    assert envvars.mux_backend() == "no-such-backend"


def test_process_host_is_a_verbatim_passthrough(monkeypatch):
    """Same contract as `mux_backend`, and for the same reason: `process_host`'s
    own registry raises `ProcessHostError` on an unregistered name rather than
    falling back to POSIX (on win32 `os.kill(pid, 0)` is destructive), so this
    reader must not filter the name on its way there."""
    monkeypatch.delenv(envvars.PROCESS_HOST, raising=False)
    assert envvars.process_host() is None

    monkeypatch.setenv(envvars.PROCESS_HOST, "posix")
    assert envvars.process_host() == "posix"

    monkeypatch.setenv(envvars.PROCESS_HOST, "bogus")
    assert envvars.process_host() == "bogus"
