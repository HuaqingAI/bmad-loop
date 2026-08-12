"""Central registry of the ``BMAD_LOOP_*`` runtime environment variables.

These operator/test override knobs used to be read inline at scattered call
sites (`engine`, `adapters.multiplexer`, `cli`, `process_host`), which left them
undiscoverable and undocumented. This module is the one place each is named,
typed, and given a reader; the operator-facing table lives in the README under
"Environment variables".

Each reader preserves its call site's exact semantics — same parse, same
fallback — so routing a site through here changes nothing observable. Only the
three core vars belong here; the `BMAD_LOOP_UNITY_*` / `BMAD_LOOP_ENGINE_*`
family read by the bundled Unity plugin's stand-alone helper scripts is that
plugin's own contract (documented in the game-engine guide) and stays with it.
"""

from __future__ import annotations

import math
import os

#: Overrides the per-session wall-clock budget, in seconds (test / E2E hook).
SESSION_TIMEOUT_S = "BMAD_LOOP_SESSION_TIMEOUT_S"
#: Forces the terminal-multiplexer backend by registered name.
MUX_BACKEND = "BMAD_LOOP_MUX_BACKEND"
#: Forces the process-host implementation by registered name (test / override).
PROCESS_HOST = "BMAD_LOOP_PROCESS_HOST"


def session_timeout_s() -> float | None:
    """The per-session wall-clock override in seconds, or ``None`` when unset.

    Anything that is not a finite positive number of seconds reads as ``None``
    (ignored), and the guard is deliberately two-sided. Rejecting zero and
    negatives keeps a fat-fingered override from silently *shortening* a real
    run's budget; rejecting non-finite values keeps ``inf`` / ``1e999`` /
    ``Infinity`` — all of which ``float()`` accepts and all of which pass a bare
    ``> 0`` — from silently *removing* it. That second half matters more than it
    looks: both adapters build their monotonic and wall-clock deadlines by
    adding this to the current time, so a non-finite budget yields a deadline
    that can never expire, and this is the outer bound every stall-grace and
    wake-nudge window defers to. Losing it means an unattended run can wedge
    with no backstop left.

    A very large *finite* value is still honoured. It is a duration, however
    unwise, and an operator asking for one is expressing intent; ``inf`` is not
    a duration at all.
    """
    raw = os.environ.get(SESSION_TIMEOUT_S)
    if raw is None:
        return None
    try:
        value = float(raw)
    except ValueError:
        return None
    if not math.isfinite(value) or value <= 0:
        return None
    return value


def mux_backend() -> str | None:
    """The forced terminal-multiplexer backend name, or ``None`` when unset.

    Returned verbatim (callers test truthiness and resolve the name), so the
    forced-selection semantics match the raw env read exactly.
    """
    return os.environ.get(MUX_BACKEND)


def process_host() -> str | None:
    """The forced process-host name, or ``None`` when unset."""
    return os.environ.get(PROCESS_HOST)
