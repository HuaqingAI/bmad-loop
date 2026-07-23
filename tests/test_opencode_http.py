"""OpencodeHttpAdapter: unit tests + fake-binary E2E against a FakeOpencode.

The E2E cases spawn the adapter's real code path end to end: a fake `opencode`
binary (a stdlib-only HTTP server implementing the pinned 1.18.2 surface —
see the ``opencode_http`` module docstring) is launched by the adapter itself via the
conftest ``write_script_launcher`` shim, scripted per scenario through env vars
riding ``spec.env`` (the same channel the engine's BMAD_LOOP_* contract uses).
Everything binds 127.0.0.1; no real opencode binary or network access anywhere.
"""

from __future__ import annotations

import importlib.util
import json
import os
import queue
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest
from conftest import write_script_launcher

from bmad_loop.adapters import generic, opencode_http
from bmad_loop.adapters.base import SessionHandle, SessionSpec
from bmad_loop.adapters.generic import BUDGET_NUDGE_TEXT, NUDGE_TEXT, STALL_NUDGE_TEXT
from bmad_loop.adapters.opencode_http import (
    OpencodeDevAdapter,
    OpencodeHttpAdapter,
    OpencodeServerError,
    _free_port,
    _now_ms,
    _parse_sse_lines,
    _ServerSession,
    _sum_usage,
)
from bmad_loop.adapters.profile import get_profile
from bmad_loop.bmadconfig import ProjectPaths
from bmad_loop.model import TokenUsage
from bmad_loop.policy import LimitsPolicy, NotifyPolicy, Policy
from bmad_loop.process_host import ProcessHostError, get_process_host

# A pinned example timestamp from the pins file (§4): OpenCode `time.*` values
# are epoch MILLISECONDS. The proof-of-work floor must live in the same unit —
# a ns-vs-ms comparison is always False and silently disables the poll
# fallback, so these tests anchor the unit explicitly.
PINNED_EPOCH_MS = 1_784_218_739_410

# ---------------------------------------------------------------- FakeOpencode
#
# Scenario contract (FAKE_OPENCODE_SCENARIO):
#   completed          prompt -> write result.json -> SSE session.idle
#   nudge-then-complete first prompt idles result-less; the second (the nudge)
#                      writes the result and idles
#   stall              every prompt idles result-less, forever
#   busy-forever       the turn never finishes and never idles
#   busy-big-usage     busy-forever, but /session/:id/message reports an
#                      assistant message with huge token counts mid-turn
#                      (completed=0, so the poll fallback never reads it as
#                      proof-of-work) — the budget-guard runaway
#   big-usage-then-complete  same huge mid-turn usage, but the turn finishes
#                      after ~0.5s (result + idle) — the warn-mode runaway
#                      that runs to its natural end
#   die-after-result   prompt -> write result.json -> the server process exits
#   die-no-result      prompt -> the server process exits
#   sse-black-hole     the SSE stream closes right after connecting (every
#                      reconnect); the turn completes result+messages only —
#                      completion is reachable only via the HTTP poll fallback
# FAKE_OPENCODE_START_FAILURES=N makes the first N spawns exit(1) pre-bind.
# FAKE_OPENCODE_SPEC_PATH/_SPEC_TEXT: a bmad-dev-auto-style terminal spec the
# turn writes wherever a scenario writes its result (and at the start of
# busy-forever, for the post-kill rescue). Unset = no-op, like RESULT_PATH.
#
# Recordings under FAKE_OPENCODE_DIR: sessions.jsonl (incl. the Authorization
# header), prompts.jsonl, aborts.jsonl, pid (the server's own pid).

FAKE_OPENCODE = r"""
import json, os, sys, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

SCENARIO = os.environ.get("FAKE_OPENCODE_SCENARIO", "completed")
REC_DIR = os.environ["FAKE_OPENCODE_DIR"]
RESULT_PATH = os.environ.get("FAKE_OPENCODE_RESULT_PATH", "")
SPEC_PATH = os.environ.get("FAKE_OPENCODE_SPEC_PATH", "")
SPEC_TEXT = os.environ.get("FAKE_OPENCODE_SPEC_TEXT", "")
START_FAILURES = int(os.environ.get("FAKE_OPENCODE_START_FAILURES", "0"))

argv = sys.argv[1:]
assert argv and argv[0] == "serve", argv
PORT = int(argv[argv.index("--port") + 1])
HOST = argv[argv.index("--hostname") + 1]

if START_FAILURES:
    counter = os.path.join(REC_DIR, "start-count")
    n = 0
    if os.path.exists(counter):
        with open(counter, encoding="utf-8") as fh:
            n = int(fh.read().strip() or 0)
    with open(counter, "w", encoding="utf-8") as fh:
        fh.write(str(n + 1))
    if n < START_FAILURES:
        sys.exit(1)

SESSION_ID = "ses_fake0000000000000000000001"
LOCK = threading.Lock()
STATE = {"busy": False, "completed_ms": 0, "prompts": 0}
EVENTS = []


def now_ms():
    return int(time.time() * 1000)


def record(name, obj):
    with LOCK:
        with open(os.path.join(REC_DIR, name), "a", encoding="utf-8") as fh:
            fh.write(json.dumps(obj) + "\n")


def push(evt):
    with LOCK:
        EVENTS.append(evt)


def pop_events():
    with LOCK:
        out, EVENTS[:] = EVENTS[:], []
    return out


def idle_event():
    return {"type": "session.idle", "properties": {"sessionID": SESSION_ID}}


def write_result():
    if RESULT_PATH:
        with open(RESULT_PATH, "w", encoding="utf-8") as fh:
            json.dump({"ok": True, "workflow": "fake-triage"}, fh)


def write_spec():
    if SPEC_PATH:
        os.makedirs(os.path.dirname(SPEC_PATH), exist_ok=True)
        with open(SPEC_PATH, "w", encoding="utf-8") as fh:
            fh.write(SPEC_TEXT)


def finish_turn():
    with LOCK:
        STATE["completed_ms"] = now_ms()
        STATE["busy"] = False


def run_turn():
    with LOCK:
        STATE["busy"] = True
        n = STATE["prompts"]
    # let the 204 flush before any scripted death tears the connection down
    time.sleep(0.15)
    if SCENARIO == "completed":
        write_result(); write_spec(); finish_turn(); push(idle_event())
    elif SCENARIO == "nudge-then-complete":
        if n >= 2:
            write_result()
        finish_turn(); push(idle_event())
    elif SCENARIO == "stall":
        finish_turn(); push(idle_event())
    elif SCENARIO in ("busy-forever", "busy-big-usage"):
        write_spec()  # visible only post-kill: the turn never ends or idles
    elif SCENARIO == "big-usage-then-complete":
        time.sleep(0.35)  # stay busy through several fast heartbeat samples
        write_result(); finish_turn(); push(idle_event())
    elif SCENARIO == "die-after-result":
        write_result(); os._exit(0)
    elif SCENARIO == "die-no-result":
        os._exit(0)
    elif SCENARIO == "sse-black-hole":
        write_result(); finish_turn()  # completion visible over HTTP only


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _json(self, code, obj):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/global/health":
            self._json(200, {"healthy": True, "version": "1.18.2"})
        elif self.path == "/session/status":
            with LOCK:
                busy = STATE["busy"]
            self._json(200, {SESSION_ID: {"type": "busy"}} if busy else {})
        elif self.path.startswith("/session/") and self.path.endswith("/message"):
            with LOCK:
                done = STATE["completed_ms"]
            msgs = []
            if SCENARIO in ("busy-big-usage", "big-usage-then-complete"):
                msgs = [{
                    "info": {
                        "id": "msg_big1", "role": "assistant",
                        "time": {"created": now_ms() - 10, "completed": done},
                        "tokens": {"input": 4000000, "output": 1000000, "reasoning": 0,
                                   "cache": {"read": 0, "write": 0}},
                        "cost": 1.0,
                    },
                    "parts": [],
                }]
            elif done:
                msgs = [{
                    "info": {
                        "id": "msg_fake1", "role": "assistant",
                        "time": {"created": done - 10, "completed": done},
                        "tokens": {"input": 100, "output": 50, "reasoning": 5,
                                   "cache": {"read": 7, "write": 3}},
                        "cost": 0.01,
                    },
                    "parts": [],
                }]
            self._json(200, msgs)
        elif self.path == "/event":
            self._sse()
        else:
            self._json(404, {"name": "NotFoundError"})

    def _sse(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()

        def emit(evt):
            self.wfile.write(b"data: " + json.dumps(evt).encode("utf-8") + b"\n\n")
            self.wfile.flush()

        try:
            emit({"type": "server.connected", "properties": {}})
            if SCENARIO == "sse-black-hole":
                return  # close the stream: events are never deliverable
            last_beat = time.time()
            while True:
                for evt in pop_events():
                    emit(evt)
                if time.time() - last_beat > 0.2:
                    emit({"type": "server.heartbeat", "properties": {}})
                    last_beat = time.time()
                time.sleep(0.02)
        except OSError:
            return  # client went away

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(length) or b"{}")
        if self.path == "/session":
            record("sessions.jsonl",
                   {"body": body, "auth": self.headers.get("Authorization", "")})
            self._json(200, {"id": SESSION_ID, "title": body.get("title", ""),
                             "cost": 0,
                             "tokens": {"input": 0, "output": 0, "reasoning": 0,
                                        "cache": {"read": 0, "write": 0}},
                             "time": {"created": now_ms(), "updated": now_ms()}})
        elif self.path.endswith("/prompt_async"):
            record("prompts.jsonl", body)
            with LOCK:
                STATE["prompts"] += 1
            threading.Thread(target=run_turn, daemon=True).start()
            self.send_response(204)
            self.end_headers()
        elif self.path.endswith("/abort"):
            record("aborts.jsonl", {"path": self.path})
            self._json(200, True)
        else:
            self._json(404, {"name": "NotFoundError"})


if sys.platform == "win32":
    # SO_REUSEADDR on Windows allows binding a port already in LISTEN — under
    # xdist two fakes could silently share one port. Bind exclusively instead.
    ThreadingHTTPServer.allow_reuse_address = False
ThreadingHTTPServer.daemon_threads = True
ThreadingHTTPServer.block_on_close = False

server = ThreadingHTTPServer((HOST, PORT), Handler)
with open(os.path.join(REC_DIR, "pid"), "w", encoding="utf-8") as fh:
    fh.write(str(os.getpid()))
server.serve_forever(poll_interval=0.05)
"""


# -------------------------------------------------------------------- helpers


def _policy(**limits) -> Policy:
    return Policy(limits=LimitsPolicy(**limits) if limits else LimitsPolicy())


def _shrink_timing(adapter: OpencodeHttpAdapter) -> OpencodeHttpAdapter:
    """Shrink every cadence for tests; the defaults are minutes of wall clock."""
    adapter.health_timeout_s = 10.0
    adapter.health_poll_s = 0.05
    adapter.reconnect_sleep_s = 0.05
    adapter.silence_threshold_s = 2.0
    adapter.poll_tick_s = 0.05
    adapter.result_grace_s = 0.5
    return adapter


def make_adapter(tmp_path: Path, binary: str = "opencode", **kwargs) -> OpencodeHttpAdapter:
    adapter = OpencodeHttpAdapter(
        run_dir=tmp_path / "run",
        policy=kwargs.pop("policy", _policy()),
        profile=get_profile("opencode"),
        binary=binary,
        **kwargs,
    )
    return _shrink_timing(adapter)


@pytest.fixture
def fake_opencode(tmp_path):
    """The fake `opencode` launcher plus its recording dir."""
    rec = tmp_path / "recordings"
    rec.mkdir()
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    launcher = write_script_launcher(bin_dir, "opencode", FAKE_OPENCODE)
    return launcher, rec


def make_spec(
    tmp_path: Path,
    rec: Path,
    scenario: str,
    task_id: str = "t-1",
    timeout_s: float = 30.0,
    stall_nudges_cap: int | None = 6,
    extra_env: dict | None = None,
    **spec_kw,
) -> SessionSpec:
    env = {
        "FAKE_OPENCODE_SCENARIO": scenario,
        "FAKE_OPENCODE_DIR": str(rec),
        "FAKE_OPENCODE_RESULT_PATH": str(tmp_path / "run" / "tasks" / task_id / "result.json"),
        **(extra_env or {}),
    }
    return SessionSpec(
        task_id=task_id,
        role="triage",
        prompt="/bmad-loop-sweep run it",
        cwd=tmp_path,
        env=env,
        timeout_s=timeout_s,
        stall_nudges_cap=stall_nudges_cap,
        **spec_kw,
    )


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def assert_server_gone(rec: Path) -> None:
    """The fake's recorded pid must be dead once the adapter is done with it —
    `opencode serve` survives parent death, so a leak here is a real leak."""
    pid_file = rec / "pid"
    if not pid_file.is_file():
        return  # never got far enough to serve
    pid = int(pid_file.read_text(encoding="utf-8"))
    host = get_process_host()
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        if not host.is_alive(pid):
            return
        time.sleep(0.05)
    raise AssertionError(f"fake opencode server (pid {pid}) is still alive")


def prompt_texts(rec: Path) -> list[str]:
    return [p["parts"][0]["text"] for p in read_jsonl(rec / "prompts.jsonl")]


# ------------------------------------------------------------------ unit tests


def test_free_port_is_bindable():
    import socket

    port = _free_port()
    assert 0 < port < 65536
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", port))  # still free (racy by design, but just picked)


def test_ms_floor_unit_matches_opencode_timestamps():
    """OpenCode timestamps are epoch ms. The completion floor derives
    from time_ns // 1e6 and MUST be comparable to them: same unit, same epoch."""
    floor = time.time_ns() // 1_000_000
    assert abs(floor - _now_ms()) < 5_000
    # comparable to the pins' pinned example (a real 1.18.2 response value):
    # the floor is later than that 2026 timestamp but within the same magnitude.
    assert PINNED_EPOCH_MS < floor < PINNED_EPOCH_MS * 10


def test_config_content_shapes(tmp_path):
    adapter = make_adapter(tmp_path)
    spec = SessionSpec(task_id="t", role="triage", prompt="p", cwd=tmp_path)
    config = json.loads(adapter._config_content(spec))
    assert config["permission"] == "allow"
    # hermetic-skills recipe (adapter docstring): the project skill tree, absolute
    assert config["skills"]["paths"] == [str(tmp_path / ".claude" / "skills")]
    assert "model" not in config

    spec_model = SessionSpec(
        task_id="t", role="triage", prompt="p", cwd=tmp_path, model="anthropic/claude-x"
    )
    config = json.loads(adapter._config_content(spec_model))
    assert config["model"] == "anthropic/claude-x"


def test_session_env_carries_contract(tmp_path):
    adapter = make_adapter(tmp_path)
    spec = SessionSpec(
        task_id="t", role="triage", prompt="p", cwd=tmp_path, env={"BMAD_LOOP_TASK_ID": "t"}
    )
    env = adapter._session_env(spec, "sekrit")
    assert env["BMAD_LOOP_TASK_ID"] == "t"
    assert env["OPENCODE_SERVER_PASSWORD"] == "sekrit"
    assert env["OPENCODE_DISABLE_EXTERNAL_SKILLS"] == "1"
    json.loads(env["OPENCODE_CONFIG_CONTENT"])  # valid JSON


def test_sse_parser_accumulates_and_tolerates_junk():
    lines = [
        "data: " + json.dumps({"type": "server.connected", "properties": {}}),
        "",
        ": a comment",
        "id: 42",
        "data: not json",
        "",
        "data: " + json.dumps({"type": "session.idle", "properties": {"sessionID": "ses_1"}}),
        "",
    ]
    events = list(_parse_sse_lines(lines))
    assert [e["type"] for e in events] == ["server.connected", "session.idle"]


def test_sse_dispatch_filters_child_sessions(tmp_path):
    """A child/subagent session's idle must not read as the parent's turn-end,
    but its frames DO count as activity (the parent is silent while a child
    streams — the tmux pane-log analogue)."""
    from bmad_loop.adapters.opencode_http import _ServerSession

    adapter = make_adapter(tmp_path)
    sess = _ServerSession(process=None, port=0, base_url="", password="", log_fh=None)
    sess.session_id = "ses_parent"
    adapter._dispatch_sse(sess, {"type": "server.heartbeat", "properties": {}})
    assert sess.activity == 0 and sess.events.empty()
    adapter._dispatch_sse(sess, {"type": "session.idle", "properties": {"sessionID": "ses_child"}})
    assert sess.activity == 1 and sess.events.empty()
    adapter._dispatch_sse(
        sess, {"type": "message.part.updated", "properties": {"sessionID": "ses_child"}}
    )
    assert sess.activity == 2 and sess.events.empty()
    adapter._dispatch_sse(sess, {"type": "session.idle", "properties": {"sessionID": "ses_parent"}})
    assert sess.events.get_nowait() == "idle"
    adapter._dispatch_sse(
        sess, {"type": "session.error", "properties": {"sessionID": "ses_parent"}}
    )
    assert sess.events.get_nowait() == "error"


def test_sum_usage_maps_opencode_tokens():
    messages = [
        {"info": {"role": "user", "tokens": {"input": 999}}},  # not assistant: ignored
        {
            "info": {
                "role": "assistant",
                "tokens": {
                    "input": 100,
                    "output": 50,
                    "reasoning": 5,
                    "cache": {"read": 7, "write": 3},
                },
            }
        },
        {
            "info": {
                "role": "assistant",
                "tokens": {
                    "input": 10,
                    "output": 20,
                    "reasoning": 0,
                    "cache": {"read": 1, "write": 2},
                },
            }
        },
        {"info": {"role": "assistant"}},  # tokenless (e.g. aborted): ignored
    ]
    usage = _sum_usage(messages)
    assert usage == TokenUsage(
        input_tokens=110, output_tokens=75, cache_read_tokens=8, cache_creation_tokens=5
    )
    assert _sum_usage("garbage") == TokenUsage()


def test_missing_httpx_names_the_extra(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "httpx", None)
    with pytest.raises(OpencodeServerError, match=r"bmad-loop\[opencode\]"):
        make_adapter(tmp_path)


def test_missing_binary_is_a_clean_error(tmp_path):
    adapter = make_adapter(tmp_path, binary="definitely-not-a-real-binary-xyz")
    spec = SessionSpec(task_id="t", role="triage", prompt="p", cwd=tmp_path)
    with pytest.raises(OpencodeServerError, match="not found on PATH"):
        adapter.start_session(spec)


def test_kill_unknown_handle_is_a_noop(tmp_path):
    adapter = make_adapter(tmp_path)
    adapter.kill(SessionHandle(task_id="never-started", native_id="ses_x"))


@pytest.mark.skipif(sys.platform == "win32", reason="os.kill(0) reap probe is POSIX")
@pytest.mark.skipif(
    not sys.platform.startswith("linux") and importlib.util.find_spec("psutil") is None,
    reason="descendant discovery off Linux needs psutil (the non-linux extra)",
)
def test_kill_process_reaps_detached_descendant(tmp_path):
    """#183 mirror on the HTTP transport, deterministic without a real opencode
    binary: a focused _kill_process test with a real Popen server whose body
    detaches a child into its own session (``start_new_session=True`` — Python's
    portable setsid; macOS ships no setsid(1) utility, so the root SIGTERM cannot
    reach it) against the REAL process host. After _kill_process the detached child
    is reaped, proving the pre-signal descendant harvest + reap covers a straggler
    the pane/pgid kill would leak (a live opencode binary is not required, and the
    live-server harness cannot easily be made to detach a child — noted in the
    report)."""
    adapter = make_adapter(tmp_path)
    adapter.kill_wait_s = 3.0
    child_pid_file = tmp_path / "detached.pid"
    # The "server" detaches a session-leader child (records its pid), then idles so
    # it is provably alive at harvest — the server (process.pid) is the parent of
    # the detached child, so host.descendants(server) finds it before the SIGTERM.
    server_body = (
        "import subprocess, sys, time\n"
        "p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(300)'],"
        " start_new_session=True)\n"
        f"open({str(child_pid_file)!r}, 'w', encoding='utf-8').write(str(p.pid))\n"
        "time.sleep(300)\n"
    )
    process = subprocess.Popen([sys.executable, "-c", server_body])
    detached_pid = None
    try:
        deadline = time.monotonic() + 10
        while not child_pid_file.is_file() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert child_pid_file.is_file(), "server never recorded its detached child"
        detached_pid = int(child_pid_file.read_text(encoding="utf-8").strip())
        # sanity: the recorded pid is the setsid'd process and is currently alive
        os.kill(detached_pid, 0)

        sess = _ServerSession(process=process, port=0, base_url="", password="", log_fh=None)
        adapter._kill_process(sess)

        assert process.poll() is not None  # root server reaped
        reap_deadline = time.monotonic() + 10
        while True:
            try:
                os.kill(detached_pid, 0)
            except ProcessLookupError:
                break  # detached child reaped by the descendant sweep
            assert time.monotonic() < reap_deadline, f"detached child {detached_pid} survived"
            time.sleep(0.05)
    finally:
        for pid in (detached_pid, process.pid):
            if pid is None:
                continue
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError:
                pass
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pass


def test_kill_process_strikes_root_before_reraising_bad_host_override(tmp_path, monkeypatch):
    """The process-host lookup precedes the first signal; an explicit-but-bogus
    BMAD_LOOP_PROCESS_HOST must still raise loudly (never silently mis-signal), but
    the server must not be left alive behind the raise — one legacy Popen root
    strike fires, then ProcessHostError propagates (the tmux adapter's
    strike-before-reraise doctrine, mirrored)."""

    class _FakePopen:
        pid = 4242

        def __init__(self):
            self.terminated = 0
            self.killed = 0

        def poll(self):
            return None  # alive → _kill_process must not early-return

        def terminate(self):
            self.terminated += 1

        def kill(self):
            self.killed += 1

    adapter = make_adapter(tmp_path)
    process = _FakePopen()
    sess = _ServerSession(process=process, port=0, base_url="", password="", log_fh=None)
    monkeypatch.setenv("BMAD_LOOP_PROCESS_HOST", "bogus-host-name")
    get_process_host.cache_clear()
    try:
        with pytest.raises(ProcessHostError):
            adapter._kill_process(sess)
        if sys.platform == "win32":
            assert process.killed == 1  # the win32 legacy strike is Popen.kill()
        else:
            assert process.terminated == 1  # struck once before the raise
    finally:
        get_process_host.cache_clear()


class _ReapRecordingHost:
    """Minimal host for the reap-gate unit test: nobody dies on SIGTERM, so the
    force-kill loop is what settles survivors — proving the identity gate, not
    terminate, decides who gets force-killed. identity is None for ``no_identity``."""

    def __init__(self, alive=(), no_identity=()):
        self.alive = set(alive)
        self.no_identity = set(no_identity)
        self.terminated: list[int] = []
        self.force_killed: list[int] = []

    def identity(self, pid):
        return None if pid in self.no_identity else float(pid)

    def alive_and_ours(self, pid, identity):
        if pid not in self.alive:
            return False
        return identity is None or identity == self.identity(pid)

    def terminate(self, pid):
        self.terminated.append(pid)  # deliberately does NOT kill — force_kill settles it

    def force_kill(self, pid):
        self.force_killed.append(pid)
        self.alive.discard(pid)


def test_reap_descendants_never_signals_none_identity(tmp_path):
    """_reap_descendants signals only identity-confirmed stragglers; a
    None-identity survivor (a possible pid reuse) is never signalled AT ALL — no
    terminate, no force-kill, no poll burn (even a SIGTERM to a recycled pid kills
    an innocent process) — the ProcessHost contract, mirrored on the HTTP teardown
    (opencode_http.py:_reap_descendants)."""
    adapter = make_adapter(tmp_path)
    adapter.kill_wait_s = 0.1  # bound the poll: nobody dies on terminate here
    host = _ReapRecordingHost(alive={200, 400}, no_identity={400})
    tree = {200: 200.0, 400: None}  # 200 identity-confirmed at harvest, 400 unconfirmable
    adapter._reap_descendants(host, tree)
    assert host.terminated == [200]  # the unconfirmable 400 is never asked to stop
    assert host.force_killed == [200]  # only the confirmed pid escalated
    assert 400 not in host.force_killed


def test_read_usage_returns_stash_by_session_id(tmp_path):
    from bmad_loop.adapters.base import SessionResult

    adapter = make_adapter(tmp_path)
    adapter._usage["ses_1"] = TokenUsage(input_tokens=1)
    assert adapter.read_usage(SessionResult(status="completed", session_id="ses_1")).total == 1
    assert adapter.read_usage(SessionResult(status="completed", session_id="ses_2")) is None
    assert adapter.read_usage(SessionResult(status="completed")) is None


def test_sample_weighted_usage_inert_on_http_failure(tmp_path):
    """The budget guard's mid-session sample must never break the wait loop:
    no live session yet, a transport error, or a non-200 all read as None
    (guard inert this tick)."""
    adapter = make_adapter(tmp_path)
    spec = SessionSpec(task_id="t", role="triage", prompt="p", cwd=tmp_path)
    sess = _ServerSession(process=None, port=0, base_url="", password="", log_fh=None)
    assert adapter._sample_weighted_usage(sess, spec) is None  # no session id yet

    sess.session_id = "ses_1"

    class _BoomClient:
        def get(self, path):
            raise RuntimeError("connection refused")

    sess.client = _BoomClient()
    assert adapter._sample_weighted_usage(sess, spec) is None

    class _Client500:
        def get(self, path):
            class _Resp:
                status_code = 500

            return _Resp()

    sess.client = _Client500()
    assert adapter._sample_weighted_usage(sess, spec) is None


# ------------------------------------------------------------------- E2E tests


def test_e2e_completed(tmp_path, fake_opencode):
    launcher, rec = fake_opencode
    adapter = make_adapter(tmp_path, binary=str(launcher))
    spec = make_spec(tmp_path, rec, "completed")

    result = adapter.run(spec)

    assert result.status == "completed"
    assert result.result_json == {"ok": True, "workflow": "fake-triage"}
    assert result.session_id and result.session_id.startswith("ses_")
    # transcript = the raw messages dump; usage stashed for read_usage
    assert result.transcript_path and Path(result.transcript_path).is_file()
    usage = adapter.read_usage(result)
    assert usage == TokenUsage(
        input_tokens=100, output_tokens=55, cache_read_tokens=7, cache_creation_tokens=3
    )
    # the prompt went through the profile's template
    assert prompt_texts(rec) == ["Use the bmad-loop-sweep skill now: run it"]
    # authenticated transport end to end
    sessions = read_jsonl(rec / "sessions.jsonl")
    assert sessions and sessions[0]["auth"].startswith("Basic ")
    # teardown: registry empty, server dead, log tee landed
    assert adapter._sessions == {}
    assert_server_gone(rec)
    assert (tmp_path / "run" / "logs" / "t-1.log").exists()


def test_e2e_result_less_stop_nudges_then_completes(tmp_path, fake_opencode):
    launcher, rec = fake_opencode
    adapter = make_adapter(tmp_path, binary=str(launcher))
    spec = make_spec(tmp_path, rec, "nudge-then-complete")

    result = adapter.run(spec)

    assert result.status == "completed"
    texts = prompt_texts(rec)
    assert len(texts) == 2
    assert texts[1] == NUDGE_TEXT  # the wake-up carried the result-contract nudge
    assert_server_gone(rec)


def test_e2e_stall_after_nudge_budget(tmp_path, fake_opencode):
    launcher, rec = fake_opencode
    adapter = make_adapter(tmp_path, binary=str(launcher))
    spec = make_spec(tmp_path, rec, "stall")

    result = adapter.run(spec)

    # default budget: 1 stop-nudge; the second result-less idle is a stall
    assert result.status == "stalled"
    assert result.result_json is None
    assert len(prompt_texts(rec)) == 2
    breadcrumbs = read_jsonl(tmp_path / "run" / "tasks" / "t-1" / "resultless-stops.jsonl")
    assert breadcrumbs and all(b["verdict"] == "no-result-json" for b in breadcrumbs)
    assert_server_gone(rec)


def test_e2e_monotonic_stall_cap(tmp_path, fake_opencode):
    """#149 on the HTTP transport: every result-less idle refills the per-silence
    wake budget, so only the monotonic spec.stall_nudges_cap bounds a session
    that keeps ending its turn without a result."""
    launcher, rec = fake_opencode
    adapter = make_adapter(tmp_path, binary=str(launcher), stop_without_result_nudges=0)
    adapter._stall_grace_s = 0.3
    adapter._stall_nudges = 99  # per-silence budget effectively unbounded
    spec = make_spec(tmp_path, rec, "stall", stall_nudges_cap=2, timeout_s=60.0)

    result = adapter.run(spec)

    assert result.status == "stalled"
    assert result.result_json is None
    texts = prompt_texts(rec)
    # initial prompt + exactly cap stall-nudges, then the stall verdict
    assert len(texts) == 3
    assert texts[1] == STALL_NUDGE_TEXT and texts[2] == STALL_NUDGE_TEXT
    assert_server_gone(rec)


def test_e2e_timeout_aborts(tmp_path, fake_opencode):
    launcher, rec = fake_opencode
    adapter = make_adapter(tmp_path, binary=str(launcher))
    spec = make_spec(tmp_path, rec, "busy-forever", timeout_s=1.5)

    result = adapter.run(spec)

    assert result.status == "timeout"
    assert result.result_json is None
    aborts = read_jsonl(rec / "aborts.jsonl")
    assert aborts and "/abort" in aborts[0]["path"]
    assert_server_gone(rec)


# ------------------------------ mid-session token-budget guard (#158)
#
# Mirrors the generic-adapter guard on the HTTP transport: cumulative usage is
# sampled from GET /session/:id/message on the heartbeat cadence (the first
# tick always samples), and an enforce-mode trip nudges, arms the grace, then
# aborts the session and returns over_budget — the timeout path's exit shape.


def _budget_policy() -> Policy:
    return Policy(limits=LimitsPolicy(), notify=NotifyPolicy(desktop=False, file=True))


def test_e2e_budget_enforce_trips_nudges_and_aborts_over_budget(tmp_path, fake_opencode):
    launcher, rec = fake_opencode
    adapter = make_adapter(tmp_path, binary=str(launcher), policy=_budget_policy())
    spec = make_spec(
        tmp_path,
        rec,
        "busy-big-usage",
        timeout_s=30.0,
        token_budget=1_000_000,
        token_budget_mode="enforce",
        token_budget_grace_s=0.3,
    )

    result = adapter.run(spec)

    assert result.status == "over_budget"
    assert result.result_json is None
    # fake reports input 4M + output 1M, no cache: weighted = 5M
    assert result.budget_weighted == 5_000_000
    texts = prompt_texts(rec)
    assert len(texts) == 2 and texts[1] == BUDGET_NUDGE_TEXT
    aborts = read_jsonl(rec / "aborts.jsonl")
    assert aborts and "/abort" in aborts[0]["path"]
    # trip actions fired exactly once: one ATTENTION line, one breadcrumb
    attention = (tmp_path / "run" / "ATTENTION").read_text(encoding="utf-8")
    assert len(attention.splitlines()) == 1
    lifecycle = read_jsonl(tmp_path / "run" / "tasks" / "t-1" / "session-lifecycle.jsonl")
    tripped = [ln for ln in lifecycle if ln["event"] == "budget-tripped"]
    assert len(tripped) == 1
    assert tripped[0]["weighted"] == 5_000_000 and tripped[0]["mode"] == "enforce"
    # the verdict leaves a breadcrumb, like timeout-fired (#157 forensics)
    fired = [ln for ln in lifecycle if ln["event"] == "over-budget-fired"]
    assert len(fired) == 1
    assert fired[0]["weighted"] == 5_000_000 and fired[0]["zero_grace"] is False
    # usage was captured over HTTP before teardown
    assert result.transcript_path and Path(result.transcript_path).is_file()
    usage = adapter.read_usage(result)
    assert usage == TokenUsage(input_tokens=4_000_000, output_tokens=1_000_000)
    assert_server_gone(rec)


def test_e2e_budget_zero_grace_terminates_at_trip_without_nudge(tmp_path, fake_opencode):
    launcher, rec = fake_opencode
    adapter = make_adapter(tmp_path, binary=str(launcher), policy=_budget_policy())
    spec = make_spec(
        tmp_path,
        rec,
        "busy-big-usage",
        timeout_s=30.0,
        token_budget=1_000_000,
        token_budget_mode="enforce",
        token_budget_grace_s=0.0,
    )

    result = adapter.run(spec)

    assert result.status == "over_budget"
    assert result.budget_weighted == 5_000_000
    assert prompt_texts(rec) == ["Use the bmad-loop-sweep skill now: run it"]  # no nudge
    assert_server_gone(rec)


def test_e2e_budget_inert_under_cap(tmp_path, fake_opencode, monkeypatch):
    """A session whose weighted usage stays under its cap never trips: no
    ATTENTION, no breadcrumb, no budget_weighted on the completed result.
    The shrunk heartbeat guarantees samples actually observe the 5M weighted
    spend (the `completed` scenario would finish before ever reporting usage,
    leaving the comparison untested)."""
    monkeypatch.setattr(opencode_http, "HEARTBEAT_INTERVAL_S", 0.05)
    launcher, rec = fake_opencode
    adapter = make_adapter(tmp_path, binary=str(launcher), policy=_budget_policy())
    spec = make_spec(
        tmp_path,
        rec,
        "big-usage-then-complete",
        timeout_s=30.0,
        token_budget=10**9,
        token_budget_mode="enforce",
        token_budget_grace_s=240.0,
    )

    result = adapter.run(spec)

    assert result.status == "completed"
    assert result.budget_weighted is None
    assert not (tmp_path / "run" / "ATTENTION").exists()
    lifecycle_path = tmp_path / "run" / "tasks" / "t-1" / "session-lifecycle.jsonl"
    if lifecycle_path.exists():
        lifecycle = read_jsonl(lifecycle_path)
        assert not [ln for ln in lifecycle if ln["event"] == "budget-tripped"]
    assert_server_gone(rec)


def test_e2e_budget_warn_trips_once_and_completes(tmp_path, fake_opencode, monkeypatch):
    """Warn mode across MULTIPLE heartbeat samples (interval shrunk to 0.05s
    while the fake stays busy ~0.5s): exactly one ATTENTION line and one
    budget-tripped breadcrumb (the trip latch), NO nudge, NO abort while the
    session ran — it completes naturally with budget_weighted on the result."""
    monkeypatch.setattr(opencode_http, "HEARTBEAT_INTERVAL_S", 0.05)
    launcher, rec = fake_opencode
    adapter = make_adapter(tmp_path, binary=str(launcher), policy=_budget_policy())
    spec = make_spec(
        tmp_path,
        rec,
        "big-usage-then-complete",
        timeout_s=30.0,
        token_budget=1_000_000,
        token_budget_mode="warn",
        token_budget_grace_s=240.0,
    )

    handle = adapter.start_session(spec)
    try:
        result = adapter.wait_for_completion(handle, spec)
        # no abort while the session ran (kill() below aborts at teardown)
        assert not (rec / "aborts.jsonl").exists()
    finally:
        adapter.kill(handle)

    assert result.status == "completed"
    assert result.result_json == {"ok": True, "workflow": "fake-triage"}
    assert result.budget_weighted == 5_000_000
    assert prompt_texts(rec) == ["Use the bmad-loop-sweep skill now: run it"]  # no nudge
    attention = (tmp_path / "run" / "ATTENTION").read_text(encoding="utf-8")
    assert len(attention.splitlines()) == 1
    lifecycle = read_jsonl(tmp_path / "run" / "tasks" / "t-1" / "session-lifecycle.jsonl")
    tripped = [ln for ln in lifecycle if ln["event"] == "budget-tripped"]
    assert len(tripped) == 1
    assert tripped[0]["mode"] == "warn"
    assert [ln for ln in lifecycle if ln["event"] == "over-budget-fired"] == []
    assert_server_gone(rec)


# Clock-driven budget unit tests: the _timeout_driven_session machinery plus a
# fake control client whose /message answer reports runaway usage (weighted 5M).


class _BigUsageClient:
    def __init__(self):
        self.posts: list[str] = []

    def get(self, path):
        class _Resp:
            status_code = 200

            @staticmethod
            def json():
                return [
                    {
                        "info": {
                            "role": "assistant",
                            "tokens": {"input": 4_000_000, "output": 1_000_000},
                        }
                    }
                ]

        return _Resp()

    def post(self, path):
        self.posts.append(path)

        class _Resp:
            status_code = 200

        return _Resp()

    def close(self):
        pass


def _budget_unit_spec(tmp_path, grace_s: float, timeout_s: float = 30_000.0) -> SessionSpec:
    return SessionSpec(
        task_id="t-1",
        role="dev",
        prompt="p",
        cwd=tmp_path,
        timeout_s=timeout_s,
        token_budget=1_000_000,
        token_budget_mode="enforce",
        token_budget_grace_s=grace_s,
    )


def test_budget_grace_fires_on_wall_clock_when_monotonic_frozen(tmp_path, monkeypatch):
    """The #157 suspend signature on the budget grace: time.monotonic() stands
    still through a host suspend, so the monotonic grace alone would stretch
    the wrap-up window by the nap's length. The wall co-bound fires anyway."""
    adapter = make_adapter(tmp_path, policy=_budget_policy())
    clock = _install_clock(monkeypatch)
    (adapter.tasks_dir / "t-1").mkdir(parents=True)

    def advance():
        clock["wall"] += 11.0  # suspended host: wall counts on, monotonic frozen

    sess = _timeout_driven_session(adapter, advance)
    sess.client = _BigUsageClient()
    adapter.send_text = lambda handle, text: None  # nudge delivery not under test

    result = adapter.wait_for_completion(
        SessionHandle(task_id="t-1", native_id="ses_1"), _budget_unit_spec(tmp_path, grace_s=50.0)
    )

    assert result.status == "over_budget"
    assert result.budget_weighted == 5_000_000
    fired = [ln for ln in _lifecycle_lines(adapter) if ln["event"] == "over-budget-fired"]
    assert len(fired) == 1 and fired[0]["zero_grace"] is False


def test_budget_nudge_send_failure_still_arms_grace(tmp_path, monkeypatch):
    """A dead/hung server can reject the wrap-up nudge (the HTTP send raises);
    the trip must survive it and the grace still arm — the session is then
    scored via the normal paths (here: grace expiry → over_budget)."""
    adapter = make_adapter(tmp_path, policy=_budget_policy())
    clock = _install_clock(monkeypatch)
    (adapter.tasks_dir / "t-1").mkdir(parents=True)

    def advance():
        clock["mono"] += 11.0

    sess = _timeout_driven_session(adapter, advance)
    sess.client = _BigUsageClient()

    def boom(handle, text):
        raise RuntimeError("http send failed")

    adapter.send_text = boom

    result = adapter.wait_for_completion(
        SessionHandle(task_id="t-1", native_id="ses_1"), _budget_unit_spec(tmp_path, grace_s=50.0)
    )

    assert result.status == "over_budget"
    assert result.budget_weighted == 5_000_000


def test_budget_zero_grace_dead_server_takes_crash_path(tmp_path, monkeypatch):
    """A trip coinciding with server death must not discard a landed artifact
    just because grace is 0: the zero-grace exit checks the process first and
    routes a dead server through the crash path, which honors the artifact."""
    adapter = make_adapter(tmp_path, policy=_budget_policy())
    _install_clock(monkeypatch)
    (adapter.tasks_dir / "t-1").mkdir(parents=True)
    (adapter.tasks_dir / "t-1" / "result.json").write_text('{"ok": true}')

    sess = _timeout_driven_session(adapter, lambda: None)
    sess.client = _BigUsageClient()

    class _DeadProc:
        def poll(self):
            return 1

    sess.process = _DeadProc()

    result = adapter.wait_for_completion(
        SessionHandle(task_id="t-1", native_id="ses_1"), _budget_unit_spec(tmp_path, grace_s=0.0)
    )

    assert result.status == "completed"  # crash path honored the artifact
    assert result.result_json == {"ok": True}
    assert result.budget_weighted == 5_000_000


def test_budget_notify_failure_does_not_break_trip(tmp_path, monkeypatch):
    """observe-degrade: an ATTENTION append failure (disk full, perms) degrades
    to a missing notification; the trip and the over_budget verdict proceed."""
    from bmad_loop import gates as gates_mod

    adapter = make_adapter(tmp_path, policy=_budget_policy())
    _install_clock(monkeypatch)
    (adapter.tasks_dir / "t-1").mkdir(parents=True)

    def boom(*args, **kwargs):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(gates_mod, "notify", boom)
    sess = _timeout_driven_session(adapter, lambda: None)
    sess.client = _BigUsageClient()

    result = adapter.wait_for_completion(
        SessionHandle(task_id="t-1", native_id="ses_1"), _budget_unit_spec(tmp_path, grace_s=0.0)
    )

    assert result.status == "over_budget"
    assert result.budget_weighted == 5_000_000


# ---------------- timeout instrumentation + wall-clock co-bound (#157)
#
# Mirrors the generic-adapter coverage on the HTTP transport: the fire moment
# stamps the result and one timeout-fired line in session-lifecycle.jsonl, and
# a wall-clock co-bound fires through a frozen time.monotonic() (the macOS-sleep
# signature) but may never EXTEND the deadline. There is no watcher here — a
# tick is one sess.events.get(), so a fake queue advances the steerable clock
# each tick, exactly as _ScriptedWatcher.on_call does for the generic adapter.


def _install_clock(monkeypatch, mono=1000.0, wall=5000.0):
    clock = {"mono": mono, "wall": wall}

    class _Clock:
        monotonic = staticmethod(lambda: clock["mono"])
        time = staticmethod(lambda: clock["wall"])
        sleep = staticmethod(lambda *_: None)
        time_ns = staticmethod(lambda: 0)

    monkeypatch.setattr(opencode_http, "time", _Clock)
    return clock


def _timeout_driven_session(adapter, advance, task_id="t-1"):
    """Register a live session whose event queue never yields a frame but runs
    ``advance`` each tick, so only a clock crossing its deadline can end the
    wait. client=None makes _abort/_capture_usage no-ops."""

    class _AliveProc:
        def poll(self):
            return None

    class _TickingQueue:
        def get(self, timeout=None):
            advance()
            raise queue.Empty

        def empty(self):
            return True

    sess = _ServerSession(process=_AliveProc(), port=0, base_url="", password="", log_fh=None)
    sess.session_id = "ses_1"
    sess.client = None
    sess.events = _TickingQueue()
    adapter._sessions[task_id] = sess
    # There is no real server behind the fake process, so the atexit sweep must
    # not try to signal its (absent) pid or close its (None) log handle.
    adapter._teardown = lambda _sess: None
    return sess


def _lifecycle_lines(adapter, task_id="t-1"):
    path = adapter.tasks_dir / task_id / "session-lifecycle.jsonl"
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _timeout_spec(tmp_path, timeout_s=30.0) -> SessionSpec:
    return SessionSpec(task_id="t-1", role="dev", prompt="p", cwd=tmp_path, timeout_s=timeout_s)


def test_timeout_monotonic_expiry_is_instrumented(tmp_path, monkeypatch):
    """A plain monotonic expiry records WHEN and BY WHICH CLOCK the deadline was
    declared elapsed — result stamps, one timeout-fired lifecycle line, and a
    heartbeat.json topped up while the loop still ran."""
    adapter = make_adapter(tmp_path)
    clock = _install_clock(monkeypatch)
    (adapter.tasks_dir / "t-1").mkdir(parents=True)  # start_session makes it in production

    def advance():
        clock["mono"] += 11.0  # wall frozen: only the monotonic clock expires

    _timeout_driven_session(adapter, advance)
    result = adapter.wait_for_completion(
        SessionHandle(task_id="t-1", native_id="ses_1"), _timeout_spec(tmp_path)
    )

    assert result.status == "timeout"
    assert result.timeout_expired_clock == "monotonic"
    assert result.timeout_fired_at == 5000.0  # the fake wall clock at fire time
    (fired,) = [ln for ln in _lifecycle_lines(adapter) if ln["event"] == "timeout-fired"]
    assert fired["expired_clock"] == "monotonic"
    assert fired["timeout_s"] == 30.0
    assert fired["mono_remaining_s"] <= 0
    hb = json.loads((adapter.tasks_dir / "t-1" / "heartbeat.json").read_text(encoding="utf-8"))
    assert hb["remaining_s"] == 30.0 and hb["stall_armed"] is False


def test_timeout_fires_on_wall_clock_when_monotonic_frozen(tmp_path, monkeypatch):
    """The #157 suspend signature on the HTTP transport: time.monotonic() stands
    still through a host suspend, so the monotonic deadline alone would stretch
    the session by the nap's length. The wall-clock co-bound fires anyway, and
    the wall-only expiry (monotonic time still to spare) is stamped as the
    evidence."""
    adapter = make_adapter(tmp_path)
    clock = _install_clock(monkeypatch)

    def advance():
        clock["wall"] += 11.0  # suspended host: wall counts on, monotonic frozen

    _timeout_driven_session(adapter, advance)
    result = adapter.wait_for_completion(
        SessionHandle(task_id="t-1", native_id="ses_1"), _timeout_spec(tmp_path)
    )

    assert result.status == "timeout"
    assert result.timeout_expired_clock == "wall"
    (fired,) = [ln for ln in _lifecycle_lines(adapter) if ln["event"] == "timeout-fired"]
    assert fired["expired_clock"] == "wall"
    assert fired["mono_remaining_s"] == 30.0  # the frozen clock never advanced


def test_timeout_wall_clock_step_back_cannot_extend_deadline(tmp_path, monkeypatch):
    """The co-bound may only EXPIRE the deadline, never stretch it: a wall clock
    stepped backward (an NTP correction) leaves the monotonic expiry on its
    original schedule."""
    adapter = make_adapter(tmp_path)
    clock = _install_clock(monkeypatch)
    ticks = {"n": 0}

    def advance():
        ticks["n"] += 1
        clock["mono"] += 11.0
        clock["wall"] -= 3600.0  # NTP step-back: must change nothing

    _timeout_driven_session(adapter, advance)
    result = adapter.wait_for_completion(
        SessionHandle(task_id="t-1", native_id="ses_1"), _timeout_spec(tmp_path)
    )

    assert result.status == "timeout"
    assert result.timeout_expired_clock == "monotonic"
    assert ticks["n"] == 3  # same tick count as an untouched wall clock


def test_e2e_server_death_with_artifact_completes(tmp_path, fake_opencode):
    """Server death ≙ window death: the crash path vouches for a landed
    result.json (accept_result=True), so finished-then-died reads completed."""
    launcher, rec = fake_opencode
    adapter = make_adapter(tmp_path, binary=str(launcher))
    spec = make_spec(tmp_path, rec, "die-after-result")

    result = adapter.run(spec)

    assert result.status == "completed"
    assert result.result_json == {"ok": True, "workflow": "fake-triage"}
    assert_server_gone(rec)


def test_e2e_server_death_without_artifact_crashes(tmp_path, fake_opencode):
    launcher, rec = fake_opencode
    adapter = make_adapter(tmp_path, binary=str(launcher))
    spec = make_spec(tmp_path, rec, "die-no-result")

    result = adapter.run(spec)

    assert result.status == "crashed"
    assert result.result_json is None
    assert_server_gone(rec)


def test_e2e_sse_loss_degrades_to_poll_fallback(tmp_path, fake_opencode):
    """The stream closes after every connect, so session.idle is never
    deliverable; completion must arrive via GET /session/status + the
    message-level proof-of-work — in epoch-ms units (the regression test for
    the ns-vs-ms floor bug, which silently disables this whole path)."""
    launcher, rec = fake_opencode
    adapter = make_adapter(tmp_path, binary=str(launcher))
    spec = make_spec(tmp_path, rec, "sse-black-hole")

    result = adapter.run(spec)

    assert result.status == "completed"
    assert result.result_json == {"ok": True, "workflow": "fake-triage"}
    assert_server_gone(rec)


def test_e2e_spawn_retry_survives_one_early_death(tmp_path, fake_opencode):
    launcher, rec = fake_opencode
    adapter = make_adapter(tmp_path, binary=str(launcher))
    spec = make_spec(tmp_path, rec, "completed", extra_env={"FAKE_OPENCODE_START_FAILURES": "1"})

    result = adapter.run(spec)

    assert result.status == "completed"
    assert int((rec / "start-count").read_text(encoding="utf-8")) == 2
    assert_server_gone(rec)


def test_e2e_spawn_gives_up_after_attempts(tmp_path, fake_opencode):
    launcher, rec = fake_opencode
    adapter = make_adapter(tmp_path, binary=str(launcher))
    spec = make_spec(tmp_path, rec, "completed", extra_env={"FAKE_OPENCODE_START_FAILURES": "99"})

    with pytest.raises(OpencodeServerError, match="after 3 attempts"):
        adapter.run(spec)
    assert adapter._sessions == {}
    # every attempt appended to one log for the task
    assert (tmp_path / "run" / "logs" / "t-1.log").exists()


# --------------------------------------------- OpencodeDevAdapter (Phase 4)
#
# Dev/review sessions run the generic bmad-dev-auto skill, which writes NO
# result.json: the outcome lives in the terminal spec it leaves on disk,
# synthesized via devcontract by _DevSynthesisMixin — the same machinery
# GenericDevAdapter uses, composed over the HTTP transport.

_DONE_SPEC = (
    "---\nstatus: done\nbaseline_revision: abc123\n---\n\n"
    "## Auto Run Result\n\nStatus: done\nImplemented.\n"
)


def make_dev_adapter(
    tmp_path: Path, binary: str = "opencode", **kwargs
) -> tuple[OpencodeDevAdapter, Path]:
    impl = tmp_path / "impl"
    impl.mkdir(exist_ok=True)
    # project root == tmp_path so rebased(spec.cwd=tmp_path) is a no-op: these
    # sessions run in place, where cwd == the project root.
    paths = ProjectPaths(
        project=tmp_path,
        implementation_artifacts=impl,
        planning_artifacts=tmp_path / "plan",
    )
    adapter = OpencodeDevAdapter(
        run_dir=tmp_path / "run",
        policy=kwargs.pop("policy", _policy()),
        profile=get_profile("opencode"),
        binary=binary,
        paths=paths,
        **kwargs,
    )
    return _shrink_timing(adapter), impl


def make_dev_spec(
    tmp_path: Path,
    rec: Path,
    scenario: str,
    spec_path: Path,
    spec_text: str = _DONE_SPEC,
    story_key: str = "3-1",
    task_id: str = "3-1-dev-1",
    timeout_s: float = 30.0,
    extra_env: dict | None = None,
) -> SessionSpec:
    env = {
        "FAKE_OPENCODE_SCENARIO": scenario,
        "FAKE_OPENCODE_DIR": str(rec),
        # Deliberately NO FAKE_OPENCODE_RESULT_PATH: the dev skill writes no
        # result.json, and the dev adapter must never lean on one.
        "FAKE_OPENCODE_SPEC_PATH": str(spec_path),
        "FAKE_OPENCODE_SPEC_TEXT": spec_text,
        "BMAD_LOOP_STORY_KEY": story_key,
        **(extra_env or {}),
    }
    return SessionSpec(
        task_id=task_id,
        role="dev",
        prompt=f"/bmad-dev-auto {story_key}",
        cwd=tmp_path,
        env=env,
        timeout_s=timeout_s,
        stall_nudges_cap=6,
    )


def test_dev_knobs_configured(tmp_path):
    """_configure_dev_knobs over the HTTP knob names: no result-contract stop
    nudges (the skill writes no result.json), stall grace + wake budget from
    policy — same contract as GenericDevAdapter."""
    adapter, _ = make_dev_adapter(
        tmp_path, policy=_policy(dev_stall_grace_s=123, dev_stall_nudges=4)
    )
    assert adapter._stop_nudges == 0
    assert adapter._stall_grace_s == 123.0
    assert adapter._stall_nudges == 4


def test_dev_probe_alive_never_none(tmp_path):
    """The post-kill liveness seam: poll() on the retained Popen handle is
    always answerable (True/False), never the tri-state unknown a tmux probe
    can hit — and a task with no retained process owns nothing alive."""

    class _Proc:
        def __init__(self, rc):
            self._rc = rc

        def poll(self):
            return self._rc

    adapter, _ = make_dev_adapter(tmp_path)
    handle = SessionHandle(task_id="t", native_id="ses_x")
    assert adapter._probe_alive(handle) is False  # never spawned
    adapter._server_procs["t"] = _Proc(None)
    assert adapter._probe_alive(handle) is True  # kill silently failed: keep verdict
    adapter._server_procs["t"] = _Proc(0)
    assert adapter._probe_alive(handle) is False


def test_dev_result_json_ignores_result_file(tmp_path):
    """MRO pin: _DevSynthesisMixin's spec synthesis shadows the core adapter's
    result.json read-back — a stray result.json with no spec on disk is not a
    dev result."""
    adapter, _ = make_dev_adapter(tmp_path)
    task_dir = adapter.tasks_dir / "3-1-dev-1"
    task_dir.mkdir(parents=True)
    (task_dir / "result.json").write_text('{"ok": true}', encoding="utf-8")
    spec = SessionSpec(
        task_id="3-1-dev-1",
        role="dev",
        prompt="/bmad-dev-auto 3-1",
        cwd=tmp_path,
        env={"BMAD_LOOP_STORY_KEY": "3-1"},
    )
    handle = SessionHandle(task_id="3-1-dev-1", native_id="ses_x")
    assert adapter._result_json(handle, spec, wait=False) is None


def test_e2e_dev_synthesizes_terminal_spec(tmp_path, fake_opencode):
    launcher, rec = fake_opencode
    adapter, impl = make_dev_adapter(tmp_path, binary=str(launcher))
    spec = make_dev_spec(tmp_path, rec, "completed", impl / "spec-3-1-foo.md")

    result = adapter.run(spec)

    assert result.status == "completed"
    rj = result.result_json
    assert rj["workflow"] == "auto-dev"
    assert rj["status"] == "done"
    assert rj["baseline_commit"] == "abc123"  # mapped from baseline_revision
    assert rj["story_key"] == "3-1"
    assert rj["escalations"] == []
    assert "post_kill_reconciled" not in rj  # vouched by the idle, not rescued
    # transport parity with the classic path: usage, template, teardown
    assert adapter.read_usage(result) == TokenUsage(
        input_tokens=100, output_tokens=55, cache_read_tokens=7, cache_creation_tokens=3
    )
    assert prompt_texts(rec) == ["Use the bmad-dev-auto skill now: 3-1"]
    assert adapter._sessions == {}
    assert_server_gone(rec)


def test_e2e_dev_stories_mode_resolves_by_id(tmp_path, fake_opencode, monkeypatch):
    """Folder+id dispatch (BMAD_LOOP_SPEC_FOLDER): the story spec is resolved
    at its deterministic id-keyed path — never via the mtime scan."""
    launcher, rec = fake_opencode
    adapter, impl = make_dev_adapter(tmp_path, binary=str(launcher))

    def boom(*a, **k):
        raise AssertionError("stories mode must not call the mtime scan")

    monkeypatch.setattr(generic.devcontract, "find_result_artifact", boom)
    spec = make_dev_spec(
        tmp_path,
        rec,
        "completed",
        tmp_path / "epic" / "stories" / "1-foo.md",
        story_key="1",
        task_id="1-dev-1",
        extra_env={"BMAD_LOOP_SPEC_FOLDER": "epic"},
    )

    result = adapter.run(spec)

    assert result.status == "completed"
    assert result.result_json["status"] == "done"
    assert result.result_json["story_key"] == "1"
    assert_server_gone(rec)


def test_e2e_dev_post_kill_rescue(tmp_path, fake_opencode):
    """#61 over HTTP: the turn wrote its terminal spec but its idle was never
    seen (the fake stays busy forever), so the loop times out — a verdict
    reached under a live server, where the artifact is advisory (#48/#53).
    run()'s kill settles the liveness question; _post_kill_reconcile re-probes
    the now-dead server process and rescues the self-consistent done spec."""
    launcher, rec = fake_opencode
    adapter, impl = make_dev_adapter(tmp_path, binary=str(launcher))
    spec = make_dev_spec(tmp_path, rec, "busy-forever", impl / "spec-3-1-foo.md", timeout_s=1.5)

    result = adapter.run(spec)

    assert result.status == "completed"
    assert result.result_json["status"] == "done"
    assert result.result_json["post_kill_reconciled"] is True
    assert_server_gone(rec)


def test_e2e_dev_wait_loop_drives_observe_tick(tmp_path, fake_opencode, monkeypatch):
    """Cross-adapter parity (#276 M2): the OpenCode wait loop invokes _observe_tick
    from its heartbeat-throttled block, exactly as the generic adapter does. The
    first tick always fires (last_heartbeat is None), so any run drives the hook —
    with this session's handle and the very spec that was dispatched. OpencodeDev-
    Adapter shares _DevSynthesisMixin, so the observation seam is live over HTTP."""
    launcher, rec = fake_opencode
    adapter, impl = make_dev_adapter(tmp_path, binary=str(launcher))
    spec = make_dev_spec(tmp_path, rec, "completed", impl / "spec-3-1-foo.md")

    seen: list[tuple[str, bool]] = []
    monkeypatch.setattr(
        adapter, "_observe_tick", lambda handle, s: seen.append((handle.task_id, s is spec))
    )

    result = adapter.run(spec)

    assert result.status == "completed"
    assert seen and all(tid == "3-1-dev-1" and same for tid, same in seen)
    assert_server_gone(rec)
