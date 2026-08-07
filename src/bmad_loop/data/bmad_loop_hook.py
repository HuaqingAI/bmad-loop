#!/usr/bin/env python3
"""Coding-CLI hook relay for bmad-loop. Stdlib only.

Each CLI's hook config registers this script under its native event names
(Claude/Codex: SessionStart/Stop/..., Gemini: AfterAgent for Stop, Copilot:
agentStop for Stop) but always passes the CANONICAL event name as argv[1] — the
orchestrator only ever sees canonical events. Payload keys vary too: snake_case
(claude/codex), conversation_id (cursor), or camelCase (copilot's sessionId/
transcriptPath, agy's conversationId/transcriptPath — protojson encoding); the
field extraction below tries each. agy alone carries no cwd, sending the
workspacePaths list instead. Reads the hook payload
from stdin and writes one event file
into the orchestrator's run directory. No-ops (exit 0) unless the session was
spawned by bmad-loop (detected via env vars set on the tmux window), so
normal interactive sessions are unaffected.
"""

import json
import os
import sys
import time


def _first_workspace(payload):
    paths = payload.get("workspacePaths")
    if isinstance(paths, list) and paths and isinstance(paths[0], str):
        return paths[0]
    return None


def _write_event(events_dir, name, event):
    """Write one event file into `events_dir`, refusing to follow a symlink.

    The events dir is the orchestrator's control plane. A driven session has
    write access to the project, so it could plant `<run_dir>/events` as a
    symlink and redirect (or swallow) the completion signal — stalling the run
    to `session_timeout_min` instead of completing. `os.makedirs(exist_ok=True)`
    `isdir()`-checks THROUGH a link, so the refusal has to come before it; that
    refusal works on every platform and is the whole defense on Windows, where
    O_NOFOLLOW/O_DIRECTORY do not exist and `os.supports_dir_fd` is empty.

    Where the platform has them, the create+replace is additionally anchored to
    a dir_fd opened O_NOFOLLOW, which closes the window between the check and
    the write. Mode is 0o600 (narrowed from the umask-derived 0644 an ordinary
    `open()` produced): only the operator running the loop reads these.

    Raises OSError on any refusal or failure; the caller degrades to a no-op.
    """
    if os.path.islink(events_dir):
        raise OSError(f"refusing to write events into a symlinked directory: {events_dir}")
    os.makedirs(events_dir, exist_ok=True)
    data = json.dumps(event).encode("utf-8")
    tmp = name + ".tmp"
    o_nofollow = getattr(os, "O_NOFOLLOW", 0)
    o_directory = getattr(os, "O_DIRECTORY", 0)
    # O_BINARY is a no-op flag on POSIX; on Windows it stops the fd from
    # newline-translating what os.write() puts through it.
    create = os.O_WRONLY | os.O_CREAT | os.O_EXCL | o_nofollow | getattr(os, "O_BINARY", 0)
    # Probe os.rename, not os.replace: CPython omits os.replace from
    # supports_dir_fd on Linux even though it accepts src_dir_fd/dst_dir_fd, so
    # probing it would leave this whole branch dead everywhere. This branch is
    # POSIX-only by construction, and there rename(2) IS the atomic-replace
    # primitive os.replace wraps — probe the function actually called.
    if o_nofollow and o_directory and {os.open, os.rename} <= os.supports_dir_fd:
        dir_fd = os.open(events_dir, os.O_RDONLY | o_directory | o_nofollow)
        try:
            fd = os.open(tmp, create, 0o600, dir_fd=dir_fd)
            try:
                os.write(fd, data)
            finally:
                os.close(fd)
            os.rename(tmp, name, src_dir_fd=dir_fd, dst_dir_fd=dir_fd)
        finally:
            os.close(dir_fd)
        return
    tmp_path = os.path.join(events_dir, tmp)
    fd = os.open(tmp_path, create, 0o600)
    try:
        os.write(fd, data)
    finally:
        os.close(fd)
    os.replace(tmp_path, os.path.join(events_dir, name))


def main() -> int:
    run_dir = os.environ.get("BMAD_LOOP_RUN_DIR")
    task_id = os.environ.get("BMAD_LOOP_TASK_ID")
    if not run_dir or not task_id:
        return 0
    event_name = sys.argv[1] if len(sys.argv) > 1 else "Unknown"
    try:
        payload = json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError):
        payload = {}
    if not isinstance(payload, dict):
        payload = {}

    ts = time.time_ns()
    event = {
        "ts": ts,
        "event": event_name,
        "task_id": task_id,
        # Payload keys vary by CLI: snake_case (claude/codex), conversation_id
        # (cursor), or camelCase (copilot's sessionId/transcriptPath, agy's
        # conversationId). Try each.
        "session_id": (
            payload.get("session_id")
            or payload.get("conversation_id")
            or payload.get("sessionId")
            or payload.get("conversationId")
        ),
        "transcript_path": payload.get("transcript_path") or payload.get("transcriptPath"),
        # agy sends no cwd — it sends workspacePaths, a list of workspace roots.
        "cwd": payload.get("cwd") or _first_workspace(payload),
    }
    try:
        _write_event(os.path.join(run_dir, "events"), f"{ts}-{task_id}-{event_name}.json", event)
    except OSError:
        # A hostile or broken events dir must degrade to the orchestrator's
        # normal session_timeout_min path, never surface as a hook failure that
        # fails the CLI window (mirrors bmad_loop_probe_hook.py's write wrap).
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
