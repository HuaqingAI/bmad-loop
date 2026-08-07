"""The hook relay script runs as a real subprocess, like Claude Code runs it."""

import json
import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parent.parent / "src" / "bmad_loop" / "data" / "bmad_loop_hook.py"


def run_hook(event: str, env: dict, payload) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), event],
        input=json.dumps(payload) if payload is not None else "",
        env={"PATH": "/usr/bin:/bin", **env},
        capture_output=True,
        text=True,
        timeout=10,
    )


def test_noop_without_env(tmp_path):
    proc = run_hook("Stop", {}, {"session_id": "s1"})
    assert proc.returncode == 0
    assert list(tmp_path.iterdir()) == []


def test_writes_event_file(tmp_path):
    env = {"BMAD_LOOP_RUN_DIR": str(tmp_path), "BMAD_LOOP_TASK_ID": "1-1-a-dev-1"}
    payload = {
        "session_id": "abc-123",
        "transcript_path": "/home/u/.claude/projects/x/abc-123.jsonl",
        "cwd": "/proj",
    }
    proc = run_hook("Stop", env, payload)
    assert proc.returncode == 0

    files = list((tmp_path / "events").glob("*.json"))
    assert len(files) == 1
    assert "1-1-a-dev-1" in files[0].name and "Stop" in files[0].name
    event = json.loads(files[0].read_text())
    assert event["event"] == "Stop"
    assert event["task_id"] == "1-1-a-dev-1"
    assert event["session_id"] == "abc-123"
    assert event["transcript_path"].endswith("abc-123.jsonl")
    assert not list((tmp_path / "events").glob("*.tmp"))


def test_conversation_id_fallback(tmp_path):
    """Cursor-style payloads carry conversation_id instead of session_id."""
    env = {"BMAD_LOOP_RUN_DIR": str(tmp_path), "BMAD_LOOP_TASK_ID": "t1"}
    proc = run_hook("Stop", env, {"conversation_id": "conv-9"})
    assert proc.returncode == 0
    files = list((tmp_path / "events").glob("*.json"))
    assert json.loads(files[0].read_text())["session_id"] == "conv-9"


def test_antigravity_payload(tmp_path):
    """agy payloads are protojson: conversationId, and workspacePaths for cwd."""
    env = {"BMAD_LOOP_RUN_DIR": str(tmp_path), "BMAD_LOOP_TASK_ID": "t1"}
    payload = {
        "conversationId": "agy-3",
        "transcriptPath": "/ws/.gemini/antigravity-cli/transcript.jsonl",
        "workspacePaths": ["/ws"],
        "terminationReason": "model_stop",
        "fullyIdle": True,
    }
    proc = run_hook("Stop", env, payload)
    assert proc.returncode == 0
    event = json.loads(next((tmp_path / "events").glob("*.json")).read_text())
    assert event["session_id"] == "agy-3"
    assert event["transcript_path"].endswith("transcript.jsonl")
    assert event["cwd"] == "/ws"


def test_workspace_paths_ignored_when_unusable(tmp_path):
    """An empty/odd workspacePaths must degrade to None, never IndexError."""
    env = {"BMAD_LOOP_RUN_DIR": str(tmp_path), "BMAD_LOOP_TASK_ID": "t1"}
    proc = run_hook("Stop", env, {"conversationId": "agy-4", "workspacePaths": []})
    assert proc.returncode == 0
    assert json.loads(next((tmp_path / "events").glob("*.json")).read_text())["cwd"] is None


def test_camelcase_payload(tmp_path):
    """Copilot payloads carry camelCase sessionId / transcriptPath."""
    env = {"BMAD_LOOP_RUN_DIR": str(tmp_path), "BMAD_LOOP_TASK_ID": "t1"}
    payload = {
        "sessionId": "cop-7",
        "transcriptPath": "/home/u/.copilot/session-state/cop-7/events.jsonl",
        "stopReason": "end_turn",
    }
    proc = run_hook("Stop", env, payload)
    assert proc.returncode == 0
    event = json.loads(next((tmp_path / "events").glob("*.json")).read_text())
    assert event["session_id"] == "cop-7"
    assert event["transcript_path"].endswith("events.jsonl")


def test_tolerates_garbage_stdin(tmp_path):
    env = {"BMAD_LOOP_RUN_DIR": str(tmp_path), "BMAD_LOOP_TASK_ID": "t1"}
    proc = run_hook("SessionEnd", env, None)  # empty stdin
    assert proc.returncode == 0
    files = list((tmp_path / "events").glob("*.json"))
    assert len(files) == 1
    assert json.loads(files[0].read_text())["session_id"] is None


@pytest.mark.skipif(os.name == "nt", reason="POSIX symlinks")
def test_symlinked_events_dir_writes_nothing_and_exits_zero(tmp_path):
    """#461 (Low): a driven session can write inside the run dir, so it can plant
    `events/` as a symlink and redirect the orchestrator's control-plane event
    stream — swallowing the Stop signal and stalling the run to timeout. The relay
    must refuse the link and degrade to a no-op, never write through it."""
    target = tmp_path / "attacker"
    target.mkdir()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "events").symlink_to(target, target_is_directory=True)

    env = {"BMAD_LOOP_RUN_DIR": str(run_dir), "BMAD_LOOP_TASK_ID": "t1"}
    proc = run_hook("Stop", env, {"session_id": "s1"})

    assert proc.returncode == 0
    assert list(target.iterdir()) == []
    assert list((run_dir / "events").iterdir()) == []


@pytest.mark.skipif(os.name == "nt", reason="POSIX file modes")
def test_event_file_mode_is_0600(tmp_path):
    """Event files carry the orchestrator's control plane; nothing but the
    operator running the loop needs to read them (narrowed from the umask-derived
    0644 a plain `open()` produced)."""
    env = {"BMAD_LOOP_RUN_DIR": str(tmp_path), "BMAD_LOOP_TASK_ID": "t1"}
    proc = run_hook("Stop", env, {"session_id": "s1"})
    assert proc.returncode == 0
    written = next((tmp_path / "events").glob("*.json"))
    assert stat.S_IMODE(written.stat().st_mode) == 0o600


def test_installed_copy_matches_source(tmp_path):
    from bmad_loop.install import install_into

    install_into(tmp_path)
    installed = (tmp_path / ".bmad-loop" / "bmad_loop_hook.py").read_text()
    assert installed == SCRIPT.read_text()
