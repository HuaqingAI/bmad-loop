"""gates.notify — the ATTENTION file sink and the cross-platform desktop
notifier dispatch added for #231 (native macOS/Windows notifications, plus the
untrusted-text-via-env invariant that keeps user text out of the command string)."""

from __future__ import annotations

import subprocess

import pytest

from bmad_loop import gates
from bmad_loop.policy import NotifyPolicy, Policy


def _policy(*, desktop: bool, file: bool) -> Policy:
    return Policy(notify=NotifyPolicy(desktop=desktop, file=file))


def _capture_run(monkeypatch):
    """Record every (argv, kwargs) `gates.notify` hands to `subprocess.run`."""
    calls: list[tuple[list[str], dict]] = []

    def fake_run(argv, **kwargs):
        calls.append((argv, kwargs))
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(gates.subprocess, "run", fake_run)
    return calls


# ------------------------------------------------------------- file sink


def test_notify_file_appends_attention_line(tmp_path):
    gates.notify(_policy(desktop=False, file=True), tmp_path, "worktree-open-failed", "mount lost")
    line = (tmp_path / gates.ATTENTION_FILE).read_text(encoding="utf-8")
    assert line.startswith("[")  # timestamp stamp
    assert "worktree-open-failed: mount lost" in line


def test_notify_file_swallows_an_unwritable_attention_path(tmp_path, monkeypatch):
    """The "never raises" contract covers the file sink too. An unwritable
    ATTENTION path is observability degrading, not a reason to break the run —
    every engine caller notifies on a path where a raise would crash the run or
    unwind a decision it has already journaled.

    A directory at the ATTENTION path is the portable way to make the append
    fail: IsADirectoryError on POSIX, PermissionError on Windows, both OSError.
    chmod would not do it under root (CI) or on Windows.
    """
    (tmp_path / gates.ATTENTION_FILE).mkdir()
    # the desktop half must not be what absorbs this
    monkeypatch.setattr(gates, "desktop_notifier_kind", lambda: None)

    gates.notify(_policy(desktop=False, file=True), tmp_path, "story deferred: 1-1-a", "verify")


# ------------------------------------------------ desktop_notifier_kind()


@pytest.mark.parametrize(
    ("platform", "present", "expected"),
    [
        ("darwin", "osascript", "osascript"),
        ("win32", "powershell", "powershell"),
        ("linux", "notify-send", "notify-send"),
    ],
)
def test_desktop_notifier_kind_per_platform(monkeypatch, platform, present, expected):
    monkeypatch.setattr(gates.sys, "platform", platform)
    monkeypatch.setattr(
        gates.shutil, "which", lambda cmd: f"/usr/bin/{cmd}" if cmd == present else None
    )
    assert gates.desktop_notifier_kind() == expected


@pytest.mark.parametrize("platform", ["darwin", "win32", "linux"])
def test_desktop_notifier_kind_none_when_tool_absent(monkeypatch, platform):
    monkeypatch.setattr(gates.sys, "platform", platform)
    monkeypatch.setattr(gates.shutil, "which", lambda _cmd: None)
    assert gates.desktop_notifier_kind() is None


def test_desktop_notifier_kind_win32_accepts_pwsh(monkeypatch):
    """PowerShell Core satisfies the Windows branch even without Windows PowerShell."""
    monkeypatch.setattr(gates.sys, "platform", "win32")
    monkeypatch.setattr(
        gates.shutil, "which", lambda cmd: "/usr/bin/pwsh" if cmd == "pwsh" else None
    )
    assert gates.desktop_notifier_kind() == "powershell"


def test_desktop_notifier_kind_linux_ignores_pwsh(monkeypatch):
    """`sys.platform` gates first: pwsh on Linux must not divert from notify-send."""
    monkeypatch.setattr(gates.sys, "platform", "linux")
    monkeypatch.setattr(
        gates.shutil, "which", lambda cmd: "/usr/bin/pwsh" if cmd == "pwsh" else None
    )
    assert gates.desktop_notifier_kind() is None  # notify-send absent → nothing, not pwsh


# --------------------------------------------------- desktop dispatch


def test_notify_macos_runs_osascript_via_env(monkeypatch, tmp_path):
    monkeypatch.setattr(gates.sys, "platform", "darwin")
    monkeypatch.setattr(
        gates.shutil, "which", lambda cmd: "/usr/bin/osascript" if cmd == "osascript" else None
    )
    calls = _capture_run(monkeypatch)

    message = 'he said "done"\nnow'  # a quote+newline that would break AppleScript if interpolated
    title = "story deferred: 1-2-a"
    gates.notify(_policy(desktop=True, file=False), tmp_path, title, message)

    assert len(calls) == 1
    argv, kwargs = calls[0]
    assert argv[0] == "osascript"
    # untrusted text travels through env, never interpolated into the command string
    assert kwargs["env"][gates._TITLE_ENV] == title
    assert kwargs["env"][gates._MESSAGE_ENV] == message
    assert not any(message in part for part in argv)
    assert not any("1-2-a" in part for part in argv)


def test_notify_windows_runs_powershell(monkeypatch, tmp_path):
    monkeypatch.setattr(gates.sys, "platform", "win32")
    monkeypatch.setattr(
        gates.shutil, "which", lambda cmd: "C:\\powershell.exe" if cmd == "powershell" else None
    )
    calls = _capture_run(monkeypatch)

    # both title and message carry PowerShell/shell-sensitive text that must never
    # reach argv — they travel through env, and the -Command script reads $env:.
    title = 'escalation `$(rm)`;"& evil'
    message = 'toast $(bad) "body"'
    gates.notify(_policy(desktop=True, file=False), tmp_path, title, message)

    assert len(calls) == 1
    argv, kwargs = calls[0]
    assert argv[0].lower().endswith(("powershell.exe", "pwsh"))
    assert "-Command" in argv
    assert kwargs["env"][gates._TITLE_ENV] == title
    assert kwargs["env"][gates._MESSAGE_ENV] == message
    # neither the title nor the message is interpolated into the command string
    assert not any(title in part for part in argv)
    assert not any(message in part for part in argv)
    # the toast is delivered under a registered AUMID (Windows drops unregistered ids)
    assert any("WindowsPowerShell" in part for part in argv)


def test_notify_linux_runs_notify_send(monkeypatch, tmp_path):
    monkeypatch.setattr(gates.sys, "platform", "linux")
    monkeypatch.setattr(
        gates.shutil, "which", lambda cmd: "/usr/bin/notify-send" if cmd == "notify-send" else None
    )
    calls = _capture_run(monkeypatch)

    # option-shaped title/message must not be parsed as notify-send options: the `--`
    # terminator forces GLib to treat them as positional SUMMARY/BODY text.
    gates.notify(_policy(desktop=True, file=False), tmp_path, "--title", "--help")

    assert len(calls) == 1
    argv, kwargs = calls[0]
    # `--` sits before the untrusted text, so a leading-dash payload stays positional
    assert argv == ["notify-send", "--app-name=bmad-loop", "--", "--title", "--help"]
    assert kwargs["env"] is None  # no env override for the notify-send path


def test_notify_desktop_swallows_value_error(monkeypatch, tmp_path):
    """An embedded NUL in the untrusted text makes subprocess.run raise ValueError
    (not a SubprocessError, which is not a ValueError subclass); the best-effort
    boundary must still swallow it rather than crash the run."""
    monkeypatch.setattr(gates.sys, "platform", "linux")
    monkeypatch.setattr(gates.shutil, "which", lambda _cmd: "/usr/bin/notify-send")

    def boom(*_a, **_k):
        raise ValueError("embedded null byte")

    monkeypatch.setattr(gates.subprocess, "run", boom)
    # must not propagate
    gates.notify(_policy(desktop=True, file=False), tmp_path, "title", "mid\x00nul")


def test_notify_windows_dispatches_via_pwsh_only(monkeypatch, tmp_path):
    """PowerShell Core alone (no powershell.exe) still dispatches the toast — the
    dispatch path must select whichever of pwsh/powershell shutil.which resolves."""
    monkeypatch.setattr(gates.sys, "platform", "win32")
    monkeypatch.setattr(
        gates.shutil, "which", lambda cmd: "/usr/bin/pwsh" if cmd == "pwsh" else None
    )
    calls = _capture_run(monkeypatch)

    gates.notify(_policy(desktop=True, file=False), tmp_path, "title", "message")

    assert len(calls) == 1
    argv, kwargs = calls[0]
    assert argv[0].endswith("pwsh")  # not powershell.exe
    assert "-Command" in argv
    assert kwargs["env"][gates._TITLE_ENV] == "title"


def test_notify_desktop_noop_when_no_notifier(monkeypatch, tmp_path):
    monkeypatch.setattr(gates.sys, "platform", "darwin")
    monkeypatch.setattr(gates.shutil, "which", lambda _cmd: None)
    calls = _capture_run(monkeypatch)

    gates.notify(_policy(desktop=True, file=False), tmp_path, "title", "message")

    assert calls == []  # no notifier resolved → subprocess.run never called


def test_notify_desktop_swallows_errors(monkeypatch, tmp_path):
    monkeypatch.setattr(gates.sys, "platform", "linux")
    monkeypatch.setattr(gates.shutil, "which", lambda _cmd: "/usr/bin/notify-send")

    def boom(*_a, **_k):
        raise OSError("no dbus")

    monkeypatch.setattr(gates.subprocess, "run", boom)
    # best-effort: a failing notifier must not propagate out of notify()
    gates.notify(_policy(desktop=True, file=False), tmp_path, "title", "message")
