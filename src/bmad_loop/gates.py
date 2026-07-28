"""Gate evaluation and human notification (desktop + ATTENTION file)."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from .policy import Policy

ATTENTION_FILE = "ATTENTION"

# The untrusted notification title/message (story keys, `str(e)` error tails) are
# handed to osascript/PowerShell through these environment variables rather than
# interpolated into the command text, so quotes/newlines/AppleScript-or-PowerShell
# metacharacters cannot break out of the string. notify-send takes them as argv,
# which is already injection-safe.
_TITLE_ENV = "BMAD_LOOP_NOTIFY_TITLE"
_MESSAGE_ENV = "BMAD_LOOP_NOTIFY_MESSAGE"

# WinRT ToastNotificationManager — the dependency-free toast path on Windows 10+
# (works under both pwsh and powershell.exe). Reads title/message from $env, so no
# PowerShell-string interpolation of user text.
_WIN_TOAST_PS = (
    "$ErrorActionPreference='Stop';"
    "[Windows.UI.Notifications.ToastNotificationManager,Windows.UI.Notifications,"
    "ContentType=WindowsRuntime]|Out-Null;"
    "$t=[Windows.UI.Notifications.ToastNotificationManager]::GetTemplateContent("
    "[Windows.UI.Notifications.ToastTemplateType]::ToastText02);"
    "$x=$t.GetElementsByTagName('text');"
    "$x.Item(0).AppendChild($t.CreateTextNode($env:BMAD_LOOP_NOTIFY_TITLE))|Out-Null;"
    "$x.Item(1).AppendChild($t.CreateTextNode($env:BMAD_LOOP_NOTIFY_MESSAGE))|Out-Null;"
    "$n=[Windows.UI.Notifications.ToastNotification]::new($t);"
    # Windows only shows a toast for a *registered* AppUserModelID; 'bmad-loop' has
    # no Start-menu shortcut carrying it, so that toast would be silently dropped.
    # Reuse Windows PowerShell's own default-registered AUMID instead (the toast is
    # attributed to "Windows PowerShell"). Raw strings: the AUMID's backslashes must
    # stay literal — `\v1.0` would otherwise be parsed as a vertical tab.
    r"[Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier("
    r"'{1AC14E77-02E7-4E5D-B744-2EB1AE5198B7}\WindowsPowerShell\v1.0\powershell.exe')"
    r".Show($n)"
)


def desktop_notifier_kind() -> str | None:
    """The desktop notifier available on THIS platform, or ``None``. Read-only —
    ``validate`` and the engine call it to decide whether ``notify.desktop`` can do
    anything here. Gated on ``sys.platform`` first (not ``which`` alone): PowerShell
    Core can exist on Linux/macOS, but only Windows should reach the toast path and
    Linux must keep picking ``notify-send``."""
    if sys.platform == "darwin":
        return "osascript" if shutil.which("osascript") else None
    if sys.platform == "win32":
        return "powershell" if (shutil.which("pwsh") or shutil.which("powershell")) else None
    return "notify-send" if shutil.which("notify-send") else None


def _notifier_argv(kind: str, title: str, message: str) -> tuple[list[str], dict[str, str]]:
    """``(argv, env-overrides)`` for ``kind``. osascript/powershell carry the
    untrusted text via env (never argv); notify-send takes it as argv."""
    if kind == "osascript":
        return (
            [
                "osascript",
                "-e",
                f'display notification (system attribute "{_MESSAGE_ENV}") '
                f'with title (system attribute "{_TITLE_ENV}")',
            ],
            {_TITLE_ENV: title, _MESSAGE_ENV: message},
        )
    if kind == "powershell":
        pwsh = shutil.which("pwsh") or shutil.which("powershell") or "powershell"
        return (
            [pwsh, "-NoProfile", "-NonInteractive", "-Command", _WIN_TOAST_PS],
            {_TITLE_ENV: title, _MESSAGE_ENV: message},
        )
    # `--` ends GLib option parsing: an untrusted title/message beginning with
    # `-`/`--` (e.g. a plugin veto reason of `--help`) is then taken as positional
    # SUMMARY/BODY text, not parsed as a notify-send option.
    return (["notify-send", "--app-name=bmad-loop", "--", title, message], {})


def notify(policy: Policy, run_dir: Path, title: str, message: str) -> None:
    """Best-effort human notification: append the ATTENTION file (if notify.file)
    and fire a native desktop notification (if notify.desktop). Never raises — a
    failing notifier must not crash the run. Headless CI cannot observe a real
    notification (no macOS job), so the native macOS/Windows paths are unit-tested
    at the command-construction level; verify visual delivery manually per OS."""
    if policy.notify.file:
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        try:
            with (run_dir / ATTENTION_FILE).open("a", encoding="utf-8") as f:
                f.write(f"[{stamp}] {title}: {message}\n")
        except OSError:
            # observe-degrade: an unwritable ATTENTION file is observability,
            # never a reason to break the loop (the _write_heartbeat doctrine,
            # already applied to this same call in the adapter budget guards).
            # Without it the "never raises" contract above was false for the
            # file half, and an unwritable run dir turned an advisory notice
            # into a run crash at every record-a-decision site. The journal
            # entry each caller writes first stays the durable record.
            pass
    # Native desktop notification per platform: osascript (macOS), a best-effort
    # WinRT PowerShell toast (Windows), notify-send (Linux). None → silently skip;
    # `validate` and run start warn separately when notify.desktop is inert here.
    if policy.notify.desktop:
        kind = desktop_notifier_kind()
        if kind:
            argv, env = _notifier_argv(kind, title, message)
            try:
                subprocess.run(
                    argv,
                    timeout=10,
                    capture_output=True,
                    env={**os.environ, **env} if env else None,
                )
            except (subprocess.SubprocessError, OSError, ValueError):
                # best-effort: a failing notifier must never crash the run. ValueError
                # covers an embedded NUL in the untrusted title/message, which reaches
                # argv (notify-send) or an env value (osascript/PowerShell) and makes
                # subprocess.run raise `ValueError: embedded null byte`.
                pass


def pause_at_epic_boundary(policy: Policy) -> bool:
    return policy.gates.mode in ("per-epic", "per-story-spec-approval")


def pause_after_spec(policy: Policy) -> bool:
    return policy.gates.mode == "per-story-spec-approval"
