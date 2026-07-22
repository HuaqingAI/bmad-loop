"""In-process Unity engine plugin — the proof the framework carries an engine.

Everything that used to be bespoke ``Engine._engine_*`` code now lives here, on
top of the generic plugin framework:

  * the **readiness gate** (``on_pre_ready_gate``) blocks until the Editor + MCP
    report ready before any session runs, in both editor modes;
  * **per_worktree setup/teardown** (``on_pre_worktree_setup`` /
    ``on_pre_worktree_teardown``) launch and reap a managed Editor per worktree;
  * a **rollback quiesce** (``on_pre_rollback`` / ``on_post_rollback``) saves +
    closes open scenes before ``verify.safe_rollback`` runs ``git reset --hard``,
    then refreshes assets after — so the reset can't rewrite a tracked ``.unity``
    file under a shared Editor and raise a run-freezing modal dialog. Best effort:
    a wedged Editor is skipped (fast) and the rollback always proceeds;
  * a failure **vetoes (defers)** the unit through the bus — the engine's generic
    ``_vetoed`` routing turns that into a deferral + notification, with no
    Unity-specific branch in the loop;
  * **MCP agent routing** reads ``ctx.agents`` (the dev + review CLIs in the
    worktree) so every agent's MCP config is pointed at the worktree's Editor;
  * **prompt-fact injection** (``on_pre_session``) appends the Unity scene-save
    discipline (from the shipped ``unity_facts.md``) to every dev/review session
    prompt, so the agent saves dirty scenes at the boundaries that would otherwise
    raise the run-freezing modals — a plugin-only change, no engine edit;
  * a **detect-only dialog probe** (``on_pre_run`` / ``on_post_run`` /
    ``on_pre_worktree_setup``) launches a detached ``unity_dialog_probe.py`` that
    watches (via xdotool, X11 only) for the modal dialogs and *reports* them
    (JSONL + ATTENTION + notify-send) — it never clicks or keys anything. It is the
    last-resort observability net for a modal the guard + facts didn't prevent;
  * **editor_mode↔scm.isolation coupling** is validated in ``validate`` at startup
    (it moved out of core ``policy.loads`` — a flat per-key schema can't express a
    cross-section coupling).

The helper scripts (``unity_ready.py`` / ``unity_setup.py`` / ``unity_teardown.py``)
are unchanged; they read the same ``BMAD_LOOP_*`` environment this module injects,
so the env contract a Unity operator relies on is identical to the engine layer it
replaces.

A newer helper, ``unity_seed_assets.py``, seeds a **scene auto-save guard** into
the project (see ``unity_assets/``). Unity-MCP GameObject tools mark scenes dirty
but never save, so a shared editor's scene sits chronically dirty — the state that
raises the two run-stalling modal dialogs (scene-changed-on-disk reload, and
save-before-quit) that freeze the MCP dispatch loop. The guard is seeded *before*
the Editor's first import in both modes (at ``pre_worktree_setup`` in per_worktree
mode, ahead of ``unity_setup.py``; at ``pre_ready_gate`` in shared mode, where
``unity_setup.py`` never runs). Seeding is best-effort — a failure is logged, never
vetoes — and happens pre-baseline, so ``verify.safe_rollback`` never reclaims it.
The seeded guard is committed into the consumer project by story-finalize's
``git add -A`` — intended.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from bmad_loop.plugins.model import Plugin, PluginError
from bmad_loop.process_host import get_process_host

# editor modes this plugin supports; the operator picks one via [plugins.unity].
EDITOR_MODES = ("shared", "per_worktree")
# the detached detect-only modal-dialog probe + its pid-file reap handle.
_DIALOG_PROBE_SCRIPT = "unity_dialog_probe.py"
_DIALOG_PROBE_PID_FILE = "unity-dialog-probe.pid"
# best-effort teardown bound so a hung Editor-quit can't stall the loop for the
# full readiness budget on every unit (was Engine._ENGINE_TEARDOWN_TIMEOUT).
_TEARDOWN_TIMEOUT = 120
# scene-guard seeding is a handful of small file copies; bound it small so a wedged
# filesystem can't eat into the readiness budget (it never vetoes either way).
_SEED_TIMEOUT = 60
# fallback overall budget for the rollback quiesce helper if quiesce_timeout_sec is
# unreadable. The helper's own per-call timeouts are the primary no-deadlock guard;
# this is the outer kill so a wedged Editor can never stall a rollback either way.
_QUIESCE_TIMEOUT = 60


class UnityPlugin(Plugin):
    """Trust-gated in-process plugin (loads only when "unity" is in
    ``[plugins] enabled``). Its lifecycle hooks gate and manage the Editor."""

    # Whether to let the post-run cleanup reclaim the IvanMurzak MCP server's
    # /tmp scratch + log. Mirrors [cleanup] clean_tmp, captured at validate()
    # since plugin hooks don't otherwise see the run policy. Default True.
    _clean_tmp = True

    def _editor_mode(self) -> str:
        return str(self.settings.get("editor_mode", "shared"))

    # ----------------------------------------------------------- validation

    def validate(self, policy: Any) -> None:
        """editor_mode and scm.isolation are coupled: a live Editor MCP must act
        on the same folder the agent edits. shared = the agent's warm Editor on
        the checkout in place (no worktree); per_worktree = one Editor per
        isolated worktree."""
        self._clean_tmp = bool(getattr(getattr(policy, "cleanup", None), "clean_tmp", True))
        mode = self._editor_mode()
        if mode not in EDITOR_MODES:
            raise PluginError(
                f"plugin 'unity': editor_mode must be one of {sorted(EDITOR_MODES)}: got {mode!r}"
            )
        isolation = getattr(getattr(policy, "scm", None), "isolation", "none")
        if mode == "shared" and isolation != "none":
            raise PluginError(
                "plugin 'unity': editor_mode = 'shared' requires scm.isolation = 'none' "
                f"(the agent works in place on the live Editor's checkout); got "
                f"scm.isolation = {isolation!r}"
            )
        if mode == "per_worktree" and isolation != "worktree":
            raise PluginError(
                "plugin 'unity': editor_mode = 'per_worktree' requires scm.isolation = 'worktree'; "
                f"got scm.isolation = {isolation!r}"
            )

    # --------------------------------------------------------------- hooks

    def on_pre_ready_gate(self, ctx) -> None:
        """Block until the Editor + MCP report ready before a unit runs (both
        modes). A non-zero exit vetoes (defers) the unit.

        In shared mode ``unity_setup.py`` never runs (isolation=none has no
        per_worktree setup stage), so this is where the scene guard is seeded —
        before the readiness gate blocks on the live Editor."""
        if self._editor_mode() == "shared":
            self._seed_scene_guard(ctx)
        rc, tail = self._run_script("unity_ready.py", ctx, timeout=self._ready_timeout())
        if rc != 0:
            ctx.veto("defer", f"Unity Editor not ready (rc={rc}): {tail}".rstrip())

    def on_pre_worktree_setup(self, ctx) -> None:
        """per_worktree: make the fresh worktree a usable Unity project + launch
        its managed Editor before the agent runs. A failure defers the unit."""
        if self._editor_mode() != "per_worktree":
            return
        # Seed the scene guard BEFORE unity_setup launches the Editor, so the
        # Editor's very first import already sees it. Best effort — never vetoes.
        self._seed_scene_guard(ctx)
        # Launch this unit's detect-only dialog probe (scoped to the worktree path
        # so teardown's argv scan can find it) before the Editor comes up, so it is
        # already watching. Best effort — a launch failure never blocks setup.
        self._start_dialog_probe(ctx, ctx.worktree or "")
        rc, tail = self._run_script("unity_setup.py", ctx, timeout=self._ready_timeout())
        if rc != 0:
            ctx.veto("defer", f"Unity worktree setup failed (rc={rc}): {tail}".rstrip())

    def on_pre_worktree_teardown(self, ctx) -> None:
        """per_worktree: quit the unit's managed Editor + undo its setup. Best
        effort — observe-only (teardown stages forbid veto); a failure is left to
        the bus journal, the unit's outcome stands."""
        if self._editor_mode() != "per_worktree":
            return
        self._run_script("unity_teardown.py", ctx, timeout=_TEARDOWN_TIMEOUT)

    def on_pre_rollback(self, ctx) -> None:
        """Quiesce the Editor before ``verify.safe_rollback`` runs ``git reset
        --hard``: save every open scene, then open an empty untitled scene so no
        tracked ``.unity`` file is open when the reset rewrites it. Without this a
        shared Editor holding a dirty scene raises a modal "scene changed on disk —
        Reload/Cancel" dialog that freezes ``EditorApplication.update`` and every
        Unity-MCP dispatch. Best effort — a wedged Editor is detected fast and
        skipped, and the rc is ignored: a failed quiesce must never block the
        rollback."""
        self._quiesce("pre", ctx)

    def on_post_rollback(self, ctx) -> None:
        """After the reset rewrote the tracked tree, tell the Editor to re-import
        so it sees the reverted assets rather than its stale in-memory copies. Best
        effort — same never-veto contract as ``on_pre_rollback``."""
        self._quiesce("post", ctx)

    def on_pre_session(self, ctx) -> None:
        """Append the Unity scene-save discipline to every dev/review session prompt
        so the agent saves dirty scenes at the boundaries that raise the run-freezing
        modals. Only the whitelisted ``proposed_prompt`` is mutated, and only when
        it is non-empty — a fast-path emit with no prompt is left untouched. The
        facts text ships as ``unity_facts.md`` next to the helper scripts; a
        project-local plugin-dir copy overrides it (the loader points ``scripts_dir``
        at the override)."""
        if not ctx.proposed_prompt:
            return
        facts = self._session_facts()
        if facts:
            ctx.proposed_prompt = ctx.proposed_prompt + "\n\n" + facts

    def on_pre_run(self, ctx) -> None:
        """Run start: sweep a stale dialog probe recorded in this run dir (a resume
        re-enters with the same run dir + a re-stamped engine pid, so a prior probe
        would otherwise linger until its next self-reap poll), then in shared mode
        launch a fresh detect-only probe that shadows the live Editor for the whole
        run. per_worktree launches its probe per unit at ``pre_worktree_setup``."""
        self._reap_dialog_probe(ctx)
        if self._editor_mode() == "shared":
            self._start_dialog_probe(ctx, ctx.repo_root or "")

    def on_post_run(self, ctx) -> None:
        """Run finished cleanly: reap the shared-mode dialog probe (per_worktree
        probes are reaped at worktree teardown; the reap is idempotent + mode-
        agnostic so it is safe here in both modes), then reclaim the IvanMurzak MCP
        server's /tmp scratch (downloaded server zips) and truncate its unbounded
        editor log. Best effort — observe-only, runs once per run in both editor
        modes, after the loop so it never races an in-flight setup-mcp download.
        CoplayDev uses a shared server with no per-project /tmp download, so the
        cleanup is skipped."""
        self._reap_dialog_probe(ctx)
        if not self._clean_tmp:
            return
        if str(self.settings.get("mcp", "ivanmurzak")) != "ivanmurzak":
            return
        self._run_script("unity_cleanup.py", ctx, timeout=_TEARDOWN_TIMEOUT)

    # -------------------------------------------------------------- helpers

    def _ready_timeout(self) -> int:
        try:
            return max(1, int(self.settings.get("ready_timeout_sec", 600)))
        except (TypeError, ValueError):
            return 600

    def _install_scene_guard(self) -> bool:
        return bool(self.settings.get("install_scene_guard", True))

    def _quiesce_on_rollback(self) -> bool:
        return bool(self.settings.get("quiesce_on_rollback", True))

    def _quiesce_timeout(self) -> int:
        try:
            return max(1, int(self.settings.get("quiesce_timeout_sec", _QUIESCE_TIMEOUT)))
        except (TypeError, ValueError):
            return _QUIESCE_TIMEOUT

    def _quiesce(self, phase: str, ctx) -> None:
        """Run the rollback quiesce helper for ``phase`` ("pre" | "post"). Skipped
        when disabled or when the MCP isn't the IvanMurzak CLI (the only server the
        helper drives). Best effort: the rc is ignored (the helper's exit codes are
        advisory) and a failure never vetoes — this is observe-only, called from the
        engine's ``pre_rollback`` / ``post_rollback`` stages which forbid a veto."""
        if not self._quiesce_on_rollback():
            return
        if str(self.settings.get("mcp", "ivanmurzak")) != "ivanmurzak":
            return
        self._run_script(
            "unity_quiesce.py",
            ctx,
            timeout=self._quiesce_timeout(),
            extra_env={"BMAD_LOOP_QUIESCE_PHASE": phase},
        )

    def _seed_scene_guard(self, ctx) -> None:
        """Seed the scene auto-save guard into the project so a chronically-dirty
        scene never raises the run-stalling modal dialogs (which freeze the MCP
        dispatch loop). Skipped when ``install_scene_guard`` is off. Best effort:
        the seeder never mutates tracked history and a failure is logged, never
        vetoes — the guard is a convenience, and its absence must not defer a
        unit."""
        if not self._install_scene_guard():
            return
        rc, tail = self._run_script("unity_seed_assets.py", ctx, timeout=_SEED_TIMEOUT)
        if rc != 0:
            sys.stderr.write(f"unity: scene-guard seeding failed (rc={rc}): {tail}".rstrip() + "\n")

    def _session_facts(self) -> str:
        """The Unity operating facts appended to every session prompt, read once from
        the shipped (or project-overridden) ``unity_facts.md``. Cached on the
        instance; a missing/unreadable file degrades to no injection."""
        cached = getattr(self, "_facts_text", None)
        if cached is not None:
            return cached
        try:
            text = (
                (Path(self.manifest.scripts_dir) / "unity_facts.md")
                .read_text(encoding="utf-8")
                .strip()
            )
        except OSError:
            text = ""
        self._facts_text = text
        return text

    # ----------------------------------------------------- detect-only dialog probe

    def _dialog_probe(self) -> bool:
        return bool(self.settings.get("dialog_probe", False))

    def _dialog_probe_interval(self) -> int:
        try:
            return max(1, int(self.settings.get("dialog_probe_interval_sec", 5)))
        except (TypeError, ValueError):
            return 5

    def _dialog_probe_notify(self) -> bool:
        return bool(self.settings.get("dialog_probe_notify", True))

    def _probe_pid_path(self, ctx) -> Path | None:
        return Path(ctx.run_dir) / _DIALOG_PROBE_PID_FILE if ctx.run_dir else None

    def _start_dialog_probe(self, ctx, scan_path: str) -> None:
        """Launch the detached detect-only dialog probe (only when enabled). Reaps
        any probe recorded in this run dir first so a re-entry never leaves two
        running. The probe self-reaps when the engine pid dies, so a leak is bounded
        regardless. ``scan_path`` (the worktree in per_worktree mode, the repo root
        in shared) is passed in argv so teardown's argv scan can find the process —
        its exe basename is ``python``, invisible to the Editor sweep. Best effort:
        a launch failure is logged, never blocks the caller."""
        if not self._dialog_probe():
            return
        self._reap_dialog_probe(ctx)  # never run two
        script = Path(self.manifest.scripts_dir) / _DIALOG_PROBE_SCRIPT
        # detach so the probe outlives this hook (mirrors unity_setup's Editor launch)
        detach: dict[str, Any] = {"start_new_session": True}  # portability: POSIX detach kwarg
        if sys.platform == "win32":  # pragma: no cover - probe is X11/Linux-only
            detach = {"creationflags": getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)}
        try:
            subprocess.Popen(  # nosec B603 - operator-enabled engine plugin script
                [sys.executable, str(script), scan_path],
                cwd=ctx.worktree or ctx.repo_root or None,
                env=self.engine_env(ctx),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                **detach,
            )
        except OSError as e:
            sys.stderr.write(f"unity: dialog-probe launch failed: {e}\n")

    def _reap_dialog_probe(self, ctx) -> None:
        """Terminate a dialog probe recorded in this run dir's pid file, identity-
        guarded so a kernel-recycled pid is never signalled. Best effort + idempotent:
        a missing/stale handle is a no-op. The probe handles SIGTERM by exiting its
        loop, so a polite terminate suffices; the run-end teardown / self-reap poll
        are the backstops."""
        path = self._probe_pid_path(ctx)
        if path is None:
            return
        from bmad_loop import runs  # lazy: keep import cost off the hot path

        pid, identity = runs.read_named_pid_identity(path)
        if pid is not None:
            host = get_process_host()
            if host.alive_and_ours(pid, identity):
                try:
                    host.terminate(pid)
                except OSError:
                    pass
        try:
            path.unlink()
        except OSError:
            pass

    def engine_env(self, ctx) -> dict[str, str]:
        """The ``BMAD_LOOP_*`` environment the helper scripts read — identity +
        worktree from the context, the Editor knobs from this plugin's settings,
        and the MCP agent ids from ``ctx.agents`` (dev + review CLIs). Identical
        to the contract the bespoke ``Engine._run_engine_hook`` used to inject."""
        worktree = ctx.worktree or ctx.repo_root or ""
        env = dict(os.environ)
        env.update(
            {
                "BMAD_LOOP_REPO_ROOT": ctx.repo_root or "",
                "BMAD_LOOP_WORKTREE": worktree,
                "BMAD_LOOP_RUN_DIR": ctx.run_dir or "",
                "BMAD_LOOP_STORY_KEY": ctx.story_key or "",
                "BMAD_LOOP_ENGINE_MCP": str(self.settings.get("mcp", "ivanmurzak")),
                "BMAD_LOOP_ENGINE_EDITOR_MODE": self._editor_mode(),
                "BMAD_LOOP_ENGINE_READY_TIMEOUT": str(self.settings.get("ready_timeout_sec", 600)),
                "BMAD_LOOP_ENGINE_READY_GRACE": str(self.settings.get("ready_grace_sec", -1)),
                "BMAD_LOOP_UNITY_PATH": str(self.settings.get("unity_path", "")),
                "BMAD_LOOP_CLEAN_TMP": "1" if self._clean_tmp else "0",
                "BMAD_LOOP_UNITY_INSTALL_SCENE_GUARD": "1" if self._install_scene_guard() else "0",
                "BMAD_LOOP_UNITY_SCENE_GUARD_DIR": str(
                    self.settings.get("scene_guard_dir", "Assets/BmadLoop/Editor")
                ),
                "BMAD_LOOP_UNITY_DIALOG_PROBE": "1" if self._dialog_probe() else "0",
                "BMAD_LOOP_UNITY_DIALOG_PROBE_INTERVAL_SEC": str(self._dialog_probe_interval()),
                "BMAD_LOOP_UNITY_DIALOG_PROBE_NOTIFY": "1" if self._dialog_probe_notify() else "0",
            }
        )
        # Tell the per_worktree setup which agent MCP configs to point at the
        # worktree's Editor (dev + review may be different CLIs, each with its own
        # config file). Omitted when no real profile is loaded so the script keeps
        # its claude-code default.
        if ctx.agents:
            env["BMAD_LOOP_ENGINE_AGENTS"] = ",".join(ctx.agents)
        return env

    def _run_script(
        self, name: str, ctx, *, timeout: int, extra_env: dict[str, str] | None = None
    ) -> tuple[int, str]:
        """Run one helper script with the engine env, returning (rc, output-tail).
        ``extra_env`` is merged over ``engine_env(ctx)`` for per-call knobs the base
        contract doesn't carry (e.g. the quiesce phase). Never raises: a launch
        failure / timeout maps to a non-zero rc so the readiness gate defers rather
        than crashing the run."""
        script = Path(self.manifest.scripts_dir) / name
        env = self.engine_env(ctx)
        if extra_env:
            env.update(extra_env)
        cwd = ctx.worktree or ctx.repo_root or None
        try:
            proc = subprocess.run(  # nosec B603 - operator-enabled engine plugin script
                # Our own interpreter, not a PATH-resolved `python3`: the helper now
                # imports `bmad_loop.process_host`, which must be importable here even
                # under a pipx-style install where PATH `python3` lacks the package.
                [sys.executable, str(script)],
                cwd=cwd,
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return -1, f"timed out after {timeout}s"
        except OSError as e:
            return -1, str(e)
        return proc.returncode, (proc.stdout + proc.stderr)[-2000:].strip()
