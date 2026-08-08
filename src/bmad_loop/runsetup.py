"""Run-composition layer for the CLI's ``run`` callback.

``cli.cmd_run`` used to build the :class:`~bmad_loop.model.RunState`, wire the
:class:`~bmad_loop.engine.Engine`, and stand up the coding-CLI adapters inline in
an argparse callback — logic that could only be exercised by round-tripping
through argv. This module lifts those pieces out as typed functions so a non-CLI
frontend (or a test) can compose a run directly:

* :func:`make_adapters` — the per-role adapter factory.
* :func:`platform_preflight` — the multiplexer/process-host readiness probe
  ``cmd_validate`` reports.
* :func:`build_run_state` / :func:`compose_run` — the RunState + Engine wiring
  for ``cmd_run``.
* :func:`compose_sweep` — the same wiring for a ``sweep`` run (``cmd_sweep`` and
  the auto-triggered child-sweep factory).
* :func:`compose_resume` — rebuilds the engine for a paused/interrupted run
  (``cmd_resume`` and ``resolve``'s re-arm), selecting the sweep/stories/plain
  variant from persisted run state.
* :func:`config_digest` — the integrity pin over the agent-writable config that
  reaches host code execution (issue #461 point 4).

The engine class and the adapter factory are *injected* into :func:`compose_run`
rather than referenced here directly: ``cli`` resolves ``Engine`` /
``StoriesEngine`` / ``_make_adapters`` from its own module namespace at call time,
so the test suite's ``monkeypatch.setattr(cli, "Engine", ...)`` (and friends)
still bites. ``cli`` re-exports :func:`make_adapters`, :func:`platform_preflight`,
:func:`mux_reason_label`, and :data:`ROLES` under their historical private names
so those seams stay importable and monkeypatchable from ``cli``.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from . import bmadconfig
from . import policy as policy_mod
from . import runs
from .checks import Finding
from .journal import Journal, save_state
from .model import RunState
from .platform_util import atomic_replace
from .runs import RUNS_DIR

if TYPE_CHECKING:
    from collections.abc import Callable

    from .adapters.base import CodingCLIAdapter
    from .adapters.profile import CLIProfile
    from .engine import Engine
    from .policy import Policy
    from .stories_engine import StoriesEngine
    from .sweep import SweepEngine

    class MakeAdapters(Protocol):
        """Call shape of :func:`make_adapters`, which ``compose_*`` takes injected.

        Spelled as a Protocol rather than a ``Callable`` alias only so the
        keyword-only ``profiles`` freeze below is part of the injected contract:
        a frontend that supplies its own factory has to accept the pre-resolved
        profiles, or silently re-read them from disk."""

        def __call__(
            self,
            project: Path,
            run_dir: Path,
            policy: Policy,
            *,
            profiles: dict[str, CLIProfile] | None = None,
        ) -> dict[str, CodingCLIAdapter]: ...


# The three adapter roles a run wires. Defined here (the composition layer that
# actually builds them) and re-exported as ``cli.ROLES``, which `cmd_validate`
# and the test suite resolve.
ROLES = ("dev", "review", "triage")


def resolve_profiles(policy: Policy, project: Path) -> dict[str, CLIProfile]:
    """Resolve every role's :class:`CLIProfile` from disk **once**, as a mapping
    the caller can then hand to both :func:`config_digest` and
    :func:`make_adapters` so the two agree on the same bytes.

    This exists for the child-sweep gate (#461 point 4). ``config_digest`` and
    ``make_adapters`` each used to call ``get_profile`` on their own, which made
    the gate a check-then-use over two *separate* reads of an agent-writable
    file: a session that leaves a background writer flipping
    ``.bmad-loop/profiles/*.toml`` between a benign and a hostile copy needs only
    the digest's read to catch the benign one and the adapter's read to catch the
    other. That race is cheap to retry — a lost round raises `sweep-auto-failed`,
    which `_maybe_auto_sweep` swallows, and the next auto-sweep trigger deals
    again — so "narrow window" is not a defense. Resolving once and threading the
    result removes the second read rather than shrinking the window.

    ``cmd_run`` and ``_resume_paused_run`` thread it too, for a DIFFERENT reason —
    they stamp a baseline rather than compare against one, and at launch the
    on-disk config is the trust anchor, so no race there grants an attacker
    anything a plain pre-launch write does not. What the second read cost them was
    accuracy: the pin they mint is what every later auto-sweep is held to, so a pin
    over bytes the run did not launch makes those children refuse the config the
    parent has been running all along. See ``cli._launch_profiles``.

    The policy half needs no equivalent: ``cli._sweep_factory`` already loads
    ``policy.toml`` once and passes that one frozen ``Policy`` to both the gate
    and the composition. Profiles were the only surface read twice.

    Deduplicated by profile name, so the common single-CLI policy touches disk
    once rather than three times. ``ProfileError`` propagates.
    """
    from .adapters.profile import get_profile

    by_name: dict[str, CLIProfile] = {}
    for role in ROLES:
        name = policy.adapter.resolved(role).name
        if name not in by_name:
            by_name[name] = get_profile(name, project)
    return {role: by_name[policy.adapter.resolved(role).name] for role in ROLES}


def config_digest(
    policy: Policy, project: Path, *, profiles: dict[str, CLIProfile] | None = None
) -> str:
    """sha256 over the agent-writable config that reaches **host** code execution.

    The driven sessions can write anywhere under the project tree, including
    ``.bmad-loop/policy.toml`` and ``.bmad-loop/profiles/*.toml`` — so a session
    can rewrite the commands ``verify`` runs (``shell=True``), the ``binary`` a
    later session is launched from, or the ``[plugins] enabled`` allowlist that
    gates in-process Python import (issue #461 point 4). A run freezes its
    ``Policy`` at launch, so the parent loop is already pinned; this digest exists
    for the one path that re-reads config mid-run with **no human present** — the
    auto-triggered child sweep in ``cli._sweep_factory``.

    Field-scoped on purpose. A whole-file hash would also fire on the benign
    ``[limits]`` live-edits #189 documents as supported, so this covers exactly
    the exec-reachable surface:

    * ``verify.commands`` — order-preserved; they run in sequence.
    * ``sorted(plugins.enabled)`` — set semantics, so order is not meaningful.
    * per :data:`ROLES`, every field that decides **which program runs and with
      what flags and environment**. That rule, not a hand-picked list, is what
      keeps this complete: walk ``GenericAdapter.interactive_argv`` and
      ``interactive_env`` and every token there traces back to one of
      ``binary`` / ``launch_args`` / ``bypass_args`` / ``model_flag`` /
      ``prompt_template`` / ``env`` on the *resolved* profile, or to
      ``extra_args`` on the resolved adapter. The opencode-http builder reads a
      strict SUBSET of those — ``_serve_argv`` takes ``binary`` and the adapter's
      ``extra_args`` and nothing else, and ``_session_env`` layers
      ``profile.env`` plus one *generated* variable, which the ``skill_tree``
      bullet below accounts for. See the union paragraph on why the subset does
      not narrow what is hashed.
    * ``hookless`` — the transport, because it decides *which of those two argv
      builders runs at all*. See the paragraph below on why a hard-coded token is
      not the same thing as a safe one.

    Three of those are easy to lose, and each was lost in an earlier cut of this
    function — which is why the rule above is stated rather than the list.
    ``binary`` (and its siblings) live in ``profiles/*.toml`` and never appear in
    ``policy_snapshot``, so a snapshot-only compare is blind to them.
    ``adapter.extra_args`` *replaces* ``profile.bypass_args`` when set — it is the
    field that carries ``--permission-mode bypassPermissions`` — so hashing the
    profile default alone leaves the flags actually launched unpinned; ``None``
    there means "fall back to the profile", a different state from ``()``, and the
    two are kept apart. And ``prompt_template`` reads like prompt *payload* but is
    not: ``interactive_argv`` places ``render_prompt(spec.prompt)`` in the argv
    list, and the template need not contain ``{prompt}`` at all, so a rewritten
    template is a verbatim attacker-chosen argv token. ``build_command``
    ``shlex.quote``\\ s it, which bounds it to ONE token — no word-splitting — but
    one token is enough for the ``--opt=value`` form.

    ``hookless`` was the fourth, and it was excluded here on a reading that turned
    out to be wrong, so the correction is worth keeping: *a hard-coded argv token
    is not the same thing as a safe one.* Flipping ``hooks.dialect`` to ``"none"``
    does not add a token — it swaps the whole builder, dropping ``launch_args``,
    the prompt and the ``bypass_args`` fallback and putting the literal ``"serve"``
    at argv[1], which ``_spawn_server`` then runs with ``cwd`` at the workspace
    root. To a CLI that is a subcommand and a bad one dies in the health poll. To
    an *interpreter* — a profile whose ``binary`` is ``python``/``sh``/``node``
    with the real program in ``launch_args``, which nothing forbids — argv[1] is a
    **script path resolved against the agent-writable tree**, and the exec happens
    before the health poll it fails (three times: ``SPAWN_ATTEMPTS``). ``binary``
    being pinned does not save it: the attacker inherits whichever binary the
    project configured and only has to write a file named ``serve``. So the token
    is a literal, and the argv is still attacker-controlled — walking the consumer
    means asking what the *launched program* does with a token, not only where the
    token came from.

    The payload is the UNION of those fields across transports, not the subset
    the role's builder actually reads, and that costs a known false positive:
    ``adapter.extra_args`` REPLACES ``bypass_args`` rather than extending it, so
    for a role that sets it the hashed ``bypass_args`` is dead, and rewriting the
    dead field alone moves this digest without moving one token of the launched
    argv. Under ``hookless``, ``bypass_args`` / ``launch_args`` / ``model_flag``
    are dead the same way. Hashing the effective projection instead means
    restating two builders' precedence rules inside the control that polices
    them, where drift is silent and lands in the UNDER-covering direction — the
    failure this function has already made four times by reasoning from one
    builder. Over-coverage fails the other way, loudly: ``sweep-auto-failed`` +
    notify, with the message naming ``bmad-loop sweep`` as the human-present path.
    Not free (the refusal burns the trigger for the life of the run, #501), but it
    needs a writer, and nothing under ``src/`` writes ``.bmad-loop/profiles/*.toml``
    at all — that overlay is hand-authored, and the TUI settings screen writes
    ``policy.toml`` (``extra_args`` included). So a dead-field rewrite arriving
    mid-run is a config change nobody automated made under a running loop, which
    is the condition this gate reports rather than a false alarm to suppress.

    Deliberately EXCLUDED:

    * The *bytes* behind ``binary``/``launch_args`` — this pins the launch
      target's SPELLING, not its content. A project-local target (``binary`` a
      path into the tree, or ``python`` with the program in ``launch_args``) can
      be rewritten in place with no config field moving. The gap is real and
      unguarded: ``profile.py`` requires only a non-empty ``binary`` string, while
      the three sibling path fields (``hooks.config_path``, ``skill_tree``,
      ``seed_files``) all reject absolute and parent refs.

      NOT excluded on "the parent execs it too" — that defence is false for the
      ``triage`` role. Base ``Engine`` wires only dev+review; ``sweep.py`` holds
      the only ``adapters["triage"]`` assignment and the only two ``role="triage"``
      dispatches, so a ``[adapter.triage]`` profile override's target is exec'd by
      a sweep and by nothing else. ``sweep.auto = "run-end"`` and worktree
      isolation give two more shapes where the child is the uniquely exposed one.

      Excluded because a hash cannot identify the target. Which ``launch_args``
      token names a file is undecidable (``-i`` vs ``tools/agent.py``);
      digest-time resolution is not the tmux shell's; and one indirection defeats
      it — this repo's own ``write_script_launcher`` is a stub that execs an
      interpreter on a sidecar, so hashing the stub misses the payload. Nor is
      the target ours to pin: it is normally a third-party CLI that self-updates,
      and a mid-run update would move a content hash and burn the auto-sweep
      trigger for the life of the run (``_maybe_auto_sweep`` records the trigger
      BEFORE calling the factory, and early-returns on it forever after).
      Confinement is the instrument, not hashing — and as a ``validate`` warning
      rather than a refusal, since "resolves inside the project" does not decide
      it either: under an active project venv ``which("python")`` IS
      ``<project>/.venv/bin/python``, and this repo's own zero-token E2E gate
      configures ``binary`` at ``<sandbox>/.bmad-loop/fake-cli.sh``. Tracked as
      #500, to land with #499's fix option 3; unreachable on stock config (all six
      shipped profiles are bare PATH names).
    * ``hooks.config_path`` — the relay is issue #461's points 1-3, hardened on
      its own track; folding it in would fire on an ordinary ``bmad-loop init``.
    * ``adapter.model`` — it cannot introduce an argv token, only fill the value
      slot behind ``model_flag``, which IS pinned here. Pinning it would refuse an
      auto-sweep after a human's mid-run model change in the TUI.
    * ``skill_tree`` — the one profile field reaching a launched session's env
      without passing through argv. For a hookless role ``_config_content`` plants
      ``cwd/skill_tree`` in that ``OPENCODE_CONFIG_CONTENT`` as ``skills.paths``,
      so a rewritten tree points the unattended child sweep at instructions of the
      writer's choosing. Excluded because the pointer is not the door — the
      content is, and the content is reachable with no config change at all, so
      there is no digest to move either way. The triage session that consumes
      ``bmad-loop-sweep`` runs at ``workspace.root`` — the main checkout, never a
      unit worktree, since ``sweep.py`` swaps the workspace only around bundle
      execution — and a driven session can write that copy: worktrees mount under
      ``.bmad-loop/runs/`` INSIDE the main checkout, and nothing confines a
      session's writes to its cwd (this docstring's opening premise). Nothing
      hashes or reseeds a skill in place: ``_copy_skills`` skips an existing skill
      dir absent ``--force-skills``.

      Be precise about the worktree case, because the obvious reading is wrong:
      under ``isolation = "worktree"`` a session editing the skill tree *in its
      own worktree* does NOT reach that child. A gitignored tree is absent from a
      fresh ``git worktree add`` checkout, so ``provision_worktree`` seeds it from
      the WHEEL (``resources.files("bmad_loop.data")``) — the per-file no-clobber
      that ``test_provision_worktree_does_not_clobber_existing_skill`` pins only
      preserves a file the destination already has, which is the *tracked*
      skill-tree case that test's own docstring names. The two copies then never
      share bytes: the worktree-local exclude blocks staging and ``git merge``
      moves only tracked content. The route above (absolute path into the main
      checkout) is what carries this bullet, not in-worktree editing.

      Skill-content integrity is a real question and not one a config hash can
      answer.
    * ``usage_parser`` — and with it the rest of the token-budget surface. It
      selects a read-only tally over a transcript the orchestrator opens anyway
      and decides no program, flag, or variable. Rewriting it to ``"none"`` DOES
      make the mid-session guard inert (``tokens.read_usage`` returns None for
      anything it does not recognize, so ``_sample_weighted_usage`` never trips
      and the session drops out of accounting) — but the guard's own controls are
      ``limits.session_budget_mode`` / ``max_tokens_per_session`` /
      ``cache_read_weight``, which sit in the ``[limits]`` excluded above and
      which the child sweep re-reads fresh from disk. ``session_budget_mode =
      "off"`` silences the guard in one line and more completely (it never
      samples at all), so pinning the parser bolts a side door in a wall whose
      main door is held open on purpose. Admitting it also widens the rule from
      the launch surface to the launch surface *plus safety knobs*, which is
      ``[limits]`` — the whole-file hash the second paragraph exists to reject.
    * ``[plugins.<name>]`` settings — an enabled plugin's resolved settings do
      reach exec (``bus.py`` exports each as ``BMAD_LOOP_SETTING_*`` into a
      ``shell=True`` hook, and the Unity plugin turns one into an ``--editor-path``
      argv token), but the TUI settings screen writes those same tables, so
      pinning them would refuse an auto-sweep after a supported human edit. They
      are inside an *already-enabled* plugin's blast radius — the trust boundary
      the gap below is about — rather than a way past the allowlist.

    Known gaps, tracked separately (#496, #497). This pins the plugin allowlist by
    NAME only, and the allowlist is not the whole plugin exec surface:

    * A project-origin ``.bmad-loop/plugins/<name>/`` overrides a same-named
      builtin (``plugins/loader.py`` overlay precedence) and
      ``trust.require_enabled`` gates on the name, so a session can swap the module
      behind an *already* enabled plugin without moving this digest.
    * Adding a plugin is caught only for one that declares ``[python]``. A
      *declarative* manifest (no ``[python]``) loads on folder-drop by design
      (``plugins/trust.py``) and ``registry.hooks_for`` hands its ``[hooks.<stage>]
      cmd`` to the bus, which runs it with ``shell=True`` — with no ``enabled``
      entry, and from a directory this digest never reads.

    Closing either is a plugin-trust-model change rather than a wider hash, so
    neither belongs here.

    Tuples are normalized to lists before ``json.dumps(sort_keys=True)`` for the
    same reason ``cli._resume_paused_run``'s ``policy_changed`` compare does it:
    the live policy carries TUPLES where a persisted round-trip yields lists, and
    a raw compare then reports "changed" every single time. ``ProfileError``
    propagates — an unresolvable profile already aborts at :func:`make_adapters`.

    ``profiles`` is an already-resolved mapping from :func:`resolve_profiles`.
    Pass it wherever the digest gates — or becomes the baseline for — something
    that then *runs* under the same config, so the bytes hashed here are the bytes
    launched rather than a second read of a file the sessions can rewrite in
    between. Both halves of that rule have a caller: ``cli._sweep_factory`` gates,
    ``cmd_run``/``_resume_paused_run`` baseline. Omit it to resolve fresh, which
    ``cmd_sweep`` does deliberately — a human started that one, and the pin it
    stamps gates no child."""
    profiles = profiles if profiles is not None else resolve_profiles(policy, project)

    launch: dict[str, dict[str, object]] = {}
    for role in ROLES:
        cfg = policy.adapter.resolved(role)
        prof = profiles[role]
        launch[role] = {
            "binary": prof.binary,
            "launch_args": list(prof.launch_args),
            "bypass_args": list(prof.bypass_args),
            "model_flag": prof.model_flag,
            # An argv element, not prompt payload: render_prompt returns this
            # template formatted, and it need not reference {prompt} at all.
            "prompt_template": prof.prompt_template,
            "env": dict(prof.env),
            # The transport, because it rewrites the argv WHOLESALE rather than
            # adding a token: hookless drops launch_args/prompt/bypass_args and
            # substitutes `serve --port … --print-logs`, whose literal "serve" an
            # interpreter binary reads as a cwd-relative script path.
            "hookless": prof.hookless,
            # None (inherit profile.bypass_args) is NOT the same state as () (an
            # explicit override to no flags at all); json.dumps keeps them apart.
            "extra_args": None if cfg.extra_args is None else list(cfg.extra_args),
        }
    payload = {
        "verify_commands": list(policy.verify.commands),
        "plugins_enabled": sorted(policy.plugins.enabled),
        "profiles": launch,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def make_adapters(
    project: Path,
    run_dir: Path,
    policy,
    *,
    profiles: dict[str, CLIProfile] | None = None,
) -> dict[str, CodingCLIAdapter]:
    """Build the per-role adapters. ``profiles`` is an already-resolved mapping
    from :func:`resolve_profiles`; when given, no profile is re-read from disk, so
    a caller that gated on :func:`config_digest` launches the *same* bytes it
    validated (#461 point 4). Omitted, each role resolves fresh as before."""
    from .adapters.generic import GenericAdapter, GenericDevAdapter
    from .adapters.multiplexer import fold_version, get_multiplexer, mux_usable
    from .adapters.profile import ProfileError, get_profile

    # The dev skill (bmad-dev-auto) writes no result.json: its adapter
    # synthesizes the result from the spec, and so needs the project paths to
    # find that spec — rebasing onto the active worktree's implementation-
    # artifacts dir under isolation, not just the main checkout's.
    paths = bmadconfig.load_paths(project)
    mux = None
    adapters: dict[str, CodingCLIAdapter] = {}
    by_cfg: dict = {}
    for role in ROLES:
        cfg = policy.adapter.resolved(role)
        # Both the dev and review sessions are now bmad-dev-auto runs (the review
        # session re-invokes the dev skill on the done spec for a follow-up pass),
        # and the skill writes no result.json — its adapter synthesizes the result
        # from the spec it leaves on disk, so it needs the project paths to find
        # that spec and cannot be shared with the triage role even on identical config.
        synthesizes = role in ("dev", "review") and policy.dev.skill == "bmad-dev-auto"
        key = (cfg, synthesizes)
        if key not in by_cfg:
            if profiles is not None:
                profile = profiles[role]
            else:
                try:
                    profile = get_profile(cfg.name, project)
                except ProfileError as e:
                    raise SystemExit(f"error: {e}") from e
            if profile.hookless:
                # Hookless profiles (opencode-http) are driven over HTTP/SSE —
                # the tmux adapters below cannot host them.
                from .adapters.opencode_http import (
                    OpencodeDevAdapter,
                    OpencodeHttpAdapter,
                    OpencodeServerError,
                )

                common = dict(
                    run_dir=run_dir,
                    policy=policy,
                    profile=profile,
                    extra_args=cfg.extra_args,
                    usage_grace_s=cfg.usage_grace_s,
                    stop_without_result_nudges=cfg.stop_without_result_nudges,
                )
                try:
                    # heterogeneous **kwargs: pyright unions the dict values; per-arg error is spurious
                    by_cfg[key] = (
                        OpencodeDevAdapter(**common, paths=paths)
                        if synthesizes
                        else OpencodeHttpAdapter(**common)  # pyright: ignore[reportArgumentType]
                    )
                except OpencodeServerError as e:
                    raise SystemExit(f"error: {e}") from e
            else:
                # Resolve and probe the shared multiplexer only when a profile
                # actually uses it; hookless HTTP/SSE runs need no transport.
                if mux is None:
                    mux = get_multiplexer()
                    if not mux_usable(mux):
                        try:
                            version = fold_version(mux.version())
                        except Exception:  # diagnosing must not mask the refusal
                            version = None
                        raise SystemExit(
                            f"error: multiplexer backend {type(mux).__name__} is not usable on "
                            f"this host (reported version: {version}); its transport binary is "
                            "missing, the version is unsupported, or a required helper is "
                            "absent (psmux needs `pwsh` on PATH); see `bmad-loop diagnose`"
                        )
                common = dict(
                    run_dir=run_dir,
                    policy=policy,
                    profile=profile,
                    extra_args=cfg.extra_args,
                    usage_grace_s=cfg.usage_grace_s,
                    stop_without_result_nudges=cfg.stop_without_result_nudges,
                    mux=mux,
                )
                # heterogeneous **kwargs: pyright unions the dict values; per-arg error is spurious
                by_cfg[key] = (
                    GenericDevAdapter(**common, paths=paths)
                    if synthesizes
                    else GenericAdapter(**common)  # pyright: ignore[reportArgumentType]
                )
        adapters[role] = by_cfg[key]
    return adapters


def mux_reason_label(reason: str) -> str:
    """Human wording for a MuxBackendInfo.reason, shared by `mux` and validate."""
    return {
        "env": "forced by BMAD_LOOP_MUX_BACKEND",
        "policy": f"set by [mux] backend in {policy_mod.POLICY_FILE}",
        "platform-default": f"platform default for {sys.platform}",
        "first-match": "first available platform match",
        "fallback": "fallback (no registered backend is available)",
    }.get(reason, reason)


def platform_preflight() -> list[Finding]:
    """Probe the platform-selected seams — the terminal multiplexer and the process
    host — for `cmd_validate`, returning the findings in emission order.

    A backend reports its own readiness through ``available()`` / ``version()``, so
    a new OS or transport surfaces here by *registering* rather than by adding a
    ``sys.platform`` branch to validate. The process host is named so a
    misselection (e.g. the Windows host picked on Linux) is visible at a glance.
    """
    from .adapters.multiplexer import (
        detect_multiplexers,
        external_backend_errors,
        fold_version,
        get_multiplexer,
    )
    from .process_host import get_process_host

    found: list[Finding] = []

    try:
        backend = get_multiplexer()
        label = type(backend).__name__
        # Defensive fold: an out-of-tree backend can break the seam's
        # single-line promise, and this string lands in an inline message.
        version = fold_version(backend.version())
        if backend.available():
            found.append(
                Finding(
                    "mux.backend",
                    "ok",
                    f"multiplexer {label} available" + (f" ({version})" if version else ""),
                    {"backend": label, "available": True, "version": version},
                )
            )
        else:
            found.append(
                Finding(
                    "mux.backend",
                    "problem",
                    f"multiplexer {label} unavailable"
                    + (f" (reports {version})" if version else "")
                    + " — its transport binary is missing, the version is unsupported, or a "
                    "required helper is absent (psmux needs `pwsh` on PATH); "
                    "see `bmad-loop diagnose`",
                    {"backend": label, "available": False, "version": version},
                )
            )
    except Exception as e:  # selection or readiness must not abort validate
        found.append(Finding("mux.preflight", "problem", f"multiplexer preflight failed: {e}"))

    try:
        infos = detect_multiplexers()
    except Exception:  # detection is advisory; never break validate
        infos = []
    if len(infos) > 1:  # a lone tmux needs no listing; keep single-backend output stable
        listed = ", ".join(
            i.name
            + ("*" if i.selected else "")
            + (
                " (available" + (f", {i.version}" if i.version else "") + ")"
                if i.available
                else " (unavailable)"
            )
            for i in infos
        )
        # The text flattens each row into a suffix soup ("tmux*, psmux
        # (unavailable)") whose trailing `*` a consumer would have to parse to
        # learn which backend is selected. The detail keeps the rows themselves.
        found.append(
            Finding(
                "mux.backends-detected",
                "ok",
                f"mux backends: {listed} — `bmad-loop mux` for details",
                {
                    "backends": [
                        {
                            "name": i.name,
                            "matches_platform": i.matches_platform,
                            "available": i.available,
                            "version": i.version,
                            "selected": i.selected,
                            "reason": i.reason,
                        }
                        for i in infos
                    ]
                },
            )
        )
    chosen = next((i for i in infos if i.selected), None)
    if chosen and chosen.reason in ("env", "policy"):
        # detail keeps the raw enum, not mux_reason_label's prose: the label is
        # wording ("set by [mux] backend in .bmad-loop/policy.toml"), the enum is
        # the value MuxBackendInfo.reason actually carries.
        found.append(
            Finding(
                "mux.selection",
                "ok",
                f"multiplexer selection {mux_reason_label(chosen.reason)}",
                {"backend": chosen.name, "reason": chosen.reason},
            )
        )

    # A warning, not a problem and not a note: an installed package the operator
    # asked for did not load, which is a real failure — but selection already
    # degraded past it (a failed external can never be the selected backend), so
    # the preflight outcome above is authoritative and the verdict must not flip.
    # `cmd_mux` has always printed this same condition as `warning:`; validate was
    # the outlier, pinned to "ok" because promoting inserts "  warning: " into the
    # text (render() keeps the double prefix by design) and the TUI rendered that
    # text verbatim. Since #210 the TUI reads `validate --json` and styles from the
    # severity field, so the severity is free to say what the message already does.
    for ep_name, reason in sorted(external_backend_errors().items()):
        found.append(
            Finding(
                "mux.external-backend",
                "warning",
                f"external mux backend '{ep_name}' failed to load: {reason}",
                {"entry_point": ep_name, "error": reason},
            )
        )

    try:
        host = type(get_process_host()).__name__
        found.append(Finding("host.process", "ok", f"process host: {host}", {"host": host}))
    except Exception as e:  # a bad BMAD_LOOP_PROCESS_HOST must report, not crash
        found.append(Finding("host.process", "problem", f"process host preflight failed: {e}"))

    return found


def build_run_state(
    *,
    run_id: str,
    project: Path,
    policy: Policy,
    epic_filter: int | None,
    story_filter: str | None,
    max_stories: int | None,
    stories_on: bool,
    spec_folder: str,
    trusted_config_digest: str,
) -> RunState:
    """Assemble the launch-time :class:`RunState` for a fresh run.

    ``policy_snapshot`` freezes ``policy`` at launch so every later display reads
    the weights the run actually launched under; ``source`` / ``spec_folder``
    record which queue the run dispatches (a stories manifest vs sprint-status).

    ``trusted_config_digest`` is passed in rather than derived from ``policy`` +
    ``project`` here so the value stamped on the run is the same string the caller
    handed the sweep factory — and so a broken profile still aborts where it always
    did (``make_adapters``'s ``SystemExit``), not from inside this constructor."""
    return RunState(
        run_id=run_id,
        project=str(project),
        started_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
        policy_snapshot=policy.to_dict(),
        trusted_config_digest=trusted_config_digest,
        epic_filter=epic_filter,
        story_filter=story_filter,
        max_stories=max_stories,
        source="stories" if stories_on else "sprint-status",
        spec_folder=spec_folder if stories_on else "",
    )


@dataclass
class ComposedRun:
    """The composed-but-not-yet-run artifacts a ``compose_*`` returns for its
    callback to render from — shared by :func:`compose_run`, :func:`compose_sweep`,
    and :func:`compose_resume`.

    ``engine`` is ready to :meth:`run`; ``run_id`` names the run for the attach
    hint. ``run_dir`` / ``state`` / ``journal`` are the persisted context a caller
    other than the CLI can inspect."""

    engine: Engine
    run_id: str
    run_dir: Path
    state: RunState
    journal: Journal


def compose_run(
    *,
    project: Path,
    paths: bmadconfig.ProjectPaths,
    policy: Policy,
    run_id: str | None,
    epic_filter: int | None,
    story_filter: str | None,
    max_stories: int | None,
    stories_on: bool,
    spec_folder: str,
    sweep_factory: Callable[[str], None],
    make_adapters: MakeAdapters,
    engine_cls: type[Engine],
    stories_engine_cls: type[StoriesEngine],
    trusted_config_digest: str,
    profiles: dict[str, CLIProfile] | None = None,
) -> ComposedRun:
    """Stand up a run: allocate the run dir, persist state + pid, build the
    adapters, and wire the engine — everything ``cmd_run`` did inline between its
    preflight gates and ``engine.run()``.

    ``profiles`` carries ``cmd_run``'s single :func:`resolve_profiles` resolution —
    the same one ``trusted_config_digest`` was computed from — so the stamped
    baseline describes the bytes these adapters are built from rather than a second
    read of an agent-writable file (#461 point 4). ``None`` resolves fresh.

    ``make_adapters`` and the engine classes are injected (rather than imported
    here) so ``cli`` supplies its own module-level names — keeping the test
    suite's ``monkeypatch.setattr(cli, "Engine"/"_make_adapters", ...)`` effective.
    """
    run_id = run_id or runs.new_run_id()
    run_dir = project / RUNS_DIR / run_id
    journal = Journal(run_dir)
    state = build_run_state(
        run_id=run_id,
        project=project,
        policy=policy,
        epic_filter=epic_filter,
        story_filter=story_filter,
        max_stories=max_stories,
        stories_on=stories_on,
        spec_folder=spec_folder,
        trusted_config_digest=trusted_config_digest,
    )
    save_state(run_dir, state)
    runs.write_pid(run_dir)
    adapters = make_adapters(project, run_dir, policy, profiles=profiles)
    journal.append(
        "run-start",
        run_id=run_id,
        source=state.source,
        adapter_dev=policy.adapter.resolved("dev").name,
        adapter_review=policy.adapter.resolved("review").name,
    )
    common = dict(
        paths=paths,
        policy=policy,
        adapter=adapters["dev"],
        review_adapter=adapters["review"],
        run_dir=run_dir,
        journal=journal,
        state=state,
        max_stories=max_stories,
        epic_filter=epic_filter,
        story_filter=story_filter,
        sweep_factory=sweep_factory,
    )
    # heterogeneous **kwargs: pyright unions the dict values; per-arg error is spurious
    engine: Engine = (
        stories_engine_cls(**common, spec_folder=spec_folder)
        if stories_on
        else engine_cls(**common)  # pyright: ignore[reportArgumentType]
    )
    return ComposedRun(engine=engine, run_id=run_id, run_dir=run_dir, state=state, journal=journal)


def compose_sweep(
    *,
    project: Path,
    paths: bmadconfig.ProjectPaths,
    policy: Policy,
    run_id: str | None,
    prompting: bool,
    decisions_only: bool,
    max_bundles: int | None,
    repeat: bool | None,
    max_cycles: int | None,
    trigger: str,
    make_adapters: MakeAdapters,
    sweep_engine_cls: type[SweepEngine],
    trusted_config_digest: str,
    profiles: dict[str, CLIProfile] | None = None,
) -> ComposedRun:
    """Stand up a sweep run: allocate the run dir, persist state + pid, record the
    sweep options, build the adapters, and wire the ``SweepEngine`` — everything
    ``cli._start_sweep`` did inline before ``engine.run()``.

    ``profiles`` carries an already-resolved :func:`resolve_profiles` mapping down
    to ``make_adapters``. The child-sweep factory passes the same one it gated on,
    so the adapters are built from the validated bytes instead of a fresh read of
    an agent-writable file (#461 point 4); ``cmd_sweep`` (human-present) omits it.

    ``sweep.json`` freezes the launch options so a resume rebuilds the same sweep
    (see :func:`compose_resume`). ``make_adapters`` and ``sweep_engine_cls`` are
    injected so ``cli`` supplies its own module-level names — keeping the test
    suite's ``monkeypatch.setattr(cli, "SweepEngine"/"_make_adapters", ...)``
    effective."""
    run_id = run_id or runs.new_run_id()
    run_dir = project / RUNS_DIR / run_id
    journal = Journal(run_dir)
    state = RunState(
        run_id=run_id,
        project=str(project),
        started_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
        policy_snapshot=policy.to_dict(),
        trusted_config_digest=trusted_config_digest,
        run_type="sweep",
    )
    save_state(run_dir, state)
    runs.write_pid(run_dir)
    options = {
        "prompting": prompting,
        "decisions_only": decisions_only,
        "max_bundles": max_bundles,
        "repeat": repeat,
        "max_cycles": max_cycles,
        "trigger": trigger,
    }
    # Persist the sweep options atomically (tmp + os.replace), the way save_state
    # writes state.json: a resume reads this back to rebuild the SweepEngine, so a
    # crash mid-write must not leave a torn file the recovery path then chokes on.
    sweep_path = run_dir / "sweep.json"
    sweep_tmp = sweep_path.with_suffix(".json.tmp")
    sweep_tmp.write_text(json.dumps(options, indent=2), encoding="utf-8")
    atomic_replace(sweep_tmp, sweep_path)
    adapters = make_adapters(project, run_dir, policy, profiles=profiles)
    journal.append("run-start", run_id=run_id, run_type="sweep", trigger=trigger)
    engine: Engine = sweep_engine_cls(
        paths=paths,
        policy=policy,
        adapter=adapters["dev"],
        review_adapter=adapters["review"],
        triage_adapter=adapters["triage"],
        run_dir=run_dir,
        journal=journal,
        state=state,
        prompting=prompting,
        decisions_only=decisions_only,
        max_bundles=max_bundles,
        repeat=repeat,
        max_cycles=max_cycles,
    )
    return ComposedRun(engine=engine, run_id=run_id, run_dir=run_dir, state=state, journal=journal)


def compose_resume(
    *,
    project: Path,
    paths: bmadconfig.ProjectPaths,
    run_dir: Path,
    state: RunState,
    policy: Policy,
    journal: Journal,
    sweep_factory: Callable[[str], None],
    make_adapters: MakeAdapters,
    engine_cls: type[Engine],
    stories_engine_cls: type[StoriesEngine],
    sweep_engine_cls: type[SweepEngine],
    profiles: dict[str, CLIProfile] | None = None,
) -> ComposedRun:
    """Rebuild the engine for a paused/interrupted run and return it ready to
    :meth:`run` — the adapter build + engine selection ``cli._resume_paused_run``
    did inline.

    ``state`` arrives already re-stamped and persisted by the caller: the resume
    policy-snapshot reconciliation and the pause/pid/graceful-stop bookkeeping stay
    CLI-side (their ordering is load-bearing — see ``_resume_paused_run``), so this
    lifts only the composition. The variant is selected from persisted run state:
    ``run_type == "sweep"`` rebuilds a ``SweepEngine`` from ``sweep.json``;
    otherwise ``source`` picks ``StoriesEngine`` vs ``Engine``, restoring the
    launching scope + cap so a resumed ``--epic N`` run keeps its filter. The engine
    classes and ``make_adapters`` are injected so ``cli``'s ``monkeypatch.setattr``
    seams bite.

    ``profiles`` carries the caller's single :func:`resolve_profiles` resolution —
    the one the re-stamped ``state.trusted_config_digest`` was computed from — so
    the new baseline describes the bytes these adapters are built from rather than
    a second read of an agent-writable file (#461 point 4). ``None`` resolves
    fresh."""
    # drop any stale agent session so the run spins up a fresh one (a stopped or
    # interrupted run can leave a lingering bmad-loop-<id> session behind).
    runs.kill_session(run_dir.name)
    adapters = make_adapters(project, run_dir, policy, profiles=profiles)
    if state.run_type == "sweep":
        opts_path = run_dir / "sweep.json"
        try:
            opts = json.loads(opts_path.read_text(encoding="utf-8")) if opts_path.is_file() else {}
        except (OSError, json.JSONDecodeError):
            # A torn/corrupt sweep.json (crash mid-write on an older run) must not
            # abort the recovery path — fall back to the same launch defaults as
            # the missing-file arm, mirroring tui.data's tolerant run-dir reads.
            opts = {}
        engine: Engine = sweep_engine_cls(
            paths=paths,
            policy=policy,
            adapter=adapters["dev"],
            review_adapter=adapters["review"],
            triage_adapter=adapters["triage"],
            run_dir=run_dir,
            journal=journal,
            state=state,
            prompting=bool(opts.get("prompting", False)),
            decisions_only=bool(opts.get("decisions_only", False)),
            max_bundles=opts.get("max_bundles"),
            repeat=opts.get("repeat"),
            max_cycles=opts.get("max_cycles"),
        )
    else:
        story_common = dict(
            paths=paths,
            policy=policy,
            adapter=adapters["dev"],
            review_adapter=adapters["review"],
            run_dir=run_dir,
            journal=journal,
            state=state,
            # restore the launching scope + cap so a resumed `--epic N` run keeps
            # picking within N instead of silently widening to every epic.
            epic_filter=state.epic_filter,
            story_filter=state.story_filter,
            max_stories=state.max_stories,
            sweep_factory=sweep_factory,
        )
        # stories mode is pinned in run state at launch, so resume rebuilds the
        # same picker (StoriesEngine) without any flag.
        # heterogeneous **kwargs: pyright unions the dict values; per-arg error is spurious
        engine = (
            stories_engine_cls(**story_common, spec_folder=state.spec_folder)
            if state.source == "stories"
            else engine_cls(**story_common)  # pyright: ignore[reportArgumentType]
        )
    return ComposedRun(
        engine=engine, run_id=run_dir.name, run_dir=run_dir, state=state, journal=journal
    )
