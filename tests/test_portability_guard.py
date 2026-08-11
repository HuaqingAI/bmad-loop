"""Regression guard against POSIX-only patterns creeping back into the core.

The POSIX-decoupling pass (multiplexer seam + portability fixes) quarantined
every Unix assumption behind a single tmux backend and a handful of
platform-guarded helpers. This guard byte/AST-scans ``src/bmad_loop`` so a new
hard POSIX dependency can't sneak in unnoticed. Each sanctioned exception lives
in an allowlisted file and — outside the wholesale tmux quarantine — carries a
``# portability:`` ack on its line, so exceptions stay deliberate.

The same single-pass scan also carries the one non-POSIX quarantine that has the
identical shape: AGENTS.md's "New core env vars register in ``envvars.py``;
plugin-owned env-var families stay with their plugin" — see
``test_bmad_loop_env_reads_only_in_the_registry``.

If this test flags something unexpected, fix the source (route it through the
seam / a platform helper) rather than widening an allowlist.
"""

from __future__ import annotations

import ast
from pathlib import Path

import bmad_loop
from bmad_loop import envvars

SRC = Path(bmad_loop.__file__).resolve().parent
# Marker an allowlisted exception line must carry. Written as ``# portability: …``;
# matched as the bare keyword so it also rides along on a ``# nosec B108 portability: …``.
ACK = "portability:"

# ----------------------------------------------------------------- allowlists

# The files allowed to shell out to ``tmux`` — the whole-file quarantine for
# tmux/POSIX-shell knowledge, split across the shared base (where the spawn
# primitive + argv live) and its POSIX leaf. No per-line ack needed: these files
# *are* the sanctioned spot (their module docstrings say so).
TMUX_BACKENDS = {"adapters/tmux_base.py", "adapters/tmux_backend.py"}

# Files that may name a bare POSIX path, each on a line carrying a `# portability:`
# ack. process_host.py's Linux identity reader walks `/proc/<pid>/stat` behind a
# sys.platform branch; the Unity teardown scripts are POSIX-only. verify.py is the
# one non-platform case: git's *diff format* spells an absent file `/dev/null` on
# every platform, so `patch_new_files` compares against it as a protocol token.
PATH_ALLOW = {
    "data/plugins/unity/unity_cleanup.py",
    "data/plugins/unity/unity_teardown.py",
    "process_host.py",
    "verify.py",
}

# The detach helpers that legitimately request POSIX `start_new_session` (each
# branches on `sys.platform` for a Windows creationflags fallback).
DETACH_ALLOW = {
    "platform_util.py",
    "data/plugins/unity/unity_setup.py",
    "data/plugins/unity/unity_plugin.py",
}

# `os.kill(pid, 0)` is a read-only existence probe on POSIX but *destructive* on
# Windows (it maps to TerminateProcess). Confine it to the platform-guarded
# liveness helpers, each on a line carrying a `# portability:` ack; everything
# else routes through the ProcessHost seam (`get_process_host().is_alive`). The
# Unity teardown no longer probes directly — it delegates to the seam.
KILL_PROBE_ALLOW = {
    "process_host.py",
}

# Broader than the signal-0 probe: *any* `os.kill(` — a real signal send is just as
# destructive-on-Windows as the probe form. Only the ProcessHost may call it directly;
# everything else routes through the seam (terminate / force_kill / is_alive).
OS_KILL_ALLOW = {
    "process_host.py",
}

# The two sanctioned `shell=True` spots: operator-authored command strings whose
# cmd/PowerShell port is an explicit out-of-scope follow-up.
SHELL_ALLOW = {
    "verify.py",
    "plugins/bus.py",
}

# The files allowed to read a `BMAD_LOOP_*` variable straight out of the process
# environment. `envvars.py` *is* the registry — the one place a core var is named,
# typed and given a reader, which every core call site then calls (AGENTS.md: "New
# core env vars register in `envvars.py`"). The two hook relays are copied OUT of
# the package into the target project and run inside the coding CLI's process under
# whatever interpreter the host has (both say "Stdlib only" in their docstrings), so
# they cannot import bmad_loop to reach the registry at all. The Unity helper
# scripts are stand-alone in the same way and read the plugin's own
# `BMAD_LOOP_UNITY_*` / `BMAD_LOOP_ENGINE_*` contract, which the second half of the
# invariant leaves with the plugin — envvars.py's docstring carves them out by name.
# Writes stay out of scope on purpose: engine/resolve/probe/plugins.bus/unity_plugin
# *build* a `BMAD_LOOP_*` env dict to inject into a child session, and that
# producing side is what these readers consume, not a second source of truth.
ENV_READ_ALLOW = {
    "envvars.py",
    "data/bmad_loop_hook.py",
    "data/bmad_loop_probe_hook.py",
    "data/plugins/unity/unity_cleanup.py",
    "data/plugins/unity/unity_dialog_probe.py",
    "data/plugins/unity/unity_quiesce.py",
    "data/plugins/unity/unity_ready.py",
    "data/plugins/unity/unity_seed_assets.py",
    "data/plugins/unity/unity_setup.py",
    "data/plugins/unity/unity_teardown.py",
}

# Bare POSIX paths that must not be hardcoded outside PATH_ALLOW. `os.devnull` is
# the portable replacement for "/dev/null".
POSIX_PATHS = ("/tmp", "/proc", "/dev/null")

# Prefix that makes an environment variable this project's to register.
ENV_PREFIX = "BMAD_LOOP_"

# ``CONSTANT_NAME -> "BMAD_LOOP_…"`` for the registry's own public constants, read
# off the live module so the guard cannot drift from it: register a fourth var in
# envvars.py and the scan resolves reads spelled through it with no edit here.
# This is what lets a read reach the guard when it borrows the registry's constant
# but skips the registry's reader — the shape a well-meaning change actually takes.
REGISTRY_NAMES = {
    name: value
    for name, value in vars(envvars).items()
    if isinstance(value, str) and value.startswith(ENV_PREFIX)
}


def _py_files() -> list[Path]:
    return sorted(SRC.rglob("*.py"))


def _rel(path: Path) -> str:
    return path.relative_to(SRC).as_posix()


def _docstring_node_ids(tree: ast.AST) -> set[int]:
    """Ids of the string-Constant nodes that are module/class/function docstrings
    — excluded from literal scans (prose, not code)."""
    ids: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            body = getattr(node, "body", [])
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                ids.add(id(body[0].value))
    return ids


def _classify_posix_path(value: str) -> str | None:
    """The POSIX path this string literal hardcodes, or None. Matches the whole
    value or a subpath of it, so big shell strings that merely *contain*
    ``2>/dev/null`` and lookalikes such as ``~/.gemini/tmp/...`` are not flagged."""
    for pat in POSIX_PATHS:
        if value == pat:
            return pat
        if pat != "/dev/null" and value.startswith(pat + "/"):
            return pat
    return None


def _is_os_environ(node: ast.expr) -> bool:
    """True for the ``os.environ`` attribute access itself."""
    return (
        isinstance(node, ast.Attribute)
        and node.attr == "environ"
        and isinstance(node.value, ast.Name)
        and node.value.id == "os"
    )


def _env_name_aliases(tree: ast.AST) -> dict[str, str]:
    """``NAME -> "BMAD_LOOP_…"`` for every constant binding in the module, so a read
    spelled through a named constant still resolves. That indirection is the norm
    here, not an edge case: the registry reads ``os.environ.get(MUX_BACKEND)`` and
    gates.py names its notify vars ``_TITLE_ENV`` / ``_MESSAGE_ENV`` — matching the
    string literal alone would miss exactly the well-behaved shape."""
    aliases: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            targets = [t for t in node.targets if isinstance(t, ast.Name)]
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            targets = [node.target]
        else:
            continue
        value = node.value
        if (
            targets
            and isinstance(value, ast.Constant)
            and isinstance(value.value, str)
            and value.value.startswith(ENV_PREFIX)
        ):
            for target in targets:
                aliases[target.id] = value.value
    return aliases


def _env_call_key_node(call: ast.Call) -> ast.expr | None:
    """The node holding the looked-up key: the first positional arg, or the ``key=``
    keyword when the call passes none.

    The keyword form is not hypothetical. ``os.environ`` is ``os._Environ``, a
    Python-level ``MutableMapping``, so its ``get`` / ``pop`` / ``setdefault`` are
    the ABC's plain-Python defs and DO bind ``key=`` — unlike ``dict.get``, whose C
    signature is positional-only and would raise. ``os.getenv(key=...)`` binds for
    the same reason. All four were confirmed against the live interpreter rather
    than assumed, because the dict intuition points the wrong way here."""
    if call.args:
        return call.args[0]
    for kw in call.keywords:
        if kw.arg == "key":
            return kw.value
    return None


def _env_read_key(node: ast.expr | None, aliases: dict[str, str]) -> str | None:
    """The ``BMAD_LOOP_*`` variable an environment-lookup key names, or None.

    No docstring exclusion here, unlike the POSIX-path scan: that one walks *every*
    string Constant in the tree and so must skip prose, but this one only ever
    inspects a key position (a call's first arg, a subscript's slice). A docstring
    is a standalone ``Expr`` statement and can never appear there, so a
    ``BMAD_LOOP_*`` mention in prose produces no finding to exclude. Verified by
    counting key-position nodes that are also docstring nodes across the whole
    tree: zero. An exclusion here would be unreachable code implying a check that
    is not happening.

    Four spellings resolve, because the interesting violation is the *half-right*
    one: someone who reuses the registry's own constant but skips its reader. A
    literal and a same-module alias were never the risky shapes — reaching for
    ``envvars.MUX_BACKEND`` is, precisely because it looks tidy.

    1. ``os.environ.get("BMAD_LOOP_X")``          — string literal
    2. ``os.environ.get(LOCAL)``                  — bound to a literal here
    3. ``os.environ.get(envvars.MUX_BACKEND)``    — qualified registry attribute
    4. ``os.environ.get(MUX_BACKEND)``            — registry constant imported in

    (3) matches on the attribute name alone rather than proving the object is the
    registry module: `import bmad_loop.envvars as ev` / `from . import envvars`
    and a rebound alias all spell it differently, and resolving that statically
    costs more than it buys. A false positive here is a review prompt on a line
    that reads like an env lookup, not a silent miss — the direction a tripwire
    should fail in."""
    if isinstance(node, ast.Constant):
        if isinstance(node.value, str) and node.value.startswith(ENV_PREFIX):
            return node.value
    if isinstance(node, ast.Name):
        # a same-module binding wins over the registry name it may shadow
        return aliases.get(node.id) or REGISTRY_NAMES.get(node.id)
    if isinstance(node, ast.Attribute):
        return REGISTRY_NAMES.get(node.attr)
    return None


def _scan():
    """Single pass over the tree → list of (kind, rel, lineno, line_text)."""
    findings = []
    for path in _py_files():
        rel = _rel(path)
        src = path.read_text(encoding="utf-8")
        lines = src.splitlines()
        tree = ast.parse(src, filename=str(path))
        docs = _docstring_node_ids(tree)
        env_aliases = _env_name_aliases(tree)

        def line_at(lineno: int) -> str:
            return lines[lineno - 1] if 1 <= lineno <= len(lines) else ""

        for node in ast.walk(tree):
            # tmux argv literal: ["tmux", ...] (not the which-list tuple ("tmux", ...))
            if isinstance(node, ast.List) and node.elts:
                first = node.elts[0]
                if isinstance(first, ast.Constant) and first.value == "tmux":
                    findings.append(("tmux", rel, node.lineno, line_at(node.lineno)))

            # bare POSIX path string literal (skip docstrings)
            if (
                isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and id(node) not in docs
                and _classify_posix_path(node.value)
            ):
                findings.append(("path", rel, node.lineno, line_at(node.lineno)))

            # signal.SIGKILL attribute access (the guarded form is a "SIGKILL"
            # *string* passed to getattr — not an attribute access — so it's clean)
            if (
                isinstance(node, ast.Attribute)
                and node.attr == "SIGKILL"
                and isinstance(node.value, ast.Name)
                and node.value.id == "signal"
            ):
                findings.append(("sigkill", rel, node.lineno, line_at(node.lineno)))

            # os.kill(<pid>, 0) — the existence-probe form (signal 0), not a real
            # signal send like os.kill(pid, SIGTERM)
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "kill"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "os"
                and len(node.args) >= 2
                and isinstance(node.args[1], ast.Constant)
                and node.args[1].value == 0
                and node.args[1].value is not False
            ):
                findings.append(("killprobe", rel, node.lineno, line_at(node.lineno)))

            # os.kill(...) in any form — every signal send maps to a destructive
            # TerminateProcess on Windows, so confine the call to the ProcessHost.
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "kill"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "os"
            ):
                findings.append(("oskill", rel, node.lineno, line_at(node.lineno)))

            # start_new_session=True as a call kwarg
            if (
                isinstance(node, ast.keyword)
                and node.arg == "start_new_session"
                and isinstance(node.value, ast.Constant)
                and node.value.value is True
            ):
                findings.append(("detach", rel, node.lineno, line_at(node.lineno)))

            # {"start_new_session": True} as a dict literal (the detach-kwargs form)
            if isinstance(node, ast.Dict):
                for key, val in zip(node.keys, node.values):
                    if (
                        isinstance(key, ast.Constant)
                        and key.value == "start_new_session"
                        and isinstance(val, ast.Constant)
                        and val.value is True
                    ):
                        findings.append(("detach", rel, key.lineno, line_at(key.lineno)))

            # shell=True as a call kwarg
            if (
                isinstance(node, ast.keyword)
                and node.arg == "shell"
                and isinstance(node.value, ast.Constant)
                and node.value.value is True
            ):
                findings.append(("shell", rel, node.lineno, line_at(node.lineno)))

            # A `BMAD_LOOP_*` variable READ out of the process environment:
            # os.environ.get(K) / os.environ.pop(K) / os.getenv(K) / os.environ[K].
            # Reads only — the env dicts modules *build* to inject into a child
            # session are the producing side, which the invariant does not constrain.
            env_key = None
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                func = node.func
                key_node = _env_call_key_node(node)
                if key_node is None:
                    pass
                elif func.attr in ("get", "pop", "setdefault") and _is_os_environ(func.value):
                    env_key = _env_read_key(key_node, env_aliases)
                elif (
                    func.attr == "getenv"
                    and isinstance(func.value, ast.Name)
                    and func.value.id == "os"
                ):
                    env_key = _env_read_key(key_node, env_aliases)
            elif (
                isinstance(node, ast.Subscript)
                and _is_os_environ(node.value)
                and isinstance(node.ctx, ast.Load)
            ):
                env_key = _env_read_key(node.slice, env_aliases)
            if env_key:
                findings.append(("envread", rel, node.lineno, line_at(node.lineno)))

    return findings


FINDINGS = _scan()


def _of(kind: str):
    return [f for f in FINDINGS if f[0] == kind]


def test_no_tmux_invocation_outside_backend():
    """Only the tmux backend may build a ``["tmux", ...]`` argv — every other call
    site goes through the multiplexer seam."""
    offenders = [(rel, ln, txt) for _, rel, ln, txt in _of("tmux") if rel not in TMUX_BACKENDS]
    assert not offenders, (
        "tmux invoked outside the tmux backend (adapters/tmux_base.py, "
        "adapters/tmux_backend.py) — route it through get_multiplexer() instead:\n"
        + "\n".join(f"  {rel}:{ln}: {txt.strip()}" for rel, ln, txt in offenders)
    )


def test_no_hardcoded_posix_paths():
    """No bare ``/tmp`` / ``/proc`` / ``/dev/null`` literal outside the allowlisted
    platform-guarded Unity files; each allowed line carries a `# portability:` ack.
    Use ``os.devnull`` / ``tempfile`` / the psutil fallback instead."""
    bad = []
    for _, rel, ln, txt in _of("path"):
        if rel not in PATH_ALLOW:
            bad.append(f"  {rel}:{ln}: {txt.strip()}  (not an allowlisted file)")
        elif ACK not in txt:
            bad.append(f"  {rel}:{ln}: {txt.strip()}  (missing '{ACK}' ack)")
    assert not bad, "hardcoded POSIX path(s):\n" + "\n".join(bad)


def test_no_unguarded_sigkill():
    """``signal.SIGKILL`` is absent on Windows — reference it only via the
    ``getattr(signal, "SIGKILL", signal.SIGTERM)`` guard, never as a bare
    attribute access."""
    offenders = _of("sigkill")
    assert not offenders, "unguarded signal.SIGKILL attribute access:\n" + "\n".join(
        f"  {rel}:{ln}: {txt.strip()}" for _, rel, ln, txt in offenders
    )


def test_pid_existence_probe_only_in_liveness_helpers():
    """``os.kill(pid, 0)`` is read-only on POSIX but destructive on Windows
    (TerminateProcess) — confine it to the platform-guarded liveness helpers, each
    line carrying a `# portability:` ack. Other call sites route through
    ``platform_util.pid_alive``."""
    bad = []
    for _, rel, ln, txt in _of("killprobe"):
        if rel not in KILL_PROBE_ALLOW:
            bad.append(f"  {rel}:{ln}: {txt.strip()}  (route through platform_util.pid_alive)")
        elif ACK not in txt:
            bad.append(f"  {rel}:{ln}: {txt.strip()}  (missing '{ACK}' ack)")
    assert not bad, "os.kill(pid, 0) outside liveness helpers:\n" + "\n".join(bad)


def test_os_kill_only_in_process_host():
    """Any reachable ``os.kill`` maps to a destructive TerminateProcess on Windows —
    confine it to ``process_host.py``. Detects the literal ``os.kill(`` form only;
    import aliases and assigned aliases are deliberately not tracked — this is a
    review tripwire, not a sandbox. Other call sites route through the ProcessHost
    seam (``terminate`` / ``force_kill`` / ``is_alive``)."""
    offenders = [(rel, ln, txt) for _, rel, ln, txt in _of("oskill") if rel not in OS_KILL_ALLOW]
    assert (
        not offenders
    ), "os.kill( outside process_host.py — route it through the ProcessHost seam:\n" + "\n".join(
        f"  {rel}:{ln}: {txt.strip()}" for rel, ln, txt in offenders
    )


def test_start_new_session_only_in_detach_helpers():
    """``start_new_session=True`` is POSIX-only — confine it to the detach helpers
    (which branch on ``sys.platform``), each line carrying a `# portability:` ack."""
    bad = []
    for _, rel, ln, txt in _of("detach"):
        if rel not in DETACH_ALLOW:
            bad.append(f"  {rel}:{ln}: {txt.strip()}  (not a detach helper)")
        elif ACK not in txt:
            bad.append(f"  {rel}:{ln}: {txt.strip()}  (missing '{ACK}' ack)")
    assert not bad, "start_new_session=True outside detach helpers:\n" + "\n".join(bad)


def test_shell_true_only_in_sanctioned_spots():
    """``shell=True`` only in the two operator-authored-command spots, each line
    carrying a `# portability:` ack."""
    bad = []
    for _, rel, ln, txt in _of("shell"):
        if rel not in SHELL_ALLOW:
            bad.append(f"  {rel}:{ln}: {txt.strip()}  (not a sanctioned shell spot)")
        elif ACK not in txt:
            bad.append(f"  {rel}:{ln}: {txt.strip()}  (missing '{ACK}' ack)")
    assert not bad, "shell=True outside verify.py / plugins/bus.py:\n" + "\n".join(bad)


def test_bmad_loop_env_reads_only_in_the_registry():
    """AGENTS.md's env invariant, enforced: "New core env vars register in
    ``envvars.py``; plugin-owned env-var families stay with their plugin." Reading a
    knob inline is what made these undiscoverable before the registry existed, so a
    core module must call an ``envvars`` reader rather than touch ``os.environ``
    itself. Detects the literal ``os.environ`` / ``os.getenv`` forms with the key
    spelled four ways — literal, same-module constant, ``envvars.MUX_BACKEND``, and
    the registry constant imported in (see ``_env_read_key``). ``from os import
    environ`` aliases are deliberately not tracked: that is a way to obscure the
    *lookup*, and this is a review tripwire, not a sandbox. Borrowing the registry's
    *key* is different — it is the shape a well-meaning change actually takes, so it
    resolves.

    Scoped to reads. Writes are a different act: engine.py, resolve.py, probe.py,
    plugins/bus.py and unity_plugin.py all BUILD a ``BMAD_LOOP_*`` dict to inject
    into a child session, and gates.py hands notify text to osascript/PowerShell the
    same way — all producing side, none of it a second place a var is *defined*.
    Reads of a SessionSpec's ``spec.env`` (adapters/generic.py) are likewise out:
    that is a plain dict handed down in-process, not the environment.

    Ablated 2026-08-11, six ways — a negative assertion passes for every reason a
    value could be absent, including a branch that never fires:
    - emptying ENV_READ_ALLOW fails listing all 52 real reads across the 10
      ENV_READ_ALLOW files, so the scan sees them rather than passing on an empty
      finding set;
    - ``_ABLATION_TIMEOUT = os.environ.get("BMAD_LOOP_SOMETHING")`` planted at module
      scope in verify.py (core, not allowlisted) fails with
      ``verify.py:37: _ABLATION_TIMEOUT = os.environ.get("BMAD_LOOP_SOMETHING")``;
    - the alias arm is live, not decoration: the same read spelled
      ``_ABLATION_ENV = "BMAD_LOOP_SOMETHING"`` / ``os.environ.get(_ABLATION_ENV)``
      fails too — drop ``_env_name_aliases`` and it passes, and so would every read
      in envvars.py, which is the only shape the registry itself uses;
    - the name written into a non-allowlisted module's docstring instead stays green.
      ⚠️ That row is a CONTROL, not an ablation: it passes because a prose mention
      creates no key-position node at all, so it would stay green no matter what the
      scan did. An earlier revision carried a docstring exclusion inside
      ``_env_read_key`` and cited this row as proof it worked; counting
      key-position nodes that are also docstring nodes across the whole tree gives
      zero, so that exclusion was unreachable and the row graded nothing. The
      exclusion is gone — keep the row for the property, not as evidence.

    The last two close a hole this guard SHIPPED WITH and a review caught (both
    verified by planting in verify.py, and both passed green before REGISTRY_NAMES
    existed — the guard asserted an invariant it did not enforce):
    - ``from . import envvars as _ev`` / ``os.environ.get(_ev.MUX_BACKEND)`` now
      fails with ``verify.py:13: _LEAK1 = os.environ.get(_ev.MUX_BACKEND)``;
    - ``from .envvars import MUX_BACKEND`` / ``os.environ.get(MUX_BACKEND)`` now
      fails with ``verify.py:13: _LEAK2 = os.environ.get(MUX_BACKEND)``.
    Both spellings borrow the registry's constant while skipping its reader, which
    is a *more* likely violation than an inline literal, not a more exotic one —
    the tidy-looking version is the one that gets written.

    A later review round closed a third hole the same way: the key passed as a
    KEYWORD. ``os.getenv(key="BMAD_LOOP_X")`` and ``os.environ.get(key=...)`` /
    ``.pop(key=...)`` / ``.setdefault(key=...)`` all leave ``node.args`` empty, and
    the scan required a positional arg — so each read green. All four bind ``key=``
    for real (``os.environ`` is ``os._Environ``, a Python-level MutableMapping, so
    its methods are the ABC's plain-Python defs; the ``dict.get`` intuition, which
    is positional-only, points the wrong way). Confirmed against the live
    interpreter before fixing, then re-verified as CAUGHT, including the keyword
    form carrying an imported registry constant."""
    offenders = [(rel, ln, txt) for _, rel, ln, txt in _of("envread") if rel not in ENV_READ_ALLOW]
    assert not offenders, (
        "BMAD_LOOP_* read outside envvars.py and the plugin-owned families — name "
        "the var in envvars.py and call its reader instead of widening the "
        "allowlist:\n" + "\n".join(f"  {rel}:{ln}: {txt.strip()}" for rel, ln, txt in offenders)
    )


def test_guard_actually_scanned_files():
    """Sanity: the scan walked a non-trivial number of files (catches a broken
    SRC root silently passing every assertion)."""
    assert len(_py_files()) > 20
