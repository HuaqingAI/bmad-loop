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

import pytest

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
    """True for the ``os.environ`` / ``os.environb`` attribute access itself.

    ``environb`` is the bytes-keyed twin (POSIX-only, absent on Windows). Nobody
    reaches for it here, but it is the same mapping and costs one string to cover,
    which is cheaper than discovering it later."""
    return (
        isinstance(node, ast.Attribute)
        and node.attr in ("environ", "environb")
        and isinstance(node.value, ast.Name)
        and node.value.id == "os"
    )


def _env_name_aliases(tree: ast.AST) -> dict[str, str]:
    """``NAME -> "BMAD_LOOP_…"`` for every constant binding in the module, so a read
    spelled through a named constant still resolves. That indirection is the norm
    here, not an edge case: the registry reads ``os.environ.get(MUX_BACKEND)`` and
    gates.py names its notify vars ``_TITLE_ENV`` / ``_MESSAGE_ENV`` — matching the
    string literal alone would miss exactly the well-behaved shape.

    A ``bytes`` constant binds too, since ``os.environb`` can only be keyed by
    bytes: the registry's own constants are ``str`` and would raise there, so a
    bytes literal or a bytes constant are the only two spellings that axis has."""
    aliases: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            targets = [t for t in node.targets if isinstance(t, ast.Name)]
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            targets = [node.target]
        else:
            continue
        value = node.value
        if not targets or not isinstance(value, ast.Constant):
            continue
        name = value.value
        if isinstance(name, bytes):
            name = name.decode("utf-8", "replace")
        if isinstance(name, str) and name.startswith(ENV_PREFIX):
            for target in targets:
                aliases[target.id] = name
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
        # bytes ride along for os.environb's b"BMAD_LOOP_…" keys
        if isinstance(node.value, bytes):
            decoded = node.value.decode("utf-8", "replace")
            return decoded if decoded.startswith(ENV_PREFIX) else None
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
        findings.extend(_scan_source(path.read_text(encoding="utf-8"), _rel(path)))
    return findings


def _scan_source(src: str, rel: str):
    """The whole per-file scan, over one source string → the same
    ``(kind, rel, lineno, line_text)`` tuples ``_scan`` collects.

    Split out from ``_scan`` so the detectors can be driven by a snippet and not
    only by what happens to be in the tree today. A repo-wide "nothing is flagged"
    assertion is green both when the invariant holds and when the detector has
    quietly stopped detecting; the probes below feed known-bad sources through
    THIS function — the same code path the real scan uses — so the two failure
    modes stop being indistinguishable."""
    findings = []
    lines = src.splitlines()
    tree = ast.parse(src, filename=rel)
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
                # `getenvb` is the bytes twin, and POSIX-only like `environb`
                func.attr in ("getenv", "getenvb")
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
        elif isinstance(node, ast.Compare):
            # `"BMAD_LOOP_X" in os.environ` / `not in` — a presence read, and the
            # most natural way to spell a boolean flag. A chain expands PAIRWISE
            # (`c == K in os.environ` means `c == K and K in os.environ`), so the
            # operand a membership tests is the one to its immediate left — the
            # PRECEDING comparator, not `node.left`, for any op past the first.
            # Carry the left operand across the pairs rather than re-reading
            # `node.left`, which resolves the wrong name on a chain.
            left = node.left
            for op, rhs in zip(node.ops, node.comparators):
                if isinstance(op, (ast.In, ast.NotIn)) and _is_os_environ(rhs):
                    env_key = _env_read_key(left, env_aliases)
                    if env_key:
                        break
                left = rhs
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
    itself.

    COVERED — the access shape: ``os.environ.get/pop/setdefault``, ``os.getenv``,
    ``os.environ[K]``, the ``key=`` keyword form of each, and ``K in os.environ`` /
    ``not in`` (including as a link in a chained comparison). Each has a POSIX-only
    bytes twin — ``os.environb`` for the mapping forms, ``os.getenvb`` for the
    function — and both twins are covered.

    Crossed with the key spelling: a literal, a same-module constant,
    ``envvars.MUX_BACKEND``, and the registry constant imported in (see
    ``_env_read_key``). Borrowing the registry's own constant while skipping its
    reader is the *likeliest* violation rather than an exotic one — the
    tidy-looking version is the one that gets written — so every key spelling
    resolves. The bytes twins take only the first two: their keys must be bytes,
    and the registry's constants are ``str``, so that half of the cross product
    cannot be written at all rather than being an uncovered case.

    ``ENV_READ_PROBES`` / ``ENV_READ_NON_PROBES`` are that matrix, executable. Read
    them for what is covered; this docstring only argues the boundary.

    NOT COVERED, deliberately — both obscure the *lookup* rather than the key:
    rebinding the mapping (``e = os.environ; e.get(K)``) and ``from os import
    environ``. This is a review tripwire, not a sandbox: it exists to catch the
    change someone writes while trying to do the right thing, not to withstand
    someone routing around it.

    Bulk copies (``dict(os.environ)``, ``{**os.environ}``) are correctly silent:
    they name no variable, so there is no var being defined outside the registry.

    Scoped to reads. Writes are a different act: engine.py, resolve.py, probe.py,
    plugins/bus.py and unity_plugin.py all BUILD a ``BMAD_LOOP_*`` dict to inject
    into a child session, and gates.py hands notify text to osascript/PowerShell the
    same way — all producing side, none of it a second place a var is *defined*.
    Reads of a SessionSpec's ``spec.env`` (adapters/generic.py) are likewise out:
    that is a plain dict handed down in-process, not the environment.

    ⚠️ THIS assertion cannot grade the detector. It says only that today's tree
    carries no unallowlisted finding — equally green when the scan has silently
    stopped scanning. Delete any single branch of the ``envread`` detector and this
    test still passes while exactly the matching ``ENV_READ_PROBES`` rows redden.
    The assertion is the invariant; the probes are the proof it is being checked,
    and neither replaces the other. What this test does grade alone is the
    allowlist: empty ``ENV_READ_ALLOW`` and it fails naming every real read, so a
    green run means the scan saw those reads rather than that it found nothing.

    ⚠️ The prose row in ``ENV_READ_NON_PROBES`` is a CONTROL, not an ablation. A
    ``BMAD_LOOP_*`` mention in a docstring creates no key-position node at all, so
    it stays silent no matter what the detector does — an earlier revision cited it
    as proof of a docstring exclusion in ``_env_read_key`` that was in fact
    unreachable. Keep the row for the property, never as evidence.

    ⚠️ New uncovered shapes keep surfacing here, and that is a property of the
    design rather than a run of bad luck: this is a denylist of access forms, so it
    is only ever as complete as the last sweep over them, and the NOT COVERED list
    is the honest boundary rather than an oversight. Sweep an axis when you touch
    it — every mapping form at once, not the one that prompted the visit — and
    extend the matrix before the branch: add the probe row, watch it fail, then fix
    the scan."""
    offenders = [(rel, ln, txt) for _, rel, ln, txt in _of("envread") if rel not in ENV_READ_ALLOW]
    assert not offenders, (
        "BMAD_LOOP_* read outside envvars.py and the plugin-owned families — name "
        "the var in envvars.py and call its reader instead of widening the "
        "allowlist:\n" + "\n".join(f"  {rel}:{ln}: {txt.strip()}" for rel, ln, txt in offenders)
    )


# Every access form the env-read detector claims to cover, as a source snippet that
# MUST produce an `envread` finding. These are the executable half of the matrix the
# test above documents: that test asserts only that today's tree is clean, which stays
# green both when the invariant holds and when the detector has silently stopped
# detecting. Driving known-bad sources through the real `_scan_source` separates
# those. Snippets are parsed, never imported, so a nonexistent relative import is fine.
# Fix order when a new form turns up: add the row here FIRST and watch it fail.
ENV_READ_PROBES = [
    ("get-literal", 'import os\nX = os.environ.get("BMAD_LOOP_X")\n'),
    ("get-local-const", 'import os\nK = "BMAD_LOOP_X"\nX = os.environ.get(K)\n'),
    (
        "get-qualified-registry",
        "import os\nfrom . import envvars\nX = os.environ.get(envvars.MUX_BACKEND)\n",
    ),
    (
        "get-aliased-registry",
        "import os\nfrom . import envvars as ev\nX = os.environ.get(ev.MUX_BACKEND)\n",
    ),
    (
        "get-imported-registry",
        "import os\nfrom .envvars import MUX_BACKEND\nX = os.environ.get(MUX_BACKEND)\n",
    ),
    ("getenv", 'import os\nX = os.getenv("BMAD_LOOP_X")\n'),
    ("subscript", 'import os\ndef f():\n    return os.environ["BMAD_LOOP_X"]\n'),
    ("getenv-keyword", 'import os\nX = os.getenv(key="BMAD_LOOP_X")\n'),
    ("get-keyword", 'import os\nX = os.environ.get(key="BMAD_LOOP_X")\n'),
    ("pop-keyword", 'import os\ndef f():\n    return os.environ.pop(key="BMAD_LOOP_X")\n'),
    ("setdefault-keyword", 'import os\nX = os.environ.setdefault(key="BMAD_LOOP_X", value="v")\n'),
    ("membership-in", 'import os\nX = "BMAD_LOOP_X" in os.environ\n'),
    ("membership-not-in", 'import os\nX = "BMAD_LOOP_X" not in os.environ\n'),
    (
        "membership-registry",
        "import os\nfrom .envvars import MUX_BACKEND\nX = MUX_BACKEND in os.environ\n",
    ),
    # A chain, where the membership's left operand is the PRECEDING comparator.
    # Keyed by a literal on purpose: the registry row above already grades key
    # resolution, so this row reddens for one reason only — chain position.
    ("membership-chained", 'import os\nc = "x"\nX = c == "BMAD_LOOP_X" in os.environ\n'),
    # The `os.environb` axis, swept rather than sampled: every mapping form above
    # has a bytes twin, and `os.getenvb` is the twin of `os.getenv`. Keys here are a
    # bytes literal or a bytes constant, which is the whole spelling axis — the
    # registry's constants are `str` and would raise against a bytes mapping.
    ("environb-get", 'import os\nX = os.environb.get(b"BMAD_LOOP_X")\n'),
    ("environb-subscript", 'import os\ndef f():\n    return os.environb[b"BMAD_LOOP_X"]\n'),
    ("environb-local-const", 'import os\nK = b"BMAD_LOOP_X"\nX = os.environb.get(K)\n'),
    ("environb-pop-keyword", 'import os\nX = os.environb.pop(key=b"BMAD_LOOP_X")\n'),
    ("environb-setdefault", 'import os\nX = os.environb.setdefault(b"BMAD_LOOP_X", b"v")\n'),
    ("environb-membership", 'import os\nX = b"BMAD_LOOP_X" in os.environb\n'),
    (
        "environb-membership-chained",
        'import os\nc = b"y"\nX = c == b"BMAD_LOOP_X" in os.environb\n',
    ),
    ("getenvb", 'import os\nX = os.getenvb(b"BMAD_LOOP_X")\n'),
    ("getenvb-keyword", 'import os\nX = os.getenvb(key=b"BMAD_LOOP_X")\n'),
]

# The other half: shapes that must stay SILENT. Without these the detector could pass
# every probe above by flagging everything, which would be just as broken — a guard
# that cries wolf gets its allowlist widened until it means nothing.
ENV_READ_NON_PROBES = [
    ("bulk-dict-copy", "import os\nX = dict(os.environ)\n"),
    ("bulk-splat-copy", "import os\nX = {**os.environ}\n"),
    ("bulk-copy-method", "import os\nX = os.environ.copy()\n"),
    ("foreign-var", 'import os\nX = os.environ.get("PATH")\n'),
    (
        "prose-in-docstring",
        'import os\ndef f():\n    """Injects BMAD_LOOP_X downstream."""\n    return 1\n',
    ),
    ("write-not-read", 'import os\nos.environ["BMAD_LOOP_X"] = "1"\n'),
    ("session-spec-env", 'def f(spec):\n    return spec.env.get("BMAD_LOOP_X")\n'),
]


@pytest.mark.parametrize(("label", "source"), ENV_READ_PROBES, ids=[p[0] for p in ENV_READ_PROBES])
def test_env_read_detector_flags_every_claimed_access_form(label, source):
    """Each documented access form really does produce a finding.

    This is the check the repo-wide assertion cannot be: delete any single branch of
    the `envread` scan and the tree-wide test stays green (nothing in `src/` uses that
    branch today), while exactly the matching row here reddens. The coverage claim
    lives here rather than in the guard's docstring alone, because prose does not
    fail a build."""
    found = [f for f in _scan_source(source, "probe.py") if f[0] == "envread"]
    assert found, (
        f"the {label!r} access form produced no `envread` finding — the detector does "
        f"not cover a shape the guard's docstring claims:\n{source}"
    )


@pytest.mark.parametrize(
    ("label", "source"), ENV_READ_NON_PROBES, ids=[p[0] for p in ENV_READ_NON_PROBES]
)
def test_env_read_detector_stays_silent_on_non_reads(label, source):
    """The complement: a bulk environment copy, a foreign variable, prose, a WRITE,
    and a `SessionSpec.env` lookup are all silent. Pins the scoping decisions the
    guard's docstring argues for, so narrowing or widening the detector has to be
    deliberate — and stops a future fix from passing the probes by flagging
    everything."""
    found = [f for f in _scan_source(source, "probe.py") if f[0] == "envread"]
    assert not found, (
        f"the {label!r} shape was flagged as an env read; it is deliberately out of "
        f"scope:\n{source}"
    )


def test_guard_actually_scanned_files():
    """Sanity: the scan walked a non-trivial number of files (catches a broken
    SRC root silently passing every assertion)."""
    assert len(_py_files()) > 20
