"""Coding-CLI adapter-registry selection + discovery proof.

The CLI axis selects its adapter *class* through a registry
(:func:`~bmad_loop.adapters.registry.register_adapter`) keyed on the
``profile.adapter`` field, rather than name-branching in
``runsetup.make_adapters``. So a new adapter family is a registration — not a core
edit — exactly as a new transport backend is
(:mod:`bmad_loop.adapters.multiplexer`). These tests pin the registry (builtin +
external registration, builtins-first-wins, unknown-kind fail-loud), the
out-of-tree entry-point discovery and its degrade-not-crash contract, and the
``runsetup.make_adapters`` dispatch: a registered kind builds, the
``(cfg, synthesizes)`` cache shares instances, ``needs_mux`` gates the transport
resolution, a construction failure becomes a clean ``SystemExit``, the
digest-gated ``profiles=`` mapping decides the kind, and — the regression pin —
``opencode-http`` still dispatches to the HTTP adapters unchanged.

The registry is deliberately simpler than the multiplexer seam: adapters are built
per-run (``runsetup.make_adapters`` owns the cache), so there is no ``lru_cache`` /
``cache_clear`` and no ``configure_*`` / platform-default machinery — the
``fresh_adapter_registry`` fixture snapshots only ``_ADAPTERS`` / the two loaded
flags / ``_EXTERNAL_ERRORS``.

Entry points are faked by monkeypatching ``importlib.metadata.entry_points``
through the ``registry`` module's own binding; one test builds a real
``*.dist-info`` on ``sys.path`` to prove the scan works against genuine packaging
metadata.
"""

from __future__ import annotations

import argparse
import json

import pytest
from conftest import install_bmad_config, write_sprint

from bmad_loop import cli
from bmad_loop import policy as policy_mod
from bmad_loop import runsetup
from bmad_loop.adapters import multiplexer as mux_mod
from bmad_loop.adapters import profile as profile_mod
from bmad_loop.adapters import registry as m
from bmad_loop.adapters.profile import CLIProfile, HookSpec
from bmad_loop.adapters.registry import AdapterBuilder, AdapterError

# --------------------------------------------------------------------------- #
# Stubs + fixtures


class _StubAdapter:
    """Plain-variant double: records the kwargs make_adapters passes so a test can
    assert the mux was (or was not) threaded in."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.profile = kwargs.get("profile")


class _StubDevAdapter(_StubAdapter):
    """Dev-variant double: mirrors the real ``(*args, paths, **kwargs)`` dev
    __init__ contract every registered family's dev class honors."""

    def __init__(self, *, paths, **kwargs):
        super().__init__(**kwargs)
        self.paths = paths


def _stub_builder(*, construct_error=()):
    return AdapterBuilder(plain=_StubAdapter, dev=_StubDevAdapter, construct_error=construct_error)


class _FakeDist:
    """Stands in for ``EntryPoint.dist``; the scan orders on its ``.name``."""

    def __init__(self, name):
        self.name = name


class _FakeEntryPoint:
    """Duck-typed importlib.metadata.EntryPoint: the loader touches ``.name``,
    ``.dist`` (the scan's tiebreak — see `_load_external_adapters`) and
    ``.load()``. ``dist`` defaults to a distinct-per-name stand-in so the ordering
    of same-named entries is only ever decided by a test that sets it."""

    def __init__(self, name, load, dist=None):
        self.name = name
        self.dist = _FakeDist(dist if dist is not None else f"{name}-dist")
        self._load = load

    def load(self):
        return self._load()


@pytest.fixture
def fresh_adapter_registry():
    """Isolate the global adapter registry: snapshot, clear, restore. No
    lru_cache / configured-choice to reset — the deliberate asymmetry vs. the mux
    seam. The externals scan is parked as already-loaded so whatever adapters are
    installed on the dev box can't leak into builtin tests (discovery tests re-arm
    it via :func:`scan_adapter_registry`). The companion profile-module external
    scan is suppressed too, so an installed ``bmad_loop.profiles`` package can't
    leak a profile into the ``make_adapters`` integration tests."""
    saved_adapters = dict(m._ADAPTERS)
    saved_loaded = m._BUILTINS_LOADED
    saved_ext_loaded = m._EXTERNALS_LOADED
    saved_ext_errors = dict(m._EXTERNAL_ERRORS)
    m._ADAPTERS.clear()
    m._BUILTINS_LOADED = False
    m._EXTERNALS_LOADED = True  # externals OFF by default; discovery tests opt back in
    m._EXTERNAL_ERRORS.clear()

    saved_prof_loaded = profile_mod._EXTERNALS_LOADED
    saved_prof_profiles = dict(profile_mod._EXTERNAL_PROFILES)
    saved_prof_errors = dict(profile_mod._PROFILE_LOAD_ERRORS)
    profile_mod._EXTERNALS_LOADED = True
    profile_mod._EXTERNAL_PROFILES.clear()
    profile_mod._PROFILE_LOAD_ERRORS.clear()

    yield m

    m._ADAPTERS.clear()
    m._ADAPTERS.update(saved_adapters)
    m._BUILTINS_LOADED = saved_loaded
    m._EXTERNALS_LOADED = saved_ext_loaded
    m._EXTERNAL_ERRORS.clear()
    m._EXTERNAL_ERRORS.update(saved_ext_errors)
    profile_mod._EXTERNALS_LOADED = saved_prof_loaded
    profile_mod._EXTERNAL_PROFILES.clear()
    profile_mod._EXTERNAL_PROFILES.update(saved_prof_profiles)
    profile_mod._PROFILE_LOAD_ERRORS.clear()
    profile_mod._PROFILE_LOAD_ERRORS.update(saved_prof_errors)


@pytest.fixture
def scan_adapter_registry(fresh_adapter_registry, monkeypatch):
    """fresh_adapter_registry with the externals scan re-armed. Yields a hook:
    call it with fake entry points (or a ``scan_error`` to raise from the scan
    itself) and the next resolution performs that scan."""

    def arm(*eps, scan_error=None):
        def fake_entry_points(*, group):
            assert group == m.ADAPTERS_GROUP
            if scan_error is not None:
                raise scan_error
            return list(eps)

        monkeypatch.setattr(m.importlib.metadata, "entry_points", fake_entry_points)
        m._EXTERNALS_LOADED = False
        m._EXTERNAL_ERRORS.clear()

    yield fresh_adapter_registry, arm


def _write_policy(project, text) -> None:
    d = project / ".bmad-loop"
    d.mkdir(parents=True, exist_ok=True)
    (d / "policy.toml").write_text(text, encoding="utf-8")


def _write_profile(project, name, *, adapter, hookless=True) -> None:
    d = project / ".bmad-loop" / "profiles"
    d.mkdir(parents=True, exist_ok=True)
    if hookless:
        hooks = '[hooks]\ndialect = "none"\n'
    else:
        hooks = (
            '[hooks]\ndialect = "claude-settings-json"\n'
            'config_path = ".hermes/settings.json"\n[hooks.events]\nStop = "Stop"\n'
        )
    (d / f"{name}.toml").write_text(
        f'name = "{name}"\nbinary = "{name}"\nadapter = "{adapter}"\n{hooks}', encoding="utf-8"
    )


def _run_dir(project):
    return project / ".bmad-loop" / "runs" / "r"


# --------------------------------------------------------------------------- #
# Registry — builtin registration + fail-loud


def test_builtin_kinds_registered_with_needs_mux(fresh_adapter_registry):
    """The two bundled kinds register with the correct transport requirement:
    generic drives tmux (needs_mux), opencode-http is hookless HTTP/SSE (does not)."""
    generic = fresh_adapter_registry.get_adapter_kind("generic")
    http = fresh_adapter_registry.get_adapter_kind("opencode-http")
    assert generic.needs_mux is True
    assert http.needs_mux is False
    assert fresh_adapter_registry.known_adapter_kinds() == ["generic", "opencode-http"]


def test_builtin_name_constants_match_the_registered_names(fresh_adapter_registry):
    """`validate`'s httpx check keys on the GENERIC/OPENCODE_HTTP constants rather
    than on literals. Pin them to what actually registers, so a rename cannot make
    that check silently stop firing — an absent finding reads as a pass."""
    assert fresh_adapter_registry.GENERIC == "generic"
    assert fresh_adapter_registry.OPENCODE_HTTP == "opencode-http"
    assert set(fresh_adapter_registry.known_adapter_kinds()) == {
        fresh_adapter_registry.GENERIC,
        fresh_adapter_registry.OPENCODE_HTTP,
    }


def test_load_thunk_returns_real_builder(fresh_adapter_registry):
    """The lazy thunk resolves to the real adapter classes — imported only now,
    never at registration/listing time."""
    from bmad_loop.adapters.generic import GenericAdapter, GenericDevAdapter

    builder = fresh_adapter_registry.get_adapter_kind("generic").load()
    assert builder.plain is GenericAdapter
    assert builder.dev is GenericDevAdapter
    assert builder.construct_error == ()


def test_unknown_kind_fails_loud_naming_known(fresh_adapter_registry):
    """An explicit but unregistered kind is a misconfiguration: fail loud, listing
    the registered kinds — never silently fall back."""
    with pytest.raises(AdapterError, match=r"nonesuch.*generic.*opencode-http"):
        fresh_adapter_registry.get_adapter_kind("nonesuch")


def test_register_adapter_first_wins(fresh_adapter_registry):
    """A duplicate registration of a name is ignored (first wins) — the mechanism
    that lets builtins load first and shrug off a same-named external."""
    registry = fresh_adapter_registry
    registry.register_adapter("dup", needs_mux=True, load=lambda: _stub_builder())
    registry.register_adapter("dup", needs_mux=False, load=lambda: _stub_builder())
    assert registry.get_adapter_kind("dup").needs_mux is True


def test_detect_adapters_labels_builtin(fresh_adapter_registry):
    """detect_adapters lists every kind, sorted, labelled builtin-vs-external,
    without invoking any load thunk (no heavy import)."""
    registry = fresh_adapter_registry
    registry.register_adapter("extra", needs_mux=False, load=lambda: _stub_builder())
    rows = {r.name: r for r in registry.detect_adapters()}
    assert set(rows) == {"generic", "opencode-http", "extra"}
    assert rows["generic"].builtin is True and rows["generic"].needs_mux is True
    assert rows["opencode-http"].builtin is True and rows["opencode-http"].needs_mux is False
    assert rows["extra"].builtin is False
    assert [r.name for r in registry.detect_adapters()] == sorted(rows)


# --------------------------------------------------------------------------- #
# External discovery (the bmad_loop.adapters entry-point scan)


def test_entry_point_adapter_registers_and_is_selectable(scan_adapter_registry):
    """The pip-install-and-go path: the entry point's module import registers the
    kind; it resolves and lists like a builtin, with no load error."""
    registry, arm = scan_adapter_registry

    def load():
        registry.register_adapter("extadapter", needs_mux=False, load=lambda: _stub_builder())

    arm(_FakeEntryPoint("extadapter", load))
    kind = registry.get_adapter_kind("extadapter")
    assert kind.needs_mux is False
    assert "extadapter" in {r.name for r in registry.detect_adapters()}
    assert registry.external_adapter_errors() == {}


def test_builtins_first_wins_over_external(scan_adapter_registry):
    """Ordering guarantee: builtins register before the scan, so an external that
    tries to register the bundled name ``generic`` cannot shadow it."""
    registry, arm = scan_adapter_registry

    def load():
        # a hostile/clumsy external claiming the builtin name with wrong needs_mux
        registry.register_adapter("generic", needs_mux=False, load=lambda: _stub_builder())

    arm(_FakeEntryPoint("shadow", load))
    kind = registry.get_adapter_kind("generic")
    assert kind.needs_mux is True  # the builtin, not the external
    assert kind.load().plain.__name__ == "GenericAdapter"


def test_builtins_win_over_an_external_imported_by_the_profile_scan(
    fresh_adapter_registry, monkeypatch
):
    """The shadowing hole that first-wins ALONE does not close.

    The documented packaging layout puts both entry points in one module, so the
    ``bmad_loop.profiles`` scan imports it too — and that scan runs on any
    ``load_profiles`` call, which every command makes long before a kind is ever
    resolved. The external's import-time ``register_adapter`` therefore lands
    before ``_load_builtin_adapters`` would have run, and ``setdefault`` keeps it
    under the bundled name: every default profile silently redirects to a
    third-party class. Only ``register_adapter`` seeding the builtins on its own
    side makes first-wins an invariant instead of an ordering coincidence.

    ABLATION: drop the ``_load_builtin_adapters()`` call from ``register_adapter``
    and this reddens — while ``test_builtins_first_wins_over_external`` above stays
    green, because ``get_adapter_kind`` seeds the builtins on that path in."""
    registry = fresh_adapter_registry

    def provider():
        # Any loadable profile does — it exists so the scan has something to
        # accept. The shadowing is the entry point's import side effect below,
        # not anything about this profile. (Its kind is deliberately not
        # `generic`: hookless + `generic` is refused as incoherent.)
        return [
            CLIProfile(name="ext", binary="ext", adapter="ext-http", hooks=HookSpec("none", "", {}))
        ]

    def ep_load():
        # The import side effect of a module carrying BOTH entry points: a clumsy
        # (or hostile) external claiming the builtin name with wrong needs_mux.
        registry.register_adapter("generic", needs_mux=False, load=lambda: _stub_builder())
        return provider

    def fake_entry_points(*, group):
        assert group == profile_mod.PROFILES_GROUP
        return [_FakeEntryPoint("dual", ep_load)]

    monkeypatch.setattr(profile_mod.importlib.metadata, "entry_points", fake_entry_points)
    profile_mod._EXTERNALS_LOADED = False  # re-arm the scan the fixture parks

    # The real process order: profiles resolve first (cmd_adapters and cmd_run
    # both do), and nothing has touched the adapter registry yet.
    assert "ext" in profile_mod.load_profiles(None)
    assert registry._ADAPTERS, "the external registered — otherwise this proves nothing"

    kind = registry.get_adapter_kind("generic")
    assert kind.needs_mux is True  # the builtin, not the external
    assert kind.load().plain.__name__ == "GenericAdapter"


def test_broken_entry_point_degrades_and_is_recorded(scan_adapter_registry):
    """A distribution whose import blows up must not break selection: the builtins
    still resolve, and the failure is recorded for adapters/validate to show."""
    registry, arm = scan_adapter_registry

    def boom():
        raise ImportError("No module named 'ghost_dependency'")

    arm(_FakeEntryPoint("brokenadapter", boom))
    assert registry.get_adapter_kind("generic").needs_mux is True  # selection still works
    errors = registry.external_adapter_errors()
    assert list(errors) == ["brokenadapter"]
    assert "ghost_dependency" in errors["brokenadapter"]


def test_one_broken_package_does_not_hide_the_rest(scan_adapter_registry):
    """Per-entry isolation: the loader keeps importing after a failure, so a
    working adapter still registers alongside a broken one."""
    registry, arm = scan_adapter_registry

    def boom():
        raise RuntimeError("half-installed")

    def load():
        registry.register_adapter("goodadapter", needs_mux=True, load=lambda: _stub_builder())

    arm(_FakeEntryPoint("brokenadapter", boom), _FakeEntryPoint("goodadapter", load))
    assert registry.get_adapter_kind("goodadapter").needs_mux is True
    assert list(registry.external_adapter_errors()) == ["brokenadapter"]


def test_scan_failure_degrades(scan_adapter_registry):
    """Even the entry-point enumeration itself blowing up leaves selection working,
    with the scan failure recorded."""
    registry, arm = scan_adapter_registry
    arm(scan_error=RuntimeError("metadata index corrupt"))
    assert registry.get_adapter_kind("generic").needs_mux is True
    assert "<entry-point scan>" in registry.external_adapter_errors()


def test_scan_runs_once_per_process(scan_adapter_registry):
    """The loaded-flag is set up front: a second resolution does not re-scan (a
    third-party import failure is not transient; re-importing would re-fail)."""
    registry, arm = scan_adapter_registry
    calls = []

    def load():
        calls.append(1)

    arm(_FakeEntryPoint("extadapter", load))
    registry.known_adapter_kinds()
    registry.known_adapter_kinds()
    assert len(calls) == 1


@pytest.mark.parametrize(
    "order", [("alpha", "zeta"), ("zeta", "alpha")], ids=["alpha-discovered-first", "zeta-first"]
)
def test_same_named_entry_points_resolve_by_distribution_not_install_order(
    scan_adapter_registry, order
):
    """Two distributions may advertise the SAME entry-point name in one group —
    `entry_points(group=...)` does not dedup across distributions — and a package
    conventionally names its entry point after the kind it registers, so packages
    colliding on a kind normally arrive as a NAME collision too. `sorted` is
    stable, so a name-only key resolves that tie in distribution-discovery order,
    which is `sys.path` order: the very same two packages would then pick
    different winners on two hosts. Ordering on the distribution as well is what
    makes first-wins a fact about the packages.

    Both parameters arm the identical pair and differ only in the order the scan
    yields them; `alpha-adapter` must win either way.

    ABLATION: drop the `getattr(e.dist, ...)` half of the sort key in
    `_load_external_adapters` and the `zeta-first` case reddens (needs_mux True)
    while `alpha-discovered-first` stays green — which is the finding: the
    name-only key is right only when the install happens to agree with it."""
    registry, arm = scan_adapter_registry

    def register(needs_mux):
        def load():
            registry.register_adapter("acme", needs_mux=needs_mux, load=lambda: _stub_builder())

        return load

    eps = {
        "alpha": _FakeEntryPoint("acme", register(False), dist="alpha-adapter"),
        "zeta": _FakeEntryPoint("acme", register(True), dist="zeta-adapter"),
    }
    arm(*(eps[k] for k in order))

    assert registry.get_adapter_kind("acme").needs_mux is False  # alpha-adapter's


def test_real_dist_info_metadata_is_discovered(fresh_adapter_registry, monkeypatch, tmp_path):
    """End-to-end against genuine packaging metadata: a real ``*.dist-info`` +
    module on sys.path is found by the unpatched importlib scan and its import
    registers the kind — proving the group name works outside our fakes."""
    site = tmp_path / "site"
    site.mkdir()
    (site / "extadapter_pkg.py").write_text(
        "from bmad_loop.adapters.registry import register_adapter\n"
        "def _load():\n"
        "    raise AssertionError('load thunk must stay lazy')\n"
        "register_adapter('extadapter-real', False, _load)\n",
        encoding="utf-8",
    )
    dist = site / "extadapter-0.1.dist-info"
    dist.mkdir()
    (dist / "METADATA").write_text("Metadata-Version: 2.1\nName: extadapter\nVersion: 0.1\n")
    (dist / "entry_points.txt").write_text(
        "[bmad_loop.adapters]\nextadapter = extadapter_pkg\n", encoding="utf-8"
    )
    monkeypatch.syspath_prepend(str(site))
    fresh_adapter_registry._EXTERNALS_LOADED = False  # re-arm the (real) scan
    assert "extadapter-real" in fresh_adapter_registry.known_adapter_kinds()
    assert fresh_adapter_registry.external_adapter_errors() == {}


# --------------------------------------------------------------------------- #
# runsetup.make_adapters dispatch through the registry


def test_cli_make_adapters_alias_is_the_runsetup_factory():
    """`cli._make_adapters` is a re-export seam, not a second implementation: the
    run/sweep/resume composers hand `make_adapters=cli._make_adapters` from this
    module's namespace so `monkeypatch.setattr(cli, "_make_adapters", ...)` keeps
    biting. Pin the identity — a rebase that resurrects a duplicate in cli.py
    would leave both halves importable and only one of them dispatching."""
    assert cli._make_adapters is runsetup.make_adapters


def test_make_adapters_dispatches_registered_kind(fresh_adapter_registry, project, monkeypatch):
    """A registered out-of-tree kind dispatches with no core branching: the
    synthesizing dev/review roles build the dev variant with ``paths`` + the shared
    mux, triage builds the plain variant, and the ``(cfg, synthesizes)`` cache
    shares the dev instance across dev+review."""
    registry = fresh_adapter_registry
    registry.register_adapter("hermes", needs_mux=True, load=lambda: _stub_builder())
    monkeypatch.setattr(mux_mod, "_usable", lambda mux: True)
    install_bmad_config(project)
    _write_profile(project.project, "hermes", adapter="hermes", hookless=False)
    _write_policy(project.project, '[adapter]\nname = "hermes"\n')
    pol = policy_mod.load(project.project / ".bmad-loop" / "policy.toml")

    adapters = runsetup.make_adapters(project.project, _run_dir(project.project), pol)

    assert isinstance(adapters["dev"], _StubDevAdapter)
    assert adapters["dev"] is adapters["review"]  # (cfg, synthesizes) sharing
    assert adapters["dev"].paths.project == project.project
    assert "mux" in adapters["dev"].kwargs  # needs_mux=True threaded the transport
    assert isinstance(adapters["triage"], _StubAdapter)
    assert not isinstance(adapters["triage"], _StubDevAdapter)
    assert adapters["triage"] is not adapters["dev"]
    assert "mux" in adapters["triage"].kwargs


def test_make_adapters_kind_comes_from_the_passed_in_profiles(
    fresh_adapter_registry, project, monkeypatch
):
    """#461 point 4, extended to the field that now picks the argv builder: when a
    caller hands in the already-resolved `profiles` it gated on, the adapter KIND
    must come from THOSE bytes — not from a second read of a file a driven session
    can rewrite in between. On-disk says `generic`; the passed-in mapping says
    `hermes`; `hermes` must be what builds.

    ABLATION: move the `get_adapter_kind` call above the `profiles is not None`
    branch (or re-read with `get_profile` for the kind) and this reddens — the
    generic tmux adapter builds instead of the stub."""
    registry = fresh_adapter_registry
    registry.register_adapter("hermes", needs_mux=False, load=lambda: _stub_builder())
    monkeypatch.setattr(mux_mod, "get_multiplexer", lambda: pytest.fail("no mux for this kind"))
    install_bmad_config(project)
    _write_profile(project.project, "swapme", adapter="generic", hookless=False)
    _write_policy(project.project, '[adapter]\nname = "swapme"\n')
    pol = policy_mod.load(project.project / ".bmad-loop" / "policy.toml")

    pinned = CLIProfile(
        name="swapme", binary="swapme", adapter="hermes", hooks=HookSpec("none", "", {})
    )
    adapters = runsetup.make_adapters(
        project.project,
        _run_dir(project.project),
        pol,
        profiles=dict.fromkeys(runsetup.ROLES, pinned),
    )
    assert isinstance(adapters["dev"], _StubDevAdapter)
    assert adapters["dev"].profile is pinned


def test_config_digest_pins_the_adapter_kind(project):
    """The kind selects the argv builder, so rewriting it mid-run must move the
    digest the auto-sweep gate compares against — otherwise a driven session swaps
    the whole launch shape without tripping the #461 pin.

    ABLATION: drop `"adapter": prof.adapter` from the launch payload and the two
    digests below become equal."""
    install_bmad_config(project)
    _write_profile(project.project, "swapme", adapter="generic", hookless=False)
    _write_policy(project.project, '[adapter]\nname = "swapme"\n')
    pol = policy_mod.load(project.project / ".bmad-loop" / "policy.toml")

    before = runsetup.config_digest(pol, project.project)
    _write_profile(project.project, "swapme", adapter="opencode-http", hookless=False)
    after = runsetup.config_digest(pol, project.project)
    assert before != after


def test_make_adapters_skips_mux_for_needs_mux_false_kind(
    fresh_adapter_registry, project, monkeypatch
):
    """A ``needs_mux=False`` kind must never resolve the multiplexer — the same
    guarantee the hookless HTTP adapter relies on."""
    registry = fresh_adapter_registry
    registry.register_adapter("noxport", needs_mux=False, load=lambda: _stub_builder())

    def no_mux():
        raise AssertionError("a needs_mux=False kind must not resolve a multiplexer")

    monkeypatch.setattr(mux_mod, "get_multiplexer", no_mux)
    install_bmad_config(project)
    _write_profile(project.project, "noxport", adapter="noxport")
    _write_policy(project.project, '[adapter]\nname = "noxport"\n')
    pol = policy_mod.load(project.project / ".bmad-loop" / "policy.toml")

    adapters = runsetup.make_adapters(project.project, _run_dir(project.project), pol)
    assert isinstance(adapters["dev"], _StubDevAdapter)
    assert "mux" not in adapters["dev"].kwargs


def test_make_adapters_needs_mux_true_kind_still_refuses_an_unusable_mux(
    fresh_adapter_registry, project, monkeypatch
):
    """The other half of the `needs_mux` gate: moving the transport probe inside it
    must not make the refusal unreachable for a family that DOES drive one.

    Ablating the gate to `if True:` leaves this green and the previous test red, so
    the pair is what pins the gate — this one alone would not."""
    registry = fresh_adapter_registry
    registry.register_adapter("hermes", needs_mux=True, load=lambda: _stub_builder())
    monkeypatch.setattr(mux_mod, "_usable", lambda mux: False)
    install_bmad_config(project)
    _write_profile(project.project, "hermes", adapter="hermes", hookless=False)
    _write_policy(project.project, '[adapter]\nname = "hermes"\n')
    pol = policy_mod.load(project.project / ".bmad-loop" / "policy.toml")

    with pytest.raises(SystemExit, match="not usable"):
        runsetup.make_adapters(project.project, _run_dir(project.project), pol)


def test_make_adapters_construct_error_becomes_systemexit(
    fresh_adapter_registry, project, monkeypatch
):
    """A family-declared construction failure raised in __init__ is converted to a
    clean SystemExit — a run aborts with a message, not a traceback."""

    class _Boom(Exception):
        pass

    class _Exploding:
        def __init__(self, **kwargs):
            raise _Boom("server would not start")

    registry = fresh_adapter_registry
    registry.register_adapter(
        "boomer",
        needs_mux=False,
        load=lambda: AdapterBuilder(plain=_Exploding, dev=_Exploding, construct_error=(_Boom,)),
    )
    install_bmad_config(project)
    _write_profile(project.project, "boomer", adapter="boomer")
    _write_policy(project.project, '[adapter]\nname = "boomer"\n')
    pol = policy_mod.load(project.project / ".bmad-loop" / "policy.toml")

    with pytest.raises(SystemExit, match="server would not start"):
        runsetup.make_adapters(project.project, _run_dir(project.project), pol)


def test_make_adapters_unrelated_construct_failure_is_not_swallowed(
    fresh_adapter_registry, project
):
    """`construct_error` is a family's DECLARED failure mode, not a catch-all: an
    exception it does not name must propagate as itself, so a genuine bug in an
    adapter surfaces as a traceback rather than a misleading `error:` line.

    ABLATION: widen the `except builder.construct_error` to `except Exception` and
    this reddens (SystemExit is raised instead)."""

    class _Exploding:
        def __init__(self, **kwargs):
            raise ZeroDivisionError("a real bug, not a declared failure")

    fresh_adapter_registry.register_adapter(
        "buggy",
        needs_mux=False,
        load=lambda: AdapterBuilder(plain=_Exploding, dev=_Exploding, construct_error=()),
    )
    install_bmad_config(project)
    _write_profile(project.project, "buggy", adapter="buggy")
    _write_policy(project.project, '[adapter]\nname = "buggy"\n')
    pol = policy_mod.load(project.project / ".bmad-loop" / "policy.toml")

    with pytest.raises(ZeroDivisionError):
        runsetup.make_adapters(project.project, _run_dir(project.project), pol)


def test_make_adapters_load_thunk_failure_becomes_systemexit(fresh_adapter_registry, project):
    """A load thunk that raises — the family's own module missing an optional
    dependency is the ordinary case — aborts with a clean SystemExit naming the
    profile and the kind, not a raw traceback.

    This is the one failure mode with no earlier gate: `validate` and `bmad-loop
    adapters` both deliberately avoid invoking the thunk, and by the time
    `make_adapters` runs, `compose_run` has already written the run state and pid,
    so an escaping ImportError strands a run directory behind a traceback.

    ABLATION: drop the try/except around `kind.load()` and this reddens (the
    ModuleNotFoundError propagates as itself)."""

    def _load():
        raise ModuleNotFoundError("No module named 'acmesdk'")

    fresh_adapter_registry.register_adapter("lazyboom", needs_mux=False, load=_load)
    install_bmad_config(project)
    _write_profile(project.project, "lazyboom", adapter="lazyboom")
    _write_policy(project.project, '[adapter]\nname = "lazyboom"\n')
    pol = policy_mod.load(project.project / ".bmad-loop" / "policy.toml")

    with pytest.raises(SystemExit, match=r"lazyboom.*failed to load.*acmesdk"):
        runsetup.make_adapters(project.project, _run_dir(project.project), pol)


def test_make_adapters_non_import_thunk_failure_is_not_swallowed(fresh_adapter_registry, project):
    """The other half of the pair above, and the same rule `construct_error` follows:
    a missing dependency is a lazy loader's DECLARED failure, but anything else is a
    bug in that package and must surface as itself. Swallowing it would hand an
    adapter author an `error:` line where the traceback was the whole diagnosis.

    ABLATION: widen the `except ImportError` to `except Exception` and this reddens
    (SystemExit is raised instead)."""

    def _load():
        raise ZeroDivisionError("a real bug, not a missing dependency")

    fresh_adapter_registry.register_adapter("lazybug", needs_mux=False, load=_load)
    install_bmad_config(project)
    _write_profile(project.project, "lazybug", adapter="lazybug")
    _write_policy(project.project, '[adapter]\nname = "lazybug"\n')
    pol = policy_mod.load(project.project / ".bmad-loop" / "policy.toml")

    with pytest.raises(ZeroDivisionError):
        runsetup.make_adapters(project.project, _run_dir(project.project), pol)


def test_make_adapters_unknown_kind_systemexit_names_profile(fresh_adapter_registry, project):
    """A profile whose ``adapter`` names no registered kind aborts the run with a
    SystemExit that names both the profile and the known kinds."""
    install_bmad_config(project)
    _write_profile(project.project, "weird", adapter="ghostkind")
    _write_policy(project.project, '[adapter]\nname = "weird"\n')
    pol = policy_mod.load(project.project / ".bmad-loop" / "policy.toml")

    with pytest.raises(SystemExit, match=r"weird.*ghostkind.*generic"):
        runsetup.make_adapters(project.project, _run_dir(project.project), pol)


def test_make_adapters_generic_shares_synthesizing_but_not_triage(
    fresh_adapter_registry, project, monkeypatch
):
    """The ``(cfg, synthesizes)`` cache with the real builtin generic kind: dev and
    review (same cfg, both the dev primitive) share one GenericDevAdapter, triage on
    the same cfg gets a separate plain GenericAdapter."""
    from bmad_loop.adapters.generic import GenericAdapter, GenericDevAdapter

    monkeypatch.setattr(mux_mod, "_usable", lambda mux: True)
    install_bmad_config(project)
    adapters = runsetup.make_adapters(
        project.project, _run_dir(project.project), policy_mod.load(None)
    )
    assert isinstance(adapters["dev"], GenericDevAdapter)
    assert adapters["dev"] is adapters["review"]
    assert isinstance(adapters["triage"], GenericAdapter)
    assert not isinstance(adapters["triage"], GenericDevAdapter)
    assert adapters["triage"] is not adapters["dev"]


def test_make_adapters_opencode_http_dispatch_unchanged(
    fresh_adapter_registry, project, monkeypatch
):
    """Regression pin: routing the ``opencode-http`` profile through the registry
    yields exactly the pre-registry adapters — OpencodeDevAdapter for the
    synthesizing roles (never resolving a mux), OpencodeHttpAdapter for triage."""
    from bmad_loop.adapters import opencode_http
    from bmad_loop.adapters.opencode_http import OpencodeDevAdapter, OpencodeHttpAdapter

    def no_mux():
        raise AssertionError("hookless opencode-http must not resolve a multiplexer")

    monkeypatch.setattr(opencode_http, "_require_httpx", lambda: object())
    monkeypatch.setattr(mux_mod, "get_multiplexer", no_mux)
    install_bmad_config(project)
    _write_policy(project.project, '[adapter]\nname = "opencode"\n')  # alias → opencode-http
    pol = policy_mod.load(project.project / ".bmad-loop" / "policy.toml")

    adapters = runsetup.make_adapters(project.project, _run_dir(project.project), pol)
    assert isinstance(adapters["dev"], OpencodeDevAdapter)
    assert adapters["dev"] is adapters["review"]
    assert adapters["dev"].profile.adapter == "opencode-http"
    assert isinstance(adapters["triage"], OpencodeHttpAdapter)
    assert not isinstance(adapters["triage"], OpencodeDevAdapter)


# --------------------------------------------------------------------------- #
# `bmad-loop adapters` + the validate findings


def test_adapters_command_lists_builtins_and_surfaces_failures(
    scan_adapter_registry, capsys, tmp_path
):
    """`bmad-loop adapters` renders the kind table and names a failed out-of-tree
    package — the one place an operator looks when an installed adapter is missing."""
    registry, arm = scan_adapter_registry

    def boom():
        raise ImportError("No module named 'ghost_dependency'")

    arm(_FakeEntryPoint("brokenadapter", boom))
    args = argparse.Namespace(project=tmp_path)
    assert cli.cmd_adapters(args) == 0
    captured = capsys.readouterr()
    assert "generic" in captured.out  # the table renders
    assert "opencode-http" in captured.out
    assert "brokenadapter" in captured.err
    assert "ghost_dependency" in captured.err


def test_adapters_command_flags_dangling_kind_reference(fresh_adapter_registry, capsys, tmp_path):
    """A project profile whose adapter kind never registered is named as a warning:
    the table can't show a kind that isn't there, so the dangling reference is
    surfaced explicitly."""
    _write_profile(tmp_path, "weird", adapter="ghostkind")
    args = argparse.Namespace(project=tmp_path)
    assert cli.cmd_adapters(args) == 0
    captured = capsys.readouterr()
    assert "ghostkind" in captured.err
    assert "weird" in captured.err


def test_adapters_command_reports_a_malformed_overlay(fresh_adapter_registry, capsys, tmp_path):
    """A listing assembled from a profile set that silently lost an entry is worse
    than a named error: a malformed project overlay exits 1 naming the file."""
    d = tmp_path / ".bmad-loop" / "profiles"
    d.mkdir(parents=True)
    (d / "bad.toml").write_text('name = "bad"\nbinary = "bad"\nadapter = 5\n[hooks]\n')
    assert cli.cmd_adapters(argparse.Namespace(project=tmp_path)) == 1
    assert "adapter must be a string" in capsys.readouterr().err


def _validate_findings(project, capsys):
    """Run the real `validate --json` and hand back its findings. Through the CLI
    on purpose: the check ids then pass the `VALIDATE_CHECKS` assert in
    `ValidationReport.add`, which is what makes a new id's absence a crash rather
    than a quiet no-op."""
    cli.main(["validate", "--project", str(project), "--json"])
    return json.loads(capsys.readouterr().out)["findings"]


def test_validate_httpx_check_keys_on_the_adapter_kind_not_hooklessness(
    fresh_adapter_registry, project, capsys
):
    """The httpx extra belongs to the opencode-http FAMILY, not to hooklessness.
    Once the two axes decoupled, a hookless profile driven by another registered
    kind must draw no `adapter.httpx` finding at all — FAILing it would tell an
    operator to `pip install bmad-loop[opencode]` for a package they do not use.

    ABLATION: re-key the check on `profile.hookless` and the `not any(...)` assert
    reddens (an httpx finding appears for a kind that never imports httpx)."""
    fresh_adapter_registry.register_adapter("hermes", needs_mux=False, load=lambda: _stub_builder())
    install_bmad_config(project)
    _write_profile(project.project, "hermes", adapter="hermes")  # hookless=True
    _write_policy(project.project, '[adapter]\nname = "hermes"\n')

    findings = _validate_findings(project.project, capsys)
    assert not any(f["check"] == "adapter.httpx" for f in findings)
    # the transport question is still answered — only the family question moved
    assert any(f["check"] == "adapter.hookless" for f in findings)
    assert [f["severity"] for f in findings if f["check"] == "adapter.kind"] == ["ok"]


def test_validate_model_format_check_keys_on_the_adapter_kind_not_hooklessness(
    fresh_adapter_registry, project, capsys
):
    """`policy.model-qualified` is the httpx check's sibling and needed the same
    re-keying: "provider/model" is a fact about the opencode SERVER's config file,
    not about whether a profile registers hooks. An out-of-tree hookless family
    whose server takes bare model names must draw no warning naming a spelling it
    does not use.

    The `adapter.hookless` assert is the positive control, and the point of the
    test: the profile IS hookless, so the old predicate would have fired here. The
    absent warning is therefore the re-keying and not a profile that failed to
    load, a model that never reached the check, or a validate that bailed early.

    ABLATION: re-key the check on `prof.hookless` and the `not any(...)` reddens."""
    fresh_adapter_registry.register_adapter("hermes", needs_mux=False, load=lambda: _stub_builder())
    install_bmad_config(project)
    _write_profile(project.project, "hermes", adapter="hermes")  # hookless=True
    _write_policy(project.project, '[adapter]\nname = "hermes"\nmodel = "haiku"\n')

    findings = _validate_findings(project.project, capsys)
    assert not any(f["check"] == "policy.model-qualified" for f in findings)
    # controls: the profile loaded, its kind resolved, and it really is hookless
    assert [f["severity"] for f in findings if f["check"] == "adapter.kind"] == ["ok"]
    assert any(f["check"] == "adapter.hookless" for f in findings)


def test_validate_model_format_warns_on_an_opencode_kind_carrying_a_hook_dialect(
    fresh_adapter_registry, project, capsys
):
    """The other direction of the same miss: keyed on `hookless`, the check also
    UNDER-fires. Decoupling the axes made `opencode-http` beside a hook dialect a
    legal profile, and its bare model still falls back to the server's default —
    the case the warning exists for — while `prof.hookless` reads False.

    ABLATION: re-key the check on `prof.hookless` and this reddens (no finding at
    all, because the profile is not hookless)."""
    install_bmad_config(project)
    _write_profile(project.project, "ochooked", adapter="opencode-http", hookless=False)
    _write_policy(project.project, '[adapter]\nname = "ochooked"\nmodel = "haiku"\n')

    findings = [
        f
        for f in _validate_findings(project.project, capsys)
        if f["check"] == "policy.model-qualified"
    ]
    assert findings, "a bare model on the opencode-http kind must warn"
    assert {f["severity"] for f in findings} == {"warning"}
    assert all("haiku" in f["message"] for f in findings)


def test_validate_flags_an_unregistered_adapter_kind(fresh_adapter_registry, project, capsys):
    """`adapter.kind` is resolved against the live registry, so a profile naming a
    kind no installed package provides is a FAIL that names the known set."""
    install_bmad_config(project)
    _write_profile(project.project, "weird", adapter="ghostkind")
    _write_policy(project.project, '[adapter]\nname = "weird"\n')

    findings = [
        f for f in _validate_findings(project.project, capsys) if f["check"] == "adapter.kind"
    ]
    assert [f["severity"] for f in findings] == ["problem"]
    assert "ghostkind" in findings[0]["message"] and "generic" in findings[0]["message"]


def test_dry_run_says_an_unregistered_kind_would_abort(fresh_adapter_registry, project, capsys):
    """`--dry-run` renders a preview from `binary`/`launch_args`/`prompt_template`
    alone, so an unregistered `adapter` is invisible in it: the operator reads a
    perfectly plausible invocation for a config `make_adapters` refuses to build.
    That is exactly the gap the honesty banner exists to close, so the unknown kind
    joins the refusals it already mirrors.

    Same contract as the other banner sources: stderr-only, the schedule still
    renders on stdout, and the exit code stays 0 — a dry-run is a diagnostic.

    ABLATION: drop the `_unknown_adapter_kinds` call from
    `_warn_preflight_would_abort` and this reddens (stderr is empty, rc still 0)."""
    write_sprint(project, {"epic-1": "backlog", "1-1-a": "ready-for-dev"})
    _write_profile(project.project, "weird", adapter="ghostkind")
    _write_policy(project.project, '[adapter]\nname = "weird"\n')
    pol = policy_mod.load(project.project / ".bmad-loop" / "policy.toml")
    args = argparse.Namespace(epic=None, story=None, max_stories=None)

    assert cli._dry_run(project, pol, args) == 0
    out, err = capsys.readouterr()
    assert "NOT runnable" in err
    assert "ghostkind" in err and "generic" in err
    assert "1-1-a" in out  # the schedule itself still rendered


def test_validate_reports_a_broken_profile_package_even_when_policy_fails(
    fresh_adapter_registry, project, monkeypatch, capsys
):
    """A broken profile package must be reported for its OWN reason, not silently
    dropped because something else in the config is also wrong.

    The `bmad_loop.profiles` scan has exactly one other trigger — `load_profiles`,
    which validate reaches via `get_profile`, inside the block a `PolicyError`
    aborts. Reading the error map without scanning would print nothing here, which
    an operator reads as "no profile package failed". The adapter half never had
    the gap, because `known_adapter_kinds()` runs unconditionally.

    ABLATION: drop the `_load_external_profiles()` call from
    `external_profile_errors` and this reddens (no adapter.external-profile
    finding) while the `policy` failure below still reports."""

    def fake_entry_points(*, group):
        assert group == profile_mod.PROFILES_GROUP

        def boom():
            raise ImportError("No module named 'ghost_profile_dep'")

        return [_FakeEntryPoint("brokenprofiles", boom)]

    monkeypatch.setattr(profile_mod.importlib.metadata, "entry_points", fake_entry_points)
    profile_mod._EXTERNALS_LOADED = False  # re-arm the scan the fixture parks

    install_bmad_config(project)
    _write_policy(project.project, "this is not = valid toml [[[\n")

    findings = _validate_findings(project.project, capsys)
    assert [f["severity"] for f in findings if f["check"] == "policy"] == ["problem"]
    external = [f for f in findings if f["check"] == "adapter.external-profile"]
    assert [f["severity"] for f in external] == ["warning"]
    assert "ghost_profile_dep" in external[0]["message"]


def test_validate_warns_on_a_broken_external_package(scan_adapter_registry, project, capsys):
    """A half-installed out-of-tree package is a WARNING, not a failure: selection
    already degraded past it, so the same non-blocking treatment a failed mux
    backend package gets."""
    _registry, arm = scan_adapter_registry

    def boom():
        raise ImportError("No module named 'ghost_dependency'")

    arm(_FakeEntryPoint("brokenadapter", boom))
    install_bmad_config(project)

    findings = [
        f for f in _validate_findings(project.project, capsys) if f["check"] == "adapter.external"
    ]
    assert [f["severity"] for f in findings] == ["warning"]
    assert "ghost_dependency" in findings[0]["message"]
