"""External-backend discovery proof (the ``bmad_loop.mux_backends`` entry-point scan).

An out-of-tree backend package advertises a module under the
``bmad_loop.mux_backends`` entry-point group; ``_load_external_backends``
imports it (after the builtins, so tmux keeps first registration) and the
module's import-time ``register_multiplexer`` call makes it selectable exactly
like a bundled backend. These tests pin the loader's contract: discovery,
ordering, and — above all — that a broken third-party distribution degrades to
a recorded, surfaced reason and can never break backend selection.

Entry points are faked by monkeypatching ``importlib.metadata.entry_points``
through the ``multiplexer`` module's own binding (it imports the module, so the
attribute path is ``m.importlib.metadata``); one test builds a real
``*.dist-info`` on ``sys.path`` to prove the scan works against genuine
packaging metadata, not just our fake.
"""

from __future__ import annotations

import sys

import pytest

# Reuse the registry-isolation fixture where it lives; importing it into this
# module's namespace is how pytest shares a non-conftest fixture across files.
from test_backend_registry import fresh_registry  # noqa: F401

from bmad_loop.adapters import multiplexer as m
from bmad_loop.adapters.tmux_backend import TmuxMultiplexer


class _FakeDist:
    """Stands in for ``EntryPoint.dist``; the scan orders on its ``.name``."""

    def __init__(self, name):
        self.name = name


class _FakeEntryPoint:
    """Duck-typed stand-in for importlib.metadata.EntryPoint: the loader touches
    ``.name``, ``.dist`` (the scan's tiebreak — see `_load_external_backends`) and
    ``.load()``. ``dist`` defaults to a distinct-per-name stand-in so the ordering
    of same-named entries is only ever decided by a test that sets it."""

    def __init__(self, name, load, dist=None):
        self.name = name
        self.dist = _FakeDist(dist if dist is not None else f"{name}-dist")
        self._load = load

    def load(self):
        return self._load()


@pytest.fixture
def scan_registry(fresh_registry, monkeypatch):  # noqa: F811 — fixture, not a redefinition
    """fresh_registry with the externals scan re-armed (the base fixture parks it
    as already-loaded so installed adapters can't leak into builtin tests).
    Yields a hook: call it with fake entry points (or an exception to raise from
    the scan itself) and the next selection performs that scan."""

    def arm(*eps, scan_error: Exception | None = None):
        def fake_entry_points(*, group):
            assert group == m.MUX_BACKENDS_GROUP
            if scan_error is not None:
                raise scan_error
            return list(eps)

        monkeypatch.setattr(m.importlib.metadata, "entry_points", fake_entry_points)
        m._EXTERNALS_LOADED = False
        m._EXTERNAL_ERRORS.clear()
        m.get_multiplexer.cache_clear()

    yield fresh_registry, arm


def test_entry_point_backend_registers_and_is_selectable(scan_registry, monkeypatch):
    """The pip-install-and-go path: the entry point's module import registers the
    backend; it lists in detect_multiplexers and a forced name selects it."""
    registry, arm = scan_registry
    sentinel = object()

    def load():
        registry.register_multiplexer("extmux", lambda p: False, lambda: sentinel)
        return None  # the loader ignores the return value; import side effect is the contract

    arm(_FakeEntryPoint("extmux", load))
    monkeypatch.setenv("BMAD_LOOP_MUX_BACKEND", "extmux")
    registry.get_multiplexer.cache_clear()
    assert registry.get_multiplexer() is sentinel
    assert registry.external_backend_errors() == {}
    monkeypatch.delenv("BMAD_LOOP_MUX_BACKEND")
    registry.get_multiplexer.cache_clear()
    rows = {r.name: r for r in registry.detect_multiplexers()}
    assert "extmux" in rows


def test_externals_load_after_builtins(scan_registry):
    """Ordering guarantee: builtins register first, so tmux keeps first-wins on a
    name collision and POSIX default selection is unchanged by installing an
    adapter. The external lands after both builtins in the registry."""
    registry, arm = scan_registry

    def load():
        registry.register_multiplexer("extmux", lambda p: True, lambda: object())

    arm(_FakeEntryPoint("extmux", load))
    registry._select()
    names = [name for name, _, _ in registry._BACKENDS]
    assert names.index("tmux") < names.index("extmux")


def test_broken_entry_point_degrades_and_is_recorded(scan_registry, monkeypatch):
    """A distribution whose import blows up must not break selection: tmux is
    still selected, and the failure is recorded for mux/validate to show."""
    registry, arm = scan_registry

    def boom():
        raise ImportError("No module named 'ghost_dependency'")

    arm(_FakeEntryPoint("brokenmux", boom))
    monkeypatch.setattr(sys, "platform", "linux")
    backend, name, _reason = registry._select()
    assert isinstance(backend, TmuxMultiplexer) and name == "tmux"
    errors = registry.external_backend_errors()
    assert list(errors) == ["brokenmux"]
    assert "ghost_dependency" in errors["brokenmux"]


def test_one_broken_package_does_not_hide_the_rest(scan_registry, monkeypatch):
    """Per-entry isolation: the loader keeps importing after a failure, so a
    working adapter still registers alongside a broken one."""
    registry, arm = scan_registry
    sentinel = object()

    def boom():
        raise RuntimeError("half-installed")

    def load():
        registry.register_multiplexer("goodmux", lambda p: False, lambda: sentinel)

    arm(_FakeEntryPoint("brokenmux", boom), _FakeEntryPoint("goodmux", load))
    monkeypatch.setenv("BMAD_LOOP_MUX_BACKEND", "goodmux")
    registry.get_multiplexer.cache_clear()
    assert registry.get_multiplexer() is sentinel
    assert list(registry.external_backend_errors()) == ["brokenmux"]


def test_scan_failure_degrades(scan_registry, monkeypatch):
    """Even the entry-point enumeration itself blowing up (exotic sys.path /
    importlib state) leaves selection working, with the scan failure recorded."""
    registry, arm = scan_registry
    arm(scan_error=RuntimeError("metadata index corrupt"))
    monkeypatch.setattr(sys, "platform", "linux")
    backend, name, _reason = registry._select()
    assert isinstance(backend, TmuxMultiplexer) and name == "tmux"
    assert "<entry-point scan>" in registry.external_backend_errors()


def test_scan_runs_once_per_process(scan_registry):
    """The loaded-flag is set up front: a second selection does not re-scan (a
    third-party import failure is not transient; re-importing would re-fail on
    every selection)."""
    registry, arm = scan_registry
    calls = []

    def load():
        calls.append(1)

    arm(_FakeEntryPoint("extmux", load))
    registry._select()
    registry.get_multiplexer.cache_clear()
    registry._select()
    assert len(calls) == 1


def test_same_named_broken_distributions_both_record_a_reason(scan_registry, monkeypatch):
    """Two distributions may advertise the SAME entry-point name in this group, and
    both may be broken. A name-keyed assignment let the second overwrite the first:
    the operator fixed the package they were shown and met the other one on the next
    run, with nothing saying it had ever been there. Both reasons are kept now, each
    labelled with its distribution — the entry-point name is not the name you
    `pip uninstall`, and two packages failing identically would otherwise render as
    the same sentence twice. (#566 names the adapter and profile scans only; this
    third site is the one it omits.)

    The fixture trap #566 itself flags: asserting only "two reasons are present"
    passes for the wrong reason if the two entry points accidentally got DIFFERENT
    names. Pinning the single shared key forecloses that, and is the shape
    decision's own assertion besides — see the comment below.

    Ablation: restore the single-key write
    (`_EXTERNAL_ERRORS[ep.name] = f"{type(exc).__name__}: {exc}"`) in
    `_load_external_backends` and this test fails on the missing `alpha-backend`
    half, while the adapter and profile twins stay green — that per-site
    independence is what proves the fix reached all three scans rather than one."""
    registry, arm = scan_registry

    def boom(msg):
        def load():
            raise ImportError(msg)

        return load

    arm(
        _FakeEntryPoint("acme", boom("No module named 'alpha_dep'"), dist="alpha-backend"),
        _FakeEntryPoint("acme", boom("No module named 'zeta_dep'"), dist="zeta-backend"),
    )
    monkeypatch.setattr(sys, "platform", "linux")
    backend, name, _reason = registry._select()
    assert isinstance(backend, TmuxMultiplexer) and name == "tmux"  # selection still works
    reason = registry.external_backend_errors()["acme"]
    assert "alpha-backend" in reason and "zeta-backend" in reason
    assert "alpha_dep" in reason and "zeta_dep" in reason
    # Still ONE key — the shape decision. The key set is what reaches
    # `detail["entry_point"]` in `validate --json`, so it deliberately does not grow;
    # only the human-facing reason string widens.
    assert list(registry.external_backend_errors()) == ["acme"]


@pytest.mark.parametrize(
    "order", [("alpha", "zeta"), ("zeta", "alpha")], ids=["alpha-discovered-first", "zeta-first"]
)
def test_same_named_entry_points_are_visited_in_distribution_order(
    scan_registry, monkeypatch, order
):
    """The mux analogue of the adapter scan's
    `test_same_named_entry_points_resolve_by_distribution_not_install_order`: this
    scan was left unsorted when commit 90a7ca9 ordered the other two.

    It matters here because the accumulated reason is now READ in append order. A
    bare `entry_points(group=...)` yields distributions in `sys.path` order, so
    without the sort the assertion below would be a fact about the machine running
    the test — the same two packages would render their two reasons in the opposite
    order on another host, and the accumulation test above would be
    non-deterministic by construction. Sorting on (name, distribution) is what makes
    the recording a fact about the packages.

    Both parameters arm the identical pair and differ only in the order the scan
    yields them; `alpha-backend`'s reason must come first either way.

    ABLATION: drop the `getattr(e.dist, ...)` half of the sort key in
    `_load_external_backends` and the `zeta-first` case reddens while
    `alpha-discovered-first` stays green — which is the finding: the name-only key
    is right only when the install happens to agree with it."""
    registry, arm = scan_registry

    def boom(msg):
        def load():
            raise ImportError(msg)

        return load

    eps = {
        "alpha": _FakeEntryPoint("acme", boom("alpha broke"), dist="alpha-backend"),
        "zeta": _FakeEntryPoint("acme", boom("zeta broke"), dist="zeta-backend"),
    }
    arm(*(eps[k] for k in order))
    monkeypatch.setattr(sys, "platform", "linux")
    registry._select()

    assert registry.external_backend_errors()["acme"].startswith("alpha-backend: ")


def test_mux_command_surfaces_load_failures(scan_registry, monkeypatch, capsys, tmp_path):
    """`bmad-loop mux` names a failed external package — the one place an operator
    looks when an installed backend is missing from the table."""
    import argparse

    from bmad_loop import cli

    _registry, arm = scan_registry

    def boom():
        raise ImportError("No module named 'ghost_dependency'")

    arm(_FakeEntryPoint("brokenmux", boom))
    monkeypatch.setattr(sys, "platform", "linux")
    args = argparse.Namespace(project=tmp_path, action=None, name=None, clear=False, force=False)
    assert cli.cmd_mux(args) == 0
    captured = capsys.readouterr()
    assert "brokenmux" in captured.err
    assert "ghost_dependency" in captured.err
    assert "tmux" in captured.out  # the table itself still renders


def test_real_dist_info_metadata_is_discovered(
    fresh_registry, monkeypatch, tmp_path  # noqa: F811 — fixture, not a redefinition
):
    """End-to-end against genuine packaging metadata: a real ``*.dist-info`` +
    module on sys.path is found by the unpatched importlib scan and its import
    registers the backend — proving the group name and value convention work
    outside our fakes."""
    site = tmp_path / "site"
    site.mkdir()
    (site / "extmux_backend.py").write_text(
        "from bmad_loop.adapters.multiplexer import register_multiplexer\n"
        "class _Probe:\n"
        "    pass\n"
        "register_multiplexer('extmux-real', lambda p: False, _Probe)\n",
        encoding="utf-8",
    )
    dist = site / "extmux-0.1.dist-info"
    dist.mkdir()
    (dist / "METADATA").write_text("Metadata-Version: 2.1\nName: extmux\nVersion: 0.1\n")
    (dist / "entry_points.txt").write_text(
        "[bmad_loop.mux_backends]\nextmux = extmux_backend\n", encoding="utf-8"
    )
    monkeypatch.syspath_prepend(str(site))
    fresh_registry._EXTERNALS_LOADED = False  # re-arm the (real) scan
    fresh_registry._select()
    assert fresh_registry.external_backend_errors().get("extmux") is None
    assert "extmux-real" in [name for name, _, _ in fresh_registry._BACKENDS]
