"""Unit tests for the Unity scene-guard seeder (``unity_seed_assets.py``).

The seeder copies the bundled scene auto-save guard payload (``unity_assets/``:
``SceneAutoSaveGuard.cs`` + its asmdef, with pre-generated fixed-GUID ``.meta``
files) into a project's ``Assets`` tree so a chronically-dirty scene never raises
the two run-stalling modal dialogs. These drive its ``main()`` end-to-end against a
temp worktree, using the real bundled payload (so the version-header + GUID contract
is exercised, not a stand-in).

The plugin-side wiring (env plumbing + the pre_worktree_setup/pre_ready_gate hooks
that invoke this seeder) lives in ``test_engine_plugin.py``.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

from bmad_loop.plugins import get_plugin

_GUARD_DIR = "Assets/BmadLoop/Editor"


def _load_seeder():
    path = os.path.join(get_plugin("unity").scripts_dir, "unity_seed_assets.py")
    spec = importlib.util.spec_from_file_location("unity_seed_assets_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _payload(mod) -> Path:
    return Path(mod.__file__).resolve().parent / "unity_assets"


def _set_env(monkeypatch, worktree, *, install="1", guard_dir=None):
    monkeypatch.setenv("BMAD_LOOP_WORKTREE", str(worktree))
    monkeypatch.setenv("BMAD_LOOP_UNITY_INSTALL_SCENE_GUARD", install)
    if guard_dir is None:
        monkeypatch.delenv("BMAD_LOOP_UNITY_SCENE_GUARD_DIR", raising=False)
    else:
        monkeypatch.setenv("BMAD_LOOP_UNITY_SCENE_GUARD_DIR", guard_dir)


# ---------------------------------------------------------------- fresh install


def test_seed_fresh_install_lays_all_payload_files(tmp_path, monkeypatch):
    mod = _load_seeder()
    _set_env(monkeypatch, tmp_path)
    (tmp_path / "Assets").mkdir()

    assert mod.main() == 0

    payload = _payload(mod)
    editor = tmp_path / _GUARD_DIR
    # content files land byte-identical to the payload (robust to a version bump)
    for name in (
        "SceneAutoSaveGuard.cs",
        "SceneAutoSaveGuard.cs.meta",
        "BmadLoop.Unity.Editor.asmdef",
        "BmadLoop.Unity.Editor.asmdef.meta",
    ):
        assert (editor / name).read_bytes() == (payload / name).read_bytes(), name
    # parent-folder metas carry the payload's fixed GUIDs (no per-worktree churn)
    assert (tmp_path / "Assets" / "BmadLoop.meta").read_bytes() == (
        payload / "_folders" / "BmadLoop.meta"
    ).read_bytes()
    assert (tmp_path / "Assets" / "BmadLoop" / "Editor.meta").read_bytes() == (
        payload / "_folders" / "Editor.meta"
    ).read_bytes()
    # Unity owns Assets/ itself — the seeder never lays an Assets.meta over it
    assert not (tmp_path / "Assets.meta").exists()


# ------------------------------------------------------------------ idempotency


def test_seed_idempotent_reinstall_is_noop(tmp_path, monkeypatch):
    mod = _load_seeder()
    _set_env(monkeypatch, tmp_path)
    (tmp_path / "Assets").mkdir()
    assert mod.main() == 0

    # a local edit that keeps the (matching) version header must survive a re-run:
    # an equal version is left untouched.
    cs = tmp_path / _GUARD_DIR / "SceneAutoSaveGuard.cs"
    cs.write_text(cs.read_text() + "\n// local edit marker\n", encoding="utf-8")

    assert mod.main() == 0
    assert "// local edit marker" in cs.read_text()  # not overwritten


# --------------------------------------------------------------- version bump


def test_seed_version_bump_reinstalls(tmp_path, monkeypatch):
    mod = _load_seeder()
    _set_env(monkeypatch, tmp_path)
    editor = tmp_path / _GUARD_DIR
    editor.mkdir(parents=True)
    cs = editor / "SceneAutoSaveGuard.cs"
    cs.write_text("// bmad-loop-scene-guard-version: 0.9.0\n// stale guard\n", encoding="utf-8")

    assert mod.main() == 0

    payload = _payload(mod)
    assert "0.9.0" not in cs.read_text()  # the stale guard was replaced
    assert cs.read_bytes() == (payload / "SceneAutoSaveGuard.cs").read_bytes()
    # the reinstall also brought the meta companions along
    assert (editor / "SceneAutoSaveGuard.cs.meta").is_file()
    assert (editor / "BmadLoop.Unity.Editor.asmdef").is_file()


def test_seed_newer_target_version_is_left_alone(tmp_path, monkeypatch):
    mod = _load_seeder()
    _set_env(monkeypatch, tmp_path)
    editor = tmp_path / _GUARD_DIR
    editor.mkdir(parents=True)
    cs = editor / "SceneAutoSaveGuard.cs"
    # a hypothetical future guard version must never be downgraded
    cs.write_text("// bmad-loop-scene-guard-version: 99.0.0\n// future guard\n", encoding="utf-8")

    assert mod.main() == 0
    assert "99.0.0" in cs.read_text()  # untouched


# ------------------------------------------------------------ graceful skips


def test_seed_skips_when_no_asset_root(tmp_path, monkeypatch):
    mod = _load_seeder()
    _set_env(monkeypatch, tmp_path)  # no Assets/ created

    assert mod.main() == 0  # benign skip, not an error
    assert not (tmp_path / "Assets").exists()  # nothing scattered into a non-Unity tree


def test_seed_disabled_via_env(tmp_path, monkeypatch):
    mod = _load_seeder()
    _set_env(monkeypatch, tmp_path, install="0")
    (tmp_path / "Assets").mkdir()

    assert mod.main() == 0
    assert not (tmp_path / "Assets" / "BmadLoop").exists()  # seeding did not run


def test_seed_errors_without_worktree(tmp_path, monkeypatch):
    mod = _load_seeder()
    monkeypatch.delenv("BMAD_LOOP_WORKTREE", raising=False)
    monkeypatch.setenv("BMAD_LOOP_UNITY_INSTALL_SCENE_GUARD", "1")
    assert mod.main() == 2  # a real error (no project to seed)


# ---------------------------------------------------------------- custom dir


def test_seed_custom_dir_keys_folder_metas_by_name(tmp_path, monkeypatch):
    mod = _load_seeder()
    _set_env(monkeypatch, tmp_path, guard_dir="Assets/Vendor/BmadLoop/Editor")
    (tmp_path / "Assets").mkdir()

    assert mod.main() == 0

    editor = tmp_path / "Assets" / "Vendor" / "BmadLoop" / "Editor"
    assert (editor / "SceneAutoSaveGuard.cs").is_file()
    # the payload keys folder metas by folder NAME, so BmadLoop/ + Editor/ still get
    # their fixed-GUID metas even under a custom parent...
    assert (tmp_path / "Assets" / "Vendor" / "BmadLoop.meta").is_file()
    assert (editor.parent / "Editor.meta").is_file()
    # ...but Vendor/ has no payload meta, so Unity auto-generates that one (we ship none)
    assert not (tmp_path / "Assets" / "Vendor.meta").exists()


# ------------------------------------------------------------- invalid dirs


def test_seed_rejects_absolute_guard_dir(tmp_path, monkeypatch):
    mod = _load_seeder()
    outside = tmp_path / "outside"
    outside.mkdir()
    worktree = tmp_path / "wt"
    (worktree / "Assets").mkdir(parents=True)
    _set_env(monkeypatch, worktree, guard_dir=str(outside / "Editor"))

    assert mod.main() == 2  # a real error, not a crash (relative_to would raise)
    assert not (outside / "Editor").exists()  # nothing written at the absolute target


def test_seed_rejects_parent_traversal_guard_dir(tmp_path, monkeypatch):
    mod = _load_seeder()
    # the escape target's parent (tmp_path) exists, so an unguarded seeder would
    # really create escaped/ out there — the mutate-check is meaningful
    worktree = tmp_path / "wt"
    (worktree / "Assets").mkdir(parents=True)
    _set_env(monkeypatch, worktree, guard_dir="Assets/../../escaped/Editor")

    assert mod.main() == 2
    assert not (tmp_path / "escaped").exists()  # no write outside the worktree


def test_seed_rejects_root_naming_guard_dir(tmp_path, monkeypatch):
    mod = _load_seeder()
    worktree = tmp_path / "wt"
    (worktree / "Assets").mkdir(parents=True)
    # `main()` `.strip()`s the env var, so the trailing-SPACE spellings collapse to
    # "." and the pre-existing `not rel.parts` arm already caught them. These are
    # the ones that survive that strip: on Windows each names the worktree itself,
    # and the asset-root probe below the guard would then find the worktree (a real
    # directory) and pass, scattering the payload across the worktree root.
    for evil in ("...", "....", ". .", ".\\"):
        _set_env(monkeypatch, worktree, guard_dir=evil)
        assert mod.main() == 2, evil
    assert not (worktree / mod._GUARD_CS).exists()  # payload never hit the root
    assert not (worktree / "BmadLoop").exists()


def test_seed_rejects_windows_flavored_escapes_on_any_platform(tmp_path, monkeypatch):
    mod = _load_seeder()
    (tmp_path / "Assets").mkdir()
    for evil in ("C:\\evil", "C:evil", "\\evil", "..\\evil"):
        _set_env(monkeypatch, tmp_path, guard_dir=evil)
        assert mod.main() == 2, evil
    assert not (tmp_path / "Assets" / "BmadLoop").exists()  # nothing seeded


# ------------------------------------------------------------ version parsing


def test_parse_version_tuple_and_missing():
    mod = _load_seeder()
    assert mod._parse_version("// bmad-loop-scene-guard-version: 1.2.3\n") == (1, 2, 3)
    assert mod._parse_version("no header here") is None
