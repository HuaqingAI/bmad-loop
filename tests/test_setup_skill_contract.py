"""Contract guards for the shipped `bmad-loop-setup` skill.

Three independent contracts meet in this one directory, and all are easy to break
silently:

1. **The BMAD custom-source resolver contract.** The fork must use a module
   code that does not collide with BMAD-METHOD's official `bmad-loop` registry
   entry. The installer resolves this repository through `plugin-resolver.js`
   strategy 2 — "a skill whose directory name ends in `-setup`, carrying
   `assets/module.yaml` **and** `assets/module-help.csv`". If any of those three
   go missing, the custom module stops installing.

2. **Source provenance.** Setup must read the exact custom module entry from the
   BMAD manifest and reinstall the Python tool from that URL/path. Falling back
   to the official repository silently replaces the fork on setup or upgrade.

3. **No writes to the legacy BMAD config layout** (#258). Setup used to write
   `_bmad/config.yaml`, `_bmad/config.user.yaml` and a root
   `_bmad/module-help.csv` — files BMAD v6.10 never reads. The BMAD installer owns
   module registration; the skill's only `_bmad/` write is the per-module help CSV.
   These are prose assertions because the skill *is* prose — the instructions are
   the executable.
"""

import csv
import json
from pathlib import Path

import pytest
import yaml
from bmad_loop.install import MODULE_SKILLS

SKILL_DIR = "bmad-loop-setup"
MODULE_CODE = "huaqing-bmad-loop"
FORK_REPOSITORY = "https://github.com/HuaqingAI/bmad-loop.git"
REPO = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def skill_root():
    from importlib import resources

    return resources.files("bmad_loop.data").joinpath("skills").joinpath(SKILL_DIR)


@pytest.fixture(scope="module")
def skill_md(skill_root):
    return skill_root.joinpath("SKILL.md").read_text(encoding="utf-8")


def test_setup_skill_is_bundled():
    # the installer's strategy-2 match keys off the trailing "-setup"
    assert SKILL_DIR in MODULE_SKILLS
    assert SKILL_DIR.endswith("-setup")


@pytest.mark.parametrize("asset", ["module.yaml", "module-help.csv"])
def test_installer_required_assets_present(skill_root, asset):
    # plugin-resolver.js strategy 2 needs BOTH or bmad-loop stops resolving
    assert skill_root.joinpath("assets").joinpath(asset).is_file()


def test_custom_module_identity_avoids_official_registry_collision(skill_root):
    metadata = yaml.safe_load(
        skill_root.joinpath("assets", "module.yaml").read_text(encoding="utf-8")
    )
    assert metadata["code"] == MODULE_CODE
    assert metadata["code"] != "bmad-loop"
    assert metadata["tool_repository"] == FORK_REPOSITORY


def test_repo_root_module_descriptor_matches_canonical_asset(skill_root):
    assert (REPO / "module.yaml").read_bytes() == skill_root.joinpath(
        "assets", "module.yaml"
    ).read_bytes()


def test_marketplace_identifies_the_fork():
    marketplace = json.loads(
        (REPO / ".claude-plugin" / "marketplace.json").read_text(encoding="utf-8")
    )
    assert marketplace["name"] == MODULE_CODE
    assert marketplace["owner"]["name"] == "HuaqingAI"
    assert marketplace["repository"] == FORK_REPOSITORY.removesuffix(".git")
    assert [plugin["name"] for plugin in marketplace["plugins"]] == [MODULE_CODE]


def test_module_help_csv_shape(skill_root):
    """`mergeModuleHelpCatalogs` parses this positionally and keys rows off column 0."""
    raw = skill_root.joinpath("assets").joinpath("module-help.csv").read_text(encoding="utf-8")
    rows = list(csv.reader(raw.splitlines()))
    header, data = rows[0], rows[1:]
    assert header[0] == "module"
    assert len(header) == 13, f"help CSV column count changed: {header}"
    assert data, "module-help.csv must carry at least one skill row"
    for row in data:
        assert len(row) == len(header), f"ragged row: {row}"
        assert row[0] == "BMAD Loop Skills", f"unexpected module column: {row[0]!r}"


# The legacy layout (#258) and the pre-rename module code: both are gone, and the
# skill must not reintroduce either. Substring match on purpose — the failure mode
# is prose drifting back, not an exact string reappearing.
@pytest.mark.parametrize(
    "forbidden",
    [
        "config.yaml",
        "config.user.yaml",
        "merge-config.py",
        "merge-help-csv.py",
        "cleanup-legacy.py",
        "bauto",
        "bmad-auto",
        ".automator",
    ],
)
def test_skill_md_drops_legacy_layout(skill_md, forbidden):
    assert forbidden not in skill_md


def test_skill_md_registers_help_under_the_dynamic_module_code(skill_md):
    # The fork's collision-free code controls the BMAD module directory. Setup
    # reads it from module.yaml rather than silently writing into the official
    # module's `_bmad/bmad-loop/` registration.
    assert "{module-dir}/module-help.csv" in skill_md
    assert "_bmad/bmad-loop/" not in skill_md


def test_skill_md_reinstalls_from_the_exact_custom_source(skill_md):
    for required in [
        "tool_repository",
        "manifest.yaml",
        "name` exactly equals `{module-code}`",
        "localPath",
        "rawSource",
        "repoUrl",
        "entry's `source` is `custom`",
        'uv tool install --force --reinstall "bmad-loop[tui] @ {tool-reference}"',
    ]:
        assert required in skill_md

    assert "https://github.com/bmad-code-org/bmad-loop" not in skill_md
    assert "uv tool upgrade bmad-loop --reinstall" not in skill_md


def test_setup_skill_ships_no_scripts(skill_root):
    # all three PEP 723 scripts were retired with #258/#259; nothing left to invoke
    assert not skill_root.joinpath("scripts").is_dir()


def test_setup_skill_documents_namespaces_and_checkout_local_hooks(skill_md):
    assert "--local-hooks --no-skills" in skill_md
    assert "--namespace L0" in skill_md
    assert "git rm --cached" in skill_md
