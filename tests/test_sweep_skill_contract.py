"""Contract guards for the shipped `bmad-loop-sweep` skill."""

import pytest

SKILL_DIR = "bmad-loop-sweep"


@pytest.fixture(scope="module")
def skill_root():
    from importlib import resources

    return resources.files("bmad_loop.data").joinpath("skills").joinpath(SKILL_DIR)


def test_sweep_skill_states_bundle_name_contract(skill_root):
    """ABLATION A3: remove the primary skill constraint and this fails its first assertion."""
    skill_md = " ".join(skill_root.joinpath("SKILL.md").read_text(encoding="utf-8").split())
    automation_md = " ".join(
        skill_root.joinpath("automation-mode.md").read_text(encoding="utf-8").split()
    )

    assert "`name` matches `^[a-z0-9][a-z0-9-]{1,39}\\Z`" in skill_md
    assert "at most 40 characters" in skill_md
    assert (
        "otherwise-valid overlong bundle name or decision option `bundle_name` is truncated "
        "to 40 characters and journaled before validation"
    ) in automation_md
    assert "post-truncation name collisions still fail validation" in automation_md
