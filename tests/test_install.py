import io
import json
import os
import re
import stat
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest
from conftest import git

import bmad_loop.install as install_mod
from bmad_loop import verify
from bmad_loop.adapters.profile import get_profile
from bmad_loop.install import (
    BASE_SKILLS,
    DEV_BASE_SKILLS,
    MODULE_SKILLS,
    _copy_traversable,
    _git_version_at_least,
    _worktree_local_exclude,
    install_into,
    merge_hooks,
    missing_base_skills,
    provision_worktree,
)


def _install_skills(root, tree, catalog):
    """Lay down stubs of exactly ``catalog`` ({skill: (marker, ...)}) under root/tree.

    Takes the catalog explicitly so a test can build one precise upstream topology
    (e.g. the v6.10.0 shape: no `bmad-review`, no verification-gap) rather than the
    everything-installed superset."""
    for skill, markers in catalog.items():
        d = root / tree / skill
        d.mkdir(parents=True, exist_ok=True)
        (d / "SKILL.md").write_text(f"# {skill}\n", encoding="utf-8")
        for marker in markers:
            (d / marker).write_text("x\n", encoding="utf-8")


def _install_base_skills(root, tree=".claude/skills"):
    """Lay down stubs of the non-bundled upstream skills the orchestrator drives."""
    _install_skills(root, tree, BASE_SKILLS)


def _wt_private_exclude(wt):
    """The file the git-add shield writes: the exclude in the worktree's OWN
    gitdir (`.git/worktrees/<id>/info/exclude`), never the repo-wide one (#384).

    Asked of git rather than composed from `.git/worktrees/<basename>`: git
    appends a disambiguating number when the basename is already taken, so a
    hand-built path is right only by luck."""
    return Path(git(wt, "rev-parse", "--absolute-git-dir")) / "info" / "exclude"


def _registrations(profile, command="python3 /x/.bmad-loop/bmad_loop_hook.py {event}"):
    return {
        native: command.format(event=canonical)
        for native, canonical in profile.hooks.events.items()
    }


def test_merge_hooks_adds_all_events():
    profile = get_profile("claude")
    settings, changed = merge_hooks({}, _registrations(profile), profile.hooks.dialect)
    assert changed
    assert set(profile.hooks.events) <= set(settings["hooks"])


def test_merge_hooks_idempotent():
    profile = get_profile("claude")
    settings, _ = merge_hooks({}, _registrations(profile), profile.hooks.dialect)
    again, changed = merge_hooks(settings, _registrations(profile), profile.hooks.dialect)
    assert not changed
    for event in profile.hooks.events:
        assert len(again["hooks"][event]) == 1


def test_merge_hooks_preserves_existing():
    profile = get_profile("claude")
    existing = {
        "hooks": {"Stop": [{"hooks": [{"type": "command", "command": "echo hi"}]}]},
        "permissions": {"allow": ["Bash(ls)"]},
    }
    settings, changed = merge_hooks(existing, _registrations(profile), profile.hooks.dialect)
    assert changed
    assert settings["permissions"] == {"allow": ["Bash(ls)"]}
    commands = [
        handler["command"] for matcher in settings["hooks"]["Stop"] for handler in matcher["hooks"]
    ]
    assert "echo hi" in commands
    assert any("bmad_loop_hook" in c for c in commands)


def test_merge_hooks_gemini_entry_shape():
    profile = get_profile("gemini")
    settings, _ = merge_hooks({}, _registrations(profile), profile.hooks.dialect)
    entry = settings["hooks"]["AfterAgent"][0]
    assert entry["matcher"] == ""
    handler = entry["hooks"][0]
    assert handler["timeout"] == 60_000  # Gemini hook timeouts are milliseconds
    # registered under the native event but relaying the canonical name
    assert handler["command"].endswith("bmad_loop_hook.py Stop")


def test_merge_hooks_copilot_entry_shape():
    profile = get_profile("copilot")
    settings, _ = merge_hooks({}, _registrations(profile), profile.hooks.dialect)
    assert settings["version"] == 1  # Copilot hook configs are versioned
    # Copilot stores the handler dict directly in the event list (no "hooks" wrapper)
    handler = settings["hooks"]["agentStop"][0]
    assert handler["type"] == "command"
    assert handler["timeoutSec"] == 60  # Copilot hook timeouts are seconds
    # registered under the native event (agentStop) but relaying the canonical name
    assert handler["command"].endswith("bmad_loop_hook.py Stop")


def test_merge_hooks_copilot_idempotent():
    # the bare-handler shape must still dedupe on a re-run
    profile = get_profile("copilot")
    settings, _ = merge_hooks({}, _registrations(profile), profile.hooks.dialect)
    again, changed = merge_hooks(settings, _registrations(profile), profile.hooks.dialect)
    assert not changed
    for event in profile.hooks.events:
        assert len(again["hooks"][event]) == 1


def test_merge_hooks_antigravity_entry_shape():
    # agy keys hooks.json by hook NAME at the top level ("bmad-loop"), and its Stop
    # event is FLAT (handler dict directly, no matcher/hooks wrapper).
    profile = get_profile("antigravity")
    settings, _ = merge_hooks({}, _registrations(profile), profile.hooks.dialect)
    assert "hooks" not in settings  # not a "hooks"-wrapped dialect
    handler = settings["bmad-loop"]["Stop"][0]
    assert handler["type"] == "command"
    assert handler["timeout"] == 60  # agy hook timeouts are seconds
    assert handler["command"].endswith("bmad_loop_hook.py Stop")


def test_merge_hooks_antigravity_idempotent():
    profile = get_profile("antigravity")
    settings, _ = merge_hooks({}, _registrations(profile), profile.hooks.dialect)
    again, changed = merge_hooks(settings, _registrations(profile), profile.hooks.dialect)
    assert not changed
    for event in profile.hooks.events:
        assert len(again["bmad-loop"][event]) == 1


def test_merge_hooks_antigravity_appends_beside_existing_stop():
    # agy stores each event as a LIST of handlers; a hooks.json that already has a
    # bmad-loop group with the user's own Stop handler must keep it and gain ours.
    profile = get_profile("antigravity")
    existing = {"bmad-loop": {"Stop": [{"type": "command", "command": "echo mine", "timeout": 5}]}}
    settings, changed = merge_hooks(existing, _registrations(profile), profile.hooks.dialect)
    assert changed
    commands = [h["command"] for h in settings["bmad-loop"]["Stop"]]
    assert "echo mine" in commands
    assert any("bmad_loop_hook" in c for c in commands)
    assert len(settings["bmad-loop"]["Stop"]) == 2


def test_merge_hooks_unrelated_bmad_loop_path_does_not_suppress_relay():
    # Dedup keys on the bmad-loop script markers, not the broad "bmad_loop"
    # substring: an unrelated handler whose command merely mentions a
    # bmad_loop-containing path must not make init skip the relay — that would
    # leave `validate` (which detects on the narrow marker) un-passable, the
    # #159 failure class through the merge/detect seam.
    profile = get_profile("claude")
    existing = {
        "hooks": {
            "Stop": [
                {
                    "hooks": [
                        {"type": "command", "command": "python /home/me/bmad_loop_fork/notify.py"}
                    ]
                }
            ]
        }
    }
    settings, changed = merge_hooks(existing, _registrations(profile), profile.hooks.dialect)
    assert changed
    commands = [
        handler["command"] for matcher in settings["hooks"]["Stop"] for handler in matcher["hooks"]
    ]
    assert "python /home/me/bmad_loop_fork/notify.py" in commands
    assert any("bmad_loop_hook" in c for c in commands)


def test_merge_hooks_antigravity_unrelated_bmad_loop_path_does_not_suppress_relay():
    # Same guarantee for agy's flat top-level-group shape (the other dedup branch).
    profile = get_profile("antigravity")
    existing = {
        "bmad-loop": {
            "Stop": [{"type": "command", "command": "python /home/me/bmad_loop_fork/notify.py"}]
        }
    }
    settings, changed = merge_hooks(existing, _registrations(profile), profile.hooks.dialect)
    assert changed
    commands = [h["command"] for h in settings["bmad-loop"]["Stop"]]
    assert "python /home/me/bmad_loop_fork/notify.py" in commands
    assert any("bmad_loop_hook" in c for c in commands)


def test_merge_hooks_antigravity_rejects_malformed_shape():
    # a malformed pre-existing hooks.json yields a clear ProfileError, not an
    # opaque AttributeError during init.
    import pytest

    from bmad_loop.adapters.profile import ProfileError

    profile = get_profile("antigravity")
    with pytest.raises(ProfileError):
        merge_hooks({"bmad-loop": "oops"}, _registrations(profile), profile.hooks.dialect)
    with pytest.raises(ProfileError):
        merge_hooks({"bmad-loop": {"Stop": "oops"}}, _registrations(profile), profile.hooks.dialect)


def test_merge_hooks_antigravity_tolerates_non_string_command():
    # a pre-existing handler whose "command" is a non-string (e.g. None) must not
    # crash the idempotency dedupe.
    profile = get_profile("antigravity")
    existing = {"bmad-loop": {"Stop": [{"type": "command", "command": None}]}}
    settings, changed = merge_hooks(existing, _registrations(profile), profile.hooks.dialect)
    assert changed
    assert any("bmad_loop_hook" in (h.get("command") or "") for h in settings["bmad-loop"]["Stop"])


def test_merge_hooks_antigravity_preserves_other_groups():
    # user/plugin hook groups sit alongside "bmad-loop" and must survive.
    profile = get_profile("antigravity")
    existing = {"lint-checker": {"PostToolUse": [{"matcher": "run_command", "hooks": []}]}}
    settings, changed = merge_hooks(existing, _registrations(profile), profile.hooks.dialect)
    assert changed
    assert settings["lint-checker"] == {"PostToolUse": [{"matcher": "run_command", "hooks": []}]}
    assert settings["bmad-loop"]["Stop"][0]["command"].endswith("bmad_loop_hook.py Stop")


def test_install_does_not_clobber_existing_policy(tmp_path):
    """An existing .bmad-loop/policy.toml is per-machine state: init must leave it
    alone rather than resetting it to the template."""
    current = tmp_path / ".bmad-loop" / "policy.toml"
    current.parent.mkdir(parents=True)
    current.write_text("CURRENT", encoding="utf-8")

    assert install_into(tmp_path) == 0
    assert current.read_text() == "CURRENT"


def test_copilot_profile_render_prompt():
    # {skill} must expand plainly (no codex-style $ prefix) into the SKILL.md path
    profile = get_profile("copilot")
    rendered = profile.render_prompt("/bmad-dev-auto 1-2-a")
    assert ".agents/skills/bmad-dev-auto/SKILL.md" in rendered
    assert "1-2-a" in rendered


def test_install_into_copilot(tmp_path):
    assert install_into(tmp_path, clis=("copilot",)) == 0
    settings = json.loads((tmp_path / ".github" / "copilot" / "settings.json").read_text())
    assert settings["version"] == 1
    # registered under the camelCase native names Copilot 1.0.63 actually fires
    # (agentStop is turn-end; PascalCase Stop never fires); relay still gets canonical
    assert set(settings["hooks"]) == {"agentStop", "sessionStart", "sessionEnd"}
    cmd = settings["hooks"]["agentStop"][0]["command"]
    # absolute path baked in (no $CLAUDE_PROJECT_DIR equivalent in copilot)
    assert str(tmp_path.resolve()) in cmd and cmd.endswith(" Stop")
    # skills land in the shared .agents/skills tree
    for skill in MODULE_SKILLS:
        assert (tmp_path / ".agents" / "skills" / skill / "SKILL.md").is_file()

    # idempotent re-run does not duplicate the bare handler
    assert install_into(tmp_path, clis=("copilot",)) == 0
    settings = json.loads((tmp_path / ".github" / "copilot" / "settings.json").read_text())
    assert len(settings["hooks"]["agentStop"]) == 1


def test_install_into_full(tmp_path):
    assert install_into(tmp_path) == 0
    assert (tmp_path / ".bmad-loop" / "bmad_loop_hook.py").is_file()
    assert (tmp_path / ".bmad-loop" / "policy.toml").is_file()
    settings = json.loads((tmp_path / ".claude" / "settings.json").read_text())
    assert "Stop" in settings["hooks"]
    gitignore = (tmp_path / ".gitignore").read_text()
    assert ".bmad-loop/runs/" in gitignore
    assert ".bmad-loop/cache/" in gitignore  # engine plugins' rebuildable caches
    assert ".bmad-loop/policy.toml" in gitignore  # per-machine config ([mux] backend)

    # all bundled skills land in claude's tree, with nested files intact
    skills_dir = tmp_path / ".claude" / "skills"
    for skill in MODULE_SKILLS:
        assert (skills_dir / skill / "SKILL.md").is_file()
    assert (skills_dir / "bmad-loop-sweep" / "deferred-work-format.md").is_file()

    # second run: idempotent, does not duplicate
    assert install_into(tmp_path) == 0
    settings = json.loads((tmp_path / ".claude" / "settings.json").read_text())
    assert len(settings["hooks"]["Stop"]) == 1
    final_gitignore = (tmp_path / ".gitignore").read_text()
    assert final_gitignore.count(".bmad-loop/runs/") == 1
    assert final_gitignore.count(".bmad-loop/cache/") == 1
    assert final_gitignore.count(".bmad-loop/policy.toml") == 1


def test_install_into_warns_when_policy_is_tracked(tmp_path, capsys):
    """A .gitignore entry doesn't untrack an already-committed policy.toml:
    upgrading repos get the one-time `git rm --cached` hint."""
    import subprocess

    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    policy_file = tmp_path / ".bmad-loop" / "policy.toml"
    policy_file.parent.mkdir(parents=True)
    policy_file.write_text("[gates]\n", encoding="utf-8")
    subprocess.run(["git", "add", ".bmad-loop/policy.toml"], cwd=tmp_path, check=True)
    assert install_into(tmp_path) == 0
    assert "git rm --cached .bmad-loop/policy.toml" in capsys.readouterr().out


def test_install_into_no_tracking_warning_outside_a_repo(tmp_path, capsys):
    assert install_into(tmp_path) == 0
    assert "git rm --cached" not in capsys.readouterr().out


def test_hook_command_uses_selected_process_host(tmp_path, monkeypatch):
    # The hook interpreter is platform-selected: forcing the Windows host swaps the
    # registered command's prefix without `install` branching on sys.platform.
    from bmad_loop.process_host import get_process_host

    monkeypatch.setenv("BMAD_LOOP_PROCESS_HOST", "windows")
    get_process_host.cache_clear()
    try:
        assert install_into(tmp_path) == 0
        settings = json.loads((tmp_path / ".claude" / "settings.json").read_text())
        cmd = settings["hooks"]["Stop"][0]["hooks"][0]["command"]
        assert cmd.startswith("uv run --no-project python ")
    finally:
        monkeypatch.delenv("BMAD_LOOP_PROCESS_HOST", raising=False)
        get_process_host.cache_clear()


def test_install_into_multiple_clis(tmp_path):
    assert install_into(tmp_path, clis=("codex", "gemini")) == 0

    codex_hooks = json.loads((tmp_path / ".codex" / "hooks.json").read_text())
    assert set(codex_hooks["hooks"]) == {"SessionStart", "Stop"}
    cmd = codex_hooks["hooks"]["Stop"][0]["hooks"][0]["command"]
    # absolute path (no $CLAUDE_PROJECT_DIR equivalent in codex/gemini)
    assert str(tmp_path.resolve()) in cmd and cmd.endswith(" Stop")

    gemini_settings = json.loads((tmp_path / ".gemini" / "settings.json").read_text())
    assert set(gemini_settings["hooks"]) == {"SessionStart", "AfterAgent", "SessionEnd"}

    # idempotent across both
    assert install_into(tmp_path, clis=("codex", "gemini")) == 0
    codex_hooks = json.loads((tmp_path / ".codex" / "hooks.json").read_text())
    assert len(codex_hooks["hooks"]["Stop"]) == 1


def test_install_skills_dedupes_agents_tree(tmp_path):
    # codex and gemini share .agents/skills — install once there, not under .claude
    assert install_into(tmp_path, clis=("codex", "gemini")) == 0
    for skill in MODULE_SKILLS:
        assert (tmp_path / ".agents" / "skills" / skill / "SKILL.md").is_file()
    assert not (tmp_path / ".claude" / "skills").exists()


def test_install_skills_skip_existing(tmp_path):
    skill_md = tmp_path / ".claude" / "skills" / "bmad-loop-sweep" / "SKILL.md"
    skill_md.parent.mkdir(parents=True)
    skill_md.write_text("CUSTOM", encoding="utf-8")
    # default run must not clobber an existing skill dir
    assert install_into(tmp_path) == 0
    assert skill_md.read_text() == "CUSTOM"
    # but a skill that was absent still gets installed
    assert (tmp_path / ".claude" / "skills" / "bmad-loop-resolve" / "SKILL.md").is_file()


def test_install_skills_force(tmp_path):
    skill_md = tmp_path / ".claude" / "skills" / "bmad-loop-resolve" / "SKILL.md"
    skill_md.parent.mkdir(parents=True)
    skill_md.write_text("CUSTOM", encoding="utf-8")
    assert install_into(tmp_path, force_skills=True) == 0
    assert skill_md.read_text() != "CUSTOM"


def test_install_no_skills(tmp_path):
    assert install_into(tmp_path, skills=False) == 0
    # hooks still installed, but no skill tree created
    assert (tmp_path / ".claude" / "settings.json").is_file()
    assert not (tmp_path / ".claude" / "skills").exists()


def test_install_unknown_cli(tmp_path):
    assert install_into(tmp_path, clis=("acme-cli",)) == 1
    assert not (tmp_path / ".bmad-loop").exists()


def test_install_resolves_legacy_alias(tmp_path):
    assert install_into(tmp_path, clis=("claude-code-tmux",)) == 0
    assert (tmp_path / ".claude" / "settings.json").is_file()


def test_provision_worktree_lays_down_skills_and_hook(tmp_path):
    """A worktree must receive the bmad-loop-* skills + signal hook even though
    those dirs are gitignored (absent from a fresh checkout), or the bundled
    skills are missing and the Stop hook never fires."""
    wt, repo = tmp_path / "wt", tmp_path / "repo"
    claude = get_profile("claude")
    provision_worktree(wt, [claude], repo)

    # skills installed into the claude skill tree
    for skill in MODULE_SKILLS:
        assert (wt / claude.skill_tree / skill / "SKILL.md").is_file()
    # hook registered, baked to the MAIN repo's relay (absolute) — nothing written
    # into the worktree's .bmad-loop/ (which a project may not gitignore)
    settings = json.loads((wt / claude.hooks.config_path).read_text())
    assert set(claude.hooks.events) <= set(settings["hooks"])
    cmd = settings["hooks"]["Stop"][0]["hooks"][0]["command"]
    assert str((repo / ".bmad-loop" / "bmad_loop_hook.py")) in cmd
    assert not (wt / ".bmad-loop").exists()


def test_provision_worktree_covers_multiple_profiles(tmp_path):
    """Dev=claude + review=codex provisions both skill trees (.claude/skills and
    .agents/skills) and both hook configs."""
    wt, repo = tmp_path / "wt", tmp_path / "repo"
    claude, codex = get_profile("claude"), get_profile("codex")
    provision_worktree(wt, [claude, codex], repo)

    assert (wt / claude.skill_tree / "bmad-loop-sweep" / "SKILL.md").is_file()
    assert (wt / codex.skill_tree / "bmad-loop-sweep" / "SKILL.md").is_file()
    assert (wt / claude.hooks.config_path).is_file()
    assert (wt / codex.hooks.config_path).is_file()


def test_provision_worktree_does_not_clobber_existing_skill(tmp_path):
    """A skill the checkout already carries (project commits its own skill tree)
    is left untouched, so no diff is merged back."""
    wt, repo = tmp_path / "wt", tmp_path / "repo"
    claude = get_profile("claude")
    existing = wt / claude.skill_tree / "bmad-loop-sweep" / "SKILL.md"
    existing.parent.mkdir(parents=True)
    existing.write_text("COMMITTED", encoding="utf-8")

    provision_worktree(wt, [claude], repo)
    assert existing.read_text() == "COMMITTED"
    # a skill that was absent is still laid down
    assert (wt / claude.skill_tree / "bmad-loop-resolve" / "SKILL.md").is_file()


def test_provision_worktree_empty_profiles_is_noop(tmp_path):
    provision_worktree(tmp_path / "wt", [], tmp_path / "repo")
    assert not (tmp_path / "wt").exists()


def test_provision_worktree_copies_base_skills_from_repo(tmp_path):
    """The upstream skills the orchestrator drives aren't bundled in the wheel, so
    the worktree must get them copied from the MAIN repo's installed tree."""
    wt, repo = tmp_path / "wt", tmp_path / "repo"
    claude = get_profile("claude")
    _install_base_skills(repo, claude.skill_tree)

    provision_worktree(wt, [claude], repo)

    for skill in BASE_SKILLS:
        assert (wt / claude.skill_tree / skill / "SKILL.md").is_file()
    # the dev primitive's marker file came along too
    assert (wt / claude.skill_tree / "bmad-dev-auto" / "step-04-review.md").is_file()


def test_missing_base_skills_reports_absent_and_incomplete(tmp_path):
    claude = get_profile("claude")
    # nothing installed → dev primitive + the two inline review layers reported
    # missing (the hunters are required whenever the merged reviewer is absent —
    # bmad-dev-auto's step-04 invokes them on every run)
    problems = missing_base_skills(tmp_path, [claude.skill_tree])
    assert len(problems) == 3
    assert all("BMAD-METHOD >= 6.10.0" in p.message for p in problems)

    # install everything → no problems
    _install_base_skills(tmp_path, claude.skill_tree)
    assert missing_base_skills(tmp_path, [claude.skill_tree]) == []

    # remove the dev primitive's step-file marker → reported as incomplete
    (tmp_path / claude.skill_tree / "bmad-dev-auto" / "step-04-review.md").unlink()
    problems = missing_base_skills(tmp_path, [claude.skill_tree])
    assert len(problems) == 1
    assert "incomplete" in problems[0].message
    assert "step-04-review.md" in problems[0].message

    # restore it, then drop customize.toml (the review-layer config marker,
    # BMAD-METHOD #2535/#2550) → a pre-July bmm install is caught as incomplete
    (tmp_path / claude.skill_tree / "bmad-dev-auto" / "step-04-review.md").write_text("x\n")
    (tmp_path / claude.skill_tree / "bmad-dev-auto" / "customize.toml").unlink()
    problems = missing_base_skills(tmp_path, [claude.skill_tree])
    assert len(problems) == 1
    assert "incomplete" in problems[0].message
    assert "customize.toml" in problems[0].message

    # #260: verification-gap is NOT a requirement — no tagged BMAD-METHOD release
    # ships it, so removing it from an otherwise complete tree must still pass
    _install_base_skills(tmp_path, claude.skill_tree)  # re-complete everything
    import shutil as _shutil

    _shutil.rmtree(tmp_path / claude.skill_tree / "bmad-review-verification-gap")
    assert missing_base_skills(tmp_path, [claude.skill_tree]) == []


def test_missing_stories_support_probes_step01_content(tmp_path):
    from bmad_loop.install import (
        STORIES_PROBE_FILE,
        STORIES_PROBE_SKILL,
        missing_stories_support,
    )

    claude = get_profile("claude")
    tree = claude.skill_tree
    step01 = tmp_path / tree / STORIES_PROBE_SKILL / STORIES_PROBE_FILE

    # step-01 absent → reported (older/half install)
    problems = missing_stories_support(tmp_path, [tree])
    assert len(problems) == 1 and "not found" in problems[0].message

    # present but WITHOUT the folder+id dispatch marker (a pre-#2549 skill)
    step01.parent.mkdir(parents=True, exist_ok=True)
    step01.write_text("# Step 1\nold clarify-and-route, no dispatch protocol\n", encoding="utf-8")
    problems = missing_stories_support(tmp_path, [tree])
    assert len(problems) == 1 and "folder+id dispatch" in problems[0].message

    # present WITH the marker → OK
    step01.write_text("route a **folder+id dispatch** invocation\n", encoding="utf-8")
    assert missing_stories_support(tmp_path, [tree]) == []


def test_missing_base_skills_findings_carry_ids_and_detail(tmp_path):
    """#205: the problems are Findings, so `validate --json` can key on the check id
    rather than on remediation prose. The two failure modes are distinct ids, and
    `missing_markers` is a list — the message's ", " join is a rendering of it, and
    a consumer must not have to split a separator the message is free to change."""
    from bmad_loop.checks import VALIDATE_CHECKS

    claude = get_profile("claude")
    absent = missing_base_skills(tmp_path, [claude.skill_tree])
    assert {f.check for f in absent} == {"skills.base-missing"}
    assert all(f.severity == "problem" for f in absent)
    assert all(f.check in VALIDATE_CHECKS for f in absent)
    assert {f.detail["skill"] for f in absent} == {
        "bmad-dev-auto",
        "bmad-review-adversarial-general",
        "bmad-review-edge-case-hunter",
    }
    assert all(f.detail["tree"] == claude.skill_tree for f in absent)

    _install_base_skills(tmp_path, claude.skill_tree)
    (tmp_path / claude.skill_tree / "bmad-dev-auto" / "step-04-review.md").unlink()
    (tmp_path / claude.skill_tree / "bmad-dev-auto" / "customize.toml").unlink()
    incomplete = missing_base_skills(tmp_path, [claude.skill_tree])
    assert len(incomplete) == 1
    assert incomplete[0].check == "skills.base-incomplete"
    # a LIST of markers, not the joined string the message renders
    assert incomplete[0].detail["missing_markers"] == ["step-04-review.md", "customize.toml"]
    for marker in incomplete[0].detail["missing_markers"]:
        assert marker in incomplete[0].message


def test_merged_bmad_review_satisfies_review_layers(tmp_path):
    """#260: post-consolidation bmm installs ship the merged `bmad-review` skill, with
    the standalone hunter IDs as thin forwarders to it. The merged reviewer provides
    every lens itself, so a tree carrying it needs none of the hunters."""
    claude = get_profile("claude")
    _install_skills(
        tmp_path,
        claude.skill_tree,
        {"bmad-dev-auto": DEV_BASE_SKILLS["bmad-dev-auto"], "bmad-review": ()},
    )
    assert missing_base_skills(tmp_path, [claude.skill_tree]) == []

    # ...but it never substitutes for the dev primitive
    import shutil as _shutil

    _shutil.rmtree(tmp_path / claude.skill_tree / "bmad-dev-auto")
    problems = missing_base_skills(tmp_path, [claude.skill_tree])
    assert len(problems) == 1
    assert problems[0].detail["skill"] == "bmad-dev-auto"


def test_verification_gap_never_required(tmp_path):
    """#260: the latest-release (v6.10.0) shape — dev primitive + the two review
    layers that release actually ships, no `bmad-review-verification-gap` (no tagged
    release has ever shipped it) and no merged reviewer — must validate. Requiring it
    made `validate` (and the run/resume/sweep preflight) unsatisfiable everywhere."""
    claude = get_profile("claude")
    _install_skills(tmp_path, claude.skill_tree, DEV_BASE_SKILLS)
    # guard the topology: neither escape hatch is present, so [] below is earned by
    # verification-gap not being required, not by the merged-reviewer bypass
    assert not (tmp_path / claude.skill_tree / "bmad-review-verification-gap").exists()
    assert not (tmp_path / claude.skill_tree / "bmad-review").exists()

    assert missing_base_skills(tmp_path, [claude.skill_tree]) == []


def test_review_hunter_missing_without_merged_review_reported(tmp_path):
    """A genuinely broken pre-consolidation install still fails — and its message must
    not misdiagnose the cause as "bmm is not installed" (#260): bmm is exactly what
    ships the layer, so a user whose bmm is installed could not act on the old line."""
    claude = get_profile("claude")
    _install_skills(
        tmp_path,
        claude.skill_tree,
        {k: v for k, v in DEV_BASE_SKILLS.items() if k != "bmad-review-edge-case-hunter"},
    )

    problems = missing_base_skills(tmp_path, [claude.skill_tree])
    assert len(problems) == 1
    assert problems[0].check == "skills.base-missing"
    assert problems[0].severity == "problem"
    assert problems[0].detail == {
        "tree": claude.skill_tree,
        "skill": "bmad-review-edge-case-hunter",
    }
    assert "bmad-review-edge-case-hunter" in problems[0].message
    assert "install the BMad Method" not in problems[0].message
    assert "update bmm" in problems[0].message


# Verbatim shape of BMAD-METHOD main's bmad-dev-auto/customize.toml: four review
# layers, three invoking the merged `bmad-review` with one lens each, plus
# intent-alignment — a self-contained prompt that invokes no skill at all.
LAYER_CUSTOMIZE = """
[workflow]
implementation_handoff = "irrelevant here"

[[workflow.review_layers]]
id = "blind-hunter"
name = "Blind Hunter"
instruction = '''
Launch a subagent with no prior conversation context, with this prompt:

> Invoke the `bmad-review` skill with only the `adversarial` lens on this diff:
>
> {diff_output}
'''

[[workflow.review_layers]]
id = "edge-case-hunter"
name = "Edge Case Hunter"
instruction = '''
> Invoke the `bmad-review` skill with only the `edge-case-hunter` lens on this diff:
>
> {diff_output}
'''

[[workflow.review_layers]]
id = "verification-gap"
name = "Verification Gap Reviewer"
instruction = '''
> Invoke the `bmad-review` skill with only the `verification-gap` lens on this diff:
>
> {diff_output}
'''

[[workflow.review_layers]]
id = "intent-alignment"
name = "Intent Alignment Auditor"
instruction = '''
> You are an intent-alignment auditor. Here is the diff:
>
> {diff_output}
'''
"""

# Pre-consolidation (v6.10.0) shape: a valid customize.toml that simply has no
# review_layers section, and a step-04 that names the two reviewers it invokes.
PRE_LAYER_CUSTOMIZE = """
[workflow]
activation_steps_prepend = []
persistent_facts = ["file:{project-root}/**/project-context.md"]
on_complete = ""
"""

STEP04_NAMED = """
### Step 2: Review layers

- Launch a subagent, with this prompt:
  > Invoke the `bmad-review-adversarial-general` skill on this diff:
  > {diff_output}
- Launch a subagent, with this prompt:
  > Invoke the `bmad-review-edge-case-hunter` skill on this diff:
  > {diff_output}
"""


def _install_dev_auto(root, tree, *, customize="x\n", step04="x\n"):
    """Install bmad-dev-auto with real customize.toml / step-04 content, so the
    preflight reads the review shape it would read on a real install."""
    d = root / tree / "bmad-dev-auto"
    d.mkdir(parents=True, exist_ok=True)
    (d / "SKILL.md").write_text("# bmad-dev-auto\n", encoding="utf-8")
    (d / "customize.toml").write_text(customize, encoding="utf-8")
    (d / "step-04-review.md").write_text(step04, encoding="utf-8")
    return d


def test_layer_driven_review_requires_the_merged_skill_it_names(tmp_path):
    """Post-consolidation topology: the layers invoke `bmad-review` by name, so that
    skill — and none of the standalone hunters — is what the tree must carry."""
    claude = get_profile("claude")
    _install_dev_auto(tmp_path, claude.skill_tree, customize=LAYER_CUSTOMIZE)
    _install_skills(tmp_path, claude.skill_tree, {"bmad-review": ()})

    assert missing_base_skills(tmp_path, [claude.skill_tree]) == []


def test_layer_driven_review_reports_unresolvable_layer(tmp_path):
    """The gap this closes: a project whose customize.toml is post-consolidation but
    whose skills are the pre-consolidation standalone hunters. The old preflight was
    green here while three of the four layers would fail on every dev run."""
    claude = get_profile("claude")
    _install_dev_auto(tmp_path, claude.skill_tree, customize=LAYER_CUSTOMIZE)
    _install_skills(
        tmp_path,
        claude.skill_tree,
        {
            "bmad-review-adversarial-general": (),
            "bmad-review-edge-case-hunter": (),
            "bmad-review-verification-gap": (),
        },
    )

    problems = missing_base_skills(tmp_path, [claude.skill_tree])
    assert len(problems) == 1
    assert problems[0].check == "skills.review-layer-missing"
    assert problems[0].severity == "problem"
    assert problems[0].detail["skill"] == "bmad-review"
    # the three lens layers name it; intent-alignment invokes no skill at all
    assert problems[0].detail["layers"] == [
        "blind-hunter",
        "edge-case-hunter",
        "verification-gap",
    ]
    assert problems[0].detail["source"] == "customize.toml"
    assert "bmad-review" in problems[0].message


def test_review_layer_check_id_is_registered(tmp_path):
    from bmad_loop.checks import VALIDATE_CHECKS

    assert "skills.review-layer-missing" in VALIDATE_CHECKS


def test_pre_consolidation_step04_requires_the_skills_it_names(tmp_path):
    """v6.10.0 shape: no review_layers, so the requirement comes from the two skills
    step-04 invokes by name — and a tree carrying only the merged reviewer does NOT
    satisfy them, because that step-04 names the hunters, not `bmad-review`."""
    claude = get_profile("claude")
    tree = claude.skill_tree
    _install_dev_auto(tmp_path, tree, customize=PRE_LAYER_CUSTOMIZE, step04=STEP04_NAMED)
    _install_skills(
        tmp_path,
        tree,
        {"bmad-review-adversarial-general": (), "bmad-review-edge-case-hunter": ()},
    )
    assert missing_base_skills(tmp_path, [tree]) == []

    import shutil as _shutil

    _shutil.rmtree(tmp_path / tree / "bmad-review-edge-case-hunter")
    problems = missing_base_skills(tmp_path, [tree])
    assert len(problems) == 1
    assert problems[0].check == "skills.review-layer-missing"
    assert problems[0].detail["skill"] == "bmad-review-edge-case-hunter"
    assert problems[0].detail["layers"] == []
    assert problems[0].detail["source"] == "step-04-review.md"


def test_disabled_review_layer_is_not_required(tmp_path):
    """An empty `instruction` disables a layer — its skill must not be required."""
    claude = get_profile("claude")
    disabled = LAYER_CUSTOMIZE.replace(
        """instruction = '''
> Invoke the `bmad-review` skill with only the `verification-gap` lens on this diff:
>
> {diff_output}
'''""",
        'instruction = ""',
    )
    _install_dev_auto(tmp_path, claude.skill_tree, customize=disabled)
    _install_skills(tmp_path, claude.skill_tree, {"bmad-review": ()})

    assert missing_base_skills(tmp_path, [claude.skill_tree]) == []


def test_project_override_replaces_review_layer(tmp_path):
    """`_bmad/custom/bmad-dev-auto.toml` merges arrays of tables by `id`. A layer
    overridden to run an external reviewer no longer requires the default's skill —
    requiring it anyway would be exactly the kind of false FAIL #260 was."""
    claude = get_profile("claude")
    only_one_layer = LAYER_CUSTOMIZE.split("[[workflow.review_layers]]")[0] + (
        """[[workflow.review_layers]]
id = "blind-hunter"
instruction = '''
> Invoke the `bmad-review` skill with only the `adversarial` lens on this diff:
'''
"""
    )
    _install_dev_auto(tmp_path, claude.skill_tree, customize=only_one_layer)
    assert len(missing_base_skills(tmp_path, [claude.skill_tree])) == 1

    override = tmp_path / "_bmad" / "custom" / "bmad-dev-auto.toml"
    override.parent.mkdir(parents=True, exist_ok=True)
    override.write_text(
        """[[workflow.review_layers]]
id = "blind-hunter"
instruction = "Run `my-external-reviewer` via bash on the diff."
""",
        encoding="utf-8",
    )
    assert missing_base_skills(tmp_path, [claude.skill_tree]) == []


def test_unreadable_customize_falls_back_to_static_catalog(tmp_path):
    """A malformed customize.toml must not crash the preflight, and must not be read
    as "no layers configured" either — fall back to the static catalog."""
    claude = get_profile("claude")
    _install_dev_auto(tmp_path, claude.skill_tree, customize="this is not = valid toml [[[")

    problems = missing_base_skills(tmp_path, [claude.skill_tree])
    assert {p.detail["skill"] for p in problems} == {
        "bmad-review-adversarial-general",
        "bmad-review-edge-case-hunter",
    }
    assert {p.check for p in problems} == {"skills.base-missing"}

    # ...and the merged reviewer still satisfies that fallback
    _install_skills(tmp_path, claude.skill_tree, {"bmad-review": ()})
    assert missing_base_skills(tmp_path, [claude.skill_tree]) == []


# --- derivation vs. what the run really resolves (PR #283 review) -------------
#
# Everything below pins the preflight to BMAD's own resolver and to step-04's own
# skip rules. The failure mode being guarded is asymmetric: requiring a skill the
# run never invokes is a false FAIL (#260), and accepting a layer the run cannot
# resolve is a green validate followed by a broken review on every story.


def _write_override(root, body, *, user=False):
    """A project override of bmad-dev-auto's shipped customize.toml."""
    suffix = "user.toml" if user else "toml"
    path = root / "_bmad" / "custom" / f"bmad-dev-auto.{suffix}"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    return path


def _layer(layer_id, skill, *, key="id", when=None, phrasing="Invoke the"):
    when_line = f'when = "{when}"\n' if when else ""
    return f"""
[[workflow.review_layers]]
{key} = "{layer_id}"
{when_line}instruction = "{phrasing} `{skill}` skill on this diff."
"""


def _severities(findings):
    return {(f.check, f.severity) for f in findings}


def test_appended_override_layer_requires_the_skill_it_names(tmp_path):
    """A new `id` appends rather than replaces, so an override that adds a reviewer
    adds a requirement — the run will invoke it, so the preflight must too."""
    claude = get_profile("claude")
    _install_dev_auto(tmp_path, claude.skill_tree, customize=LAYER_CUSTOMIZE)
    _install_skills(tmp_path, claude.skill_tree, {"bmad-review": ()})
    assert missing_base_skills(tmp_path, [claude.skill_tree]) == []

    _write_override(tmp_path, _layer("house-style", "bmad-review-company"))
    problems = missing_base_skills(tmp_path, [claude.skill_tree])
    assert len(problems) == 1
    assert problems[0].check == "skills.review-layer-missing"
    assert problems[0].severity == "problem"
    assert problems[0].detail["skill"] == "bmad-review-company"
    assert problems[0].detail["layers"] == ["house-style"]

    # ...and installing it clears the problem, rather than the skill being
    # unreachable because it is not in any catalog this package pins.
    _install_skills(tmp_path, claude.skill_tree, {"bmad-review-company": ()})
    assert missing_base_skills(tmp_path, [claude.skill_tree]) == []


def test_standalone_verification_gap_required_when_a_layer_names_it(tmp_path):
    """Dropping bmad-review-verification-gap from the static catalog must not make
    it unrequirable: a project whose layers DO name it still needs it installed."""
    claude = get_profile("claude")
    customize = "[workflow]\n" + _layer("verification-gap", "bmad-review-verification-gap")
    _install_dev_auto(tmp_path, claude.skill_tree, customize=customize)

    problems = missing_base_skills(tmp_path, [claude.skill_tree])
    assert [p.detail["skill"] for p in problems] == ["bmad-review-verification-gap"]
    assert problems[0].severity == "problem"

    _install_skills(tmp_path, claude.skill_tree, {"bmad-review-verification-gap": ()})
    assert missing_base_skills(tmp_path, [claude.skill_tree]) == []


def test_pre_and_post_consolidation_trees_resolve_independently(tmp_path):
    """A project can carry a post-consolidation .claude tree and a pre-merge .agents
    one at once. Each tree's requirement comes from ITS OWN installed skill."""
    claude, codex = get_profile("claude"), get_profile("codex")
    assert claude.skill_tree != codex.skill_tree

    _install_dev_auto(tmp_path, claude.skill_tree, customize=LAYER_CUSTOMIZE)
    _install_skills(tmp_path, claude.skill_tree, {"bmad-review": ()})
    _install_dev_auto(
        tmp_path, codex.skill_tree, customize=PRE_LAYER_CUSTOMIZE, step04=STEP04_NAMED
    )
    _install_skills(
        tmp_path,
        codex.skill_tree,
        {"bmad-review-adversarial-general": (), "bmad-review-edge-case-hunter": ()},
    )
    assert missing_base_skills(tmp_path, [claude.skill_tree, codex.skill_tree]) == []

    # the merged reviewer in the OTHER tree must not satisfy the pre-merge one
    import shutil as _shutil

    _shutil.rmtree(tmp_path / codex.skill_tree / "bmad-review-edge-case-hunter")
    problems = missing_base_skills(tmp_path, [claude.skill_tree, codex.skill_tree])
    assert len(problems) == 1
    assert problems[0].detail["tree"] == codex.skill_tree
    assert problems[0].detail["skill"] == "bmad-review-edge-case-hunter"


def test_malformed_override_warns_and_still_resolves_base_layers(tmp_path):
    """BMAD's resolver warns on an unparseable override and carries on with the
    layers below it. Falling back to a static catalog instead would preflight a
    requirement set the run does not use."""
    claude = get_profile("claude")
    _install_dev_auto(tmp_path, claude.skill_tree, customize=LAYER_CUSTOMIZE)
    _write_override(tmp_path, "this is not = valid toml [[[")

    findings = missing_base_skills(tmp_path, [claude.skill_tree])
    assert ("skills.customize-unreadable", "warning") in _severities(findings)
    # the base layers still drive the requirement: `bmad-review`, not the static
    # two-hunter catalog the old code fell back to
    problems = [f for f in findings if f.severity == "problem"]
    assert [p.detail["skill"] for p in problems] == ["bmad-review"]
    assert all(p.check == "skills.review-layer-missing" for p in problems)

    _install_skills(tmp_path, claude.skill_tree, {"bmad-review": ()})
    findings = missing_base_skills(tmp_path, [claude.skill_tree])
    # the broken file is still surfaced, but nothing blocks
    assert [f.severity for f in findings] == ["warning"]
    assert findings[0].detail["file"].endswith("bmad-dev-auto.toml")


def test_user_override_wins_over_team_override(tmp_path):
    """Precedence is base -> team -> user, so the personal layer decides."""
    claude = get_profile("claude")
    _install_dev_auto(
        tmp_path, claude.skill_tree, customize="[workflow]\n" + _layer("blind", "bmad-review")
    )
    _write_override(tmp_path, _layer("blind", "bmad-review-team"))
    _write_override(tmp_path, _layer("blind", "bmad-review-personal"), user=True)

    problems = missing_base_skills(tmp_path, [claude.skill_tree])
    assert [p.detail["skill"] for p in problems] == ["bmad-review-personal"]


def test_every_review_layer_disabled_is_reported(tmp_path):
    """Emptying every `instruction` disables every layer, and step-04 then HALTs
    blocked with 'no active review layers'. Preflight must not call that green."""
    claude = get_profile("claude")
    disabled = re.sub(
        r"instruction = '''.*?'''", 'instruction = ""', LAYER_CUSTOMIZE, flags=re.DOTALL
    )
    _install_dev_auto(tmp_path, claude.skill_tree, customize=disabled)
    _install_skills(tmp_path, claude.skill_tree, {"bmad-review": ()})

    problems = missing_base_skills(tmp_path, [claude.skill_tree])
    assert len(problems) == 1
    assert problems[0].check == "skills.review-layers-empty"
    assert problems[0].severity == "problem"
    assert "no active review layers" in problems[0].message


def test_code_keyed_layers_merge_by_code_like_upstream(tmp_path):
    """BMAD's resolver keys arrays of tables on `code` OR `id` — `code` first. A
    code-keyed layer that an override replaces must not survive as a requirement."""
    claude = get_profile("claude")
    _install_dev_auto(
        tmp_path,
        claude.skill_tree,
        customize="[workflow]\n" + _layer("R1", "bmad-review-old", key="code"),
    )
    _write_override(tmp_path, _layer("R1", "bmad-review-new", key="code"))

    problems = missing_base_skills(tmp_path, [claude.skill_tree])
    # replaced in place: only the override's skill is required
    assert [p.detail["skill"] for p in problems] == ["bmad-review-new"]


def test_keyless_override_item_forces_append_like_upstream(tmp_path):
    """The keyed merge is opt-in for the array as a WHOLE: one override item with no
    identifier and the resolver appends everything, leaving the base layer in place.
    Replacing by id anyway drops a reviewer the run still executes — a green
    validate followed by a review that fails on a skill nobody checked for."""
    claude = get_profile("claude")
    _install_dev_auto(
        tmp_path,
        claude.skill_tree,
        customize="[workflow]\n" + _layer("blind", "bmad-review-base"),
    )
    _write_override(
        tmp_path,
        _layer("blind", "bmad-review-replacement")
        + '\n[[workflow.review_layers]]\ninstruction = "Invoke the `bmad-review-extra` skill."\n',
    )

    problems = missing_base_skills(tmp_path, [claude.skill_tree])
    assert sorted(p.detail["skill"] for p in problems) == [
        "bmad-review-base",
        "bmad-review-extra",
        "bmad-review-replacement",
    ]
    # the id-less layer is still reported by position, so the finding is actionable
    extra = next(p for p in problems if p.detail["skill"] == "bmad-review-extra")
    assert extra.detail["layers"] == ["#3"]


@pytest.mark.parametrize("value", ['"wrong shape"', "[1, 2]", "42", "true"])
def test_non_table_workflow_does_not_crash_the_preflight(tmp_path, value):
    """Syntactically valid TOML of the wrong SHAPE used to raise AttributeError out
    of the preflight, taking validate/run/resume/sweep with it."""
    claude = get_profile("claude")
    _install_dev_auto(tmp_path, claude.skill_tree, customize=f"workflow = {value}\n")

    # no layers readable -> the static fallback, not an exception
    findings = missing_base_skills(tmp_path, [claude.skill_tree])
    assert {f.check for f in findings} == {"skills.base-missing"}


def test_when_gated_layer_warns_instead_of_blocking(tmp_path):
    """step-04 skips every layer whose `when` does not hold, and that condition is
    evaluated by the model in run context. Undecidable here, so it must not FAIL."""
    claude = get_profile("claude")
    _install_dev_auto(
        tmp_path,
        claude.skill_tree,
        customize="[workflow]\n"
        + _layer("blind", "bmad-review")
        + _layer("perf", "bmad-review-performance", when="the diff touches hot paths"),
    )
    _install_skills(tmp_path, claude.skill_tree, {"bmad-review": ()})

    findings = missing_base_skills(tmp_path, [claude.skill_tree])
    assert len(findings) == 1
    assert findings[0].check == "skills.review-layer-unresolved"
    assert findings[0].severity == "warning"
    assert findings[0].detail["skill"] == "bmad-review-performance"
    assert findings[0].detail["layers"] == ["perf"]


def test_unrecognized_handoff_phrasing_warns_instead_of_blocking(tmp_path):
    """The invocation phrasing is a convention, not a contract — upstream itself
    writes "use the `x` skill" elsewhere. An unconfirmable reference is surfaced,
    never blocked on: guessing wrong rebuilds #260."""
    claude = get_profile("claude")
    _install_dev_auto(
        tmp_path,
        claude.skill_tree,
        customize="[workflow]\n"
        + _layer("blind", "bmad-review")
        + _layer("house", "bmad-review-company", phrasing="Use the"),
    )
    _install_skills(tmp_path, claude.skill_tree, {"bmad-review": ()})

    findings = missing_base_skills(tmp_path, [claude.skill_tree])
    assert [(f.check, f.severity) for f in findings] == [
        ("skills.review-layer-unresolved", "warning")
    ]
    assert findings[0].detail["skill"] == "bmad-review-company"


def test_skill_required_by_one_layer_is_never_also_advisory(tmp_path):
    """A hard requirement wins: the same skill named by a gated layer and an
    ungated one is reported once, as a problem."""
    claude = get_profile("claude")
    _install_dev_auto(
        tmp_path,
        claude.skill_tree,
        customize="[workflow]\n"
        + _layer("blind", "bmad-review")
        + _layer("perf", "bmad-review", when="sometimes"),
    )

    findings = missing_base_skills(tmp_path, [claude.skill_tree])
    assert [(f.check, f.severity) for f in findings] == [("skills.review-layer-missing", "problem")]


def test_new_review_check_ids_are_registered():
    """A check id that isn't in the registry asserts at emit time — i.e. ships as a
    crash on exactly the misconfigured project it was meant to report."""
    from bmad_loop.checks import VALIDATE_CHECKS

    assert {
        "skills.review-layer-unresolved",
        "skills.review-layers-empty",
        "skills.customize-unreadable",
    } <= VALIDATE_CHECKS


def test_provision_worktree_copies_derived_review_skill(tmp_path):
    """Validating a custom reviewer and then not provisioning it is how preflight
    passes in the main checkout while the isolated review fails on a skill that was
    never there. The worktree gets what the layers actually name."""
    wt, repo = tmp_path / "wt", tmp_path / "repo"
    claude = get_profile("claude")
    _install_base_skills(repo, claude.skill_tree)
    _install_dev_auto(
        repo,
        claude.skill_tree,
        customize="[workflow]\n" + _layer("house-style", "bmad-review-company"),
    )
    _install_skills(repo, claude.skill_tree, {"bmad-review-company": ()})

    provision_worktree(wt, [claude], repo)

    assert (wt / claude.skill_tree / "bmad-review-company" / "SKILL.md").is_file()
    # the floor is still copied, so a tree whose config we cannot read is unchanged
    for skill in BASE_SKILLS:
        assert (wt / claude.skill_tree / skill / "SKILL.md").is_file()


def test_provision_worktree_seeds_bmad_custom(tmp_path):
    """The run inside the worktree resolves review layers from ITS OWN project
    root. `*.user.toml` is gitignored by the upstream installer (and plenty of
    projects gitignore `_bmad/` whole), so without seeding, validate approves a
    layer set the isolated run never resolves."""
    wt, repo = tmp_path / "wt", tmp_path / "repo"
    claude = get_profile("claude")
    _install_base_skills(repo, claude.skill_tree)
    _write_override(repo, _layer("house-style", "bmad-review-company"), user=True)

    provision_worktree(wt, [claude], repo)

    seeded = wt / "_bmad" / "custom" / "bmad-dev-auto.user.toml"
    assert seeded.is_file()
    assert "bmad-review-company" in seeded.read_text(encoding="utf-8")


def test_provision_worktree_bmad_custom_does_not_clobber_checkout(tmp_path):
    """A checkout that tracks its own customization keeps every tracked file — only
    the children it lacks (the gitignored personal layer) are filled in."""
    wt, repo = tmp_path / "wt", tmp_path / "repo"
    claude = get_profile("claude")
    _install_base_skills(repo, claude.skill_tree)
    _write_override(repo, _layer("team", "bmad-review-repo-side"))
    _write_override(repo, _layer("personal", "bmad-review-mine"), user=True)
    # the worktree checked out the TRACKED team layer, at a different revision
    tracked = wt / "_bmad" / "custom" / "bmad-dev-auto.toml"
    tracked.parent.mkdir(parents=True, exist_ok=True)
    tracked.write_text("# checked out\n", encoding="utf-8")

    provision_worktree(wt, [claude], repo)

    assert tracked.read_text(encoding="utf-8") == "# checked out\n"
    assert (wt / "_bmad" / "custom" / "bmad-dev-auto.user.toml").is_file()


def test_provision_worktree_bmad_custom_shielded_in_local_exclude(project, tmp_path):
    """Seeded customization must stay out of the unit's `git add -A` — a project
    that doesn't gitignore `_bmad/` would otherwise merge it back on every story.

    The pattern lands in the worktree's PRIVATE exclude, and the repo-wide one is
    byte-identical afterwards (#384)."""
    repo = project.project
    _write_override(repo, _layer("house-style", "bmad-review-company"), user=True)
    wt = tmp_path / "wt"
    verify.worktree_add(repo, wt, "feat", "main")
    shared = repo / ".git" / "info" / "exclude"
    before = shared.read_bytes()

    provision_worktree(wt, [get_profile("claude")], repo)

    assert (wt / "_bmad" / "custom" / "bmad-dev-auto.user.toml").is_file()
    exclude = _wt_private_exclude(wt).read_text(encoding="utf-8")
    assert "/_bmad/custom" in exclude.splitlines()
    assert shared.read_bytes() == before


def test_missing_stories_support_findings_split_absent_from_stale(tmp_path):
    """#205: a half install and a too-old install are different conditions with
    different remediations, so they get different check ids — a script pinning a
    version bump must be able to tell "reinstall" from "update"."""
    from bmad_loop.checks import VALIDATE_CHECKS
    from bmad_loop.install import (
        STORIES_PROBE_FILE,
        STORIES_PROBE_SKILL,
        STORIES_PROBE_TEXT,
        missing_stories_support,
    )

    claude = get_profile("claude")
    tree = claude.skill_tree
    step01 = tmp_path / tree / STORIES_PROBE_SKILL / STORIES_PROBE_FILE

    absent = missing_stories_support(tmp_path, [tree])
    assert [f.check for f in absent] == ["skills.stories-dispatch-missing"]
    assert absent[0].detail == {
        "tree": tree,
        "skill": STORIES_PROBE_SKILL,
        "file": STORIES_PROBE_FILE,
    }

    step01.parent.mkdir(parents=True, exist_ok=True)
    step01.write_text("old clarify-and-route, no dispatch protocol\n", encoding="utf-8")
    stale = missing_stories_support(tmp_path, [tree])
    assert [f.check for f in stale] == ["skills.stories-dispatch-stale"]
    assert stale[0].detail["marker"] == STORIES_PROBE_TEXT
    assert all(f.check in VALIDATE_CHECKS for f in (*absent, *stale))


def test_missing_stories_support_reports_non_utf8_probe_without_crashing(tmp_path):
    """C1: a binary/non-UTF-8 step-01 file must be reported as a problem, not crash
    the preflight — read_text(encoding="utf-8") raises UnicodeDecodeError (a
    ValueError, NOT an OSError), so the content probe has to catch it explicitly."""
    from bmad_loop.install import (
        STORIES_PROBE_FILE,
        STORIES_PROBE_SKILL,
        missing_stories_support,
    )

    claude = get_profile("claude")
    tree = claude.skill_tree
    step01 = tmp_path / tree / STORIES_PROBE_SKILL / STORIES_PROBE_FILE
    step01.parent.mkdir(parents=True, exist_ok=True)
    step01.write_bytes(b"\xff\xfe\x00\x01 not utf-8 \x80\x81")  # invalid UTF-8

    problems = missing_stories_support(tmp_path, [tree])
    assert len(problems) == 1 and "not found" in problems[0].message


def test_new_dev_auto_skill_is_additive_for_sprint_mode(tmp_path):
    """Scenario 6 additivity: installing the *new* bmad-dev-auto (folder+id
    dispatch present) satisfies both preflights — sprint mode's file-existence
    check (`missing_base_skills`, which never inspects the dispatch content) and
    stories mode's content probe (`missing_stories_support`). The new skill
    breaks neither pipeline."""
    from bmad_loop.install import (
        STORIES_PROBE_FILE,
        STORIES_PROBE_SKILL,
        missing_stories_support,
    )

    claude = get_profile("claude")
    tree = claude.skill_tree
    _install_base_skills(tmp_path, tree)
    # upgrade bmad-dev-auto in place to the folder+id dispatch version
    step01 = tmp_path / tree / STORIES_PROBE_SKILL / STORIES_PROBE_FILE
    step01.write_text("route a **folder+id dispatch** invocation\n", encoding="utf-8")

    # sprint mode (file existence) is unaffected by the new dispatch content …
    assert missing_base_skills(tmp_path, [tree]) == []
    # … and stories mode now also passes its stricter content probe
    assert missing_stories_support(tmp_path, [tree]) == []


def test_provision_worktree_seeds_gitignored_config(tmp_path):
    """A gitignored config present in the main repo is copied into the worktree
    (a `git worktree add` checkout would omit it)."""
    wt, repo = tmp_path / "wt", tmp_path / "repo"
    repo.mkdir()
    (repo / ".mcp.json").write_text('{"mcpServers": {}}', encoding="utf-8")
    provision_worktree(wt, [], repo, seed_files=[".mcp.json"])
    assert (wt / ".mcp.json").read_text() == '{"mcpServers": {}}'


def test_provision_worktree_seed_skips_missing_source(tmp_path):
    wt, repo = tmp_path / "wt", tmp_path / "repo"
    repo.mkdir()
    provision_worktree(wt, [], repo, seed_files=[".mcp.json"])
    assert not (wt / ".mcp.json").exists()


def test_provision_worktree_seed_does_not_clobber_existing(tmp_path):
    """A seed target already present in the worktree (tracked/committed) is left
    untouched, so no diff is merged back."""
    wt, repo = tmp_path / "wt", tmp_path / "repo"
    repo.mkdir()
    (repo / ".mcp.json").write_text("FROM_REPO", encoding="utf-8")
    dst = wt / ".mcp.json"
    dst.parent.mkdir(parents=True)
    dst.write_text("IN_WORKTREE", encoding="utf-8")
    provision_worktree(wt, [], repo, seed_files=[".mcp.json"])
    assert dst.read_text() == "IN_WORKTREE"


def test_provision_worktree_reports_seed_skipped_as_noop(tmp_path):
    """A seed entry left untouched because the destination exists is REPORTED, so a
    `worktree_seed` that copies nothing cannot look like applied configuration."""
    wt, repo = tmp_path / "wt", tmp_path / "repo"
    repo.mkdir()
    (repo / ".mcp.json").write_text("FROM_REPO", encoding="utf-8")
    dst = wt / ".mcp.json"
    dst.parent.mkdir(parents=True)
    dst.write_text("IN_WORKTREE", encoding="utf-8")
    assert provision_worktree(wt, [], repo, seed_files=[".mcp.json"]) == [".mcp.json"]


def test_provision_worktree_seeds_absent_children_of_existing_dir(tmp_path):
    """The case that motivated #230: a worktree checks out tracked files, so a seed
    DIRECTORY with any tracked child already exists. Its absent children are seeded
    anyway (they clobber nothing), and the entry is no longer reported as a no-op."""
    wt, repo = tmp_path / "wt", tmp_path / "repo"
    (repo / "_bmad" / "custom").mkdir(parents=True)  # tracked child
    (repo / "_bmad" / "bmm").mkdir()  # gitignored sibling, absent from the checkout
    (repo / "_bmad" / "bmm" / "config.yaml").write_text("SEED ME", encoding="utf-8")
    (wt / "_bmad" / "custom").mkdir(parents=True)  # what `git worktree add` lays down

    assert provision_worktree(wt, [], repo, seed_files=["_bmad"]) == []
    assert (wt / "_bmad" / "bmm" / "config.yaml").read_text() == "SEED ME"


def test_provision_worktree_seed_dir_does_not_clobber_existing_children(tmp_path):
    """Seeding into an existing dir stays no-clobber at FILE granularity: a child the
    checkout carries keeps its content while its absent siblings are copied in."""
    wt, repo = tmp_path / "wt", tmp_path / "repo"
    (repo / "cfg").mkdir(parents=True)
    (repo / "cfg" / "tracked.yaml").write_text("FROM_REPO", encoding="utf-8")
    (repo / "cfg" / "ignored.yaml").write_text("SEED ME", encoding="utf-8")
    (repo / "cfg" / "nested").mkdir()
    (repo / "cfg" / "nested" / "deep.yaml").write_text("SEED ME TOO", encoding="utf-8")
    (wt / "cfg").mkdir(parents=True)
    (wt / "cfg" / "tracked.yaml").write_text("IN_WORKTREE", encoding="utf-8")

    assert provision_worktree(wt, [], repo, seed_files=["cfg"]) == []
    assert (wt / "cfg" / "tracked.yaml").read_text() == "IN_WORKTREE"  # untouched
    assert (wt / "cfg" / "ignored.yaml").read_text() == "SEED ME"
    assert (wt / "cfg" / "nested" / "deep.yaml").read_text() == "SEED ME TOO"


def test_provision_worktree_reports_seed_dir_with_nothing_to_copy(tmp_path):
    """A directory entry whose children ALL already exist copied nothing, so it is
    still a silent no-op and still reported — only a partial seed stops being."""
    wt, repo = tmp_path / "wt", tmp_path / "repo"
    (repo / "cfg").mkdir(parents=True)
    (repo / "cfg" / "tracked.yaml").write_text("FROM_REPO", encoding="utf-8")
    (wt / "cfg").mkdir(parents=True)
    (wt / "cfg" / "tracked.yaml").write_text("IN_WORKTREE", encoding="utf-8")

    assert provision_worktree(wt, [], repo, seed_files=["cfg"]) == ["cfg"]
    assert (wt / "cfg" / "tracked.yaml").read_text() == "IN_WORKTREE"


def test_provision_worktree_seed_dir_over_existing_file_is_skipped(tmp_path):
    """A directory entry whose destination is a FILE is a type mismatch: recursing
    would mkdir over the file. The file wins (no-clobber) and the entry is reported
    skipped like any other whose destination already exists."""
    wt, repo = tmp_path / "wt", tmp_path / "repo"
    (repo / "cfg").mkdir(parents=True)
    (repo / "cfg" / "ignored.yaml").write_text("SEED ME", encoding="utf-8")
    wt.mkdir()
    (wt / "cfg").write_text("A FILE, NOT A DIR", encoding="utf-8")

    assert provision_worktree(wt, [], repo, seed_files=["cfg"]) == ["cfg"]
    assert (wt / "cfg").read_text() == "A FILE, NOT A DIR"  # untouched


def test_provision_worktree_seed_skips_nested_file_typed_as_dir(tmp_path):
    """The same type mismatch one level down: a child that is a dir in the repo but
    a FILE in the checkout is skipped whole (never mkdir'd over), while its absent
    siblings still seed — so the entry counts as applied."""
    wt, repo = tmp_path / "wt", tmp_path / "repo"
    (repo / "cfg" / "sub").mkdir(parents=True)
    (repo / "cfg" / "sub" / "deep.yaml").write_text("SEED ME", encoding="utf-8")
    (repo / "cfg" / "ignored.yaml").write_text("SEED ME TOO", encoding="utf-8")
    (wt / "cfg").mkdir(parents=True)
    (wt / "cfg" / "sub").write_text("A FILE, NOT A DIR", encoding="utf-8")

    assert provision_worktree(wt, [], repo, seed_files=["cfg"]) == []
    assert (wt / "cfg" / "sub").read_text() == "A FILE, NOT A DIR"  # untouched
    assert (wt / "cfg" / "ignored.yaml").read_text() == "SEED ME TOO"


def test_provision_worktree_seeds_absent_empty_child_dir(tmp_path):
    """Creating a missing EMPTY child directory is a write: the entry modified the
    worktree, so it is treated as applied rather than reported as a no-op."""
    wt, repo = tmp_path / "wt", tmp_path / "repo"
    (repo / "cfg" / "empty").mkdir(parents=True)
    (repo / "cfg" / "tracked.yaml").write_text("FROM_REPO", encoding="utf-8")
    (wt / "cfg").mkdir(parents=True)
    (wt / "cfg" / "tracked.yaml").write_text("IN_WORKTREE", encoding="utf-8")

    assert provision_worktree(wt, [], repo, seed_files=["cfg"]) == []
    assert (wt / "cfg" / "empty").is_dir()
    assert (wt / "cfg" / "tracked.yaml").read_text() == "IN_WORKTREE"  # untouched


def test_provision_worktree_reports_nothing_when_seeding_succeeds(tmp_path):
    """A seed that actually copies is not reported — the signal stays specific to
    entries that silently did nothing. A missing source is also not a no-op report:
    it is already covered as its own case."""
    wt, repo = tmp_path / "wt", tmp_path / "repo"
    repo.mkdir()
    (repo / ".mcp.json").write_text("FROM_REPO", encoding="utf-8")
    assert provision_worktree(wt, [], repo, seed_files=[".mcp.json", "absent.json"]) == []
    assert (wt / ".mcp.json").read_text() == "FROM_REPO"


def test_provision_worktree_seed_rejects_escaping_path(tmp_path):
    """A seed entry resolving outside the repo/worktree is skipped — never copies
    a file from outside the project tree into the worktree."""
    wt, repo = tmp_path / "wt", tmp_path / "repo"
    repo.mkdir()
    (tmp_path / "outside.txt").write_text("SECRET", encoding="utf-8")
    provision_worktree(wt, [], repo, seed_files=["../outside.txt"])
    assert not wt.exists()  # nothing copied, no dirs created


def test_provision_worktree_seed_then_hook_merge_preserves_settings(tmp_path):
    """A seeded settings file that is also the hook config_path keeps its real
    content (seeded first), then gets the Stop hook merged in — not recreated empty."""
    wt, repo = tmp_path / "wt", tmp_path / "repo"
    repo.mkdir()
    claude = get_profile("claude")
    cfg = repo / claude.hooks.config_path  # .claude/settings.json
    cfg.parent.mkdir(parents=True)
    cfg.write_text(json.dumps({"permissions": {"allow": ["Bash(ls)"]}}), encoding="utf-8")

    provision_worktree(wt, [claude], repo, seed_files=[claude.hooks.config_path])

    seeded = json.loads((wt / claude.hooks.config_path).read_text())
    assert seeded["permissions"] == {"allow": ["Bash(ls)"]}  # real content survived
    assert "Stop" in seeded["hooks"]  # signal hook merged in on top


def test_provision_worktree_seed_shielded_in_local_exclude(project, tmp_path):
    """Seeded configs are added to the worktree's private git exclude so a project
    that doesn't gitignore them won't have the unit's `git add -A` stage them —
    without a line reaching the repo-wide exclude the main checkout shares (#384)."""
    repo = project.project
    (repo / ".mcp.json").write_text("{}", encoding="utf-8")
    wt = tmp_path / "wt"
    verify.worktree_add(repo, wt, "feat", "main")
    shared = repo / ".git" / "info" / "exclude"
    before = shared.read_bytes()

    provision_worktree(wt, [get_profile("claude")], repo, seed_files=[".mcp.json"])

    assert (wt / ".mcp.json").is_file()
    exclude = _wt_private_exclude(wt).read_text(encoding="utf-8")
    assert "/.mcp.json" in exclude.splitlines()
    assert shared.read_bytes() == before


def test_provision_worktree_partial_seed_dir_shielded_in_local_exclude(project, tmp_path):
    """A directory entry that only PARTIALLY seeds (its destination already existed)
    still gets its exclude pattern written — otherwise the children just seeded would
    be staged by the unit's `git add -A`."""
    repo = project.project
    (repo / "cfg").mkdir(exist_ok=True)
    (repo / "cfg" / "tracked.yaml").write_text("FROM_REPO", encoding="utf-8")
    (repo / "cfg" / "ignored.yaml").write_text("SEED ME", encoding="utf-8")
    wt = tmp_path / "wt"
    verify.worktree_add(repo, wt, "feat", "main")
    # what the checkout lays down: the tracked child, so the seed dir already exists
    (wt / "cfg").mkdir(parents=True)
    (wt / "cfg" / "tracked.yaml").write_text("FROM_REPO", encoding="utf-8")

    shared = repo / ".git" / "info" / "exclude"
    before = shared.read_bytes()

    provision_worktree(wt, [get_profile("claude")], repo, seed_files=["cfg"])

    assert (wt / "cfg" / "ignored.yaml").read_text() == "SEED ME"
    exclude = _wt_private_exclude(wt).read_text(encoding="utf-8")
    assert "/cfg" in exclude.splitlines()
    assert shared.read_bytes() == before


# ------------------------------------------- the shield is scoped to the worktree (issue #384)
#
# Real linked worktrees throughout (`verify.worktree_add` on the `project` fixture),
# never a stand-in: every property under test is git's, not the filesystem's — which
# exclude file git reads, which config scope wins, what `git worktree remove` deletes.
# A mocked git would assert the test author's model of those instead of git's answer.


def test_shield_never_touches_main_checkout(project, tmp_path):
    """THE #384 regression: after a worktree is provisioned, a NEW file under a
    TRACKED tool dir must still be staged by `git add -A` in the main checkout.

    The reported harm exactly. `.claude/skills` is a directory projects legitimately
    track, the shield names it, and the repo-wide `.git/info/exclude` the helper
    used to append to is shared with the operator's own checkout, permanent, and
    unversioned. Their next `git add -A` then captured only diffs to files git
    already tracked, while every newly created sibling silently vanished — 51 files
    and three whole skills across two upgrade commits in the reporter's repo, with
    nothing in either diff able to reveal why.

    Both assertions are needed. Git's own answer (the file stages) is the harm; the
    shared file's BYTES are the mechanism, and pin that nothing was appended even in
    a form that happens not to match this fixture's paths.

    Ablation: target `common_dir / "info" / "exclude"` again and both fail."""
    repo = project.project
    tracked = repo / ".claude" / "skills" / "committed-skill"
    tracked.mkdir(parents=True)
    (tracked / "SKILL.md").write_text("# tracked\n", encoding="utf-8")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "track the skill tree")
    shared = repo / ".git" / "info" / "exclude"
    before = shared.read_bytes()
    wt = tmp_path / "wt"
    verify.worktree_add(repo, wt, "feat", "main")

    provision_worktree(wt, [get_profile("claude")], repo)

    # what the operator does next, in their own checkout: add a skill
    fresh = repo / ".claude" / "skills" / "written-after-the-run"
    fresh.mkdir(parents=True)
    (fresh / "SKILL.md").write_text("# new\n", encoding="utf-8")
    git(repo, "add", "-A")
    staged = git(repo, "diff", "--cached", "--name-only").splitlines()
    assert ".claude/skills/written-after-the-run/SKILL.md" in staged
    assert shared.read_bytes() == before


def test_shield_excludes_only_inside_the_worktree(project, tmp_path):
    """The shield still shields: the very path the main checkout must keep seeing
    is invisible to the WORKTREE's `git add -A`.

    Paired with the regression above deliberately. Issue #384 measured that a
    private exclude alone does NOTHING — git reads only `$GIT_COMMON_DIR/info/exclude`
    — so writing the file and skipping the `config --worktree core.excludesFile`
    activation would satisfy that test while quietly committing the tool files here.
    Neither test is meaningful without the other.

    Ablation: drop the activation call and this fails — the provisioned skills and
    settings.json show up staged."""
    repo = project.project
    wt = tmp_path / "wt"
    verify.worktree_add(repo, wt, "feat", "main")

    provision_worktree(wt, [get_profile("claude")], repo)

    assert (wt / ".claude" / "skills" / "bmad-loop-sweep" / "SKILL.md").is_file()
    assert (wt / ".claude" / "settings.json").is_file()
    git(wt, "add", "-A")
    assert git(wt, "diff", "--cached", "--name-only") == ""


def test_shield_dies_with_the_worktree(project, tmp_path):
    """Lifetime, the other half of #384: `git worktree remove` deletes the whole
    per-worktree gitdir, taking the private exclude AND the `config.worktree` that
    points at it. The shield expires exactly when the thing it shields does.

    That is why this fix needs no remover. The old design had none either, and that
    was the bug: `gc_run_worktrees` reclaimed the worktree and left the patterns in
    the shared exclude forever — surviving `isolation` going back to `"none"`, and
    the run, and the release.

    Ablation is git's own behavior, so the inverse is the check that matters: a
    shield written to the shared exclude survives this removal by construction, and
    this test would fail against it."""
    repo = project.project
    wt = tmp_path / "wt"
    verify.worktree_add(repo, wt, "feat", "main")

    provision_worktree(wt, [get_profile("claude")], repo)

    private = _wt_private_exclude(wt)
    assert private.is_file()  # it really was there to be lost
    gitdir = private.parent.parent

    verify.worktree_remove(repo, wt, force=True)

    assert not private.exists()
    assert not gitdir.exists()  # the whole .git/worktrees/<id>, config.worktree too


def test_shield_refuses_to_enable_extension_over_core_worktree(project, tmp_path):
    """`core.worktree` in the shared config is one of the two shapes git's own docs
    (git-worktree(1), CONFIGURATION FILE) say must be moved into the main worktree's
    `config.worktree` BEFORE `extensions.worktreeConfig` is enabled: enabling drops
    the exception that confines those keys to the main worktree, so they would start
    applying to every worktree.

    Rearranging an operator's repo layout is not something an installer may do
    behind their back, so the shield degrades and the run continues unshielded.
    Enabling anyway is the loud failure this refuses to risk.

    Ablation: delete the `core.worktree` branch in `_shield_enable_worktree_config`
    and this fails — `worktreeConfig` appears in the shared config."""
    repo = project.project
    wt = tmp_path / "wt"
    verify.worktree_add(repo, wt, "feat", "main")
    git(repo, "config", "core.worktree", str(repo))
    shared_exclude = repo / ".git" / "info" / "exclude"
    before = shared_exclude.read_bytes()

    reason = _worktree_local_exclude(wt, ["/probe-384"])

    assert reason is not None and "core.worktree" in reason
    # read as TEXT, not via `git config --get`: conftest's git() is check=True and
    # an unset key exits 1, which would error the test rather than assert it.
    assert "worktreeConfig" not in (repo / ".git" / "config").read_text(encoding="utf-8")
    assert not _wt_private_exclude(wt).exists()
    assert shared_exclude.read_bytes() == before


def test_shield_refuses_to_enable_extension_over_core_bare(project, tmp_path):
    """The second refused shape: `core.bare = true` in the shared config. Same
    reasoning as its `core.worktree` sibling, different key, and it needs its own
    case because only a TRUE value is disqualifying — `core.bare = false` is written
    into every ordinary repo by `git init` and must not block the shield.

    Set after the worktree is mounted, and nothing below asks git about the repo
    itself: a repo declaring itself bare answers most commands with a refusal,
    which is precisely why git wants the key moved.

    Ablation: delete the `core.bare` branch and this fails; narrow the value check
    to "is the key present" and every ordinary repo (`bare = false`) stops being
    shielded, which the sibling tests above catch."""
    repo = project.project
    wt = tmp_path / "wt"
    verify.worktree_add(repo, wt, "feat", "main")
    shared_exclude = repo / ".git" / "info" / "exclude"
    before = shared_exclude.read_bytes()
    git(repo, "config", "core.bare", "true")

    reason = _worktree_local_exclude(wt, ["/probe-384"])

    assert reason is not None and "core.bare" in reason
    assert "worktreeConfig" not in (repo / ".git" / "config").read_text(encoding="utf-8")
    assert shared_exclude.read_bytes() == before


@pytest.mark.parametrize(
    ("reported", "supported"),
    [
        ("git version 2.20.0\n", True),  # the boundary itself
        ("git version 2.19.4\n", False),  # one minor below it
        ("git version 2.9.5\n", False),  # numeric, not lexicographic: "9" > "20" as text
        ("git version 2.44.0.windows.1\n", True),
        ("git version 2.39.5 (Apple Git-154)\n", True),
        ("git version 3.0\n", True),
        ("", False),  # nothing at all: a spawn that produced no stdout
        ("fatal: not a git repository\n", False),
        # No `git version` prefix. Refused deliberately: a bare-number answer is
        # not this program's output, and the caller is about to make a permanent
        # repo-format change on the strength of it.
        ("2.55.0\n", False),
    ],
)
def test_git_version_at_least_reads_only_a_git_version_line(reported, supported):
    """The parse behind the shield's 2.20 gate. Unreadable answers must come back
    False, because the caller reads False as "do not touch this repository" — an
    optimistic parse is the only failure mode that costs anything."""
    assert _git_version_at_least(reported, (2, 20)) is supported


def test_shield_refuses_to_enable_extension_over_old_git(project, tmp_path, monkeypatch):
    """`extensions.worktreeConfig` and `git config --worktree` are both git 2.20.
    Below that the write buys a PERMANENT repo-format change that shields nothing —
    and git-worktree(1) says older git refuses a repository carrying the extension.
    So the version is checked before any of it.

    The gate also sits above the two probes underneath it on purpose: `--type=` is
    git 2.18, so on an older git the `core.bare` read exits non-zero and is read as
    "not bare" — the safety gate opening silently. Refusing here keeps that pair
    unreachable.

    Only the `version` call is faked; every other call runs against the real repo,
    so the assertion below reads the actual shared config rather than a stub's log.
    The enable is ALSO made to raise, because "the key is absent afterwards" would
    hold if the write merely failed.

    Ablation: delete the version gate in `_shield_enable_worktree_config` and this
    fails — the enable fires, the fake raises, and the key lands in the config."""
    repo = project.project
    wt = tmp_path / "wt"
    verify.worktree_add(repo, wt, "feat", "main")
    shared_before = (repo / ".git" / "info" / "exclude").read_bytes()
    real = install_mod.git_bytes

    def ancient(worktree, *args):
        if args == ("version",):
            return subprocess.CompletedProcess(
                args=["git", "version"], returncode=0, stdout=b"git version 2.19.4\n", stderr=b""
            )
        if args[:1] == ("config",) and "extensions.worktreeConfig" in args and "--get" not in args:
            raise AssertionError(f"made a permanent format change on git 2.19.4: {args}")
        return real(worktree, *args)

    monkeypatch.setattr(install_mod, "git_bytes", ancient)

    reason = _worktree_local_exclude(wt, ["/probe-384"])

    assert reason is not None and "2.19.4" in reason and "git 2.20" in reason
    # the repo's own format is untouched. Read as TEXT for the reason the sibling
    # refusal tests give: conftest's git() is check=True and an unset key exits 1.
    assert "worktreeConfig" not in (repo / ".git" / "config").read_text(encoding="utf-8")
    assert not _wt_private_exclude(wt).exists()
    assert (repo / ".git" / "info" / "exclude").read_bytes() == shared_before


def test_shield_degrade_leaves_no_permanent_repo_format_change(project, tmp_path, monkeypatch):
    """`extensions.worktreeConfig` is a PERMANENT repo-format change bmad-loop never
    reverses, so it must not be paid for a shield that then degrades away. Round 5:
    `_shield_enable_worktree_config` used to perform the enable itself, ahead of the
    seed, the mkdir, the write and the activation — so every degrade below it left
    the operator's repo marked forever for a shield that never applied. It now only
    PROBES; the write happens one line above the activation.

    Driven through the seed fault the round-5 GitError fix introduced, because that
    is the degrade path this reordering exists for — a fault the old code could not
    even reach, since it swallowed instead of degrading.

    Read as TEXT rather than via `git config --get`, for the reason the sibling
    refusal tests give: conftest's git() is check=True and an unset key exits 1.

    Ablation: move the enable back above the seed (into
    `_shield_enable_worktree_config`, where it was) and this fails —
    `worktreeConfig` is in the shared config after a degrade that shielded nothing."""
    repo = project.project
    wt = tmp_path / "wt"
    verify.worktree_add(repo, wt, "feat", "main")
    users = tmp_path / "my-global-ignores"
    users.write_text("mine.log\n", encoding="utf-8")
    git(repo, "config", "core.excludesFile", str(users))
    assert "worktreeConfig" not in (repo / ".git" / "config").read_text(encoding="utf-8")
    real = install_mod.git_bytes

    def unanswerable_excludes_read(worktree, *args):
        if "core.excludesFile" in args and "--get" in args:
            raise verify.GitError("git config did not answer")
        return real(worktree, *args)

    monkeypatch.setattr(install_mod, "git_bytes", unanswerable_excludes_read)

    reason = _worktree_local_exclude(wt, ["/probe-384"])

    assert reason is not None  # it really did degrade
    assert "worktreeConfig" not in (repo / ".git" / "config").read_text(encoding="utf-8")


def test_shield_enables_the_extension_on_the_path_that_activates(project, tmp_path):
    """The other half of the deferral: moving the enable down must not lose it. A
    happy path still leaves the repo carrying the extension — that write is what
    `git config --worktree` needs to exist at all, so without it the activation
    fatals and the shield does nothing.

    The sibling above proves the flag is absent after a degrade; this proves the
    reordering did not simply stop setting it. Neither alone distinguishes the fix
    from a deletion."""
    repo = project.project
    wt = tmp_path / "wt"
    verify.worktree_add(repo, wt, "feat", "main")
    assert "worktreeConfig" not in (repo / ".git" / "config").read_text(encoding="utf-8")

    assert _worktree_local_exclude(wt, ["/probe-384"]) is None

    assert "worktreeConfig" in (repo / ".git" / "config").read_text(encoding="utf-8")
    # ...and it bought a shield that actually holds
    (wt / "probe-384").write_text("noise\n", encoding="utf-8")
    assert git(wt, "status", "--short") == ""


def test_shield_rolls_back_the_extension_when_activation_fails(project, tmp_path, monkeypatch):
    """The one degrade left that can still have made a permanent repo-format change:
    the ACTIVATION's own failure, which happens one line after the enable. Round 5
    moved the enable down to close every path above it; Codex then pointed out this
    one, and it is real — read-only `.git` and a lock on `config.worktree` both reach
    it. So the arm rolls the flag back.

    Measured, not assumed: `--unset` removes the key *and* the now-empty
    `[extensions]` section, and `git config --worktree` refuses again afterwards, so
    the repo is genuinely returned to the state we found rather than cosmetically
    tidied.

    Only `--worktree` writes are failed, so the enable itself succeeds — which is
    what makes this the enable-then-fail ordering rather than a short-circuit.

    Ablation: drop the `--unset` rollback and this fails — `worktreeConfig` is left
    in the shared config after a degrade that shielded nothing."""
    repo = project.project
    wt = tmp_path / "wt"
    verify.worktree_add(repo, wt, "feat", "main")
    assert "worktreeConfig" not in (repo / ".git" / "config").read_text(encoding="utf-8")
    real = install_mod.git_bytes

    def fail_worktree_writes(worktree, *args):
        if "--worktree" in args:
            return subprocess.CompletedProcess(
                args=["git", *args], returncode=1, stdout=b"", stderr=b"fatal: read-only .git\n"
            )
        return real(worktree, *args)

    monkeypatch.setattr(install_mod, "git_bytes", fail_worktree_writes)

    reason = _worktree_local_exclude(wt, ["/probe-384"])

    assert reason is not None and "read-only .git" in reason
    assert "worktreeConfig" not in (repo / ".git" / "config").read_text(encoding="utf-8")
    # the rollback is silent when it works: the operator is told the shield failed,
    # not handed a second fault that did not happen
    assert "could NOT be rolled back" not in reason


def test_shield_reports_a_rollback_it_could_not_make(project, tmp_path, monkeypatch):
    """If the rollback ALSO fails the operator has to be told, because the repo then
    really does keep a permanent format change that shields nothing — the coherent
    case, since a read-only `.git` fails both writes. Reported rather than raised:
    this function is contracted never to propagate.

    Both faults are named in the one reason, so the second is not lost behind the
    first.

    Ablation: drop the `undone.returncode != 0` branch and this fails — the reason
    mentions only the activation, and the format change goes unreported."""
    repo = project.project
    wt = tmp_path / "wt"
    verify.worktree_add(repo, wt, "feat", "main")
    real = install_mod.git_bytes

    def fail_worktree_writes_and_the_unset(worktree, *args):
        if "--worktree" in args:
            return subprocess.CompletedProcess(
                args=["git", *args], returncode=1, stdout=b"", stderr=b"fatal: read-only .git\n"
            )
        if "--unset" in args:
            return subprocess.CompletedProcess(
                args=["git", *args], returncode=1, stdout=b"", stderr=b"fatal: could not lock\n"
            )
        return real(worktree, *args)

    monkeypatch.setattr(install_mod, "git_bytes", fail_worktree_writes_and_the_unset)

    reason = _worktree_local_exclude(wt, ["/probe-384"])

    assert reason is not None
    assert "read-only .git" in reason  # the activation fault
    assert "could NOT be rolled back" in reason and "could not lock" in reason
    assert "permanent format change" in reason


def test_shield_reports_a_rollback_whose_own_git_faulted(project, tmp_path, monkeypatch):
    """The rollback's OWN git can raise, not just return non-zero — it is another
    chokepoint call. `_shield_undo_extension` catches that and reports it, because it
    is called from a function contracted never to propagate, and because a raise
    escaping it would lose the activation fault that prompted the rollback.

    The coherent case rather than a contrived one: a dead git or a read-only `.git`
    fails the unset for the same reason it failed the activation.

    Ablation: drop the `except GitError` in `_shield_undo_extension` and this fails —
    the fault escapes into the caller's tail and the reason names neither the
    activation fault nor the retained format change."""
    repo = project.project
    wt = tmp_path / "wt"
    verify.worktree_add(repo, wt, "feat", "main")
    real = install_mod.git_bytes

    def fail_activation_and_raise_on_unset(worktree, *args):
        if "--unset" in args:
            raise verify.GitSpawnError("git could not spawn")
        if "--worktree" in args:
            return subprocess.CompletedProcess(
                args=["git", *args], returncode=1, stdout=b"", stderr=b"fatal: read-only .git\n"
            )
        return real(worktree, *args)

    monkeypatch.setattr(install_mod, "git_bytes", fail_activation_and_raise_on_unset)

    reason = _worktree_local_exclude(wt, ["/probe-384"])

    assert reason is not None
    assert "read-only .git" in reason  # the activation fault survived the rollback
    assert "could NOT be rolled back" in reason and "could not spawn" in reason
    assert "permanent format change" in reason


def test_shield_never_unsets_an_extension_the_repo_already_carried(project, tmp_path, monkeypatch):
    """The rollback is gated on having enabled the flag IN THIS CALL, and that gate is
    a safety property rather than a micro-optimization: a repo that already carries
    `extensions.worktreeConfig` may have `config.worktree` files other worktrees
    depend on, and unsetting it would stop git reading them. So an activation failure
    against an already-enabled repo must leave the flag alone.

    The `--unset` is injected as a hard failure rather than asserted on the config
    text, for the reason the already-enabled sibling test gives: the flag being
    present afterwards would also hold if the unset had run and failed. Forbidding
    the CALL is the only form that bites.

    Ablation: drop the `if needs_enable:` gate on the rollback and this fails — the
    fake raises."""
    repo = project.project
    wt = tmp_path / "wt"
    verify.worktree_add(repo, wt, "feat", "main")
    git(repo, "config", "extensions.worktreeConfig", "true")
    real = install_mod.git_bytes

    def fail_worktree_writes_forbid_unset(worktree, *args):
        if "--unset" in args and "extensions.worktreeConfig" in args:
            raise AssertionError(f"unset an extension this call did not enable: {args}")
        if "--worktree" in args:
            return subprocess.CompletedProcess(
                args=["git", *args], returncode=1, stdout=b"", stderr=b"fatal: read-only .git\n"
            )
        return real(worktree, *args)

    monkeypatch.setattr(install_mod, "git_bytes", fail_worktree_writes_forbid_unset)

    reason = _worktree_local_exclude(wt, ["/probe-384"])

    assert reason is not None and "read-only .git" in reason
    assert "worktreeConfig" in (repo / ".git" / "config").read_text(encoding="utf-8")


def test_shield_reuses_already_enabled_extension(project, tmp_path, monkeypatch):
    """A repo that already carries the extension is used as found — the second
    isolated run in a repo must not re-assert a permanent format change, and must
    not re-run the refusal gates against a repo whose format it did not change.

    Injected as a hard failure rather than an equality assertion on the shared
    config: `git config` rewriting `true` over `true` leaves the file byte-identical,
    so a bytes comparison would pass with the early return deleted. Forbidding the
    WRITE is the only form of this that bites.

    Ablation: drop the already-true early return in `_shield_enable_worktree_config`
    and this fails — the enable call fires and the fake raises."""
    repo = project.project
    wt = tmp_path / "wt"
    verify.worktree_add(repo, wt, "feat", "main")
    git(repo, "config", "extensions.worktreeConfig", "true")
    real = install_mod.git_bytes

    def no_reenable(worktree, *args):
        # reads of the key must still pass through, or nothing works at all
        if args[:1] == ("config",) and "extensions.worktreeConfig" in args and "--get" not in args:
            raise AssertionError(f"re-enabled an extension already on: {args}")
        return real(worktree, *args)

    monkeypatch.setattr(install_mod, "git_bytes", no_reenable)

    assert _worktree_local_exclude(wt, ["/probe-384"]) is None

    assert "/probe-384" in _wt_private_exclude(wt).read_text(encoding="utf-8").splitlines()


def test_shield_enables_extension_when_only_the_global_config_claims_it(
    project, tmp_path, monkeypatch
):
    """`extensions.*` is repo-format state git honors ONLY from the repository's own
    config, so the "is it already enabled" question has exactly one legitimate place
    to be asked. Asked unscoped, a `worktreeConfig = true` in the operator's
    ~/.gitconfig answers it — the shield concludes the repo is ready, skips the
    write, and the activation then dies with git's own `--worktree cannot be used
    with multiple working trees` — a repo one write away from being shielded,
    skipped over instead.

    The global value is planted through GIT_CONFIG_GLOBAL, pinned around the shield
    call alone for the reason the XDG test gives at length: an env pin that outlives
    the code under test changes what git thinks of files already checked out.

    Ablation: drop `--file <shared>` from the extension probe and this fails — the
    global value is believed, the extension never reaches the repo config, and the
    activation returns a reason instead of None."""
    repo = project.project
    wt = tmp_path / "wt"
    verify.worktree_add(repo, wt, "feat", "main")
    fake_global = tmp_path / "gitconfig-with-the-extension"
    fake_global.write_text("[extensions]\n\tworktreeConfig = true\n", encoding="utf-8")

    with monkeypatch.context() as pinned:
        pinned.setenv("GIT_CONFIG_GLOBAL", str(fake_global))
        reason = _worktree_local_exclude(wt, ["/probe-384"])

    assert reason is None
    # the repo's OWN format was changed, i.e. the global claim was not mistaken
    # for one about this repository
    assert "worktreeConfig" in (repo / ".git" / "config").read_text(encoding="utf-8")
    # ...and the shield it unlocks actually holds, which is what the unscoped probe
    # cost: without the write above, `config --worktree` cannot store the pointer.
    assert "/probe-384" in _wt_private_exclude(wt).read_text(encoding="utf-8").splitlines()
    (wt / "probe-384").write_text("noise\n", encoding="utf-8")
    assert git(wt, "status", "--short") == ""


def test_shield_seeds_users_excludesfile(project, tmp_path):
    """A worktree-scoped `core.excludesFile` SHADOWS the operator's own — git reads
    the key from the most specific scope that sets it and does not concatenate
    across scopes (verified) — so their patterns are copied into the private file
    when it is created. Without the copy, activating the shield silently un-ignores,
    inside the worktree, everything they ignore globally, and the unit's
    `git add -A` commits it: a shield that creates the leak it exists to plug.

    Asserted through git rather than the file's content alone: the content could be
    right while the activation pointed somewhere else.

    Ablation: make `_shield_inherited_excludes` return "" and this fails —
    `mine.log` comes back untracked in the worktree."""
    repo = project.project
    wt = tmp_path / "wt"
    verify.worktree_add(repo, wt, "feat", "main")
    users = tmp_path / "my-global-ignores"
    users.write_text("mine.log\n", encoding="utf-8")
    git(repo, "config", "core.excludesFile", str(users))

    assert _worktree_local_exclude(wt, ["/probe-384"]) is None

    lines = _wt_private_exclude(wt).read_text(encoding="utf-8").splitlines()
    assert "mine.log" in lines and "/probe-384" in lines
    (wt / "mine.log").write_text("noise\n", encoding="utf-8")
    assert git(wt, "status", "--short") == ""


@pytest.mark.skipif(
    sys.platform == "win32", reason="POSIX-only: Windows strips a trailing space from a filename"
)
def test_shield_seeds_an_excludesfile_whose_path_has_edge_whitespace(project, tmp_path):
    """Round 5 (#384, Codex P2): the same leak as the encoding bug, through the PATH
    rather than the content. A `.strip()` on git's answer destroyed a legal POSIX
    path — `git config --type=path --get` returns leading and trailing whitespace
    VERBATIM (git writes such a value quoted, so it round-trips; measured at 2.20.4
    and 2.55.0) — so the stripped path read absent via `is_file()`, the seed came
    back empty, and the activation then SHADOWED the excludes it had failed to copy.
    Silent: `_shield_inherited_excludes` returned empty rather than raising, and the
    caller only journals a non-None reason.

    `-z` is the fix: it terminates the value with NUL, so the path survives whatever
    whitespace it carries.

    One filename covers BOTH sides of the strip — `" my-global-ignores "` has a
    leading and a trailing space — so a half-fix (`rstrip` only, say) still fails.

    Both halves asserted, as in the non-UTF-8 sibling below: the private file's
    content (the mechanism) and git's own answer inside the worktree (the harm).

    Ablation: restore `.strip()` in place of the `-z` split in
    `_shield_inherited_excludes` and this fails — `mine.log` comes back untracked."""
    repo = project.project
    wt = tmp_path / "wt"
    verify.worktree_add(repo, wt, "feat", "main")
    users = tmp_path / " my-global-ignores "
    users.write_text("mine.log\n", encoding="utf-8")
    git(repo, "config", "core.excludesFile", str(users))
    # git really did keep the whitespace, so the seed below has something to lose.
    # Read from the config FILE, not through `git config --get`: conftest's git()
    # strips its stdout, which would defeat the very check being made here.
    assert f'"{users}"' in (repo / ".git" / "config").read_text(encoding="utf-8")

    assert _worktree_local_exclude(wt, ["/probe-384"]) is None

    lines = _wt_private_exclude(wt).read_text(encoding="utf-8").splitlines()
    assert "mine.log" in lines and "/probe-384" in lines
    (wt / "mine.log").write_text("noise\n", encoding="utf-8")
    assert git(wt, "status", "--short") == ""


@pytest.mark.skipif(
    sys.platform == "win32", reason="POSIX-only: a 0xff byte cannot be a Windows filename"
)
def test_shield_preserves_a_non_utf8_excludesfile(project, tmp_path):
    """THE round-4 regression (#384, Codex P1): the seed is a VERBATIM BYTE COPY, so
    an operator's excludes file that is not UTF-8 survives it intact.

    A `read_text(encoding="utf-8")` inside a handler naming `UnicodeError` made one
    non-UTF-8 byte anywhere in their file collapse the WHOLE seed to empty — and the
    activation that follows then SHADOWED the excludes it had just failed to copy,
    because git takes `core.excludesFile` from the most specific scope that sets it
    and never concatenates across scopes. So `git add -A` committed a file the
    operator had told git to ignore: the exact leak the seed exists to prevent,
    caused by the seed. An exclude file holds path patterns and POSIX paths are
    arbitrary bytes, so a legacy 8-bit encoding here is ordinary, not exotic.

    Both halves are asserted because either alone would pass against a different
    bug: the private file's BYTES (the mechanism — a lossy copy is still a copy) and
    git's own answer inside the worktree (the harm).

    Ablation: restore `source.read_text(encoding="utf-8")` in
    `_shield_inherited_excludes` and this fails — the seed comes back empty and
    `secret-\\377` shows up untracked in the worktree, ready for `git add -A`."""
    repo = project.project
    wt = tmp_path / "wt"
    verify.worktree_add(repo, wt, "feat", "main")
    users = tmp_path / "my-global-ignores"
    users.write_bytes(b"secret-\xff\nplain.log\n")
    git(repo, "config", "core.excludesFile", str(users))

    assert _worktree_local_exclude(wt, ["/probe-384"]) is None

    assert _wt_private_exclude(wt).read_bytes() == b"secret-\xff\nplain.log\n/probe-384\n"
    (wt / os.fsdecode(b"secret-\xff")).write_text("noise\n", encoding="utf-8")
    (wt / "plain.log").write_text("noise\n", encoding="utf-8")
    assert git(wt, "status", "--short") == ""


def test_shield_degrades_when_the_users_excludesfile_cannot_be_read(project, tmp_path, monkeypatch):
    """An excludes file that EXISTS but cannot be read must skip the shield, not
    activate over patterns it never copied. Absent is silent (the common case, and
    there is nothing to shadow); unreadable is a degrade, because the shadow is real.

    The other half of the round-4 fix, and untested in either direction before it:
    `OSError` was swallowed beside the decode fault, so an EACCES/EIO on their file
    produced the identical silent empty seed — and then the shadow.

    Injected at `Path.read_bytes` rather than `chmod(0o000)`, for a reason the
    assertions depend on: git runs as this same user, so a file WE cannot read is
    one git cannot read either, and "their ignores still apply" — the property that
    makes the shadow a harm — would be untestable. With the fault injected the file
    stays readable to git, so the third assertion is git's own answer. (It also
    sidesteps a root-owned CI runner ignoring the mode bits, the reason the sibling
    write-fault test injects too.)

    The last assertion is the non-vacuity check: without it this passes just as well
    if the shield had silently excluded everything.

    Ablation: put `OSError` back in `_shield_inherited_excludes`'s except tuple and
    this fails — `reason` comes back None, the activation shadows their file, and
    `mine.log` shows up untracked."""
    repo = project.project
    wt = tmp_path / "wt"
    verify.worktree_add(repo, wt, "feat", "main")
    users = tmp_path / "my-global-ignores"
    users.write_text("mine.log\n", encoding="utf-8")
    git(repo, "config", "core.excludesFile", str(users))
    real_read_bytes = Path.read_bytes

    def unreadable(self):
        if self == users:
            raise PermissionError(13, "Permission denied", str(users))
        return real_read_bytes(self)

    monkeypatch.setattr(Path, "read_bytes", unreadable)

    reason = _worktree_local_exclude(wt, ["/probe-384"])

    monkeypatch.undo()  # the assertions below do their own file and git I/O
    assert reason is not None and "Permission denied" in reason
    assert not _wt_private_exclude(wt).exists()  # nothing to point a shadowing key at
    (wt / "mine.log").write_text("noise\n", encoding="utf-8")
    (wt / "probe-384").write_text("noise\n", encoding="utf-8")
    status = git(wt, "status", "--short")
    assert "mine.log" not in status  # their ignore was never shadowed
    assert "probe-384" in status  # ...and the shield really was skipped


def test_shield_reseeds_after_an_interrupted_creation(project, tmp_path, monkeypatch):
    """A creation that died between `touch()` and the write must not leave a
    placeholder that the NEXT attempt reads as authoritative.

    `atomic_write_bytes` "leaves the original untouched and removes the temp" — and the
    original here is the zero-byte file `touch()` just made, so an ENOSPC in between
    leaves it behind. That attempt is harmless by itself: the degrade arm returns
    above the `config --worktree`, and an unactivated private exclude does nothing.
    The harm is deferred to the next one — an empty file taken as authoritative skips
    the inherited-excludes seed and THEN activates a shadowing `core.excludesFile`,
    reintroducing through the back door the exact leak
    `test_shield_seeds_users_excludesfile` exists to plug.

    Two attempts against the helper directly, because the engine cannot produce this
    sequence today: every re-mount of a unit path runs `discard_worktree`, whose
    `worktree_prune` deletes the whole per-worktree gitdir — placeholder included —
    and when that best-effort prune fails, `git worktree add` refuses the stale path
    outright and the unit defers before provisioning runs. What this pins is the
    helper's OWN contract, so a future caller cannot inherit the defect. Do not
    "simplify" the size check away on the grounds that nothing reaches it.

    The zero-byte assertion between the attempts is not decoration: without it this
    passes just as well if the first attempt left no file at all, i.e. whenever the
    placeholder this test is about never existed.

    Ablation: restore `existed = exclude.is_file()` and this fails — `mine.log` is
    missing from the private exclude and comes back untracked in the worktree."""
    repo = project.project
    wt = tmp_path / "wt"
    verify.worktree_add(repo, wt, "feat", "main")
    users = tmp_path / "my-global-ignores"
    users.write_text("mine.log\n", encoding="utf-8")
    git(repo, "config", "core.excludesFile", str(users))

    def boom(*a, **kw):
        raise OSError("no space left on device")

    # patched at install's OWN binding, as everywhere else in this file: the write
    # goes through atomic_write_bytes (#375, #384 round 4) on an mkstemp fd via
    # os.fdopen, so a Path.write_bytes patch never fires and would pass vacuously.
    # The NAME matters as much as the binding — left pointing at atomic_write_text
    # this patch became a silent no-op and the test failed on the degrade assertion.
    monkeypatch.setattr(install_mod, "atomic_write_bytes", boom)
    assert _worktree_local_exclude(wt, ["/probe-384"]) is not None
    monkeypatch.undo()  # the assertions below do their own file and git I/O
    assert _wt_private_exclude(wt).stat().st_size == 0  # the placeholder touch() left

    assert _worktree_local_exclude(wt, ["/probe-384"]) is None

    lines = _wt_private_exclude(wt).read_text(encoding="utf-8").splitlines()
    assert "mine.log" in lines and "/probe-384" in lines
    (wt / "mine.log").write_text("noise\n", encoding="utf-8")
    assert git(wt, "status", "--short") == ""


def test_shield_reprovision_does_not_duplicate_patterns(project, tmp_path):
    """Provisioning the SAME worktree twice must leave the private exclude
    byte-identical. The dedupe is what makes that true, and it compares the
    already-present lines against the patterns being added — so both sides have to
    be the same type. Left as `str` against a `bytes` set (the shape the round-4
    bytes conversion invites), every pattern reads as absent and is re-appended on
    every single re-provision, growing the file without bound.

    Nothing pinned this before: `test_shield_reseeds_after_an_interrupted_creation`
    does run the helper twice, but the second run sees a ZERO-BYTE file, i.e. the
    create path both times — it never exercises the dedupe at all.

    Through `provision_worktree` rather than the helper, because the real pattern
    set is what a re-provision re-offers, and `on_degraded` is collected so a
    surprise degrade cannot make the two runs match by both doing nothing.

    Ablation: drop the `os.fsencode` and compare `str` patterns against the bytes
    `present` set — this fails, with every tool pattern appearing twice."""
    repo = project.project
    wt = tmp_path / "wt"
    verify.worktree_add(repo, wt, "feat", "main")
    msgs: list[str] = []

    provision_worktree(wt, [get_profile("claude")], repo, on_degraded=msgs.append)
    first = _wt_private_exclude(wt).read_bytes()
    provision_worktree(wt, [get_profile("claude")], repo, on_degraded=msgs.append)

    assert msgs == []
    assert _wt_private_exclude(wt).read_bytes() == first
    lines = first.splitlines()
    assert lines and len(set(lines)) == len(lines)  # nor duplicated within one run


def test_shield_dedupe_does_not_split_on_non_git_line_breaks(project, tmp_path):
    """The dedupe splits the existing content into lines the way GIT does, which is
    the second thing the bytes conversion bought (#384 round 4).

    `str.splitlines()` breaks on `\\x0b`, `\\x0c`, `\\x1c`, `\\x1d`, `\\x1e` and
    `\\x85` as well as on newlines; git treats none of those as a line boundary, and
    every one of them is a legal byte in a POSIX filename. So a legitimate pattern
    containing one fragmented into two wrong dedupe keys, the identical pattern then
    read as absent, and it was appended a second time. `bytes.splitlines()` splits on
    `\\n`, `\\r` and `\\r\\n` only — git's own boundary set (it also strips a trailing
    `\\r`, which is why `\\r` costs nothing here).

    Asserted on the bytes rather than through `git status`: the subject is which
    dedupe KEY the pattern produces, and a `\\x0c` in a filename is POSIX-only
    while this fault is not.

    Ablation: restore `set(existing.splitlines())` over decoded text and this fails —
    the seeded pattern fragments into `weird`/`pattern`, the identical pattern reads
    as absent, and the file comes back carrying it twice."""
    repo = project.project
    wt = tmp_path / "wt"
    verify.worktree_add(repo, wt, "feat", "main")
    users = tmp_path / "my-global-ignores"
    users.write_bytes(b"weird\x0cpattern\n")
    git(repo, "config", "core.excludesFile", str(users))

    assert _worktree_local_exclude(wt, ["weird\x0cpattern", "/probe-384"]) is None

    assert _wt_private_exclude(wt).read_bytes() == b"weird\x0cpattern\n/probe-384\n"


def test_shield_seeds_a_relative_excludesfile_resolved_like_git(project, tmp_path, monkeypatch):
    """`--type=path` expands `~` and stops there — a RELATIVE `core.excludesFile`
    comes back from git verbatim. Git resolves such a value against the worktree's
    top level; `Path(value)` resolves it against whatever directory the orchestrator
    happens to have been launched from.

    The miss is silent, which is what makes it worth a test: `is_file()` comes back
    false, the seed returns "", and the activation below then SHADOWS the operator's
    real excludes file — so their patterns stop applying inside the worktree and the
    unit's `git add -A` stages what they told git to ignore.

    `monkeypatch.chdir` is load-bearing, not hygiene. Run from the worktree, the
    unfixed code resolves the same relative path by accident and this test passes
    against the bug it exists to catch.

    Ablation: drop the `is_absolute()` branch and this fails — `mine.log` is missing
    from the private exclude and comes back untracked in the worktree."""
    repo = project.project
    wt = tmp_path / "wt"
    verify.worktree_add(repo, wt, "feat", "main")
    (wt / "ignores").mkdir()
    (wt / "ignores" / "global").write_text("mine.log\n", encoding="utf-8")
    git(repo, "config", "core.excludesFile", "ignores/global")
    # anywhere that is NOT the worktree, and where no `ignores/global` exists
    monkeypatch.chdir(tmp_path)

    assert _worktree_local_exclude(wt, ["/probe-384"]) is None

    lines = _wt_private_exclude(wt).read_text(encoding="utf-8").splitlines()
    assert "mine.log" in lines and "/probe-384" in lines
    (wt / "mine.log").write_text("noise\n", encoding="utf-8")
    # `ignores/` is the fixture's own scaffolding, not the subject — it is untracked
    # either way, so exclude it rather than let it mask the assertion.
    assert "mine.log" not in git(wt, "status", "--short")


def test_shield_seeds_xdg_default_when_unset(project, tmp_path, monkeypatch):
    """With `core.excludesFile` unset git falls back to `$XDG_CONFIG_HOME/git/ignore`
    (gitignore(5)) — a real file on plenty of developer boxes and shadowed just as
    hard. The probe has to reproduce that fallback because git cannot be asked for
    it: `config --get` answers "unset", not "here is the default I would use".

    The environment is pinned three ways, and each one is load-bearing:
    XDG_CONFIG_HOME so the file under test is this test's, GIT_CONFIG_GLOBAL and
    GIT_CONFIG_NOSYSTEM so a developer box whose own `~/.gitconfig` sets
    `core.excludesFile` reaches the fallback branch at all instead of passing
    through the branch above and never testing this one.

    That pinning is scoped to the probe rather than the whole test, because
    switching the system config off mid-test changes what git thinks of files
    already on disk. Git for Windows ships `core.autocrlf = true` in its SYSTEM
    config, so `worktree_add` above checks out CRLF; with `GIT_CONFIG_NOSYSTEM`
    still set, the status below re-reads those same files with autocrlf off and
    every tracked one reads as modified. The env is what the probe needs, not
    what the assertions need — so it ends with the probe.

    Ablation: return "" when the key is unset (drop the XDG branch) and this fails."""
    repo = project.project
    wt = tmp_path / "wt"
    verify.worktree_add(repo, wt, "feat", "main")
    xdg = tmp_path / "xdg"
    (xdg / "git").mkdir(parents=True)
    (xdg / "git" / "ignore").write_text("xdg-ignored.tmp\n", encoding="utf-8")

    with monkeypatch.context() as pinned:
        pinned.setenv("XDG_CONFIG_HOME", str(xdg))
        pinned.setenv("GIT_CONFIG_GLOBAL", str(tmp_path / "no-such-gitconfig"))
        pinned.setenv("GIT_CONFIG_NOSYSTEM", "1")
        assert _worktree_local_exclude(wt, ["/probe-384"]) is None

    assert "xdg-ignored.tmp" in _wt_private_exclude(wt).read_text(encoding="utf-8").splitlines()
    (wt / "xdg-ignored.tmp").write_text("noise\n", encoding="utf-8")
    # The exclude is activated through a worktree-scoped `core.excludesFile`, i.e.
    # repo-local config — so this holds without the env pinning, which is the point.
    assert git(wt, "status", "--short") == ""


def test_shield_config_fault_skips_shield_entirely(project, tmp_path, monkeypatch):
    """When the activation fails, the shield stops there. It must NEVER fall back to
    the repository-wide exclude: that fallback is #384 itself, trading one reported
    degrade for permanent, silent, unreviewable damage to the operator's checkout.

    The fault is name-filtered — only `config --worktree` calls fail — so everything
    before it runs for real and this pins the LAST step rather than a short-circuit
    somewhere earlier. Returned as a non-zero rc rather than raised, because that is
    the shape git actually produces and the shield reads rc, not exceptions.

    Ablation: add any shared-exclude fallback on this path and the last two
    assertions fail."""
    repo = project.project
    wt = tmp_path / "wt"
    verify.worktree_add(repo, wt, "feat", "main")
    shared = repo / ".git" / "info" / "exclude"
    before = shared.read_bytes()
    real = install_mod.git_bytes

    def fail_worktree_writes(worktree, *args):
        if "--worktree" in args:
            return subprocess.CompletedProcess(
                args=["git", *args], returncode=1, stdout=b"", stderr=b"fatal: read-only .git\n"
            )
        return real(worktree, *args)

    monkeypatch.setattr(install_mod, "git_bytes", fail_worktree_writes)

    reason = _worktree_local_exclude(wt, ["/probe-384"])

    assert reason is not None and "read-only .git" in reason
    assert shared.read_bytes() == before
    # and nothing else the MAIN checkout consults changed either: a file created
    # there afterwards is still visible to git
    (repo / "written-after-the-run.tmp").write_text("x\n", encoding="utf-8")
    assert "written-after-the-run.tmp" in git(repo, "status", "--short")


def test_shield_main_checkout_degrades(project):
    """Handed a MAIN checkout there is nothing to scope to — its gitdir IS the
    common dir — so the only exclude file on offer is the shared one. That is the
    #384 write, so the shield refuses and says why rather than falling back.

    Not a hypothetical caller: `provision_worktree` is handed plain directories and
    project roots by tests and by non-isolated paths, and the pre-fix helper wrote
    the repo-wide exclude for every one of them.

    Ablation: restore the common-dir target and this fails — the reason is None and
    `/probe-384` lands in the shared exclude."""
    repo = project.project
    shared = repo / ".git" / "info" / "exclude"
    before = shared.read_bytes()

    reason = _worktree_local_exclude(repo, ["/probe-384"])

    assert reason is not None and "not a linked worktree" in reason
    assert shared.read_bytes() == before


@pytest.mark.skipif(sys.platform == "win32", reason="newline is a legal POSIX path byte")
def test_shield_survives_a_newline_in_the_repository_path(tmp_path):
    """A repository directory carrying a NEWLINE must still get its shield.

    `git rev-parse` separates its answers with a newline, so asking ONE call for both
    `--absolute-git-dir` and `--git-common-dir` read a legal path byte as a record
    delimiter: measured on git 2.55, the two answers below split into FOUR entries and
    the helper returned a degrade instead. Degrading is not a safe default here —
    `provision_worktree` has already copied the tool files by the time the shield runs,
    so the unit's `git add -A` commits them into the story's merge.

    The degrade was also a FALSE one, which is why the assertion that matters is git's
    own answer rather than the private file's content: git honors every step of this at
    such a path. `config --worktree core.excludesFile` round-trips the newline (escaped
    as `\\n` in `config.worktree`) and the pattern then applies.

    A fresh repo rather than the `project` fixture, whose path has no newline. The
    worktree itself may sit at a plain path — the private gitdir inherits the newline
    through the repo, which is where `--absolute-git-dir` picks it up.

    Ablation: ask for both dirs in one `rev-parse` again and this fails on the first
    assertion, with "could not read this worktree's git dirs"."""
    repo = tmp_path / "re\npo"
    repo.mkdir()
    git(repo, "init", "-q", "-b", "main")
    git(repo, "config", "user.email", "test@test")
    git(repo, "config", "user.name", "test")
    (repo / "seed.txt").write_text("x\n", encoding="utf-8")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "seed")
    shared = repo / ".git" / "info" / "exclude"
    before = shared.read_bytes()
    wt = tmp_path / "wt"
    verify.worktree_add(repo, wt, "feat", "main")

    assert _worktree_local_exclude(wt, ["/probe-384"]) is None

    assert "/probe-384" in _wt_private_exclude(wt).read_text(encoding="utf-8").splitlines()
    (wt / "probe-384").write_text("noise\n", encoding="utf-8")
    assert git(wt, "status", "--short") == ""
    assert shared.read_bytes() == before


# ------------------------------------------------- local exclude, best-effort (issue #359)
#
# The helper's filesystem tail was unguarded while its docstring promised
# best-effort. Faults below the `git rev-parse` call are INJECTED via name-filtered
# monkeypatch (the test_engine.py:1553 pattern), not built for real: this box and
# the 3.13+ legs resolve a real symlink loop without raising, so a hand-built loop
# would false-green. `_worktree_local_exclude` returns None on success AND on the
# expected "git can't be queried" skip; a str is the degrade reason.


def test_worktree_local_exclude_degrades_on_symlink_loop_resolve(project, monkeypatch):
    """A symlink loop under `.resolve()` raises RuntimeError, not OSError, on the
    3.11/3.12 CI legs — so the tail's except tuple must name RuntimeError or the
    fault escapes a function documented as best-effort.

    Both dirs are resolved before they are compared (a symlinked repo path is
    absolute but not canonical, and an unresolved comparison would read a main
    checkout as a linked worktree), so the injected fault fires on the common dir
    every shape has. The `project` MAIN CHECKOUT is the subject only because it
    needs no worktree to mount.

    Not vacuous through the main-checkout refusal that also returns a reason: the
    assertion is on the LOOP's message, which only the propagating fault produces.

    Ablation: drop `RuntimeError` from the tail's except tuple and this fails —
    the RuntimeError propagates out of the call instead of becoming a reason."""
    real_resolve = Path.resolve

    def unresolvable(self, *a, **kw):
        if self.name == ".git":
            raise RuntimeError(f"Symlink loop from {str(self)!r}")
        return real_resolve(self, *a, **kw)

    monkeypatch.setattr(Path, "resolve", unresolvable)

    reason = _worktree_local_exclude(project.project, ["/probe-359"])

    assert reason is not None and "Symlink loop" in reason
    assert str(project.project) in reason  # names WHICH worktree lost its exclude
    exclude = project.project / ".git" / "info" / "exclude"
    assert not exclude.is_file() or "/probe-359" not in exclude.read_text(encoding="utf-8")


def test_worktree_local_exclude_degrades_on_write_fault(project, tmp_path, monkeypatch):
    """A write fault on the exclude file itself (e.g. a read-only .git) degrades to
    a reason instead of propagating — this pins the guard reaching the LAST
    statement of the tail, not just the early resolve/mkdir steps.

    Injected rather than chmod: a root-owned CI runner ignores a read-only bit.

    A real LINKED worktree is the subject here and in the rest of this block: since
    #384 the tail's write only happens for one (a main checkout is refused before
    it), so a main-checkout subject would degrade for the wrong reason and pass
    without ever reaching the statement under test.

    Ablation: drop the tail's `try` (or `OSError` from its tuple) and this fails —
    the OSError propagates out of `_worktree_local_exclude`."""
    wt = tmp_path / "wt"
    verify.worktree_add(project.project, wt, "feat", "main")

    def boom(*a, **kw):
        raise OSError("read-only .git")

    # patched at install's OWN binding: the exclude write goes through
    # atomic_write_bytes (#375, #384 round 4), which writes via os.fdopen on an
    # mkstemp fd, so a Path.write_bytes/Path.open patch never fires and would pass
    # vacuously — and so would a patch left on the atomic_write_TEXT this replaced.
    monkeypatch.setattr(install_mod, "atomic_write_bytes", boom)

    reason = _worktree_local_exclude(wt, ["/probe-359"])

    assert reason is not None and "read-only .git" in reason


def test_worktree_local_exclude_appends_to_a_non_utf8_private_exclude(project, tmp_path):
    """REPURPOSED, and the reversal is the fix (#384 review round 4). This case used
    to assert that a private exclude which is not UTF-8 DEGRADES the shield away,
    because `read_text` raised UnicodeDecodeError on it. The payload is bytes
    end-to-end now, so those bytes survive untouched *and* the shield still applies
    — strictly better than the old policy, which preserved the file by giving up on
    shielding the worktree at all.

    The legacy bytes are asserted byte-identical in place, as before: rewriting
    someone's legacy-encoded exclude would still be worse than skipping it.

    Ablation: restore `exclude.read_text(encoding="utf-8")` for the re-read and this
    fails — UnicodeDecodeError comes back as a degrade reason and `/probe-359` never
    reaches the file."""
    wt = tmp_path / "wt"
    verify.worktree_add(project.project, wt, "feat", "main")
    exclude = _wt_private_exclude(wt)
    exclude.parent.mkdir(parents=True, exist_ok=True)
    exclude.write_bytes(b"\xff\xfe legacy-encoded\n")

    assert _worktree_local_exclude(wt, ["/probe-359"]) is None

    assert exclude.read_bytes() == b"\xff\xfe legacy-encoded\n/probe-359\n"
    (wt / "probe-359").write_text("noise\n", encoding="utf-8")
    assert git(wt, "status", "--short") == ""


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="fsencode uses utf-8/surrogatepass on Windows, which encodes any surrogate",
)
def test_worktree_local_exclude_degrades_on_unencodable_pattern(project, tmp_path):
    """RE-POINTED (#384 review round 4) at the one shape that still raises. The
    encode fault used to be reachable from a real filename: `provision_worktree`
    builds a `seed_globs` pattern via `rel.as_posix()`, a non-UTF-8 name arrived
    surrogate-escaped, and `write_text` could not encode it back. The payload is
    bytes now and `os.fsencode` is surrogateescape's inverse, so THAT name
    round-trips to its exact original bytes and no longer degrades anything —
    `test_shield_preserves_a_non_utf8_excludesfile` covers the improvement.

    What remains is a pattern carrying a surrogate surrogateescape never produces:
    a hand-authored `"\\ud800"` in a config string (`worktree_seed`, a plugin's
    `seed_globs`) rather than a decoded filename. `os.fsencode` rejects it on POSIX,
    because surrogateescape only round-trips \\udc80-\\udcff. Windows is skipped
    rather than adapted: its filesystem codec is utf-8/surrogatepass, which encodes
    every surrogate, so there is no encode fault to pin there at all.

    `"\\udcff"` — what this test used to use — is deliberately NOT the subject any
    more: it is the case that now succeeds.

    Ablation: drop `UnicodeError` from the tail's except tuple and this fails —
    UnicodeEncodeError propagates out of a function contracted never to."""
    pattern = "/vendor/weird-\ud800-name"
    wt = tmp_path / "wt"
    verify.worktree_add(project.project, wt, "feat", "main")

    reason = _worktree_local_exclude(wt, [pattern])

    assert reason is not None and "surrogates not allowed" in reason


def test_worktree_local_exclude_short_write_leaves_exclude_intact(project, tmp_path, monkeypatch):
    """#375: a fault PARTWAY THROUGH the write must leave the operator's exclude
    byte-identical, not truncated. `write_bytes` opens "wb" (truncate-then-write)
    and this is a read-modify-REWRITE carrying their content in `prefix`, so a
    direct write left the file cut mid-content while the reason still said
    "could not update". The surviving tail is a valid git pattern and a prefix of
    the intended one — cut one character in and the last line is `/`, which
    excludes the whole worktree.

    Blast radius is smaller since #384 (the file is this worktree's own, not the
    main repo's shared exclude) but the fault is not: the content carried through
    `prefix` is the operator's global excludes, copied in when the private file was
    created, and `/` still swallows the whole worktree.

    The fault is injected at `Path.open`, NOT at `Path.write_bytes`: patching
    write_bytes means the file is never opened, so the truncation cannot happen
    and the test would pass against the very bug it exists to catch. Going
    through the real `open(mode="wb")` is the whole point — that is what
    truncates.

    Ablation: swap `atomic_write_bytes` back to a direct `exclude.write_bytes` and
    this fails — the file comes back truncated."""
    wt = tmp_path / "wt"
    verify.worktree_add(project.project, wt, "feat", "main")
    exclude = _wt_private_exclude(wt)
    exclude.parent.mkdir(parents=True, exist_ok=True)
    exclude.write_text("# OPERATOR'S OWN\n/secret-local\n", encoding="utf-8")
    before = exclude.read_bytes()

    real_open = io.open

    class ShortWriter:
        def __init__(self, f):
            self._f = f

        def write(self, s):
            self._f.write(s[:8])  # a partial write, then the device gives up
            raise OSError(28, "No space left on device")

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            self._f.close()
            return False

        def __getattr__(self, name):
            return getattr(self._f, name)

    def opener(file, *a, **kw):
        mode = a[0] if a else kw.get("mode", "r")
        handle = real_open(file, *a, **kw)
        # io.open is the ONE seam both shapes share: the fix reaches it through
        # os.fdopen on an mkstemp fd (an int), the ablation through Path.open (a
        # path). Patching either one alone injects into only one of them, so the
        # ablation would silently stop biting.
        if "w" in mode:
            return ShortWriter(handle)
        return handle

    monkeypatch.setattr(io, "open", opener)

    reason = _worktree_local_exclude(wt, ["/probe-359"])

    monkeypatch.undo()
    assert reason is not None and "No space left" in reason
    assert exclude.read_bytes() == before  # untouched, not half-rewritten
    # no scratch file left lying next to it. Globbed, not a fixed name: the
    # helper's temp is uniquely named (mkstemp), so asserting one literal
    # filename is absent would pass no matter what the code did.
    assert [p.name for p in exclude.parent.iterdir()] == ["exclude"]


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="POSIX-only fault: Windows paths are UTF-16, so git cannot emit an undecodable path",
)
def test_worktree_local_exclude_undecodable_git_output_degrades(tmp_path, monkeypatch):
    """#374: the git-query arm used `text=True`, decoding stdout strictly, so a
    repo path with bytes invalid in the locale encoding raised UnicodeDecodeError
    — a type in NEITHER arm's tuple, straight out of a function whose whole
    contract is that it never propagates.

    Driven through a REAL stub `git` on PATH rather than a monkeypatched
    `subprocess.run`. That distinction is the test: replacing subprocess.run
    hands the code bytes directly and never runs the stdlib's decoding at all, so
    such a test passes identically with `text=True` restored — it cannot see the
    bug. Verified: the run-patching version did NOT fail its own ablation.

    Ablation: restore `text=True` on the subprocess call and this fails — the
    UnicodeDecodeError propagates from the arm that promises a silent skip."""
    bin_dir = tmp_path / "stubbin"
    bin_dir.mkdir()
    stub = bin_dir / "git"
    # Emits paths carrying byte 0xff, exactly what a repo whose directory name is
    # not valid UTF-8 makes `git rev-parse` print. Rooted in tmp_path, NOT a
    # literal /tmp: the helper mkdir(parents=True)s the gitdir it is handed, so a
    # hardcoded path would create real directories in the shared system /tmp on
    # every run, outside pytest's cleanup.
    root = os.fsencode(str(tmp_path)) + b"/repo-\377-name/.git"
    gitdir = root + b"/worktrees/wt"  # distinct from the common dir: a LINKED worktree

    # SINGLE-QUOTED into the script. Unquoted, a temp root carrying whitespace
    # (TMPDIR, --basetemp, a username with a space) word-splits into a second
    # printf argument, and `printf '%s\n'` repeats its format per argument — so
    # the helper is handed a path with an embedded NEWLINE and mkdir(parents=True)s
    # it as a sibling of the basetemp pytest reaps. Measured with a spaced
    # --basetemp: it leaked such a directory and the test still passed.
    def sq(raw):
        return b"'" + raw.replace(b"'", b"'\\''") + b"'"

    # Dispatches on the subcommand, because the shield asks git several questions
    # now and a stub that answered them all identically would not get past the
    # first. `shift 2` drops the `-C <worktree>` every call carries. The version
    # clears the 2.20 gate, the extension reads as already enabled so the stub is
    # never asked to change repo state, and every other `--get` exits 1, git's own
    # "unset".
    #
    # `rev-parse` dispatches one level further, on the FLAG, because the shield asks
    # for the two dirs in two calls — a newline is a legal byte in a POSIX path, so it
    # cannot serve as a record separator between them. Answering both here regardless
    # of the flag is what real git does not do, and it would hand the helper one
    # two-line "path" for each dir, making them equal and reading as a main checkout.
    stub.write_bytes(
        b'#!/bin/sh\nshift 2\ncase "$1" in\n'
        b"version) printf 'git version 2.55.0\\n' ;;\n"
        b'rev-parse) case "$2" in\n'
        b"  --absolute-git-dir) printf '%s\\n' " + sq(gitdir) + b" ;;\n"
        b"  *) printf '%s\\n' " + sq(root) + b" ;;\n"
        b"  esac ;;\n"
        b'config) case "$*" in\n'
        b"  *--worktree*) exit 0 ;;\n"
        b"  *extensions.worktreeConfig*) printf 'true\\n' ;;\n"
        b"  *) exit 1 ;;\n"
        b"  esac ;;\n"
        b"*) exit 1 ;;\nesac\n"
    )
    stub.chmod(0o755)
    # PATH is REPLACED, not prepended: the stub has to be the only resolvable
    # `git`, or the box's real git answers first with a decodable path and this
    # passes without ever exercising the decode.
    monkeypatch.setenv("PATH", str(bin_dir))
    # the excludes-file seed falls back to the XDG default when the key is unset
    # (which the stub reports); pointed at an empty dir so the runner's own
    # ~/.config/git/ignore cannot leak into the assertion below.
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))

    # must not raise: the contract is best-effort in both arms
    reason = _worktree_local_exclude(tmp_path / "wt", ["/probe-374"])

    assert reason is None or "could not update" in reason
    # The tail actually RAN, against the decoded path. Not decoration: without
    # it this test is vacuous whenever the stub cannot exec — a noexec temp dir
    # (ordinary CI hardening), no /bin/sh, a filesystem that drops the exec bit.
    # subprocess.run then raises OSError, the git-query arm swallows it as the
    # expected skip it is, `reason is None` holds, and the ablation above
    # evaporates with it. Measured: at mode 0644 the stub never runs and every
    # assertion above still passes.
    #
    # This also pins containment better than globbing shared /tmp for a leak:
    # that glob is non-recursive and tmp_path sits three levels down, so it
    # could not see this test's own output, while it DID fail for anyone whose
    # box still carried litter from the hardcoded-path version of this test.
    landed = Path(os.fsdecode(gitdir)) / "info" / "exclude"
    assert landed.is_file()
    assert "/probe-374" in landed.read_text(encoding="utf-8")
    # and not in the common dir's exclude, which is where #384 put it
    assert not (Path(os.fsdecode(root)) / "info" / "exclude").exists()


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX mode bits; Windows has no umask")
def test_worktree_local_exclude_created_exclude_stays_readable(project, tmp_path):
    """An exclude this helper CREATES keeps the mode the `write_text` it replaced
    would have produced, not `mkstemp`'s private 0600. `atomic_write_bytes` carries
    a mode over only when the target already EXISTS (the shared contract on
    `atomic_write_text` is explicit),
    so a fresh one lands 0600 — and git treats an exclude it cannot read as one
    that is not there. Measured: git warns `unable to access`, exits 0, and
    `git add -A` stages the very files the exclude was written to shield, with no
    degrade reason at all, because the write SUCCEEDED. That is the harm the
    helper's own docstring calls out, arrived at through a successful write.

    Reachable where the gitdir is read under another UID —
    `core.sharedRepository`, a shared checkout — and since #384 the create path is
    live for EVERY worktree, not just a repo whose `.git/info/exclude` is missing:
    the private exclude never exists until the shield writes it.

    The umask is pinned rather than inherited: at the box's own 0077 the ablated
    code produces 0600 too, and the ablation would not bite.

    Ablation: drop the `exclude.touch()` and this fails, reporting 0o600."""
    old_umask = os.umask(0o022)
    try:
        wt = tmp_path / "wt"
        verify.worktree_add(project.project, wt, "feat", "main")
        exclude = _wt_private_exclude(wt)
        exclude.parent.mkdir(parents=True, exist_ok=True)
        assert not exclude.exists()
        # what the replaced write_text would have produced, measured not assumed
        probe = exclude.with_name("umask-probe")
        probe.write_text("x", encoding="utf-8")
        expected = stat.S_IMODE(probe.stat().st_mode)
        probe.unlink()

        assert _worktree_local_exclude(wt, ["/probe-mode"]) is None

        assert stat.S_IMODE(exclude.stat().st_mode) == expected, oct(
            stat.S_IMODE(exclude.stat().st_mode)
        )
    finally:
        os.umask(old_umask)


def test_worktree_local_exclude_non_git_dir_returns_none(tmp_path):
    """Not-a-repo is an EXPECTED skip, not a degradation, and must stay silent —
    while a fault in the tail means "surface it". That is why the two arms cannot
    merge.

    Two corrections to what this comment used to say, both worth keeping straight:
    not-a-repo is not a raised fault at all — `rev-parse` answers rc 128 and the
    silent arm reads that returncode — and the arm's `except` catches `GitError`,
    not `OSError`, since the chokepoint translates a failed spawn before it arrives.

    INVERSE ablation (a negative assertion needs one): make the rev-parse arm
    return a reason string instead of None and this fails."""
    assert _worktree_local_exclude(tmp_path, ["/probe-359"]) is None


def test_worktree_local_exclude_success_returns_none(project, tmp_path):
    """The happy path returns None too — `None` is "nothing to report", not
    "nothing happened": the pattern really lands in the worktree's own exclude,
    and nowhere the main checkout reads."""
    repo = project.project
    wt = tmp_path / "wt"
    verify.worktree_add(repo, wt, "feat", "main")
    shared = repo / ".git" / "info" / "exclude"
    before = shared.read_bytes()

    assert _worktree_local_exclude(wt, ["/probe-359"]) is None

    lines = _wt_private_exclude(wt).read_text(encoding="utf-8").splitlines()
    assert "/probe-359" in lines
    assert shared.read_bytes() == before


def test_provision_worktree_exclude_fault_reports_on_degraded(project, tmp_path, monkeypatch):
    """Provisioning forwards the degrade reason to `on_degraded` and otherwise
    completes. The swallow and the surfacing have to ship together: without this
    call a lost exclude turns a loud crash into a silent `git add -A` that commits
    the tool files into the story's merge.

    Ablation: delete the `on_degraded(reason)` call in `provision_worktree` and
    this fails (`msgs` stays empty)."""
    repo = project.project
    wt = tmp_path / "wt"
    verify.worktree_add(repo, wt, "feat", "main")

    def boom(*a, **kw):
        raise OSError("read-only .git")

    # patched at install's OWN binding: the exclude write goes through
    # atomic_write_bytes (#375, #384 round 4), which writes via os.fdopen on an
    # mkstemp fd, so a Path.write_bytes/Path.open patch never fires and would pass
    # vacuously — and so would a patch left on the atomic_write_TEXT this replaced.
    monkeypatch.setattr(install_mod, "atomic_write_bytes", boom)
    msgs: list[str] = []

    provision_worktree(wt, [get_profile("claude")], repo, on_degraded=msgs.append)

    assert len(msgs) == 1 and "read-only .git" in msgs[0]
    # provisioning itself still finished — only the shielding step degraded
    assert (wt / ".claude" / "skills" / "bmad-loop-sweep" / "SKILL.md").is_file()
    assert (wt / ".claude" / "settings.json").is_file()


def test_provision_worktree_non_git_no_degrade_callback(tmp_path):
    """The expected skip must not reach `on_degraded`: many callers provision plain
    non-repo directories, and a degrade event per call would train operators to
    ignore the one that matters.

    INVERSE ablation: make the subprocess arm return a reason string instead of
    None and this fails (`msgs` gets an entry)."""
    wt, repo = tmp_path / "wt", tmp_path / "repo"
    repo.mkdir()
    msgs: list[str] = []

    provision_worktree(wt, [get_profile("claude")], repo, on_degraded=msgs.append)

    assert msgs == []
    assert (wt / ".claude" / "skills" / "bmad-loop-sweep" / "SKILL.md").is_file()


# ------------------------------------------- the shield runs on the chokepoint (issue #389)
#
# The shield's git used to be a bare `subprocess.run` in this module, so its
# guards caught `subprocess.SubprocessError` and `OSError` — the raw forms of a
# timeout and a failed spawn. Through `verify.git_bytes` neither type ever
# arrives: the chokepoint translates them into `GitError` and its `GitSpawnError`
# subclass first. Every guard was re-derived for that, and these are the tests
# that would have caught leaving one behind. Each injects at the boundary the
# chokepoint actually raises from, which is why the fakes RAISE rather than
# return a non-zero rc — an rc is an answer here, never a fault.


@pytest.mark.parametrize("fault", [verify.GitError, verify.GitSpawnError])
def test_worktree_local_exclude_git_fault_before_answering_is_a_silent_skip(
    project, tmp_path, monkeypatch, fault
):
    """A git that never answered leaves the FIRST arm's contract untouched: silence.
    Both classes are covered because they enter differently — a timeout is raised
    as `GitError` itself, a spawn failure as the `GitSpawnError` subclass — and a
    guard naming only one of them looks correct until the other happens.

    Ablation: narrow the `rev-parse` guard to `subprocess.SubprocessError` (what it
    caught before the chokepoint) and both cases fail — the fault propagates out of
    a function whose whole first arm promises it never will."""
    wt = tmp_path / "wt"
    verify.worktree_add(project.project, wt, "feat", "main")

    def unanswerable(worktree, *args):
        raise fault(f"git {args[0]} did not answer in {worktree}")

    monkeypatch.setattr(install_mod, "git_bytes", unanswerable)

    assert _worktree_local_exclude(wt, ["/probe-389"]) is None


@pytest.mark.parametrize("fault", [verify.GitError, verify.GitSpawnError])
def test_worktree_local_exclude_git_fault_after_answering_degrades(
    project, tmp_path, monkeypatch, fault
):
    """Once `rev-parse` has answered, the same fault means the opposite thing — the
    shield was owed and did not happen — so it must come back as a reason the
    caller can journal, not as silence.

    Name-filtered onto the activation so everything before it runs for real; that
    is also what makes this the SECOND arm rather than the first.

    THE FLAG ASSERTION IS THE POINT OF THE SECOND HALF. This test injected exactly
    the raised activation fault for two review rounds while asserting only the
    returned reason — so it sat directly on top of a live defect and could not see
    it: the enable one line above had already made a permanent repo-format change,
    and the raise bypassed the rollback that the non-zero-rc arm had. Codex found it
    by reading this test rather than the code. A degrade assertion that stops at the
    reason string cannot tell a clean degrade from one that left state behind.

    Parametrized over both classes because they enter differently — a timeout is
    raised as `GitError` itself, a spawn failure as the `GitSpawnError` subclass.

    Ablation: drop the `except GitError` around the activation and both cases fail on
    the flag assertion (the fault reaches the tail, which cannot roll back). Note the
    ablation this docstring USED to name — "drop `GitError` from the tail's except
    tuple" — no longer applies here, since the activation's fault is now caught
    locally; the tail's guard is pinned by
    `test_shield_degrades_when_git_will_not_say_what_the_users_excludes_are`."""
    repo = project.project
    wt = tmp_path / "wt"
    verify.worktree_add(repo, wt, "feat", "main")
    assert "worktreeConfig" not in (repo / ".git" / "config").read_text(encoding="utf-8")
    real = install_mod.git_bytes

    def fault_on_activation(worktree, *args):
        if "--worktree" in args:
            raise fault("git config timed out after 120s")
        return real(worktree, *args)

    monkeypatch.setattr(install_mod, "git_bytes", fault_on_activation)

    reason = _worktree_local_exclude(wt, ["/probe-389"])

    assert reason is not None and "timed out" in reason
    # ...and the permanent format change the enable made one line earlier is gone
    assert "worktreeConfig" not in (repo / ".git" / "config").read_text(encoding="utf-8")


@pytest.mark.parametrize("fault", [verify.GitError, verify.GitSpawnError])
def test_shield_degrades_when_git_will_not_say_what_the_users_excludes_are(
    project, tmp_path, monkeypatch, fault
):
    """A git fault reading `core.excludesFile` means we did NOT FIND OUT which file
    applies — and that must skip the shield, not activate over it.

    THIS TEST REVERSES what it asserted before round 5. It used to pin the swallow,
    on the premise that "git unqueryable ⇒ no key to resolve ⇒ nothing to copy" and
    that skipping was "strictly worse than losing the seed". Both halves were wrong:

    - The premise is a false inference. It holds for `rc != 0`, which is an ANSWER
      (an unset key exits 1, and that arm is still correctly silent). A raised fault
      is not an answer, so the code mapped UNKNOWN onto ABSENT.
    - The outcome was never "a lost seed". It was a lost seed PLUS a worktree-scoped
      key that SHADOWS the file the seed failed to read — the compound harm round 4
      named as the bug. Skipping stages a bounded, journaled set of bmad-loop's own
      files; continuing silently un-ignores whatever the operator's personal excludes
      cover, which is by definition what their `.gitignore` does not, and merges it
      back into their checkout as tracked content.

    Assertion 3 is the harm and is why the operator's excludes are planted FIRST: the
    version of this test that pinned the swallow planted none at all, so it could not
    observe the shadow in either direction. Assertion 4 is the non-vacuity check —
    without it this passes just as well if the shield had excluded everything.

    Parametrized because of RE-INTRODUCTION, not coverage: `GitSpawnError` exists so
    callers can tell "git said no" from "the machine is broken", which makes
    `except GitSpawnError: return b""` the most plausible future re-narrowing — and a
    `GitError`-only test would pass against it.

    Ablation: restore `except GitError: return b""` in `_shield_inherited_excludes`
    and both cases fail — on assertion 1 (`reason` comes back None) and again on
    assertion 3 (`mine.log` shows up untracked, the shadow).

    Asserts on "did not answer", NOT "timed out": the latter is a lie for
    `GitSpawnError`, and it is the substring the activation-fault test above already
    asserts, so it could not tell which call faulted."""
    repo = project.project
    wt = tmp_path / "wt"
    verify.worktree_add(repo, wt, "feat", "main")
    users = tmp_path / "my-global-ignores"
    users.write_text("mine.log\n", encoding="utf-8")
    git(repo, "config", "core.excludesFile", str(users))
    real = install_mod.git_bytes

    def unanswerable_excludes_read(worktree, *args):
        if "core.excludesFile" in args and "--get" in args:
            raise fault(f"git config did not answer in {worktree}")
        return real(worktree, *args)

    # No monkeypatch.undo() below, unlike the read_bytes sibling: this patch is on
    # `install_mod.git_bytes`, while conftest's git() is a bare subprocess.run — so
    # the assertions' own git is untouched.
    monkeypatch.setattr(install_mod, "git_bytes", unanswerable_excludes_read)

    reason = _worktree_local_exclude(wt, ["/probe-389"])

    assert reason is not None and "did not answer" in reason
    assert not _wt_private_exclude(wt).exists()  # nothing to point a shadowing key at
    (wt / "mine.log").write_text("noise\n", encoding="utf-8")
    (wt / "probe-389").write_text("noise\n", encoding="utf-8")
    status = git(wt, "status", "--short")
    assert "mine.log" not in status  # their ignore was never shadowed — THE HARM
    assert "probe-389" in status  # ...and the shield really was skipped


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX /bin/sh stub git")
def test_shield_git_calls_carry_the_locale_pin(project, tmp_path, monkeypatch):
    """`LC_ALL=C` is one of the two things standing outside the chokepoint used to
    cost (the other, the engine-configured timeout, has no observable seam here).
    The shield embeds git's stderr verbatim in its degrade reasons, so without the
    pin those reasons are whatever language the operator's box speaks — and #236
    exists because a translated git message had already been misread once.

    Recorded from a real child process rather than asserted on the argv: the pin is
    an `env=` on the spawn, so only the child can testify that it arrived.

    The ambient `LC_ALL` is deliberately set to something else first — without that
    the recorded value could be "C" because the runner's box already said so, and
    the ablation below would pass.

    Ablation: spawn with a bare `subprocess.run` (no `env=`) and this fails — the
    stub records `en_US.UTF-8`."""
    bin_dir = tmp_path / "stubbin"
    bin_dir.mkdir()
    seen = tmp_path / "locale-seen"
    stub = bin_dir / "git"
    stub.write_text(
        f'#!/bin/sh\nprintf "%s\\n" "${{LC_ALL-<unset>}}" >> {seen}\nexit 1\n', encoding="utf-8"
    )
    stub.chmod(0o755)
    monkeypatch.setenv("PATH", str(bin_dir))
    monkeypatch.setenv("LC_ALL", "en_US.UTF-8")

    # rc 1 from the stub is the "not a repo" answer, i.e. the silent skip — all
    # this test needs is that one call was spawned at all.
    assert _worktree_local_exclude(tmp_path / "wt", ["/probe-389"]) is None

    # non-empty is itself an assertion: an unexecutable stub would leave no file,
    # and every claim below would hold vacuously.
    recorded = seen.read_text(encoding="utf-8").split()
    assert recorded and set(recorded) == {"C"}


# ----------------------------------------------------------------- hookless profiles


def test_install_into_hookless_skips_hook_registration(tmp_path, capsys):
    """A hookless profile (opencode-http) gets skills but never a hook config —
    there is nothing to register for an HTTP/SSE-transport adapter."""
    assert install_into(tmp_path, clis=("opencode-http",)) == 0
    for skill in MODULE_SKILLS:
        assert (tmp_path / ".claude" / "skills" / skill / "SKILL.md").is_file()
    assert not (tmp_path / ".claude" / "settings.json").exists()
    assert "no hooks needed (opencode-http)" in capsys.readouterr().out


def test_install_resolves_opencode_alias(tmp_path):
    assert install_into(tmp_path, clis=("opencode",)) == 0
    assert (tmp_path / ".claude" / "skills" / "bmad-loop-sweep" / "SKILL.md").is_file()
    assert not (tmp_path / ".claude" / "settings.json").exists()


def test_provision_worktree_hookless_skips_hook_merge(tmp_path):
    """Worktree provisioning for a hookless profile lays down the skill tree but
    writes no hook config (and still nothing into the worktree's .bmad-loop/)."""
    wt, repo = tmp_path / "wt", tmp_path / "repo"
    opencode = get_profile("opencode-http")
    provision_worktree(wt, [opencode], repo)

    for skill in MODULE_SKILLS:
        assert (wt / opencode.skill_tree / skill / "SKILL.md").is_file()
    assert not (wt / ".claude" / "settings.json").exists()
    assert not (wt / ".bmad-loop").exists()


def test_provision_worktree_hookless_exclude_never_blankets_worktree(project, tmp_path):
    """A hookless profile has config_path == "", which must not become the git
    exclude pattern "/" (that would exclude the entire worktree from the unit's
    `git add -A` commit). Only the skill tree is shielded."""
    repo = project.project
    wt = tmp_path / "wt"
    verify.worktree_add(repo, wt, "feat", "main")

    shared = repo / ".git" / "info" / "exclude"
    before = shared.read_bytes()

    provision_worktree(wt, [get_profile("opencode-http")], repo)

    lines = _wt_private_exclude(wt).read_text(encoding="utf-8").splitlines()
    assert "/.claude/skills" in lines
    assert "/" not in lines
    assert shared.read_bytes() == before


# ----------------------------------------------------------------- seed_globs (engine plugin)


def test_provision_worktree_seed_globs_copies_matching_tree(tmp_path):
    """A glob pattern expands against the main repo; every match is copied into
    the worktree (this is how an engine plugin's MCP skill dirs reach a worktree)."""
    wt, repo = tmp_path / "wt", tmp_path / "repo"
    skills = repo / ".claude" / "skills"
    (skills / "gameobject-create").mkdir(parents=True)
    (skills / "gameobject-create" / "SKILL.md").write_text("tool", encoding="utf-8")
    (skills / "scene-open").mkdir(parents=True)
    (skills / "scene-open" / "SKILL.md").write_text("tool", encoding="utf-8")

    provision_worktree(wt, [], repo, seed_globs=[".claude/skills/*"])

    assert (wt / ".claude" / "skills" / "gameobject-create" / "SKILL.md").read_text() == "tool"
    assert (wt / ".claude" / "skills" / "scene-open" / "SKILL.md").read_text() == "tool"


def test_provision_worktree_seed_globs_skip_existing_and_noop_when_unmatched(tmp_path):
    """Glob seeding never clobbers a match already in the worktree, and an empty
    expansion writes nothing."""
    wt, repo = tmp_path / "wt", tmp_path / "repo"
    src = repo / ".claude" / "skills" / "ping"
    src.mkdir(parents=True)
    (src / "SKILL.md").write_text("FROM_REPO", encoding="utf-8")
    dst = wt / ".claude" / "skills" / "ping"
    dst.mkdir(parents=True)
    (dst / "SKILL.md").write_text("IN_WORKTREE", encoding="utf-8")

    # one matching dir already present, plus a pattern that matches nothing
    provision_worktree(wt, [], repo, seed_globs=[".claude/skills/*", ".mcp/*"])

    assert (dst / "SKILL.md").read_text() == "IN_WORKTREE"  # not clobbered


def test_provision_worktree_seed_globs_shielded_in_local_exclude(project, tmp_path):
    """Glob-seeded paths join the worktree's local git exclude alongside seed_files,
    so a project that doesn't gitignore its skill tree won't stage them."""
    repo = project.project
    skill = repo / ".claude" / "skills" / "tests-run"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text("tool", encoding="utf-8")
    wt = tmp_path / "wt"
    verify.worktree_add(repo, wt, "feat", "main")
    shared = repo / ".git" / "info" / "exclude"
    before = shared.read_bytes()

    provision_worktree(wt, [get_profile("claude")], repo, seed_globs=[".claude/skills/*"])

    assert (wt / ".claude" / "skills" / "tests-run" / "SKILL.md").is_file()
    exclude = _wt_private_exclude(wt).read_text(encoding="utf-8").splitlines()
    assert "/.claude/skills/tests-run" in exclude
    assert git(wt, "status", "--short", "--", ".claude/skills/tests-run") == ""
    assert shared.read_bytes() == before


# ----------------------------------------------------------------- seed file modes (issue #126)


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX exec bits")
def test_provision_worktree_seed_preserves_exec_bit(tmp_path):
    """A seeded executable (vendor/bin/*) keeps +x in the worktree — a byte-only
    copy would strip the mode and the first verify command dies rc=127."""
    wt, repo = tmp_path / "wt", tmp_path / "repo"
    tool = repo / "vendor" / "bin" / "tool"
    tool.parent.mkdir(parents=True)
    tool.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    tool.chmod(0o755)

    provision_worktree(wt, [], repo, seed_files=["vendor/bin/tool"])

    assert (wt / "vendor" / "bin" / "tool").stat().st_mode & 0o111


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX exec bits")
def test_provision_worktree_seed_globs_preserve_exec_bit(tmp_path):
    """Exec bits survive the recursive directory walk of a glob-seeded tree."""
    wt, repo = tmp_path / "wt", tmp_path / "repo"
    tool = repo / "node_modules" / ".bin" / "eslint"
    tool.parent.mkdir(parents=True)
    tool.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    tool.chmod(0o755)

    provision_worktree(wt, [], repo, seed_globs=["node_modules/*"])

    assert (wt / "node_modules" / ".bin" / "eslint").stat().st_mode & 0o111


def test_copy_traversable_zip_source_copies_content(tmp_path):
    """The zip-import fallback: a zipfile.Path source has no .stat(), so the
    copy must stay content-only and not crash (the docstring's contract)."""
    zf_path = tmp_path / "pkg.zip"
    with zipfile.ZipFile(zf_path, "w") as zf:
        zf.writestr("pkg/skill/SKILL.md", "tool")
    dst = tmp_path / "out"

    _copy_traversable(zipfile.Path(zf_path, "pkg/"), dst)

    assert (dst / "skill" / "SKILL.md").read_text() == "tool"


def test_copy_traversable_skip_existing_holds_on_zip_source(tmp_path):
    """`skip_existing` guards the zip-import branch too, not just the copy2 one:
    an existing destination file survives while its absent sibling is written."""
    zf_path = tmp_path / "pkg.zip"
    with zipfile.ZipFile(zf_path, "w") as zf:
        zf.writestr("pkg/skill/SKILL.md", "FROM_ZIP")
        zf.writestr("pkg/skill/EXTRA.md", "FROM_ZIP")
    dst = tmp_path / "out"
    (dst / "skill").mkdir(parents=True)
    (dst / "skill" / "SKILL.md").write_text("ON_DISK", encoding="utf-8")

    assert _copy_traversable(zipfile.Path(zf_path, "pkg/"), dst, skip_existing=True) is True

    assert (dst / "skill" / "SKILL.md").read_text() == "ON_DISK"  # untouched
    assert (dst / "skill" / "EXTRA.md").read_text() == "FROM_ZIP"


def test_copy_traversable_skip_existing_reports_total_noop(tmp_path):
    """Nothing left to copy -> False, which is how a seed entry that copied nothing
    is still told apart from one that partially seeded."""
    src, dst = tmp_path / "src", tmp_path / "dst"
    (src / "d").mkdir(parents=True)
    (src / "d" / "f.txt").write_text("FROM_SRC", encoding="utf-8")
    (dst / "d").mkdir(parents=True)
    (dst / "d" / "f.txt").write_text("ON_DISK", encoding="utf-8")

    assert _copy_traversable(src, dst, skip_existing=True) is False
    assert (dst / "d" / "f.txt").read_text() == "ON_DISK"


def test_copy_traversable_skip_existing_never_mkdirs_over_file(tmp_path):
    """A destination FILE standing where the source has a directory is left alone:
    without the guard, mkdir(exist_ok=True) on the file raises FileExistsError."""
    src, dst = tmp_path / "src", tmp_path / "dst"
    (src / "d").mkdir(parents=True)
    (src / "d" / "f.txt").write_text("FROM_SRC", encoding="utf-8")
    dst.mkdir()
    (dst / "d").write_text("A FILE, NOT A DIR", encoding="utf-8")

    assert _copy_traversable(src, dst, skip_existing=True) is False
    assert (dst / "d").read_text() == "A FILE, NOT A DIR"
