"""Deterministic full-CLI sandbox E2E for stories mode (folder+id dispatch).

Drives the REAL `bmad-loop run/resolve/resume` binaries through REAL tmux with a
scripted fake `claude` wired in as a custom profile — no LLM, no cost, fully
reproducible. The fake fakes both the CLI and its Stop hook (it writes the
SessionStart/Stop event files itself), so no `bmad-loop init` is needed; it
leaves the id-keyed story spec on disk and the `GenericDevAdapter` synthesizes
the result from it (`_stories_synth_result`, keyed on `BMAD_LOOP_SPEC_FOLDER`).

The fake routes on the story spec's frontmatter status, like `bmad-dev-auto`
step-01: no spec / `ready-for-dev` → implement to `done`; `BMAD_LOOP_PLAN_HALT`
→ halt at `ready-for-dev` (plan-checkpoint leg); a `.block-<id>` marker → write a
`blocked` spec (a CRITICAL escalation). This exercises the CLI wiring the mock
adapter bypasses: arg parsing, prompt render, hook-signal completion, the
stories read-back, git commit, and the resolve/resume dance.

Covers, through the real binary: (1) two-story happy path, (2) `spec_checkpoint`
two-leg plan-halt + resume, (4) blocked → resolve → re-dispatch, (6)
sprint-mode regression (the new folder+id-capable dev skill installed, yet a
plain sprint run drives dev → verify → commit → sprint-status advance untouched
by the stories wiring), (7) sprint-mode intent-gap patch-restore (halt saves
the attempt as a patch → `resolve --restore-patch` re-arms to in-review +
re-stamps the spec baseline → resume re-applies the patch and dispatches an
explicit spec pointer, resuming review instead of re-implementing), and (8) the
same intent-gap patch-restore for a `sweep` deferred-work bundle (#75) —
triage → bundle halt → `resolve --restore-patch` → resume, driving the
sweep-specific CLI resolve→resume path (SweepEngine rebuilt from sweep.json).
Scenarios (3) `done_checkpoint` and (5) worktree isolation are covered
deterministically at the engine level in test_stories_engine.py; here we prove
the end-to-end CLI stack.

(9) is the post-rename row (BMAD-METHOD #2651): the same two-story happy path
against a project carrying only `bmad-build-auto`, proving the real CLI resolves
the invoked primitive off disk instead of spelling a constant.

(10) is the renderer preflight row: content-confirmed renderer stubs with missing
project-global or skill-relative inputs are refused before tmux can spawn the
fake, dry-run stays diagnostic, and the complete surface reaches the same
zero-token real-tmux success path.

(11) runs that complete renderer project under worktree isolation after removing
the renderer script unit and central config from the index. The real tmux session
can start only if runtime provisioning reconstructs the ignored `_bmad` surface.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest
import yaml
from conftest import (
    RENDERER_SCRIPT_IMPORTING_SIBLING,
    install_build_auto_skill,
    install_dev_base_skills,
)

from bmad_loop.install import (
    BMAD_SCRIPTS_SEED_REL,
    CENTRAL_CONFIG_REL,
    DEV_PRIMITIVE_NEW,
    RENDERER_CONFIG_UTILS_REL,
    RENDERER_ENTRY_REL,
    RENDERER_SCRIPT_REL,
)

# Linux-only, not merely non-win32: every fake CLI below is bash + GNU coreutils
# (`date +%s%N`; the detached-writer fake also needs setsid(1)) — BSD/macOS date
# has no %N and macOS ships no setsid utility, so skipping honestly beats failing.
HAVE_TMUX = sys.platform.startswith("linux") and shutil.which("tmux") is not None
pytestmark = pytest.mark.skipif(not HAVE_TMUX, reason="stories E2E needs real tmux on Linux")

# The fake CLI: reads the story id + spec folder from the session env (as the real
# folder+id adapter does), writes the id-keyed story spec BEFORE the Stop event so
# the adapter's read-back finds a terminal spec, and stays alive until the engine
# kills its window. Routes on the existing spec status so one script drives a fresh
# dispatch, a plan-halt leg, its post-checkpoint implement leg, and a blocked halt.
FAKE_CLI = r"""#!/usr/bin/env bash
set -e
rd="$BMAD_LOOP_RUN_DIR"; tid="$BMAD_LOOP_TASK_ID"
story="$BMAD_LOOP_STORY_KEY"; folder="$BMAD_LOOP_SPEC_FOLDER"
prompt="${1:-}"
ts=$(date +%s%N)
mkdir -p "$rd/events" "$rd/tasks/$tid"
printf '{"ts": %s, "event": "SessionStart", "task_id": "%s", "session_id": "fake-1"}' \
    "$ts" "$tid" > "$rd/events/$ts-$tid-SessionStart.json"
# argv as it ARRIVED, after profile render + tmux quoting — the orchestrator's own
# tasks/<id>/prompt.txt records the pre-render prompt, so only this file can prove a
# dispatched skill NAME actually reached the binary.
printf '%s' "$prompt" > "$rd/tasks/$tid/fake-prompt.txt"
baseline=$(git rev-parse HEAD)

# SWEEP triage (`/bmad-loop-sweep`): the triage adapter is a plain GenericAdapter
# that reads a real result.json (not the spec-synthesizing dev adapter), so write
# the partition ourselves — one bundle "fix" owning the single open ledger id — and
# Stop. Bundle dev sessions carry no BMAD_LOOP_SPEC_FOLDER and fall through to the
# SPRINT branch below, which already drives spec-<key>.md and the intent-gap /
# patch-restore contracts (here <key> = dw-fix, the bundle task key).
if printf '%s' "$prompt" | grep -q "bmad-loop-sweep"; then
    tdir="$rd/tasks/$tid"; mkdir -p "$tdir"
    printf '%s' '{"workflow": "deferred-sweep-triage", "open_ids": ["DW-1"], "already_resolved": [], "bundles": [{"name": "fix", "dw_ids": ["DW-1"], "intent": "resolve DW-1"}], "blocked": [], "skip": [], "decisions": [], "escalations": []}' \
        > "$tdir/result.json"
    ts2=$(( ts + 1 ))
    printf '{"ts": %s, "event": "Stop", "task_id": "%s", "session_id": "fake-1"}' \
        "$ts2" "$tid" > "$rd/events/$ts2-$tid-Stop.json"
    sleep 30
    exit 0
fi

# SPRINT mode (no BMAD_LOOP_SPEC_FOLDER in env): the folder+id-capable skill is
# installed, but a plain sprint run must still work. Write the result artifact
# the orchestrator scans by mtime under implementation-artifacts, make a real
# code change, and Stop — the orchestrator (not the skill) advances sprint-status.
# Routes like step-01: an in-review spec is a patch-restore re-drive (#2564); a
# committed `.intent-gap-<story>` marker makes the first dispatch halt the way
# bmad-dev-auto's review does on an intent gap (save attempt as patch, revert,
# block); otherwise a plain implement-to-done.
if [ -z "$folder" ]; then
    impl="_bmad-output/implementation-artifacts"
    mkdir -p "$impl"
    spec="$impl/spec-$story.md"
    patch="$impl/attempt-$story.patch"
    status=""
    [ -f "$spec" ] && status=$(sed -n 's/^status:[[:space:]]*//p' "$spec" | head -1 | tr -d "'\" ")
    if [ "$status" = "in-review" ]; then
        # Patch-restore re-drive: resume REVIEW on the restored diff — never
        # re-implement. Enforce the two orchestrator-side contracts the way the
        # real step-01 would: the prompt must point at the spec explicitly (an
        # in-review spec only routes to step-04 through the spec-pointer intent
        # check), and the attempted change must already be back on the tree.
        if ! printf '%s' "$prompt" | grep -qF "$spec"; then
            printf -- '---\ntitle: %s\nstatus: blocked\nbaseline_commit: %s\n---\n\n## Auto Run Result\n\n- Status: blocked\n\nprompt lacks the spec pointer.\n' \
                "$story" "$baseline" > "$spec"
        elif ! grep -q "attempted reading" src.txt; then
            printf -- '---\ntitle: %s\nstatus: blocked\nbaseline_commit: %s\n---\n\n## Auto Run Result\n\n- Status: blocked\n\ntree was not restored.\n' \
                "$story" "$baseline" > "$spec"
        else
            printf -- '---\ntitle: %s\nstatus: done\nbaseline_commit: %s\n---\n\n## Intent\n\nx\n\n## Auto Run Result\n\n- Status: done\n\nSummary: reviewed the restored change.\n' \
                "$story" "$baseline" > "$spec"
        fi
    elif [ -f ".intent-gap-$story" ] && [ ! -f "$patch" ]; then
        echo "attempted reading for $story" >> src.txt
        git diff HEAD > "$patch"
        git checkout -- src.txt
        printf -- '---\ntitle: %s\nstatus: blocked\nbaseline_commit: %s\n---\n\n## Intent\n\nx\n\n## Auto Run Result\n\n- Status: blocked\n\nintent gap; saved patch: %s\n' \
            "$story" "$baseline" "$patch" > "$spec"
    else
        echo "impl for $story" >> src.txt
        printf -- '---\ntitle: %s\nstatus: done\nbaseline_commit: %s\n---\n\n## Intent\n\nx\n\n## Auto Run Result\n\n- Status: done\n\nSummary: sprint.\n' \
            "$story" "$baseline" > "$spec"
    fi
    ts2=$(( ts + 1 ))
    printf '{"ts": %s, "event": "Stop", "task_id": "%s", "session_id": "fake-1"}' \
        "$ts2" "$tid" > "$rd/events/$ts2-$tid-Stop.json"
    sleep 30
    exit 0
fi

sdir="$folder/stories"
mkdir -p "$sdir"
spec="$sdir/$story-slug.md"
existing=$(ls "$sdir/$story"-*.md 2>/dev/null | head -1 || true)
status=""
[ -n "$existing" ] && status=$(sed -n 's/^status:[[:space:]]*//p' "$existing" | head -1 | tr -d "'\" ")

write_done() {
    echo "impl for $story" >> src.txt
    printf -- '---\ntitle: %s\nstatus: done\nbaseline_commit: %s\n---\n\n# %s\nimplemented.\n' \
        "$story" "$baseline" "$story" > "$spec"
}
write_planned() {
    printf -- '---\ntitle: %s\nstatus: ready-for-dev\nbaseline_commit: %s\n---\n\n# %s\nplanned.\n' \
        "$story" "$baseline" "$story" > "$spec"
}
write_blocked() {
    printf -- '---\ntitle: %s\nstatus: blocked\nbaseline_commit: %s\n---\n\n# %s\n\n## Auto Run Result\n\n- Status: blocked\n\nNeeds a human decision.\n' \
        "$story" "$baseline" "$story" > "$spec"
}

if [ "$status" = "ready-for-dev" ] || [ "$status" = "in-progress" ] || [ "$status" = "draft" ]; then
    write_done                       # re-dispatch after a plan-checkpoint or a re-arm
elif [ -n "$BMAD_LOOP_PLAN_HALT" ]; then
    write_planned                    # spec_checkpoint leg 1: halt after planning
elif [ -f "$folder/.block-$story" ]; then
    write_blocked                    # poisoned story: first dispatch blocks
else
    write_done                       # normal fresh dispatch
fi

ts2=$(( ts + 1 ))
printf '{"ts": %s, "event": "Stop", "task_id": "%s", "session_id": "fake-1"}' \
    "$ts2" "$tid" > "$rd/events/$ts2-$tid-Stop.json"
sleep 30
"""

PROFILE_TOML = """\
name = "fakestories"
binary = "{binary}"
bypass_args = []
usage_parser = "none"
skill_tree = ".claude/skills"

[hooks]
dialect = "claude-settings-json"
config_path = ".claude/settings.json"
events = {{ SessionStart = "SessionStart", Stop = "Stop" }}
"""

SPEC_FOLDER = "_bmad-output/epic-1"
CLI = [sys.executable, "-m", "bmad_loop.cli"]

# A fake CLI that writes SessionStart and then sleeps forever — it NEVER fires a
# Stop hook, so the dev session can only end via the orchestrator's own timeout
# fire + bounded teardown (#157). Because the session never ends a turn, the
# result-less-Stop stall machinery never engages either, exactly the wedged-in-a-
# tool-call shape the issue reported.
TIMEOUT_FAKE_CLI = r"""#!/usr/bin/env bash
set -e
rd="$BMAD_LOOP_RUN_DIR"; tid="$BMAD_LOOP_TASK_ID"
ts=$(date +%s%N)
mkdir -p "$rd/events"
printf '{"ts": %s, "event": "SessionStart", "task_id": "%s", "session_id": "fake-1"}' \
    "$ts" "$tid" > "$rd/events/$ts-$tid-SessionStart.json"
# Background + wait keeps the same process group as a foreground sleep, but
# records the child's pid so the test can prove teardown reaped descendants,
# not just this shell (whose cmdline is all the pgrep check can see).
sleep 100000 &
child=$!
printf '%s\n' "$child" > "$rd/tasks/$tid/fake-child.pid"
wait "$child"
"""

# A fake CLI that ends CLEANLY (writes a `done` spec + Stop, then idles like a real
# interactive session) but first `setsid`-detaches a straggler into its OWN session.
# Unlike the :185-204 same-pgid child (which tmux's SIGHUP reaps), a setsid child
# escapes the pane pgid entirely — the #183/#139 repro the pre-harvest descendant
# reap must cover before the worktree is merged and removed. Sprint mode: writes the
# id-keyed done spec + a real code change so the run merges and tears the worktree down.
DETACHED_WRITER_FAKE_CLI = r"""#!/usr/bin/env bash
set -e
rd="$BMAD_LOOP_RUN_DIR"; tid="$BMAD_LOOP_TASK_ID"; story="$BMAD_LOOP_STORY_KEY"
ts=$(date +%s%N)
mkdir -p "$rd/events"
printf '{"ts": %s, "event": "SessionStart", "task_id": "%s", "session_id": "fake-1"}' \
    "$ts" "$tid" > "$rd/events/$ts-$tid-SessionStart.json"
baseline=$(git rev-parse HEAD)

# Detach a straggler into a NEW session (setsid): $! is the setsid'd process itself
# (a non-interactive shell runs background jobs in its own pgrp, so setsid does not
# fork) — it now leads its own session and survives the pane pgid's SIGHUP.
setsid sleep 100000 &
child=$!
printf '%s\n' "$child" > "$rd/tasks/$tid/fake-child.pid"

# Sprint-mode result: a real code change + the id-keyed done spec the dev synthesis
# reads back (written under the worktree cwd in isolation mode).
impl="_bmad-output/implementation-artifacts"
mkdir -p "$impl"
echo "impl for $story" >> src.txt
printf -- '---\ntitle: %s\nstatus: done\nbaseline_commit: %s\n---\n\n## Intent\n\nx\n\n## Auto Run Result\n\n- Status: done\n\nSummary: sprint.\n' \
    "$story" "$baseline" > "$impl/spec-$story.md"

ts2=$(( ts + 1 ))
printf '{"ts": %s, "event": "Stop", "task_id": "%s", "session_id": "fake-1"}' \
    "$ts2" "$tid" > "$rd/events/$ts2-$tid-Stop.json"
# Stay alive like an idle interactive session so the pane shell is still live when
# the engine kills it: the harvest sees the detached child as our descendant only
# while we (its parent) are still around. Long enough to outlast Stop -> kill_window
# on a loaded CI host (the window kill terminates this sleep anyway, so it costs
# nothing) — else the shell could exit first, reparenting the child to init.
sleep 600
"""


def _git(root: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True, text=True)


def _entry(story_id: str, **over) -> dict:
    d = {"id": story_id, "title": f"Story {story_id}", "description": "does a thing"}
    d.update(over)
    return d


def _scaffold(root: Path, entries: list[dict], *, install_skills=install_dev_base_skills) -> None:
    """A committed, clean sandbox: git repo, BMAD config + artifact dirs, the
    base-skill stubs the stories preflight requires (incl. the folder+id dispatch
    probe), a stories.yaml + SPEC.md, the fake-CLI profile, and a stories-mode
    policy — everything committed so the run-start worktree_clean gate passes.

    ``install_skills`` picks the dev-primitive ERA laid on disk: the legacy
    `install_dev_base_skills` by default, `install_build_auto_skill` for the
    post-rename tree. Both take ``(root, *, folder_id)`` and write the folder+id
    dispatch probe under whichever name `resolve_dev_primitive` will pick."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "src.txt").write_text("original\n", encoding="utf-8")
    (root / ".gitignore").write_text(".bmad-loop/runs/\n", encoding="utf-8")

    cfg = root / "_bmad" / "bmm"
    cfg.mkdir(parents=True)
    (cfg / "config.yaml").write_text(
        "implementation_artifacts: '{project-root}/_bmad-output/implementation-artifacts'\n"
        "planning_artifacts: '{project-root}/_bmad-output/planning-artifacts'\n",
        encoding="utf-8",
    )
    for sub in ("implementation-artifacts", "planning-artifacts"):
        (root / "_bmad-output" / sub).mkdir(parents=True, exist_ok=True)
        (root / "_bmad-output" / sub / ".keep").write_text("", encoding="utf-8")

    install_skills(root, folder_id=True)  # tree matches PROFILE_TOML's skill_tree

    folder = root / SPEC_FOLDER
    (folder / "stories").mkdir(parents=True)
    (folder / "SPEC.md").write_text("---\ntitle: Epic 1\n---\n# Epic 1\n", encoding="utf-8")
    (folder / "stories.yaml").write_text(yaml.safe_dump(entries, sort_keys=False), encoding="utf-8")

    fake = root / ".bmad-loop" / "fake-cli.sh"
    fake.parent.mkdir(parents=True, exist_ok=True)
    fake.write_text(FAKE_CLI, encoding="utf-8")
    os.chmod(fake, 0o755)
    profiles = root / ".bmad-loop" / "profiles"
    profiles.mkdir(parents=True)
    (profiles / "fakestories.toml").write_text(
        PROFILE_TOML.format(binary=str(fake)), encoding="utf-8"
    )
    (root / ".bmad-loop" / "policy.toml").write_text(
        '[adapter]\nname = "fakestories"\n\n'
        "[review]\nenabled = false\n\n"
        f'[stories]\nsource = "stories"\nspec_folder = "{SPEC_FOLDER}"\n',
        encoding="utf-8",
    )

    _git(root, "init", "-q", "-b", "main")
    _git(root, "config", "user.email", "e2e@test")
    _git(root, "config", "user.name", "e2e")
    _git(root, "config", "core.fsync", "none")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "sandbox")


def _scaffold_renderer(
    root: Path, *, script: bool = True, config: bool = True, workflow: bool = True
) -> None:
    """A clean renderer-era stories sandbox with independently mutable surfaces.

    The helper is always present so ``script=False`` isolates the entry-point gate,
    while ``config=False`` isolates the central-config gate. Every mutation after
    :func:`_scaffold` is committed so the real run's clean-worktree check cannot
    hide a missing renderer finding behind an earlier refusal.
    """

    def install_renderer_skills(skill_root: Path, *, folder_id: bool) -> None:
        install_build_auto_skill(skill_root, folder_id=folder_id, renderer_stub=True)

    _scaffold(root, [_entry("1")], install_skills=install_renderer_skills)

    helper = root / RENDERER_CONFIG_UTILS_REL
    helper.parent.mkdir(parents=True, exist_ok=True)
    helper.write_text("# renderer config helper\n", encoding="utf-8")
    if script:
        (root / RENDERER_SCRIPT_REL).write_text(RENDERER_SCRIPT_IMPORTING_SIBLING, encoding="utf-8")
    if config:
        central = root / CENTRAL_CONFIG_REL
        central.parent.mkdir(parents=True, exist_ok=True)
        central.write_text("[core]\nname = 'renderer-e2e'\n", encoding="utf-8")
    if not workflow:
        (root / ".claude" / "skills" / DEV_PRIMITIVE_NEW / RENDERER_ENTRY_REL).unlink()

    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "renderer fixture")


def _scaffold_sprint(
    root: Path, story_key: str, fake_cli: str = FAKE_CLI, extra_policy: str = ""
) -> None:
    """A committed, clean SPRINT-mode sandbox carrying the SAME new folder+id-
    capable bmad-dev-auto skill stub as `_scaffold` (with the folder+id dispatch
    probe content) — the regression point: installing that skill must not disturb
    the default sprint path. sprint-status.yaml holds one ready-for-dev story; the
    policy has NO [stories] section, so the run is plain sprint mode. ``fake_cli``
    swaps the CLI script (e.g. a never-Stop timeout fake); ``extra_policy`` appends
    to policy.toml (e.g. a [limits] block)."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "src.txt").write_text("original\n", encoding="utf-8")
    (root / ".gitignore").write_text(".bmad-loop/runs/\n", encoding="utf-8")

    cfg = root / "_bmad" / "bmm"
    cfg.mkdir(parents=True)
    (cfg / "config.yaml").write_text(
        "implementation_artifacts: '{project-root}/_bmad-output/implementation-artifacts'\n"
        "planning_artifacts: '{project-root}/_bmad-output/planning-artifacts'\n",
        encoding="utf-8",
    )
    impl = root / "_bmad-output" / "implementation-artifacts"
    for sub in ("implementation-artifacts", "planning-artifacts"):
        (root / "_bmad-output" / sub).mkdir(parents=True, exist_ok=True)
        (root / "_bmad-output" / sub / ".keep").write_text("", encoding="utf-8")

    # the SAME new folder+id-capable skill stub the stories scaffold installs
    install_dev_base_skills(root, folder_id=True)  # tree matches PROFILE_TOML's skill_tree

    sprint = {
        "generated": "01-06-2026 10:00",
        "last_updated": "01-06-2026 10:00",
        "project": "sandbox",
        "project_key": "NOKEY",
        "tracking_system": "file-system",
        "development_status": {story_key: "ready-for-dev"},
    }
    (impl / "sprint-status.yaml").write_text(
        yaml.safe_dump(sprint, sort_keys=False), encoding="utf-8"
    )

    fake = root / ".bmad-loop" / "fake-cli.sh"
    fake.parent.mkdir(parents=True, exist_ok=True)
    fake.write_text(fake_cli, encoding="utf-8")
    os.chmod(fake, 0o755)
    profiles = root / ".bmad-loop" / "profiles"
    profiles.mkdir(parents=True)
    (profiles / "fakestories.toml").write_text(
        PROFILE_TOML.format(binary=str(fake)), encoding="utf-8"
    )
    (root / ".bmad-loop" / "policy.toml").write_text(
        '[adapter]\nname = "fakestories"\n\n'
        "[review]\nenabled = false\n\n"
        '[gates]\nmode = "none"\n' + extra_policy,
        encoding="utf-8",
    )

    _git(root, "init", "-q", "-b", "main")
    _git(root, "config", "user.email", "e2e@test")
    _git(root, "config", "user.name", "e2e")
    _git(root, "config", "core.fsync", "none")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "sandbox")


def _scaffold_sweep(root: Path) -> None:
    """A committed, clean SWEEP-mode sandbox: same folder+id-capable dev-skill
    stubs as `_scaffold_sprint`, but no sprint-status — instead a canonical
    deferred-work.md ledger with one open entry (DW-1). The policy has no
    [stories] section; the run is a plain `bmad-loop sweep`."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "src.txt").write_text("original\n", encoding="utf-8")
    (root / ".gitignore").write_text(".bmad-loop/runs/\n", encoding="utf-8")

    cfg = root / "_bmad" / "bmm"
    cfg.mkdir(parents=True)
    (cfg / "config.yaml").write_text(
        "implementation_artifacts: '{project-root}/_bmad-output/implementation-artifacts'\n"
        "planning_artifacts: '{project-root}/_bmad-output/planning-artifacts'\n",
        encoding="utf-8",
    )
    impl = root / "_bmad-output" / "implementation-artifacts"
    for sub in ("implementation-artifacts", "planning-artifacts"):
        (root / "_bmad-output" / sub).mkdir(parents=True, exist_ok=True)
        (root / "_bmad-output" / sub / ".keep").write_text("", encoding="utf-8")

    # the SAME folder+id-capable skill stubs the other scaffolds install
    install_dev_base_skills(root, folder_id=True)  # tree matches PROFILE_TOML's skill_tree

    # canonical DW-format ledger (no legacy content → migration is skipped)
    (impl / "deferred-work.md").write_text(
        "# Deferred Work\n\n"
        "### DW-1: item DW-1\n\n"
        "origin: test, 2026-06-01\nlocation: src.txt:1\nreason: test entry.\nstatus: open\n",
        encoding="utf-8",
    )

    fake = root / ".bmad-loop" / "fake-cli.sh"
    fake.parent.mkdir(parents=True, exist_ok=True)
    fake.write_text(FAKE_CLI, encoding="utf-8")
    os.chmod(fake, 0o755)
    profiles = root / ".bmad-loop" / "profiles"
    profiles.mkdir(parents=True)
    (profiles / "fakestories.toml").write_text(
        PROFILE_TOML.format(binary=str(fake)), encoding="utf-8"
    )
    (root / ".bmad-loop" / "policy.toml").write_text(
        '[adapter]\nname = "fakestories"\n\n'
        "[review]\nenabled = false\n\n"
        '[gates]\nmode = "none"\n',
        encoding="utf-8",
    )

    _git(root, "init", "-q", "-b", "main")
    _git(root, "config", "user.email", "e2e@test")
    _git(root, "config", "user.name", "e2e")
    _git(root, "config", "core.fsync", "none")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "sandbox")


def _dw_status(root: Path, dw_id: str) -> str:
    """The status field of one deferred-work ledger entry ('' if absent)."""
    text = (root / "_bmad-output" / "implementation-artifacts" / "deferred-work.md").read_text(
        encoding="utf-8"
    )
    in_entry = False
    for line in text.splitlines():
        if line.startswith(f"### {dw_id}:"):
            in_entry = True
        elif line.startswith("### "):
            in_entry = False
        elif in_entry and line.startswith("status:"):
            return line.split(":", 1)[1].strip()
    return ""


def _sprint_status(root: Path, story_key: str) -> str:
    doc = yaml.safe_load(
        (root / "_bmad-output" / "implementation-artifacts" / "sprint-status.yaml").read_text(
            encoding="utf-8"
        )
    )
    return doc.get("development_status", {}).get(story_key, "?")


def _run(root: Path, *args: str, timeout: float = 150) -> subprocess.CompletedProcess:
    return subprocess.run(
        [*CLI, args[0], "--project", str(root), *args[1:]],
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=str(root),
    )


def _status(root: Path, story_id: str) -> str:
    spec = root / SPEC_FOLDER / "stories" / f"{story_id}-slug.md"
    if not spec.is_file():
        return "pending"
    for line in spec.read_text(encoding="utf-8").splitlines():
        if line.startswith("status:"):
            return line.split(":", 1)[1].strip().strip("'\"")
    return "?"


def _commit_count(root: Path) -> int:
    out = subprocess.run(
        ["git", "-C", str(root), "rev-list", "--count", "HEAD"],
        capture_output=True,
        text=True,
    )
    return int(out.stdout.strip())


def _run_id(root: Path) -> str:
    runs = sorted((root / ".bmad-loop" / "runs").iterdir())
    assert runs, "no run dir created"
    return runs[-1].name


def test_e2e_two_story_happy_path(tmp_path):
    root = tmp_path / "sbx"
    _scaffold(root, [_entry("1"), _entry("2")])
    base = _commit_count(root)

    proc = _run(root, "run")
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert _status(root, "1") == "done"
    assert _status(root, "2") == "done"
    # one squashed story commit per story above the sandbox baseline
    assert _commit_count(root) == base + 2


def test_e2e_two_story_happy_path_build_auto(tmp_path):
    """Scenario 9, the post-rename twin of the happy path (BMAD-METHOD #2651):
    the project carries ONLY `bmad-build-auto`, so the real CLI has to resolve the
    invoked primitive off disk. The fake routes on env vars and spec status and
    never on the skill name, so the run lands identically either era — the name
    assertion below is the whole difference between this row and the legacy one,
    and without it this test would pass with the resolution ablated.

    Asserted against the prompt the FAKE recorded (post-render, post-tmux argv),
    not the orchestrator's own prompt.txt: a name that never reached the binary
    would still be in prompt.txt."""
    root = tmp_path / "sbx"
    _scaffold(root, [_entry("1"), _entry("2")], install_skills=install_build_auto_skill)
    base = _commit_count(root)

    proc = _run(root, "run")
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert _status(root, "1") == "done"
    assert _status(root, "2") == "done"
    assert _commit_count(root) == base + 2

    run_dir = root / ".bmad-loop" / "runs" / _run_id(root)
    dispatched = [
        p.read_text(encoding="utf-8") for p in (run_dir / "tasks").glob("*/fake-prompt.txt")
    ]
    assert len(dispatched) == 2, dispatched  # one dev session per story, both recorded
    # The fakestories profile renders `{prompt}` verbatim, so the dispatch is the
    # literal skill invocation; a codex-shaped profile would rewrite it to `$skill`.
    assert all(p.startswith("/bmad-build-auto Spec folder: ") for p in dispatched), dispatched
    assert not any("bmad-dev-auto" in p for p in dispatched), dispatched


def test_e2e_renderer_missing_script_fails_validate_and_refuses_before_spawn(tmp_path):
    root = tmp_path / "sbx"
    _scaffold_renderer(root, script=False)

    validate = _run(root, "validate", "--json")
    assert validate.returncode == 1
    findings = {finding["check"]: finding for finding in json.loads(validate.stdout)["findings"]}
    renderer = findings["skills.dev-renderer"]
    assert renderer["severity"] == "problem"
    assert renderer["detail"] == {
        "tree": ".claude/skills",
        "skill": DEV_PRIMITIVE_NEW,
        "missing_scripts": [RENDERER_SCRIPT_REL],
    }

    run = _run(root, "run")
    assert run.returncode == 1
    assert RENDERER_SCRIPT_REL in run.stderr and "HALT" in run.stderr
    assert not (root / ".bmad-loop" / "runs").exists(), "preflight must precede session spawn"

    dry = _run(root, "run", "--dry-run")
    assert dry.returncode == 0, dry.stderr or dry.stdout
    assert "NOT runnable" in dry.stderr and RENDERER_SCRIPT_REL in dry.stderr
    assert "Story id: 1." in dry.stdout
    assert not (root / ".bmad-loop" / "runs").exists()


def test_e2e_renderer_missing_central_config_refuses_before_spawn(tmp_path):
    root = tmp_path / "sbx"
    _scaffold_renderer(root, config=False)

    run = _run(root, "run")
    assert run.returncode == 1
    assert CENTRAL_CONFIG_REL in run.stderr and "HALT" in run.stderr
    assert not (root / ".bmad-loop" / "runs").exists(), "preflight must precede session spawn"


def test_e2e_renderer_missing_workflow_refuses_before_spawn(tmp_path):
    root = tmp_path / "sbx"
    _scaffold_renderer(root, workflow=False)

    run = _run(root, "run")
    assert run.returncode == 1
    assert RENDERER_ENTRY_REL in run.stderr and "HALT" in run.stderr
    assert not (root / ".bmad-loop" / "runs").exists(), "preflight must precede session spawn"


def test_e2e_renderer_complete_surface_reaches_real_tmux_fake_cli(tmp_path):
    root = tmp_path / "sbx"
    _scaffold_renderer(root)
    base = _commit_count(root)

    run = _run(root, "run")
    assert run.returncode == 0, run.stderr or run.stdout
    assert _status(root, "1") == "done"
    assert _commit_count(root) == base + 1

    run_dir = root / ".bmad-loop" / "runs" / _run_id(root)
    dispatched = list((run_dir / "tasks").glob("*/fake-prompt.txt"))
    assert len(dispatched) == 1, dispatched
    assert dispatched[0].read_text(encoding="utf-8").startswith("/bmad-build-auto Spec folder: ")


def test_e2e_renderer_surface_is_seeded_into_isolated_real_tmux_session(tmp_path):
    """The required sandbox E2E for the runtime half of renderer support.

    The main checkout keeps the renderer files as ignored working-tree content,
    while the linked worktree starts without them. A green completion crosses the
    real CLI, worktree provision, completeness gates, tmux fake, verification and
    merge path; no LLM is invoked.
    """
    root = tmp_path / "sbx"
    _scaffold_renderer(root)
    gitignore = root / ".gitignore"
    existing = gitignore.read_text(encoding="utf-8")
    prefix = existing if not existing or existing.endswith("\n") else existing + "\n"
    gitignore.write_text(
        prefix + f"{BMAD_SCRIPTS_SEED_REL}/\n{CENTRAL_CONFIG_REL}\n",
        encoding="utf-8",
    )
    policy = root / ".bmad-loop" / "policy.toml"
    policy.write_text(
        policy.read_text(encoding="utf-8") + '\n[scm]\nisolation = "worktree"\n',
        encoding="utf-8",
    )
    _git(root, "rm", "-r", "--cached", "--", BMAD_SCRIPTS_SEED_REL, CENTRAL_CONFIG_REL)
    _git(root, "add", ".gitignore", ".bmad-loop/policy.toml")
    _git(root, "commit", "-q", "-m", "ignore renderer runtime surface")
    base = _commit_count(root)

    run = _run(root, "run")

    assert run.returncode == 0, run.stderr or run.stdout
    assert _status(root, "1") == "done"
    # Isolation records the unit commit and the local integration commit.
    assert _commit_count(root) == base + 2
    run_dir = root / ".bmad-loop" / "runs" / _run_id(root)
    kinds = [
        json.loads(line)["kind"]
        for line in (run_dir / "journal.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert "worktree-opened" in kinds and "unit-merged" in kinds
    assert "story-escalated" not in kinds


def test_e2e_spec_checkpoint_two_leg(tmp_path):
    root = tmp_path / "sbx"
    _scaffold(root, [_entry("1", spec_checkpoint=True)])
    base = _commit_count(root)

    # leg 1: dispatch halts after planning → run pauses at the plan checkpoint
    proc = _run(root, "run")
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert _status(root, "1") == "ready-for-dev"  # planned, not implemented
    assert _commit_count(root) == base  # no commit yet

    run_id = _run_id(root)
    st = _run(root, "status", run_id)
    # deterministic status line: `PAUSED (plan-checkpoint) — …` — assert BOTH the
    # paused state and the specific stage, not either-or (a weak `or` would pass on
    # any paused run regardless of stage).
    out = st.stdout.lower()
    assert "paused" in out and "plan-checkpoint" in out

    # leg 2: resume re-dispatches straight to implementation → done + commit
    resume = _run(root, "resume", run_id)
    assert resume.returncode == 0, resume.stderr or resume.stdout
    assert _status(root, "1") == "done"
    assert _commit_count(root) == base + 1


def test_e2e_blocked_resolve_redispatch(tmp_path):
    root = tmp_path / "sbx"
    _scaffold(root, [_entry("1"), _entry("2")])
    (root / SPEC_FOLDER / ".block-1").write_text("", encoding="utf-8")  # story 1 poisoned
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "poison story 1")
    base = _commit_count(root)

    # story 1 blocks → run pauses at the escalation (story 2 not leapfrogged)
    proc = _run(root, "run")
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert _status(root, "1") == "blocked"
    assert _status(root, "2") == "pending"
    run_id = _run_id(root)

    # resolve (non-interactive) re-arms blocked → ready-for-dev + strips the halt
    resolve = _run(root, "resolve", run_id, "--no-interactive", "--no-resume")
    assert resolve.returncode == 0, resolve.stderr or resolve.stdout
    assert _status(root, "1") == "ready-for-dev"

    # resume re-dispatches story 1 to done and continues to story 2
    resume = _run(root, "resume", run_id)
    assert resume.returncode == 0, resume.stderr or resume.stdout
    assert _status(root, "1") == "done"
    assert _status(root, "2") == "done"
    assert _commit_count(root) == base + 2


def test_e2e_sprint_intent_gap_patch_restore(tmp_path):
    # Scenario 7 (review F1/F2, end-to-end): a sprint-mode intent-gap halt saves
    # the attempted change as a patch and reverts; `resolve --restore-patch`
    # re-arms the spec to in-review and re-stamps its baseline; resume re-applies
    # the patch onto the tree and dispatches an EXPLICIT spec pointer, so the
    # (fake) skill resumes review on the restored diff instead of re-implementing.
    # The fake blocks loudly if the prompt lacks the pointer or the tree was not
    # restored, so a `done` landing proves both contracts held.
    root = tmp_path / "sbx"
    story = "1-1-thing"
    _scaffold_sprint(root, story)
    (root / f".intent-gap-{story}").write_text("", encoding="utf-8")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "poison: intent gap")
    base = _commit_count(root)
    impl = root / "_bmad-output" / "implementation-artifacts"
    spec = impl / f"spec-{story}.md"
    patch = impl / f"attempt-{story}.patch"

    # leg 1: the dev session halts on the intent gap — patch saved, tree reverted
    proc = _run(root, "run")
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert "status: blocked" in spec.read_text(encoding="utf-8")
    assert patch.is_file()  # the attempted change survives the revert
    assert "attempted reading" not in (root / "src.txt").read_text(encoding="utf-8")
    run_id = _run_id(root)

    # the human confirms the attempted reading: latch the restore
    resolve = _run(
        root, "resolve", run_id, "--no-interactive", "--no-resume", "--restore-patch", str(patch)
    )
    assert resolve.returncode == 0, resolve.stderr or resolve.stdout
    text = spec.read_text(encoding="utf-8")
    assert "status: in-review" in text  # restore routing: step-01 -> step-04
    head = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    assert f"baseline_revision: {head}" in text  # F2: spec baseline re-stamped

    # resume: patch re-applied, review resumed, story lands done + committed
    resume = _run(root, "resume", run_id)
    assert resume.returncode == 0, resume.stderr or resume.stdout
    final = spec.read_text(encoding="utf-8")
    assert "status: done" in final, final  # fake blocks loudly on a broken contract
    assert _sprint_status(root, story) == "done"
    assert _commit_count(root) == base + 1
    src = (root / "src.txt").read_text(encoding="utf-8")
    assert src.count("attempted reading") == 1  # restored from the patch, not re-implemented
    # F1: the re-drive dispatch pointed at the spec, never the bare story key
    run_dir = root / ".bmad-loop" / "runs" / run_id
    prompts = [p.read_text(encoding="utf-8") for p in (run_dir / "tasks").glob("*/prompt.txt")]
    assert any(str(spec) in p for p in prompts)


def _tmux_has_session(name: str) -> bool:
    return subprocess.run(["tmux", "has-session", "-t", name], capture_output=True).returncode == 0


def test_e2e_session_timeout_teardown(tmp_path, monkeypatch):
    """#157 end to end through the real binary + real tmux: a dev session wedged
    forever (SessionStart, then sleep — never a Stop) is bounded only by the
    session timeout, and the fix makes that firing timely and observable. The
    1-minute policy floor is too coarse for a fast test, so the engine's
    BMAD_LOOP_SESSION_TIMEOUT_S seam drives a 3-second budget."""
    root = tmp_path / "sbx"
    story = "1-1-timeout"
    _scaffold_sprint(
        root,
        story,
        fake_cli=TIMEOUT_FAKE_CLI,
        extra_policy="\n[limits]\nmax_dev_attempts = 1\nteardown_grace_s = 5\n",
    )
    # inherited by the `bmad-loop run` subprocess (_run passes no env=)
    monkeypatch.setenv("BMAD_LOOP_SESSION_TIMEOUT_S", "3")

    proc = _run(root, "run", timeout=90)
    assert proc.returncode == 0, proc.stderr or proc.stdout

    run_id = _run_id(root)
    run_dir = root / ".bmad-loop" / "runs" / run_id

    # (1) session-end status=timeout, journaled promptly, with the fire forensics
    journal = [
        json.loads(ln)
        for ln in (run_dir / "journal.jsonl").read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    ends = [j for j in journal if j["kind"] == "session-end" and j.get("status") == "timeout"]
    assert ends, f"no session-end status=timeout: {[j['kind'] for j in journal]}"
    end = ends[0]
    assert end.get("fired_at"), end
    assert end["teardown_s"] < 15.0, f"teardown gap not small (kill hung?): {end['teardown_s']}"
    assert end.get("expired_clock") in ("monotonic", "wall", "both"), end
    task_id = end["task_id"]

    # (2) the fire moment left a timeout-fired breadcrumb, distinct from teardown
    tdir = run_dir / "tasks" / task_id
    life = [
        json.loads(ln)
        for ln in (tdir / "session-lifecycle.jsonl").read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    assert any(ln.get("event") == "timeout-fired" for ln in life), life

    # (3) the wait loop's proof-of-life exists and is recent (not the frozen gap)
    hb = json.loads((tdir / "heartbeat.json").read_text(encoding="utf-8"))
    assert time.time() - hb["ts"] < 120, hb

    # (4) teardown actually reaped the session — no orphan tmux session/process
    assert not _tmux_has_session(f"bmad-loop-{run_id}")
    if shutil.which("pgrep"):
        pg = subprocess.run(["pgrep", "-af", "fake-cli.sh"], capture_output=True, text=True)
        assert not [ln for ln in pg.stdout.splitlines() if str(root) in ln], pg.stdout
    # The pgrep filter can only see the shell's cmdline; probe the recorded sleep
    # descendant directly — the escalation force-kills pane-root pids, so a
    # regression there would leak exactly this child while pgrep stays clean.
    # Poll briefly: a just-killed child can linger as a zombie (kill 0 succeeds)
    # until init reaps it after the shell died.
    pid_file = tdir / "fake-child.pid"
    assert pid_file.is_file(), "fake CLI never recorded its sleep child"
    fake_pid = int(pid_file.read_text(encoding="utf-8"))
    deadline = time.monotonic() + 10
    while True:
        try:
            os.kill(fake_pid, 0)
        except ProcessLookupError:
            break  # dead and reaped — teardown covered the descendant
        assert time.monotonic() < deadline, f"sleep child {fake_pid} survived teardown"
        time.sleep(0.1)


def test_e2e_detached_writer_reaped_before_worktree_teardown(tmp_path):
    """#183/#139 end to end: a dev session `setsid`-detaches a straggler into its own
    session (escaping the pane pgid tmux's SIGHUP reaps), then ends CLEANLY via a
    Stop + done spec. The verified-kill reap must chase the harvested descendant tree
    and reap the straggler BEFORE the worktree is merged and removed — so the
    detached pid is dead, the worktree is gone, and no `worktree-teardown-degraded`
    fires (the #139 signature). Worktree isolation makes the teardown real."""
    root = tmp_path / "sbx"
    story = "1-1-detach"
    _scaffold_sprint(
        root,
        story,
        fake_cli=DETACHED_WRITER_FAKE_CLI,
        extra_policy=(
            '\n[scm]\nisolation = "worktree"\n\n'
            "[limits]\nmax_dev_attempts = 1\nteardown_grace_s = 10\n"
        ),
    )
    detached_pid: int | None = None
    try:
        proc = _run(root, "run", timeout=120)
        assert proc.returncode == 0, proc.stderr or proc.stdout

        run_id = _run_id(root)
        run_dir = root / ".bmad-loop" / "runs" / run_id

        # (1) clean end: the story landed done and merged, not a timeout/stall
        assert _sprint_status(root, story) == "done"
        journal = [
            json.loads(ln)
            for ln in (run_dir / "journal.jsonl").read_text(encoding="utf-8").splitlines()
            if ln.strip()
        ]
        kinds = [j["kind"] for j in journal]
        assert "unit-merged" in kinds, kinds
        # (2) the #139 failure signature is ABSENT — the worktree teardown was clean
        assert "worktree-teardown-degraded" not in kinds, kinds

        # (3) no unit worktree survives (git sees only the main checkout)
        wt = subprocess.run(
            ["git", "-C", str(root), "worktree", "list", "--porcelain"],
            capture_output=True,
            text=True,
        )
        mounts = [ln for ln in wt.stdout.splitlines() if ln.startswith("worktree ")]
        assert len(mounts) == 1, wt.stdout  # only the primary checkout remains

        # (4) the detached straggler was reaped within the grace: kill-0 -> gone.
        pid_files = list((run_dir / "tasks").glob("*/fake-child.pid"))
        assert pid_files, "fake CLI never recorded its setsid child"
        detached_pid = int(pid_files[0].read_text(encoding="utf-8").strip())
        deadline = time.monotonic() + 10
        while True:
            try:
                os.kill(detached_pid, 0)
            except ProcessLookupError:
                break  # reaped by the descendant sweep before teardown
            assert time.monotonic() < deadline, f"detached child {detached_pid} survived teardown"
            time.sleep(0.1)
    finally:
        # never leak the detached sleep if the test fails before the reap check
        if detached_pid is not None:
            try:
                os.kill(detached_pid, signal.SIGKILL)
            except OSError:
                pass


def test_e2e_sweep_intent_gap_patch_restore(tmp_path):
    # Scenario 8 (#75): a SWEEP deferred-work bundle hits an intent gap during its
    # dev session — the patch is saved, the tree reverted, and the run escalates.
    # `resolve --restore-patch` re-arms the bundle spec to in-review + re-stamps its
    # baseline; resume re-applies the patch and dispatches an EXPLICIT spec pointer
    # (Change A) so the bundle resumes review on the restored diff instead of
    # re-implementing. This is the only scenario that drives the sweep-specific CLI
    # resolve→resume path (SweepEngine rebuilt from sweep.json in _resume_paused_run).
    root = tmp_path / "sbx"
    _scaffold_sweep(root)
    story = "dw-fix"  # triage names the bundle "fix" → task key dw-fix
    (root / f".intent-gap-{story}").write_text("", encoding="utf-8")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "poison: intent gap")
    impl = root / "_bmad-output" / "implementation-artifacts"
    spec = impl / f"spec-{story}.md"
    patch = impl / f"attempt-{story}.patch"

    # triage → bundle dev halts on the intent gap: patch saved, tree reverted
    proc = _run(root, "sweep", "--no-prompt")
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert "status: blocked" in spec.read_text(encoding="utf-8")
    assert patch.is_file()  # the attempted change survives the revert
    assert "attempted reading" not in (root / "src.txt").read_text(encoding="utf-8")
    assert _dw_status(root, "DW-1").startswith("open")  # a blocked pass does not close it
    run_id = _run_id(root)

    # resolve latches the restore: bundle spec → in-review + baseline re-stamped
    resolve = _run(
        root, "resolve", run_id, "--no-interactive", "--no-resume", "--restore-patch", str(patch)
    )
    assert resolve.returncode == 0, resolve.stderr or resolve.stdout
    text = spec.read_text(encoding="utf-8")
    assert "status: in-review" in text  # restore routing: step-01 → step-04
    head = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    assert f"baseline_revision: {head}" in text  # F2: spec baseline re-stamped

    # resume: patch re-applied, review resumed, bundle lands done + ledger closed
    resume = _run(root, "resume", run_id)
    assert resume.returncode == 0, resume.stderr or resume.stdout
    final = spec.read_text(encoding="utf-8")
    assert "status: done" in final, final  # fake blocks loudly on a broken contract
    assert _dw_status(root, "DW-1").startswith("done")  # the bundle closed the ledger id
    src = (root / "src.txt").read_text(encoding="utf-8")
    assert src.count("attempted reading") == 1  # restored from the patch, not re-implemented
    # Change A: the re-drive dispatch pointed at the bundle spec, never the intent.md
    run_dir = root / ".bmad-loop" / "runs" / run_id
    prompts = [p.read_text(encoding="utf-8") for p in (run_dir / "tasks").glob("*/prompt.txt")]
    assert any(str(spec) in p for p in prompts)


def test_e2e_sprint_mode_regression(tmp_path):
    # Scenario 6 (audit MAJOR-2): the new folder+id-capable bmad-dev-auto skill is
    # installed, but this is a plain SPRINT-mode run. It must drive dev → verify →
    # commit and let the orchestrator advance sprint-status to done through the
    # real CLI — unaffected by the stories wiring (the adapter's
    # BMAD_LOOP_SPEC_FOLDER read-back branch, the per-session env exports, etc.).
    root = tmp_path / "sbx"
    story = "1-1-thing"
    _scaffold_sprint(root, story)
    base = _commit_count(root)

    proc = _run(root, "run")
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert _sprint_status(root, story) == "done"
    assert _commit_count(root) == base + 1
