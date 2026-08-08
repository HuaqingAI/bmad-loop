"""Run-composition layer — chiefly `config_digest`, the integrity pin over the
agent-writable config that reaches HOST code execution (issue #461 point 4).

The digest's whole job is to be exact in two directions at once: it must move for
every field that reaches exec, and hold still for everything else. A digest that
under-covers lets a mid-run rewrite through the auto-sweep gate; one that
over-covers refuses every auto-sweep after a `[limits]` live-edit #189 documents
as supported. Both halves are pinned below.
"""

import dataclasses

import pytest

from bmad_loop import policy as policy_mod
from bmad_loop import runsetup
from bmad_loop.adapters.profile import ProfileError

# A profile overlay carrying the whole launch surface the digest covers. It lives
# under .bmad-loop/profiles/, inside the tree every driven session can write.
PROFILE = """\
name = "mycli"
binary = "mycli"
launch_args = ["-i"]
bypass_args = ["--yes"]
env = { FOO = "bar" }

[hooks]
dialect = "claude-settings-json"
config_path = ".mycli/settings.json"
events = { SessionStart = "SessionStart", Stop = "Stop" }
"""

POLICY = """\
[adapter]
name = "mycli"

[verify]
commands = ["ruff check .", "pytest -q"]
"""

PROFILE_REL = ".bmad-loop/profiles/mycli.toml"


@pytest.fixture
def pinned(tmp_path):
    """A project whose entire host-exec surface is expressible on disk: a policy
    naming the verify commands, and a profile overlay carrying the launch
    binary/args/env.

    Deliberately `tmp_path` rather than the `project` conftest sandbox: this is a
    pure-core unit test of `config_digest`, which reads only `.bmad-loop/`. It
    needs no git repo, no BMAD artifact dirs and no sprint board, so the sandbox's
    per-test copytree would buy nothing. The end-to-end gate behavior is tested on
    the real sandbox in tests/test_cli.py."""
    profiles = tmp_path / ".bmad-loop" / "profiles"
    profiles.mkdir(parents=True)
    (profiles / "mycli.toml").write_text(PROFILE, encoding="utf-8")
    return tmp_path


def _digest(project, policy_text=POLICY) -> str:
    return runsetup.config_digest(policy_mod.loads(policy_text), project)


def _with_adapter_key(line: str) -> str:
    """POLICY with one more key inside its EXISTING `[adapter]` table — appending a
    second `[adapter]` header is a TOML redeclaration error, not an override."""
    return POLICY.replace('name = "mycli"', f'name = "mycli"\n{line}', 1)


def _rewrite_profile(project, old, new) -> None:
    """The mid-run profile rewrite a driven session can perform."""
    assert old in PROFILE, f"fixture drift: {old!r} no longer in PROFILE"
    (project / PROFILE_REL).write_text(PROFILE.replace(old, new), encoding="utf-8")


def test_digest_is_a_stable_sha256_over_an_unchanged_tree(pinned):
    """The null case, and the one that matters most in production: an unstable
    digest would refuse every auto-sweep in the product, not just a tampered one."""
    first = _digest(pinned)
    assert len(first) == 64 and int(first, 16) >= 0  # sha256 hex
    assert _digest(pinned) == first


def test_digest_ignores_a_benign_limits_edit(pinned):
    """Why this is field-scoped rather than a whole-file hash. #189 documents
    live-editing `[limits]` under a running loop as supported; a file hash would
    turn every such edit into a refused auto-sweep."""
    assert _digest(pinned, POLICY + "\n[limits]\ncache_read_weight = 0.5\n") == _digest(pinned)


def test_digest_ignores_the_hook_config_path(pinned):
    """Deliberate exclusion: the relay is issue #461's points 1-3 and is hardened
    on its own track. Folding it in would make the auto-sweep gate refuse after an
    ordinary `bmad-loop init` re-registration, which is not an attack."""
    before = _digest(pinned)
    _rewrite_profile(pinned, 'config_path = ".mycli/settings.json"', 'config_path = ".x/s.json"')
    assert _digest(pinned) == before


def test_digest_moves_on_a_rewritten_verify_command(pinned):
    """`[verify] commands` runs with shell=True on the host (verify.py), outside
    any session's sandbox."""
    assert _digest(pinned, POLICY.replace("pytest -q", "touch pwned")) != _digest(pinned)


def test_digest_preserves_verify_command_order(pinned):
    """They run in sequence, so a reorder is a different execution — a canonical
    that sorted them would let one be silently resequenced."""
    swapped = POLICY.replace('["ruff check .", "pytest -q"]', '["pytest -q", "ruff check ."]')
    assert _digest(pinned, swapped) != _digest(pinned)


@pytest.mark.parametrize(
    "old,new",
    [
        pytest.param('binary = "mycli"', 'binary = "rogue-cli"', id="binary"),
        pytest.param('launch_args = ["-i"]', 'launch_args = ["-i", "--evil"]', id="launch_args"),
        pytest.param('bypass_args = ["--yes"]', 'bypass_args = ["--all"]', id="bypass_args"),
        pytest.param('env = { FOO = "bar" }', 'env = { FOO = "evil" }', id="env"),
        # model_flag is an argv FLAG, not a value: an overlay that turns "--model"
        # into some other option changes what the CLI does with the model string.
        pytest.param(
            'binary = "mycli"', 'model_flag = "--exec"\nbinary = "mycli"', id="model_flag"
        ),
        # prompt_template reads like prompt payload but is an argv ELEMENT:
        # interactive_argv places render_prompt(spec.prompt) in the list, and the
        # template need not reference {prompt} at all. shlex.quote bounds it to one
        # token, which is still enough for the --opt=value form.
        pytest.param(
            'binary = "mycli"',
            'prompt_template = "--mcp-config=/tmp/evil.json"\nbinary = "mycli"',
            id="prompt_template",
        ),
    ],
)
def test_digest_moves_on_any_resolved_profile_launch_field(pinned, old, new):
    """The launch surface issue #461 names, and the reason the digest RESOLVES
    profiles instead of hashing `policy_snapshot`: not one of these fields appears
    in the snapshot, so a snapshot-only compare is blind to all four. Parametrized
    so a field quietly dropped from the canonical fails on its own row."""
    before = _digest(pinned)
    _rewrite_profile(pinned, old, new)
    assert _digest(pinned) != before


def test_digest_moves_on_rewritten_adapter_extra_args(pinned):
    """`extra_args` is the field that carries `--permission-mode bypassPermissions`,
    and `GenericAdapter.interactive_argv` prefers it over `profile.bypass_args`
    whenever it is set — so hashing the profile default alone leaves the flags the
    host CLI is actually launched with unpinned. It lives in policy.toml, which
    every driven session can write."""
    assert _digest(pinned, _with_adapter_key('extra_args = ["--yolo"]')) != _digest(pinned)


@pytest.mark.parametrize("role", ["dev", "review", "triage"])
def test_digest_moves_on_rewritten_per_stage_extra_args(pinned, role):
    """Per-stage `[adapter.<role>] extra_args` overrides the base for that role
    only, so a digest reading just the base would miss a rewrite aimed at one
    stage — and the review stage is the one that runs after the dev work lands."""
    staged = POLICY + f'\n[adapter.{role}]\nextra_args = ["--yolo"]\n'
    assert _digest(pinned, staged) != _digest(pinned)


def test_digest_separates_absent_extra_args_from_an_empty_override(pinned):
    """`None` means "fall back to profile.bypass_args"; `[]` means "launch with no
    flags at all". Two different command lines, so they must not collide — a
    canonical that coerced None to [] would let one be swapped for the other."""
    inherit = _digest(pinned)  # extra_args absent entirely
    explicit_none = _digest(pinned, _with_adapter_key("extra_args = []"))
    assert inherit != explicit_none


def test_digest_ignores_the_adapter_model(pinned):
    """The documented exclusion, pinned so it stays deliberate. `model` cannot
    introduce an argv token — it only fills the value slot behind `model_flag`,
    which IS pinned above — and including it would refuse an auto-sweep after an
    operator's mid-run model change in the TUI."""
    assert _digest(pinned, _with_adapter_key('model = "some-other-model"')) == _digest(pinned)


def test_digest_moves_on_a_widened_plugin_allowlist(pinned):
    """`[plugins] enabled` gates in-process Python import (plugins/trust.py) —
    a straight path from a workspace write to code inside the orchestrator."""
    assert _digest(pinned, POLICY + '\n[plugins]\nenabled = ["rogue"]\n') != _digest(pinned)


def test_digest_is_insensitive_to_plugin_allowlist_order(pinned):
    """`enabled` is a trust SET; listing the same two names the other way round is
    not a config change and must not refuse an auto-sweep."""
    both = POLICY + '\n[plugins]\nenabled = ["alpha", "beta"]\n'
    reordered = POLICY + '\n[plugins]\nenabled = ["beta", "alpha"]\n'
    assert _digest(pinned, both) == _digest(pinned, reordered)


def test_digest_is_invariant_to_tuple_vs_list_shapes(pinned):
    """The false-"changed" trap `cli._resume_paused_run`'s policy compare already
    documents: these fields are TUPLES on a live Policy and lists on anything that
    round-tripped through JSON. The canonical normalizes both to lists before
    hashing, so a digest can never move for shape alone — which would refuse every
    auto-sweep while looking exactly like a real tamper."""
    pol = policy_mod.loads(POLICY + '\n[plugins]\nenabled = ["alpha", "beta"]\n')
    assert isinstance(pol.verify.commands, tuple) and isinstance(pol.plugins.enabled, tuple)
    listy = dataclasses.replace(
        pol,
        verify=dataclasses.replace(pol.verify, commands=list(pol.verify.commands)),
        plugins=dataclasses.replace(pol.plugins, enabled=list(pol.plugins.enabled)),
    )

    assert runsetup.config_digest(listy, pinned) == runsetup.config_digest(pol, pinned)


def test_digest_covers_every_adapter_role(pinned):
    """A per-stage override points a role at a different profile, so hashing only
    the base adapter would leave the review and triage launch surfaces unpinned."""
    before = _digest(pinned)
    (pinned / ".bmad-loop" / "profiles" / "other.toml").write_text(
        PROFILE.replace('name = "mycli"', 'name = "other"').replace(
            'binary = "mycli"', 'binary = "other"'
        ),
        encoding="utf-8",
    )
    for role in runsetup.ROLES:
        assert _digest(pinned, POLICY + f'\n[adapter.{role}]\nname = "other"\n') != before


def test_digest_raises_on_an_unresolvable_profile(pinned):
    """ProfileError propagates by design: an unknown `[adapter] name` already
    aborts the run at make_adapters, so the digest must not paper over one by
    hashing a hole where the launch surface should be."""
    with pytest.raises(ProfileError):
        _digest(pinned, '[adapter]\nname = "no-such-cli"\n')
