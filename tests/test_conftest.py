"""Contract tests for the sandbox fixtures in `tests/conftest.py`.

`project` hands every test a copytree clone of a session-scoped template repo, so
the template's shape is a shared dependency of most of the suite. What is pinned
here is the part of that shape other modules rely on without asserting it.
"""

from __future__ import annotations

from conftest import make_git_noisy

from bmad_loop import verify


def test_template_drops_sample_hooks_but_keeps_hooks_dir_and_exclude(project):
    """`git init` seeds 14 dead `*.sample` hooks that nothing reads; the template
    deletes the files so each per-test copy stops replicating them.

    Both halves are load-bearing, and they pull against each other: the obvious
    shortcut for the first (`git init --template=` pointed at an empty dir) also
    removes `.git/hooks/` itself and `.git/info/exclude`, which the suite does
    depend on — three sites write `.git/hooks/pre-commit` with no `mkdir`
    (tests/test_engine.py twice, tests/test_verify.py once) and
    tests/test_install.py reads `.git/info/exclude`. Deleting the sample files
    is therefore the only cleanup that satisfies both.

    Ablation target: delete the `sample.unlink()` loop from `_project_template`
    and the first assertion fails alone; swap that loop for an empty
    `git init --template=` and the first assertion passes while the two
    survival assertions fail instead — disjoint failures, which is what proves
    the two halves are independent rather than one implying the other."""
    git_dir = project.project / ".git"
    hooks = git_dir / "hooks"

    assert list(hooks.glob("*.sample")) == []
    assert hooks.is_dir()
    assert (git_dir / "info" / "exclude").is_file()


def test_make_git_noisy_produces_rc_zero_stderr(project):
    """The anti-vacuity guard for the suite's only host-noise dimension (#442).

    `make_git_noisy` is what makes the merged and the stdout-alone reads
    distinguishable; its whole premise is that an unknown VALUE for a known config
    KEY is a `warning:` on stderr at rc 0, not an error. If a future git ever stops
    emitting it, THIS test fails loudly — instead of the four tests that depend on
    the helper all passing for the wrong reason, with the bug restored. Do not
    delete it as a duplicate of them: it is the only row that would notice.

    Ablation target: delete the `git config` line from `make_git_noisy` and the
    premise is dead — the helper's own probe catches it, so this row and the four
    that depend on the helper all report SKIPPED, none PASSED. Delete the probe as
    well, so nothing masks the dead premise, and this row FAILS on the stderr
    assertion. Deleting the probe ALONE reddens nothing while the host git still
    warns, and that is the point rather than a gap: the probe is what turns a future
    silent git into four skips instead of four false greens, and this row into the
    one loud failure."""
    repo = project.project
    make_git_noisy(repo)

    proc = verify._run_git(["git", "-C", str(repo), "rev-parse", "HEAD"], repo)

    assert proc.returncode == 0  # a warning, not a failure
    assert proc.stderr.strip()  # git really did write to stderr
    sha = proc.stdout.strip()
    assert len(sha) == 40 and all(c in "0123456789abcdef" for c in sha)
