"""Contract tests for the sandbox fixtures in `tests/conftest.py`.

`project` hands every test a copytree clone of a session-scoped template repo, so
the template's shape is a shared dependency of most of the suite. What is pinned
here is the part of that shape other modules rely on without asserting it.
"""

from __future__ import annotations


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
