"""Shared recording for the three ``bmad_loop.*`` entry-point scans.

The adapter registry (:mod:`~.registry`), the profile loader (:mod:`~.profile`)
and the multiplexer registry (:mod:`~.multiplexer`) each scan their own
entry-point group, and each degrades a broken third-party distribution to a
*recorded* reason rather than a crash. This module owns the one thing all three
recordings must agree on: how a failure becomes an entry in that map.

A leaf on purpose — standard library only, and **no import of a sibling
adapters module**. Those three do not import each other today, and the package's
builtins load lazily to keep that so; an edge from here into any of them would
put a cycle one refactor away.

Why the map stays keyed on the entry-point NAME. The key is what reaches
``detail["entry_point"]`` in ``bmad-loop validate --json``. That document is
schema-versioned and evolves additively (see ``documents.validate_document``,
which contracts ``check`` as the matchable identity and states outright that
``message``/``detail`` are for humans) — so widening the key to
``(distribution, name)`` would change a value consumers can already see, while
widening only the reason TEXT does not.

Why reasons accumulate instead of overwriting.
``importlib.metadata.entry_points(group=...)`` does not deduplicate across
distributions: two installed packages may both advertise ``acme`` in one group,
and the scan yields both. A plain assignment let the second failure silently
overwrite the first, so an operator fixed one package and met the other on the
next run with no sign it had ever been there. Reasons are joined with ``"; "``
— reasons already contain colons, so a colon separator would be unreadable.

Why the distribution labels each reason. The entry-point name is not the name
you ``pip uninstall``, and two packages failing the same way otherwise render as
the same sentence twice, with nothing to tell the operator there are two.

The honest limit: a same-named collision still records ONE row, whose text now
carries both reasons. The row count does not double — that is the price of
leaving the key (and therefore ``--json``) untouched, and it still puts every
reason in front of the operator.
"""

from __future__ import annotations

from typing import Any


def record_load_error(errors: dict[str, str], ep: Any, exc: BaseException) -> None:
    """Record ``exc`` against ``ep``'s name in ``errors``, appending to whatever a
    same-named entry point already recorded.

    ``ep`` is annotated ``Any`` deliberately: the callers pass a real
    ``importlib.metadata.EntryPoint`` but every test passes a hand-rolled double,
    so naming ``EntryPoint`` here would claim a contract this function does not
    require (it touches ``.name`` and, defensively, ``.dist.name``).

    The doubled ``getattr`` tolerates a double with no ``dist`` attribute at all
    as well as a real entry point whose ``dist`` is ``None``; the truthiness
    check then treats an empty distribution name as absent, the same
    normalization the scans' ``or ""`` sort keys apply."""
    dist = getattr(getattr(ep, "dist", None), "name", None)
    reason = f"{type(exc).__name__}: {exc}"
    if dist:
        reason = f"{dist}: {reason}"
    prior = errors.get(ep.name)
    errors[ep.name] = f"{prior}; {reason}" if prior else reason
