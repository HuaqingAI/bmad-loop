"""The machine-readable output contract for CLI ``--json`` modes.

``--json`` in its pure-document form means exactly one JSON object on stdout
and nothing else — no trailers, no fenced blocks, no log lines. Every document
carries an inline integer ``schema_version``; each command owns its own version
constant (e.g. ``cli.STATUS_SCHEMA_VERSION``) so the documents evolve
independently, the same convention as ``diagnostics.SCHEMA_VERSION``. Evolution
is additive-only: new fields may appear, but anything breaking — removing or
renaming a field, changing a type or the meaning of a value — bumps that
command's version.

Errors never produce a partial or error document: the message goes to stderr,
stdout stays empty, and the exit code is nonzero. Consumers may rely on
"stdout is either one complete valid document or empty".

An **error** is a command that could not do its job — no runs to dump, an
unresolvable run ref, a policy that will not parse. A command whose job is to
report a *verdict* is a different thing: it did its job, and it exits nonzero to
carry the answer. ``validate --json`` is the case — a failing check is the
finding it was asked for, and the document is still owed. So the rule for
consumers is positive rather than inferred from the exit code: **parse non-empty
stdout whatever the exit code, and take the verdict from the document's own
field** (``ok`` on ``validate``). That field, unlike rc, separates "the checks
failed" from "the command broke": rc 1 is both, ``ok: false`` is only the first,
and a command that broke leaves nothing on stdout to read the field from. Note
that every gate in ``cmd_validate`` runs inside a ``try``, so the command has no
error path of its own — its rc ∈ {0, 1} is purely the verdict.

**Success does not imply a document on stdout.** ``diagnose`` and
``probe-adapter`` also take ``--out FILE``; with both flags the document goes to
the file and stdout is legitimately empty at exit 0, with only a confirmation on
stderr. So empty stdout means "no document *here*" — check the exit code to tell
a redirect from a failure, and do not treat rc 0 as a promise of bytes to parse.
That file is held to the same standard as the stream: :func:`write_document` is
the only way it is written, and validates exactly as :func:`emit_document` does.

Every command that takes ``--json`` shares this contract; there is no exception
(#195 removed the last two, ``diagnose`` and ``probe-adapter``, which used to
append a fenced JSON block to a markdown/text report). A command adopting the
flag adopts the whole of it — the contract is the flag's meaning, not a style
the earlier commands happen to follow.

Two ways in, by what the caller already holds:

- :func:`emit` takes a ``dict`` and serializes it — ``status`` and ``list``.
- :func:`emit_document` takes an already-serialized ``str`` — ``diagnose`` and
  ``probe-adapter``, whose renderers return text. For both this is a hard
  constraint: each serializes *before* running the shared leak self-check
  (``sanitize.guard``), making those exact bytes the ones the check verified —
  re-encoding them here would emit bytes nothing verified.

:func:`write_document` is the ``--out`` sibling of :func:`emit_document`, taking
the same already-serialized string to a file instead of stdout.

The serializer flags differ by family on purpose and are not to be unified: the
renderers behind :func:`emit_document` sort their keys (a diff-stable dump is
worth more than field order there) while :func:`emit`'s dicts are built in the
order they are meant to be read, and the two guarded renderers pass
``ensure_ascii=False`` because the leak guard has to scan the values unescaped.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def emit(doc: dict[str, object]) -> None:
    """The single stdout write in JSON mode."""
    emit_document(json.dumps(doc, indent=2))


def _validated(rendered: str) -> str:
    """Return ``rendered`` unchanged, or raise if it is not a whole JSON document.

    Parse to assert, never to re-serialize: the caller's exact bytes are what gets
    written. The ``diagnose``/``probe-adapter`` renderers run their leak self-check
    *before* handing the string over, so re-encoding here would ship bytes nothing
    checked.
    """
    try:
        json.loads(rendered)
    except json.JSONDecodeError as e:
        raise ValueError(f"refusing to emit a malformed JSON document: {e}") from e
    return rendered


def emit_document(rendered: str) -> None:
    """The single stdout write, for a document that is *already* serialized.

    Verifies well-formedness before writing, then writes the ORIGINAL string —
    never a re-serialization of the parsed result. Half the commands reach stdout
    through here rather than through :func:`emit`, so without this the contract
    would hold for them by convention only; but the guarded renderers validated
    these exact bytes with their leak self-check, so emitting anything re-derived
    would ship bytes nothing checked. Parse to assert, print verbatim.

    Raises :class:`ValueError` on a malformed document — a bug in the caller's
    renderer, and far better surfaced as a crash with empty stdout (which the
    contract permits) than as a half-parsable stream a consumer has to diagnose.

    stdout is switched to UTF-8 first, because a document is not necessarily
    ASCII — the guarded renderers dump with ``ensure_ascii=False`` so the
    leak guard can scan the values unescaped, which lets a non-sensitive
    non-ASCII field (a localized ``platform.release()``, say) through to here
    verbatim. Encoding it for a legacy non-UTF-8 console then raised
    :class:`UnicodeEncodeError` before a byte was written (#200). Escaping the
    output instead would have been the smaller change and the wrong one: it
    breaks the invariant this function exists for, since the guard verified the
    unescaped bytes. So the stream is made able to carry the document rather
    than the document cut down to fit the stream.
    """
    document = _validated(rendered)
    # Guarded: a substituted stdout (pytest capture, an exotic stream) may not be
    # a TextIOWrapper at all. Falling through leaves the pre-#200 behaviour, which
    # is a crash with stdout still empty — permitted by the contract above.
    if hasattr(sys.stdout, "reconfigure"):
        # hasattr-guarded: reconfigure exists on TextIOWrapper, not the TextIO base.
        sys.stdout.reconfigure(  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]
            encoding="utf-8"
        )
    print(document)


def write_document(path: Path, rendered: str) -> None:
    """The single file write in JSON mode — the bytes :func:`emit_document` prints.

    ``--out FILE`` is the other half of the same contract, so it gets the same
    validation and the same trailing newline: ``--json --out FILE`` and
    ``--json > FILE`` produce byte-identical files. Without the parse the file was
    the weaker half — stdout refused a malformed document while the file accepted
    it, which is backwards, since a document written to a file is the one nobody
    eyeballs before feeding it to a parser.

    Raises :class:`ValueError` on a malformed document, before the file is created.
    """
    path.write_text(_validated(rendered) + "\n", encoding="utf-8")


def add_json_flag(parser: argparse.ArgumentParser, what: str) -> None:
    """Register ``--json`` with the standard help text."""
    parser.add_argument(
        "--json",
        action="store_true",
        help=f"emit a stable machine-readable JSON document ({what}) instead of text",
    )
