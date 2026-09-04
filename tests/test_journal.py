"""State persistence: the atomic write must survive the transient Windows
sharing violation (WinError 5) a concurrent TUI reader triggers. The retry
lives in platform_util.atomic_replace (unit-tested there); this proves
save_state still rides it end to end."""

from __future__ import annotations

import json
import os
import stat
import threading
from contextlib import contextmanager
from pathlib import Path

import pytest

from bmad_loop import journal as journal_mod
from bmad_loop import platform_util, runs
from bmad_loop.journal import Journal, load_state, save_state, state_lock
from bmad_loop.model import RunState


def test_save_state_retries_transient_sharing_violation(tmp_path, monkeypatch):
    """On win32, os.replace denied by a concurrent reader is retried, not fatal."""
    monkeypatch.setattr(platform_util.sys, "platform", "win32")
    monkeypatch.setattr(platform_util.time, "sleep", lambda _s: None)  # no real backoff
    monkeypatch.setattr(
        journal_mod, "file_lock", contextmanager(lambda _path, **_kw: iter((None,)))
    )

    real_replace = os.replace
    calls = {"n": 0}

    def flaky_replace(src, dst):
        calls["n"] += 1
        if calls["n"] < 3:  # first two collide, third lands
            raise PermissionError(5, "Access is denied")
        real_replace(src, dst)

    monkeypatch.setattr(platform_util.os, "replace", flaky_replace)

    save_state(tmp_path, RunState(run_id="r1", project="p", started_at="2026-07-06T21:00:00"))

    assert calls["n"] == 3
    assert load_state(tmp_path).run_id == "r1"


def test_state_lock_holds_the_canonical_run_sidecar(tmp_path):
    run_dir = tmp_path / "run"
    lock_path = runs.lock_path_for(run_dir / journal_mod.STATE_FILE, follow_final_symlink=False)

    with state_lock(run_dir):
        with pytest.raises(OSError):
            with platform_util.file_lock(lock_path, blocking=False):
                pytest.fail("a rival acquired the held run-state sidecar")


def test_state_lock_same_run_nesting_acquires_os_lock_once(tmp_path, monkeypatch):
    acquired: list[object] = []

    @contextmanager
    def recording_lock(path, **_kw):
        acquired.append(path)
        yield

    monkeypatch.setattr(journal_mod, "file_lock", recording_lock)

    with state_lock(tmp_path):
        with state_lock(tmp_path / "."):
            save_state(
                tmp_path,
                RunState(run_id="r1", project="p", started_at="2026-09-01T00:00:00"),
            )

    assert acquired == [
        runs.lock_path_for(tmp_path / journal_mod.STATE_FILE, follow_final_symlink=False)
    ]


def test_state_lock_same_run_symlink_spellings_acquire_os_lock_once(tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    alias = tmp_path / "run-alias"
    try:
        alias.symlink_to(run_dir, target_is_directory=True)
    except (NotImplementedError, OSError) as e:
        pytest.skip(f"directory symlinks unavailable: {e}")
    acquired: list[object] = []

    @contextmanager
    def recording_lock(path, **_kw):
        acquired.append(path)
        yield

    monkeypatch.setattr(journal_mod, "file_lock", recording_lock)

    with state_lock(run_dir):
        with state_lock(alias):
            pass

    assert acquired == [
        runs.lock_path_for(run_dir / journal_mod.STATE_FILE, follow_final_symlink=False)
    ]


def test_state_lock_identity_survives_replacing_a_final_state_symlink(tmp_path):
    """Ablation: follow the final state.json symlink in state_lock and nested
    save_state changes sidecars when atomic_replace replaces the link."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    elsewhere = tmp_path / "elsewhere.json"
    elsewhere.write_text("{}", encoding="utf-8")
    state_path = run_dir / journal_mod.STATE_FILE
    state_path.symlink_to(elsewhere)
    logical_lock = runs.lock_path_for(state_path, follow_final_symlink=False)

    # The default remains referent-based for ledgers and every other caller.
    assert runs.lock_path_for(state_path) == runs.lock_path_for(elsewhere)
    assert runs.lock_path_for(state_path) != logical_lock

    with state_lock(run_dir):
        save_state(
            run_dir,
            RunState(run_id="r1", project="p", started_at="2026-09-01T00:00:00"),
        )
        assert not state_path.is_symlink()
        with state_lock(run_dir / "."):
            with pytest.raises(OSError):
                with platform_util.file_lock(logical_lock, blocking=False):
                    pytest.fail("a rival acquired the original logical sidecar")

    assert elsewhere.read_text(encoding="utf-8") == "{}"
    assert load_state(run_dir).run_id == "r1"


def test_state_lock_refuses_different_run_nesting_before_second_acquire(tmp_path, monkeypatch):
    acquired: list[object] = []

    @contextmanager
    def recording_lock(path, **_kw):
        acquired.append(path)
        yield

    monkeypatch.setattr(journal_mod, "file_lock", recording_lock)

    with state_lock(tmp_path / "one"):
        with pytest.raises(RuntimeError, match="different runs"):
            with state_lock(tmp_path / "two"):
                pytest.fail("cross-run nesting was allowed")

    assert len(acquired) == 1


def test_state_lock_failure_clears_thread_guard(tmp_path, monkeypatch):
    acquired: list[object] = []

    @contextmanager
    def recording_lock(path, **_kw):
        acquired.append(path)
        yield

    monkeypatch.setattr(journal_mod, "file_lock", recording_lock)

    with pytest.raises(ValueError, match="boom"):
        with state_lock(tmp_path / "one"):
            raise ValueError("boom")
    with state_lock(tmp_path / "two"):
        pass

    assert len(acquired) == 2


def test_save_state_acquisition_error_writes_nothing(tmp_path, monkeypatch):
    run_dir = tmp_path / "run"

    @contextmanager
    def refusing_lock(_path, **_kw):
        raise OSError("lock unavailable")
        yield

    monkeypatch.setattr(journal_mod, "file_lock", refusing_lock)

    with pytest.raises(OSError, match="lock unavailable"):
        save_state(
            run_dir,
            RunState(run_id="r1", project="p", started_at="2026-09-01T00:00:00"),
        )

    assert not run_dir.exists()


def test_save_state_root_error_writes_nothing(tmp_path, monkeypatch):
    run_dir = tmp_path / "run"

    def no_state_root(_path, **_kwargs):
        raise runs.StateRootError("no state root")

    monkeypatch.setattr(runs, "lock_path_for", no_state_root)

    with pytest.raises(runs.StateRootError, match="no state root"):
        save_state(
            run_dir,
            RunState(run_id="r1", project="p", started_at="2026-09-01T00:00:00"),
        )

    assert not run_dir.exists()


def test_two_concurrent_saves_never_share_the_fixed_temp_file(tmp_path, monkeypatch):
    """Ablation: remove save_state's state_lock and the second replace enters while
    the first is paused, so both calls race on state.json.tmp and one loses it."""
    real_replace = journal_mod.atomic_replace
    real_file_lock = journal_mod.file_lock
    first_entered = threading.Event()
    second_attempted = threading.Event()
    release_first = threading.Event()
    replace_threads: list[str] = []

    @contextmanager
    def observed_file_lock(path, **_kw):
        if threading.current_thread().name == "second":
            second_attempted.set()
        with real_file_lock(path, **_kw):
            yield

    def controlled_replace(src, dst):
        replace_threads.append(threading.current_thread().name)
        if len(replace_threads) == 1:
            first_entered.set()
            assert release_first.wait(2)
        real_replace(src, dst)

    monkeypatch.setattr(journal_mod, "atomic_replace", controlled_replace)
    monkeypatch.setattr(journal_mod, "file_lock", observed_file_lock)
    errors: list[BaseException] = []

    def writer(run_id: str) -> None:
        try:
            save_state(
                tmp_path,
                RunState(run_id=run_id, project="p", started_at="2026-09-01T00:00:00"),
            )
        except BaseException as e:
            errors.append(e)

    first = threading.Thread(target=writer, args=("first",), name="first")
    second = threading.Thread(target=writer, args=("second",), name="second")
    first.start()
    assert first_entered.wait(2)
    second.start()
    assert second_attempted.wait(2)
    assert replace_threads == ["first"]
    release_first.set()
    first.join(2)
    second.join(2)

    assert errors == []
    assert sorted(replace_threads) == ["first", "second"]
    assert load_state(tmp_path).run_id in {"first", "second"}


def _planted_verify_symlink(tmp_path):
    """A run dir whose `verify/` a session has already replaced with a link out."""
    run_dir, elsewhere = tmp_path / "run", tmp_path / "elsewhere"
    run_dir.mkdir()
    elsewhere.mkdir()
    (run_dir / "verify").symlink_to(elsewhere, target_is_directory=True)
    return Journal(run_dir), elsewhere


@pytest.mark.skipif(not journal_mod.DIR_FD_ANCHORED_WRITES, reason="dir-fd anchoring is POSIX-only")
def test_write_verify_stream_refuses_a_symlinked_verify_directory(tmp_path):
    """A session that plants `verify/` as a link cannot redirect verifier output.

    Sessions are handed the run directory (`BMAD_LOOP_RUN_DIR`) and write their
    own result.json into it, so this is a writer that really can plant the link.
    `mkdir(parents=True, exist_ok=True)` ACCEPTS a symlink-to-directory — it
    re-raises only when `is_dir()` is false, and that follows links — and
    `follow_symlinks=False` covers the final component, never its parent. Without
    the confinement walk the write lands in `elsewhere/`, outside the run dir.

    The refusal is an OSError because that is the caller's existing degrade path:
    the journal record still lands, with a null pointer and `capture_error`.

    Ablation, measured, and the two guards OVERLAP — which is the part worth
    writing down. Dropping the `open_dir_confined` arm alone reddens this test on
    the *message* only, because the win32 `is_symlink()` fallback below still
    refuses; so that ablation proves the arm is reached, not that it prevents the
    escape. Removing BOTH guards is what proves the harm: each test then fails
    `DID NOT RAISE`, and the same planted link writes `v.stdout.log` into
    `elsewhere/` while `write_verify_stream` returns the pointer
    `verify/v.stdout.log` — the file is outside the run dir and the record claims
    it is inside.
    """
    journal, elsewhere = _planted_verify_symlink(tmp_path)

    with pytest.raises(OSError, match=r"unconfined verify directory"):
        journal.write_verify_stream("v.stdout.log", "verifier output")

    # the assertion that actually pins the fix: nothing escaped the run dir
    assert list(elsewhere.iterdir()) == []


@pytest.mark.skipif(not journal_mod.DIR_FD_ANCHORED_WRITES, reason="dir-fd anchoring is POSIX-only")
def test_write_verify_stream_refuses_a_symlinked_verify_directory_on_the_win32_path(
    tmp_path, monkeypatch
):
    """win32 has no *at() family, so it keeps a check-then-write — which must
    still refuse the planted link rather than fall through to the write.

    Ablation: delete the `is_link_like(verify_dir)` guard and this fails
    `DID NOT RAISE`, with the file landing in `elsewhere/` exactly as the
    unguarded POSIX path did.
    """
    monkeypatch.setattr(journal_mod, "DIR_FD_ANCHORED_WRITES", False)
    journal, elsewhere = _planted_verify_symlink(tmp_path)

    with pytest.raises(OSError, match=r"redirected verify directory"):
        journal.write_verify_stream("v.stdout.log", "verifier output")

    assert list(elsewhere.iterdir()) == []


def test_write_verify_stream_writes_an_ordinary_verify_directory(tmp_path):
    """The positive control: an unplanted run dir still retains its streams.

    Without this, both refusal tests above pass for a `write_verify_stream` that
    refuses everything unconditionally — a negative assertion is green for every
    reason a file could be absent.
    """
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    journal = Journal(run_dir)

    pointer = journal.write_verify_stream("v.stdout.log", "verifier output")

    assert pointer == "verify/v.stdout.log"
    assert (run_dir / pointer).read_text(encoding="utf-8") == "verifier output"


class _ReparseStat:
    """os.lstat() of a Windows junction: a DIRECTORY mode — which is why
    Path.is_symlink() answers False — carrying a reparse tag."""

    st_mode = stat.S_IFDIR | 0o755
    st_reparse_tag = 0xA0000003  # IO_REPARSE_TAG_MOUNT_POINT


def test_write_verify_stream_refuses_a_junctioned_verify_directory(tmp_path, monkeypatch):
    """The win32 fallback must refuse a DIRECTORY JUNCTION, not just a symlink.

    `mklink /J` needs no elevation, while a directory symlink needs
    SeCreateSymbolicLinkPrivilege or Developer Mode — so on Windows the junction
    is the unprivileged half of the same escape, and `Path.is_symlink()` reports
    False for it. A guard written as `is_symlink()` would leave that half open
    with no race to win. Windows-only in reality; the logic is driven here so it
    does not ship unexercised.

    Ablation: point the guard back at `verify_dir.is_symlink()` and this fails
    `DID NOT RAISE` — verified.
    """
    monkeypatch.setattr(journal_mod, "DIR_FD_ANCHORED_WRITES", False)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    verify_dir = run_dir / "verify"
    verify_dir.mkdir()  # a real directory: is_symlink() is False, as for a junction

    # Patch the TAG TUPLE in platform_util, not `is_link_like` itself: journal.py
    # bound the function by value at import, so replacing the name there would not
    # reach this call — but the predicate reads `_LINK_REPARSE_TAGS` from its own
    # module globals on every call, so this does.
    real_lstat = os.lstat
    monkeypatch.setattr(platform_util, "_LINK_REPARSE_TAGS", (_ReparseStat.st_reparse_tag,))
    monkeypatch.setattr(
        os,
        "lstat",
        lambda p, *a, **k: _ReparseStat() if str(p) == str(verify_dir) else real_lstat(p),
    )

    with pytest.raises(OSError, match=r"redirected verify directory"):
        Journal(run_dir).write_verify_stream("v.stdout.log", "verifier output")


# ------------------------------------------------- append tail heal (DW-97)


def _journal_path(run_dir):
    return run_dir / "journal.jsonl"


def test_append_leaves_a_terminated_tail_alone(tmp_path):
    """The common case pays nothing: a journal ending in a newline gains exactly
    one line and no blank one.

    Ablation: heal unconditionally (drop the `_tail_is_terminated` guard) and this
    reddens on the blank line between the two records."""
    journal = Journal(tmp_path)
    journal.append("run-start")
    journal.append("session-start", task_id="t0")

    raw = _journal_path(tmp_path).read_text(encoding="utf-8")
    assert "\n\n" not in raw
    assert [e["kind"] for e in journal.entries()] == ["run-start", "session-start"]


def test_append_to_an_absent_or_empty_journal_writes_no_leading_newline(tmp_path):
    """A missing file and a zero-length one are both "terminated": there is no
    fragment to close, so neither may gain a leading blank line."""
    journal = Journal(tmp_path)
    assert not _journal_path(tmp_path).exists()
    journal.append("run-start")
    assert _journal_path(tmp_path).read_text(encoding="utf-8").startswith('{"ts"')

    other = tmp_path / "other"
    other.mkdir()
    _journal_path(other).write_text("", encoding="utf-8")
    Journal(other).append("run-start")
    assert _journal_path(other).read_text(encoding="utf-8").startswith('{"ts"')


def test_append_heals_an_unterminated_tail(tmp_path):
    """A partially flushed record ends the file mid-line. The next append must
    terminate that fragment on its OWN line rather than concatenating onto it, so
    the new record stays parseable."""
    _journal_path(tmp_path).write_text('{"ts": 1, "kind": "unit-merge-star', encoding="utf-8")
    journal = Journal(tmp_path)
    journal.append("unit-merged", unit="u1")

    lines = _journal_path(tmp_path).read_text(encoding="utf-8").splitlines()
    assert lines[0] == '{"ts": 1, "kind": "unit-merge-star'
    assert json.loads(lines[1])["kind"] == "unit-merged"
    kinds = [e["kind"] for e in journal.entries()]
    assert kinds == [journal_mod.UNREADABLE_LINE_KIND, "unit-merged"]


def test_one_partial_flush_costs_one_record_not_two(tmp_path):
    """The regression this defect is about: WITHOUT the heal the first append
    concatenates onto the fragment and both are dropped as one unparseable line, so
    a single fault costs TWO records — and a swallowed `unit-merged` re-drives
    already-merged work in `engine._replay_unlatched_ledger_carries`.

    Ablation: delete the `if not self._tail_is_terminated()` prepend in
    `Journal.append` and this reddens — `unit-merged` is missing from the kinds and
    only two entries come back (verified)."""
    _journal_path(tmp_path).write_text('{"ts": 1, "kind": "unit-merge-star', encoding="utf-8")
    journal = Journal(tmp_path)
    journal.append("unit-merged", unit="u1")
    journal.append("run-complete")

    entries = journal.entries()
    assert [e["kind"] for e in entries] == [
        journal_mod.UNREADABLE_LINE_KIND,
        "unit-merged",
        "run-complete",
    ]
    assert entries[1]["unit"] == "u1"


def test_append_heals_when_the_probe_cannot_open_an_existing_journal(tmp_path, monkeypatch):
    """An unknown tail fails TOWARD the heal. The probe's `open` can fail on a file
    that exists — a transient EACCES/EMFILE, or the Windows sharing violation
    `atomic_replace` already retries for — and answering "already terminated" there
    would skip the heal over a real fragment and reproduce the two-record loss on
    exactly the unlucky path this change exists to close. Costs at worst one blank
    line, which both readers skip.

    Only `FileNotFoundError` may answer True, and its own test above
    (`test_append_to_an_absent_or_empty_journal_writes_no_leading_newline`) is what
    keeps that arm honest — otherwise every fresh journal would open with a blank
    line.

    Ablation: widen the arm back to `except OSError: return True` and this reddens —
    `unit-merged` is swallowed by the fragment (verified)."""
    _journal_path(tmp_path).write_text('{"ts": 1, "kind": "unit-merge-star', encoding="utf-8")
    journal = Journal(tmp_path)

    real_open = Path.open
    seen: list[str] = []

    def deny_the_probe(self, mode="r", *args, **kwargs):
        # Fail ONLY the "rb" probe read; the append's own "a" open must proceed, or
        # the test would prove nothing about which direction the probe answered.
        if mode == "rb" and self == journal.path:
            seen.append(mode)
            raise PermissionError(13, "denied")
        return real_open(self, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", deny_the_probe)
    journal.append("unit-merged", unit="u1")
    monkeypatch.undo()

    assert seen == ["rb"], "the probe never ran; the test would be vacuous"
    lines = _journal_path(tmp_path).read_text(encoding="utf-8").splitlines()
    assert lines[0] == '{"ts": 1, "kind": "unit-merge-star'
    assert json.loads(lines[1])["kind"] == "unit-merged"
    assert [e["kind"] for e in journal.entries()] == [
        journal_mod.UNREADABLE_LINE_KIND,
        "unit-merged",
    ]


def test_append_heals_only_once_per_fragment(tmp_path):
    """Two appends after one fragment leave one blank-free join: the second append
    sees a terminated tail and adds nothing.

    The blank-line count alone would stay green if the second append wrote NOTHING,
    so the kinds are asserted too — the claim is "adds no blank line", not "adds
    nothing"."""
    _journal_path(tmp_path).write_text("frag", encoding="utf-8")
    journal = Journal(tmp_path)
    journal.append("a")
    journal.append("b")
    assert _journal_path(tmp_path).read_text(encoding="utf-8").count("\n\n") == 0
    assert [e["kind"] for e in journal.entries()] == [
        journal_mod.UNREADABLE_LINE_KIND,
        "a",
        "b",
    ]


def test_append_preserves_a_complete_record_left_without_a_final_newline(tmp_path):
    """An unterminated tail is not always a TORN record: a whole record whose final
    newline never landed is complete JSON, and the heal must give it its own line so
    it still parses. Without the prepend the next record concatenates onto it and BOTH
    are lost — the same two-record fault, with the first record intact on disk.

    Ablation: drop the `if not self._tail_is_terminated()` prepend and this reddens —
    only the marker comes back (verified)."""
    _journal_path(tmp_path).write_text('{"ts": 1, "kind": "unit-merged"}', encoding="utf-8")
    journal = Journal(tmp_path)
    journal.append("run-complete")

    assert [e["kind"] for e in journal.entries()] == ["unit-merged", "run-complete"]


def test_append_leaves_an_existing_crlf_tail_alone(tmp_path):
    r"""A journal written through Windows text mode ends `\r\n`, whose LAST byte is
    still `\n` — so the probe reads it as terminated and no blank line is added, and
    the CRLF record still parses (`entries()` strips the `\r`).

    This is the CRLF half of a two-file pin; `test_append_leaves_a_terminated_tail_alone`
    is the LF half. The mutation only THIS half catches is a probe that reads a `\r\n`
    tail as a foreign writer's torn record and heals it —
    `tail.endswith(b"\n") and not tail.endswith(b"\r\n")` — which leaves the LF sibling
    green and reddens the blank-line assertion here (verified)."""
    _journal_path(tmp_path).write_bytes(b'{"ts": 1, "kind": "run-start"}\r\n')
    journal = Journal(tmp_path)
    journal.append("session-start", task_id="t0")

    # read_text normalizes `\r\n` to `\n`, so this catches a wrongly-healed blank line
    # in either spelling — including the `\r\n` one Windows text mode would write it as.
    assert _journal_path(tmp_path).read_text(encoding="utf-8").count("\n\n") == 0
    assert [e["kind"] for e in journal.entries()] == ["run-start", "session-start"]


def test_rearm_journal_subclass_inherits_the_heal(tmp_path):
    """`runs._RearmJournal.append` forwards to `super().append`, so the heal is not
    something a subclass has to remember."""
    _journal_path(tmp_path).write_text('{"kind": "frag', encoding="utf-8")
    runs._RearmJournal(tmp_path).append("rearm-ok", story_key="1-1")
    assert [e["kind"] for e in Journal(tmp_path).entries()] == [
        journal_mod.UNREADABLE_LINE_KIND,
        "rearm-ok",
    ]


# ------------------------------------------- entries() unreadable-line marker


def test_entries_reports_an_unreadable_line_in_its_stream_position(tmp_path):
    """The marker takes the lost record's SLOT, so its position still carries the
    ordering information the entry itself would have.

    The torn line is deliberately PADDED with leading and trailing whitespace, so the
    `bytes` count can distinguish the raw line (12) from the stripped spelling the
    parse was attempted on (8). With an unpadded fixture both spellings give the same
    number and the choice is untestable.

    Ablation: restore `except json.JSONDecodeError: continue` and this reddens with
    the marker absent; count `len(line.encode(...))` (the stripped spelling) instead
    of the raw line and the `bytes` assertion reddens 8 != 12."""
    torn = "  not json  "
    _journal_path(tmp_path).write_text(
        f'{torn}\n{{"ts": 1, "kind": "run-start"}}\n', encoding="utf-8"
    )
    entries = Journal(tmp_path).entries()
    assert entries == [
        {"kind": journal_mod.UNREADABLE_LINE_KIND, "bytes": 12},
        {"ts": 1, "kind": "run-start"},
    ]
    assert len(torn) == 12 and len(torn.strip()) == 8  # the two spellings differ


def test_entries_marker_carries_no_ts_and_no_line_content(tmp_path):
    """`diagnostics.summarize_journal` derives first_ts/last_ts/duration_s from
    entry timestamps, so a fabricated `ts` would corrupt them; and a journal line
    can carry session text, so only a byte count is reported."""
    secret = '{"kind": "dev-decision", "note": "swordfish"'
    _journal_path(tmp_path).write_text(secret + "\n", encoding="utf-8")
    (marker,) = Journal(tmp_path).entries()
    assert marker == {"kind": journal_mod.UNREADABLE_LINE_KIND, "bytes": len(secret)}
    assert "swordfish" not in json.dumps(marker)


def test_entries_still_skips_blank_lines(tmp_path):
    """A blank line lost no record — and the heal itself can introduce one when a
    rival appender terminated the tail first — so blanks stay silent."""
    _journal_path(tmp_path).write_text('{"kind": "a"}\n\n\n{"kind": "b"}\n   \n', encoding="utf-8")
    assert [e["kind"] for e in Journal(tmp_path).entries()] == ["a", "b"]


def test_entries_still_passes_through_a_non_mapping_line(tmp_path):
    """A bare `3` PARSES, so it is not an unreadable line: it survives unchanged, as
    today, and `runs.journal_entries_or_none` is where non-dicts are filtered."""
    _journal_path(tmp_path).write_text('3\n{"kind": "a"}\n', encoding="utf-8")
    assert Journal(tmp_path).entries() == [3, {"kind": "a"}]


def test_entries_still_propagates_invalid_utf8(tmp_path):
    """Only `JSONDecodeError` becomes a marker. `runs.journal_entries_or_none`
    catches `UnicodeDecodeError` deliberately to return None (a journal it cannot
    read) rather than an empty list, and widening the marker to cover it would take
    that distinction away.

    Ablation: catch `UnicodeDecodeError` in `entries()` too and this fails
    `DID NOT RAISE`."""
    _journal_path(tmp_path).write_bytes(b'{"kind": "a"}\n\xff\xfe\n')
    with pytest.raises(UnicodeDecodeError):
        Journal(tmp_path).entries()
    assert runs.journal_entries_or_none(tmp_path) is None


def test_marker_survives_the_journal_entries_or_none_dict_filter(tmp_path):
    """The marker must be a plain dict: `journal_entries_or_none` drops non-dicts,
    and both its callers DIFF two reads on `len(before)` — a marker that vanished
    from one read and not the other would move the re-arm watermark."""
    _journal_path(tmp_path).write_text('bad\n{"kind": "a"}\n', encoding="utf-8")
    before = runs.journal_entries_or_none(tmp_path)
    after = runs.journal_entries_or_none(tmp_path)
    assert before is not None and after is not None
    assert len(before) == 2 and before == after
    assert before[0]["kind"] == journal_mod.UNREADABLE_LINE_KIND
