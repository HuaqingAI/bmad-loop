"""Publication authority and byte preservation without involving receipt proof."""

import base64
import os

import pytest

from bmad_loop import artifact_publication as publication
from bmad_loop.model import StoryTask


@pytest.fixture
def publication_case(project, monkeypatch):
    source = project.rebased(project.project / "unit")
    source.implementation_artifacts.mkdir(parents=True)
    spec = source.implementation_artifacts / "spec.md"
    spec.write_text("---\nstatus: done\nartifact_deliverables: [report.bin]\n---\n")
    (source.implementation_artifacts / "report.bin").write_bytes(b"\xff\x00\r\nreport")
    task = StoryTask(story_key="dw-fix", epic=0, dw_ids=["DW-1"], spec_file=str(spec))
    monkeypatch.setattr(publication.verify, "path_ignored", lambda *_: True)
    monkeypatch.setattr(publication.verify, "path_tracked", lambda *_: False)
    publication.capture(task, project)
    return task, project, source


def test_exact_selection_and_frozen_binary_payload(publication_case):
    task, paths, source = publication_case
    (source.implementation_artifacts / "unrelated.md").write_text("unrelated")
    publication.prepare(task, paths, source)
    (source.implementation_artifacts / "report.bin").write_bytes(b"later source edit")
    publication.publish(task, paths)
    assert (paths.implementation_artifacts / "report.bin").read_bytes() == b"\xff\x00\r\nreport"
    assert (paths.implementation_artifacts / "spec.md").read_bytes() == (
        source.implementation_artifacts / "spec.md"
    ).read_bytes()
    assert not (paths.implementation_artifacts / "unrelated.md").exists()
    assert task.artifact_publication_complete


def test_late_declaration_does_not_capture_late_destination(publication_case):
    task, paths, source = publication_case
    destination = paths.implementation_artifacts / "report.bin"
    destination.write_bytes(b"operator")
    publication.prepare(task, paths, source)
    with pytest.raises(publication.PublicationError, match="conflict.*report.bin"):
        publication.publish(task, paths)
    assert destination.read_bytes() == b"operator"
    assert not task.artifact_publication_complete
    assert task.artifact_payload is not None


@pytest.mark.parametrize("relative", ["report.bin", "errata/correction.md"])
def test_existing_baseline_allows_replace(publication_case, relative):
    task, paths, source = publication_case
    destination = paths.implementation_artifacts / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(b"before")
    output = source.implementation_artifacts / relative
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(b"\xff\x00\r\nreport")
    (source.implementation_artifacts / "spec.md").write_text(
        f"---\nstatus: done\nartifact_deliverables: [{relative}]\n---\n"
    )
    publication.capture(task, paths)
    publication.prepare(task, paths, source)
    publication.publish(task, paths)
    assert destination.read_bytes() == b"\xff\x00\r\nreport"


def test_partial_write_replays_saved_intent(publication_case, monkeypatch):
    task, paths, source = publication_case
    publication.prepare(task, paths, source)
    writer = publication.atomic_write_bytes_confined

    def interrupted(path, data, **kw):
        writer(path, data, **kw)
        raise OSError("host lost after replacement")

    monkeypatch.setattr(publication, "atomic_write_bytes_confined", interrupted)
    with pytest.raises(OSError, match="host lost"):
        publication.publish(task, paths)
    back = StoryTask.from_dict(task.to_dict())
    assert not back.artifact_publication_complete
    (source.implementation_artifacts / "report.bin").write_bytes(b"unverified")
    monkeypatch.setattr(publication, "atomic_write_bytes_confined", writer)
    publication.publish(back, paths)
    assert (paths.implementation_artifacts / "report.bin").read_bytes() == b"\xff\x00\r\nreport"
    assert back.artifact_publication_complete


@pytest.mark.parametrize(
    "declaration",
    [
        "../escape",
        "/absolute",
        "C:/absolute",
        "*.md",
        "dir/../x",
        "deferred-work.md",
        "sprint-status.yaml",
        "report.bin/",
        "SPRINT-STATUS.YAML",
        "Deferred-Work.md",
        "sprint-status.yaml. ",
        "deferred-work.md ",
        "dir./report.bin",
        "NUL.txt",
        "report.bin:stream",
        "...",
    ],
)
def test_invalid_paths_refused(publication_case, declaration):
    task, paths, source = publication_case
    (source.implementation_artifacts / "spec.md").write_text(
        f"---\nartifact_deliverables: ['{declaration}']\n---\n"
    )
    with pytest.raises(publication.PublicationError, match="invalid artifact|reserved"):
        publication.prepare(task, paths, source)
    assert task.artifact_payload is None


@pytest.mark.parametrize("declaration", ["null", "report.bin", "{}", "[null]"])
def test_malformed_list_refused(publication_case, declaration):
    task, paths, source = publication_case
    (source.implementation_artifacts / "spec.md").write_text(
        f"---\nartifact_deliverables: {declaration}\n---\n"
    )
    with pytest.raises(publication.PublicationError):
        publication.prepare(task, paths, source)


@pytest.mark.parametrize("kind", ["directory", "symlink", "parent-symlink", "missing"])
def test_nonregular_sources_refused(publication_case, kind):
    task, paths, source = publication_case
    report = source.implementation_artifacts / "report.bin"
    report.unlink()
    if kind == "directory":
        report.mkdir()
    elif kind == "symlink":
        report.symlink_to(source.implementation_artifacts / "spec.md")
    elif kind == "parent-symlink":
        linked = source.implementation_artifacts / "linked"
        linked.symlink_to(paths.implementation_artifacts, target_is_directory=True)
        (paths.implementation_artifacts / "report.bin").write_bytes(b"outside")
        (source.implementation_artifacts / "spec.md").write_text(
            "---\nartifact_deliverables: [linked/report.bin]\n---\n"
        )
    with pytest.raises(publication.PublicationError):
        publication.prepare(task, paths, source)


def test_old_state_cannot_create_overwrite_authority(publication_case):
    task, paths, source = publication_case
    task.artifact_baseline = None
    publication.prepare(task, paths, source)
    with pytest.raises(publication.PublicationError, match="no pre-execution"):
        publication.publish(task, paths)
    assert task.artifact_payload is not None
    assert base64.b64decode(task.artifact_payload["report.bin"]) == b"\xff\x00\r\nreport"


def test_destination_symlink_refused_even_when_equal(publication_case):
    task, paths, source = publication_case
    publication.prepare(task, paths, source)
    destination = paths.implementation_artifacts / "report.bin"
    destination.symlink_to(source.implementation_artifacts / "report.bin")
    with pytest.raises(publication.PublicationError, match="symlink"):
        publication.publish(task, paths)
    assert destination.is_symlink()


def test_destination_changed_during_git_probe_is_preserved(publication_case, monkeypatch):
    task, paths, source = publication_case
    publication.prepare(task, paths, source)
    destination = paths.implementation_artifacts / "report.bin"

    def operator_edit(*_):
        destination.write_bytes(b"operator while git ran")
        return False

    monkeypatch.setattr(publication.verify, "path_tracked", operator_edit)
    with pytest.raises(publication.PublicationError, match="changed during publication"):
        publication.publish(task, paths)
    assert destination.read_bytes() == b"operator while git ran"


def test_tracked_deliverables_ride_git(publication_case, monkeypatch):
    task, paths, source = publication_case
    monkeypatch.setattr(publication.verify, "path_tracked", lambda *_: True)
    publication.prepare(task, paths, source)
    assert task.artifact_payload == {}
    publication.publish(task, paths)
    assert task.artifact_publication_complete


def test_destination_that_becomes_tracked_is_refused(publication_case, monkeypatch):
    task, paths, source = publication_case
    publication.prepare(task, paths, source)
    destination = paths.implementation_artifacts / "report.bin"
    monkeypatch.setattr(publication.verify, "path_tracked", lambda *_: True)
    with pytest.raises(publication.PublicationError, match="became tracked"):
        publication.publish(task, paths)
    assert not destination.exists()
    assert not task.artifact_publication_complete


def test_destination_that_becomes_unignored_is_refused(publication_case, monkeypatch):
    task, paths, source = publication_case
    publication.prepare(task, paths, source)
    destination = paths.implementation_artifacts / "report.bin"
    monkeypatch.setattr(publication.verify, "path_ignored", lambda *_: False)
    with pytest.raises(publication.PublicationError, match="no longer ignored"):
        publication.publish(task, paths)
    assert not destination.exists()
    assert not task.artifact_publication_complete


def test_unignored_declaration_refused(publication_case, monkeypatch):
    task, paths, source = publication_case
    monkeypatch.setattr(publication.verify, "path_ignored", lambda *_: False)
    with pytest.raises(publication.PublicationError, match="not ignored"):
        publication.prepare(task, paths, source)


def test_read_fault_retains_baseline_and_refuses_payload(publication_case, monkeypatch):
    task, paths, source = publication_case
    report = source.implementation_artifacts / "report.bin"
    read = publication._contents

    def unreadable(root, path):
        if path == report:
            raise OSError("unreadable report.bin")
        return read(root, path)

    monkeypatch.setattr(publication, "_contents", unreadable)
    with pytest.raises(OSError, match="unreadable"):
        publication.prepare(task, paths, source)
    assert task.artifact_baseline is not None
    assert task.artifact_payload is None


def test_accepted_spec_parent_traversal_refused_before_read(publication_case, monkeypatch):
    task, paths, source = publication_case
    task.spec_file = str(source.project / ".." / "escaped.md")
    escaped = source.project.parent / "escaped.md"
    escaped.write_text("---\nstatus: done\n---\n")
    with pytest.raises(publication.PublicationError, match="parent traversal"):
        publication.prepare(task, paths, source)
    assert task.artifact_payload is None


@pytest.mark.parametrize(
    "text",
    [
        "---\nartifact_deliverables: [report.bin\n---\n",
        "---\n- report.bin\n---\n",
        "---\nstatus: done\n",
        "not frontmatter",
        "---\n{}\n---\n",
    ],
)
def test_malformed_frontmatter_refuses_publication_intent(publication_case, text):
    task, paths, source = publication_case
    (source.implementation_artifacts / "spec.md").write_text(text)
    with pytest.raises(publication.PublicationError, match="invalid accepted spec frontmatter"):
        publication.prepare(task, paths, source)
    assert task.artifact_payload is None


def test_external_declaration_is_refused(publication_case, tmp_path):
    from dataclasses import replace

    task, paths, source = publication_case
    external = replace(source, implementation_artifacts=tmp_path / "external")
    external.implementation_artifacts.mkdir()
    with pytest.raises(publication.PublicationError, match="strictly inside"):
        publication.prepare(task, paths, external)
    assert task.artifact_payload is None


@pytest.mark.parametrize("fallback", [False, True])
def test_destination_edit_during_fsync_refuses_replace(publication_case, monkeypatch, fallback):
    from bmad_loop import platform_util

    task, paths, source = publication_case
    publication.prepare(task, paths, source)
    destination = paths.implementation_artifacts / "report.bin"
    fsync = os.fsync

    def edit_during_fsync(fd):
        fsync(fd)
        destination.write_bytes(b"operator during fsync")

    if fallback:
        monkeypatch.setattr(platform_util, "DIR_FD_ANCHORED_WRITES", False)
        monkeypatch.setattr(publication, "DIR_FD_ANCHORED_WRITES", False)
    monkeypatch.setattr(os, "fsync", edit_during_fsync)
    with pytest.raises(publication.PublicationError, match="changed during publication"):
        publication.publish(task, paths)
    assert destination.read_bytes() == b"operator during fsync"
    assert not task.artifact_publication_complete
    assert list(destination.parent.glob("*.tmp")) == []


def test_publication_refuses_when_confined_write_is_not_visible(publication_case, monkeypatch):
    task, paths, source = publication_case
    publication.prepare(task, paths, source)
    destination = paths.implementation_artifacts / "report.bin"

    def detached_write(path, data, **kwargs):
        kwargs["_before_replace"]()
        (path.parent / "detached-report.bin").write_bytes(data)

    monkeypatch.setattr(publication, "atomic_write_bytes_confined", detached_write)
    with pytest.raises(publication.PublicationError, match="not visible at destination"):
        publication.publish(task, paths)
    assert not destination.exists()
    assert not task.artifact_publication_complete


@pytest.mark.skipif(not publication.DIR_FD_ANCHORED_WRITES, reason="POSIX descriptor reads")
@pytest.mark.parametrize("swap", ["file", "parent"])
def test_source_swap_between_check_and_read_is_refused(publication_case, monkeypatch, swap):
    task, paths, source = publication_case
    report = source.implementation_artifacts / "report.bin"
    outside = paths.implementation_artifacts / "outside"
    outside.mkdir()
    (outside / report.name).write_bytes(b"outside secrets")
    if swap == "file":
        opener = os.open

        def swap_file(name, flags, *args, **kwargs):
            if name == report.name and "dir_fd" in kwargs:
                report.unlink()
                report.symlink_to(outside / report.name)
            return opener(name, flags, *args, **kwargs)

        monkeypatch.setattr(os, "open", swap_file)
    else:
        opener = publication.open_dir_confined

        def swap_parent(root, parent):
            report.parent.rename(report.parent.with_name("original"))
            report.parent.symlink_to(outside, target_is_directory=True)
            return opener(root, parent)

        monkeypatch.setattr(publication, "open_dir_confined", swap_parent)
    with pytest.raises((OSError, publication.PublicationError)):
        publication._contents(source.project, report)
    assert (outside / report.name).read_bytes() == b"outside secrets"


@pytest.mark.skipif(not publication.DIR_FD_ANCHORED_WRITES, reason="POSIX descriptor inventory")
def test_baseline_directory_swap_never_reads_redirected_contents(publication_case, monkeypatch):
    task, paths, _ = publication_case
    root = paths.implementation_artifacts
    directory = root / "nested"
    directory.mkdir()
    (directory / "report.bin").write_bytes(b"before")
    outside = paths.project / "outside"
    outside.mkdir()
    (outside / "report.bin").write_bytes(b"outside secrets")
    inode = directory.stat().st_ino
    scandir = os.scandir

    def swap_directory(fd):
        if isinstance(fd, int) and os.fstat(fd).st_ino == inode:
            directory.rename(root / "original")
            directory.symlink_to(outside, target_is_directory=True)
        return scandir(fd)

    monkeypatch.setattr(os, "scandir", swap_directory)
    with pytest.raises(publication.PublicationError, match="symlink"):
        publication.capture(task, paths)


def test_undecodable_accepted_spec_refuses_intent(publication_case):
    task, paths, source = publication_case
    (source.implementation_artifacts / "spec.md").write_bytes(b"---\nstatus: done\n\xff\n---\n")
    with pytest.raises(UnicodeDecodeError):
        publication.prepare(task, paths, source)
    assert task.artifact_payload is None
