"""Explicit ignored bundle deliverables, independent of artifact-only receipts.

A complete pre-execution inventory grants comparison authority, never selection
authority. The accepted spec selects exact files; base64 snapshots in task state
freeze their bytes before integration. Publication is replayable after any partial
write because each destination must still equal its baseline or intended bytes.
Files are created privately (0600); modes and xattrs are not preserved. The
check after staging is not a lock or atomic compare-and-swap: a noncooperating
filesystem writer can still race the final check and replacement. On platforms
without descriptor-relative reads, source reads use the checked fallback and
retain its check/read race.
"""

from __future__ import annotations

import base64
import hashlib
import os
import stat
from pathlib import Path, PureWindowsPath

from . import verify
from .bmadconfig import ProjectPaths
from .frontmatter import parse_frontmatter
from .model import StoryTask
from .platform_util import (
    DIR_FD_ANCHORED_WRITES,
    atomic_write_bytes_confined,
    has_parent_ref,
    names_tree_root,
    names_win32_alias,
    open_dir_confined,
)


class PublicationError(Exception):
    """Publication refused; the mounted source and persisted payload must survive."""


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _relative(value: object) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value.startswith("/")
        or PureWindowsPath(value).drive
        or any(c in value for c in '\\*?[]<>:"|')
        or any(ord(c) < 32 for c in value)
        or names_win32_alias(value)
        or names_tree_root(value)
        or any(p in ("", ".", "..") for p in value.split("/"))
    ):
        raise PublicationError(f"invalid artifact deliverable path: {value!r}")
    if value.casefold() in ("deferred-work.md", "sprint-status.yaml"):
        raise PublicationError(f"reserved orchestrator artifact: {value}")
    return value


def _confined(root: Path, path: Path) -> None:
    """Reject links at every component, including the configured root."""
    if has_parent_ref(path):
        raise PublicationError(f"artifact path contains parent traversal: {path}")
    if not path.is_relative_to(root):
        raise PublicationError(f"artifact is outside {root}: {path}")
    for part in (path, *path.parents):
        try:
            mode = part.lstat().st_mode
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(mode):
            raise PublicationError(f"artifact path is a symlink: {part}")
        if part != path and not stat.S_ISDIR(mode):
            raise PublicationError(f"artifact parent is not a directory: {part}")


def _contents(root: Path, path: Path) -> bytes | None:
    _confined(root, path)
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError:
        return None
    if not stat.S_ISREG(mode):
        raise PublicationError(f"artifact is not a regular file: {path}")
    if not DIR_FD_ANCHORED_WRITES:
        return path.read_bytes()  # checked fallback; no descriptor-relative API
    parent_fd = open_dir_confined(root, path.parent)
    if parent_fd is None:
        raise PublicationError(f"artifact parent was redirected: {path}")
    try:
        fd = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent_fd)
        with os.fdopen(fd, "rb") as stream:
            if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                raise PublicationError(f"artifact is not a regular file: {path}")
            return stream.read()
    finally:
        os.close(parent_fd)


def _local(paths: ProjectPaths) -> bool:
    root = paths.implementation_artifacts
    return root != paths.repo_root and root.is_relative_to(paths.repo_root)


def _root(paths: ProjectPaths) -> Path:
    root = paths.implementation_artifacts
    if not _local(paths):
        raise PublicationError(f"artifact directory must be strictly inside the repository: {root}")
    _confined(paths.repo_root, root)
    return root


def capture(task: StoryTask, paths: ProjectPaths) -> None:
    """Called only for a newly opened bundle mount, before its first execution."""
    root = paths.implementation_artifacts
    inventory: dict[str, str] = {}

    def walk(directory: Path) -> None:
        _confined(root, directory)
        directory_fd = open_dir_confined(root, directory) if DIR_FD_ANCHORED_WRITES else None
        if DIR_FD_ANCHORED_WRITES and directory_fd is None:
            raise PublicationError(f"artifact inventory directory was redirected: {directory}")
        try:
            with os.scandir(directory_fd if directory_fd is not None else directory) as entries:
                for entry in entries:
                    path = directory / entry.name
                    mode = entry.stat(follow_symlinks=False).st_mode
                    rel = path.relative_to(root).as_posix()
                    if stat.S_ISDIR(mode):
                        inventory[rel] = "directory"
                        walk(path)
                    elif stat.S_ISREG(mode):
                        data = _contents(root, path)
                        if data is None:
                            raise PublicationError(f"artifact disappeared during inventory: {path}")
                        inventory[rel] = _digest(data)
                    else:
                        inventory[rel] = "nonregular"
        finally:
            if directory_fd is not None:
                os.close(directory_fd)

    # Shared external artifacts already live at their destination. They neither
    # need implicit transport nor grant authority for explicit publication.
    if _local(paths):
        _root(paths)
        if root.exists():
            walk(root)
    task.artifact_baseline = inventory
    task.artifact_destination = str(root)
    task.artifact_payload = None
    task.artifact_publication_complete = False


def prepare(task: StoryTask, paths: ProjectPaths, source: ProjectPaths) -> None:
    """Freeze only explicit ignored regular deliverables from the accepted spec."""
    if task.artifact_payload is not None:
        return
    if not task.spec_file:
        raise PublicationError("accepted bundle spec is missing")
    spec = verify.resolve_spec_path(task.spec_file, source)
    root = source.implementation_artifacts
    # Specs outside implementation_artifacts are not automatically published,
    # but their explicit declaration is still subject to the same confinement.
    spec_root = source.project if spec.is_relative_to(source.project) else root
    spec_data = _contents(spec_root, spec)
    if spec_data is None:
        raise PublicationError(f"accepted bundle spec is missing: {spec}")
    fm = parse_frontmatter(spec_data.decode("utf-8"))
    if not fm:
        raise PublicationError(f"invalid accepted spec frontmatter: {spec}")
    declarations = fm.get("artifact_deliverables", [])
    if not isinstance(declarations, list):
        raise PublicationError(f"artifact_deliverables must be a list of exact paths: {spec}")
    selected = {_relative(item) for item in declarations}
    if selected:
        _root(source)  # explicit unsafe external declarations must fail
    if _local(source) and spec != root and spec.is_relative_to(root):
        selected.add(_relative(spec.relative_to(root).as_posix()))
    payload: dict[str, str] = {}
    for rel in sorted(selected):
        path = root / rel
        data = spec_data if path == spec else _contents(root, path)
        if data is None:
            raise PublicationError(f"artifact deliverable is missing: {path}")
        if verify.path_tracked(source.repo_root, path.relative_to(source.repo_root).as_posix()):
            continue  # tracked deliverables ride Git
        if not verify.path_ignored(source.repo_root, path):
            raise PublicationError(f"artifact deliverable is not ignored: {path}")
        payload[rel] = base64.b64encode(data).decode("ascii")
    task.artifact_payload = payload
    # Store intended bytes even for a legacy task: refusal must retain recovery
    # material, and a later resume must never read changed source bytes as intent.
    if task.artifact_destination is None:
        task.artifact_destination = str(paths.implementation_artifacts)


def publish(task: StoryTask, paths: ProjectPaths) -> None:
    """Compare each destination against pre-execution evidence, then replace."""
    if task.artifact_publication_complete:
        return
    root = paths.implementation_artifacts
    if task.artifact_destination != str(root):
        raise PublicationError(f"artifact destination changed since execution: {root}")
    if task.artifact_payload is None:
        raise PublicationError("artifact publication intent is missing")
    if task.artifact_payload:
        _root(paths)
    for rel, encoded in task.artifact_payload.items():
        path = root / _relative(rel)
        intended = base64.b64decode(encoded, validate=True)
        current = _contents(root, path)
        if current == intended:
            continue
        if task.artifact_baseline is None:
            raise PublicationError(f"no pre-execution artifact baseline for {path}")
        before = (task.artifact_baseline or {}).get(rel)
        now = None if current is None else _digest(current)
        if now != before:
            raise PublicationError(f"artifact destination conflict: {path}")
        if verify.path_tracked(paths.repo_root, path.relative_to(paths.repo_root).as_posix()):
            raise PublicationError(f"artifact destination became tracked: {path}")
        if not verify.path_ignored(paths.repo_root, path):
            raise PublicationError(f"artifact destination is no longer ignored: {path}")
        _confined(root, path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if _contents(root, path) != current:
            raise PublicationError(f"artifact destination changed during publication: {path}")

        def validate_destination() -> None:
            if _contents(root, path) != current:
                raise PublicationError(f"artifact destination changed during publication: {path}")

        atomic_write_bytes_confined(
            path, intended, confine_root=root, _before_replace=validate_destination
        )
        if _contents(root, path) != intended:
            raise PublicationError(f"published artifact is not visible at destination: {path}")
    task.artifact_publication_complete = True
