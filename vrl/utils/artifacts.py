"""Shared artifact paths, atomic publication and provenance contracts."""

from __future__ import annotations

import hashlib
import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import IO, Any, Literal

DATA_ROOT_ENV = "VRL_DATA_ROOT"


@contextmanager
def atomic_file(
    path: str | Path,
    *,
    binary: bool = False,
    overwrite: bool = True,
) -> Iterator[IO[Any]]:
    """Publish a completed file from a temporary sibling on successful exit.

    Flush and fsync content before replacement (or exclusive hard-link creation).
    Exceptions clean up the temporary file. This does not fsync the directory
    or promise cleanup after abrupt process termination. Path policy belongs to
    the caller; no root restriction or tilde expansion is applied here.
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb" if binary else "w",
            encoding=None if binary else "utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            yield handle
            handle.flush()
            os.fsync(handle.fileno())
        if overwrite:
            os.replace(temporary, path)
        else:
            os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def fsync_directory(path: str | Path) -> None:
    """Flush published directory entries on filesystems supporting directory fsync."""
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def sha256_file(path: str | Path) -> str:
    """Canonical SHA-256 hex digest of one file's bytes.

    The single file-integrity implementation shared across domains: reward
    artifact writer and service validator (which must hash identically or
    shared-filesystem integrity checks fail), checkpoint/manifest identity in
    eval reports, and dataset derivation manifests.
    """

    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


# Media file extensions used only to classify a path as image-vs-video (reward
# frame decode, manifest readability probe). This is plain extension taxonomy —
# it is NOT part of the artifact/manifest contract, so it lives in the torch-free
# artifacts leaf rather than duplicated inside either consumer.
IMAGE_SUFFIXES = frozenset({".bmp", ".gif", ".jpeg", ".jpg", ".png", ".ppm", ".webp"})


class ArtifactManifestError(ValueError):
    """Raised when an artifact path violates storage policy."""


class PathOutsideRootsError(ArtifactManifestError):
    """A resolved path escaped its configured storage roots."""

    def __init__(self, path: Path) -> None:
        self.path = path
        super().__init__(f"path is outside configured roots: {path}")


class RootedPaths:
    """Where artifact references may point: one or more roots, and one ``resolve``.

    The policy is fixed when the object is built, so every reference a
    consumer resolves is judged the same way:

    - a relative reference lands under the single root (several roots cannot
      say which one, so relative references are refused there);
    - an absolute reference is, per ``absolute``, kept ``"inside"`` a root
      after symlinks resolve (the reward service's sandbox), refused outright
      (``"reject"``, the default for manifests under the data root), or taken
      from ``"anywhere"`` (manifests allowed to point outside the data root).

    Symlink escapes are caught after resolution. Protocol-specific errors and
    file writers remain with the caller.
    """

    def __init__(
        self,
        root: str | Path,
        *additional_roots: str | Path,
        absolute: Literal["inside", "reject", "anywhere"] = "inside",
    ) -> None:
        self.roots = tuple(user_path(value) for value in (root, *additional_roots))
        self.absolute = absolute

    @classmethod
    def data_root(
        cls, value: str | Path | None = None, *, allow_absolute: bool = False
    ) -> RootedPaths:
        """The manifest data root: ``value``, else ``VRL_DATA_ROOT``, else the local default."""

        if value is None:
            value = os.environ.get(DATA_ROOT_ENV, "").strip() or repo_root() / "data" / "external"
        return cls(value, absolute="anywhere" if allow_absolute else "reject")

    @property
    def root(self) -> Path:
        """The single root, for callers that write under it."""

        if len(self.roots) != 1:
            raise ValueError("root is defined only for a single storage root")
        return self.roots[0]

    def resolve(self, raw_path: str | Path, *, strict: bool = False) -> Path:
        """One reference to an absolute path under the policy; ``strict`` requires it to exist."""

        text = str(raw_path).strip()
        if not text:
            raise ArtifactManifestError("artifact path is empty")
        path = Path(text).expanduser()
        if path.is_absolute():
            if self.absolute == "reject":
                raise ArtifactManifestError(
                    f"absolute artifact paths are not allowed by default: {text}",
                )
            if self.absolute == "anywhere":
                return path.resolve(strict=strict)
        else:
            if len(self.roots) != 1:
                raise ValueError("relative paths require exactly one storage root")
            path = self.roots[0] / path
        resolved = path.resolve(strict=strict)
        if not any(resolved.is_relative_to(root) for root in self.roots):
            raise PathOutsideRootsError(resolved)
        return resolved


def repo_root() -> Path:
    """Return the repository root for local ignored artifact defaults."""

    return Path(__file__).resolve().parents[2]


def user_path(value: str | Path) -> Path:
    """A path the user wrote (CLI argument or config value): ``~`` expanded, made absolute."""

    return Path(value).expanduser().resolve()


__all__ = [
    "DATA_ROOT_ENV",
    "IMAGE_SUFFIXES",
    "ArtifactManifestError",
    "PathOutsideRootsError",
    "RootedPaths",
    "atomic_file",
    "repo_root",
    "user_path",
]
