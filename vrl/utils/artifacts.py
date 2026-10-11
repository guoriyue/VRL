"""Atomic file publication, file digests and the artifact-policy error.

Where a path may point is ``vrl.utils.paths``; this module only writes and
hashes files once a caller has a path.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import IO, Any


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
    """An artifact manifest row or path that violates the storage policy."""


__all__ = [
    "IMAGE_SUFFIXES",
    "ArtifactManifestError",
    "atomic_file",
    "fsync_directory",
    "sha256_file",
]
