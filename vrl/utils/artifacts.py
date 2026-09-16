"""Shared artifact paths, atomic publication and provenance contracts."""

from __future__ import annotations

import hashlib
import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
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


class PathOutsideRootsError(ValueError):
    """A resolved path escaped its configured storage roots."""

    def __init__(self, path: Path) -> None:
        self.path = path
        super().__init__(f"path is outside configured roots: {path}")


class RootedPaths:
    """Resolve file references within one or more configured storage roots.

    Relative references require a single root. Absolute references must also
    stay within a root after resolving symlinks. Input syntax policies and
    protocol-specific errors remain with the caller; file writers are separate.
    """

    def __init__(self, root: str | Path, *additional_roots: str | Path) -> None:
        self.roots = tuple(
            Path(value).expanduser().resolve() for value in (root, *additional_roots)
        )

    def resolve(self, raw_path: str | Path, *, strict: bool = False) -> Path:
        """Resolve a reference and reject escapes; strict requires an existing path."""

        path = Path(raw_path).expanduser()
        if not path.is_absolute():
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


def default_data_root() -> Path:
    """Resolve the artifact data root from ``VRL_DATA_ROOT`` or local ignored data."""

    env_value = os.environ.get(DATA_ROOT_ENV, "").strip()
    if env_value:
        return Path(env_value).expanduser().resolve()
    return (repo_root() / "data" / "external").resolve()


def resolve_artifact_path(
    raw_path: str | Path,
    *,
    data_root: str | Path | None = None,
    allow_absolute: bool = False,
) -> Path:
    """Resolve one manifest artifact path under ``data_root``."""

    text = str(raw_path).strip()
    if not text:
        raise ArtifactManifestError("artifact path is empty")
    path = Path(text).expanduser()
    root = coerce_data_root(data_root)
    if path.is_absolute():
        if not allow_absolute:
            raise ArtifactManifestError(
                f"absolute artifact paths are not allowed by default: {text}",
            )
        return path.resolve()
    if any(part == ".." for part in path.parts):
        raise ArtifactManifestError(f"artifact paths must stay under data root: {text}")
    try:
        return RootedPaths(root).resolve(path)
    except PathOutsideRootsError as error:
        raise ArtifactManifestError(f"artifact paths must stay under data root: {text}") from error


def coerce_data_root(value: str | Path | None) -> Path:
    """Normalize an optional data-root override to an absolute path."""

    return Path(value).expanduser().resolve() if value is not None else default_data_root()


# ---- reward artifact wire contract -------------------------------------------
# Shared by the generation worker (writes the files) and the reward layer
# (admits them). Neither layer may import the other, so the contract lives here
# with the other artifact primitives.


@dataclass(frozen=True, slots=True)
class RewardArtifactSpec:
    """One reward component's request that the worker write its media to disk.

    Projected by the collector from the reward function's disk artifact
    stores. The worker writes one file per sample under ``root`` in the
    component's format and returns ``MaterializedArtifact`` references; the
    decoded media then never crosses the worker->driver wire and the driver
    never encodes video. ``fps`` is the mp4 encode rate, filled from the
    request's sampling when the store did not pin one.
    """

    name: str
    root: str
    media_type: Literal["image", "video"]
    artifact_format: Literal["tensor", "mp4"]
    fps: float | None = None

    def __post_init__(self) -> None:
        if not self.name or not self.root:
            raise ValueError("RewardArtifactSpec needs a component name and a root directory")
        if self.media_type not in ("image", "video"):
            raise ValueError(
                f"RewardArtifactSpec.media_type must be image or video, got {self.media_type!r}",
            )
        if self.artifact_format not in ("tensor", "mp4"):
            raise ValueError(
                f"RewardArtifactSpec.artifact_format must be tensor or mp4, got {self.artifact_format!r}",
            )
        if self.artifact_format == "mp4" and self.media_type != "video":
            raise ValueError("RewardArtifactSpec: artifact_format=mp4 requires media_type=video")


@dataclass(frozen=True, slots=True)
class MaterializedArtifact:
    """A reward media file already written for one sample by the rollout worker.

    Carries exactly what the reward service needs to admit the file (path +
    integrity); the driver never decodes or re-encodes the media it refers to.
    """

    path: str
    size_bytes: int
    sha256: str

    def __post_init__(self) -> None:
        if not self.path:
            raise ValueError("MaterializedArtifact.path must be non-empty")
        if isinstance(self.size_bytes, bool) or int(self.size_bytes) < 0:
            raise ValueError("MaterializedArtifact.size_bytes must be >= 0")
        if len(self.sha256) != 64:
            raise ValueError("MaterializedArtifact.sha256 must be a hex digest")


__all__ = [
    "DATA_ROOT_ENV",
    "IMAGE_SUFFIXES",
    "ArtifactManifestError",
    "MaterializedArtifact",
    "PathOutsideRootsError",
    "RewardArtifactSpec",
    "RootedPaths",
    "atomic_file",
    "coerce_data_root",
    "default_data_root",
    "repo_root",
    "resolve_artifact_path",
]
