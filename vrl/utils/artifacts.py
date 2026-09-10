"""Shared artifact path and provenance contracts."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

DATA_ROOT_ENV = "VRL_DATA_ROOT"


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

# Ordered manifest provenance contract shared by data validation and reward
# artifact materialization. Order is load-bearing: validators report the first
# missing field, so this must remain an explicit schema rather than a set.
SOURCE_BACKED_VIDEO_WORLD_METADATA_FIELDS = (
    "source",
    "source_repo",
    "source_split",
    "source_episode",
    "source_video",
    "source_frame_index",
    "decode_method",
    "conditioning",
)


class ArtifactManifestError(ValueError):
    """Raised when an artifact path violates storage policy."""


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
    return (root / path).resolve()


def coerce_data_root(value: str | Path | None) -> Path:
    """Normalize an optional data-root override to an absolute path."""

    return Path(value).expanduser().resolve() if value is not None else default_data_root()


__all__ = [
    "DATA_ROOT_ENV",
    "IMAGE_SUFFIXES",
    "SOURCE_BACKED_VIDEO_WORLD_METADATA_FIELDS",
    "ArtifactManifestError",
    "coerce_data_root",
    "default_data_root",
    "repo_root",
    "resolve_artifact_path",
]
