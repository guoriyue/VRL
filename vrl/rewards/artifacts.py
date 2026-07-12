"""Materialize video/image reward artifacts for worker-side scoring."""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any, Literal, get_args

import torch

from vrl.rewards.inference import MEDIA_TYPES, MediaType, RewardInferenceArtifact
from vrl.rewards.types import RewardRollout
from vrl.trainers.data.artifacts import (
    DEFAULT_ARTIFACT_FIELDS,
    SOURCE_BACKED_VIDEO_WORLD_METADATA_FIELDS,
)
from vrl.utils.media import write_mp4

# On-disk artifact container. ``ArtifactFormat`` is the single source of truth;
# ARTIFACT_FORMATS derives from it (mp4 = real video container decord can read,
# tensor = torch.save .pt). media_type (image/video) is a separate axis.
ArtifactFormat = Literal["tensor", "mp4"]
ARTIFACT_FORMATS = frozenset(get_args(ArtifactFormat))


class VideoRewardArtifactStore:
    """Driver-side writer for stable reward media artifacts."""

    def __init__(
        self,
        root: str | Path,
        *,
        media_type: MediaType = "video",
        artifact_format: ArtifactFormat = "tensor",
        manifest_name: str = "manifest.jsonl",
    ) -> None:
        if media_type not in MEDIA_TYPES:
            raise ValueError(
                f"media_type must be one of {', '.join(get_args(MediaType))}",
            )
        if artifact_format not in ARTIFACT_FORMATS:
            raise ValueError(
                f"artifact_format must be one of {', '.join(get_args(ArtifactFormat))}",
            )
        if artifact_format == "mp4" and media_type != "video":
            raise ValueError("artifact_format=mp4 requires media_type=video")
        self.root = Path(root)
        self.media_type = media_type
        self.artifact_format = artifact_format
        self.manifest_path = self.root / manifest_name
        self._owned_paths: set[Path] = set()
        self.root.mkdir(parents=True, exist_ok=True)

    def materialize(self, rollouts: list[RewardRollout]) -> list[RewardInferenceArtifact]:
        artifacts: list[RewardInferenceArtifact] = []
        try:
            for rollout in rollouts:
                artifacts.append(self._write_one(rollout))
        except BaseException:
            self.release(artifacts)
            raise
        return artifacts

    def release(self, artifacts: list[RewardInferenceArtifact]) -> None:
        """Delete materializations owned by this store; safe to retry."""

        errors: list[OSError] = []
        for artifact in artifacts:
            if not artifact.path:
                continue
            path = Path(artifact.path)
            if path not in self._owned_paths:
                continue
            try:
                path.unlink(missing_ok=True)
            except OSError as error:
                errors.append(error)
            else:
                self._owned_paths.discard(path)
        if errors:
            raise OSError(
                f"failed to release {len(errors)} reward artifacts",
            ) from errors[0]

    def retain(self, artifacts: list[RewardInferenceArtifact]) -> None:
        """Transfer retained debug artifacts out of the store's ownership."""

        for artifact in artifacts:
            if artifact.path:
                self._owned_paths.discard(Path(artifact.path))

    def _write_one(self, rollout: RewardRollout) -> RewardInferenceArtifact:
        output = rollout.output
        if not isinstance(output, torch.Tensor):
            raise TypeError(
                "video reward artifact materialization requires tensor rollout output",
            )
        tensor = output.detach().cpu()
        _validate_media_shape(tensor, self.media_type)

        metadata = dict(rollout.metadata or {})
        materialization_id = uuid.uuid4().hex
        artifact_id = f"{rollout.source_request_id}:{rollout.sample_id}:{materialization_id}"
        fps = _fps(metadata)
        suffix = "mp4" if self.artifact_format == "mp4" else "pt"
        path = (self.root / f"{materialization_id}.{suffix}").resolve()
        self._owned_paths.add(path)
        try:
            if self.artifact_format == "mp4":
                write_mp4(tensor, path, fps=fps)
            else:
                torch.save(tensor, path)
            size_bytes = path.stat().st_size
            artifact = RewardInferenceArtifact(
                artifact_id=artifact_id,
                path=str(path),
                media_type=self.media_type,
                prompt=str(rollout.prompt),
                source_request_id=rollout.source_request_id,
                sample_id=rollout.sample_id,
                group_id=rollout.group_id,
                trajectory_id=rollout.trajectory_id,
                policy_version=rollout.policy_version,
                size_bytes=size_bytes,
                sha256=_sha256_file(path),
                metadata={
                    "shape": list(tensor.shape),
                    "dtype": str(tensor.dtype),
                    "artifact_format": self.artifact_format,
                    "fps": fps,
                    **_artifact_provenance(metadata),
                },
            )
            self._append_manifest(artifact)
        except BaseException:
            path.unlink(missing_ok=True)
            self._owned_paths.discard(path)
            raise
        return artifact

    def _append_manifest(self, artifact: RewardInferenceArtifact) -> None:
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with self.manifest_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(asdict(artifact), sort_keys=True) + "\n")


def _validate_media_shape(tensor: torch.Tensor, media_type: str) -> None:
    if tensor.numel() == 0:
        raise ValueError("video reward artifact tensor must be non-empty")
    if media_type == "image" and tensor.ndim not in {3, 4}:
        raise ValueError(
            "image reward artifact expects [C,H,W] or [B,C,H,W] tensor, "
            f"got shape={tuple(tensor.shape)}",
        )
    if media_type == "video" and tensor.ndim not in {4, 5}:
        raise ValueError(
            "video reward artifact expects [C,T,H,W] or [B,C,T,H,W] tensor, "
            f"got shape={tuple(tensor.shape)}",
        )


def _fps(metadata: dict[str, Any]) -> float:
    value = metadata.get("video_fps", metadata.get("fps", 8.0))
    return float(value) if value is not None else 8.0


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_provenance(metadata: dict[str, Any]) -> dict[str, Any]:
    # task_type is store-local (stamped by the collector, not a manifest field);
    # the rest derives from the manifest vocabulary. "references" is excluded on
    # purpose: it is list-valued and incompatible with the scalar filter below.
    keys = (
        "task_type",
        *(f for f in DEFAULT_ARTIFACT_FIELDS if f != "references"),
        *SOURCE_BACKED_VIDEO_WORLD_METADATA_FIELDS,
    )
    return {
        key: metadata[key]
        for key in keys
        if key in metadata and metadata[key] is not None and str(metadata[key]).strip()
    }


__all__ = ["VideoRewardArtifactStore"]
