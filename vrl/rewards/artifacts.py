"""Materialize video/image reward artifacts for worker-side scoring."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any, Literal, get_args

import torch

from vrl.rewards.inference import (
    MEDIA_TYPES,
    MediaType,
    RewardInferenceArtifact,
    sha256_file,
)
from vrl.rewards.types import RewardSample
from vrl.utils.artifacts import (
    SOURCE_BACKED_VIDEO_WORLD_METADATA_FIELDS,
)
from vrl.utils.media import write_mp4

# On-disk artifact container. ``ArtifactFormat`` is the single source of truth;
# ARTIFACT_FORMATS derives from it (mp4 = real video container decord can read,
# tensor = torch.save .pt). media_type (image/video) is a separate axis.
ArtifactFormat = Literal["tensor", "mp4"]
ARTIFACT_FORMATS = frozenset(get_args(ArtifactFormat))

# Reward-inference artifact wire allow-list. This is intentionally not derived
# from PromptExample: list-valued manifest references do not cross this scalar
# boundary, while rollout task/source provenance does. Each schema therefore
# keeps its own explicit contract instead of coupling rewards to trainer data.
_REWARD_ARTIFACT_PROVENANCE_FIELDS = (
    "task_type",
    "reference_image",
    "reference_video",
    "target_image",
    "target_video",
    *SOURCE_BACKED_VIDEO_WORLD_METADATA_FIELDS,
)


class VideoRewardArtifactStore:
    """Driver-side writer for stable reward media artifacts."""

    def __init__(
        self,
        root: str | Path,
        *,
        media_type: MediaType = "video",
        artifact_format: ArtifactFormat = "tensor",
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
        self._owned_paths: set[Path] = set()
        self.root.mkdir(parents=True, exist_ok=True)

    def materialize(self, samples: list[RewardSample]) -> list[RewardInferenceArtifact]:
        artifacts: list[RewardInferenceArtifact] = []
        try:
            for sample in samples:
                artifacts.append(self._write_one(sample))
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

    def _write_one(self, sample: RewardSample) -> RewardInferenceArtifact:
        output = sample.output
        if not isinstance(output, torch.Tensor):
            raise TypeError(
                "video reward artifact materialization requires tensor sample output",
            )
        tensor = output.detach().cpu()
        _validate_media_shape(tensor, self.media_type)

        metadata = dict(sample.metadata or {})
        materialization_id = uuid.uuid4().hex
        artifact_id = f"{sample.sample_id}:{materialization_id}"
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
            # Per-request audit trails are the opt-in debug_dir JSONLs owned by
            # InferenceRewardFunction._write_debug; the store writes media only.
            artifact = RewardInferenceArtifact(
                artifact_id=artifact_id,
                path=str(path),
                prompt=str(sample.prompt),
                size_bytes=size_bytes,
                sha256=sha256_file(path),
                metadata=_artifact_provenance(metadata),
            )
        except BaseException:
            path.unlink(missing_ok=True)
            self._owned_paths.discard(path)
            raise
        return artifact


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


def _artifact_provenance(metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        key: metadata[key]
        for key in _REWARD_ARTIFACT_PROVENANCE_FIELDS
        if key in metadata and metadata[key] is not None and str(metadata[key]).strip()
    }


__all__ = ["VideoRewardArtifactStore"]
