"""Materialize video/image reward artifacts for worker-side scoring."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Literal, get_args

import torch

from vrl.rewards.inference import MEDIA_TYPES, MediaType, RewardInferenceArtifact
from vrl.rewards.types import RewardRollout
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
        self.root.mkdir(parents=True, exist_ok=True)

    def materialize(self, rollouts: list[RewardRollout]) -> list[RewardInferenceArtifact]:
        artifacts: list[RewardInferenceArtifact] = []
        for index, rollout in enumerate(rollouts):
            artifact = self._write_one(rollout, index)
            artifacts.append(artifact)
        return artifacts

    def _write_one(self, rollout: RewardRollout, index: int) -> RewardInferenceArtifact:
        output = rollout.trajectory.output
        if not isinstance(output, torch.Tensor):
            raise TypeError(
                "video reward artifact materialization requires tensor rollout output",
            )
        tensor = output.detach().cpu()
        _validate_media_shape(tensor, self.media_type)

        metadata = dict(rollout.metadata or {})
        sample_id = _sample_id(metadata, index)
        artifact_id = f"{sample_id}-{index}"
        fps = _fps(metadata)
        policy_version = metadata.get("policy_version")
        policy_version = int(policy_version) if policy_version is not None else None
        if self.artifact_format == "mp4":
            path = self.root / f"{artifact_id}.mp4"
            write_mp4(tensor, path, fps=fps)
        else:
            path = self.root / f"{artifact_id}.pt"
            torch.save(tensor, path)
        artifact = RewardInferenceArtifact(
            artifact_id=artifact_id,
            path=str(path.resolve()),
            media_type=self.media_type,
            prompt=str(rollout.trajectory.prompt),
            sample_id=sample_id,
            policy_version=policy_version,
            metadata={
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "artifact_format": self.artifact_format,
                "fps": fps,
                **_artifact_provenance(metadata),
            },
        )
        self._append_manifest(artifact)
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


def _sample_id(metadata: dict[str, Any], index: int) -> str:
    sample_ids = metadata.get("sample_ids")
    if isinstance(sample_ids, list) and index < len(sample_ids):
        return str(sample_ids[index])
    if "sample_id" in metadata:
        return str(metadata["sample_id"])
    return f"sample-{index}"


def _artifact_provenance(metadata: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "task_type",
        "reference_image",
        "reference_video",
        "source",
        "source_repo",
        "source_split",
        "source_episode",
        "source_video",
        "source_frame_index",
        "decode_method",
        "conditioning",
    )
    return {
        key: metadata[key]
        for key in keys
        if key in metadata and metadata[key] is not None and str(metadata[key]).strip()
    }


__all__ = ["VideoRewardArtifactStore"]
