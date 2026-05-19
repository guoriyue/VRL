"""Materialize video/image reward artifacts for worker-side scoring."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from vrl.rewards.inference import RewardInferenceArtifact
from vrl.rewards.types import RewardRollout


class VideoRewardArtifactStore:
    """Driver-side writer for stable reward media artifacts."""

    def __init__(
        self,
        root: str | Path,
        *,
        media_type: str = "video",
        manifest_name: str = "manifest.jsonl",
    ) -> None:
        if media_type not in {"image", "video", "tensor"}:
            raise ValueError("media_type must be image, video, or tensor")
        self.root = Path(root)
        self.media_type = media_type
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
        policy_version = _policy_version(metadata)
        artifact_id = f"{sample_id}-{index}"
        path = self.root / f"{artifact_id}.pt"
        torch.save(tensor, path)
        artifact = RewardInferenceArtifact(
            artifact_id=artifact_id,
            path=str(path),
            media_type=self.media_type,
            prompt=str(rollout.trajectory.prompt),
            sample_id=sample_id,
            policy_version=policy_version,
            metadata={
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "fps": metadata.get("video_fps", metadata.get("fps")),
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


def _sample_id(metadata: dict[str, Any], index: int) -> str:
    sample_ids = metadata.get("sample_ids")
    if isinstance(sample_ids, list) and index < len(sample_ids):
        return str(sample_ids[index])
    if "sample_id" in metadata:
        return str(metadata["sample_id"])
    return f"sample-{index}"


def _policy_version(metadata: dict[str, Any]) -> int | None:
    value = metadata.get("policy_version")
    if value is None:
        return None
    return int(value)


__all__ = ["VideoRewardArtifactStore"]
