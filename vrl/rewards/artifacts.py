"""Materialize video/image reward artifacts for worker-side scoring."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
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
        artifact_format: str = "tensor",
        manifest_name: str = "manifest.jsonl",
    ) -> None:
        if media_type not in {"image", "video", "tensor"}:
            raise ValueError("media_type must be image, video, or tensor")
        if artifact_format not in {"tensor", "mp4"}:
            raise ValueError("artifact_format must be tensor or mp4")
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
        policy_version = _policy_version(metadata)
        artifact_id = f"{sample_id}-{index}"
        if self.artifact_format == "mp4":
            path = self.root / f"{artifact_id}.mp4"
            fps = _fps(metadata)
            _write_video_mp4(tensor, path, fps=fps)
        else:
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
                "artifact_format": self.artifact_format,
                "fps": _fps(metadata),
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


def _write_video_mp4(tensor: torch.Tensor, path: Path, *, fps: float) -> None:
    """Write a channel-first video tensor to an mp4 file for external reward models."""

    frames = _video_tensor_to_uint8_frames(tensor)
    path.parent.mkdir(parents=True, exist_ok=True)
    import imageio.v2 as imageio

    with imageio.get_writer(path, fps=fps, codec="libx264", macro_block_size=1) as writer:
        for frame in frames:
            writer.append_data(frame)


def _video_tensor_to_uint8_frames(tensor: torch.Tensor) -> np.ndarray:
    if tensor.ndim == 5:
        if tensor.shape[0] != 1:
            raise ValueError(
                "artifact_format=mp4 expects one video per rollout tensor; "
                f"got batch={tensor.shape[0]}",
            )
        tensor = tensor[0]
    if tensor.ndim != 4:
        raise ValueError(
            "artifact_format=mp4 expects [C,T,H,W] or [1,C,T,H,W] tensor, "
            f"got shape={tuple(tensor.shape)}",
        )
    if tensor.shape[0] not in {1, 3, 4}:
        raise ValueError(
            "artifact_format=mp4 expects channel-first video with 1, 3, or 4 channels, "
            f"got shape={tuple(tensor.shape)}",
        )
    video = tensor.float()
    if float(video.min().item()) < -0.01:
        video = (video + 1.0) / 2.0
    video = video.clamp(0.0, 1.0)
    if video.shape[0] == 1:
        video = video.repeat(3, 1, 1, 1)
    if video.shape[0] == 4:
        video = video[:3]
    video = video.permute(1, 2, 3, 0).contiguous()
    return (video.numpy() * 255.0).round().astype(np.uint8)


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


def _policy_version(metadata: dict[str, Any]) -> int | None:
    value = metadata.get("policy_version")
    if value is None:
        return None
    return int(value)


__all__ = ["VideoRewardArtifactStore"]
