"""Target-aware video similarity reward for robot-domain Video2World.

The first implementation is intentionally lightweight and local: it compares
generated frames with a manifest-provided ``target_video`` or ``target_image``
using low-resolution RGB features. This is a probeable baseline and a stable
contract for later DINO/CLIP/video-encoder replacements without changing the
training pipeline.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from vrl.trainers.data.artifacts import default_data_root, resolve_artifact_path
from vrl.utils.media import image_to_uint8_hwc, video_tensor_to_uint8_frames


class TargetVideoSimilarityModel:
    """Compare generated video frames against target media from artifact metadata."""

    def __init__(self, worker_config: Mapping[str, Any]) -> None:
        cfg = dict(worker_config)
        self.worker_config = cfg
        self.data_root = str(cfg.get("data_root") or default_data_root())
        self.num_frames = max(1, int(cfg.get("num_frames", 8)))
        self.feature_size = max(8, int(cfg.get("feature_size", 64)))
        self.sequence_weight = float(cfg.get("sequence_weight", 0.5))
        self.final_weight = float(cfg.get("final_weight", 0.3))
        self.temporal_weight = float(cfg.get("temporal_weight", 0.2))
        self.allow_absolute_paths = bool(cfg.get("allow_absolute_paths", True))

    def __call__(self, *, artifact: Any, request: Any) -> Mapping[str, float]:
        del request
        metadata = dict(getattr(artifact, "metadata", {}) or {})
        target_video = str(metadata.get("target_video", "") or "").strip()
        target_image = str(metadata.get("target_image", "") or "").strip()
        if not target_video and not target_image:
            raise ValueError(
                f"artifact {artifact.artifact_id!r} has no target_video or target_image metadata",
            )

        generated = _frames_from_artifact(artifact, self.num_frames)
        target = (
            _frames_from_video_path(self._resolve(target_video), self.num_frames)
            if target_video
            else _frames_from_image_path(self._resolve(target_image))
        )
        if len(target) == 1 and len(generated) > 1:
            target = target.expand(len(generated), -1, -1, -1)
        generated, target = _align_frame_counts(generated, target)

        gen_features = _features(generated, self.feature_size)
        tgt_features = _features(target, self.feature_size)
        sequence = _unit_similarity(gen_features, tgt_features)
        final = _unit_similarity(gen_features[-1:], tgt_features[-1:])
        temporal = _temporal_similarity(gen_features, tgt_features)
        total_weight = self.sequence_weight + self.final_weight + self.temporal_weight
        if total_weight <= 0:
            raise ValueError("target_video_similarity weights must sum to a positive value")
        overall = (
            self.sequence_weight * sequence
            + self.final_weight * final
            + self.temporal_weight * temporal
        ) / total_weight
        return {
            "target_similarity": float(overall),
            "target_sequence_similarity": float(sequence),
            "target_final_frame_similarity": float(final),
            "target_temporal_similarity": float(temporal),
        }

    def _resolve(self, raw_path: str) -> Path:
        return resolve_artifact_path(
            raw_path,
            data_root=self.data_root,
            allow_absolute=self.allow_absolute_paths,
        )


def _frames_from_artifact(artifact: Any, num_frames: int) -> torch.Tensor:
    path = str(getattr(artifact, "path", "") or "")
    if path and not path.endswith(".pt"):
        suffix = Path(path).suffix.lower()
        if suffix in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}:
            return _frames_from_image_path(Path(path))
        return _frames_from_video_path(Path(path), num_frames)
    media = artifact.as_media()
    if isinstance(media, torch.Tensor):
        if media.ndim in {4, 5}:
            frames = torch.from_numpy(video_tensor_to_uint8_frames(media))
            return _sample_frames(_thwc_to_float(frames), num_frames)
        if media.ndim in {3, 4}:
            image = torch.from_numpy(image_to_uint8_hwc(media))
            return _thwc_to_float(image.unsqueeze(0))
    raise TypeError(
        f"target_video_similarity expected image/video tensor or media path, got {type(media)}",
    )


def _frames_from_image_path(path: Path) -> torch.Tensor:
    from PIL import Image

    with Image.open(path) as image:
        frame = torch.from_numpy(image_to_uint8_hwc(image.convert("RGB")))
    return _thwc_to_float(frame.unsqueeze(0))


def _frames_from_video_path(path: Path, num_frames: int) -> torch.Tensor:
    import imageio.v2 as imageio

    reader = imageio.get_reader(path)
    frames: list[torch.Tensor] = []
    try:
        for frame in reader:
            frames.append(torch.from_numpy(image_to_uint8_hwc(frame)))
    finally:
        reader.close()
    if not frames:
        raise ValueError(f"target_video_similarity decoded no frames from {path}")
    return _sample_frames(_thwc_to_float(torch.stack(frames, dim=0)), num_frames)


def _sample_frames(frames: torch.Tensor, num_frames: int) -> torch.Tensor:
    if frames.ndim != 4:
        raise ValueError(f"frames must be [T,H,W,C], got {tuple(frames.shape)}")
    if frames.shape[0] <= num_frames:
        return frames
    indices = torch.linspace(0, frames.shape[0] - 1, steps=num_frames).round().long()
    return frames.index_select(0, indices)


def _align_frame_counts(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    count = min(int(a.shape[0]), int(b.shape[0]))
    if count < 1:
        raise ValueError("target_video_similarity needs at least one frame per side")
    if a.shape[0] != count:
        a = _sample_frames(a, count)
    if b.shape[0] != count:
        b = _sample_frames(b, count)
    return a, b


def _thwc_to_float(frames: torch.Tensor) -> torch.Tensor:
    if frames.ndim != 4 or frames.shape[-1] not in {1, 3, 4}:
        raise ValueError(f"frames must be [T,H,W,C], got {tuple(frames.shape)}")
    frames = frames[..., :3].float() / 255.0
    if frames.shape[-1] == 1:
        frames = frames.repeat(1, 1, 1, 3)
    return frames


def _features(frames: torch.Tensor, size: int) -> torch.Tensor:
    nchw = frames.permute(0, 3, 1, 2).contiguous()
    resized = F.interpolate(nchw, size=(size, size), mode="bilinear", align_corners=False)
    return resized.flatten(start_dim=1)


def _unit_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    distance = (a - b).abs().mean().clamp(0.0, 1.0)
    return float((1.0 - distance).item())


def _temporal_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.shape[0] < 2 or b.shape[0] < 2:
        return _unit_similarity(a, b)
    return _unit_similarity(a[1:] - a[:-1], b[1:] - b[:-1])


__all__ = ["TargetVideoSimilarityModel"]
