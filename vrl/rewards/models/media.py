"""Media decoding shared by artifact-backed reward models."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from vrl.rewards.inference import RewardInferenceArtifact

if TYPE_CHECKING:
    import torch


def decode_artifact_frames(
    artifact: RewardInferenceArtifact,
    num_frames: int | None = None,
) -> torch.Tensor:
    """Decode a reward artifact to a ``[T,H,W,3]`` float frame stack.

    Reward models accept either a materialized image/video path or the
    collector's in-memory channel-first tensor. Both representations must reach
    model scoring with identical ``[0,1]`` pixel semantics.
    """

    import torch

    from vrl.utils.artifacts import IMAGE_SUFFIXES
    from vrl.utils.media import (
        frames_thwc_to_float,
        image_to_uint8_hwc,
        read_image_as_frames,
        read_video_frames,
        sample_frames,
        video_tensor_to_uint8_frames,
    )

    path = artifact.path
    if path and not path.endswith(".pt"):
        if Path(path).suffix.lower() in IMAGE_SUFFIXES:
            return read_image_as_frames(path)
        return read_video_frames(path, num_frames)
    media = artifact.as_media()
    if isinstance(media, torch.Tensor):
        if media.ndim in {4, 5}:
            frames = torch.from_numpy(video_tensor_to_uint8_frames(media))
            return sample_frames(frames_thwc_to_float(frames), num_frames)
        if media.ndim == 3:
            image = torch.from_numpy(image_to_uint8_hwc(media))
            return frames_thwc_to_float(image.unsqueeze(0))
    raise TypeError(
        f"reward artifact expected image/video tensor or media path, got {type(media)}",
    )


__all__ = ["decode_artifact_frames"]
