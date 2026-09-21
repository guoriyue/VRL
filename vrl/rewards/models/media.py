"""Media decoding shared by artifact-backed reward models.

Two entry points, one layout contract. ``decode_artifact_frames`` reads a
materialized artifact or in-memory media into a ``[T,H,W,3]`` float stack for
frame-based models; ``pil_frames_from_media`` turns the in-memory payload a
``TorchRewardModel.score_media`` receives into per-sample PIL frame lists for
CLIP-style scorers. Tensor layouts follow the collector contract everywhere:
``[C,H,W]`` is one image, ``[C,T,H,W]`` is one video, ``[B,C,T,H,W]`` is a
batch of videos. Which frames a scorer then uses (the middle one, three
evenly spaced, a fixed window) is that scorer's own decision.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from vrl.rewards.inference import RewardInferenceArtifact

if TYPE_CHECKING:
    import torch
    from PIL import Image


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


def artifact_middle_frame_image(artifact: RewardInferenceArtifact) -> Image.Image:
    """Middle frame of a reward artifact as an RGB PIL image.

    The input every image-domain detector/tagger reward consumes: an in-memory
    PIL image passes through, a tensor or media path is decoded to frames and
    the middle one is taken (a still image has exactly one frame).
    """

    from PIL import Image

    from vrl.utils.media import to_pil_image

    if not artifact.path or artifact.path.endswith(".pt"):
        media = artifact.as_media()
        if isinstance(media, Image.Image):
            return to_pil_image(media)
    frames = decode_artifact_frames(artifact, 1)
    return to_pil_image(frames[frames.shape[0] // 2])


def pil_frames_from_media(media: Any) -> list[list[Image.Image]]:
    """Per-sample RGB PIL frame lists from a ``score_media`` payload.

    Tensors: ``[C,H,W]`` -> one sample with one frame; ``[C,T,H,W]`` -> one sample
    with ``T`` frames; ``[B,C,T,H,W]`` -> ``B`` samples. Tensor videos are always channel-first;
    callers with ``[T,C,H,W]`` frames must permute to ``[C,T,H,W]`` explicitly.
    ``numpy`` ``HWC`` / ``THWC`` arrays, a PIL image, and a list of PIL images are
    one sample each. Empty media raises ``ValueError``; unsupported types raise ``TypeError``.
    """

    import numpy as np
    import torch
    from PIL import Image

    from vrl.utils.media import to_pil_image, video_tensor_to_uint8_frames

    if isinstance(media, Image.Image):
        if media.width == 0 or media.height == 0:
            raise ValueError("reward received an empty media image")
        return [[to_pil_image(media)]]
    if isinstance(media, (list, tuple)) and all(isinstance(item, Image.Image) for item in media):
        if not media or any(item.width == 0 or item.height == 0 for item in media):
            raise ValueError("reward received an empty media frame list")
        return [[to_pil_image(item) for item in media]]
    if isinstance(media, np.ndarray):
        if media.size == 0:
            raise ValueError("reward received an empty media array")
        if media.ndim == 3:
            return [[to_pil_image(media)]]
        if media.ndim == 4:
            return [[to_pil_image(frame) for frame in media]]
        raise TypeError(f"reward media array must be HWC or THWC, got {media.shape}")
    if not isinstance(media, torch.Tensor):
        raise TypeError(f"reward media must be a tensor, array, or PIL image, got {type(media)}")
    if media.numel() == 0:
        raise ValueError("reward received an empty media tensor")
    if media.ndim == 3:
        return [[to_pil_image(media)]]
    videos = [media] if media.ndim == 4 else list(media) if media.ndim == 5 else None
    if videos is None:
        raise TypeError(f"reward media tensor must be 3-5 dimensional, got {tuple(media.shape)}")
    return [
        [Image.fromarray(frame, mode="RGB") for frame in video_tensor_to_uint8_frames(video)]
        for video in videos
    ]


def evenly_spaced_frames(frames: list[Image.Image], count: int) -> list[Image.Image]:
    """``count`` frames at evenly spaced interior positions (``t//4, t//2, 3t//4`` for 3).

    Fewer frames than ``count`` are returned as they are.
    """

    total = len(frames)
    if total <= count:
        return list(frames)
    return [frames[total * (index + 1) // (count + 1)] for index in range(count)]


__all__ = ["decode_artifact_frames", "evenly_spaced_frames", "pil_frames_from_media"]
