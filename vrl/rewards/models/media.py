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


def pil_frames_from_media(media: Any) -> list[list[Image.Image]]:
    """Per-sample RGB PIL frame lists from a ``score_media`` payload.

    Tensors: ``[C,H,W]`` -> one sample with one frame; ``[C,T,H,W]`` -> one sample
    with ``T`` frames; ``[B,C,T,H,W]`` -> ``B`` samples. A 4-D tensor whose
    leading dim is not a channel count (1/3/4) is read as ``[N,C,H,W]`` frames of
    one sample, the layout image-only rewards are handed for frame windows.
    ``numpy`` ``HWC`` / ``THWC`` arrays, a PIL image, and a list of PIL images are
    one sample each. Anything else raises ``TypeError``.
    """

    import numpy as np
    import torch
    from PIL import Image

    from vrl.utils.media import to_pil_image, video_tensor_to_uint8_frames

    if isinstance(media, Image.Image):
        return [[media.convert("RGB")]]
    if isinstance(media, (list, tuple)) and all(isinstance(item, Image.Image) for item in media):
        return [[item.convert("RGB") for item in media]]
    if isinstance(media, np.ndarray):
        if media.ndim == 3:
            return [[to_pil_image(media)]]
        if media.ndim == 4:
            return [[to_pil_image(frame) for frame in media]]
        raise TypeError(f"reward media array must be HWC or THWC, got {media.shape}")
    if not isinstance(media, torch.Tensor):
        raise TypeError(f"reward media must be a tensor, array, or PIL image, got {type(media)}")
    if media.ndim == 3:
        return [[to_pil_image(media)]]
    if media.ndim == 4 and media.shape[0] not in (1, 3, 4):
        return [[to_pil_image(frame) for frame in media]]
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
