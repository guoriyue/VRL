"""Image/video tensor format conversion and media file IO.

Neutral home (no domain deps) so reward models, generation scripts, and data
prep all share one implementation. Two conversion entry points with different
contracts:

- ``to_uint8``: strict ``[0, 1]`` float tensor -> ``uint8`` (no denormalize).
  Reward media is guaranteed unit-range by the collector rescale step, so this
  only scales.
- ``image_to_uint8_hwc`` / ``to_pil_image``: defensive converter for inputs of
  uncertain range/layout (raw generation output, dataset arrays). Denormalizes
  ``[-1, 1]`` when ``min < 0`` and normalizes channel layout to HWC RGB.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

if TYPE_CHECKING:
    from PIL import Image as PILImage


def to_uint8(media: torch.Tensor) -> torch.Tensor:
    """Convert a unit-range ``[0, 1]`` float image/video tensor to ``uint8``.

    The round-then-clamp order matters: rounding first then clamping to
    ``[0, 255]`` avoids 256 overflow on values at the top of the range.
    """

    return (media * 255).round().clamp(0, 255).to(torch.uint8)


def image_to_uint8_hwc(image: Any) -> np.ndarray:
    """Convert a tensor/ndarray image to an ``HWC`` RGB ``uint8`` array.

    Accepts torch tensors or numpy arrays in ``CHW``/``HWC`` (optionally a
    leading singleton batch). Floats in ``[-1, 1]`` are denormalized when the
    min is negative; ``uint8``/integer inputs are passed through with clipping.
    """

    if isinstance(image, torch.Tensor):
        array = image.detach().cpu()
        if array.ndim == 4:
            if array.shape[0] != 1:
                raise ValueError(f"expected one image, got batch shape {tuple(array.shape)}")
            array = array[0]
        if array.ndim == 3 and array.shape[0] in {1, 3, 4}:
            array = array[:3].permute(1, 2, 0)
        if torch.is_floating_point(array):
            # NumPy cannot represent bfloat16. Normalize tensor float dtypes to
            # float32 while preserving integer values for the pass-through path.
            array = array.float()
        array_np = array.numpy()
    else:
        array_np = np.asarray(image)

    if array_np.ndim != 3:
        raise ValueError(f"expected an image with 3 dimensions, got shape {array_np.shape}")
    if array_np.shape[0] in {1, 3, 4} and array_np.shape[-1] not in {1, 3, 4}:
        array_np = array_np[:3].transpose(1, 2, 0)
    if array_np.shape[-1] == 1:
        array_np = np.repeat(array_np, 3, axis=-1)
    if array_np.shape[-1] == 4:
        array_np = array_np[..., :3]
    if array_np.shape[-1] != 3:
        raise ValueError(f"expected an RGB image, got shape {array_np.shape}")

    if np.issubdtype(array_np.dtype, np.floating):
        if float(np.nanmin(array_np)) < 0.0:
            array_np = (array_np + 1.0) * 0.5
        array_np = np.nan_to_num(array_np, nan=0.0, posinf=1.0, neginf=0.0)
        array_np = np.clip(array_np, 0.0, 1.0)
        return (array_np * 255.0).round().astype(np.uint8)
    return np.clip(array_np, 0, 255).astype(np.uint8)


def to_pil_image(image: Any) -> PILImage.Image:
    """Convert a tensor/ndarray/PIL image to an RGB ``PIL.Image``."""

    from PIL import Image

    if isinstance(image, Image.Image):
        return image.convert("RGB")
    return Image.fromarray(image_to_uint8_hwc(image), mode="RGB")


def load_reference_image(reference_image: Any) -> Any:
    """Load a reference-image path into an RGB ``PIL.Image``.

    Non-string or empty values pass through unchanged, so callers can forward an
    already-loaded image (or ``None``) without special-casing.
    """

    if not isinstance(reference_image, str) or not reference_image:
        return reference_image
    from PIL import Image

    return Image.open(reference_image).convert("RGB")


def write_png(image: Any, path: str | Path) -> None:
    """Write a tensor/ndarray/PIL image to an RGB PNG file."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    to_pil_image(image).save(path, format="PNG")


def video_tensor_to_uint8_frames(tensor: torch.Tensor) -> np.ndarray:
    """Convert a channel-first ``[C,T,H,W]`` (or ``[1,C,T,H,W]``) video to ``THWC`` ``uint8``."""

    if tensor.ndim == 5:
        if tensor.shape[0] != 1:
            raise ValueError(
                "video frame conversion expects one video per tensor; "
                f"got batch={tensor.shape[0]}",
            )
        tensor = tensor[0]
    if tensor.ndim != 4:
        raise ValueError(
            "video frame conversion expects [C,T,H,W] or [1,C,T,H,W] tensor, "
            f"got shape={tuple(tensor.shape)}",
        )
    if tensor.shape[0] not in {1, 3, 4}:
        raise ValueError(
            "video frame conversion expects channel-first video with 1, 3, or 4 channels, "
            f"got shape={tuple(tensor.shape)}",
        )
    # .cpu(): this converter ends in .numpy(), which requires a host tensor. The
    # single-worker pipelined rollout (forward_chunks_pipelined) can hand the reward
    # an artifact whose video tensor is still on CUDA, unlike the per-chunk path's
    # worker._to_cpu. Mirror image_to_uint8_hwc's intake so any device works.
    video = tensor.detach().cpu()
    input_is_float = torch.is_floating_point(video)
    if input_is_float:
        video = video.float()
        if float(video.min().item()) < -0.01:
            video = (video + 1.0) / 2.0
        video = video.clamp(0.0, 1.0)
    else:
        # Generation workers pack decoded videos as uint8 before the Ray wire.
        # Preserve those byte values when a direct executor consumer writes an
        # artifact instead of interpreting every nonzero byte as unit-range 1.0.
        video = video.clamp(0, 255).to(torch.uint8)
    if video.shape[0] == 1:
        video = video.repeat(3, 1, 1, 1)
    if video.shape[0] == 4:
        video = video[:3]
    video = video.permute(1, 2, 3, 0).contiguous()
    if input_is_float:
        return (video.numpy() * 255.0).round().astype(np.uint8)
    return video.numpy()


def frames_thwc_to_float(frames: torch.Tensor) -> torch.Tensor:
    """Normalize a ``[T,H,W,C]`` uint8/float frame stack to ``[T,H,W,3]`` float in ``[0,1]``."""

    if frames.ndim != 4 or frames.shape[-1] not in {1, 3, 4}:
        raise ValueError(f"frames must be [T,H,W,C], got {tuple(frames.shape)}")
    rgb = frames[..., :3]
    if torch.is_floating_point(rgb):
        # Float frames are assumed unit-range; tolerate an unusual 0-255 float source.
        out = rgb.float()
        if float(out.max().item()) > 1.5:
            out = out / 255.0
    else:
        out = rgb.float() / 255.0  # integer 0-255 frames
    if out.shape[-1] == 1:
        out = out.repeat(1, 1, 1, 3)
    return out.clamp(0.0, 1.0)


def sample_frames(frames: torch.Tensor, num_frames: int | None) -> torch.Tensor:
    """Evenly subsample a ``[T,...]`` stack to ``num_frames`` (no-op when fewer/None)."""

    if num_frames is None or frames.shape[0] <= num_frames:
        return frames
    indices = torch.linspace(0, frames.shape[0] - 1, steps=num_frames).round().long()
    return frames.index_select(0, indices)


def align_frame_counts(
    a: torch.Tensor,
    b: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Subsample both stacks to their common (minimum) frame count."""

    count = min(int(a.shape[0]), int(b.shape[0]))
    if count < 1:
        raise ValueError("need at least one frame per side")
    return sample_frames(a, count), sample_frames(b, count)


def read_video_frames(
    path: str | Path,
    num_frames: int | None = None,
) -> torch.Tensor:
    """Decode a video file to a ``[T,H,W,3]`` float tensor in ``[0,1]`` (optionally subsampled)."""

    import imageio.v2 as imageio

    reader = imageio.get_reader(Path(path))
    frames: list[torch.Tensor] = []
    try:
        for frame in reader:
            frames.append(torch.from_numpy(image_to_uint8_hwc(frame)))
    finally:
        reader.close()
    if not frames:
        raise ValueError(f"decoded no frames from {path}")
    return sample_frames(frames_thwc_to_float(torch.stack(frames, dim=0)), num_frames)


def read_image_as_frames(path: str | Path) -> torch.Tensor:
    """Decode an image file to a single-frame ``[1,H,W,3]`` float tensor in ``[0,1]``."""

    from PIL import Image

    with Image.open(path) as image:
        frame = torch.from_numpy(image_to_uint8_hwc(image.convert("RGB")))
    return frames_thwc_to_float(frame.unsqueeze(0))


def write_mp4(tensor: torch.Tensor, path: str | Path, *, fps: float) -> None:
    """Write a channel-first video tensor to an mp4 file."""

    frames = video_tensor_to_uint8_frames(tensor)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    import imageio.v2 as imageio

    with imageio.get_writer(path, fps=fps, codec="libx264", macro_block_size=1) as writer:
        for frame in frames:
            writer.append_data(frame)


__all__ = [
    "align_frame_counts",
    "frames_thwc_to_float",
    "image_to_uint8_hwc",
    "load_reference_image",
    "read_image_as_frames",
    "read_video_frames",
    "sample_frames",
    "to_pil_image",
    "to_uint8",
    "video_tensor_to_uint8_frames",
    "write_mp4",
    "write_png",
]
