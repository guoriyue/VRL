"""Shared image/video generation through the stepwise denoise model boundary.

This does not replace native pipeline protocols such as the frozen SANA
evaluation, or token-autoregressive generation.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import platform
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any, Protocol

import torch

from vrl.generation.types import DenoiseRequest
from vrl.math.denoise.flow_matching import sde_step_with_logprob
from vrl.utils.media import to_pil_image

if TYPE_CHECKING:
    from PIL import Image

    from vrl.models.steps.denoise.base import DiffusionModelBase, DiffusionSamplingStateBase


class ImageSampling(Protocol):
    """Sampling values consumed by native stepwise image generation."""

    @property
    def width(self) -> int: ...

    @property
    def height(self) -> int: ...

    @property
    def num_steps(self) -> int: ...

    @property
    def guidance_scale(self) -> float: ...

    @property
    def max_sequence_length(self) -> int: ...


def generator_runtime_identity() -> dict[str, Any]:
    """Bind paired archives to the generator code and core package versions."""

    package_root = Path(__file__).resolve().parents[2]
    digest = hashlib.sha256()
    for path in sorted(package_root.rglob("*.py")):
        relative = path.relative_to(package_root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")

    versions: dict[str, str | None] = {}
    for package in ("torch", "diffusers", "transformers", "peft", "safetensors"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {
        "vrl_python_tree_sha256": digest.hexdigest(),
        "python": platform.python_version(),
        "packages": versions,
    }


def seed_for(
    *,
    base_seed: int,
    prompt_index: int,
    sample_index: int,
    samples_per_prompt: int,
) -> int:
    """Return one checkpoint-independent seed for a prompt/sample cell."""

    return int(base_seed) + int(prompt_index) * int(samples_per_prompt) + int(sample_index)


def generate_one_video(
    model: DiffusionModelBase,
    *,
    prompt: str,
    seed: int,
    sampling: dict[str, Any],
) -> torch.Tensor:
    """Generate one video through the registered full-sequence model boundary."""

    encoded = model.encode_prompt(
        prompt,
        None,
        max_sequence_length=int(sampling["max_sequence_length"]),
        guidance_scale=float(sampling["guidance_scale"]),
    )
    request = DenoiseRequest(
        width=int(sampling["width"]),
        height=int(sampling["height"]),
        frame_count=int(sampling["num_frames"]),
        num_steps=int(sampling["num_steps"]),
        guidance_scale=float(sampling["guidance_scale"]),
        seed=int(seed),
        fps=int(sampling["fps"]),
    )
    state = model.prepare_sampling(request, encoded)
    generator = torch.Generator(device=state.latents.device)
    generator.manual_seed(int(seed))
    with torch.no_grad():
        if str(sampling["denoise_mode"]) == "native":
            _denoise_native(model, state)
        else:
            for step_idx, timestep in enumerate(state.timesteps):
                step_output = model.forward_step(state, step_idx)
                state.latents = sde_step_with_logprob(
                    state.scheduler,
                    step_output["noise_pred"].float(),
                    timestep.unsqueeze(0),
                    state.latents.float(),
                    generator=generator,
                    deterministic=False,
                    return_dt=False,
                    noise_level=float(sampling["noise_level"]),
                    sde_type=str(sampling["sde_type"]),
                    step_index=step_idx,
                ).prev_sample
        decoded = model.decode_latents(state.latents)
    return video_to_cthw(decoded.detach().cpu())


def generate_images(
    model: DiffusionModelBase,
    *,
    prompt: str,
    negative_prompt: str,
    seed: int,
    samples_per_prompt: int,
    sampling: ImageSampling,
    torch: ModuleType,
) -> list[Image.Image]:
    """Generate one reproducible image batch through native scheduler steps."""

    prompts = [prompt] * samples_per_prompt
    negative_prompts = [negative_prompt] * samples_per_prompt
    encoded = model.encode_prompt(
        prompts,
        negative_prompts,
        max_sequence_length=sampling.max_sequence_length,
        guidance_scale=sampling.guidance_scale,
    )
    request = DenoiseRequest(
        negative_prompt=negative_prompt,
        width=sampling.width,
        height=sampling.height,
        frame_count=1,
        num_steps=sampling.num_steps,
        guidance_scale=sampling.guidance_scale,
        seed=int(seed),
    )
    state = model.prepare_sampling(request, encoded)
    with torch.no_grad():
        _denoise_native(model, state)
    decoded = model.decode_latents(state.latents)
    return [to_pil_image(image) for image in decoded]


def _denoise_native(model: DiffusionModelBase, state: DiffusionSamplingStateBase) -> None:
    """Keep native scheduler arithmetic identical for image and video callers.

    The callers own their grad/decode contexts; models retain their configured
    forward precision while scheduler arithmetic stays in float32.
    """

    for step_idx, timestep in enumerate(state.timesteps):
        step_output = model.forward_step(state, step_idx)
        state.latents = state.scheduler.step(
            step_output["noise_pred"].float(),
            timestep,
            state.latents.float(),
            return_dict=False,
        )[0]


def video_to_cthw(video: torch.Tensor) -> torch.Tensor:
    """Normalize a decoded video to channel-first ``[C,T,H,W]``."""

    if video.ndim == 5:
        if video.shape[0] != 1:
            raise ValueError(f"expected one decoded video, got shape={tuple(video.shape)}")
        video = video[0]
    if video.ndim != 4:
        raise ValueError(f"expected decoded video rank 4/5, got shape={tuple(video.shape)}")
    if video.shape[0] in {1, 3, 4}:
        return video[:3]
    if video.shape[1] in {1, 3, 4}:
        return video[:, :3].permute(1, 0, 2, 3).contiguous()
    raise ValueError(f"cannot infer channel axis for decoded video shape={tuple(video.shape)}")


__all__ = [
    "ImageSampling",
    "generate_images",
    "generate_one_video",
    "generator_runtime_identity",
    "seed_for",
    "video_to_cthw",
]
