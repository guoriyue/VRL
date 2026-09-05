"""Shared image/video generation through the stepwise denoise model boundary.

This does not replace native pipeline protocols such as the frozen SANA
evaluation, or token-autoregressive generation.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import math
import platform
from collections.abc import Mapping
from dataclasses import dataclass, fields
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any

import torch

from vrl.generation.types import DenoiseRequest
from vrl.math.denoise.flow_matching import sde_step_with_logprob
from vrl.utils.media import to_pil_image

if TYPE_CHECKING:
    from PIL import Image

    from vrl.config.schema import RootConfig
    from vrl.models.steps.denoise.base import DiffusionModelBase, DiffusionSamplingStateBase


@dataclass(frozen=True, slots=True)
class ImageSampling:
    """Resolved image sampling values shared by generation and evaluation.

    One type serves the checkpoint evaluators and the Anima generation archive:
    ``from_root`` projects the parsed config through ``resolve_eval_sampling``
    (no defaults of its own), ``from_mapping`` re-reads a persisted record and
    fails closed on missing or unknown keys, and ``to_record`` writes it back
    with keys derived from the fields.
    """

    width: int
    height: int
    num_steps: int
    guidance_scale: float
    max_sequence_length: int

    def __post_init__(self) -> None:
        for name in ("width", "height", "num_steps", "max_sequence_length"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"sampling.{name} must be a positive integer")
        guidance_scale = self.guidance_scale
        if (
            isinstance(guidance_scale, bool)
            or not isinstance(guidance_scale, (int, float))
            or not math.isfinite(guidance_scale)
            or guidance_scale < 0
        ):
            raise ValueError("sampling.guidance_scale must be finite and non-negative")
        object.__setattr__(self, "guidance_scale", float(guidance_scale))

    @classmethod
    def from_root(
        cls,
        root: RootConfig,
        *,
        overrides: Mapping[str, Any] | None = None,
    ) -> ImageSampling:
        """Project the parsed config (CLI ``overrides`` on top) into the image fields."""

        from vrl.scripts.eval._sampling import resolve_eval_sampling

        sampling = resolve_eval_sampling(root, overrides=overrides)
        return cls(**{field.name: sampling[field.name] for field in fields(cls)})

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any] | Any,
        *,
        what: str = "sampling",
    ) -> ImageSampling:
        """Parse one fail-closed persisted sampling record."""

        if not isinstance(value, Mapping):
            raise TypeError(f"{what} must be a mapping")
        expected = {field.name for field in fields(cls)}
        missing = sorted(expected - set(value))
        unknown = sorted(set(value) - expected)
        if missing or unknown:
            raise ValueError(f"invalid {what} fields: missing={missing} unknown={unknown}")
        return cls(**{name: value[name] for name in expected})

    def to_record(self) -> dict[str, int | float]:
        """Serialize with keys derived from the typed source of truth."""

        return {field.name: getattr(self, field.name) for field in fields(self)}


@dataclass(frozen=True, slots=True)
class GeneratorRuntimeIdentity:
    """Bind paired archives to the generator code and core package versions.

    ``capture()`` is the only producer; archives re-read it with
    ``from_mapping`` and compare whole values, so a runtime drift between
    preflight and generation, or between two paired archives, fails closed.
    """

    python: str
    packages: dict[str, str | None]
    # Digest over every vrl/**/*.py file. Broader than what produces pixels
    # (reward and evaluator code count too), so paired evaluation compares only
    # ``python`` + ``packages`` and leaves this to causal audits.
    vrl_python_tree_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.python, str) or not self.python:
            raise ValueError("generator runtime python must be a non-empty string")
        if not isinstance(self.packages, Mapping) or not self.packages:
            raise ValueError("generator runtime packages must be a non-empty mapping")
        digest = self.vrl_python_tree_sha256
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError(
                "generator runtime requires a lowercase hexadecimal vrl_python_tree_sha256",
            )
        object.__setattr__(self, "packages", dict(self.packages))

    @classmethod
    def capture(cls) -> GeneratorRuntimeIdentity:
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
        return cls(
            python=platform.python_version(),
            packages=versions,
            vrl_python_tree_sha256=digest.hexdigest(),
        )

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any] | Any,
        *,
        what: str = "generator runtime",
    ) -> GeneratorRuntimeIdentity:
        """Parse one fail-closed persisted runtime record."""

        if not isinstance(value, Mapping):
            raise TypeError(f"{what} must be a mapping")
        expected = {field.name for field in fields(cls)}
        missing = sorted(expected - set(value))
        unknown = sorted(set(value) - expected)
        if missing or unknown:
            raise ValueError(f"invalid {what} fields: missing={missing} unknown={unknown}")
        try:
            return cls(**{name: value[name] for name in expected})
        except ValueError as error:
            raise ValueError(f"{what}: {error}") from error

    def to_record(self) -> dict[str, Any]:
        return {field.name: getattr(self, field.name) for field in fields(self)}


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
    "GeneratorRuntimeIdentity",
    "ImageSampling",
    "generate_images",
    "generate_one_video",
    "seed_for",
    "video_to_cthw",
]
