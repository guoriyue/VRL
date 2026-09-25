"""Shared image/video generation through the stepwise denoise model boundary.

This does not replace native pipeline protocols such as frozen SANA evaluation.
"""

from __future__ import annotations

import importlib.metadata
import math
import platform
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any

import torch

from vrl.generation.types import DenoiseRequest
from vrl.math.denoise.flow_matching import sde_step_with_logprob
from vrl.models.source_integrity import runtime_source_tree_sha256
from vrl.utils.media import to_pil_image
from vrl.utils.validation import require_mapping_keys

if TYPE_CHECKING:
    from PIL import Image

    from vrl.config.schema import RootConfig
    from vrl.models.steps.denoise.base import DenoiseModelBase, DenoiseSamplingStateBase


@dataclass(frozen=True, slots=True)
class ImageSampling:
    """Resolved image sampling values shared by generation and evaluation.

    One type serves the checkpoint evaluators and the generation archive:
    ``from_root`` projects the parsed config through ``resolve_eval_sampling``
    (no defaults of its own), ``from_mapping`` re-reads a persisted record and
    fails closed on missing or unknown keys, and ``to_record`` writes it back
    with keys derived from the fields.
    """

    width: int
    height: int
    num_steps: int
    guidance_scale: float
    # None for families whose sampling section declares no prompt-length knob
    # (Qwen-Image-2.1); the projection carries the key only when declared.
    max_sequence_length: int | None = None
    # Every other key the family's sampling section declares beyond the shared
    # geometry (Qwen-Image-2.1: reference_resolution, output_mode), resolved
    # by the same projection and handed to the family executor as-is.
    family: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("width", "height", "num_steps", "max_sequence_length"):
            value = getattr(self, name)
            if name == "max_sequence_length" and value is None:
                continue
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

        from vrl.scripts.eval._sampling import family_sampling_keys, resolve_eval_sampling

        sampling = resolve_eval_sampling(root, overrides=overrides)
        named = cls._named_fields()
        return cls(
            **{name: sampling[name] for name in named if name != "max_sequence_length"},
            max_sequence_length=sampling.get("max_sequence_length"),
            family={
                key: sampling[key]
                for key in family_sampling_keys(root.sampling)
                if key not in named
            },
        )

    @classmethod
    def _named_fields(cls) -> tuple[str, ...]:
        return tuple(f.name for f in fields(cls) if f.name != "family")

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any] | Any,
        *,
        what: str = "sampling",
    ) -> ImageSampling:
        """Build sampling values from one persisted record (flat, as ``to_record`` wrote it).

        The named fields must all be present; any other key is a family key
        and rides along unchanged.
        """

        if not isinstance(value, Mapping):
            raise TypeError(f"{what} must be a mapping, got {type(value).__name__}")
        named = cls._named_fields()
        missing = sorted(set(named) - set(value))
        if missing:
            raise ValueError(f"{what} missing keys: {missing}")
        return cls(
            **{name: value[name] for name in named},
            family={key: item for key, item in value.items() if key not in named},
        )

    def to_record(self) -> dict[str, Any]:
        """One flat mapping of every effective sampling value, family keys included."""

        return {**{name: getattr(self, name) for name in self._named_fields()}, **self.family}


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
        digest = runtime_source_tree_sha256(package_root, include_globs=("**/*.py",))

        versions: dict[str, str | None] = {}
        for package in ("torch", "diffusers", "transformers", "peft", "safetensors"):
            try:
                versions[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                versions[package] = None
        return cls(
            python=platform.python_version(),
            packages=versions,
            vrl_python_tree_sha256=digest,
        )

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any] | Any,
        *,
        what: str = "generator runtime",
    ) -> GeneratorRuntimeIdentity:
        """Parse one fail-closed persisted runtime record."""

        try:
            return cls(
                **require_mapping_keys(value, (field.name for field in fields(cls)), what=what)
            )
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
    model: DenoiseModelBase,
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
    model: DenoiseModelBase,
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


def _denoise_native(model: DenoiseModelBase, state: DenoiseSamplingStateBase) -> None:
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
