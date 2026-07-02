"""FLUX.1 family runtime.

Backend imports live inside the model's ``from_spec`` so the shared runtime does
not import diffusers eagerly. Mirrors the sd3_5 runtime four-builder shape.
"""

from __future__ import annotations

from typing import Any

from vrl.generation.diffusion import (
    DiffusionChunkExecutorBase,
    DiffusionSamplingParams,
)
from vrl.generation.diffusion.layout import VideoGenerationRequest
from vrl.generation.execution.chunks import SampleChunk
from vrl.generation.types import GenerationRequest
from vrl.models.diffusion.build import (
    build_diffusion_replay_runtime_bundle,
    build_diffusion_runtime_bundle,
)
from vrl.models.diffusion.capabilities import diffusion_family_capability
from vrl.models.interfaces.runtime import (
    RuntimeBuildSpec,
    RuntimeBundle,
)
from vrl.models.runtime_config import (
    extract_runtime_spec,
)
from vrl.utils.logging import init_logger

logger = init_logger(__name__)
FLUX_FAMILY_CAPABILITY = diffusion_family_capability("flux", "t2i")


def extract_flux_runtime_spec(cfg: Any, device: Any, weight_dtype: Any) -> RuntimeBuildSpec:
    """Slice the runtime-relevant subset out of a whole RL cfg."""
    return extract_runtime_spec(
        cfg,
        device,
        weight_dtype,
        task_variant="t2i",
    )


def build_flux_runtime_bundle(
    spec: RuntimeBuildSpec,
    *,
    attach_previous_adapter: bool = False,
) -> RuntimeBundle:
    """Thin family stub over the shared diffusion runtime builder.

    ``attach_previous_adapter`` is the DiffusionNFT-only switch: when set, build a
    frozen ``previous`` LoRA adapter mirroring ``default`` (the NFT previous-policy
    forward), routed through the shared builder's ``after_lora`` hook. Left False
    for GRPO so its build path is bit-for-bit unchanged.
    """
    from vrl.models.diffusion.flux.model import FluxModel

    logger.info("Building flux runtime bundle")

    after_lora = None
    if attach_previous_adapter:

        def after_lora(model: Any, spec: RuntimeBuildSpec) -> None:
            model.attach_previous_policy_adapter(spec)
            logger.info("Attached frozen DiffusionNFT `previous` LoRA adapter")

    return build_diffusion_runtime_bundle(
        spec,
        model_cls=FluxModel,
        capability=FLUX_FAMILY_CAPABILITY,
        memory_owner="FLUX VAE",
        after_lora=after_lora,
    )


def build_flux_replay_runtime_bundle(
    spec: RuntimeBuildSpec,
    *,
    attach_previous_adapter: bool = False,
) -> RuntimeBundle:
    """Thin family stub over the shared diffusion replay builder.

    This is the model that actually runs the DiffusionNFT loss (the trainer
    optimizes it), so ``attach_previous_adapter`` must be set here too. FLUX also
    needs its dynamic-shift replay timesteps set right after construction; both
    ride the shared builder's ``after_construct`` / ``after_lora`` hooks so the
    generic body stays family-agnostic.
    """

    from vrl.models.diffusion.flux.model import FluxReplayModel

    logger.info(
        "Building flux replay runtime bundle from %s",
        spec.model_name_or_path,
    )

    def after_construct(model: Any, spec: RuntimeBuildSpec) -> None:
        # FLUX's dynamic-shifting scheduler was loaded WITHOUT timesteps (mu
        # unknown in the generic loader). The replay SDE log-prob math reads
        # scheduler.sigmas + index_for_timestep, so the replay scheduler must
        # carry the SAME mu-shifted schedule the rollout used. Resolution is
        # fixed per run, so derive the packed image_seq_len from it and set the
        # dynamic timesteps now — identical to the rollout's prepare_sampling.
        # (debug.first_step asserts old==new log-prob, so any drift here surfaces
        # immediately.) FLUX packs an 8x VAE + 2x2 patch grid, so
        # seq_len = (H // 16) * (W // 16).
        sampling = spec.sampling_config or {}
        num_steps = spec.num_steps
        height, width = sampling.get("height"), sampling.get("width")
        if num_steps is not None and height and width:
            image_seq_len = (int(height) // 16) * (int(width) // 16)
            model._set_dynamic_timesteps(int(num_steps), image_seq_len, spec.device)

    after_lora = None
    if attach_previous_adapter:

        def after_lora(model: Any, spec: RuntimeBuildSpec) -> None:
            model.attach_previous_policy_adapter(spec)
            logger.info("Attached frozen DiffusionNFT `previous` LoRA adapter (replay)")

    return build_diffusion_replay_runtime_bundle(
        spec,
        replay_cls=FluxReplayModel,
        transformer_classname="FluxTransformer2DModel",
        capability=FLUX_FAMILY_CAPABILITY,
        after_construct=after_construct,
        after_lora=after_lora,
    )


def build_flux_runtime_bundle_from_cfg(
    cfg: Any, device: Any, weight_dtype: Any,
) -> RuntimeBundle:
    """Outer convenience: whole-cfg → spec → bundle."""
    spec = extract_flux_runtime_spec(cfg, device, weight_dtype)
    return build_flux_runtime_bundle(spec)


def build_flux_replay_runtime_bundle_from_cfg(
    cfg: Any,
    device: Any,
    weight_dtype: Any,
) -> RuntimeBundle:
    """Outer convenience: whole-cfg → spec → replay bundle."""
    spec = extract_flux_runtime_spec(cfg, device, weight_dtype)
    return build_flux_replay_runtime_bundle(spec)


class FluxChunkExecutor(DiffusionChunkExecutorBase):
    """Diffusion executor for FLUX.1 text-to-image rollouts."""

    family: str = "flux"
    task: str = "t2i"
    family_capability = FLUX_FAMILY_CAPABILITY
    default_num_frames: int = 1
    default_max_sequence_length: int = 512

    def __init__(
        self,
        model: Any,  # FluxModel
        *,
        samples_per_chunk: int = 8,
    ) -> None:
        self.model = model
        self.default_samples_per_chunk = max(1, int(samples_per_chunk))

    def build_chunk_encoded(
        self,
        *,
        encoded: dict[str, Any],
        generation_request: GenerationRequest,
        video_request: VideoGenerationRequest,
        params: DiffusionSamplingParams,
        chunk: SampleChunk,
    ) -> dict[str, Any]:
        """Repeat FLUX prompt/pooled embeds across the chunk batch.

        ``text_ids`` is batch-shared (shape ``[seq, 3]``, no batch dim), so it is
        passed through unrepeated — repeating it would corrupt its leading dim.
        """

        del generation_request, video_request, params
        chunk_g = chunk.sample_count
        return {
            "prompt_embeds": self.layout.repeat_batch(
                encoded["prompt_embeds"],
                chunk_g,
            ),
            "pooled_prompt_embeds": self.layout.repeat_batch(
                encoded["pooled_prompt_embeds"],
                chunk_g,
            ),
            "text_ids": encoded["text_ids"],
        }


__all__ = [
    "FluxChunkExecutor",
    "build_flux_replay_runtime_bundle",
    "build_flux_replay_runtime_bundle_from_cfg",
    "build_flux_runtime_bundle",
    "build_flux_runtime_bundle_from_cfg",
    "extract_flux_runtime_spec",
]
