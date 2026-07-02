"""SD 3.5 family runtime.

Backend imports live inside the model's ``from_spec`` so the shared runtime
does not import diffusers or future native backends eagerly.
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
SD3_5_FAMILY_CAPABILITY = diffusion_family_capability("sd3_5", "t2i")


def extract_sd3_5_runtime_spec(cfg: Any, device: Any, weight_dtype: Any) -> RuntimeBuildSpec:
    """Slice the runtime-relevant subset out of a whole RL cfg."""
    return extract_runtime_spec(
        cfg,
        device,
        weight_dtype,
        task_variant="t2i",
    )


def build_sd3_5_runtime_bundle(spec: RuntimeBuildSpec) -> RuntimeBundle:
    """Thin family stub over the shared diffusion runtime builder."""
    from vrl.models.diffusion.sd3_5.model import SD3_5Model

    logger.info("Building sd3_5 runtime bundle")
    return build_diffusion_runtime_bundle(
        spec,
        model_cls=SD3_5Model,
        capability=SD3_5_FAMILY_CAPABILITY,
        memory_owner="SD3.5 VAE",
    )


def build_sd3_5_replay_runtime_bundle(spec: RuntimeBuildSpec) -> RuntimeBundle:
    """Thin family stub over the shared diffusion replay builder."""
    from vrl.models.diffusion.sd3_5.model import SD3_5ReplayModel

    logger.info(
        "Building sd3_5 replay runtime bundle from %s",
        spec.model_name_or_path,
    )
    return build_diffusion_replay_runtime_bundle(
        spec,
        replay_cls=SD3_5ReplayModel,
        transformer_classname="SD3Transformer2DModel",
        capability=SD3_5_FAMILY_CAPABILITY,
    )


def build_sd3_5_runtime_bundle_from_cfg(
    cfg: Any, device: Any, weight_dtype: Any,
) -> RuntimeBundle:
    """Outer convenience: whole-cfg → spec → bundle."""
    spec = extract_sd3_5_runtime_spec(cfg, device, weight_dtype)
    return build_sd3_5_runtime_bundle(spec)


def build_sd3_5_replay_runtime_bundle_from_cfg(
    cfg: Any,
    device: Any,
    weight_dtype: Any,
) -> RuntimeBundle:
    """Outer convenience: whole-cfg → spec → replay bundle."""
    spec = extract_sd3_5_runtime_spec(cfg, device, weight_dtype)
    return build_sd3_5_replay_runtime_bundle(spec)


class SD3_5ChunkExecutor(DiffusionChunkExecutorBase):
    """Diffusion executor for SD3.5-M text-to-image rollouts."""

    family: str = "sd3_5"
    task: str = "t2i"
    family_capability = SD3_5_FAMILY_CAPABILITY
    default_num_frames: int = 1
    default_max_sequence_length: int = 128

    def __init__(
        self,
        model: Any,  # SD3_5Model
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
        """Repeat SD3 prompt and pooled embeds across the chunk batch."""

        del generation_request, video_request, params
        chunk_g = chunk.sample_count
        chunk_encoded: dict[str, Any] = {
            "prompt_embeds": self.layout.repeat_batch(
                encoded["prompt_embeds"],
                chunk_g,
            ),
            "pooled_prompt_embeds": self.layout.repeat_batch(
                encoded["pooled_prompt_embeds"],
                chunk_g,
            ),
        }
        neg = encoded.get("negative_prompt_embeds")
        neg_pool = encoded.get("negative_pooled_prompt_embeds")
        if neg is not None:
            chunk_encoded["negative_prompt_embeds"] = self.layout.repeat_batch(
                neg,
                chunk_g,
            )
        if neg_pool is not None:
            chunk_encoded["negative_pooled_prompt_embeds"] = self.layout.repeat_batch(
                neg_pool,
                chunk_g,
            )
        return chunk_encoded


__all__ = [
    "SD3_5ChunkExecutor",
    "build_sd3_5_replay_runtime_bundle",
    "build_sd3_5_replay_runtime_bundle_from_cfg",
    "build_sd3_5_runtime_bundle",
    "build_sd3_5_runtime_bundle_from_cfg",
    "extract_sd3_5_runtime_spec",
]
