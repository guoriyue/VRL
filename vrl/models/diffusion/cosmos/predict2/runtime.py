"""Cosmos Predict2 family runtime.

Backend imports live inside the model's ``from_spec`` so the shared runtime
does not import diffusers or cosmos-library backends eagerly.
"""

from __future__ import annotations

from typing import Any

from vrl.generation.diffusion import (
    DiffusionChunkExecutorBase,
    DiffusionSamplingParams,
)
from vrl.generation.diffusion.executor import ReferenceConditionedChunks
from vrl.generation.diffusion.layout import VideoGenerationRequest
from vrl.generation.execution.chunks import SampleChunk
from vrl.generation.types import GenerationRequest
from vrl.models.diffusion.build import (
    build_diffusion_replay_runtime_bundle,
    build_diffusion_runtime_bundle,
)
from vrl.models.diffusion.capabilities import diffusion_family_capability
from vrl.models.interfaces.runtime import RuntimeBuildSpec, RuntimeBundle
from vrl.models.runtime_config import (
    extract_runtime_spec,
)
from vrl.utils.logging import init_logger
from vrl.utils.media import load_reference_image

logger = init_logger(__name__)
COSMOS_PREDICT2_FAMILY_CAPABILITY = diffusion_family_capability(
    "cosmos-predict2",
    "v2w",
    supports_reference_conditioning=True,
)


def extract_cosmos_predict2_runtime_spec(
    cfg: Any, device: Any, weight_dtype: Any,
) -> RuntimeBuildSpec:
    """Slice the runtime-relevant subset out of a whole RL cfg."""
    return extract_runtime_spec(
        cfg,
        device,
        weight_dtype,
        task_variant="video2world",
    )


def build_cosmos_predict2_runtime_bundle(
    spec: RuntimeBuildSpec,
) -> RuntimeBundle:
    """Thin family stub over the shared diffusion runtime builder.

    Passes the historical caps dict verbatim (no ``family_capability`` key —
    the worker falls back to the registry-declared capability) and merges the
    v2w ``reference_image`` into metadata.
    """
    from vrl.models.diffusion.cosmos.predict2.model import CosmosPredict2Model

    logger.info(
        "Building cosmos-predict2 runtime bundle from %s",
        spec.model_name_or_path,
    )
    return build_diffusion_runtime_bundle(
        spec,
        model_cls=CosmosPredict2Model,
        capability=COSMOS_PREDICT2_FAMILY_CAPABILITY,
        memory_owner="Cosmos Predict2 VAE",
        runtime_caps={"supports_reference_conditioning": True},
    )


def build_cosmos_predict2_replay_runtime_bundle(
    spec: RuntimeBuildSpec,
) -> RuntimeBundle:
    """Thin family stub over the shared diffusion replay builder."""

    from vrl.models.diffusion.cosmos.predict2.model import CosmosPredict2ReplayModel

    logger.info(
        "Building cosmos-predict2 replay runtime bundle from %s",
        spec.model_name_or_path,
    )
    return build_diffusion_replay_runtime_bundle(
        spec,
        replay_cls=CosmosPredict2ReplayModel,
        transformer_classname="CosmosTransformer3DModel",
        capability=COSMOS_PREDICT2_FAMILY_CAPABILITY,
        runtime_caps={"supports_reference_conditioning": True},
    )


def build_cosmos_predict2_runtime_bundle_from_cfg(
    cfg: Any, device: Any, weight_dtype: Any,
) -> RuntimeBundle:
    """Outer convenience: whole-cfg → spec → bundle."""
    spec = extract_cosmos_predict2_runtime_spec(cfg, device, weight_dtype)
    return build_cosmos_predict2_runtime_bundle(spec)


def build_cosmos_predict2_replay_runtime_bundle_from_cfg(
    cfg: Any,
    device: Any,
    weight_dtype: Any,
) -> RuntimeBundle:
    """Outer convenience: whole-cfg → spec → replay bundle."""
    spec = extract_cosmos_predict2_runtime_spec(cfg, device, weight_dtype)
    return build_cosmos_predict2_replay_runtime_bundle(spec)


class CosmosChunkExecutor(ReferenceConditionedChunks, DiffusionChunkExecutorBase):
    """Diffusion executor for Cosmos Predict2 Video2World rollouts."""

    family: str = "cosmos-predict2"
    task: str = "v2w"
    family_capability = COSMOS_PREDICT2_FAMILY_CAPABILITY
    default_num_frames: int = 93
    default_fps: int | None = 16
    default_max_sequence_length: int = 512
    include_max_sequence_length_extra: bool = False

    def __init__(
        self,
        model: Any,  # CosmosPredict2Model
        *,
        reference_image: Any = None,
        samples_per_chunk: int = 8,
    ) -> None:
        self.model = model
        self.reference_image = load_reference_image(reference_image)
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
        """Repeat Cosmos text embeds and pass reference image through unchanged."""

        del video_request, params
        chunk_g = chunk.sample_count
        reference_image = self._reference_image_for_request(generation_request)
        chunk_encoded: dict[str, Any] = {
            "prompt_embeds": self.layout.repeat_batch(
                encoded["prompt_embeds"],
                chunk_g,
            ),
            "reference_image": encoded.get("reference_image", reference_image),
        }
        neg = encoded.get("negative_prompt_embeds")
        if neg is not None:
            chunk_encoded["negative_prompt_embeds"] = self.layout.repeat_batch(
                neg,
                chunk_g,
            )
        else:
            chunk_encoded["negative_prompt_embeds"] = None
        return chunk_encoded


__all__ = [
    "CosmosChunkExecutor",
    "build_cosmos_predict2_replay_runtime_bundle",
    "build_cosmos_predict2_replay_runtime_bundle_from_cfg",
    "build_cosmos_predict2_runtime_bundle",
    "build_cosmos_predict2_runtime_bundle_from_cfg",
    "extract_cosmos_predict2_runtime_spec",
]
