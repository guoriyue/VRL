"""Cosmos Predict2.5 family runtime."""

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
from vrl.models.interfaces.runtime import RuntimeBuildSpec, RuntimeBundle
from vrl.models.runtime_config import (
    extract_runtime_spec,
)
from vrl.utils.logging import init_logger

logger = init_logger(__name__)
COSMOS_PREDICT25_FAMILY_CAPABILITY = diffusion_family_capability(
    "cosmos-predict2.5",
    "t2w",
)


def extract_cosmos_predict25_runtime_spec(
    cfg: Any,
    device: Any,
    weight_dtype: Any,
) -> RuntimeBuildSpec:
    return extract_runtime_spec(
        cfg,
        device,
        weight_dtype,
        task_variant="text2world",
    )


def _model_revision_from_spec(spec: RuntimeBuildSpec) -> str | None:
    """Bundle-metadata model revision; empty cfg value reads as ``None``."""

    return (spec.model_config or {}).get("revision") or None


def _skip_text_encoder_from_spec(spec: RuntimeBuildSpec) -> bool:
    return bool((spec.model_config or {}).get("skip_text_encoder", False))


def build_cosmos_predict25_runtime_bundle(spec: RuntimeBuildSpec) -> RuntimeBundle:
    """Thin family stub over the shared diffusion runtime builder.

    The non-LoRA branch still fails loud: the shared builder calls
    ``model.apply_full_finetune()``, which Predict2.5 defines as a
    DiffusionNFT-requires-LoRA error. Historical caps dict passed verbatim.
    """
    from vrl.models.diffusion.cosmos.predict2_5.model import CosmosPredict25Model

    logger.info(
        "Building cosmos-predict2.5 runtime bundle from %s",
        spec.model_name_or_path,
    )
    return build_diffusion_runtime_bundle(
        spec,
        model_cls=CosmosPredict25Model,
        capability=COSMOS_PREDICT25_FAMILY_CAPABILITY,
        memory_owner="Cosmos Predict2.5 VAE",
        runtime_caps={"supports_diffusion_nft": True},
        extra_metadata=lambda model, spec: {
            "model_revision": _model_revision_from_spec(spec),
            "skip_text_encoder": _skip_text_encoder_from_spec(spec),
        },
    )


def build_cosmos_predict25_replay_runtime_bundle(spec: RuntimeBuildSpec) -> RuntimeBundle:
    """Thin family stub over the shared diffusion replay builder.

    Family preamble guards the NFT constraint BEFORE the transformer loads:
    DiffusionNFT needs the trainable ``default`` + frozen ``previous`` adapters,
    which only exist on the LoRA path (the model's own ``apply_full_finetune``
    raises the same message, but only after paying the full transformer load).
    ``scheduler_classname`` mirrors the rollout pipeline's shipped UniPC so
    replay log-probs recompute under the same schedule.
    """

    from vrl.models.diffusion.cosmos.predict2_5.model import CosmosPredict25ReplayModel

    if not spec.use_lora:
        raise RuntimeError(
            "Cosmos Predict2.5 DiffusionNFT requires LoRA with default+previous "
            "adapters; set model.use_lora=true.",
        )

    logger.info(
        "Building cosmos-predict2.5 replay runtime bundle from %s",
        spec.model_name_or_path,
    )
    return build_diffusion_replay_runtime_bundle(
        spec,
        replay_cls=CosmosPredict25ReplayModel,
        transformer_classname="CosmosTransformer3DModel",
        scheduler_classname="UniPCMultistepScheduler",
        capability=COSMOS_PREDICT25_FAMILY_CAPABILITY,
        runtime_caps={"supports_diffusion_nft": True},
        extra_metadata=lambda model, spec: {
            "model_revision": _model_revision_from_spec(spec),
            "skip_text_encoder": _skip_text_encoder_from_spec(spec),
        },
    )


def build_cosmos_predict25_runtime_bundle_from_cfg(
    cfg: Any,
    device: Any,
    weight_dtype: Any,
) -> RuntimeBundle:
    spec = extract_cosmos_predict25_runtime_spec(cfg, device, weight_dtype)
    return build_cosmos_predict25_runtime_bundle(spec)


def build_cosmos_predict25_replay_runtime_bundle_from_cfg(
    cfg: Any,
    device: Any,
    weight_dtype: Any,
) -> RuntimeBundle:
    spec = extract_cosmos_predict25_runtime_spec(cfg, device, weight_dtype)
    return build_cosmos_predict25_replay_runtime_bundle(spec)


class CosmosPredict25ChunkExecutor(DiffusionChunkExecutorBase):
    """Diffusion executor for Cosmos Predict2.5 text-to-world rollouts."""

    family: str = "cosmos-predict2.5"
    task: str = "t2w"
    family_capability = COSMOS_PREDICT25_FAMILY_CAPABILITY
    default_num_frames: int = 93
    default_fps: int | None = 16
    default_max_sequence_length: int = 512

    def __init__(
        self,
        model: Any,
        *,
        samples_per_chunk: int = 1,
    ) -> None:
        self.model = model
        self.default_samples_per_chunk = max(1, int(samples_per_chunk))

    def encode_prompt_for_chunk(
        self,
        *,
        generation_request: GenerationRequest,
        video_request: VideoGenerationRequest,
        params: DiffusionSamplingParams,
        chunk: SampleChunk,
    ) -> dict[str, Any]:
        del generation_request
        return self.model.encode_prompt(
            chunk.prompt,
            video_request.negative_prompt or None,
            max_sequence_length=params.base.max_sequence_length,
            guidance_scale=params.base.guidance_scale,
        )

    def build_chunk_encoded(
        self,
        *,
        encoded: dict[str, Any],
        generation_request: GenerationRequest,
        video_request: VideoGenerationRequest,
        params: DiffusionSamplingParams,
        chunk: SampleChunk,
    ) -> dict[str, Any]:
        del generation_request, video_request, params
        return {
            key: self.layout.repeat_batch(value, chunk.sample_count)
            for key, value in encoded.items()
        }


__all__ = [
    "CosmosPredict25ChunkExecutor",
    "build_cosmos_predict25_replay_runtime_bundle",
    "build_cosmos_predict25_replay_runtime_bundle_from_cfg",
    "build_cosmos_predict25_runtime_bundle",
    "build_cosmos_predict25_runtime_bundle_from_cfg",
    "extract_cosmos_predict25_runtime_spec",
]
