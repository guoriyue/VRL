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
from vrl.models.diffusion.capabilities import diffusion_family_capability
from vrl.models.diffusion.common.vae_decode_memory import (
    apply_generation_memory_policy,
)
from vrl.models.interfaces.runtime import RuntimeBuildSpec, RuntimeBundle
from vrl.models.loader import (
    apply_rollout_quantization,
    load_diffusers_scheduler,
    load_diffusers_transformer,
)
from vrl.models.replay_loading import (
    full_generation_bundle_metadata,
    minimal_replay_bundle_metadata,
)
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
    from vrl.models.diffusion.cosmos.predict2_5.model import CosmosPredict25Model

    logger.info(
        "Building cosmos-predict2.5 runtime bundle from %s",
        spec.model_name_or_path,
    )
    use_lora = spec.use_lora
    model = CosmosPredict25Model.from_spec(spec)
    if use_lora:
        model.apply_lora(spec)
    else:
        model.enable_full_finetune()

    apply_rollout_quantization(model, spec)

    compile_cfg = spec.torch_compile or {}
    if compile_cfg.get("enable"):
        model.torch_compile_transformer(compile_cfg["mode"])

    num_steps = spec.num_steps
    if num_steps is not None:
        model.set_num_steps(num_steps)

    return RuntimeBundle(
        model=model,
        trainable_modules=model.trainable_modules,
        scheduler=model.scheduler,
        backend_handle=model.backend_handle,
        runtime_caps={
            "supports_diffusion_nft": True,
        },
        metadata={
            "model_path": spec.model_name_or_path,
            "task_variant": spec.task_variant,
            "dtype": str(spec.dtype),
            "use_lora": use_lora,
            "model_revision": _model_revision_from_spec(spec),
            "skip_text_encoder": _skip_text_encoder_from_spec(spec),
            **full_generation_bundle_metadata(),
            **apply_generation_memory_policy(
                model,
                memory_config=getattr(spec, "memory", None),
                owner="Cosmos Predict2.5 VAE",
            ),
        },
    )


def build_cosmos_predict25_replay_runtime_bundle(spec: RuntimeBuildSpec) -> RuntimeBundle:
    """Build the trainer replay bundle without Cosmos2.5 text/VAE modules."""

    from vrl.models.diffusion.cosmos.predict2_5.model import CosmosPredict25ReplayModel

    logger.info(
        "Building cosmos-predict2.5 replay runtime bundle from %s",
        spec.model_name_or_path,
    )
    model = CosmosPredict25ReplayModel(
        transformer=load_diffusers_transformer(
            spec,
            "CosmosTransformer3DModel",
        ),
        scheduler=load_diffusers_scheduler(
            spec,
            "UniPCMultistepScheduler",
        ),
        device=spec.device,
    )
    use_lora = spec.use_lora
    if use_lora:
        model.apply_lora(spec)
    else:
        model.enable_full_finetune()

    compile_cfg = spec.torch_compile or {}
    if compile_cfg.get("enable"):
        model.torch_compile_transformer(compile_cfg["mode"])

    return RuntimeBundle(
        model=model,
        trainable_modules=model.trainable_modules,
        scheduler=model.scheduler,
        backend_handle=None,
        runtime_caps={
            "supports_diffusion_nft": True,
        },
        metadata={
            "model_path": spec.model_name_or_path,
            "task_variant": spec.task_variant,
            "dtype": str(spec.dtype),
            "use_lora": use_lora,
            "model_revision": _model_revision_from_spec(spec),
            "skip_text_encoder": _skip_text_encoder_from_spec(spec),
            **minimal_replay_bundle_metadata(),
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
        sample_batch_size: int = 1,
    ) -> None:
        self.model = model
        self.default_sample_batch_size = max(1, int(sample_batch_size))

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
