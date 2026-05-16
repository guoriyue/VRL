"""Cosmos Predict2.5 family runtime."""

from __future__ import annotations

import logging
from typing import Any

from vrl.engine.core.types import GenerationRequest
from vrl.engine.diffusion import (
    DiffusionPipelineExecutorBase,
    DiffusionSamplingParams,
)
from vrl.engine.diffusion.layout import VideoGenerationRequest
from vrl.engine.execution.microbatching import MicroBatchSample
from vrl.models.capability_builders import diffusion_family_capability
from vrl.models.interfaces.runtime import RuntimeBuildSpec, RuntimeBundle
from vrl.models.replay_loading import (
    full_generation_bundle_metadata,
    load_diffusers_scheduler_component,
    load_diffusers_transformer_component,
    minimal_replay_bundle_metadata,
)

logger = logging.getLogger(__name__)
COSMOS_PREDICT25_FAMILY_CAPABILITY = diffusion_family_capability(
    "cosmos-predict2.5",
    "t2w",
)


def extract_cosmos_predict25_runtime_spec(
    cfg: Any,
    device: Any,
    weight_dtype: Any,
) -> RuntimeBuildSpec:
    lora_cfg: dict[str, Any] | None = None
    lora_path: str | None = None
    if cfg.model.use_lora:
        lora_path = cfg.model.lora.path or None
        lora_cfg = {
            "rank": int(cfg.model.lora.rank),
            "alpha": int(cfg.model.lora.alpha),
            "target_modules": list(cfg.model.lora.target_modules),
        }

    extra: dict[str, Any] = {}
    revision = getattr(cfg.model, "revision", None)
    if revision:
        extra["model_revision"] = str(revision)
    if bool(getattr(cfg.model, "skip_text_encoder", False)):
        extra["skip_text_encoder"] = True
    if getattr(cfg.model, "torch_compile", None) is not None and cfg.model.torch_compile.enable:
        extra["torch_compile"] = {
            "enable": True,
            "mode": cfg.model.torch_compile.mode,
        }

    return RuntimeBuildSpec(
        model_name_or_path=cfg.model.path,
        device=device,
        dtype=weight_dtype,
        backend_preference=("diffusers",),
        task_variant="text2world",
        use_lora=bool(cfg.model.use_lora),
        lora_path=lora_path,
        lora_config=lora_cfg,
        scheduler_config={"num_steps": int(cfg.sampling.num_steps)},
        extra=extra,
    )


def build_cosmos_predict25_runtime_bundle(spec: RuntimeBuildSpec) -> RuntimeBundle:
    from vrl.models.families.cosmos.predict2_5.model import CosmosPredict25Model

    logger.info(
        "Building cosmos-predict2.5 runtime bundle (backend=diffusers) from %s",
        spec.model_name_or_path,
    )
    model = CosmosPredict25Model.from_spec(spec)
    if spec.use_lora:
        model.apply_lora(spec)
    else:
        model.enable_full_finetune()

    compile_cfg = (spec.extra or {}).get("torch_compile") or {}
    if compile_cfg.get("enable"):
        model.torch_compile_transformer(compile_cfg["mode"])

    num_steps = (spec.scheduler_config or {}).get("num_steps")
    if num_steps is not None:
        model.set_num_steps(num_steps)

    return RuntimeBundle(
        model=model,
        trainable_modules=model.trainable_modules,
        scheduler=model.scheduler,
        backend_kind="diffusers",
        backend_handle=model.backend_handle,
        runtime_caps={
            "supports_stepwise": True,
            "supports_cfg": True,
            "supports_batched_decode": True,
            "supports_diffusion_nft": True,
        },
        metadata={
            "model_path": spec.model_name_or_path,
            "task_variant": spec.task_variant,
            "dtype": str(spec.dtype),
            "use_lora": spec.use_lora,
            "model_revision": (spec.extra or {}).get("model_revision"),
            "skip_text_encoder": bool((spec.extra or {}).get("skip_text_encoder", False)),
            **full_generation_bundle_metadata(),
        },
    )


def build_cosmos_predict25_replay_runtime_bundle(spec: RuntimeBuildSpec) -> RuntimeBundle:
    """Build the trainer replay bundle without Cosmos2.5 text/VAE modules."""

    from vrl.models.families.cosmos.predict2_5.model import CosmosPredict25ReplayModel

    logger.info(
        "Building cosmos-predict2.5 replay runtime bundle (backend=diffusers) from %s",
        spec.model_name_or_path,
    )
    model = CosmosPredict25ReplayModel(
        transformer=load_diffusers_transformer_component(
            spec,
            "CosmosTransformer3DModel",
        ),
        scheduler=load_diffusers_scheduler_component(
            spec,
            "UniPCMultistepScheduler",
        ),
        device=spec.device,
    )
    if spec.use_lora:
        model.apply_lora(spec)
    else:
        model.enable_full_finetune()

    compile_cfg = (spec.extra or {}).get("torch_compile") or {}
    if compile_cfg.get("enable"):
        model.torch_compile_transformer(compile_cfg["mode"])

    return RuntimeBundle(
        model=model,
        trainable_modules=model.trainable_modules,
        scheduler=model.scheduler,
        backend_kind="diffusers",
        backend_handle=None,
        runtime_caps={
            "supports_stepwise": True,
            "supports_cfg": True,
            "supports_batched_decode": False,
            "supports_diffusion_nft": True,
        },
        metadata={
            "model_path": spec.model_name_or_path,
            "task_variant": spec.task_variant,
            "dtype": str(spec.dtype),
            "use_lora": spec.use_lora,
            "model_revision": (spec.extra or {}).get("model_revision"),
            "skip_text_encoder": bool((spec.extra or {}).get("skip_text_encoder", False)),
            **minimal_replay_bundle_metadata(
                replay_modules=("transformer", "scheduler"),
                generation_only_modules=("text_encoder", "vae", "pipeline"),
            ),
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


"""Cosmos Predict2.5 diffusion executor."""


class CosmosPredict25PipelineExecutor(DiffusionPipelineExecutorBase):
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
        chunk: MicroBatchSample,
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
        chunk: MicroBatchSample,
    ) -> dict[str, Any]:
        del generation_request, video_request, params
        return {
            key: self.layout.repeat_batch(value, chunk.sample_count)
            for key, value in encoded.items()
        }


__all__ = [
    "CosmosPredict25PipelineExecutor",
    "build_cosmos_predict25_replay_runtime_bundle",
    "build_cosmos_predict25_replay_runtime_bundle_from_cfg",
    "build_cosmos_predict25_runtime_bundle",
    "build_cosmos_predict25_runtime_bundle_from_cfg",
    "extract_cosmos_predict25_runtime_spec",
]
