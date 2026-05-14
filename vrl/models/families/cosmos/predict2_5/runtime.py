"""Cosmos Predict2.5 family runtime."""

from __future__ import annotations

import logging
from typing import Any

from vrl.engine.core.types import GenerationRequest
from vrl.engine.diffusion import (
    DiffusionGenerationSpec,
    DiffusionPipelineExecutorBase,
    repeat_tensor_batch,
)
from vrl.engine.execution.microbatching import MicroBatchPlan
from vrl.models.interfaces.diffusion_policy import VideoGenerationRequest
from vrl.models.interfaces.runtime import RuntimeBuildSpec, RuntimeBundle
from vrl.models.registry import register_diffusion_model

logger = logging.getLogger(__name__)


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
    from vrl.models.families.cosmos.predict2_5.policy import CosmosPredict25Policy

    logger.info(
        "Building cosmos-predict2.5 runtime bundle (backend=diffusers) from %s",
        spec.model_name_or_path,
    )
    adapter = CosmosPredict25Policy.from_spec(spec)
    if spec.use_lora:
        adapter.apply_lora(spec)
    else:
        adapter.enable_full_finetune()

    compile_cfg = (spec.extra or {}).get("torch_compile") or {}
    if compile_cfg.get("enable"):
        adapter.torch_compile_transformer(compile_cfg["mode"])

    num_steps = (spec.scheduler_config or {}).get("num_steps")
    if num_steps is not None:
        adapter.set_num_steps(num_steps)

    return RuntimeBundle(
        policy=adapter,
        trainable_modules=adapter.trainable_modules,
        scheduler=adapter.scheduler,
        backend_kind="diffusers",
        backend_handle=adapter.backend_handle,
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
            "runtime_role": "full_generation_policy",
            "loads_full_generation_modules": True,
            "requires_minimal_replay_loader": True,
        },
    )


def build_cosmos_predict25_runtime_bundle_from_cfg(
    cfg: Any,
    device: Any,
    weight_dtype: Any,
) -> RuntimeBundle:
    spec = extract_cosmos_predict25_runtime_spec(cfg, device, weight_dtype)
    return build_cosmos_predict25_runtime_bundle(spec)


"""Cosmos Predict2.5 diffusion executor."""


class CosmosPredict25PipelineExecutor(DiffusionPipelineExecutorBase):
    """Diffusion executor for Cosmos Predict2.5 text-to-world rollouts."""

    family: str = "cosmos-predict2.5"
    task: str = "t2w"
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
        spec: DiffusionGenerationSpec,
        chunk: MicroBatchPlan,
    ) -> dict[str, Any]:
        del generation_request
        return self.model.encode_prompt(
            chunk.prompt,
            video_request.negative_prompt or None,
            max_sequence_length=spec.base.max_sequence_length,
            guidance_scale=spec.base.guidance_scale,
        )

    def build_chunk_encoded(
        self,
        *,
        encoded: dict[str, Any],
        generation_request: GenerationRequest,
        video_request: VideoGenerationRequest,
        spec: DiffusionGenerationSpec,
        chunk: MicroBatchPlan,
    ) -> dict[str, Any]:
        del generation_request, video_request, spec
        return {
            key: repeat_tensor_batch(value, chunk.sample_count)
            for key, value in encoded.items()
        }


COSMOS_PREDICT25_MODEL = register_diffusion_model(
    "cosmos-predict2.5",
    task="t2w",
    aliases=("cosmos_predict25", "cosmos_predict2_5", "cosmos_predict2_5_2b"),
    executor_cls=CosmosPredict25PipelineExecutor,
    runtime_builder=build_cosmos_predict25_runtime_bundle,
    runtime_spec_extractor=extract_cosmos_predict25_runtime_spec,
    collector_config_cls="vrl.rollouts.collector.configs:CosmosPredict2CollectorConfig",
    request_prefix="cosmos-predict2.5",
    default_task_type="text_to_video",
    error_prefix="Cosmos Predict2.5",
    video=True,
    extra_sampling_fields=("fps",),
    gatherer_kwargs={"model_family": "cosmos-predict2.5"},
)


__all__ = [
    "COSMOS_PREDICT25_MODEL",
    "CosmosPredict25PipelineExecutor",
    "build_cosmos_predict25_runtime_bundle",
    "build_cosmos_predict25_runtime_bundle_from_cfg",
    "extract_cosmos_predict25_runtime_spec",
]
