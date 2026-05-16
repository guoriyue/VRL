"""SD 3.5 family runtime.

The runtime picks the backend model class by ``spec.backend_preference``.
Backend imports live inside the model's ``from_spec`` so the shared runtime
does not import diffusers or future native backends eagerly.
"""

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
    apply_lora_to_transformer,
    compile_transformer,
    enable_transformer_full_finetune,
    full_generation_bundle_metadata,
    load_diffusers_transformer_component,
    load_flow_match_scheduler_component,
    minimal_replay_bundle_metadata,
)

logger = logging.getLogger(__name__)
SD3_5_FAMILY_CAPABILITY = diffusion_family_capability("sd3_5", "t2i")

_MODEL_BY_BACKEND: dict[str, str] = {
    "diffusers": "vrl.models.families.sd3_5.model:SD3_5Model",
}


def _resolve_model_cls(backend: str) -> type:
    import importlib

    if backend not in _MODEL_BY_BACKEND:
        raise NotImplementedError(
            f"sd3_5 has no model for backend={backend!r}; "
            f"registered: {sorted(_MODEL_BY_BACKEND)}",
        )
    spec = _MODEL_BY_BACKEND[backend]
    mod_path, cls_name = spec.rsplit(":", 1)
    return getattr(importlib.import_module(mod_path), cls_name)


def extract_sd3_5_runtime_spec(cfg: Any, device: Any, weight_dtype: Any) -> RuntimeBuildSpec:
    """Slice the runtime-relevant subset out of a whole RL cfg."""
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
    if _dtype_name(weight_dtype) == "float32":
        # Match Flow-GRPO's SD3 LoRA memory contract: keep the trainable
        # denoiser in fp32, but keep frozen text encoders in fp16.
        extra["frozen_dtype"] = "float16"
    if cfg.model.torch_compile.enable:
        extra["torch_compile"] = {
            "enable": True,
            "mode": cfg.model.torch_compile.mode,
        }

    return RuntimeBuildSpec(
        model_name_or_path=cfg.model.path,
        device=device,
        dtype=weight_dtype,
        backend_preference=("diffusers",),
        task_variant="t2i",
        use_lora=bool(cfg.model.use_lora),
        lora_path=lora_path,
        lora_config=lora_cfg,
        scheduler_config={"num_steps": int(cfg.sampling.num_steps)},
        extra=extra,
    )


def build_sd3_5_runtime_bundle(spec: RuntimeBuildSpec) -> RuntimeBundle:
    """Generic build: dispatch the backend model by runtime spec."""
    backend = spec.backend_preference[0]
    model_cls = _resolve_model_cls(backend)

    logger.info("Building sd3_5 runtime bundle (backend=%s)", backend)
    model = model_cls.from_spec(spec)

    if spec.use_lora:
        model.apply_lora(spec)
        if spec.lora_config:
            logger.info(
                "Applied LoRA (rank=%d, alpha=%d)",
                spec.lora_config["rank"], spec.lora_config["alpha"],
            )
    else:
        model.enable_full_finetune()

    compile_cfg = (spec.extra or {}).get("torch_compile") or {}
    if compile_cfg.get("enable"):
        logger.info("Compiling transformer with mode=%s", compile_cfg["mode"])
        model.torch_compile_transformer(compile_cfg["mode"])

    num_steps = (spec.scheduler_config or {}).get("num_steps")
    if num_steps is not None:
        model.set_num_steps(num_steps)
    # If None, caller (e.g. DPO trainer) will set scheduler timesteps itself.

    return RuntimeBundle(
        model=model,
        trainable_modules=model.trainable_modules,
        scheduler=model.scheduler,
        backend_kind=backend,
        backend_handle=model.backend_handle,
        runtime_caps={
            "family_capability": SD3_5_FAMILY_CAPABILITY.to_dict(),
            "supports_stepwise": True,
            "supports_cfg": True,
            "supports_batched_decode": True,
            "supports_reference_conditioning": False,
        },
        metadata={
            "model_path": spec.model_name_or_path,
            "task_variant": spec.task_variant,
            "dtype": str(spec.dtype),
            "use_lora": spec.use_lora,
            **full_generation_bundle_metadata(),
        },
    )


def build_sd3_5_replay_runtime_bundle(spec: RuntimeBuildSpec) -> RuntimeBundle:
    """Build the trainer replay bundle without loading SD3 prompt/VAE modules."""

    from vrl.models.families.sd3_5.model import SD3_5ReplayModel

    backend = spec.backend_preference[0]
    if backend != "diffusers":
        raise NotImplementedError("sd3_5 replay runtime currently supports diffusers only")

    logger.info(
        "Building sd3_5 replay runtime bundle (backend=%s) from %s",
        backend,
        spec.model_name_or_path,
    )
    model = SD3_5ReplayModel(
        transformer=load_diffusers_transformer_component(
            spec,
            "SD3Transformer2DModel",
        ),
        scheduler=load_flow_match_scheduler_component(spec),
        device=spec.device,
    )

    if spec.use_lora:
        apply_lora_to_transformer(model, spec)
    else:
        enable_transformer_full_finetune(model)

    compile_cfg = (spec.extra or {}).get("torch_compile") or {}
    if compile_cfg.get("enable"):
        compile_transformer(model, compile_cfg["mode"])

    return RuntimeBundle(
        model=model,
        trainable_modules=model.trainable_modules,
        scheduler=model.scheduler,
        backend_kind=backend,
        backend_handle=None,
        runtime_caps={
            "family_capability": SD3_5_FAMILY_CAPABILITY.to_dict(),
            "supports_stepwise": True,
            "supports_cfg": True,
            "supports_batched_decode": False,
            "supports_reference_conditioning": False,
        },
        metadata={
            "model_path": spec.model_name_or_path,
            "task_variant": spec.task_variant,
            "dtype": str(spec.dtype),
            "use_lora": spec.use_lora,
            **minimal_replay_bundle_metadata(
                replay_modules=("transformer", "scheduler"),
                generation_only_modules=(
                    "text_encoder",
                    "text_encoder_2",
                    "text_encoder_3",
                    "vae",
                    "pipeline",
                ),
            ),
        },
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


def _dtype_name(value: Any) -> str:
    return str(value).removeprefix("torch.").lower()

class SD3_5PipelineExecutor(DiffusionPipelineExecutorBase):
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
        sample_batch_size: int = 8,
    ) -> None:
        self.model = model
        self.default_sample_batch_size = max(1, int(sample_batch_size))

    def build_chunk_encoded(
        self,
        *,
        encoded: dict[str, Any],
        generation_request: GenerationRequest,
        video_request: VideoGenerationRequest,
        params: DiffusionSamplingParams,
        chunk: MicroBatchSample,
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
    "SD3_5PipelineExecutor",
    "build_sd3_5_replay_runtime_bundle",
    "build_sd3_5_replay_runtime_bundle_from_cfg",
    "build_sd3_5_runtime_bundle",
    "build_sd3_5_runtime_bundle_from_cfg",
    "extract_sd3_5_runtime_spec",
]
