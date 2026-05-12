"""SD 3.5 family runtime.

The runtime picks the adapter class by ``spec.backend_preference``,
let the adapter load itself + apply LoRA, then assemble the bundle. No
backend imports live here — every backend lib (diffusers / future native)
imports inside its own adapter's ``from_spec``.
"""

from __future__ import annotations

import logging
from typing import Any

from vrl.models.runtime import RuntimeBuildSpec, RuntimeBundle

logger = logging.getLogger(__name__)

_ADAPTER_BY_BACKEND: dict[str, str] = {
    "diffusers": "vrl.models.families.sd3_5.policy:SD3_5Policy",
}


def _resolve_adapter_cls(backend: str) -> type:
    import importlib

    if backend not in _ADAPTER_BY_BACKEND:
        raise NotImplementedError(
            f"sd3_5 has no adapter for backend={backend!r}; "
            f"registered: {sorted(_ADAPTER_BY_BACKEND)}",
        )
    spec = _ADAPTER_BY_BACKEND[backend]
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
    """Generic build: dispatch adapter by backend, let it own its load."""
    backend = spec.backend_preference[0]
    adapter_cls = _resolve_adapter_cls(backend)

    logger.info("Building sd3_5 runtime bundle (backend=%s)", backend)
    adapter = adapter_cls.from_spec(spec)

    if spec.use_lora:
        adapter.apply_lora(spec)
        if spec.lora_config:
            logger.info(
                "Applied LoRA (rank=%d, alpha=%d)",
                spec.lora_config["rank"], spec.lora_config["alpha"],
            )
    else:
        adapter.enable_full_finetune()

    compile_cfg = (spec.extra or {}).get("torch_compile") or {}
    if compile_cfg.get("enable"):
        logger.info("Compiling transformer with mode=%s", compile_cfg["mode"])
        adapter.torch_compile_transformer(compile_cfg["mode"])

    num_steps = (spec.scheduler_config or {}).get("num_steps")
    if num_steps is not None:
        adapter.set_num_steps(num_steps)
    # If None, caller (e.g. DPO trainer) will set scheduler timesteps itself.

    return RuntimeBundle(
        policy=adapter,
        trainable_modules=adapter.trainable_modules,
        scheduler=adapter.scheduler,
        backend_kind=backend,
        backend_handle=adapter.backend_handle,
        runtime_caps={
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
        },
    )


def build_sd3_5_runtime_bundle_from_cfg(
    cfg: Any, device: Any, weight_dtype: Any,
) -> RuntimeBundle:
    """Outer convenience: whole-cfg → spec → bundle."""
    spec = extract_sd3_5_runtime_spec(cfg, device, weight_dtype)
    return build_sd3_5_runtime_bundle(spec)


def _dtype_name(value: Any) -> str:
    return str(value).removeprefix("torch.").lower()

"""SD3.5-Medium diffusion pipeline executor."""


from typing import Any

from vrl.engine.core.types import GenerationRequest
from vrl.engine.diffusion import (
    DiffusionGenerationSpec,
    DiffusionPipelineExecutorBase,
    repeat_tensor_batch,
)
from vrl.engine.microbatching import MicroBatchPlan
from vrl.models.diffusion import VideoGenerationRequest


class SD3_5PipelineExecutor(DiffusionPipelineExecutorBase):
    """Diffusion executor for SD3.5-M text-to-image rollouts."""

    family: str = "sd3_5"
    task: str = "t2i"
    default_num_frames: int = 1
    default_max_sequence_length: int = 128

    def __init__(
        self,
        model: Any,  # SD3_5Policy
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
        spec: DiffusionGenerationSpec,
        chunk: MicroBatchPlan,
    ) -> dict[str, Any]:
        """Repeat SD3 prompt and pooled embeds across the chunk batch."""

        del generation_request, video_request, spec
        chunk_g = chunk.sample_count
        chunk_encoded: dict[str, Any] = {
            "prompt_embeds": repeat_tensor_batch(
                encoded["prompt_embeds"],
                chunk_g,
            ),
            "pooled_prompt_embeds": repeat_tensor_batch(
                encoded["pooled_prompt_embeds"],
                chunk_g,
            ),
        }
        neg = encoded.get("negative_prompt_embeds")
        neg_pool = encoded.get("negative_pooled_prompt_embeds")
        if neg is not None:
            chunk_encoded["negative_prompt_embeds"] = repeat_tensor_batch(
                neg,
                chunk_g,
            )
        if neg_pool is not None:
            chunk_encoded["negative_pooled_prompt_embeds"] = repeat_tensor_batch(
                neg_pool,
                chunk_g,
            )
        return chunk_encoded


__all__ = [
    "SD3_5PipelineExecutor",
    "build_sd3_5_runtime_bundle",
    "build_sd3_5_runtime_bundle_from_cfg",
    "extract_sd3_5_runtime_spec",
]
