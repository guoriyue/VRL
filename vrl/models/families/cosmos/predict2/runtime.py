"""Cosmos Predict2 family runtime.

The runtime picks the adapter class by ``spec.backend_preference``,
let the adapter load itself + apply LoRA, then assemble the bundle. No
backend imports live here — diffusers / cosmos-library imports stay
inside each adapter's ``from_spec``.
"""

from __future__ import annotations

import logging
from typing import Any

from vrl.models.runtime import RuntimeBuildSpec, RuntimeBundle

logger = logging.getLogger(__name__)

_ADAPTER_BY_BACKEND: dict[str, str] = {
    "diffusers": "vrl.models.families.cosmos.predict2.policy:CosmosPredict2Policy",
}


def _resolve_adapter_cls(backend: str) -> type:
    import importlib

    if backend not in _ADAPTER_BY_BACKEND:
        raise NotImplementedError(
            f"cosmos-predict2 has no adapter for backend={backend!r}; "
            f"registered: {sorted(_ADAPTER_BY_BACKEND)}",
        )
    spec = _ADAPTER_BY_BACKEND[backend]
    mod_path, cls_name = spec.rsplit(":", 1)
    return getattr(importlib.import_module(mod_path), cls_name)


def extract_cosmos_predict2_runtime_spec(
    cfg: Any, device: Any, weight_dtype: Any,
) -> RuntimeBuildSpec:
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
    if getattr(cfg.model, "torch_compile", None) is not None and cfg.model.torch_compile.enable:
        extra["torch_compile"] = {
            "enable": True,
            "mode": cfg.model.torch_compile.mode,
        }
    ref_image = getattr(cfg.model, "reference_image", None)
    if ref_image:
        extra["reference_image"] = str(ref_image)

    return RuntimeBuildSpec(
        model_name_or_path=cfg.model.path,
        device=device,
        dtype=weight_dtype,
        backend_preference=("diffusers",),
        task_variant="video2world",
        use_lora=bool(cfg.model.use_lora),
        lora_path=lora_path,
        lora_config=lora_cfg,
        scheduler_config={"num_steps": int(cfg.sampling.num_steps)},
        extra=extra,
    )


def build_cosmos_predict2_runtime_bundle(
    spec: RuntimeBuildSpec,
) -> RuntimeBundle:
    """Generic build: dispatch adapter by backend, let it own its load."""
    backend = spec.backend_preference[0]
    adapter_cls = _resolve_adapter_cls(backend)

    logger.info(
        "Building cosmos-predict2 runtime bundle (backend=%s) from %s",
        backend, spec.model_name_or_path,
    )
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
            "supports_reference_conditioning": True,
        },
        metadata={
            "model_path": spec.model_name_or_path,
            "task_variant": spec.task_variant,
            "dtype": str(spec.dtype),
            "use_lora": spec.use_lora,
            "reference_image": (spec.extra or {}).get("reference_image"),
        },
    )


def build_cosmos_predict2_runtime_bundle_from_cfg(
    cfg: Any, device: Any, weight_dtype: Any,
) -> RuntimeBundle:
    """Outer convenience: whole-cfg → spec → bundle."""
    spec = extract_cosmos_predict2_runtime_spec(cfg, device, weight_dtype)
    return build_cosmos_predict2_runtime_bundle(spec)

"""Cosmos Predict2 Video2World diffusion pipeline executor."""


from typing import Any

from vrl.engine.core.types import GenerationRequest
from vrl.engine.diffusion import (
    DiffusionGenerationSpec,
    DiffusionPipelineExecutorBase,
    repeat_tensor_batch,
)
from vrl.engine.microbatching import MicroBatchPlan
from vrl.models.diffusion import VideoGenerationRequest


class CosmosPipelineExecutor(DiffusionPipelineExecutorBase):
    """Diffusion executor for Cosmos Predict2 Video2World rollouts."""

    family: str = "cosmos-predict2"
    task: str = "v2w"
    default_num_frames: int = 93
    default_fps: int | None = 16
    default_max_sequence_length: int = 512
    respect_cfg_flag: bool = False
    include_max_sequence_length_extra: bool = False

    def __init__(
        self,
        model: Any,  # CosmosPredict2Policy
        *,
        reference_image: Any = None,
        sample_batch_size: int = 8,
    ) -> None:
        self.model = model
        self.reference_image = _load_reference_image(reference_image)
        self.default_sample_batch_size = max(1, int(sample_batch_size))

    def encode_prompt_for_chunk(
        self,
        *,
        generation_request: GenerationRequest,
        video_request: VideoGenerationRequest,
        spec: DiffusionGenerationSpec,
        chunk: MicroBatchPlan,
    ) -> dict[str, Any]:
        """Encode Cosmos text and preserve the Video2World reference image."""

        reference_image = self._reference_image_for_request(generation_request)
        return self.model.encode_prompt(
            chunk.prompt,
            video_request.negative_prompt or None,
            max_sequence_length=spec.base.max_sequence_length,
            guidance_scale=spec.base.guidance_scale,
            reference_image=reference_image,
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
        """Repeat Cosmos text embeds and pass reference image through unchanged."""

        del video_request, spec
        chunk_g = chunk.sample_count
        reference_image = self._reference_image_for_request(generation_request)
        chunk_encoded: dict[str, Any] = {
            "prompt_embeds": repeat_tensor_batch(
                encoded["prompt_embeds"],
                chunk_g,
            ),
            "reference_image": encoded.get("reference_image", reference_image),
        }
        neg = encoded.get("negative_prompt_embeds")
        if neg is not None:
            chunk_encoded["negative_prompt_embeds"] = repeat_tensor_batch(
                neg,
                chunk_g,
            )
        else:
            chunk_encoded["negative_prompt_embeds"] = None
        return chunk_encoded

    def build_prepare_kwargs(
        self,
        *,
        encoded: dict[str, Any],
        generation_request: GenerationRequest,
        video_request: VideoGenerationRequest,
        spec: DiffusionGenerationSpec,
        chunk: MicroBatchPlan,
    ) -> dict[str, Any]:
        """Thread the active reference image into Cosmos prepare_sampling."""

        del encoded, video_request, spec, chunk
        return {
            "reference_image": self._reference_image_for_request(
                generation_request,
            ),
        }

    def _reference_image_for_request(self, request: GenerationRequest) -> Any:
        return _load_reference_image(
            request.metadata.get("reference_image", self.reference_image),
        )


def _load_reference_image(reference_image: Any) -> Any:
    if not isinstance(reference_image, str) or not reference_image:
        return reference_image
    from PIL import Image

    return Image.open(reference_image).convert("RGB")


__all__ = [
    "CosmosPipelineExecutor",
    "build_cosmos_predict2_runtime_bundle",
    "build_cosmos_predict2_runtime_bundle_from_cfg",
    "extract_cosmos_predict2_runtime_spec",
]
