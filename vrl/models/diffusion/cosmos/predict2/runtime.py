"""Cosmos Predict2 family runtime.

The runtime picks the backend model class by ``spec.backend_preference``.
Backend imports live inside the model's ``from_spec`` so the shared runtime
does not import diffusers or cosmos-library backends eagerly.
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
from vrl.models.diffusion.capabilities import diffusion_family_capability
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
COSMOS_PREDICT2_FAMILY_CAPABILITY = diffusion_family_capability(
    "cosmos-predict2",
    "v2w",
    supports_reference_conditioning=True,
)

_MODEL_BY_BACKEND: dict[str, str] = {
    "diffusers": "vrl.models.diffusion.cosmos.predict2.model:CosmosPredict2Model",
}


def _resolve_model_cls(backend: str) -> type:
    import importlib

    if backend not in _MODEL_BY_BACKEND:
        raise NotImplementedError(
            f"cosmos-predict2 has no model for backend={backend!r}; "
            f"registered: {sorted(_MODEL_BY_BACKEND)}",
        )
    spec = _MODEL_BY_BACKEND[backend]
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
    """Generic build: dispatch the backend model by runtime spec."""
    backend = spec.backend_preference[0]
    model_cls = _resolve_model_cls(backend)

    logger.info(
        "Building cosmos-predict2 runtime bundle (backend=%s) from %s",
        backend, spec.model_name_or_path,
    )
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
            **full_generation_bundle_metadata(),
        },
    )


def build_cosmos_predict2_replay_runtime_bundle(
    spec: RuntimeBuildSpec,
) -> RuntimeBundle:
    """Build the trainer replay bundle without Cosmos generation-only modules."""

    from vrl.models.diffusion.cosmos.predict2.model import CosmosPredict2ReplayModel

    backend = spec.backend_preference[0]
    if backend != "diffusers":
        raise NotImplementedError(
            "cosmos-predict2 replay runtime currently supports diffusers only",
        )

    logger.info(
        "Building cosmos-predict2 replay runtime bundle (backend=%s) from %s",
        backend,
        spec.model_name_or_path,
    )
    model = CosmosPredict2ReplayModel(
        transformer=load_diffusers_transformer_component(
            spec,
            "CosmosTransformer3DModel",
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
            "supports_stepwise": True,
            "supports_cfg": True,
            "supports_batched_decode": False,
            "supports_reference_conditioning": True,
        },
        metadata={
            "model_path": spec.model_name_or_path,
            "task_variant": spec.task_variant,
            "dtype": str(spec.dtype),
            "use_lora": spec.use_lora,
            "reference_image": (spec.extra or {}).get("reference_image"),
            **minimal_replay_bundle_metadata(
                replay_modules=("transformer", "scheduler"),
                generation_only_modules=(
                    "text_encoder",
                    "vae",
                    "safety_checker",
                    "pipeline",
                ),
            ),
        },
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


"""Cosmos Predict2 Video2World diffusion pipeline executor."""


class CosmosPipelineExecutor(DiffusionPipelineExecutorBase):
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
        params: DiffusionSamplingParams,
        chunk: MicroBatchSample,
    ) -> dict[str, Any]:
        """Encode Cosmos text and preserve the Video2World reference image."""

        reference_image = self._reference_image_for_request(generation_request)
        return self.model.encode_prompt(
            chunk.prompt,
            video_request.negative_prompt or None,
            max_sequence_length=params.base.max_sequence_length,
            guidance_scale=params.base.guidance_scale,
            reference_image=reference_image,
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

    def build_prepare_kwargs(
        self,
        *,
        encoded: dict[str, Any],
        generation_request: GenerationRequest,
        video_request: VideoGenerationRequest,
        params: DiffusionSamplingParams,
        chunk: MicroBatchSample,
    ) -> dict[str, Any]:
        """Thread the active reference image into Cosmos prepare_sampling."""

        del encoded, video_request, params, chunk
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
    "build_cosmos_predict2_replay_runtime_bundle",
    "build_cosmos_predict2_replay_runtime_bundle_from_cfg",
    "build_cosmos_predict2_runtime_bundle",
    "build_cosmos_predict2_runtime_bundle_from_cfg",
    "extract_cosmos_predict2_runtime_spec",
]
