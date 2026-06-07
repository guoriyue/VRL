"""Wan 2.1 family runtime.

The runtime picks the backend model class by task variant (t2v vs i2v).
Backend imports live inside the model's ``from_spec`` so the shared runtime
does not import diffusers or wan-library backends eagerly.
"""

from __future__ import annotations

import logging
from typing import Any

from vrl.generation.diffusion import (
    DiffusionPipelineExecutorBase,
    DiffusionSamplingParams,
)
from vrl.generation.diffusion.layout import VideoGenerationRequest
from vrl.generation.execution.chunks import SampleChunk
from vrl.generation.types import GenerationRequest
from vrl.models.diffusion.capabilities import diffusion_family_capability
from vrl.models.interfaces.runtime import (
    RuntimeBuildSpec,
    RuntimeBundle,
)
from vrl.models.loader import (
    apply_lora_to_transformer,
    compile_transformer,
    enable_transformer_full_finetune,
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
from vrl.utils.config import cfg_get
from vrl.utils.media import load_reference_image

logger = logging.getLogger(__name__)
WAN_2_1_FAMILY_CAPABILITY = diffusion_family_capability("wan_2_1", "t2v")
WAN_2_1_I2V_FAMILY_CAPABILITY = diffusion_family_capability(
    "wan_2_1_i2v",
    "i2v",
    supports_reference_conditioning=True,
)

_MODEL_BY_TASK: dict[str, str] = {
    "t2v": "vrl.models.diffusion.wan_2_1.model:WanT2VDiffusersModel",
    "i2v": "vrl.models.diffusion.wan_2_1.model:WanI2VDiffusersModel",
}


def _resolve_model_cls(task_variant: str | None) -> type:
    import importlib

    task = _normalize_task_variant(task_variant)
    if task not in _MODEL_BY_TASK:
        raise NotImplementedError(
            f"wan_2_1 has no model for task={task!r}; "
            f"registered: {sorted(_MODEL_BY_TASK)}",
        )
    spec = _MODEL_BY_TASK[task]
    mod_path, cls_name = spec.rsplit(":", 1)
    return getattr(importlib.import_module(mod_path), cls_name)


def extract_wan_2_1_runtime_spec(cfg: Any, device: Any, weight_dtype: Any) -> RuntimeBuildSpec:
    """Slice the runtime-relevant subset out of a whole RL cfg."""
    # task_variant (t2v vs i2v) is the one runtime field that varies per cfg;
    # enable_model_cpu_offload / reference_image now ride in model_config and are
    # read by the consumer, so no per-field extra building here.
    return extract_runtime_spec(
        cfg,
        device,
        weight_dtype,
        task_variant=_task_variant_from_cfg(cfg),
    )


def build_wan_2_1_runtime_bundle(spec: RuntimeBuildSpec) -> RuntimeBundle:
    """Generic build: dispatch the backend model by runtime spec."""
    task_variant = _normalize_task_variant(spec.task_variant)
    model_cls = _resolve_model_cls(task_variant)

    logger.info(
        "Building wan_2_1 runtime bundle (task=%s)",
        task_variant,
    )
    use_lora = spec.use_lora
    model = model_cls.from_spec(spec)

    if use_lora:
        model.apply_lora(spec)
        lora_config = spec.lora
        if lora_config:
            logger.info(
                "Applied LoRA (rank=%d, alpha=%d)",
                lora_config["rank"], lora_config["alpha"],
            )
    else:
        model.enable_full_finetune()

    compile_cfg = spec.torch_compile or {}
    if compile_cfg.get("enable"):
        logger.info("Compiling transformer with mode=%s", compile_cfg["mode"])
        model.torch_compile_transformer(compile_cfg["mode"])

    num_steps = spec.num_steps
    if num_steps is not None:
        model.set_num_steps(num_steps)
    # If None, caller (e.g. DPO trainer) will set scheduler timesteps itself.

    metadata = {
        "model_path": spec.model_name_or_path,
        "task_variant": task_variant,
        "dtype": str(spec.dtype),
        "use_lora": use_lora,
        "reference_image": (spec.model_config or {}).get("reference_image"),
        **full_generation_bundle_metadata(),
    }
    metadata.update(getattr(model, "memory_metadata", None) or {})
    return RuntimeBundle(
        model=model,
        trainable_modules=model.trainable_modules,
        scheduler=model.scheduler,
        backend_handle=model.backend_handle,
        runtime_caps={
            "supports_stepwise": True,
            "supports_cfg": True,
            "supports_batched_decode": True,
            "supports_reference_conditioning": task_variant == "i2v",
        },
        metadata=metadata,
    )


def build_wan_2_1_replay_runtime_bundle(spec: RuntimeBuildSpec) -> RuntimeBundle:
    """Build the trainer replay bundle without loading Wan text/VAE modules."""

    from vrl.models.diffusion.wan_2_1.model import (
        WanI2VReplayModel,
        WanT2VReplayModel,
    )

    task_variant = _normalize_task_variant(spec.task_variant)
    replay_cls = WanI2VReplayModel if task_variant == "i2v" else WanT2VReplayModel

    logger.info(
        "Building wan_2_1 replay runtime bundle (task=%s) from %s",
        task_variant,
        spec.model_name_or_path,
    )
    model = replay_cls(
        transformer=load_diffusers_transformer(
            spec,
            "WanTransformer3DModel",
        ),
        scheduler=load_diffusers_scheduler(
            spec,
            "UniPCMultistepScheduler",
        ),
        device=spec.device,
    )

    use_lora = spec.use_lora
    if use_lora:
        apply_lora_to_transformer(model, spec)
    else:
        enable_transformer_full_finetune(model)

    compile_cfg = spec.torch_compile or {}
    if compile_cfg.get("enable"):
        compile_transformer(model, compile_cfg["mode"])

    return RuntimeBundle(
        model=model,
        trainable_modules=model.trainable_modules,
        scheduler=model.scheduler,
        backend_handle=None,
        runtime_caps={
            "supports_stepwise": True,
            "supports_cfg": True,
            "supports_batched_decode": False,
            "supports_reference_conditioning": task_variant == "i2v",
        },
        metadata={
            "model_path": spec.model_name_or_path,
            "task_variant": task_variant,
            "dtype": str(spec.dtype),
            "use_lora": use_lora,
            "reference_image": (spec.model_config or {}).get("reference_image"),
            **minimal_replay_bundle_metadata(
                replay_modules=("transformer", "scheduler"),
                generation_only_modules=(
                    "text_encoder",
                    "image_encoder",
                    "image_processor",
                    "vae",
                    "pipeline",
                )
                if task_variant == "i2v"
                else ("text_encoder", "vae", "pipeline"),
            ),
        },
    )


def build_wan_2_1_runtime_bundle_from_cfg(
    cfg: Any, device: Any, weight_dtype: Any,
) -> RuntimeBundle:
    """Outer convenience: whole-cfg → spec → bundle."""
    spec = extract_wan_2_1_runtime_spec(cfg, device, weight_dtype)
    return build_wan_2_1_runtime_bundle(spec)


def build_wan_2_1_replay_runtime_bundle_from_cfg(
    cfg: Any,
    device: Any,
    weight_dtype: Any,
) -> RuntimeBundle:
    """Outer convenience: whole-cfg → spec → replay bundle."""
    spec = extract_wan_2_1_runtime_spec(cfg, device, weight_dtype)
    return build_wan_2_1_replay_runtime_bundle(spec)


"""Wan 2.1 diffusion pipeline executor."""


class Wan_2_1PipelineExecutor(DiffusionPipelineExecutorBase):
    """Diffusion executor for Wan 2.1 text-to-video rollouts."""

    family: str = "wan_2_1"
    task: str = "t2v"
    family_capability = WAN_2_1_FAMILY_CAPABILITY
    default_num_frames: int = 1
    default_max_sequence_length: int = 512

    def __init__(
        self,
        model: Any,
        *,
        sample_batch_size: int = 1,
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
        chunk: SampleChunk,
    ) -> dict[str, Any]:
        """Repeat Wan text embeds across the chunk batch."""

        del generation_request, video_request, params
        chunk_g = chunk.sample_count
        chunk_encoded: dict[str, Any] = {
            "prompt_embeds": self.layout.repeat_batch(
                encoded["prompt_embeds"],
                chunk_g,
            ),
        }
        neg = encoded.get("negative_prompt_embeds")
        if neg is not None:
            chunk_encoded["negative_prompt_embeds"] = self.layout.repeat_batch(
                neg,
                chunk_g,
            )
        return chunk_encoded


class Wan_2_1I2VPipelineExecutor(DiffusionPipelineExecutorBase):
    """Diffusion executor for Wan 2.1 image-to-video rollouts."""

    family: str = "wan_2_1_i2v"
    task: str = "i2v"
    family_capability = WAN_2_1_I2V_FAMILY_CAPABILITY
    default_num_frames: int = 81
    default_max_sequence_length: int = 512

    def __init__(
        self,
        model: Any,
        *,
        reference_image: Any = None,
        sample_batch_size: int = 1,
    ) -> None:
        self.model = model
        self.reference_image = reference_image
        self.default_sample_batch_size = max(1, int(sample_batch_size))

    def encode_prompt_for_chunk(
        self,
        *,
        generation_request: GenerationRequest,
        video_request: VideoGenerationRequest,
        params: DiffusionSamplingParams,
        chunk: SampleChunk,
    ) -> dict[str, Any]:
        """Encode text and image conditioning for one I2V prompt chunk."""

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
        chunk: SampleChunk,
    ) -> dict[str, Any]:
        """Repeat encoded tensors across K samples and keep the image handle shared."""

        del generation_request, video_request, params
        chunk_g = chunk.sample_count
        chunk_encoded: dict[str, Any] = {
            "prompt_embeds": self.layout.repeat_batch(
                encoded["prompt_embeds"],
                chunk_g,
            ),
            "reference_image": encoded.get("reference_image"),
        }
        neg = encoded.get("negative_prompt_embeds")
        if neg is not None:
            chunk_encoded["negative_prompt_embeds"] = self.layout.repeat_batch(
                neg,
                chunk_g,
            )
        else:
            chunk_encoded["negative_prompt_embeds"] = None
        image_embeds = encoded.get("image_embeds")
        if image_embeds is not None:
            chunk_encoded["image_embeds"] = self.layout.repeat_batch(
                image_embeds,
                chunk_g,
            )
        return chunk_encoded

    def build_prepare_kwargs(
        self,
        *,
        encoded: dict[str, Any],
        generation_request: GenerationRequest,
        video_request: VideoGenerationRequest,
        params: DiffusionSamplingParams,
        chunk: SampleChunk,
    ) -> dict[str, Any]:
        """Thread the active reference image into Wan I2V prepare_sampling."""

        del encoded, video_request, params, chunk
        return {
            "reference_image": self._reference_image_for_request(
                generation_request,
            ),
        }

    def _reference_image_for_request(self, request: GenerationRequest) -> Any:
        return load_reference_image(
            request.metadata.get("reference_image", self.reference_image),
        )


def _task_variant_from_cfg(cfg: Any) -> str:
    explicit = cfg_get(cfg.model, "task_variant", None)
    if explicit:
        return _normalize_task_variant(str(explicit))
    task = cfg_get(cfg.model, "task", None)
    if task:
        return _normalize_task_variant(str(task))
    family = str(cfg_get(cfg.model, "family", ""))
    if "i2v" in family or "image" in family:
        return "i2v"
    return "t2v"


def _normalize_task_variant(task_variant: str | None) -> str:
    # Accept any non-i2v value as t2v so generic test fixtures that use a
    # neutral placeholder like "t2i" do not need a special-case branch per
    # family. Real i2v dispatch only fires for the explicit i2v aliases below
    # or when cfg.model.family/task_variant declares i2v.
    text = str(task_variant or "t2v").strip().lower()
    if text in {"image_to_video", "image-to-video", "i2v"}:
        return "i2v"
    return "t2v"


__all__ = [
    "Wan_2_1I2VPipelineExecutor",
    "Wan_2_1PipelineExecutor",
    "build_wan_2_1_replay_runtime_bundle",
    "build_wan_2_1_replay_runtime_bundle_from_cfg",
    "build_wan_2_1_runtime_bundle",
    "build_wan_2_1_runtime_bundle_from_cfg",
    "extract_wan_2_1_runtime_spec",
]
