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
from vrl.models.diffusion.capabilities import diffusion_family_capability
from vrl.models.diffusion.common.vae_decode_memory import (
    apply_generation_memory_policy,
)
from vrl.models.interfaces.runtime import RuntimeBuildSpec, RuntimeBundle
from vrl.models.loader import (
    apply_rollout_quantization,
    compile_transformer,
    enable_transformer_full_finetune,
    load_diffusers_transformer,
    load_flow_match_scheduler,
)
from vrl.models.replay_loading import (
    full_generation_bundle_metadata,
    minimal_replay_bundle_metadata,
)
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


def _reference_image_from_spec(spec: RuntimeBuildSpec) -> str | None:
    """Bundle-metadata reference image; empty cfg value reads as ``None``."""

    return (spec.model_config or {}).get("reference_image") or None


def build_cosmos_predict2_runtime_bundle(
    spec: RuntimeBuildSpec,
) -> RuntimeBundle:
    """Generic build: dispatch the backend model by runtime spec."""
    from vrl.models.diffusion.cosmos.predict2.model import CosmosPredict2Model

    logger.info(
        "Building cosmos-predict2 runtime bundle from %s",
        spec.model_name_or_path,
    )
    use_lora = spec.use_lora
    model = CosmosPredict2Model.from_spec(spec)

    if use_lora:
        model.apply_lora(spec)
        lora_config = spec.lora
        if lora_config:
            logger.info(
                "Applied LoRA (rank=%d, alpha=%d)",
                lora_config["rank"], lora_config["alpha"],
            )
    else:
        model.apply_full_finetune()

    apply_rollout_quantization(model, spec)

    compile_cfg = spec.torch_compile or {}
    if compile_cfg.get("enable"):
        logger.info("Compiling transformer with mode=%s", compile_cfg["mode"])
        model.torch_compile_transformer(compile_cfg["mode"])

    num_steps = spec.num_steps
    if num_steps is not None:
        model.set_num_steps(num_steps)
    # If None, caller (e.g. DPO trainer) will set scheduler timesteps itself.

    return RuntimeBundle(
        model=model,
        trainable_modules=model.trainable_modules,
        scheduler=model.scheduler,
        raw_handle=model.raw_handle,
        runtime_caps={
            "supports_reference_conditioning": True,
        },
        metadata={
            "model_path": spec.model_name_or_path,
            "family": COSMOS_PREDICT2_FAMILY_CAPABILITY.family,
            "task_variant": spec.task_variant,
            "dtype": str(spec.dtype),
            "use_lora": use_lora,
            "reference_image": _reference_image_from_spec(spec),
            **full_generation_bundle_metadata(),
            **apply_generation_memory_policy(
                model,
                memory_config=getattr(spec, "memory", None),
                owner="Cosmos Predict2 VAE",
            ),
        },
    )


def build_cosmos_predict2_replay_runtime_bundle(
    spec: RuntimeBuildSpec,
) -> RuntimeBundle:
    """Build the trainer replay bundle without Cosmos generation-only modules."""

    from vrl.models.diffusion.cosmos.predict2.model import CosmosPredict2ReplayModel

    logger.info(
        "Building cosmos-predict2 replay runtime bundle from %s",
        spec.model_name_or_path,
    )
    model = CosmosPredict2ReplayModel(
        transformer=load_diffusers_transformer(
            spec,
            "CosmosTransformer3DModel",
        ),
        scheduler=load_flow_match_scheduler(spec),
        device=spec.device,
    )

    use_lora = spec.use_lora
    if use_lora:
        model.apply_lora(spec)
    else:
        enable_transformer_full_finetune(model)

    compile_cfg = spec.torch_compile or {}
    if compile_cfg.get("enable"):
        compile_transformer(model, compile_cfg["mode"])

    return RuntimeBundle(
        model=model,
        trainable_modules=model.trainable_modules,
        scheduler=model.scheduler,
        raw_handle=None,
        runtime_caps={
            "supports_reference_conditioning": True,
        },
        metadata={
            "model_path": spec.model_name_or_path,
            "family": COSMOS_PREDICT2_FAMILY_CAPABILITY.family,
            "task_variant": spec.task_variant,
            "dtype": str(spec.dtype),
            "use_lora": use_lora,
            "reference_image": _reference_image_from_spec(spec),
            **minimal_replay_bundle_metadata(),
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
        sample_batch_size: int = 8,
    ) -> None:
        self.model = model
        self.reference_image = load_reference_image(reference_image)
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
