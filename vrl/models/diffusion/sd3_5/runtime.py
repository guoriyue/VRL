"""SD 3.5 family runtime.

Backend imports live inside the model's ``from_spec`` so the shared runtime
does not import diffusers or future native backends eagerly.
"""

from __future__ import annotations

from typing import Any

from vrl.generation.diffusion import (
    DiffusionChunkExecutorBase,
    DiffusionSamplingParams,
)
from vrl.generation.diffusion.layout import VideoGenerationRequest
from vrl.generation.execution.chunks import SampleChunk
from vrl.generation.types import GenerationRequest
from vrl.models.diffusion.common.vae_decode_memory import (
    apply_generation_memory_policy,
)
from vrl.models.diffusion.capabilities import diffusion_family_capability
from vrl.models.interfaces.runtime import (
    RuntimeBuildSpec,
    RuntimeBundle,
)
from vrl.models.loader import (
    apply_lora_to_transformer,
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

logger = init_logger(__name__)
SD3_5_FAMILY_CAPABILITY = diffusion_family_capability("sd3_5", "t2i")


def extract_sd3_5_runtime_spec(cfg: Any, device: Any, weight_dtype: Any) -> RuntimeBuildSpec:
    """Slice the runtime-relevant subset out of a whole RL cfg."""
    return extract_runtime_spec(
        cfg,
        device,
        weight_dtype,
        task_variant="t2i",
    )


def build_sd3_5_runtime_bundle(spec: RuntimeBuildSpec) -> RuntimeBundle:
    """Generic build: dispatch the backend model by runtime spec."""
    from vrl.models.diffusion.sd3_5.model import SD3_5Model

    logger.info("Building sd3_5 runtime bundle")
    use_lora = spec.use_lora
    model = SD3_5Model.from_spec(spec)

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

    return RuntimeBundle(
        model=model,
        trainable_modules=model.trainable_modules,
        scheduler=model.scheduler,
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
            "use_lora": use_lora,
            **full_generation_bundle_metadata(),
            **apply_generation_memory_policy(
                model,
                memory_config=getattr(spec, "memory", None),
                owner="SD3.5 VAE",
            ),
        },
    )


def build_sd3_5_replay_runtime_bundle(spec: RuntimeBuildSpec) -> RuntimeBundle:
    """Build the trainer replay bundle without loading SD3 prompt/VAE modules."""

    from vrl.models.diffusion.sd3_5.model import SD3_5ReplayModel

    logger.info(
        "Building sd3_5 replay runtime bundle from %s",
        spec.model_name_or_path,
    )
    model = SD3_5ReplayModel(
        transformer=load_diffusers_transformer(
            spec,
            "SD3Transformer2DModel",
        ),
        scheduler=load_flow_match_scheduler(spec),
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
            "use_lora": use_lora,
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


class SD3_5ChunkExecutor(DiffusionChunkExecutorBase):
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
        chunk: SampleChunk,
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
    "SD3_5ChunkExecutor",
    "build_sd3_5_replay_runtime_bundle",
    "build_sd3_5_replay_runtime_bundle_from_cfg",
    "build_sd3_5_runtime_bundle",
    "build_sd3_5_runtime_bundle_from_cfg",
    "extract_sd3_5_runtime_spec",
]
