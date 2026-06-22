"""FLUX.1 family runtime.

Backend imports live inside the model's ``from_spec`` so the shared runtime does
not import diffusers eagerly. Mirrors the sd3_5 runtime four-builder shape.
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
from vrl.models.diffusion.capabilities import diffusion_family_capability
from vrl.models.diffusion.common.vae_decode_memory import (
    apply_generation_memory_policy,
)
from vrl.models.interfaces.runtime import (
    RuntimeBuildSpec,
    RuntimeBundle,
)
from vrl.models.loader import (
    apply_lora_to_transformer,
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

logger = init_logger(__name__)
FLUX_FAMILY_CAPABILITY = diffusion_family_capability("flux", "t2i")


def extract_flux_runtime_spec(cfg: Any, device: Any, weight_dtype: Any) -> RuntimeBuildSpec:
    """Slice the runtime-relevant subset out of a whole RL cfg."""
    return extract_runtime_spec(
        cfg,
        device,
        weight_dtype,
        task_variant="t2i",
    )


def build_flux_runtime_bundle(
    spec: RuntimeBuildSpec,
    *,
    attach_previous_adapter: bool = False,
) -> RuntimeBundle:
    """Generic build: dispatch the backend model by runtime spec.

    ``attach_previous_adapter`` is the DiffusionNFT-only switch: when set, build a
    frozen ``previous`` LoRA adapter mirroring ``default`` (the NFT previous-policy
    forward). Left False for GRPO so its build path is bit-for-bit unchanged.
    """
    from vrl.models.diffusion.flux.model import FluxModel

    logger.info("Building flux runtime bundle")
    use_lora = spec.use_lora
    model = FluxModel.from_spec(spec)

    if use_lora:
        model.apply_lora(spec)
        lora_config = spec.lora
        if lora_config:
            logger.info(
                "Applied LoRA (rank=%d, alpha=%d)",
                lora_config["rank"], lora_config["alpha"],
            )
        if attach_previous_adapter:
            model.attach_previous_policy_adapter(spec)
            logger.info("Attached frozen DiffusionNFT `previous` LoRA adapter")
    else:
        model.enable_full_finetune()

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
            "family_capability": FLUX_FAMILY_CAPABILITY.to_dict(),
            "supports_reference_conditioning": False,
        },
        metadata={
            "model_path": spec.model_name_or_path,
            "family": FLUX_FAMILY_CAPABILITY.family,
            "task_variant": spec.task_variant,
            "dtype": str(spec.dtype),
            "use_lora": use_lora,
            **full_generation_bundle_metadata(),
            **apply_generation_memory_policy(
                model,
                memory_config=getattr(spec, "memory", None),
                owner="FLUX VAE",
            ),
        },
    )


def build_flux_replay_runtime_bundle(
    spec: RuntimeBuildSpec,
    *,
    attach_previous_adapter: bool = False,
) -> RuntimeBundle:
    """Build the trainer replay bundle without loading FLUX prompt/VAE modules.

    This is the model that actually runs the DiffusionNFT loss (the trainer
    optimizes it), so ``attach_previous_adapter`` must be set here too — it builds
    the frozen ``previous`` adapter the NFT forward activates. GRPO leaves it off.
    """

    from vrl.models.diffusion.flux.model import FluxReplayModel

    logger.info(
        "Building flux replay runtime bundle from %s",
        spec.model_name_or_path,
    )
    model = FluxReplayModel(
        transformer=load_diffusers_transformer(
            spec,
            "FluxTransformer2DModel",
        ),
        scheduler=load_flow_match_scheduler(spec),
        device=spec.device,
    )

    use_lora = spec.use_lora
    if use_lora:
        apply_lora_to_transformer(model, spec)
        if attach_previous_adapter:
            model.attach_previous_policy_adapter(spec)
            logger.info("Attached frozen DiffusionNFT `previous` LoRA adapter (replay)")
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
            "family_capability": FLUX_FAMILY_CAPABILITY.to_dict(),
            "supports_reference_conditioning": False,
        },
        metadata={
            "model_path": spec.model_name_or_path,
            "family": FLUX_FAMILY_CAPABILITY.family,
            "task_variant": spec.task_variant,
            "dtype": str(spec.dtype),
            "use_lora": use_lora,
            **minimal_replay_bundle_metadata(),
        },
    )


def build_flux_runtime_bundle_from_cfg(
    cfg: Any, device: Any, weight_dtype: Any,
) -> RuntimeBundle:
    """Outer convenience: whole-cfg → spec → bundle."""
    spec = extract_flux_runtime_spec(cfg, device, weight_dtype)
    return build_flux_runtime_bundle(spec)


def build_flux_replay_runtime_bundle_from_cfg(
    cfg: Any,
    device: Any,
    weight_dtype: Any,
) -> RuntimeBundle:
    """Outer convenience: whole-cfg → spec → replay bundle."""
    spec = extract_flux_runtime_spec(cfg, device, weight_dtype)
    return build_flux_replay_runtime_bundle(spec)


class FluxChunkExecutor(DiffusionChunkExecutorBase):
    """Diffusion executor for FLUX.1 text-to-image rollouts."""

    family: str = "flux"
    task: str = "t2i"
    family_capability = FLUX_FAMILY_CAPABILITY
    default_num_frames: int = 1
    default_max_sequence_length: int = 512

    def __init__(
        self,
        model: Any,  # FluxModel
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
        """Repeat FLUX prompt/pooled embeds across the chunk batch.

        ``text_ids`` is batch-shared (shape ``[seq, 3]``, no batch dim), so it is
        passed through unrepeated — repeating it would corrupt its leading dim.
        """

        del generation_request, video_request, params
        chunk_g = chunk.sample_count
        return {
            "prompt_embeds": self.layout.repeat_batch(
                encoded["prompt_embeds"],
                chunk_g,
            ),
            "pooled_prompt_embeds": self.layout.repeat_batch(
                encoded["pooled_prompt_embeds"],
                chunk_g,
            ),
            "text_ids": encoded["text_ids"],
        }


__all__ = [
    "FluxChunkExecutor",
    "build_flux_replay_runtime_bundle",
    "build_flux_replay_runtime_bundle_from_cfg",
    "build_flux_runtime_bundle",
    "build_flux_runtime_bundle_from_cfg",
    "extract_flux_runtime_spec",
]
