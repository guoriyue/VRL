"""Qwen-Image family runtime.

Backend imports live inside the model's ``from_spec`` so the shared runtime does
not import diffusers eagerly. Mirrors the sd3_5 runtime four-builder shape.
"""

from __future__ import annotations

from typing import Any

from vrl.generation.diffusion import (
    DiffusionChunkExecutorBase,
)
from vrl.models.diffusion.capabilities import diffusion_family_capability
from vrl.models.diffusion.common.vae_decode_memory import (
    apply_generation_memory_policy,
)
from vrl.models.interfaces.runtime import (
    RuntimeBuildSpec,
    RuntimeBundle,
)
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

logger = init_logger(__name__)
QWEN_IMAGE_FAMILY_CAPABILITY = diffusion_family_capability("qwen_image", "t2i")


def extract_qwen_image_runtime_spec(
    cfg: Any, device: Any, weight_dtype: Any,
) -> RuntimeBuildSpec:
    """Slice the runtime-relevant subset out of a whole RL cfg."""
    return extract_runtime_spec(
        cfg,
        device,
        weight_dtype,
        task_variant="t2i",
    )


def build_qwen_image_runtime_bundle(spec: RuntimeBuildSpec) -> RuntimeBundle:
    """Generic build: dispatch the backend model by runtime spec."""
    from vrl.models.diffusion.qwen_image.model import QwenImageModel

    logger.info("Building qwen_image runtime bundle")
    use_lora = spec.use_lora
    model = QwenImageModel.from_spec(spec)

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

    return RuntimeBundle(
        model=model,
        trainable_modules=model.trainable_modules,
        scheduler=model.scheduler,
        raw_handle=model.raw_handle,
        runtime_caps={
            "family_capability": QWEN_IMAGE_FAMILY_CAPABILITY.to_dict(),
            "supports_reference_conditioning": False,
        },
        metadata={
            "model_path": spec.model_name_or_path,
            "family": QWEN_IMAGE_FAMILY_CAPABILITY.family,
            "task_variant": spec.task_variant,
            "dtype": str(spec.dtype),
            "use_lora": use_lora,
            **full_generation_bundle_metadata(),
            **apply_generation_memory_policy(
                model,
                memory_config=getattr(spec, "memory", None),
                owner="Qwen-Image VAE",
            ),
        },
    )


def build_qwen_image_replay_runtime_bundle(spec: RuntimeBuildSpec) -> RuntimeBundle:
    """Build the trainer replay bundle without loading Qwen-Image prompt/VAE modules."""

    from vrl.models.diffusion.qwen_image.model import QwenImageReplayModel

    logger.info(
        "Building qwen_image replay runtime bundle from %s",
        spec.model_name_or_path,
    )
    model = QwenImageReplayModel(
        transformer=load_diffusers_transformer(
            spec,
            "QwenImageTransformer2DModel",
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
            "family_capability": QWEN_IMAGE_FAMILY_CAPABILITY.to_dict(),
            "supports_reference_conditioning": False,
        },
        metadata={
            "model_path": spec.model_name_or_path,
            "family": QWEN_IMAGE_FAMILY_CAPABILITY.family,
            "task_variant": spec.task_variant,
            "dtype": str(spec.dtype),
            "use_lora": use_lora,
            **minimal_replay_bundle_metadata(),
        },
    )


def build_qwen_image_runtime_bundle_from_cfg(
    cfg: Any, device: Any, weight_dtype: Any,
) -> RuntimeBundle:
    """Outer convenience: whole-cfg → spec → bundle."""
    spec = extract_qwen_image_runtime_spec(cfg, device, weight_dtype)
    return build_qwen_image_runtime_bundle(spec)


def build_qwen_image_replay_runtime_bundle_from_cfg(
    cfg: Any,
    device: Any,
    weight_dtype: Any,
) -> RuntimeBundle:
    """Outer convenience: whole-cfg → spec → replay bundle."""
    spec = extract_qwen_image_runtime_spec(cfg, device, weight_dtype)
    return build_qwen_image_replay_runtime_bundle(spec)


class QwenImageChunkExecutor(DiffusionChunkExecutorBase):
    """Diffusion executor for Qwen-Image text-to-image rollouts.

    All encoded values (prompt/negative embeds + masks) are batched tensors or
    None, so the base ``build_chunk_encoded`` repeat path needs no override
    (unlike FLUX's batch-shared ``text_ids``).
    """

    family: str = "qwen_image"
    task: str = "t2i"
    family_capability = QWEN_IMAGE_FAMILY_CAPABILITY
    default_num_frames: int = 1
    default_max_sequence_length: int = 1024

    def __init__(
        self,
        model: Any,  # QwenImageModel
        *,
        sample_batch_size: int = 8,
    ) -> None:
        self.model = model
        self.default_sample_batch_size = max(1, int(sample_batch_size))


__all__ = [
    "QwenImageChunkExecutor",
    "build_qwen_image_replay_runtime_bundle",
    "build_qwen_image_replay_runtime_bundle_from_cfg",
    "build_qwen_image_runtime_bundle",
    "build_qwen_image_runtime_bundle_from_cfg",
    "extract_qwen_image_runtime_spec",
]
