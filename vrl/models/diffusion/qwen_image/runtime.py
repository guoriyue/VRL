"""Qwen-Image family runtime.

Backend imports live inside the model's ``from_spec`` so the shared runtime does
not import diffusers eagerly. Mirrors the sd3_5 runtime four-builder shape.
"""

from __future__ import annotations

from typing import Any

from vrl.generation.diffusion import (
    DiffusionChunkExecutorBase,
)
from vrl.models.diffusion.build import (
    build_diffusion_replay_runtime_bundle,
    build_diffusion_runtime_bundle,
)
from vrl.models.diffusion.capabilities import diffusion_family_capability
from vrl.models.interfaces.runtime import (
    RuntimeBuildSpec,
    RuntimeBundle,
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
    """Thin family stub over the shared diffusion runtime builder."""
    from vrl.models.diffusion.qwen_image.model import QwenImageModel

    logger.info("Building qwen_image runtime bundle")
    return build_diffusion_runtime_bundle(
        spec,
        model_cls=QwenImageModel,
        capability=QWEN_IMAGE_FAMILY_CAPABILITY,
        memory_owner="Qwen-Image VAE",
    )


def build_qwen_image_replay_runtime_bundle(spec: RuntimeBuildSpec) -> RuntimeBundle:
    """Thin family stub over the shared diffusion replay builder."""
    from vrl.models.diffusion.qwen_image.model import QwenImageReplayModel

    logger.info(
        "Building qwen_image replay runtime bundle from %s",
        spec.model_name_or_path,
    )
    return build_diffusion_replay_runtime_bundle(
        spec,
        replay_cls=QwenImageReplayModel,
        transformer_classname="QwenImageTransformer2DModel",
        capability=QWEN_IMAGE_FAMILY_CAPABILITY,
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
        samples_per_chunk: int = 8,
    ) -> None:
        self.model = model
        self.default_samples_per_chunk = max(1, int(samples_per_chunk))


__all__ = [
    "QwenImageChunkExecutor",
    "build_qwen_image_replay_runtime_bundle",
    "build_qwen_image_replay_runtime_bundle_from_cfg",
    "build_qwen_image_runtime_bundle",
    "build_qwen_image_runtime_bundle_from_cfg",
    "extract_qwen_image_runtime_spec",
]
