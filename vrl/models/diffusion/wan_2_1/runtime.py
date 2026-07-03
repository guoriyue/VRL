"""Wan 2.1 family runtime.

The runtime picks the backend model class by task variant (t2v vs i2v).
Backend imports live inside the model's ``from_spec`` so the shared runtime
does not import diffusers or wan-library backends eagerly.
"""

from __future__ import annotations

from typing import Any

from vrl.generation.diffusion import DiffusionChunkExecutorBase
from vrl.generation.diffusion.executor import ReferenceConditionedChunks
from vrl.models.diffusion.capabilities import diffusion_family_capability
from vrl.models.interfaces.runtime import (
    RuntimeBuildSpec,
    RuntimeBundle,
)
from vrl.models.loader import (
    load_diffusers_scheduler,
    load_diffusers_transformer,
)
from vrl.models.replay_loading import (
    minimal_replay_bundle_metadata,
)
from vrl.utils.logging import init_logger

logger = init_logger(__name__)
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


def build_wan_2_1_replay_runtime_bundle(spec: RuntimeBuildSpec) -> RuntimeBundle:
    """Build the trainer replay bundle without loading Wan text/VAE modules."""

    from vrl.models.diffusion.wan_2_1.model import (
        WanI2VReplayModel,
        WanT2VReplayModel,
    )

    task_variant = _normalize_task_variant(spec.task_variant)
    replay_cls = WanI2VReplayModel if task_variant == "i2v" else WanT2VReplayModel
    boundary_ratio = _boundary_ratio_from_spec(spec)
    transformer_2 = (
        load_diffusers_transformer(
            spec,
            "WanTransformer3DModel",
            subfolder="transformer_2",
        )
        if boundary_ratio is not None
        else None
    )
    trainable_transformers = (spec.model_config or {}).get("trainable_transformers")

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
        transformer_2=transformer_2,
        boundary_ratio=boundary_ratio,
        trainable_transformers=trainable_transformers,
        scheduler=load_diffusers_scheduler(
            spec,
            "UniPCMultistepScheduler",
        ),
        device=spec.device,
    )

    use_lora = spec.use_lora
    if use_lora:
        model.apply_lora(spec)
    else:
        model.apply_full_finetune()

    compile_cfg = spec.torch_compile or {}
    if compile_cfg.get("enable"):
        model.torch_compile_transformer(compile_cfg["mode"])

    return RuntimeBundle(
        model=model,
        trainable_modules=model.trainable_modules,
        scheduler=model.scheduler,
        raw_handle=None,
        runtime_caps={
            "supports_reference_conditioning": task_variant == "i2v",
        },
        metadata={
            "model_path": spec.model_name_or_path,
            "family": (
                WAN_2_1_I2V_FAMILY_CAPABILITY.family
                if task_variant == "i2v"
                else WAN_2_1_FAMILY_CAPABILITY.family
            ),
            "task_variant": task_variant,
            "dtype": str(spec.dtype),
            "use_lora": use_lora,
            **minimal_replay_bundle_metadata(),
        },
    )


def build_wan_2_1_replay_runtime_bundle_from_cfg(
    cfg: Any,
    device: Any,
    weight_dtype: Any,
) -> RuntimeBundle:
    """Outer convenience: whole-cfg -> spec -> replay bundle.

    The spec comes from the generic descriptor extractor (task_variant is
    decided by which wan registry entry cfg.model.family selects); only the
    multi-transformer replay construction itself stays hand-written.
    """
    from vrl.models.diffusion.build import extract_family_runtime_spec

    return build_wan_2_1_replay_runtime_bundle(
        extract_family_runtime_spec(cfg, device, weight_dtype),
    )


"""Wan 2.1 diffusion pipeline executor."""


class Wan_2_1ChunkExecutor(DiffusionChunkExecutorBase):
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
        samples_per_chunk: int = 1,
    ) -> None:
        self.model = model
        self.default_samples_per_chunk = max(1, int(samples_per_chunk))

class Wan_2_1I2VChunkExecutor(ReferenceConditionedChunks, DiffusionChunkExecutorBase):
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
        samples_per_chunk: int = 1,
    ) -> None:
        self.model = model
        self.reference_image = reference_image
        self.default_samples_per_chunk = max(1, int(samples_per_chunk))

def _normalize_task_variant(task_variant: str | None) -> str:
    # Accept any non-i2v value as t2v so generic test fixtures that use a
    # neutral placeholder like "t2i" do not need a special-case branch per
    # family. Real i2v dispatch only fires for the explicit i2v aliases below
    # or when cfg.model.family/task_variant declares i2v.
    text = str(task_variant or "t2v").strip().lower()
    if text in {"image_to_video", "image-to-video", "i2v"}:
        return "i2v"
    return "t2v"


def _boundary_ratio_from_spec(spec: RuntimeBuildSpec) -> float | None:
    from vrl.models.diffusion.wan_2_1.model import _optional_float

    model_config = spec.model_config or {}
    if "boundary_ratio" in model_config:
        return _optional_float(model_config.get("boundary_ratio"), "model.boundary_ratio")
    if "Wan2.2" not in str(spec.model_name_or_path):
        return None
    return _load_boundary_ratio_from_pipeline_config(spec)


def _load_boundary_ratio_from_pipeline_config(spec: RuntimeBuildSpec) -> float | None:
    from diffusers import DiffusionPipeline

    from vrl.models.diffusion.wan_2_1.model import _optional_float

    config = DiffusionPipeline.load_config(spec.model_name_or_path)
    return _optional_float(config.get("boundary_ratio"), "pipeline boundary_ratio")


__all__ = [
    "Wan_2_1ChunkExecutor",
    "Wan_2_1I2VChunkExecutor",
    "build_wan_2_1_replay_runtime_bundle",
    "build_wan_2_1_replay_runtime_bundle_from_cfg",
    "build_wan_2_1_runtime_bundle",
    "build_wan_2_1_runtime_bundle_from_cfg",
    "extract_wan_2_1_runtime_spec",
]
