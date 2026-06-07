"""SD 3.5 GRPO training recipe."""

from __future__ import annotations

from typing import Any

from omegaconf import DictConfig

from vrl.scripts.common.online import (
    default_reference_model,
    enable_transformer_gradient_checkpointing,
    export_transformer_lora,
    run_online_recipe,
)
from vrl.scripts.common.types import OnlineRecipeDefinition
from vrl.trainers.frozen_module import (
    FrozenModuleOffload,
    frozen_module_offload_from_config,
    frozen_module_offload_metadata,
    park_frozen_modules,
)
from vrl.trainers.precision import torch_dtype_for_trainer_precision
from vrl.utils.config import plain_mapping


async def train_sd3_5_grpo(cfg: DictConfig) -> None:
    """Run SD 3.5 GRPO training driven by the common online recipe."""

    await run_online_recipe(
        cfg,
        OnlineRecipeDefinition(
            family="sd3_5",
            build_bundle=_build_bundle,
            build_replay_bundle=_build_replay_bundle,
            after_bundle_built=_after_bundle_built,
            reference_model_getter=default_reference_model,
            export_modules_getter=export_transformer_lora,
            weight_dtype_getter=_resolve_weight_dtype,
        ),
    )


def _build_bundle(cfg: DictConfig, device: Any, weight_dtype: Any) -> Any:
    from vrl.models.diffusion.sd3_5.runtime import build_sd3_5_runtime_bundle_from_cfg

    return build_sd3_5_runtime_bundle_from_cfg(cfg, device, weight_dtype)


def _build_replay_bundle(cfg: DictConfig, device: Any, weight_dtype: Any) -> Any:
    from vrl.models.diffusion.sd3_5.runtime import (
        build_sd3_5_replay_runtime_bundle_from_cfg,
    )

    return build_sd3_5_replay_runtime_bundle_from_cfg(cfg, device, weight_dtype)


def _after_bundle_built(bundle: Any, cfg: DictConfig) -> None:
    enable_transformer_gradient_checkpointing(bundle, cfg)
    bundle.metadata.update(_offload_driver_frozen_modules(bundle.model, cfg))


def _resolve_weight_dtype(cfg: DictConfig, trainer_config: Any, torch: Any) -> Any:
    del cfg
    return torch_dtype_for_trainer_precision(trainer_config, torch)


def _offload_driver_frozen_modules(
    model: object,
    cfg: DictConfig | None = None,
) -> dict[str, dict[str, Any]]:
    """Move frozen driver-only modules off CUDA before Ray workers load."""

    pipeline = getattr(model, "_pipeline", None)
    if pipeline is None:
        return {}
    defaults = FrozenModuleOffload(
        enable=True,
        module_names=(
            "text_encoder",
            "text_encoder_2",
            "text_encoder_3",
            "vae",
        ),
        empty_cache=True,
    )
    section = _extract_frozen_offload_config(cfg)
    offload = frozen_module_offload_from_config(section, defaults=defaults)
    moved = park_frozen_modules(pipeline, offload)
    return frozen_module_offload_metadata(offload, moved=moved)


def _extract_frozen_offload_config(cfg: DictConfig | None) -> dict[str, Any] | None:
    """Return the trainer lifecycle memory section owned by SD3 training."""

    if cfg is None:
        return None
    model_cfg = getattr(cfg, "model", None)
    if model_cfg is None:
        return None
    raw = getattr(model_cfg, "memory", None)
    if raw is None:
        return None
    memory = plain_mapping(raw, field_name="model.memory")
    section = memory.get("frozen_offload")
    if section is None:
        return None
    return plain_mapping(section, field_name="model.memory.frozen_offload")


__all__ = ["train_sd3_5_grpo"]
