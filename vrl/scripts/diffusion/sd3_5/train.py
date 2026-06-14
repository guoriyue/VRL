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
from vrl.trainers.precision import torch_dtype_for_trainer_precision


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


def _resolve_weight_dtype(cfg: DictConfig, trainer_config: Any, torch: Any) -> Any:
    del cfg
    return torch_dtype_for_trainer_precision(trainer_config, torch)


__all__ = ["train_sd3_5_grpo"]
