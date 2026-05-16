"""Wan 2.1 GRPO training recipe."""

from __future__ import annotations

from typing import Any

from omegaconf import DictConfig

from vrl.scripts.common.online import run_online_recipe
from vrl.scripts.common.types import OnlineRecipeDefinition
from vrl.trainers.checkpointing import LORA_WEIGHTS_NAME


async def train_wan_2_1_grpo(cfg: DictConfig) -> None:
    """Run Wan-family GRPO training driven by a merged YAML config."""

    await run_online_recipe(
        cfg,
        OnlineRecipeDefinition(
            family="wan_2_1",
            build_bundle=_build_bundle,
            build_replay_bundle=_build_replay_bundle,
            after_bundle_built=_after_bundle_built,
            reference_model_getter=_reference_model,
            export_modules_getter=_export_modules,
        ),
    )


def _build_bundle(cfg: DictConfig, device: Any, weight_dtype: Any) -> Any:
    from vrl.models.diffusion.wan_2_1.runtime import build_wan_2_1_runtime_bundle_from_cfg

    return build_wan_2_1_runtime_bundle_from_cfg(cfg, device, weight_dtype)


def _build_replay_bundle(cfg: DictConfig, device: Any, weight_dtype: Any) -> Any:
    from vrl.models.diffusion.wan_2_1.runtime import (
        build_wan_2_1_replay_runtime_bundle_from_cfg,
    )

    return build_wan_2_1_replay_runtime_bundle_from_cfg(cfg, device, weight_dtype)


def _after_bundle_built(bundle: Any, cfg: DictConfig) -> None:
    transformer = bundle.model.transformer
    if bool(cfg.actor.gradient_checkpointing):
        transformer.enable_gradient_checkpointing()


def _reference_model(bundle: Any, cfg: DictConfig) -> Any | None:
    init_kl_coef = float(getattr(cfg.algorithm, "init_kl_coef", 0.0))
    if bool(cfg.model.use_lora) and init_kl_coef > 0:
        return bundle.model
    return None


def _export_modules(bundle: Any, cfg: DictConfig) -> dict[str, Any] | None:
    transformer = bundle.model.transformer
    if bool(cfg.model.use_lora) and hasattr(transformer, "save_pretrained"):
        return {LORA_WEIGHTS_NAME: transformer}
    return None


__all__ = ["train_wan_2_1_grpo"]
