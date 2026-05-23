"""SD 3.5 GRPO training recipe."""

from __future__ import annotations

from typing import Any

from omegaconf import DictConfig

from vrl.scripts.common.online import run_online_recipe
from vrl.scripts.common.types import OnlineRecipeDefinition
from vrl.trainers.checkpointing import LORA_WEIGHTS_NAME
from vrl.trainers.precision import torch_dtype_for_trainer_precision
from vrl.utils.cuda_memory import empty_cuda_cache


async def train_sd3_5_grpo(cfg: DictConfig) -> None:
    """Run SD 3.5 GRPO training driven by the common online recipe."""

    await run_online_recipe(
        cfg,
        OnlineRecipeDefinition(
            family="sd3_5",
            build_bundle=_build_bundle,
            build_replay_bundle=_build_replay_bundle,
            after_bundle_built=_after_bundle_built,
            reference_model_getter=_reference_model,
            export_modules_getter=_export_modules,
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
    transformer = bundle.model.transformer
    if bool(cfg.actor.gradient_checkpointing):
        transformer.enable_gradient_checkpointing()
    _offload_driver_frozen_modules(bundle.model)


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


def _resolve_weight_dtype(cfg: DictConfig, trainer_config: Any, torch: Any) -> Any:
    del cfg
    return torch_dtype_for_trainer_precision(trainer_config, torch)


def _offload_driver_frozen_modules(policy: object) -> None:
    """Move frozen driver-only modules off CUDA before Ray workers load."""

    pipeline = getattr(policy, "_pipeline", None)
    if pipeline is None:
        return

    for name in ("text_encoder", "text_encoder_2", "text_encoder_3", "vae"):
        module = getattr(pipeline, name, None)
        if module is not None:
            module.to("cpu")

    empty_cuda_cache()


__all__ = ["train_sd3_5_grpo"]
