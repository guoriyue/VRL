"""JoyAI-Echo GRPO training recipe (single-shot video flow-matching policy).

Echo's LTX velocity transformer plugs into the shared flow-matching online
recipe exactly like SD3.5/Flux/Wan — its x0->velocity conversion lives in
``EchoModel.forward_step`` and the rest (SDE log-prob replay, GRPO loss, LoRA
weight sync) is family-agnostic. The only Echo-specific seam is gradient
checkpointing, whose flag lives on the inner LTXModel rather than the diffusers
``enable_gradient_checkpointing`` the generic helper drives.
"""

from __future__ import annotations

from typing import Any

from omegaconf import DictConfig, OmegaConf

from vrl.scripts.common.online import (
    default_reference_model,
    export_transformer_lora,
    run_online_recipe,
)
from vrl.scripts.common.types import OnlineRecipeDefinition
from vrl.trainers.precision import torch_dtype_for_trainer_precision


async def train_echo_grpo(cfg: DictConfig) -> None:
    """Run JoyAI-Echo GRPO training driven by the common online recipe."""

    await run_online_recipe(
        cfg,
        OnlineRecipeDefinition(
            family="echo",
            build_bundle=_build_bundle,
            build_replay_bundle=_build_replay_bundle,
            after_bundle_built=_after_bundle_built,
            reference_model_getter=default_reference_model,
            export_modules_getter=export_transformer_lora,
            weight_dtype_getter=_resolve_weight_dtype,
        ),
    )


def _build_bundle(cfg: DictConfig, device: Any, weight_dtype: Any) -> Any:
    from vrl.models.diffusion.echo.runtime import build_echo_runtime_bundle_from_cfg

    return build_echo_runtime_bundle_from_cfg(cfg, device, weight_dtype)


def _build_replay_bundle(cfg: DictConfig, device: Any, weight_dtype: Any) -> Any:
    from vrl.models.diffusion.echo.runtime import (
        build_echo_replay_runtime_bundle_from_cfg,
    )

    return build_echo_replay_runtime_bundle_from_cfg(cfg, device, weight_dtype)


def _after_bundle_built(bundle: Any, cfg: DictConfig) -> None:
    """Enable gradient checkpointing on the LTX velocity blocks when requested.

    Mirrors the generic gate (absent key -> TrainerConfig default) but routes to
    Echo's own ``enable_gradient_checkpointing`` because the LTX transformer does
    not expose the diffusers method the shared helper calls.
    """

    from vrl.trainers.core.types import TrainerConfig

    enabled = OmegaConf.select(cfg, "actor.gradient_checkpointing")
    if enabled is None:
        enabled = TrainerConfig.__dataclass_fields__["gradient_checkpointing"].default
    if bool(enabled):
        bundle.model.enable_gradient_checkpointing()


def _resolve_weight_dtype(cfg: DictConfig, trainer_config: Any, torch: Any) -> Any:
    del cfg
    return torch_dtype_for_trainer_precision(trainer_config, torch)


__all__ = ["train_echo_grpo"]
