"""FLUX.1 DiffusionNFT training recipe (GRPO uses the generic diffusion recipe)."""

from __future__ import annotations

from typing import Any

from omegaconf import DictConfig

from vrl.scripts.common.online import (
    enable_transformer_gradient_checkpointing,
    export_transformer_lora,
    run_online_recipe,
)
from vrl.scripts.common.types import OnlineRecipeDefinition
from vrl.trainers.precision import torch_dtype_for_trainer_precision


async def train_flux_diffusion_nft(cfg: DictConfig) -> None:
    """Run FLUX.1 DiffusionNFT training driven by the common online recipe.

    Differs from the generic GRPO recipe in one way: no
    ``reference_model_getter`` — NFT's KL reference is the base model behind
    ``model.disable_adapter()``, not a separately built frozen reference model.
    The frozen ``previous`` adapter is config-driven: NFT experiments must set
    ``model.nft_previous_adapter: true`` (FluxModel.apply_lora attaches it; the
    build fails loud if the key is set without LoRA).
    """

    await run_online_recipe(
        cfg,
        OnlineRecipeDefinition(
            family="flux",
            build_bundle=_build_nft_bundle,
            build_replay_bundle=_build_nft_replay_bundle,
            after_bundle_built=_after_bundle_built,
            export_modules_getter=export_transformer_lora,
            weight_dtype_getter=_resolve_weight_dtype,
        ),
    )


def _build_nft_bundle(cfg: DictConfig, device: Any, weight_dtype: Any) -> Any:
    from vrl.models.diffusion.build import build_family_runtime_bundle_from_cfg

    return build_family_runtime_bundle_from_cfg(cfg, device, weight_dtype)


def _build_nft_replay_bundle(cfg: DictConfig, device: Any, weight_dtype: Any) -> Any:
    from vrl.models.diffusion.build import (
        build_family_replay_runtime_bundle_from_cfg,
    )

    return build_family_replay_runtime_bundle_from_cfg(cfg, device, weight_dtype)


def _after_bundle_built(bundle: Any, cfg: DictConfig) -> None:
    enable_transformer_gradient_checkpointing(bundle, cfg)


def _resolve_weight_dtype(cfg: DictConfig, trainer_config: Any, torch: Any) -> Any:
    del cfg
    return torch_dtype_for_trainer_precision(trainer_config, torch)


__all__ = ["train_flux_diffusion_nft"]
