"""FLUX.1 DiffusionNFT training recipe (GRPO uses the generic diffusion recipe)."""

from __future__ import annotations

from omegaconf import DictConfig

from vrl.scripts.common.online import (
    enable_transformer_gradient_checkpointing,
    export_transformer_lora,
    run_online_recipe,
)
from vrl.scripts.common.types import OnlineRecipeDefinition
from vrl.scripts.diffusion.train import build_replay_bundle


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
            build_replay_bundle=build_replay_bundle,
            after_bundle_built=enable_transformer_gradient_checkpointing,
            export_modules_getter=export_transformer_lora,
        ),
    )


__all__ = ["train_flux_diffusion_nft"]
