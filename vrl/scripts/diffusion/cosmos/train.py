"""Cosmos online training recipes."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from vrl.scripts.common.online import (
    default_reference_model,
    enable_transformer_gradient_checkpointing,
    export_transformer_lora,
    run_online_recipe,
)
from vrl.scripts.common.types import OnlineRecipeDefinition
from vrl.scripts.diffusion.train import build_replay_bundle
from vrl.trainers.data.artifacts import require_reference_images


async def train_cosmos_predict2_grpo(cfg: DictConfig) -> None:
    """Run Cosmos Predict2 GRPO through the common online recipe."""

    await run_online_recipe(
        cfg,
        OnlineRecipeDefinition(
            family="cosmos-predict2",
            build_replay_bundle=build_replay_bundle,
            after_bundle_built=_after_bundle_built,
            reference_model_getter=default_reference_model,
            export_modules_getter=export_transformer_lora,
            collector_kwargs_getter=_predict2_collector_kwargs,
        ),
    )


async def train_cosmos_predict25_grpo(cfg: DictConfig) -> None:
    """Run Cosmos Predict2.5 GRPO through the common online recipe.

    The paper's RL recipe (§4.2.2) is GRPO; the NFT entrypoint below is the
    likelihood-free variant. Predict2.5 is text-to-world, so unlike the
    Predict2 Video2World entrypoint there is no reference-image collector
    wiring here.
    """

    await run_online_recipe(
        cfg,
        OnlineRecipeDefinition(
            family="cosmos-predict2.5",
            build_replay_bundle=build_replay_bundle,
            after_bundle_built=_after_bundle_built,
            reference_model_getter=default_reference_model,
            export_modules_getter=export_transformer_lora,
        ),
    )


async def train_cosmos_predict25_diffusion_nft(cfg: DictConfig) -> None:
    """Run Cosmos Predict2.5 DiffusionNFT through the common online recipe."""

    await run_online_recipe(
        cfg,
        OnlineRecipeDefinition(
            family="cosmos-predict2.5",
            build_replay_bundle=build_replay_bundle,
            after_bundle_built=_after_bundle_built,
            export_modules_getter=export_transformer_lora,
        ),
    )


def _after_bundle_built(bundle: Any, cfg: DictConfig) -> None:
    enable_transformer_gradient_checkpointing(bundle, cfg, require_method=False)


def _predict2_collector_kwargs(cfg: DictConfig, examples: Any) -> dict[str, Any]:
    conditioning = OmegaConf.select(cfg, "data.preprocessing.conditioning", default=None)
    if conditioning != "reference_image":
        raise ValueError(
            "Cosmos Predict2 requires data.preprocessing.conditioning=reference_image",
        )
    require_reference_images(
        examples,
        manifest_path=Path(str(cfg.data.manifest)),
        default_reference_image=OmegaConf.select(
            cfg,
            "data.preprocessing.reference_image",
            default=None,
        ),
    )
    return {}


__all__ = [
    "train_cosmos_predict2_grpo",
    "train_cosmos_predict25_diffusion_nft",
    "train_cosmos_predict25_grpo",
]
