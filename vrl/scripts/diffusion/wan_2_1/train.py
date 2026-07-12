"""Wan I2V GRPO training recipe.

Wan T2V trains through the generic ``vrl.scripts.diffusion.train:
train_diffusion_grpo`` entrypoint (family dispatch via ``model.family``); only
the I2V variant keeps a recipe here for its per-sample reference-image
collector hook.
"""

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


async def train_wan_2_1_i2v_grpo(cfg: DictConfig) -> None:
    """Run Wan I2V GRPO training driven by a merged YAML config."""

    await run_online_recipe(
        cfg,
        OnlineRecipeDefinition(
            family="wan_2_1_i2v",
            build_replay_bundle=build_replay_bundle,
            after_bundle_built=enable_transformer_gradient_checkpointing,
            reference_model_getter=default_reference_model,
            export_modules_getter=export_transformer_lora,
            collector_kwargs_getter=_i2v_collector_kwargs,
        ),
    )


def _i2v_collector_kwargs(cfg: DictConfig, examples: list[Any]) -> dict[str, Any]:
    """Validate per-sample reference images before rollout collection.

    Paths were already resolved at prompt load time (run_online_recipe ->
    ``_resolve_reference_artifacts``); this hook keeps only the wan-specific
    checks: missing-vs-global fallback and existence.
    """

    manifest_path = Path(str(OmegaConf.select(cfg, "data.manifest", default="manifest")))
    conditioning = OmegaConf.select(cfg, "data.preprocessing.conditioning", default=None)
    if conditioning != "reference_image":
        raise ValueError(
            "Wan I2V requires data.preprocessing.conditioning=reference_image",
        )
    require_reference_images(
        examples,
        manifest_path=manifest_path,
        default_reference_image=OmegaConf.select(
            cfg,
            "data.preprocessing.reference_image",
            default=None,
        ),
    )
    return {}


__all__ = ["train_wan_2_1_i2v_grpo"]
