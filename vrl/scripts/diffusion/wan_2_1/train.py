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

    global_reference = str(OmegaConf.select(cfg, "model.reference_image", default="") or "")
    manifest_path = Path(str(OmegaConf.select(cfg, "data.manifest", default="manifest")))

    for row_index, example in enumerate(examples):
        raw_path = str(getattr(example, "reference_image", "") or "").strip()
        if not raw_path:
            if global_reference:
                continue
            raise ValueError(
                f"{manifest_path}: row {row_index} is missing required field "
                "reference_image",
            )
        if not Path(raw_path).exists():
            raise FileNotFoundError(
                f"{manifest_path}: row {row_index} reference_image does not exist: "
                f"{raw_path}",
            )
    return {}


__all__ = ["train_wan_2_1_i2v_grpo"]
