"""Family-generic diffusion GRPO training recipe.

Registry-descriptor families (those with a ``DiffusionFamilyBuild`` on their
rollout registry entry) train through this single entrypoint instead of a
per-family ``train.py``: the family comes from ``cfg.model.family`` and the
bundles from the generic descriptor-driven builders. The per-family recipes
that remain (sd3_5, flux NFT, ...) either predate the descriptor path or carry
recipe-level hooks this generic body must not learn.

YAML wiring: ``trainer.entrypoint: vrl.scripts.diffusion.train:train_diffusion_grpo``.
"""

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


async def train_diffusion_grpo(cfg: DictConfig) -> None:
    """Run GRPO training for any registry-descriptor diffusion family."""

    from vrl.rollouts.families.registry import normalize_rollout_family

    family = normalize_rollout_family(str(cfg.model.family))
    await run_online_recipe(
        cfg,
        OnlineRecipeDefinition(
            family=family,
            build_bundle=_build_bundle,
            build_replay_bundle=_build_replay_bundle,
            after_bundle_built=_after_bundle_built,
            reference_model_getter=default_reference_model,
            export_modules_getter=export_transformer_lora,
        ),
    )


def _build_bundle(cfg: DictConfig, device: Any, weight_dtype: Any) -> Any:
    from vrl.models.diffusion.build import build_family_runtime_bundle_from_cfg

    return build_family_runtime_bundle_from_cfg(cfg, device, weight_dtype)


def _build_replay_bundle(cfg: DictConfig, device: Any, weight_dtype: Any) -> Any:
    from vrl.models.diffusion.build import (
        build_family_replay_runtime_bundle_from_cfg,
    )

    return build_family_replay_runtime_bundle_from_cfg(cfg, device, weight_dtype)


def _after_bundle_built(bundle: Any, cfg: DictConfig) -> None:
    enable_transformer_gradient_checkpointing(bundle, cfg)


__all__ = ["train_diffusion_grpo"]
