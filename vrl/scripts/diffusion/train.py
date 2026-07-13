"""Family-generic diffusion GRPO training recipe.

Registry-descriptor families (those with a ``DiffusionFamilyBuild`` on their
rollout registry entry) train through this single entrypoint instead of a
per-family ``train.py``: the family comes from ``cfg.model.family`` and the
replay bundle from the generic descriptor-driven builder. The per-family
recipes that remain (cosmos, flux NFT, wan i2v) either predate the descriptor
path or carry recipe-level hooks this generic body must not learn — they
import ``build_replay_bundle`` from here instead of keeping their own
lazy-import copies.

YAML wiring: ``trainer.entrypoint: vrl.scripts.diffusion.train:train_diffusion_grpo``.
"""

from __future__ import annotations

from omegaconf import DictConfig

from vrl.models.interfaces import ModelBuild, RuntimeBundle
from vrl.scripts.common.online import (
    default_reference_model,
    enable_transformer_gradient_checkpointing,
    export_transformer_lora,
    run_online_recipe,
)
from vrl.scripts.common.types import OnlineRecipeDefinition


async def train_diffusion_grpo(cfg: DictConfig) -> None:
    """Run GRPO training for any registry-descriptor diffusion family."""

    await run_online_recipe(
        cfg,
        OnlineRecipeDefinition(
            build_replay_bundle=build_replay_bundle,
            after_bundle_built=enable_transformer_gradient_checkpointing,
            reference_model_getter=default_reference_model,
            export_modules_getter=export_transformer_lora,
        ),
    )


def build_replay_bundle(build: ModelBuild) -> RuntimeBundle:
    """Lazy-import boundary over the generic descriptor-driven replay builder."""

    from vrl.models.diffusion.build import build_family_replay_runtime_bundle

    return build_family_replay_runtime_bundle(build)


__all__ = ["build_replay_bundle", "train_diffusion_grpo"]
