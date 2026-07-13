"""Family-generic AR GRPO training recipe.

AR families train through this single entrypoint instead of a per-family
``train.py`` (the janus_pro / nextstep_1 scripts it replaces were pure
forwarding). ``model.family`` names the exact registry entry; algorithm choice
never rewrites model identity behind the config's back.
The common runner resolves one replay ``ModelBuild`` through the same registry
entry used by Ray rollout workers; this module then invokes that entry's
``replay_runtime_builder`` without reading config again.

Mirrors ``vrl/scripts/diffusion/train.py`` (the registry-descriptor diffusion
entrypoint) so both sides read the same way.

YAML wiring: ``trainer.entrypoint: vrl.scripts.ar.train:train_ar_grpo``.
"""

from __future__ import annotations

from omegaconf import DictConfig

from vrl.models.interfaces import ModelBuild, RuntimeBundle
from vrl.scripts.common.online import (
    export_language_model_lora,
    run_online_recipe,
)
from vrl.scripts.common.types import OnlineRecipeDefinition


async def train_ar_grpo(cfg: DictConfig) -> None:
    """Run token GRPO for any AR family (reward chosen by config)."""

    await run_online_recipe(
        cfg,
        OnlineRecipeDefinition(
            build_replay_bundle=_build_replay_bundle,
            export_modules_getter=export_language_model_lora,
        ),
    )


def _build_replay_bundle(build: ModelBuild) -> RuntimeBundle:
    from vrl.rollouts.families import get_rollout_family_entry
    from vrl.utils.config import import_from_path

    if not build.family:
        raise ValueError("AR replay build requires a resolved family")
    entry = get_rollout_family_entry(build.family)
    if entry.replay_runtime_builder is None:
        raise ValueError(
            f"rollout family {entry.family!r} declares no replay_runtime_builder; "
            "the generic AR entrypoint needs it for the trainer replay bundle",
        )
    return import_from_path(entry.replay_runtime_builder)(build)


__all__ = ["train_ar_grpo"]
