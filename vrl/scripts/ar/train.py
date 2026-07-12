"""Family-generic AR GRPO training recipe.

AR families train through this single entrypoint instead of a per-family
``train.py`` (the janus_pro / nextstep_1 scripts it replaces were pure
forwarding — the family string was their only difference). The online family
comes from ``resolve_online_family``, which also maps algorithm-selected
variants (janus_pro + token_grpo_multisegment -> janus_pro_r1) that plain
``model.family`` normalization would miss.
The replay bundle builder comes from the rollout family registry:
``replay_runtime_builder`` + ``model_build_resolver`` (the same resolver
string the Ray workers import for the rollout side).

Mirrors ``vrl/scripts/diffusion/train.py`` (the registry-descriptor diffusion
entrypoint) so both sides read the same way.

YAML wiring: ``trainer.entrypoint: vrl.scripts.ar.train:train_ar_grpo``.
"""

from __future__ import annotations

from typing import Any

from omegaconf import DictConfig

from vrl.scripts.common.online import (
    export_language_model_lora,
    run_online_recipe,
)
from vrl.scripts.common.types import OnlineRecipeDefinition


async def train_ar_grpo(cfg: DictConfig) -> None:
    """Run token GRPO for any AR family (reward chosen by config)."""

    # resolve_online_family, not plain normalization: the shipped R1 recipe
    # keeps model.family=janus_pro and selects janus_pro_r1 via
    # algorithm.kind=token_grpo_multisegment. Passing the base family here
    # short-circuits `family or resolve_online_family(cfg)` in the factory
    # and the multisegment guard rejects the run.
    from vrl.scripts.common.factory import resolve_online_family

    family = resolve_online_family(cfg)
    await run_online_recipe(
        cfg,
        OnlineRecipeDefinition(
            family=family,
            build_replay_bundle=_build_replay_bundle,
            export_modules_getter=export_language_model_lora,
        ),
    )


def _build_replay_bundle(cfg: DictConfig, device: Any) -> Any:
    from vrl.rollouts.families.registry import get_rollout_family_entry
    from vrl.scripts.common.factory import resolve_online_family
    from vrl.utils.config import import_from_path

    entry = get_rollout_family_entry(resolve_online_family(cfg))
    if entry.replay_runtime_builder is None:
        raise ValueError(
            f"rollout family {entry.family!r} declares no replay_runtime_builder; "
            "the generic AR entrypoint needs it for the trainer replay bundle",
        )
    build_replay = import_from_path(entry.replay_runtime_builder)
    resolve_build = import_from_path(entry.model_build_resolver)
    return build_replay(resolve_build(cfg, device, for_rollout=False))


__all__ = ["train_ar_grpo"]
