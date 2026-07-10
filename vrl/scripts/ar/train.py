"""Family-generic AR GRPO training recipe.

AR families train through this single entrypoint instead of a per-family
``train.py`` (the janus_pro / nextstep_1 scripts it replaces were pure
forwarding — the family string was their only difference, and the r1 variant
already derived its ar_task from ``cfg.model.family`` inside the extractor).
The bundle builders come from the rollout family registry: ``runtime_builder``
/ ``runtime_spec_extractor`` (the same strings the Ray workers import) plus
``replay_runtime_builder`` for the trainer-side replay bundle.

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

    from vrl.rollouts.families.registry import normalize_rollout_family

    family = normalize_rollout_family(str(cfg.model.family))
    await run_online_recipe(
        cfg,
        OnlineRecipeDefinition(
            family=family,
            build_bundle=_build_bundle,
            build_replay_bundle=_build_replay_bundle,
            export_modules_getter=export_language_model_lora,
        ),
    )


def _resolve_family_imports(cfg: DictConfig) -> tuple[Any, Any, Any]:
    from vrl.rollouts.families.registry import (
        get_rollout_family_entry,
        normalize_rollout_family,
    )
    from vrl.utils.config import import_from_path

    entry = get_rollout_family_entry(normalize_rollout_family(str(cfg.model.family)))
    if entry.replay_runtime_builder is None:
        raise ValueError(
            f"rollout family {entry.family!r} declares no replay_runtime_builder; "
            "the generic AR entrypoint needs it for the trainer replay bundle",
        )
    return (
        import_from_path(entry.runtime_builder),
        import_from_path(entry.replay_runtime_builder),
        import_from_path(entry.runtime_spec_extractor),
    )


def _build_bundle(cfg: DictConfig, device: Any, weight_dtype: Any) -> Any:
    build, _, extract = _resolve_family_imports(cfg)
    return build(extract(cfg, device, weight_dtype))


def _build_replay_bundle(cfg: DictConfig, device: Any, weight_dtype: Any) -> Any:
    _, build_replay, extract = _resolve_family_imports(cfg)
    return build_replay(extract(cfg, device, weight_dtype))


__all__ = ["train_ar_grpo"]
