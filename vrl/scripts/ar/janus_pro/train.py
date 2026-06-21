"""Janus-Pro online GRPO training recipes."""

from __future__ import annotations

from typing import Any

from omegaconf import DictConfig

from vrl.scripts.common.online import (
    export_language_model_lora,
    run_online_recipe,
)
from vrl.scripts.common.types import OnlineRecipeDefinition


async def train_janus_pro_grpo(cfg: DictConfig) -> None:
    """Run Janus-Pro token GRPO (reward chosen by config, not this entrypoint)."""

    await _run_janus_recipe(cfg, family="janus_pro")


async def train_janus_pro_r1_grpo(cfg: DictConfig) -> None:
    """Run Janus-Pro-R1 multi-segment token GRPO (reward chosen by config)."""

    await _run_janus_recipe(cfg, family="janus_pro_r1")


async def _run_janus_recipe(cfg: DictConfig, *, family: str) -> None:
    await run_online_recipe(
        cfg,
        OnlineRecipeDefinition(
            family=family,
            build_bundle=lambda cfg, device, dtype: _build_bundle(
                cfg,
                device,
                dtype,
                family=family,
            ),
            build_replay_bundle=lambda cfg, device, dtype: _build_replay_bundle(
                cfg,
                device,
                dtype,
                family=family,
            ),
            export_modules_getter=export_language_model_lora,
        ),
    )


def _build_bundle(cfg: DictConfig, device: Any, weight_dtype: Any, *, family: str) -> Any:
    from vrl.models.ar.janus_pro.runtime import (
        build_janus_pro_runtime_bundle,
        extract_janus_pro_runtime_spec,
    )

    spec = extract_janus_pro_runtime_spec(cfg, device, weight_dtype)
    if family == "janus_pro_r1":
        spec.ar_task = "ar_t2i_r1"
    return build_janus_pro_runtime_bundle(spec)


def _build_replay_bundle(
    cfg: DictConfig,
    device: Any,
    weight_dtype: Any,
    *,
    family: str,
) -> Any:
    from vrl.models.ar.janus_pro.runtime import (
        build_janus_pro_replay_runtime_bundle,
        extract_janus_pro_runtime_spec,
    )

    spec = extract_janus_pro_runtime_spec(cfg, device, weight_dtype)
    if family == "janus_pro_r1":
        spec.ar_task = "ar_t2i_r1"
    return build_janus_pro_replay_runtime_bundle(spec)


__all__ = [
    "train_janus_pro_grpo",
    "train_janus_pro_r1_grpo",
]
