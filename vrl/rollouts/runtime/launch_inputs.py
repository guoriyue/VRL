"""Compatibility builders for rollout runtime inputs."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from vrl.generation.runtime.launch_inputs import (
    GenerationRuntimeInputs,
    build_generation_runtime_inputs,
)
from vrl.rollouts.family_registry import get_rollout_family_entry

RolloutRuntimeInputs = GenerationRuntimeInputs


def build_rollout_runtime_inputs(
    cfg: Any,
    family: str,
    *,
    weight_dtype: Any,
    executor_kwargs: Mapping[str, Any] | None = None,
    policy_version: int = 0,
) -> RolloutRuntimeInputs:
    """Build generation runtime inputs from the rollout family registry."""

    return build_generation_runtime_inputs(
        cfg,
        get_rollout_family_entry(family),
        weight_dtype=weight_dtype,
        executor_kwargs=executor_kwargs,
        policy_version=policy_version,
    )


__all__ = ["RolloutRuntimeInputs", "build_rollout_runtime_inputs"]
