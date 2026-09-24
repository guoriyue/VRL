"""Load a qualified reward deployment receipt before training builds scorers.

A receipt (written by ``reward_lab.qualification``) binds a frozen combination,
the scoring recipe it was fitted on, the runtime configuration and measured
raw-axis parity on pinned images. Loading checks the receipt against the
resolved runtime configuration; it does not certify preference quality.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import asdict
from typing import Any

from vrl.config.builders import RewardRuntimeConfig
from vrl.rewards.calibration import FrozenRewardCombination
from vrl.rewards.evaluation import ScoringConfig
from vrl.utils.json_files import canonical_json_sha256


def runtime_recipe(config: RewardRuntimeConfig) -> dict[str, Any]:
    return {
        "components": config.weights,
        "kwargs": config.kwargs,
        "inference": {name: asdict(value) for name, value in config.inference_configs.items()},
    }


def validate_binding(
    config: RewardRuntimeConfig,
    recipe: dict[str, Any],
    frozen: FrozenRewardCombination,
    axis_mapping: Mapping[str, str],
) -> None:
    config.require_online_training()
    if not config.all_external_inference:
        raise ValueError("calibrated deployment currently requires HTTP scoring components")
    if any(weight != 1 for weight in config.weights.values()):
        raise ValueError("calibrated deployment requires unit component weights")
    if set(axis_mapping) != set(frozen.axes) or len(set(axis_mapping.values())) != len(
        axis_mapping
    ):
        raise ValueError("axis mapping must cover frozen axes with unique destinations")
    for axis, destination in axis_mapping.items():
        if not isinstance(destination, str):
            raise ValueError("axis mapping destinations must be strings")
        component, separator, raw_axis = destination.partition("/")
        if not separator or component not in config.weights or not raw_axis:
            raise ValueError(f"axis mapping must select a raw component axis: {destination}")
        if recipe.get("schema") == "vrl.reward-scoring-join.v1":
            alias, separator, source_axis = axis.partition("/")
            if not separator or alias not in recipe["scorers"]:
                raise ValueError(f"unknown source scorer for axis: {axis}")
            source = recipe["scorers"][alias]
        else:
            source_axis, source = axis, recipe
        scoring = ScoringConfig.model_validate(source)
        if raw_axis != source_axis or scoring.inference != config.inference_configs[component]:
            raise ValueError(f"runtime scorer or raw axis differs from calibration: {axis}")
        # These are client-only options. Model options on an HTTP adapter cannot
        # configure its service and must not masquerade as a qualified change.
        unsupported = set(config.kwargs[component]) - {
            "score_key",
            "reward_name",
            "archive_dir",
            "debug_dir",
        }
        if unsupported:
            raise ValueError(f"calibrated HTTP component has unsupported kwargs: {unsupported}")


def validate_observations(payload: dict[str, Any], frozen: FrozenRewardCombination) -> None:
    atol, rtol = payload["atol"], payload["rtol"]
    if any(isinstance(v, bool) or not math.isfinite(v) or v < 0 for v in (atol, rtol)):
        raise ValueError("qualification tolerances must be finite and nonnegative")
    observations = payload["observations"]
    if not observations:
        raise ValueError("qualification requires measured observations")
    for sample_id, row in observations.items():
        offline, runtime = row["offline"], row["runtime"]
        if set(offline) != set(frozen.axes) or set(runtime) != set(frozen.axes):
            raise ValueError(f"qualification axes differ: {sample_id}")
        frozen.apply(offline, sample_id=sample_id)
        frozen.apply(runtime, sample_id=sample_id)
        for axis in frozen.axes:
            if abs(runtime[axis] - offline[axis]) > atol + rtol * abs(offline[axis]):
                raise ValueError(f"raw reward parity failed: {sample_id}/{axis}")


def load_reward_deployment(
    config: RewardRuntimeConfig,
) -> tuple[FrozenRewardCombination, dict[str, str]]:
    """Check the pinned receipt and resolved configuration before creating scorers."""
    reference = config.calibration
    if reference is None:
        raise ValueError("reward calibration reference is missing")
    saved = json.loads(reference.deployment_path.read_text())
    payload = {key: value for key, value in saved.items() if key != "deployment_id"}
    if (
        payload.get("schema") != "vrl.reward-deployment.v1"
        or canonical_json_sha256(payload, allow_nan=False) != reference.deployment_id
        or saved.get("deployment_id") != reference.deployment_id
    ):
        raise ValueError("reward deployment digest or schema mismatch")
    if payload["runtime"] != runtime_recipe(config):
        raise ValueError("reward runtime differs from qualified deployment")
    if payload["input_contract"] != "image-float32-cthw-unit-range.v1":
        raise ValueError("unsupported reward deployment input contract")
    frozen = FrozenRewardCombination(
        payload["combination"], scoring_config=payload["scoring_config"]
    )
    mapping = payload["axis_mapping"]
    validate_binding(config, payload["scoring_config"], frozen, mapping)
    validate_observations(payload, frozen)
    return frozen, dict(mapping)
