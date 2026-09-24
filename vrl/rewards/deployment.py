"""Qualify frozen reward arithmetic against the actual online HTTP scoring path.

A receipt binds configuration and measured raw-axis parity on pinned images. It
does not certify preference quality, arbitrary future inputs, or an operator's
model-version declaration. No fitting or automatic tolerance selection occurs.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

from vrl.config.builders import RewardRuntimeConfig
from vrl.rewards.calibration import FrozenRewardCombination
from vrl.rewards.evaluation import ScoringConfig
from vrl.rewards.types import RewardSample
from vrl.utils.artifacts import sha256_file
from vrl.utils.json_files import canonical_json_sha256


def _runtime_recipe(config: RewardRuntimeConfig) -> dict[str, Any]:
    return {
        "components": config.weights,
        "kwargs": config.kwargs,
        "inference": {name: asdict(value) for name, value in config.inference_configs.items()},
    }


def _validate_binding(
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


def _validate_observations(payload: dict[str, Any], frozen: FrozenRewardCombination) -> None:
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
    if payload["runtime"] != _runtime_recipe(config):
        raise ValueError("reward runtime differs from qualified deployment")
    if payload["input_contract"] != "image-float32-cthw-unit-range.v1":
        raise ValueError("unsupported reward deployment input contract")
    frozen = FrozenRewardCombination(
        payload["combination"], scoring_config=payload["scoring_config"]
    )
    mapping = payload["axis_mapping"]
    _validate_binding(config, payload["scoring_config"], frozen, mapping)
    _validate_observations(payload, frozen)
    return frozen, dict(mapping)


async def qualify_reward_deployment(
    evaluation: dict[str, Any],
    combination: dict[str, Any],
    config: RewardRuntimeConfig,
    *,
    axis_mapping: Mapping[str, str],
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    """Measure file-to-training-tensor parity without changing the frozen objective.

    One image per request matches the visual episode judge. All source records
    must succeed; missing/error rows cannot disappear from qualification. Input
    and auxiliary file hashes are checked before and after each scoring call.
    The caller owns writing the returned receipt only after this function passes.
    """
    import numpy as np
    import torch
    from PIL import Image

    from vrl.rewards.functions.registry import MultiReward
    from vrl.utils.media import to_pil_image

    if config.calibration is not None:
        raise ValueError("qualification requires an uncalibrated runtime configuration")
    if any(isinstance(v, bool) or not math.isfinite(v) or v < 0 for v in (atol, rtol)):
        raise ValueError("qualification tolerances must be finite and nonnegative")
    frozen = FrozenRewardCombination(combination, scoring_config=evaluation["config"])
    _validate_binding(config, evaluation["config"], frozen, axis_mapping)
    records = evaluation["records"]
    if not records or any(row["status"] != "success" for row in records.values()):
        raise ValueError("qualification requires nonempty, entirely successful source scores")

    def verify_inputs(sample_id: str, item: dict[str, Any]) -> None:
        paths = {"output": item["path"], **item["assets"]}
        digests = {"output": item["sha256"], **item["asset_sha256"]}
        if set(paths) != set(digests) or any(
            sha256_file(Path(path)) != digests[key] for key, path in paths.items()
        ):
            raise ValueError(f"qualification media changed: {sample_id}")

    for sample_id, row in records.items():
        verify_inputs(sample_id, row["input"])
        frozen.apply(row["result"]["scores"], sample_id=sample_id)
    observations = {}
    reward = MultiReward.from_dict(
        config.weights,
        device="cpu",
        reward_kwargs=config.kwargs,
        inference_configs=config.inference_configs,
        image_float32_inputs=True,
    )
    try:
        await reward.preflight()
        await reward.activate()
        for sample_id, row in records.items():
            item = row["input"]
            verify_inputs(sample_id, item)
            with Image.open(item["path"]) as image:
                if getattr(image, "n_frames", 1) != 1:
                    raise ValueError("qualification currently supports still images only")
                pixels = np.array(to_pil_image(image, preserve_alpha=True), copy=True)
            media = torch.from_numpy(pixels).permute(2, 0, 1).unsqueeze(1).float() / 255.0
            output = await reward.score_batch(
                [
                    RewardSample(
                        prompt=item["prompt"],
                        output=media,
                        sample_id=sample_id,
                        metadata={**item["metadata"], **item["assets"]},
                    )
                ]
            )
            verify_inputs(sample_id, item)
            observations[sample_id] = {
                "input": item,
                "offline": {axis: row["result"]["scores"][axis] for axis in frozen.axes},
                "runtime": {axis: output.components[key][0] for axis, key in axis_mapping.items()},
            }
    finally:
        try:
            await reward.park_memory()
        finally:
            await reward.shutdown()
    payload = {
        "schema": "vrl.reward-deployment.v1",
        "runtime": _runtime_recipe(config),
        "combination": combination,
        "scoring_config": evaluation["config"],
        "axis_mapping": dict(axis_mapping),
        "input_contract": "image-float32-cthw-unit-range.v1",
        "source_run_id": evaluation["run_id"],
        "atol": atol,
        "rtol": rtol,
        "observations": observations,
    }
    _validate_observations(payload, frozen)
    return {"deployment_id": canonical_json_sha256(payload, allow_nan=False), **payload}
