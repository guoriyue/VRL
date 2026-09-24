"""Qualify frozen reward arithmetic against the actual online HTTP scoring path.

The receipt this writes is what ``vrl.rewards.deployment.load_reward_deployment``
checks at training time. No fitting or automatic tolerance selection occurs.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from vrl.config.builders import RewardRuntimeConfig
from vrl.rewards.calibration import FrozenRewardCombination
from vrl.rewards.deployment import runtime_recipe, validate_binding, validate_observations
from vrl.rewards.types import RewardSample
from vrl.utils.artifacts import sha256_file
from vrl.utils.json_files import canonical_json_sha256


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
    validate_binding(config, evaluation["config"], frozen, axis_mapping)
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
        "runtime": runtime_recipe(config),
        "combination": combination,
        "scoring_config": evaluation["config"],
        "axis_mapping": dict(axis_mapping),
        "input_contract": "image-float32-cthw-unit-range.v1",
        "source_run_id": evaluation["run_id"],
        "atol": atol,
        "rtol": rtol,
        "observations": observations,
    }
    validate_observations(payload, frozen)
    return {"deployment_id": canonical_json_sha256(payload, allow_nan=False), **payload}
