"""A qualified reward deployment: a frozen combination bound to a runtime recipe.

``RewardDeployment.qualify`` scores pinned images through the actual online
scoring path and records that the raw axes match the offline scores the
combination was fitted on; ``write`` saves that receipt. ``RewardDeployment.load``
checks a receipt against the resolved runtime configuration before training
builds its scorers. Neither certifies preference quality.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from vrl.config.builders import RewardRuntimeConfig
from vrl.rewards.calibration import FrozenRewardCombination
from vrl.rewards.evaluation import Evaluation, ScoringConfig
from vrl.utils.json_files import write_json

DEPLOYMENT_SCHEMA = "vrl.reward-deployment.v1"
INPUT_CONTRACT = "image-float32-cthw-unit-range.v1"


def _runtime_recipe(config: RewardRuntimeConfig) -> dict[str, Any]:
    return {
        "components": config.weights,
        "kwargs": config.kwargs,
        "inference": {name: asdict(value) for name, value in config.inference_configs.items()},
    }


@dataclass(frozen=True)
class RewardDeployment:
    combination: FrozenRewardCombination
    # Frozen axis -> "<component>/<raw axis>" in the online reward output.
    axis_mapping: dict[str, str]
    receipt: dict[str, Any]

    @classmethod
    def load(cls, config: RewardRuntimeConfig) -> RewardDeployment:
        """Read the receipt named by ``config.calibration`` and check it fits ``config``."""

        reference = config.calibration
        if reference is None:
            raise ValueError("reward calibration reference is missing")
        receipt = json.loads(Path(reference.deployment_path).read_text())
        if receipt.get("schema") != DEPLOYMENT_SCHEMA:
            raise ValueError("unsupported reward deployment schema")
        if receipt["runtime"] != _runtime_recipe(config):
            raise ValueError("reward runtime differs from qualified deployment")
        combination = FrozenRewardCombination(
            receipt["combination"], scoring_config=receipt["scoring_config"]
        )
        deployment = cls(combination, dict(receipt["axis_mapping"]), receipt)
        deployment._check_binding(config, receipt["scoring_config"])
        return deployment

    @classmethod
    async def qualify(
        cls,
        evaluation: Evaluation,
        combination: dict[str, Any],
        config: RewardRuntimeConfig,
        *,
        axis_mapping: Mapping[str, str],
        atol: float,
        rtol: float,
    ) -> RewardDeployment:
        """Score the evaluation's images through the online path and compare raw axes.

        Every offline sample must have scored; the online reward is built from
        ``config`` exactly as training builds it, with float image inputs.
        """

        import numpy as np
        import torch
        from PIL import Image

        from vrl.rewards.functions.registry import MultiReward
        from vrl.rewards.types import RewardSample
        from vrl.utils.media import to_pil_image

        if config.calibration is not None:
            raise ValueError("qualification requires an uncalibrated runtime configuration")
        if any(isinstance(v, bool) or not math.isfinite(v) or v < 0 for v in (atol, rtol)):
            raise ValueError("qualification tolerances must be finite and nonnegative")
        frozen = FrozenRewardCombination(combination, scoring_config=evaluation.config)
        deployment = cls(frozen, dict(axis_mapping), {})
        deployment._check_binding(config, evaluation.config)
        records = evaluation.records
        if not records or any(row["status"] != "success" for row in records.values()):
            raise ValueError("qualification requires nonempty, entirely successful source scores")
        reward = MultiReward.from_dict(
            config.weights,
            device="cpu",
            reward_kwargs=config.kwargs,
            inference_configs=config.inference_configs,
            image_float32_inputs=True,
        )
        observations = {}
        try:
            await reward.preflight()
            await reward.activate()
            for sample_id, row in records.items():
                item = row["input"]
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
                observations[sample_id] = {
                    "input": item,
                    "offline": {axis: row["result"]["scores"][axis] for axis in frozen.axes},
                    "runtime": {
                        axis: output.components[key][0] for axis, key in axis_mapping.items()
                    },
                }
        finally:
            try:
                await reward.park_memory()
            finally:
                await reward.shutdown()
        for sample_id, row in observations.items():
            for axis in frozen.axes:
                offline, runtime = row["offline"][axis], row["runtime"][axis]
                if abs(runtime - offline) > atol + rtol * abs(offline):
                    raise ValueError(f"raw reward parity failed: {sample_id}/{axis}")
        receipt = {
            "schema": DEPLOYMENT_SCHEMA,
            "runtime": _runtime_recipe(config),
            "combination": combination,
            "scoring_config": evaluation.config,
            "axis_mapping": dict(axis_mapping),
            "input_contract": INPUT_CONTRACT,
            "source_run_id": evaluation.run_id,
            "atol": atol,
            "rtol": rtol,
            "observations": observations,
        }
        return cls(frozen, dict(axis_mapping), receipt)

    def write(self, path: Path) -> None:
        write_json(path, self.receipt)

    def _check_binding(self, config: RewardRuntimeConfig, recipe: dict[str, Any]) -> None:
        """The runtime's HTTP components and raw axes must be the ones the fit used."""

        config.require_online_training()
        if not config.all_external_inference:
            raise ValueError("calibrated deployment currently requires HTTP scoring components")
        if any(weight != 1 for weight in config.weights.values()):
            raise ValueError("calibrated deployment requires unit component weights")
        mapping = self.axis_mapping
        if set(mapping) != set(self.combination.axes) or len(set(mapping.values())) != len(
            mapping
        ):
            raise ValueError("axis mapping must cover frozen axes with unique destinations")
        for axis, destination in mapping.items():
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
            # Client-only options; model options on an HTTP adapter cannot
            # configure its service and must not masquerade as a qualified change.
            unsupported = set(config.kwargs[component]) - {
                "score_key",
                "reward_name",
                "archive_dir",
                "debug_dir",
            }
            if unsupported:
                raise ValueError(
                    f"calibrated HTTP component has unsupported kwargs: {unsupported}"
                )


__all__ = ["DEPLOYMENT_SCHEMA", "INPUT_CONTRACT", "RewardDeployment"]
