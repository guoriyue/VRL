"""Qualification binds real HTTP raw scores, float transport, and online construction."""

import copy
import json
from dataclasses import replace

import numpy as np
import pytest
import torch
from PIL import Image

from vrl.config.builders import RewardRuntimeConfig
from vrl.config.reward_calibration import RewardCalibrationConfig
from vrl.config.reward_inference import RewardInferenceConfig
from vrl.config.schema import RewardConfig
from vrl.rewards.calibration import FrozenRewardCombination
from vrl.rewards.deployment import load_reward_deployment, qualify_reward_deployment
from vrl.rewards.diagnostics import read_evaluation
from vrl.rewards.evaluation import ScoringConfig, rescore_media
from vrl.rewards.runtime import InProcessRewardScorer
from vrl.rewards.service.server import RewardService
from vrl.rewards.types import RewardSample
from vrl.run import ResolvedReward
from vrl.scripts.common.factory import build_reward_function
from vrl.utils.json_files import canonical_json_sha256


@pytest.mark.asyncio
async def test_real_service_qualification_pins_objective_and_rejects_drift(tmp_path):
    pixels = np.zeros((8, 8, 4), dtype=np.uint8)
    pixels[2:6, 2:6] = [128, 37, 220, 102]
    target = tmp_path / "target.png"
    Image.fromarray(pixels).save(target)
    rows = []
    for index, alpha in enumerate((102, 220)):
        candidate = pixels.copy()
        candidate[2:6, 2:6, 3] = alpha
        path = tmp_path / f"candidate-{index}.png"
        Image.fromarray(candidate).save(path)
        rows.append(
            {
                "sample_id": str(index),
                "prompt_id": "task",
                "prompt": "match target",
                "path": str(path),
                "assets": {"target_image": str(target)},
            }
        )
    manifest = tmp_path / "media.jsonl"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))
    service = RewardService(
        InProcessRewardScorer(
            {
                "model_factory": "vrl.rewards.models.image_sharpness:ImageSharpnessRewardModel",
                "scale": 10.0,
                "device": "cpu",
            }
        ),
        artifact_roots=[tmp_path],
        port=0,
        model_name="sharpness",
        model_version="test-v1",
        generation_overlap_safe=True,
    )
    await service.start()
    host, port = service.address
    inference = RewardInferenceConfig(
        kind="http",
        endpoint=f"http://{host}:{port}",
        expected_model="sharpness",
        expected_model_version="test-v1",
    )
    scoring = ScoringConfig(
        name="sharpness",
        revision="test-v1",
        preprocessing_revision="file-v1",
        rubric_revision="laplacian-v1",
        inference=inference,
    )
    config = RewardRuntimeConfig.from_cfg(
        RewardConfig(
            components={"image_sharpness": 1},
            inference={"image_sharpness": inference},
        )
    )
    runtime = None
    try:
        await rescore_media(manifest, scoring, tmp_path / "scores")
        evaluation = read_evaluation(tmp_path / "scores")
        axes = ["image_sharpness"]
        payload = {
            "schema": "vrl.reward-combination.v1",
            "scoring_config_hash": canonical_json_sha256(evaluation["config"], allow_nan=False),
            "dimension": "fixture",
            "axes": axes,
            "means": [0.01],
            "scales": [0.01],
            "weights": [1.0],
            "tie_margin": 0,
        }
        combination = {
            "combination_id": canonical_json_sha256(payload, allow_nan=False),
            **payload,
        }
        mapping = {axis: f"image_sharpness/{axis}" for axis in axes}
        receipt = await qualify_reward_deployment(
            evaluation,
            combination,
            config,
            axis_mapping=mapping,
            atol=1e-6,
            rtol=0,
        )
        path = tmp_path / "deployment.json"
        path.write_text(json.dumps(receipt))
        qualified = replace(
            config,
            calibration=RewardCalibrationConfig(
                deployment_path=path,
                deployment_id=receipt["deployment_id"],
            ),
        )
        runtime = build_reward_function(
            ResolvedReward(
                config=qualified,
                device="cpu",
                memory_parking_required=False,
            )
        )
        await runtime.preflight()
        sample = RewardSample(
            "match target",
            torch.from_numpy(pixels).permute(2, 0, 1).unsqueeze(1).float() / 255,
            "0",
            {"target_image": str(target)},
        )
        output = await runtime.score_batch([sample])
        frozen = FrozenRewardCombination(combination, scoring_config=evaluation["config"])
        assert output.scores[0] == frozen.apply(receipt["observations"]["0"]["runtime"])[0]
        assert output.components["image_sharpness/image_sharpness"][0] > 0
        with pytest.raises(ValueError, match="float32 RGB/RGBA"):
            await runtime.score_batch([replace(sample, output=sample.output.double())])
        changed = replace(qualified, kwargs={"image_sharpness": {"score_key": "sharpness_v2"}})
        with pytest.raises(ValueError, match="differs from qualified deployment"):
            build_reward_function(
                ResolvedReward(
                    config=changed,
                    device="cpu",
                    memory_parking_required=False,
                )
            )
        altered = copy.deepcopy(receipt)
        altered["combination"]["weights"][0] = 99
        path.write_text(json.dumps(altered))
        with pytest.raises(ValueError, match="digest"):
            load_reward_deployment(qualified)
        # An already constructed run retains detached frozen vectors.
        assert (await runtime.score_batch([sample])).scores == output.scores
        Image.new("RGBA", (8, 8), "red").save(target)
        with pytest.raises(ValueError, match="media changed"):
            await qualify_reward_deployment(
                evaluation,
                combination,
                config,
                axis_mapping=mapping,
                atol=1e-6,
                rtol=0,
            )
    finally:
        if runtime is not None:
            await runtime.shutdown()
        await service.shutdown_async()
