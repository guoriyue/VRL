"""A qualified deployment reproduces offline arithmetic online and refuses a changed runtime."""

import json
from dataclasses import replace

import numpy as np
import pytest
import torch
from PIL import Image

from reward_lab.calibration import Calibration
from vrl.config.builders import RewardRuntimeConfig
from vrl.config.reward_calibration import RewardCalibrationConfig
from vrl.config.reward_inference import RewardInferenceConfig
from vrl.config.schema import RewardConfig
from vrl.rewards.base import RewardFunction
from vrl.rewards.calibration import FrozenRewardCombination
from vrl.rewards.deployment import RewardDeployment
from vrl.rewards.evaluation import Evaluation, ScoringConfig
from vrl.rewards.functions.registry import MultiReward
from vrl.rewards.runtime import InProcessRewardScorer
from vrl.rewards.service.server import RewardService
from vrl.rewards.types import RewardOutput, RewardSample
from vrl.run import ResolvedReward
from vrl.scripts.common.factory import build_reward_function
from vrl.utils.json_files import canonical_json_sha256


def _combination(recipe, **overrides):
    payload = {
        "schema": "vrl.reward-combination.v1",
        "scoring_config_hash": canonical_json_sha256(recipe, allow_nan=False),
        "dimension": "overall",
        "axes": ["quality", "damage"],
        "means": [2.5, 0.125],
        "scales": [0.75, 0.25],
        "weights": [1.3, -2.1],
        "tie_margin": 0.1,
        **overrides,
    }
    return {"combination_id": canonical_json_sha256(payload, allow_nan=False), **payload}


@pytest.mark.asyncio
async def test_runtime_uses_identical_signed_frozen_arithmetic_and_retains_axis_evidence():
    recipe = {"revision": "fixture-source-v1"}
    artifact = _combination(recipe)
    frozen = FrozenRewardCombination(artifact, scoring_config=recipe)

    class MeasuredReward(RewardFunction):
        async def score_batch(self, samples):
            return RewardOutput(
                scores=tuple(999.0 for _ in samples),
                components={
                    "quality": tuple(sample.metadata["quality"] for sample in samples),
                    "damage": tuple(sample.metadata["damage"] for sample in samples),
                },
                timing_ms={"inference_ms": 5.0},
            )

    reward = MultiReward(
        [("judge", 1.0, MeasuredReward())],
        combination=frozen,
        axis_mapping={"quality": "judge/quality", "damage": "judge/damage"},
    )
    samples = [
        RewardSample("same", None, str(i), {"quality": q, "damage": d})
        for i, (q, d) in enumerate(((1.0, 0.3), (7.0, 0.0), (100.0, 10.0)))
    ]
    evaluation = Evaluation(
        "fixture",
        recipe,
        {
            sample.sample_id: {
                "input": {"sample_id": sample.sample_id},
                "status": "success",
                "result": {"scores": sample.metadata},
            }
            for sample in samples
        },
    )
    offline = Calibration(evaluation).apply(artifact)
    online = await reward.score_batch(samples)
    assert online.scores == tuple(offline["records"][s.sample_id]["score"] for s in samples)
    assert online.components["calibration/contribution/damage"] == tuple(
        offline["records"][s.sample_id]["contributions"]["damage"] for s in samples
    )
    assert online.components["judge"] == (999, 999, 999)
    # Caller mutation cannot change an already loaded objective.
    artifact["weights"][0] = 1000
    assert (await reward.score_batch(samples)).scores == online.scores
    with pytest.raises(ValueError, match="recipe differs"):
        FrozenRewardCombination(_combination(recipe), scoring_config={"revision": "changed"})
    with pytest.raises(ValueError, match="unit component weights"):
        MultiReward(
            [("judge", 2.0, MeasuredReward())],
            combination=frozen,
            axis_mapping={"quality": "judge/quality", "damage": "judge/damage"},
        )
    with pytest.raises(ValueError, match="vectors"):
        FrozenRewardCombination(_combination(recipe, scales=[0.0, 0.25]), scoring_config=recipe)


@pytest.mark.asyncio
async def test_real_service_qualification_pins_objective_and_rejects_runtime_drift(tmp_path):
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
        RewardConfig(components={"image_sharpness": 1}, inference={"image_sharpness": inference})
    )
    runtime = None
    try:
        evaluation = await Evaluation.score(manifest, scoring, tmp_path / "scores")
        combination = _combination(
            evaluation.config, axes=["image_sharpness"], means=[0.01], scales=[0.01], weights=[1.0]
        )
        mapping = {"image_sharpness": "image_sharpness/image_sharpness"}
        deployment = await RewardDeployment.qualify(
            evaluation, combination, config, axis_mapping=mapping, atol=1e-6, rtol=0
        )
        path = tmp_path / "deployment.json"
        deployment.write(path)
        qualified = replace(config, calibration=RewardCalibrationConfig(deployment_path=path))
        loaded = RewardDeployment.load(qualified)
        assert loaded.axis_mapping == mapping and loaded.combination == deployment.combination
        runtime = build_reward_function(
            ResolvedReward(config=qualified, device="cpu", memory_parking_required=False)
        )
        await runtime.preflight()
        sample = RewardSample(
            "match target",
            torch.from_numpy(pixels).permute(2, 0, 1).unsqueeze(1).float() / 255,
            "0",
            {"target_image": str(target)},
        )
        output = await runtime.score_batch([sample])
        observed = deployment.receipt["observations"]["0"]["runtime"]
        assert output.scores[0] == deployment.combination.apply(observed)[0]
        assert output.components["image_sharpness/image_sharpness"][0] > 0
        changed = replace(qualified, kwargs={"image_sharpness": {"score_key": "sharpness_v2"}})
        with pytest.raises(ValueError, match="differs from qualified deployment"):
            build_reward_function(
                ResolvedReward(config=changed, device="cpu", memory_parking_required=False)
            )
        with pytest.raises(ValueError, match="float32 RGB/RGBA"):
            await runtime.score_batch([replace(sample, output=sample.output.double())])
        with pytest.raises(ValueError, match="uncalibrated"):
            await RewardDeployment.qualify(
                evaluation, combination, qualified, axis_mapping=mapping, atol=1e-6, rtol=0
            )
    finally:
        if runtime is not None:
            await runtime.shutdown()
        await service.shutdown_async()
