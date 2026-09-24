"""Frozen runtime aggregation must reproduce offline arithmetic without batch drift."""

import copy

import pytest

from reward_lab.calibration import apply_combination
from vrl.rewards.base import RewardFunction
from vrl.rewards.calibration import FrozenRewardCombination
from vrl.rewards.functions.registry import MultiReward
from vrl.rewards.types import RewardOutput, RewardSample
from vrl.utils.json_files import canonical_json_sha256


@pytest.mark.asyncio
async def test_runtime_uses_identical_signed_frozen_arithmetic_and_retains_axis_evidence():
    recipe = {"revision": "fixture-source-v1"}
    payload = {
        "schema": "vrl.reward-combination.v1",
        "scoring_config_hash": canonical_json_sha256(recipe, allow_nan=False),
        "dimension": "overall",
        "axes": ["quality", "damage"],
        "means": [2.5, 0.125],
        "scales": [0.75, 0.25],
        "weights": [1.3, -2.1],
        "tie_margin": 0.1,
    }
    artifact = {"combination_id": canonical_json_sha256(payload, allow_nan=False), **payload}
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
    evaluation = {
        "run_id": "fixture",
        "config": recipe,
        "records": {
            sample.sample_id: {
                "input": {"sample_id": sample.sample_id},
                "status": "success",
                "result": {"scores": sample.metadata},
            }
            for sample in samples
        },
    }
    offline = apply_combination(evaluation, artifact)
    online = await reward.score_batch(samples)
    assert online.scores == tuple(offline["records"][s.sample_id]["score"] for s in samples)
    assert online.components["calibration/contribution/damage"] == tuple(
        offline["records"][s.sample_id]["contributions"]["damage"] for s in samples
    )
    assert online.components["judge"] == (999, 999, 999)
    assert online.timing_ms["inference_ms"] == 5
    assert (await reward.score_batch(samples[:1])).scores == online.scores[:1]
    # Caller mutation cannot change an already loaded objective.
    artifact["weights"][0] = 1000
    artifact["axes"][0] = "different"
    assert (await reward.score_batch(samples)).scores == online.scores
    assert (await reward.score_batch([])).scores == ()
    with pytest.raises(ValueError, match="recipe differs"):
        FrozenRewardCombination(
            {"combination_id": canonical_json_sha256(payload, allow_nan=False), **payload},
            scoring_config={"revision": "changed"},
        )


@pytest.mark.asyncio
async def test_missing_runtime_axis_and_ambiguous_weighting_fail_instead_of_falling_back():
    recipe = {}
    payload = {
        "schema": "vrl.reward-combination.v1",
        "scoring_config_hash": canonical_json_sha256(recipe, allow_nan=False),
        "dimension": "overall",
        "axes": ["quality"],
        "means": [0],
        "scales": [1],
        "weights": [1],
        "tie_margin": 0,
    }
    artifact = {"combination_id": canonical_json_sha256(payload, allow_nan=False), **payload}
    frozen = FrozenRewardCombination(artifact, scoring_config=recipe)

    class WrongAxis(RewardFunction):
        async def score_batch(self, samples):
            return RewardOutput(scores=(1.0,), components={"other": (1.0,)})

    with pytest.raises(ValueError, match="unit component weights"):
        MultiReward(
            [("judge", 2.0, WrongAxis())],
            combination=frozen,
            axis_mapping={"quality": "judge/quality"},
        )
    reward = MultiReward(
        [("judge", 1.0, WrongAxis())],
        combination=frozen,
        axis_mapping={"quality": "judge/quality"},
    )
    with pytest.raises(ValueError, match="axes missing"):
        await reward.score_batch([RewardSample("p", None, "a")])
    changed = copy.deepcopy(artifact)
    changed["weights"][0] = 2
    with pytest.raises(ValueError, match="digest"):
        FrozenRewardCombination(changed, scoring_config=recipe)
