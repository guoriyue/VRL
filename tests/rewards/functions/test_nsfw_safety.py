"""Tests for the NSFW safety penalty reward."""

from __future__ import annotations

import pytest
import torch
from PIL import Image

from vrl.rewards.functions.nsfw_safety import NSFWSafetyReward
from vrl.rewards.types import RewardRollout, RewardTrajectory


def _rollout(output: object) -> RewardRollout:
    return RewardRollout(
        request=None,
        trajectory=RewardTrajectory(
            prompt="anime portrait",
            seed=0,
            steps=[],
            output=output,
        ),
    )


@pytest.mark.asyncio
async def test_nsfw_safety_reward_only_penalizes_scores_above_threshold() -> None:
    reward = NSFWSafetyReward(
        model_name="test",
        threshold=0.25,
        penalty_scale=2.0,
        max_penalty=1.5,
        scorer=lambda images: [0.10, 0.75],
    )
    image = Image.new("RGB", (8, 8), color=(128, 128, 128))

    scores = await reward.score_batch([_rollout(image), _rollout(image)])

    assert scores[0] == pytest.approx(0.0)
    assert scores[1] == pytest.approx(-1.3333333333)
    assert all(score <= 0.0 for score in scores)


@pytest.mark.asyncio
async def test_nsfw_safety_reward_uses_max_probability_for_image_batches() -> None:
    reward = NSFWSafetyReward(
        model_name="test",
        threshold=0.50,
        penalty_scale=1.0,
        image_sample_count=2,
        scorer=lambda images: [0.20, 0.90],
    )
    output = torch.rand(2, 3, 8, 8)

    score = await reward.score(_rollout(output))

    assert score == pytest.approx(-0.8)


def test_nsfw_classifier_result_parsing_prefers_nsfw_labels() -> None:
    reward = NSFWSafetyReward(model_name="test", threshold=0.35)

    probability = reward._probability_from_classifier_result(
        [
            {"label": "normal", "score": 0.80},
            {"label": "unsafe", "score": 0.20},
        ],
    )

    assert probability == pytest.approx(0.20)


def test_nsfw_safety_reward_exposes_raw_probabilities_for_audits() -> None:
    reward = NSFWSafetyReward(model_name="test", scorer=lambda images: [0.10, 0.90])

    probabilities = reward.probability_batch([object(), object()])

    assert probabilities == pytest.approx([0.10, 0.90])


def test_nsfw_safety_reward_rejects_invalid_reverse_like_config() -> None:
    with pytest.raises(ValueError, match="penalty_scale"):
        NSFWSafetyReward(model_name="test", penalty_scale=-1.0)
