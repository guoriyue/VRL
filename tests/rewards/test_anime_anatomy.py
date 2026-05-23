from __future__ import annotations

import pytest
from PIL import Image

from vrl.rewards.functions.anime_anatomy import (
    AnimeAnatomyPlausibilityReward,
    AnimeHandQualityReward,
)
from vrl.rewards.functions.registry import MultiReward
from vrl.rewards.types import RewardRollout, RewardTrajectory


def _rollout() -> RewardRollout:
    return RewardRollout(
        request=None,
        trajectory=RewardTrajectory(
            prompt="anime woman, full body, both hands visible",
            seed=0,
            steps=[],
            output=Image.new("RGB", (8, 8), color=(128, 128, 128)),
        ),
    )


@pytest.mark.asyncio
async def test_anime_anatomy_reward_scores_classifier_labels() -> None:
    reward = AnimeAnatomyPlausibilityReward(
        device="cpu",
        scorer=lambda images: [
            [
                {"label": "plausible_limbs", "score": 0.8},
                {"label": "missing_feet", "score": 0.2},
            ]
            for _ in images
        ],
        positive_labels=("plausible_limbs",),
        negative_labels=("missing_feet",),
    )

    scores = await reward.score_batch([_rollout()])

    assert scores == pytest.approx([0.8])


@pytest.mark.asyncio
async def test_anime_hand_reward_scores_numeric_scorer() -> None:
    reward = AnimeHandQualityReward(device="cpu", scorer=lambda images: [0.35 for _ in images])

    assert await reward.score(_rollout()) == pytest.approx(0.35)


@pytest.mark.asyncio
async def test_anime_reward_requires_model_or_scorer_when_scoring_images() -> None:
    reward = AnimeHandQualityReward(device="cpu", model_name="")

    with pytest.raises(ValueError, match="requires model_name or an injected scorer"):
        await reward.score(_rollout())


def test_anime_rewards_are_registered() -> None:
    reward = MultiReward.from_dict(
        {"anime_anatomy_plausibility": 1.0, "anime_hand_quality": 0.4},
        device="cpu",
        reward_kwargs={
            "anime_anatomy_plausibility": {"scorer": lambda images: [1.0 for _ in images]},
            "anime_hand_quality": {"scorer": lambda images: [0.5 for _ in images]},
        },
    )

    assert [name for name, _, _ in reward.rewards] == [
        "anime_anatomy_plausibility",
        "anime_hand_quality",
    ]
