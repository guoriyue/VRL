"""Tests for the WD tagger requested-tag recall reward."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from vrl.rewards.functions.wd_tagger import WDTaggerReward
from vrl.rewards.models.wd_tagger import prepare_wd14_input
from vrl.rewards.types import RewardSample

_WANTED = ["long_hair", "lingerie", "smile"]


def _sample(
    output: object,
    *,
    sample_id: str = "sample-0",
    tags: list[str] | None = _WANTED,
) -> RewardSample:
    metadata = {} if tags is None else {"adherence_tags": tags}
    return RewardSample(
        prompt="1girl, long hair, lingerie, smile",
        output=output,
        sample_id=sample_id,
        metadata=metadata,
    )


def _image() -> Image.Image:
    return Image.new("RGB", (8, 8), color=(128, 128, 128))


@pytest.mark.asyncio
async def test_wd_tagger_reward_scores_recall_over_wanted_tags() -> None:
    """Checks tag adherence reward scores recall over wanted tags."""
    reward = WDTaggerReward(
        threshold=0.35,
        tagger=lambda images: [{"long_hair": 0.9, "smile": 0.5, "lingerie": 0.1}] * len(images),
    )

    score = await reward.score(_sample(_image()))

    assert score == pytest.approx(2.0 / 3.0)


@pytest.mark.asyncio
async def test_wd_tagger_reward_threshold_is_inclusive() -> None:
    """Checks tag adherence reward threshold is inclusive."""
    reward = WDTaggerReward(
        threshold=0.5,
        tagger=lambda images: [{"long_hair": 0.5, "smile": 0.4999, "lingerie": 0.0}] * len(images),
    )

    score = await reward.score(_sample(_image()))

    assert score == pytest.approx(1.0 / 3.0)


@pytest.mark.asyncio
async def test_wd_tagger_reward_matches_case_insensitively_and_ignores_extras() -> None:
    """Checks tag adherence reward matches case insensitively and ignores extras."""
    reward = WDTaggerReward(
        threshold=0.35,
        tagger=lambda images: (
            [
                {"Long_Hair": 0.9, "LINGERIE": 0.8, "smile": 0.9, "1girl": 0.99, "solo": 0.97},
            ]
            * len(images)
        ),
    )

    score = await reward.score(_sample(_image(), tags=["long_hair", "Lingerie", "SMILE"]))

    assert score == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_wd_tagger_reward_score_batch_preserves_order() -> None:
    """Checks tag adherence reward score_batch preserves order."""
    reward = WDTaggerReward(
        threshold=0.35,
        tagger=lambda images: [
            {"long_hair": 0.9, "lingerie": 0.9, "smile": 0.9},
            {"long_hair": 0.9, "lingerie": 0.0, "smile": 0.0},
        ],
    )

    output = await reward.score_batch(
        [_sample(_image()), _sample(_image(), sample_id="sample-1")],
    )

    assert output.scores == pytest.approx((1.0, 1.0 / 3.0))


@pytest.mark.asyncio
@pytest.mark.parametrize("tags", [None, []])
async def test_wd_tagger_reward_rejects_missing_or_empty_tags(tags: list[str] | None) -> None:
    """Checks tag adherence reward rejects missing or empty tags instead of scoring 0."""
    reward = WDTaggerReward(tagger=lambda images: [{}] * len(images))

    with pytest.raises(ValueError, match="adherence_tags"):
        await reward.score(_sample(_image(), tags=tags))


@pytest.mark.parametrize("threshold", [1.5, -0.1])
def test_wd_tagger_reward_rejects_invalid_threshold(threshold: float) -> None:
    """Checks tag adherence reward rejects invalid threshold."""
    with pytest.raises(ValueError, match="threshold"):
        WDTaggerReward(threshold=threshold, tagger=lambda images: [])


def test_prepare_wd14_input_pads_white_resizes_and_swaps_to_bgr() -> None:
    """Checks prepare_wd14_input pads white, resizes to 448, and swaps to BGR."""
    red = Image.new("RGB", (100, 60), color=(255, 0, 0))

    batch = prepare_wd14_input(red)

    assert batch.shape == (1, 448, 448, 3)
    assert batch.dtype == np.float32
    # Top rows are padding (60 tall pasted at offset 20 of a 100 canvas): white.
    assert np.array_equal(batch[0, 5, 224], np.array([255.0, 255.0, 255.0], dtype=np.float32))
    # Center is the pasted red region: BGR puts red in channel 2.
    assert np.array_equal(batch[0, 224, 224], np.array([0.0, 0.0, 255.0], dtype=np.float32))
