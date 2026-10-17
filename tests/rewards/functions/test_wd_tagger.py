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
async def test_recall_counts_tags_at_or_above_threshold_case_insensitively_per_sample() -> None:
    """Recall is wanted ∩ detected over wanted: inclusive threshold, lowercase match, extras ignored."""
    reward = WDTaggerReward(
        score_key="wd_tagger_recall",
        threshold=0.5,
        tagger=lambda images: [
            {"Long_Hair": 0.5, "LINGERIE": 0.4999, "smile": 0.0, "1girl": 0.99, "solo": 0.97},
            {"long_hair": 0.9, "lingerie": 0.9, "smile": 0.9},
        ],
    )

    output = await reward.score_batch(
        [
            _sample(_image(), tags=["long_hair", "Lingerie", "SMILE"]),
            _sample(_image(), sample_id="sample-1"),
        ],
    )

    assert output.scores == pytest.approx((1.0 / 3.0, 1.0))


@pytest.mark.asyncio
async def test_wd_tagger_reward_rejects_missing_tags() -> None:
    """Checks tag adherence reward rejects missing tags instead of scoring 0."""
    reward = WDTaggerReward(tagger=lambda images: [{}] * len(images))

    with pytest.raises(ValueError, match="adherence_tags"):
        (await reward.score_batch([_sample(_image(), tags=None)])).scores[0]


def test_wd_tagger_reward_rejects_invalid_threshold() -> None:
    """Checks tag adherence reward rejects invalid threshold."""
    with pytest.raises(ValueError, match="threshold"):
        WDTaggerReward(threshold=1.5, tagger=lambda images: [])


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


@pytest.mark.asyncio
async def test_wd_tagger_rejects_non_string_tags_before_inference() -> None:
    calls = []
    reward = WDTaggerReward(tagger=lambda images: calls.append(images))
    with pytest.raises(ValueError, match="must contain only tag strings"):
        (await reward.score_batch([_sample(_image(), tags=["smile", 42])])).scores[0]
    assert calls == []


@pytest.mark.asyncio
async def test_dense_score_orders_images_the_recall_calls_identical() -> None:
    """Two images missing the same tag must still be ordered by how close they came."""

    def reward_for(lingerie: float) -> WDTaggerReward:
        return WDTaggerReward(
            threshold=0.35,
            tagger=lambda images, p=lingerie: (
                [{"long_hair": 0.9, "smile": 0.9, "lingerie": p}] * len(images)
            ),
        )

    sample = [_sample(_image())]
    (faint,) = (await reward_for(0.05).score_batch(sample)).scores
    (nearly,) = (await reward_for(0.30).score_batch(sample)).scores
    recall_reward = WDTaggerReward(
        score_key="wd_tagger_recall",
        threshold=0.35,
        tagger=lambda images: [{"long_hair": 0.9, "smile": 0.9, "lingerie": 0.05}] * len(images),
    )
    (recall,) = (await recall_reward.score_batch(sample)).scores

    assert recall == pytest.approx(2.0 / 3.0)
    assert faint < nearly
    assert faint == pytest.approx((1.0 + 1.0 + 0.05 / 0.35) / 3)


@pytest.mark.asyncio
async def test_dense_caps_each_tag_at_the_threshold_and_equals_recall_outside_the_gap() -> None:
    """Confidence beyond the threshold pays nothing; absent-or-confident tags give recall."""
    sample = [_sample(_image())]
    for level in (0.4, 1.0):
        certain = WDTaggerReward(
            threshold=0.35,
            tagger=lambda images, p=level: (
                [{"long_hair": p, "smile": p, "lingerie": p}] * len(images)
            ),
        )
        assert (await certain.score_batch(sample)).scores[0] == pytest.approx(1.0)

    tagger = lambda images: [{"long_hair": 0.9, "smile": 0.0, "lingerie": 0.0}] * len(images)  # noqa: E731
    (dense,) = (await WDTaggerReward(threshold=0.35, tagger=tagger).score_batch(sample)).scores
    recall_reward = WDTaggerReward(score_key="wd_tagger_recall", threshold=0.35, tagger=tagger)
    (recall,) = (await recall_reward.score_batch(sample)).scores

    assert dense == recall == pytest.approx(1.0 / 3.0)
