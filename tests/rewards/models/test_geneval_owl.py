"""Tests for the OWLv2 + CLIP GenEval reward: decision rules over a fake detector."""

from __future__ import annotations

import pytest
from PIL import Image

from vrl.rewards.functions.geneval_owl import GenEvalOwlReward
from vrl.rewards.models.geneval_owl import GenEvalOwlRewardModel, nms, relative_position
from vrl.rewards.types import RewardSample

_IMAGE = Image.new("RGB", (64, 64), color=(128, 128, 128))


def _model(detections, colors=None, **kwargs) -> GenEvalOwlRewardModel:
    """Model whose detector returns ``detections`` for every image."""

    return GenEvalOwlRewardModel(
        {
            "device": "cpu",
            "detector": lambda images, classes: [detections] * len(images),
            "color_fn": lambda image, box, cls: (colors or {}).get(tuple(box), "red"),
            **kwargs,
        }
    )


def test_single_object_partial_and_strict_agree_when_all_conditions_hold() -> None:
    verdict = _model({"bench": [((0, 0, 10, 10), 0.9)]}).judge(
        _IMAGE, {"include": [{"class": "bench", "count": 1}]}
    )
    assert (verdict.strict, verdict.partial, verdict.why) == (1.0, 1.0, "ok")


def test_missing_object_fails_strict_and_zeroes_partial() -> None:
    verdict = _model({"bench": []}).judge(_IMAGE, {"include": [{"class": "bench", "count": 1}]})
    assert (verdict.strict, verdict.partial, verdict.why) == (0.0, 0.0, "missing:bench")


def test_counting_uses_strict_exclude_and_partial_credit() -> None:
    """Two clocks asked for, three detected: include passes, exclude fails."""
    dets = {"clock": [((0, 0, 5, 5), 0.9), ((10, 0, 15, 5), 0.8), ((20, 0, 25, 5), 0.7)]}
    verdict = _model(dets).judge(
        _IMAGE,
        {"include": [{"class": "clock", "count": 2}], "exclude": [{"class": "clock", "count": 3}]},
    )
    assert verdict.strict == 0.0
    assert verdict.partial == pytest.approx(0.5)
    assert verdict.why == "exclude:clock>=3"


def test_color_attribute_binds_per_object() -> None:
    dets = {"bus": [((0, 0, 10, 10), 0.9)], "handbag": [((20, 20, 30, 30), 0.9)]}
    colors = {(0.0, 0.0, 10.0, 10.0): "yellow", (20.0, 20.0, 30.0, 30.0): "blue"}
    spec = {
        "include": [
            {"class": "bus", "color": "yellow", "count": 1},
            {"class": "handbag", "color": "orange", "count": 1},
        ]
    }
    verdict = _model(dets, colors).judge(_IMAGE, spec)
    # 4 conditions: bus found, bus yellow, handbag found, handbag orange (fails).
    assert verdict.strict == 0.0
    assert verdict.partial == pytest.approx(0.75)
    assert verdict.why == "color:handbag!=orange"


def test_position_is_checked_against_the_earlier_include_item() -> None:
    dets = {"kite": [((0, 40, 10, 50), 0.9)], "wine glass": [((0, 0, 10, 10), 0.9)]}
    spec = {
        "include": [
            {"class": "kite", "count": 1},
            {"class": "wine glass", "count": 1, "position": ["above", 0]},
        ]
    }
    assert _model(dets).judge(_IMAGE, spec).strict == 1.0
    below = {
        "include": [
            {"class": "kite", "count": 1},
            {"class": "wine glass", "count": 1, "position": ["below", 0]},
        ]
    }
    verdict = _model(dets).judge(_IMAGE, below)
    assert verdict.strict == 0.0
    assert verdict.partial == pytest.approx(2.0 / 3.0)
    assert verdict.why == "position:below"


def test_position_with_missing_reference_fails_that_condition_only() -> None:
    dets = {"kite": [], "wine glass": [((0, 0, 10, 10), 0.9)]}
    spec = {
        "include": [
            {"class": "kite", "count": 1},
            {"class": "wine glass", "count": 1, "position": ["above", 0]},
        ]
    }
    verdict = _model(dets).judge(_IMAGE, spec)
    assert verdict.partial == pytest.approx(1.0 / 3.0)
    assert verdict.why == "missing:kite;position:above"


def test_relative_position_official_margin_zeroes_touching_objects() -> None:
    assert relative_position((0, 0, 10, 10), (0, 12, 10, 22), 0.1) == {"above"}
    assert relative_position((0, 0, 10, 10), (0, 1, 10, 11), 0.1) == set()
    assert relative_position((30, 0, 40, 10), (0, 0, 10, 10), 0.1) == {"right of"}


def test_nms_keeps_highest_scored_non_overlapping_boxes() -> None:
    kept = nms([((0, 0, 10, 10), 0.5), ((1, 1, 11, 11), 0.9), ((50, 50, 60, 60), 0.4)], 0.5)
    assert kept == [((1, 1, 11, 11), 0.9), ((50, 50, 60, 60), 0.4)]


def test_spec_without_conditions_is_rejected() -> None:
    with pytest.raises(ValueError, match="include nor exclude"):
        _model({}).judge(_IMAGE, {"tag": "single_object"})


@pytest.mark.parametrize("key", ["detection_threshold", "nms_iou", "position_threshold"])
def test_thresholds_must_be_probabilities(key: str) -> None:
    with pytest.raises(ValueError, match=key):
        _model({}, **{key: 1.5})


@pytest.mark.asyncio
async def test_reward_reads_spec_from_metadata_and_scores_partial() -> None:
    reward = GenEvalOwlReward(
        device="cpu",
        detector=lambda images, classes: (
            [{"bench": [], "cow": [((0, 0, 5, 5), 0.9)]}] * len(images)
        ),
    )
    sample = RewardSample(
        prompt="an anime illustration of a bench and a cow",
        output=_IMAGE,
        sample_id="s0",
        metadata={
            "geneval": {"include": [{"class": "bench", "count": 1}, {"class": "cow", "count": 1}]}
        },
    )
    assert await reward.score(sample) == pytest.approx(0.5)


@pytest.mark.asyncio
async def test_reward_requires_geneval_metadata() -> None:
    reward = GenEvalOwlReward(device="cpu", detector=lambda images, classes: [{}] * len(images))
    sample = RewardSample(prompt="p", output=_IMAGE, sample_id="s0", metadata={})
    with pytest.raises(ValueError, match="metadata\\['geneval'\\]"):
        await reward.score(sample)


def test_nms_drops_a_group_box_spanning_two_instances() -> None:
    """A lower-scored box around both vases must not count as a third vase."""
    kept = nms(
        [((74, 243, 246, 506), 0.78), ((271, 246, 432, 506), 0.78), ((79, 246, 433, 508), 0.22)],
        0.5,
    )
    assert kept == [((74, 243, 246, 506), 0.78), ((271, 246, 432, 506), 0.78)]


def test_nms_keeps_a_box_that_contains_only_one_other() -> None:
    """A single nested detection is a duplicate, not a group; both survive IoU rules."""
    kept = nms([((0, 0, 100, 100), 0.9), ((10, 10, 30, 30), 0.5)], 0.5)
    assert kept == [((0, 0, 100, 100), 0.9), ((10, 10, 30, 30), 0.5)]


def test_dense_presence_orders_near_misses_the_verdict_calls_identical() -> None:
    """Two images the verdict both fails must still be ordered by detection confidence."""
    spec = {"include": [{"class": "cow", "count": 1}]}
    weak = _model({"cow": [((0, 0, 5, 5), 0.06)]}).judge(_IMAGE, spec)
    nearly = _model({"cow": [((0, 0, 5, 5), 0.14)]}).judge(_IMAGE, spec)

    assert (weak.strict, nearly.strict) == (0.0, 0.0)
    assert (weak.partial, nearly.partial) == (0.0, 0.0)
    assert weak.dense < nearly.dense
    # Saturates at twice the verdict floor, so a confident detection reads 1.0.
    assert _model({"cow": [((0, 0, 5, 5), 0.9)]}).judge(_IMAGE, spec).dense == pytest.approx(1.0)


def test_dense_counting_credits_each_found_instance() -> None:
    """Three asked for, two confidently found: partial says 0, dense says two thirds."""
    dets = {"apple": [((0, 0, 5, 5), 0.9), ((10, 0, 15, 5), 0.9)]}
    verdict = _model(dets).judge(_IMAGE, {"include": [{"class": "apple", "count": 3}]})

    assert verdict.partial == 0.0
    assert verdict.dense == pytest.approx(2.0 / 3.0)


def test_dense_color_reads_the_probability_not_the_argmax() -> None:
    spec = {"include": [{"class": "bus", "color": "yellow", "count": 1}]}
    dets = {"bus": [((0, 0, 10, 10), 0.9)]}
    close = _model(dets, {(0.0, 0.0, 10.0, 10.0): {"yellow": 0.45, "orange": 0.55}}).judge(
        _IMAGE, spec
    )
    far = _model(dets, {(0.0, 0.0, 10.0, 10.0): {"yellow": 0.02, "blue": 0.98}}).judge(
        _IMAGE, spec
    )

    assert (close.strict, far.strict) == (0.0, 0.0)
    assert close.dense > far.dense
    # Presence term is 1.0 for both; only the colour term separates them.
    assert close.dense == pytest.approx((1.0 + 0.45) / 2)


def test_dense_position_reads_the_margin() -> None:
    spec = {
        "include": [
            {"class": "kite", "count": 1},
            {"class": "wine glass", "count": 1, "position": ["above", 0]},
        ]
    }
    near = {"kite": [((0, 40, 10, 50), 0.9)], "wine glass": [((0, 25, 10, 35), 0.9)]}
    far = {"kite": [((0, 40, 10, 50), 0.9)], "wine glass": [((0, 0, 10, 10), 0.9)]}

    assert _model(near).judge(_IMAGE, spec).dense < _model(far).judge(_IMAGE, spec).dense


def test_dense_exclude_falls_as_the_forbidden_instance_gets_confident() -> None:
    spec = {
        "include": [{"class": "clock", "count": 2}],
        "exclude": [{"class": "clock", "count": 3}],
    }
    two = _model({"clock": [((0, 0, 5, 5), 0.9), ((10, 0, 15, 5), 0.9)]}).judge(_IMAGE, spec)
    faint_third = _model(
        {"clock": [((0, 0, 5, 5), 0.9), ((10, 0, 15, 5), 0.9), ((20, 0, 25, 5), 0.09)]}
    ).judge(_IMAGE, spec)

    assert (two.strict, faint_third.strict) == (1.0, 1.0)
    assert faint_third.dense < two.dense == pytest.approx(1.0)


def test_reward_rejects_an_unknown_score_key() -> None:
    with pytest.raises(ValueError, match="score_key"):
        GenEvalOwlReward(device="cpu", score_key="geneval_owl")
