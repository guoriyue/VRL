"""Visible grouping must not reward discarding a pure object fragment."""

import numpy as np
import pytest

from tools.layering_reward_audit import from_masks
from vrl.rewards.models.layered_grouping import LayeredGroupingRewardModel, LayeredReference


def test_pure_fragment_in_background_loses_credit_without_prediction_normalization():
    a = np.zeros((32, 32), bool)
    a[2:12, 2:12] = True
    b = np.zeros_like(a)
    b[20:30, 20:30] = True
    major = a.copy()
    major[:, 8:] = False
    minor = a & ~major
    source = np.full((32, 32, 3), 0.4, np.float32)
    reference = LayeredReference(source, [a, b])
    scorer = LayeredGroupingRewardModel({"grouping_mode": "partition"})
    perfect = scorer.score(from_masks([a, b], source), reference)["grouping"]
    split = scorer.score(from_masks([major, minor, b], source), reference)["grouping"]
    dropped = scorer.score(from_masks([major, b], source), reference)["grouping"]
    merged = scorer.score(from_masks([a | b], source), reference)["grouping"]
    assert perfect == pytest.approx(1)
    assert perfect > split > dropped
    assert perfect > merged
    # Legacy mode remains unchanged and explicitly exposes the old tradeoff.
    legacy = LayeredGroupingRewardModel({})
    assert (
        legacy.score(from_masks([major, b], source), reference)["grouping"]
        > legacy.score(from_masks([major, minor, b], source), reference)["grouping"]
    )


def test_unknown_pixels_and_empty_slots_do_not_change_annotated_partition():
    a = np.zeros((24, 24), bool)
    a[2:10, 2:10] = True
    b = np.zeros_like(a)
    b[14:22, 14:22] = True
    unknown = np.zeros_like(a)
    unknown[2:10, 14:22] = True
    source = np.full((24, 24, 3), 0.4, np.float32)
    reference = LayeredReference(source, [a, b])
    scorer = LayeredGroupingRewardModel({"grouping_mode": "partition"})
    perfect = scorer.score(from_masks([a, b], source), reference)["grouping"]
    padded = scorer.score(from_masks([a | unknown, b, np.zeros_like(a)], source), reference)[
        "grouping"
    ]
    separate_unknown = scorer.score(from_masks([a, b, unknown], source), reference)["grouping"]
    assert padded == separate_unknown == perfect


def test_full_reference_score_preserves_legal_overlap_and_rejects_hidden_duplicates():
    from vrl.rewards.models.layered_grouping import LayerStack

    rgba = np.zeros((3, 32, 32, 4), np.float32)
    rgba[0] = [0.3, 0.3, 0.3, 1]
    rgba[1, 4:24, 4:24] = [0.8, 0.2, 0.1, 1]
    rgba[2, 12:28, 12:28] = [0.1, 0.2, 0.8, 1]
    truth = LayerStack.from_rgba(rgba)
    reference = LayeredReference(
        truth.composite(), list(truth.regions[1:]), truth, complete_layer_truth=True
    )
    scorer = LayeredGroupingRewardModel(
        {
            "grouping_mode": "partition",
            "integrity_mode": "reference",
            "integrity_weight": 1,
            "completion_weight": 0,
        }
    )
    correct = scorer.score(truth, reference)
    assert correct["integrity"] == pytest.approx(1)
    copied = LayerStack.from_rgba(np.concatenate([rgba[:1], rgba[1:2], rgba[1:]]))
    assert np.array_equal(copied.composite(), truth.composite())
    assert scorer.score(copied, reference)["layered_grouping"] < correct["layered_grouping"]
    left, right = rgba[1].copy(), rgba[1].copy()
    left[:, 14:] = 0
    right[:, :14] = 0
    split = LayerStack.from_rgba(np.stack([rgba[0], left, right, rgba[2]]))
    background = rgba[0].copy()
    mask = right[..., 3] == 1
    background[mask] = right[mask]
    dropped = LayerStack.from_rgba(np.stack([background, left, rgba[2]]))
    assert np.array_equal(split.composite(), dropped.composite())
    assert (
        scorer.score(split, reference)["layered_grouping"]
        > scorer.score(dropped, reference)["layered_grouping"]
    )
