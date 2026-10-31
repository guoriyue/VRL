"""Independent excess penalties distinguish hidden copies from legal overlap."""

import numpy as np
import pytest

from vrl.rewards.models.layered_grouping import (
    LayeredGroupingRewardModel,
    LayeredReference,
    LayerStack,
)


def test_hidden_copies_lose_credit_without_penalizing_overlap_or_empty_slots():
    rgba = np.zeros((3, 32, 32, 4), dtype=np.float32)
    rgba[0] = [0.3, 0.3, 0.3, 1]
    rgba[1, 4:24, 4:24] = [0.8, 0.2, 0.1, 1]
    rgba[2, 12:28, 12:28] = [0.8, 0.2, 0.1, 1]
    truth = LayerStack.from_rgba(rgba)
    reference = LayeredReference(
        truth.composite(), list(truth.regions[1:]), truth, complete_layer_truth=True
    )
    config = dict(grouping_mode="partition", integrity_mode="reference", integrity_weight=1)
    old = LayeredGroupingRewardModel(config)
    new = LayeredGroupingRewardModel({**config, "excess_weight": 1})
    copied = LayerStack.from_rgba(np.concatenate([rgba[:1], rgba[1:2], rgba[1:]]))
    padded = LayerStack.from_rgba(np.concatenate([rgba, np.zeros_like(rgba[:1])]))
    assert np.array_equal(copied.composite(), truth.composite())
    assert reference.excess_foreground_opacity(truth) == 0
    assert (
        new.score(truth, reference)["layered_grouping"]
        == old.score(truth, reference)["layered_grouping"]
    )
    assert (
        new.score(padded, reference)["layered_grouping"]
        == new.score(truth, reference)["layered_grouping"]
    )
    assert (
        new.score(copied, reference)["layered_grouping"]
        < old.score(copied, reference)["layered_grouping"]
    )
    # A visible mask must not masquerade as an amodal reference.
    partial = LayeredReference(truth.composite(), list(truth.regions[1:]), truth)
    assert (
        new.score(copied, partial)["layered_grouping"]
        == old.score(copied, partial)["layered_grouping"]
    )


def test_hiding_a_visible_fragment_in_background_does_not_gain_reward():
    rgba = np.zeros((2, 32, 32, 4), dtype=np.float32)
    rgba[0] = [0.3, 0.3, 0.3, 1]
    rgba[1, 4:28, 4:28] = [0.8, 0.2, 0.1, 1]
    truth = LayerStack.from_rgba(rgba)
    reference = LayeredReference(
        truth.composite(), list(truth.regions[1:]), truth, complete_layer_truth=True
    )
    major, minor = rgba[1].copy(), rgba[1].copy()
    major[:, :12, 3] = 0
    minor[:, 12:, 3] = 0
    split = LayerStack.from_rgba(np.stack([rgba[0], minor, major]))
    hidden = split.rgba.copy()
    mask = minor[..., 3] > 0
    hidden[0, mask] = minor[mask]
    hidden[1] = 0
    dropped = LayerStack.from_rgba(hidden)
    assert np.array_equal(split.composite(), dropped.composite())
    scorer = LayeredGroupingRewardModel(
        dict(
            grouping_mode="partition",
            integrity_mode="reference",
            integrity_weight=1,
            excess_weight=1,
        )
    )
    assert (
        scorer.score(truth, reference)["layered_grouping"]
        > scorer.score(split, reference)["layered_grouping"]
        > scorer.score(dropped, reference)["layered_grouping"]
    )
    with pytest.raises(ValueError, match="excess_weight"):
        LayeredGroupingRewardModel({"excess_weight": float("nan")})
