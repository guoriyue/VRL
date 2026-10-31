"""Invisible duplicates must lose reward without penalizing legitimate occlusion."""

import numpy as np
import pytest

from vrl.rewards.models.layered_grouping import (
    LayeredGroupingRewardModel,
    LayeredReference,
    LayerStack,
)


def test_full_truth_rejects_hidden_copies_and_background_residuals():
    rgba = np.zeros((3, 32, 32, 4), dtype=np.float32)
    rgba[0] = [0.4, 0.4, 0.4, 1]
    rgba[1, 4:20, 4:20] = [0.8, 0.2, 0.2, 1]
    rgba[2, 12:28, 12:28] = [0.2, 0.2, 0.8, 1]
    truth = LayerStack.from_rgba(rgba)
    hidden = (rgba[1, ..., 3] > 0.5) & ~truth.regions[1]
    reference = LayeredReference(truth.composite(), list(truth.regions[1:]), truth, hidden, True)
    model = LayeredGroupingRewardModel({"integrity_weight": 1})
    correct = model.score(truth, reference)
    assert correct["integrity"] == pytest.approx(1)
    assert correct["completion"] == pytest.approx(1)
    extra = LayerStack.from_rgba(np.concatenate([rgba[:1], rgba[1:2], rgba[1:]]))
    background_copy = rgba.copy()
    mask = rgba[1, ..., 3] > 0.5
    background_copy[0, mask, :3] = rgba[1, mask, :3]
    residual = LayerStack.from_rgba(background_copy)
    assert np.array_equal(extra.composite(), truth.composite())
    assert np.array_equal(residual.composite(), truth.composite())
    assert model.score(extra, reference)["full_object_quality"] < 1
    assert model.score(residual, reference)["clean_background"] < 1
    assert model.score(extra, reference)["layered_grouping"] < correct["layered_grouping"]
    assert model.score(residual, reference)["layered_grouping"] < correct["layered_grouping"]
    empty = np.zeros_like(rgba[:1])
    padded = LayerStack.from_rgba(np.concatenate([rgba[:1], empty, rgba[1:], empty]))
    assert model.score(padded, reference)["integrity"] == pytest.approx(1)
    # The same truth is not exhaustive unless explicitly declared. A synthetic
    # photo occlusion with only two known objects cannot score the whole scene.
    partial = LayeredReference(truth.composite(), list(truth.regions[1:]), truth, hidden)
    assert model.score(extra, partial)["integrity"] == 0
    assert model.score(truth, partial)["integrity"] == 0


def test_full_content_checks_hidden_rgb_and_wrong_layer_alpha():
    rgba = np.zeros((3, 16, 16, 4), dtype=np.float32)
    rgba[0] = [0.3, 0.3, 0.3, 1]
    rgba[1, 2:12, 2:12] = [0.8, 0.1, 0.1, 1]
    rgba[2, 7:15, 7:15] = [0.1, 0.1, 0.8, 1]
    truth = LayerStack.from_rgba(rgba)
    reference = LayeredReference(
        truth.composite(), list(truth.regions[1:]), truth, complete_layer_truth=True
    )
    bad = rgba.copy()
    bad[1, 7:12, 7:12, :3] = [0.1, 0.1, 0.8]
    corrupted = LayerStack.from_rgba(bad)
    assert np.array_equal(corrupted.composite(), truth.composite())
    assert reference.integrity(corrupted)[0] < reference.integrity(truth)[0]
    with pytest.raises(ValueError, match="truth layers"):
        LayeredReference(truth.composite(), [], complete_layer_truth=True).integrity(truth)


def test_copying_a_merged_layer_cannot_claim_a_second_visible_object():
    rgba = np.zeros((3, 32, 32, 4), dtype=np.float32)
    rgba[0] = [0.3, 0.3, 0.3, 1]
    rgba[1, 4:12, 4:12] = [0.8, 0.1, 0.1, 1]
    rgba[2, 20:28, 20:28] = [0.1, 0.8, 0.1, 1]
    truth = LayerStack.from_rgba(rgba)
    reference = LayeredReference(
        truth.composite(),
        list(truth.regions[1:]),
        truth,
        np.zeros((32, 32), dtype=bool),
        True,
    )
    merged = np.maximum(rgba[1], rgba[2])
    original = LayerStack.from_rgba(np.stack([rgba[0], merged]))
    copies = np.stack([rgba[0], merged, merged])
    hidden = LayerStack.from_rgba(copies)
    scorer = LayeredGroupingRewardModel({"integrity_weight": 1})
    assert np.array_equal(original.composite(), hidden.composite())
    assert (
        scorer.score(hidden, reference)["layered_grouping"]
        < scorer.score(original, reference)["layered_grouping"]
    )
    # A single exposed pixel must not allow the second copy to collect credit
    # for an entire second object either.
    copies[2, 20, 20, 3] = 0
    token_visible = LayerStack.from_rgba(copies)
    assert np.array_equal(original.composite(), token_visible.composite())
    assert reference.integrity(token_visible)[0] < reference.integrity(original)[0]
