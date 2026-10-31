"""Removal must reveal copies without punishing unrelated separated objects."""

import numpy as np
import pytest

from vrl.rewards.models.layered_grouping import LayeredReference, LayerStack
from vrl.rewards.models.layered_removal import (
    single_object_removal_quality,
    target_removal_quality,
)


def test_removal_detects_hidden_duplicates_but_allows_unannotated_objects():
    rgba = np.zeros((3, 16, 16, 4), np.float32)
    rgba[0] = [0.4, 0.4, 0.4, 1]
    rgba[1, 2:12, 2:12] = [0.2, 0.2, 0.8, 1]  # An unselected object behind the target.
    rgba[2, 6:14, 6:14] = [0.8, 0.2, 0.2, 1]
    correct = LayerStack.from_rgba(rgba)
    background = LayerStack.from_rgba(rgba[:2]).composite()
    partial = np.stack([np.concatenate([background, np.ones((16, 16, 1))], -1), rgba[2]])
    truth = LayerStack.from_rgba(partial)
    ref = LayeredReference(correct.composite(), list(truth.regions[1:]), truth)
    # The reference leaves the blue object in its background. Both ways of
    # representing that unrelated object must be accepted.
    assert target_removal_quality(correct, ref) == pytest.approx(1)
    assert target_removal_quality(truth, ref) == pytest.approx(1)
    assert single_object_removal_quality(correct, ref)[0]["quality"] == pytest.approx(1)
    assert single_object_removal_quality(truth, ref)[0]["quality"] == pytest.approx(1)
    copied = LayerStack.from_rgba(np.concatenate([rgba[:2], rgba[2:3], rgba[2:3]]))
    assert np.array_equal(copied.composite(), correct.composite())
    assert target_removal_quality(copied, ref) < 1
    assert single_object_removal_quality(copied, ref)[0]["quality"] < 1
    residual = rgba.copy()
    mask = rgba[2, ..., 3] == 1
    residual[0, mask, :3] = rgba[2, mask, :3]
    assert target_removal_quality(LayerStack.from_rgba(residual), ref) < 1
    # Removing a merged target+unselected-object layer also removes content that
    # should remain behind the target, so it cannot receive a perfect edit score.
    merged = np.zeros_like(rgba[1])
    merged[..., :3] = correct.composite()
    merged[..., 3] = np.maximum(rgba[1, ..., 3], rgba[2, ..., 3])
    assert target_removal_quality(LayerStack.from_rgba(np.stack([rgba[0], merged])), ref) < 1


def test_removing_a_target_must_not_erase_a_disjoint_unselected_object():
    rgba = np.zeros((3, 16, 16, 4), np.float32)
    rgba[0] = [0.4, 0.4, 0.4, 1]
    rgba[1, 1:5, 1:5] = [0.2, 0.2, 0.8, 1]
    rgba[2, 10:14, 10:14] = [0.8, 0.2, 0.2, 1]
    correct = LayerStack.from_rgba(rgba)
    bg = LayerStack.from_rgba(rgba[:2]).composite()
    truth = LayerStack.from_rgba(
        np.stack([np.concatenate([bg, np.ones((16, 16, 1))], -1), rgba[2]])
    )
    ref = LayeredReference(correct.composite(), list(truth.regions[1:]), truth)
    merged = np.zeros_like(rgba[1])
    merged[..., :3] = correct.composite()
    merged[..., 3] = np.maximum(rgba[1, ..., 3], rgba[2, ..., 3])
    bad = LayerStack.from_rgba(np.stack([rgba[0], merged]))
    assert target_removal_quality(correct, ref) == pytest.approx(1)
    assert target_removal_quality(bad, ref) < 1


def test_individual_edits_reject_merging_all_annotated_targets():
    rgba = np.zeros((3, 16, 16, 4), np.float32)
    rgba[0] = [0.4, 0.4, 0.4, 1]
    rgba[1, 1:5, 1:5] = [0.2, 0.2, 0.8, 1]
    rgba[2, 10:14, 10:14] = [0.8, 0.2, 0.2, 1]
    truth = LayerStack.from_rgba(rgba)
    ref = LayeredReference(truth.composite(), list(truth.regions[1:]), truth)
    merged = LayerStack.from_rgba(np.stack([rgba[0], np.maximum(rgba[1], rgba[2])]))
    # Deleting the merged layer removes both targets, so the all-target edit
    # alone misleadingly looks perfect. A one-object edit exposes the problem.
    assert target_removal_quality(merged, ref) == pytest.approx(1)
    assert all(r["quality"] == pytest.approx(1) for r in single_object_removal_quality(truth, ref))
    assert all(r["quality"] < 1 for r in single_object_removal_quality(merged, ref))
