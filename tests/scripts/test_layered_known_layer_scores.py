"""Known-layer scores: the truth scores one, and uneditable recompositions lose."""

import numpy as np
import pytest

from vrl.scripts.eval.layered_metrics import SCORE_KEYS, composite, score_layers


def _scene():
    """Background, a box whose lower right sits behind a bar, and the bar in front."""
    layers = np.zeros((3, 16, 16, 4))
    layers[0] = (0.8, 0.7, 0.6, 1)
    layers[1, 2:11, 2:10] = (0.1, 0.3, 0.8, 1)
    layers[2, 6:14, 6:14] = (0.8, 0.2, 0.1, 1)
    return layers


def _axes(truth, candidate, **masks):
    return score_layers(truth, candidate, **masks)["axes"]


def test_truth_scores_one_on_every_axis_and_ignores_transparent_rgb_and_empty_slots():
    truth = _scene()
    padded = np.concatenate((truth, np.zeros_like(truth[:1])))
    padded[..., :3][padded[..., 3] == 0] = (0.9, 0.1, 0.7)
    for candidate in (truth, padded):
        assert _axes(truth, candidate) == pytest.approx(dict.fromkeys(SCORE_KEYS, 1.0))


def test_hidden_corruption_is_penalized_despite_exact_recomposition():
    truth = _scene()
    corrupt = truth.copy()
    corrupt[0, 3:9, 3:9, :3] = 0
    corrupt[1, 7:10, 7:9, :3] = 0
    np.testing.assert_array_equal(composite(truth), composite(corrupt))
    result = score_layers(truth, corrupt)
    assert result["axes"]["layered_reconstruction"] == 1
    assert result["objects"][0]["hidden_rgba_mae"] > 0
    assert result["objects"][1]["hidden_rgba_mae"] > 0
    assert result["axes"]["layered_hidden_completion"] < 1
    assert result["axes"]["layered_edit_holes"] < 1


def test_merging_objects_cannot_win_by_matching_the_composite():
    truth = _scene()
    merged = truth.copy()
    merged[1] = np.where(truth[2, ..., 3:] == 1, truth[2], truth[1])
    merged[2] = 0
    np.testing.assert_array_equal(composite(truth), composite(merged))
    axes = _axes(truth, merged)
    assert axes["layered_object_match"] < 0.5
    assert axes["layered_edit_holes"] < 1


def test_fragmentation_and_duplicate_layers_are_not_free():
    truth = _scene()
    split = np.concatenate((truth[:2], np.zeros_like(truth[:1]), truth[2:]))
    split[2, :, 6:] = split[1, :, 6:]
    split[1, :, 6:] = 0
    duplicate = np.concatenate((truth, truth[2:]))
    for invalid in (split, duplicate):
        np.testing.assert_array_equal(composite(truth), composite(invalid))
        assert _axes(truth, invalid)["layered_nonredundancy"] < 1
    assert _axes(truth, split)["layered_object_match"] < 1


def test_matching_allows_any_layer_order_but_scores_the_resulting_depth_order():
    truth = _scene()
    result = score_layers(truth, truth[[0, 2, 1]])
    assert result["assignment"] == [0, 2, 1]
    assert result["axes"]["layered_object_match"] == pytest.approx(1)
    assert result["axes"]["layered_reconstruction"] < 1
    # Once the two objects no longer overlap, their order changes nothing.
    truth[2] = 0
    truth[2, 12:15, 12:15] = (0.8, 0.2, 0.1, 1)
    assert _axes(truth, truth[[0, 2, 1]]) == pytest.approx(dict.fromkeys(SCORE_KEYS, 1.0))


def test_object_match_follows_every_object_and_a_deleted_object_zeroes_it():
    truth = _scene()
    discolored = truth.copy()
    discolored[2, ..., :3] *= 0.5
    worse = discolored.copy()
    worse[1, ..., :3] *= 0.9
    missing = truth.copy()
    missing[2] = 0
    assert (
        _axes(truth, discolored)["layered_object_match"]
        > _axes(truth, worse)["layered_object_match"]
        > _axes(truth, missing)["layered_object_match"]
        == 0
    )


def test_fully_hidden_damage_does_not_leak_into_visible_retention():
    truth = np.zeros((2, 12, 12, 4))
    truth[0] = (0.7, 0.6, 0.5, 1)
    truth[1, 3:9, 3:9] = (0.1, 0.2, 0.3, 1)
    changed = truth.copy()
    changed[0, 3:9, 3:9, :3] = 0
    result = score_layers(truth, changed)
    assert result["objects"][0]["hidden_rgba_mae"] > 0
    assert result["axes"]["layered_visible_retention"] == 1
    # A completely hidden layer has no visible evidence, not perfect fidelity.
    truth[1, ..., 3] = 1
    assert score_layers(truth, truth)["objects"][0]["visible_rgba_mae"] is None


def test_region_axes_separate_deducible_from_unknowable_hidden_pixels():
    """Erasing the box's hidden pixels fails the inferable axis against truth; the
    unknowable axis asks only that the layer continues there in its own colors."""
    truth = _scene()
    hidden = (truth[1, ..., 3] > 0.5) & (truth[2, ..., 3] > 0.5)
    left = np.zeros_like(hidden)
    left[:, :8] = True
    masks = {"inferable": hidden & left, "unknowable": hidden & ~left}

    erased = truth.copy()
    erased[1][hidden] = 0
    gone = _axes(truth, erased, **masks)
    assert gone["layered_inferable_completion"] < 0.5
    assert gone["layered_unknowable_plausibility"] == 0

    recolored = truth.copy()
    recolored[1][hidden & ~left, :3] = truth[1, 3, 3, :3]
    assert _axes(truth, recolored, **masks)["layered_unknowable_plausibility"] == pytest.approx(1)
    recolored[1][hidden & ~left, :3] = 1
    assert _axes(truth, recolored, **masks)["layered_unknowable_plausibility"] < 0.5


def test_mismatched_geometry_is_rejected():
    truth = _scene()
    with pytest.raises(ValueError, match="identical H,W,4"):
        score_layers(truth, truth[:, ::2, ::2])
