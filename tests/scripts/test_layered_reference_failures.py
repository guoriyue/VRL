"""Known-reference scoring rejects uneditable pixel-perfect recompositions."""

import numpy as np
import pytest

from vrl.scripts.eval.layered_reference_score import composite, score_layers


def _scene():
    layers = np.zeros((3, 16, 16, 4))
    layers[0] = (0.8, 0.7, 0.6, 1)
    layers[1, 2:11, 2:10] = (0.1, 0.3, 0.8, 1)
    layers[2, 6:14, 6:14] = (0.8, 0.2, 0.1, 1)
    return layers


def test_hidden_corruption_is_penalized_despite_exact_recomposition():
    truth = _scene()
    corrupt = truth.copy()
    corrupt[0, 3:9, 3:9, :3] = 0
    corrupt[1, 7:10, 7:9, :3] = 0
    np.testing.assert_array_equal(composite(truth), composite(corrupt))
    result = score_layers(truth, corrupt)
    assert result["reconstruction_rgba_mae"] == 0
    assert result["objects"][0]["hidden_rgba_mae"] > 0
    assert result["objects"][1]["hidden_rgba_mae"] > 0
    assert result["components"]["edit_quality"] < 1
    assert result["diagnostic_score"] < score_layers(truth, truth)["diagnostic_score"]


def test_merging_objects_cannot_win_by_matching_the_composite():
    truth = _scene()
    merged = truth.copy()
    merged[1] = np.where(truth[2, ..., 3:] == 1, truth[2], truth[1])
    merged[2] = 0
    np.testing.assert_array_equal(composite(truth), composite(merged))
    result = score_layers(truth, merged)
    assert result["components"]["mean_layer_quality"] < 0.8
    assert result["components"]["edit_quality"] < 1


def test_fragmentation_and_duplicate_layers_are_not_free():
    truth = _scene()
    split = np.concatenate((truth[:2], np.zeros_like(truth[:1]), truth[2:]))
    split[2, :, 6:] = split[1, :, 6:]
    split[1, :, 6:] = 0
    duplicate = np.concatenate((truth, truth[2:]))
    for invalid in (split, duplicate):
        np.testing.assert_array_equal(composite(truth), composite(invalid))
        result = score_layers(truth, invalid)
        assert result["diagnostic_score"] < 1
        assert result["components"]["nonredundancy"] < 1


def test_transparent_rgb_and_unused_empty_slots_do_not_change_quality():
    truth = _scene()
    candidate = np.concatenate((truth, np.zeros_like(truth[:1])))
    candidate[..., :3][candidate[..., 3] == 0] = (0.9, 0.1, 0.7)
    assert score_layers(truth, candidate)["diagnostic_score"] == pytest.approx(1)


def test_matching_preserves_permutation_freedom_but_checks_depth_order():
    truth = _scene()
    wrong_order = truth[[0, 2, 1]]
    result = score_layers(truth, wrong_order)
    assert result["assignment"] == [0, 2, 1]
    assert result["components"]["mean_layer_quality"] == pytest.approx(1)
    assert result["reconstruction_rgba_mae"] > 0
    assert result["diagnostic_score"] < 1
    truth[2] = 0
    truth[2, 12:15, 12:15] = (0.8, 0.2, 0.1, 1)
    assert score_layers(truth, truth[[0, 2, 1]])["diagnostic_score"] == pytest.approx(1)
