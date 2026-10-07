"""Object-only decomposition must score edits to its first layer too."""

import numpy as np
import pytest

from vrl.scripts.eval.layered_reference_score import score_layers


def test_first_object_move_exposes_color_damage_without_affecting_hidden_empty_canvas():
    truth = np.zeros((1, 16, 16, 4))
    truth[0, 4:12, 4:12] = [1, 0, 0, 1]
    wrong_color = truth.copy()
    wrong_color[0, 4:12, 4:12] = [0, 0, 1, 1]

    legacy = score_layers(truth, wrong_color)
    edited = score_layers(truth, wrong_color, include_first_layer_edits=True)
    assert legacy["components"]["edit_quality"] == 1
    # Removing the sole object leaves an empty canvas for either color.
    assert edited["objects"][0]["hide_rgba_mae"] == 0
    # A two-pixel shift leaves eight damaged columns inside ten union columns.
    assert edited["objects"][0]["move_rgba_mae"] == pytest.approx(0.4)
    assert edited["components"]["edit_quality"] == pytest.approx(0.8)
    exact = score_layers(truth, truth, include_first_layer_edits=True)
    assert exact["components"]["edit_quality"] == 1
    assert edited["reconstruction_rgba_mae"] == legacy["reconstruction_rgba_mae"]
