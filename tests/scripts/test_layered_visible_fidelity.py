"""Visible detail diagnostics reject blur/noise without penalizing hidden changes."""

import numpy as np

from vrl.scripts.eval.layered_reference_score import score_layers


def test_blur_and_noise_cannot_improve_reference_texture_fidelity():
    truth = np.ones((1, 12, 12, 4))
    truth[0, ..., :3] = (np.indices((12, 12)).sum(axis=0) % 2)[..., None]
    blurred = truth.copy()
    blurred[0, ..., :3] = 0.5
    noisy = truth.copy()
    noisy[0, 3:9, 3:9, :3] = 1 - noisy[0, 3:9, 3:9, :3]
    exact = score_layers(truth, truth)["objects"][0]
    assert exact["visible_rgba_mae"] == exact["visible_gradient_mae"] == 0
    for changed in (blurred, noisy):
        result = score_layers(truth, changed)["objects"][0]
        assert result["visible_rgba_mae"] > 0
        assert result["visible_gradient_mae"] > 0


def test_fully_hidden_damage_does_not_leak_into_visible_diagnostics():
    truth = np.zeros((2, 12, 12, 4))
    truth[0] = (0.7, 0.6, 0.5, 1)
    truth[1, 3:9, 3:9] = (0.1, 0.2, 0.3, 1)
    changed = truth.copy()
    changed[0, 3:9, 3:9, :3] = 0
    result = score_layers(truth, changed)["objects"][0]
    assert result["hidden_rgba_mae"] > 0
    assert result["visible_rgba_mae"] == result["visible_gradient_mae"] == 0
    # A completely hidden layer has no visible evidence, not perfect fidelity.
    truth[1, ..., 3] = 1
    hidden = score_layers(truth, truth)["objects"][0]
    assert hidden["visible_rgba_mae"] is None
    assert hidden["visible_gradient_mae"] is None
