"""Exterior spill cannot buy a higher constrained reward through better completion."""

import numpy as np
import pytest

from vrl.scripts.eval.layered_reference_score import constrained_fidelity_reward, score_layers


def test_completion_gain_cannot_compensate_for_exterior_spill():
    truth = np.zeros((2, 16, 16, 4))
    truth[0, 2:14, 2:14] = (0.7, 0.5, 0.3, 1)
    truth[1, 5:11, 5:11] = (0.9, 0.9, 0.2, 1)
    damaged = truth.copy()
    damaged[0, 5:11, 5:11, :3] = 0
    restored_with_spill = truth.copy()
    exterior = truth[..., 3].max(axis=0) == 0
    restored_with_spill[0, exterior, 3] = 1 / 255
    settings = dict(min_alpha_iou=0.75, min_alpha_recall=0.98, max_surplus_fraction=0.01)
    damaged_score = score_layers(truth, damaged)
    spill_score = score_layers(truth, restored_with_spill)
    old_damaged = constrained_fidelity_reward(damaged_score, **settings)
    old_spill = constrained_fidelity_reward(spill_score, **settings)
    assert old_spill["value"] > old_damaged["value"]
    protected_damaged = constrained_fidelity_reward(
        damaged_score, **settings, max_exterior_alpha=0
    )
    protected_spill = constrained_fidelity_reward(spill_score, **settings, max_exterior_alpha=0)
    assert protected_spill["geometry_gate_passed"]
    assert not protected_spill["constraints_passed"]
    assert protected_spill["value"] < 0 <= protected_damaged["value"]
    assert protected_damaged["constraints_passed"]
    assert (
        constrained_fidelity_reward(score_layers(truth, truth), **settings, max_exterior_alpha=0)[
            "value"
        ]
        > protected_damaged["value"]
    )


def test_empty_layers_cannot_dilute_exterior_spill_and_transparent_rgb_is_ignored():
    truth = np.zeros((1, 8, 8, 4))
    truth[0, 2:6, 2:6] = (1, 0, 0, 1)
    candidate = truth.copy()
    candidate[0, 0, 0] = (0, 0, 1, 0.2)
    score = score_layers(truth, candidate)
    padded = np.concatenate([candidate, np.zeros_like(candidate)])
    padded[1, ..., :3] = 1
    assert score["exterior_pixel_count"] == 48
    assert score["exterior_alpha_mean"] == pytest.approx(0.2 / 48)
    assert score_layers(truth, padded)["exterior_alpha_mean"] == score["exterior_alpha_mean"]
    padded[1, 0, 0, 3] = 0.3
    assert score_layers(truth, padded)["exterior_alpha_mean"] == pytest.approx(0.44 / 48)


def test_opaque_reference_has_no_exterior_and_cap_rejects_invalid_numbers():
    truth = np.ones((1, 8, 8, 4))
    score = score_layers(truth, truth)
    assert score["exterior_pixel_count"] == 0
    settings = dict(min_alpha_iou=0.75, min_alpha_recall=0.98, max_surplus_fraction=0.01)
    assert constrained_fidelity_reward(score, **settings, max_exterior_alpha=0)[
        "constraints_passed"
    ]
    with pytest.raises(ValueError, match="finite"):
        constrained_fidelity_reward(score, **settings, max_exterior_alpha=float("nan"))
