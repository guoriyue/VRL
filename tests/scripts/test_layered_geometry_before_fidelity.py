"""Good appearance must not compensate for missing or duplicate layer geometry."""

import numpy as np

from vrl.scripts.eval.layered_reference_score import constrained_fidelity_reward, score_layers


def test_missing_and_duplicate_objects_lose_to_complete_discolored_layers():
    truth = np.zeros((2, 16, 16, 4))
    truth[0, 2:12, 2:12] = (0.7, 0.5, 0.3, 1)
    truth[1, 4:8, 4:8] = (0.9, 0.9, 0.2, 1)
    discolored = truth.copy()
    discolored[..., :3] *= 0.5
    deleted = truth.copy()
    deleted[1] = 0
    duplicate = np.concatenate([truth, truth[:1]])
    scores = [
        constrained_fidelity_reward(
            score_layers(truth, candidate),
            min_alpha_iou=0.75,
            min_alpha_recall=0.98,
            max_surplus_fraction=0.01,
        )
        for candidate in [truth, discolored, deleted, duplicate]
    ]
    assert scores[0]["value"] > scores[1]["value"] >= 0
    assert scores[0]["geometry_gate_passed"] and scores[1]["geometry_gate_passed"]
    assert all(not score["geometry_gate_passed"] and score["value"] < 0 for score in scores[2:])
