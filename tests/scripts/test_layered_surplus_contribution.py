"""Useful unmatched fragments and covered duplicates need different penalties."""

import numpy as np

from vrl.scripts.eval.layered_reference_score import coverage_fidelity_reward, score_layers


def test_useful_fragment_is_not_redundant_but_still_loses_to_complete_layer():
    truth = np.ones((1, 12, 12, 4))
    truth[0, :, :6, :3] = 0.2
    truth[0, :, 6:, :3] = 0.8
    split = np.zeros((2, 12, 12, 4))
    split[0] = truth[0]
    split[0, :, 6:, :3] = 0.2
    split[1, :, 6:] = truth[0, :, 6:]
    historical = score_layers(truth, split, min_match_quality=0.1)
    repaired = score_layers(truth, split, min_match_quality=0.1, surplus_mode="counterfactual")
    intact = score_layers(truth, truth, min_match_quality=0.1, surplus_mode="counterfactual")
    duplicate = score_layers(
        truth, np.repeat(truth, 2, axis=0), min_match_quality=0.1, surplus_mode="counterfactual"
    )
    assert repaired["assignment"] == historical["assignment"] == [0]
    assert repaired["components"]["nonredundancy"] == 1
    assert historical["components"]["nonredundancy"] < 1
    assert duplicate["components"]["nonredundancy"] < 1
    assert coverage_fidelity_reward(intact)["value"] > coverage_fidelity_reward(repaired)["value"]
    assert coverage_fidelity_reward(intact)["value"] > coverage_fidelity_reward(duplicate)["value"]
