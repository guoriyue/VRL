"""A complete-layer reward must respond beyond its worst-object bottleneck."""

import numpy as np

from vrl.scripts.eval.layered_reference_score import score_layers


def test_better_object_damage_is_not_hidden_by_worst_object_plateau():
    truth = np.zeros((2, 12, 12, 4))
    truth[0, :6] = (0.8, 0.8, 0.8, 1)
    truth[1, 6:] = (0.8, 0.8, 0.8, 1)
    candidate = truth.copy()
    candidate[1, ..., :3] *= 0.5
    worse = candidate.copy()
    worse[0, ..., :3] *= 0.9
    original = score_layers(truth, candidate)
    damaged = score_layers(truth, worse)
    assert original["diagnostic_score"] == damaged["diagnostic_score"]
    assert original["reward_proposal"]["value"] > damaged["reward_proposal"]["value"]
    # Fixing the plateau must not reintroduce compensation for a deleted object.
    deleted = truth.copy()
    deleted[1] = 0
    assert score_layers(truth, deleted)["reward_proposal"]["value"] == 0
    assert damaged["reward_proposal"]["value"] > 0
