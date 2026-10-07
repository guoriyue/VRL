"""Small visible objects cannot disappear for a cleaner matched layer reward."""

import numpy as np
import pytest

from vrl.scripts.eval.layered_reference_score import coverage_fidelity_reward, score_layers


def test_source_error_is_scaled_by_object_deletion_not_full_canvas_area():
    truth = np.zeros((2, 24, 24, 4))
    truth[0] = (0.8, 0.8, 0.8, 1)
    truth[1, 10:12, 10:12] = (0.9, 0.9, 0.7, 1)
    deleted = truth.copy()
    deleted[1] = 0
    score = score_layers(truth, deleted, include_source_preservation=True)
    source = score["source_preservation"]
    assert score["reconstruction_rgba_mae"] < 0.001
    assert source["objects"][1]["deletion_normalized_error"] == pytest.approx(1)
    assert source["quality"] == pytest.approx(0.5)


def test_hidden_object_does_not_create_a_fake_source_preservation_signal():
    truth = np.zeros((3, 12, 12, 4))
    truth[0] = (0.8, 0.8, 0.8, 1)
    truth[1, 4:8, 4:8] = (0.2, 0.3, 0.4, 1)
    truth[2, 2:10, 2:10] = (0.9, 0.8, 0.2, 1)
    corrupt = truth.copy()
    corrupt[1, ..., :3] = 0
    score = score_layers(truth, corrupt, include_source_preservation=True)
    assert score["source_preservation"]["objects"][1]["deletion_normalized_error"] is None
    assert score["source_preservation"]["quality"] == 1
    assert score["objects"][1]["hidden_rgba_mae"] > 0


def test_deleting_a_merged_small_object_loses_even_when_other_layer_gets_cleaner():
    truth = np.zeros((3, 32, 32, 4))
    truth[0] = (0.8, 0.7, 0.6, 1)
    truth[1, 10:24, 10:24] = (0.5, 0.3, 0.1, 1)
    truth[2, 6:10, 12:16] = (0.9, 0.9, 0.2, 1)
    merged = truth.copy()
    merged[1] = np.where(truth[2, ..., 3:] == 1, truth[2], truth[1])
    merged[2] = 0
    deleted = merged.copy()
    deleted[1, 6:10, 12:16] = 0
    before, after = [
        score_layers(truth, candidate, min_match_quality=0.1, include_source_preservation=True)
        for candidate in [merged, deleted]
    ]
    assert before["assignment"][2] is None and after["assignment"][2] is None
    assert after["objects"][1]["layer_quality"] > before["objects"][1]["layer_quality"]
    assert coverage_fidelity_reward(before)["value"] > coverage_fidelity_reward(after)["value"]
