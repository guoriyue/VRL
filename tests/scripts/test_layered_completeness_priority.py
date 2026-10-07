"""Appearance cannot compensate for lost support, nor separation for deletion."""

import numpy as np

from vrl.scripts.eval.layered_reference_score import completeness_priority_reward, score_layers


def test_complete_discolored_object_beats_missing_parts_without_a_fidelity_plateau():
    truth = np.zeros((2, 20, 20, 4))
    truth[0] = (0.8, 0.8, 0.8, 1)
    truth[1, 4:16, 4:16] = (0.4, 0.2, 0.1, 1)
    discolored = truth.copy()
    discolored[1, ..., :3] *= 0.9
    incomplete = truth.copy()
    incomplete[1, 14:16] = 0
    incomplete_discolored = incomplete.copy()
    incomplete_discolored[1, ..., :3] *= 0.9
    rows = [
        completeness_priority_reward(
            score_layers(truth, x, min_match_quality=0.1, include_source_preservation=True),
            min_alpha_iou=0.75,
            min_alpha_recall=0.98,
            max_normalized_source_error=1,
        )
        for x in [discolored, incomplete, incomplete_discolored]
    ]
    assert rows[0]["complete_support_references"] == 2
    assert rows[1]["complete_support_references"] == rows[2]["complete_support_references"] == 1
    assert rows[0]["value"] > rows[1]["value"] > rows[2]["value"]


def test_deleting_merged_object_cannot_win_by_completing_another_support():
    truth = np.zeros((3, 24, 24, 4))
    truth[0] = (0.8, 0.8, 0.8, 1)
    truth[1, 12:20, 8:16] = (0.3, 0.2, 0.1, 1)
    truth[2, 4:12, 8:16] = (0.9, 0.9, 0.2, 1)
    merged = truth.copy()
    merged[1, 4:12] = truth[2, 4:12]
    merged[2] = 0
    deleted = merged.copy()
    deleted[1, 4:12] = 0
    rows = [
        completeness_priority_reward(
            score_layers(truth, x, min_match_quality=0.1, include_source_preservation=True),
            min_alpha_iou=0.75,
            min_alpha_recall=0.98,
            max_normalized_source_error=1,
        )
        for x in [merged, deleted]
    ]
    assert rows[1]["complete_support_references"] > rows[0]["complete_support_references"]
    assert rows[1]["preserved_visible_references"] < rows[0]["preserved_visible_references"]
    assert rows[1]["value"] < rows[0]["value"]
