"""Incomplete candidates must not earn object credit from unrelated fragments."""

import numpy as np
import pytest

from vrl.scripts.eval.layered_reference_score import (
    composite,
    coverage_fidelity_reward,
    score_layers,
)


def test_rejoining_room_fragment_does_not_erase_credit_for_remaining_objects():
    truth = np.zeros((3, 32, 32, 4))
    truth[0] = (0.8, 0.7, 0.6, 1)
    truth[1, 10:24, 10:24] = (0.5, 0.3, 0.1, 1)
    truth[2, 6:12, 12:16] = (0.9, 0.9, 0.2, 1)
    fragmented = truth.copy()
    fragmented[1] = np.where(truth[2, ..., 3:] == 1, truth[2], truth[1])
    fragmented[2] = 0
    fragmented[2, 6] = truth[0, 6]
    fragmented[0, 6] = 0
    # Place the room fragment below the merged foreground.
    fragmented = fragmented[[0, 2, 1]]
    rejoined = fragmented.copy()
    rejoined[0, 6] = rejoined[1, 6]
    rejoined[1] = 0
    np.testing.assert_array_equal(composite(fragmented), composite(rejoined))
    old = score_layers(truth, fragmented)
    assert old["assignment"][2] == 1
    assert (
        old["reward_proposal"]["value"] > score_layers(truth, rejoined)["reward_proposal"]["value"]
    )
    before = score_layers(truth, fragmented, min_match_quality=0.1)
    after = score_layers(truth, rejoined, min_match_quality=0.1)
    assert before["assignment"][2] is None and after["assignment"][2] is None
    assert coverage_fidelity_reward(after)["value"] > coverage_fidelity_reward(before)["value"]
    damaged = rejoined.copy()
    damaged[0, 15:20, 15:20, :3] = 0
    np.testing.assert_array_equal(composite(rejoined), composite(damaged))
    assert (
        coverage_fidelity_reward(after)["value"]
        > coverage_fidelity_reward(score_layers(truth, damaged, min_match_quality=0.1))["value"]
    )


def test_coverage_priority_does_not_reward_deleting_or_duplicating_an_object():
    truth = np.zeros((2, 16, 16, 4))
    truth[0, 2:12, 2:12] = (0.7, 0.5, 0.3, 1)
    truth[1, 4:8, 4:8] = (0.9, 0.9, 0.2, 1)
    discolored = truth.copy()
    discolored[..., :3] = 0
    deleted = truth.copy()
    deleted[1] = 0
    duplicate = np.concatenate([truth, truth[:1]])
    values = [
        coverage_fidelity_reward(score_layers(truth, x, min_match_quality=0.1))["value"]
        for x in [truth, discolored, deleted, duplicate]
    ]
    assert values[0] == pytest.approx(1)
    assert values[0] > values[1] > values[2]
    assert values[0] > values[3]


def test_match_floor_requires_an_explicit_valid_setting():
    truth = np.ones((1, 2, 2, 4))
    with pytest.raises(ValueError, match="positive match floor"):
        coverage_fidelity_reward(score_layers(truth, truth))
    for floor in [-0.1, 1, float("nan")]:
        with pytest.raises(ValueError, match="Minimum match quality"):
            score_layers(truth, truth, min_match_quality=floor)


def test_complete_parts_cannot_compensate_for_merging_two_objects():
    truth = np.zeros((4, 32, 40, 4))
    truth[0] = (0.8, 0.7, 0.6, 1)
    truth[1, 12:24, 8:24] = (0.5, 0.3, 0.1, 1)
    truth[1, 24:28, 8:10] = (0.5, 0.3, 0.1, 1)
    truth[1, 24:28, 22:24] = (0.5, 0.3, 0.1, 1)
    truth[2, 6:12, 14:18] = (0.9, 0.9, 0.2, 1)
    truth[3, 12:28, 28:36] = (0.2, 0.6, 0.4, 1)
    footless = truth.copy()
    footless[1, 24:] = 0
    merged = truth.copy()
    merged[1] = np.where(truth[2, ..., 3:] == 1, truth[2], truth[1])
    merged[2] = 0
    separate_score = score_layers(truth, footless, min_match_quality=0.1)
    merged_score = score_layers(truth, merged, min_match_quality=0.1)
    separate = coverage_fidelity_reward(separate_score)
    coupled = coverage_fidelity_reward(merged_score)
    # Reproduce the failure of an unrestricted part bonus before checking repair.
    assert 0.75 * coupled["value"] + 0.25 > 0.75 * separate["value"]
    repaired_separate = coverage_fidelity_reward(separate_score, part_quality=0)
    repaired_merged = coverage_fidelity_reward(merged_score, part_quality=1)
    assert repaired_separate["value"] > repaired_merged["value"]
    assert (
        coverage_fidelity_reward(separate_score, part_quality=1)["value"]
        > repaired_separate["value"]
    )
    assert (
        coverage_fidelity_reward(separate_score, part_quality=1, part_weight=0)["value"]
        == separate["value"]
    )
    for invalid in (-0.1, 1.1, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="Part quality"):
            coverage_fidelity_reward(separate_score, part_quality=invalid)
        with pytest.raises(ValueError, match="Part weight"):
            coverage_fidelity_reward(separate_score, part_quality=0.5, part_weight=invalid)
