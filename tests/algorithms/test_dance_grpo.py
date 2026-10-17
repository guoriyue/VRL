"""DanceGRPO = FlowGRPO loss + random denoise-timestep selection.

verl-omni registers dance_grpo to the same FlowGRPOLoss as flow_grpo, so vrl
reuses the GRPO algorithm/config unchanged; the only behavioral delta is the
trainer's per-update random timestep subset (vs the default strided subset).
These pin the selection helper.
"""

from __future__ import annotations

import torch

from vrl.trainers.online.trainer import OnlineTrainer

_pick = OnlineTrainer._train_timestep_indices


# --------------------------------------------------------- timestep selection


def test_strided_is_deterministic_and_evenly_spaced() -> None:
    a = _pick(10, 0.5, "strided")
    b = _pick(10, 0.5, "strided")
    assert a == b  # fixed every call
    assert a == [0, 2, 4, 6, 8]
    assert len(a) == 5


def test_random_picks_a_valid_sorted_subset_of_the_right_size() -> None:
    torch.manual_seed(0)
    picks = _pick(20, 0.5, "random")
    assert len(picks) == 10
    assert picks == sorted(picks)  # returned sorted; loop sums over indices
    assert len(set(picks)) == 10  # no repeats (randperm, not sampling w/ replace)
    assert all(0 <= i < 20 for i in picks)


def test_random_resamples_across_calls() -> None:
    # Fresh subset each update is the whole point — decorrelates step coverage.
    torch.manual_seed(0)
    first = _pick(50, 0.4, "random")
    second = _pick(50, 0.4, "random")
    assert first != second


def test_at_least_one_step_even_at_tiny_fraction() -> None:
    assert len(_pick(10, 0.0, "strided")) == 1
    assert len(_pick(10, 0.0, "random")) == 1
