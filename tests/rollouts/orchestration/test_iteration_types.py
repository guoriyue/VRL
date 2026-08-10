"""Tests for the trainer-facing rollout iteration boundary."""

from __future__ import annotations

import torch

from vrl.rollouts.batch import RolloutBatch
from vrl.rollouts.orchestration.types import RolloutIteration
from vrl.rollouts.stats import RolloutStats


def _batch(sample_count: int) -> RolloutBatch:
    return RolloutBatch(
        rewards=torch.zeros(sample_count),
        group_ids=torch.zeros(sample_count, dtype=torch.long),
    )


def test_iteration_hands_batches_and_stats_to_the_trainer() -> None:
    batches = [_batch(2)]
    stats = RolloutStats()

    iteration = RolloutIteration(batches=batches, stats=stats)

    assert iteration.batches is batches
    assert iteration.stats is stats


def test_iteration_stats_default_is_not_shared() -> None:
    first = RolloutIteration(batches=[])
    second = RolloutIteration(batches=[])

    assert first.stats is not second.stats
