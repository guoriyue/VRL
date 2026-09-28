"""Tests for the bounded continuous rollout ready queue container."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest
import torch

from vrl.rollouts.batch import RolloutBatch
from vrl.rollouts.orchestration.continuous.scored_queue import ScoredRolloutQueue
from vrl.rollouts.orchestration.continuous.types import (
    ScoredRollout,
)


def _item(
    group_slot: int,
    version: int | None,
    *,
    samples: int = 2,
    batch_id: int = 0,
) -> ScoredRollout:
    batch = RolloutBatch(
        rewards=torch.zeros(samples),
        group_ids=torch.zeros(samples, dtype=torch.long),
    )
    return ScoredRollout(
        batch_id=batch_id,
        group_slot=group_slot,
        rollout_policy_version=version,
        batch=batch,
    )


@pytest.mark.parametrize("field, replacement", [("batch_id", 5), ("rollout_policy_version", 9)])
def test_ready_receipt_fields_cannot_change_after_admission(field, replacement) -> None:
    queue = ScoredRolloutQueue(max_items=1)
    item = _item(group_slot=0, version=1)
    queue.put(item)
    with pytest.raises(FrozenInstanceError):
        setattr(item, field, replacement)
    queue.remove([item])
    assert queue.size() == 0


def test_item_limit_can_grow_but_cannot_discard_resident_items() -> None:
    queue = ScoredRolloutQueue(max_items=1)
    queue.set_item_limit(2)
    assert queue.max_items == 2

    queue.put(_item(group_slot=0, version=1))
    queue.set_item_limit(3)
    queue.put(_item(group_slot=1, version=1))
    with pytest.raises(RuntimeError, match="below resident items"):
        queue.set_item_limit(1)
    assert queue.max_items == 3


def test_snapshot_and_remove_are_pure_container_ops() -> None:
    """snapshot() reads FIFO order; remove() drops by identity."""
    queue = ScoredRolloutQueue(max_items=8)
    queue.put(_item(group_slot=0, version=1))
    queue.put(_item(group_slot=1, version=1))
    snap = queue.snapshot()
    assert [item.group_slot for item in snap] == [0, 1]

    queue.remove([snap[0]])
    assert [item.group_slot for item in queue.snapshot()] == [1]


def test_item_count_overflow_fails_before_mutation() -> None:
    queue = ScoredRolloutQueue(max_items=2)
    queue.put(_item(group_slot=0, version=1))
    queue.put(_item(group_slot=1, version=1))

    with pytest.raises(ValueError, match="item limit"):
        queue.put(_item(group_slot=2, version=1))

    assert queue.size() == 2
    assert [item.group_slot for item in queue.snapshot()] == [0, 1]


def test_stats_shape() -> None:
    """``stats()`` reports ready items and the oldest item age under exactly those keys."""
    queue = ScoredRolloutQueue(max_items=8)
    queue.put(_item(group_slot=0, version=1))
    stats = queue.stats()
    assert stats.keys() == {"ready_items", "oldest_item_age_s"}
    assert stats["ready_items"] == 1.0
