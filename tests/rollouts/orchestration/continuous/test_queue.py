"""Tests for the bounded continuous rollout ready queue (container mechanism).

Version/staleness behavior (validation and homogeneous select) lives on the
``RolloutScheduler`` now and is covered by ``test_scheduler.py``; this file pins
only the container: byte/count backpressure and stats.
"""

from __future__ import annotations

import torch

from vrl.rollouts.batch import RolloutBatch
from vrl.rollouts.orchestration.continuous.queue import ContinuousRolloutQueue
from vrl.rollouts.orchestration.continuous.types import (
    ContinuousRolloutItem,
    estimate_batch_bytes,
)


def _item(
    group_key: int,
    version: int | None,
    *,
    samples: int = 2,
    nbytes: int = 0,
) -> ContinuousRolloutItem:
    batch = RolloutBatch(
        observations=torch.zeros(samples, 1),
        actions=torch.zeros(samples, 1),
        rewards=torch.zeros(samples),
        group_ids=torch.zeros(samples, dtype=torch.long),
    )
    return ContinuousRolloutItem(
        group_key=group_key,
        rollout_policy_version=version,
        batch=batch,
        nbytes=nbytes,
    )


def test_snapshot_and_remove_are_pure_container_ops() -> None:
    """snapshot() reads FIFO order; remove() drops by identity and fixes bytes."""
    queue = ContinuousRolloutQueue(max_items=8)
    queue.put(_item(group_key=0, version=1, nbytes=4))
    queue.put(_item(group_key=1, version=1, nbytes=6))
    snap = queue.snapshot()
    assert [item.group_key for item in snap] == [0, 1]

    queue.remove([snap[0]])
    assert queue.size() == 1
    assert queue.ready_bytes() == 6


def test_item_count_backpressure_drops_oldest() -> None:
    """Checks item count backpressure drops oldest."""
    queue = ContinuousRolloutQueue(max_items=2)
    queue.put(_item(group_key=0, version=1))
    queue.put(_item(group_key=1, version=1))
    evicted = queue.put(_item(group_key=2, version=1))
    assert queue.size() == 2
    assert [item.group_key for item in evicted] == [0]
    # Oldest group (slot 0) evicted.
    remaining = {item.group_key for item in queue._items}
    assert remaining == {1, 2}


def test_byte_cap_backpressure() -> None:
    """Checks byte cap backpressure."""
    queue = ContinuousRolloutQueue(max_items=100, max_bytes=10)
    queue.put(_item(group_key=0, version=1, nbytes=6))
    evicted = queue.put(_item(group_key=1, version=1, nbytes=6))
    assert queue.ready_bytes() <= 10
    assert [item.group_key for item in evicted] == [0]


def test_batch_byte_estimate_counts_only_trainer_transport_tensors() -> None:
    batch = _item(group_key=0, version=1).batch
    batch.extras["component"] = torch.zeros(3, dtype=torch.float64)

    expected = sum(
        tensor.element_size() * tensor.nelement()
        for tensor in (
            batch.observations,
            batch.actions,
            batch.rewards,
            batch.group_ids,
            batch.extras["component"],
        )
    )

    assert estimate_batch_bytes(batch) == expected


def test_stats_shape() -> None:
    """Checks stats shape."""
    queue = ContinuousRolloutQueue(max_items=8)
    queue.put(_item(group_key=0, version=1, nbytes=4))
    stats = queue.stats()
    assert stats["ready_items"] == 1.0
    assert stats["ready_groups"] == 1.0
    assert stats["ready_bytes"] == 4.0


def test_ready_group_count_spans_policy_versions() -> None:
    """Queue stats count group slots; version selection belongs to the scheduler."""
    queue = ContinuousRolloutQueue(max_items=8)
    queue.put(_item(group_key=0, version=1))
    queue.put(_item(group_key=0, version=2))
    queue.put(_item(group_key=1, version=2))

    assert queue.distinct_group_count() == 2
