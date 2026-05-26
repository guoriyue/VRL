"""Tests for the bounded continuous rollout ready queue."""

from __future__ import annotations

import pytest
import torch

from vrl.rollouts.batch import RolloutBatch
from vrl.rollouts.orchestration.continuous.queue import ContinuousRolloutQueue
from vrl.rollouts.orchestration.continuous.staleness import StalenessPolicy
from vrl.rollouts.orchestration.continuous.types import ContinuousRolloutItem


def _item(
    item_id: int,
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
        dones=torch.ones(samples, dtype=torch.bool),
        group_ids=torch.zeros(samples, dtype=torch.long),
        prompts=[f"p{group_key}"] * samples,
    )
    return ContinuousRolloutItem(
        item_id=item_id,
        group_key=group_key,
        rollout_policy_version=version,
        batch=batch,
        nbytes=nbytes,
    )


def _staleness(max_stale: int = 0) -> StalenessPolicy:
    return StalenessPolicy(max_stale_policy_versions=max_stale)


def test_select_drains_distinct_groups_at_one_version() -> None:
    queue = ContinuousRolloutQueue(max_items=8)
    queue.put(_item(0, group_key=0, version=1))
    queue.put(_item(1, group_key=1, version=1))

    selected = queue.select_iteration(
        min_groups=2,
        current_version=1,
        staleness=_staleness(0),
    )
    assert selected is not None
    version, items = selected
    assert version == 1
    assert sorted(item.group_key for item in items) == [0, 1]
    assert queue.size() == 0


def test_select_waits_until_full_set_present() -> None:
    queue = ContinuousRolloutQueue(max_items=8)
    queue.put(_item(0, group_key=0, version=1))
    # Only one of two required groups present yet.
    assert (
        queue.select_iteration(min_groups=2, current_version=1, staleness=_staleness(0))
        is None
    )


def test_select_never_mixes_policy_versions() -> None:
    queue = ContinuousRolloutQueue(max_items=8)
    # One group at v1, one at v2; max_stale=1 admits both versions but an
    # iteration must stay homogeneous, so only the newest full-enough version
    # (v2, with a single group) cannot fill min_groups=2 -> None.
    queue.put(_item(0, group_key=0, version=1))
    queue.put(_item(1, group_key=1, version=2))
    assert (
        queue.select_iteration(min_groups=2, current_version=2, staleness=_staleness(1))
        is None
    )

    # Add the second v2 group; now v2 alone fills the iteration.
    queue.put(_item(2, group_key=0, version=2))
    selected = queue.select_iteration(
        min_groups=2,
        current_version=2,
        staleness=_staleness(1),
    )
    assert selected is not None
    version, items = selected
    assert version == 2
    assert {item.rollout_policy_version for item in items} == {2}


def test_drop_too_stale_counts_and_removes() -> None:
    queue = ContinuousRolloutQueue(max_items=8)
    queue.put(_item(0, group_key=0, version=1))
    queue.put(_item(1, group_key=1, version=3))
    dropped = queue.drop_too_stale(current_version=3, staleness=_staleness(0))
    assert dropped == 1
    assert queue.size() == 1
    assert queue.dropped_stale == 1


def test_future_item_fails_fast() -> None:
    queue = ContinuousRolloutQueue(max_items=8)
    queue.put(_item(0, group_key=0, version=5))
    with pytest.raises(RuntimeError, match="newer than the trainer policy"):
        queue.drop_too_stale(current_version=4, staleness=_staleness(0))


def test_item_count_backpressure_drops_oldest() -> None:
    queue = ContinuousRolloutQueue(max_items=2, drop_policy="drop_oldest")
    queue.put(_item(0, group_key=0, version=1))
    queue.put(_item(1, group_key=1, version=1))
    queue.put(_item(2, group_key=2, version=1))
    assert queue.size() == 2
    assert queue.dropped_backpressure == 1
    # Oldest (item 0) evicted.
    remaining = {item.item_id for item in queue._items}
    assert remaining == {1, 2}


def test_byte_cap_backpressure() -> None:
    queue = ContinuousRolloutQueue(max_items=100, max_bytes=10, drop_policy="drop_oldest")
    queue.put(_item(0, group_key=0, version=1, nbytes=6))
    queue.put(_item(1, group_key=1, version=1, nbytes=6))  # 12 > 10 -> drop one
    assert queue.ready_bytes() <= 10
    assert queue.dropped_backpressure == 1


def test_stats_shape() -> None:
    queue = ContinuousRolloutQueue(max_items=8)
    queue.put(_item(0, group_key=0, version=1, nbytes=4))
    stats = queue.stats()
    assert stats["ready_items"] == 1.0
    assert stats["ready_groups"] == 1.0
    assert stats["ready_bytes"] == 4.0
