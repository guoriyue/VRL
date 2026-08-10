"""Tests for the bounded continuous rollout ready queue (container mechanism).

Version/staleness behavior (validation and homogeneous select) lives on the
``RolloutScheduler`` now and is covered by ``test_scheduler.py``; this file pins
only the container: byte/count backpressure and stats.
"""

from __future__ import annotations

import torch

from vrl.generation import GenerationRequest, GenerationSampleRow
from vrl.rollouts.batch import RolloutBatch
from vrl.rollouts.orchestration.continuous.queue import ContinuousRolloutQueue
from vrl.rollouts.orchestration.continuous.types import (
    ContinuousRolloutItem,
    estimate_batch_bytes,
)
from vrl.trajectory import build_ar_discrete_trajectory, trajectory_tensor_bytes


def _item(
    group_key: int,
    version: int | None,
    *,
    samples: int = 2,
    nbytes: int = 0,
) -> ContinuousRolloutItem:
    batch = RolloutBatch(
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
            batch.rewards,
            batch.group_ids,
            batch.extras["component"],
        )
    )

    assert estimate_batch_bytes(batch) == expected


def test_batch_byte_estimate_counts_trajectory_without_flat_aliases_twice() -> None:
    token_ids = torch.tensor([[1, 2]], dtype=torch.long)
    prompt_input_ids = torch.tensor([[3, 4, 5]], dtype=torch.long)
    request = GenerationRequest(
        request_id="req",
        family="janus_pro",
        task="ar_t2i",
        inputs=["p"],
        samples_per_prompt=1,
    )
    trajectory = build_ar_discrete_trajectory(
        request=request,
        sample_rows=[
            GenerationSampleRow(
                prompt_index=0,
                sample_index=0,
                prompt="p",
                group_id="g0",
                sample_id="s0",
                trajectory_id="t0",
            )
        ],
        token_ids=token_ids,
        token_log_probs=torch.zeros(1, 2),
        token_mask=torch.ones(1, 2),
        prompt_input_ids=prompt_input_ids,
        prompt_attention_mask=torch.ones(1, 3, dtype=torch.long),
        uncond_input_ids=torch.zeros(1, 3, dtype=torch.long),
        uncond_attention_mask=torch.ones(1, 3, dtype=torch.long),
        context={},
    )
    rewards = torch.zeros(1)
    group_ids = torch.zeros(1, dtype=torch.long)
    component = torch.zeros(3, dtype=torch.float64)
    batch = RolloutBatch(
        rewards=rewards,
        group_ids=group_ids,
        extras={"component": component},
        trajectory=trajectory,
    )

    expected = trajectory_tensor_bytes(trajectory) + sum(
        tensor.numel() * tensor.element_size() for tensor in (rewards, group_ids, component)
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
