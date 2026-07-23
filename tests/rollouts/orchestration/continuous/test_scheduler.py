"""Admission and policy-version decisions owned by RolloutScheduler.

Admission covers the two limits reachable by one finite prompt batch: live
in-flight collection and bytes already held by the ready queue. Version checks
reject stale/future work and select homogeneous iterations.

Zero-bound cases below exercise the scheduler mechanism only. Production
continuous config requires a window of at least one policy version; a zero
window selects the strict-on-policy schedule instead.
"""

from __future__ import annotations

import pytest
import torch

from vrl.rollouts.batch import RolloutBatch
from vrl.rollouts.orchestration.continuous.queue import ContinuousRolloutQueue
from vrl.rollouts.orchestration.continuous.scheduler import RolloutScheduler
from vrl.rollouts.orchestration.continuous.staleness import StalenessPolicy
from vrl.rollouts.orchestration.continuous.types import ContinuousRolloutItem


def _scheduler(
    *,
    max_inflight_groups: int = 4,
    max_bytes: int = 0,
    max_stale: int = 0,
) -> RolloutScheduler:
    return RolloutScheduler(
        staleness=StalenessPolicy(max_stale_policy_versions=max_stale),
        max_inflight_groups=max_inflight_groups,
        max_bytes=max_bytes,
    )


def _admit(sched: RolloutScheduler, inflight: int, ready_bytes: int = 0):
    return sched.can_admit(
        inflight_count=inflight,
        ready_bytes=ready_bytes,
    )


def _item(
    group_key: int,
    version: int | None,
) -> ContinuousRolloutItem:
    batch = RolloutBatch(
        rewards=torch.zeros(2),
        group_ids=torch.zeros(2, dtype=torch.long),
    )
    return ContinuousRolloutItem(
        group_key=group_key,
        rollout_policy_version=version,
        batch=batch,
    )


# ----------------------------------------------------------- construction


def test_rejects_nonpositive_inflight_budget() -> None:
    with pytest.raises(ValueError):
        _scheduler(max_inflight_groups=0)


# ------------------------------------------------------------- inflight cap


def test_inflight_cap_is_the_binding_reason() -> None:
    sched = _scheduler(max_inflight_groups=2)
    assert _admit(sched, inflight=1).admit is True
    blocked = _admit(sched, inflight=2)
    assert blocked.admit is False
    assert blocked.reason == "inflight_full"


# -------------------------------------------------------------- byte budget


def test_byte_budget_is_visible_to_admission() -> None:
    sched = _scheduler(max_inflight_groups=8, max_bytes=1000)
    assert _admit(sched, inflight=0, ready_bytes=999).admit is True
    blocked = _admit(sched, inflight=0, ready_bytes=1000)
    assert blocked.admit is False
    assert blocked.reason == "byte_budget_full"


def test_byte_budget_disabled_when_zero() -> None:
    sched = _scheduler(max_bytes=0, max_stale=100)
    # Huge byte count never blocks when the cap is disabled.
    assert _admit(sched, inflight=0, ready_bytes=10**12).admit is True


# ------------------------------------------------------ precedence of reasons


def test_inflight_cap_checked_before_byte_budget() -> None:
    sched = _scheduler(max_inflight_groups=1, max_bytes=1)
    assert _admit(sched, inflight=1, ready_bytes=1).reason == "inflight_full"


# -------------------------------------------- version checks / select on queue
# These pin the version decisions the scheduler took over from the queue, which
# is now a pure container.


def test_select_drains_distinct_groups_at_one_version() -> None:
    queue = ContinuousRolloutQueue(max_items=8)
    queue.put(_item(group_key=0, version=1))
    queue.put(_item(group_key=1, version=1))

    selected = _scheduler(max_stale=0).select_iteration(
        queue,
        min_groups=2,
        current_version=1,
    )
    assert selected is not None
    version, items = selected
    assert version == 1
    assert sorted(item.group_key for item in items) == [0, 1]
    assert queue.size() == 0


def test_select_waits_until_full_set_present() -> None:
    queue = ContinuousRolloutQueue(max_items=8)
    queue.put(_item(group_key=0, version=1))
    # Only one of two required groups present yet.
    assert (
        _scheduler(max_stale=0).select_iteration(
            queue,
            min_groups=2,
            current_version=1,
        )
        is None
    )


def test_select_rejects_nonpositive_group_count() -> None:
    with pytest.raises(ValueError, match="min_groups"):
        _scheduler().select_iteration(
            ContinuousRolloutQueue(max_items=1),
            min_groups=0,
            current_version=1,
        )


def test_select_rejects_duplicate_group_slots() -> None:
    queue = ContinuousRolloutQueue(max_items=8)
    queue.put(_item(group_key=0, version=1))
    queue.put(_item(group_key=0, version=1))

    with pytest.raises(RuntimeError, match="duplicate group slots"):
        _scheduler(max_stale=0).select_iteration(
            queue,
            min_groups=2,
            current_version=1,
        )


def test_select_rejects_mixed_policy_versions() -> None:
    queue = ContinuousRolloutQueue(max_items=8)
    queue.put(_item(group_key=0, version=1))
    queue.put(_item(group_key=1, version=2))
    with pytest.raises(RuntimeError, match="mixes policy versions"):
        _scheduler(max_stale=1).select_iteration(
            queue,
            min_groups=2,
            current_version=2,
        )


def test_validate_ready_versions_rejects_stale_without_removing_it() -> None:
    queue = ContinuousRolloutQueue(max_items=8)
    queue.put(_item(group_key=0, version=1))
    queue.put(_item(group_key=1, version=3))
    with pytest.raises(RuntimeError, match="older than the policy window"):
        _scheduler(max_stale=0).validate_ready_versions(queue, current_version=3)
    assert queue.size() == 2


def test_future_item_fails_fast() -> None:
    queue = ContinuousRolloutQueue(max_items=8)
    queue.put(_item(group_key=0, version=5))
    with pytest.raises(RuntimeError, match="newer than the trainer policy"):
        _scheduler(max_stale=0).validate_ready_versions(queue, current_version=4)


def test_receipt_staleness_matches_window() -> None:
    sched = _scheduler(max_stale=1)
    assert sched.is_stale_at_receipt(3, 4) is False  # staleness 1 <= 1
    assert sched.is_stale_at_receipt(2, 4) is True  # staleness 2 > 1
    assert sched.is_stale_at_receipt(None, 4) is False
    assert sched.is_stale_at_receipt(5, 4) is False  # future -> consumer fail-fast
