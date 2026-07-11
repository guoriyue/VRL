"""Admission decisions owned by RolloutScheduler.

These pin the consolidation P2 added: one predicate decides admission across the
in-flight cap, a single item budget spanning in-flight + ready, the ready byte
budget (now visible to admission), and the admit-time predicted-version
staleness throttle. The producer/queue keep their mechanisms; this is the
decision owner.
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
    capacity: int = 16,
    max_bytes: int = 0,
    groups_per_iteration: int = 4,
    max_stale: int = 0,
) -> RolloutScheduler:
    return RolloutScheduler(
        staleness=StalenessPolicy(max_stale_policy_versions=max_stale),
        max_inflight_groups=max_inflight_groups,
        capacity=capacity,
        max_bytes=max_bytes,
        groups_per_iteration=groups_per_iteration,
    )


def _admit(sched: RolloutScheduler, inflight: int, ready: int, ready_bytes: int = 0):
    return sched.can_admit(
        inflight_count=inflight, ready_items=ready, ready_bytes=ready_bytes,
    )


def _item(
    group_key: int,
    version: int | None,
    *,
    prompt_set_id: int = 0,
) -> ContinuousRolloutItem:
    batch = RolloutBatch(
        observations=torch.zeros(2, 1),
        actions=torch.zeros(2, 1),
        rewards=torch.zeros(2),
        dones=torch.ones(2, dtype=torch.bool),
        group_ids=torch.zeros(2, dtype=torch.long),
        prompts=[f"p{group_key}"] * 2,
    )
    return ContinuousRolloutItem(
        group_key=group_key,
        rollout_policy_version=version,
        prompt_set_id=prompt_set_id,
        batch=batch,
    )


# ----------------------------------------------------------- construction


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_inflight_groups": 0},
        {"capacity": 0},
        {"groups_per_iteration": 0},
    ],
)
def test_rejects_nonpositive_budgets(kwargs: dict[str, int]) -> None:
    with pytest.raises(ValueError):
        _scheduler(**kwargs)


# ------------------------------------------------------------- inflight cap


def test_inflight_cap_is_the_binding_reason() -> None:
    # Large capacity + window so only the in-flight cap can bind.
    sched = _scheduler(max_inflight_groups=2, capacity=100, max_stale=100)
    assert _admit(sched, inflight=1, ready=0).admit is True
    blocked = _admit(sched, inflight=2, ready=0)
    assert blocked.admit is False
    assert blocked.reason == "inflight_full"


# --------------------------------------------------------- single item budget


def test_item_budget_spans_inflight_plus_ready() -> None:
    # One owner counts in-flight + ready against capacity (no two diverging
    # counters): 3 in-flight + 1 ready == capacity 4 -> full.
    sched = _scheduler(max_inflight_groups=8, capacity=4, max_stale=100)
    assert _admit(sched, inflight=2, ready=1).admit is True
    blocked = _admit(sched, inflight=3, ready=1)
    assert blocked.admit is False
    assert blocked.reason == "item_budget_full"


# -------------------------------------------------------------- byte budget


def test_byte_budget_is_visible_to_admission() -> None:
    sched = _scheduler(max_inflight_groups=8, capacity=100, max_bytes=1000, max_stale=100)
    assert _admit(sched, inflight=0, ready=1, ready_bytes=999).admit is True
    blocked = _admit(sched, inflight=0, ready=1, ready_bytes=1000)
    assert blocked.admit is False
    assert blocked.reason == "byte_budget_full"


def test_byte_budget_disabled_when_zero() -> None:
    sched = _scheduler(max_bytes=0, max_stale=100)
    # Huge byte count never blocks when the cap is disabled.
    assert _admit(sched, inflight=0, ready=1, ready_bytes=10**12).admit is True


# ----------------------------------------------- predicted-version throttle


def test_predicted_staleness_counts_full_iterations_ahead() -> None:
    sched = _scheduler(groups_per_iteration=4)
    # Less than one full iteration of backlog -> lands at the current version.
    assert sched.predicted_landing_staleness(inflight_count=1, ready_items=2) == 0
    # Exactly one iteration of backlog -> one version bump before consumption.
    assert sched.predicted_landing_staleness(inflight_count=2, ready_items=2) == 1
    # Two iterations of backlog -> two bumps.
    assert sched.predicted_landing_staleness(inflight_count=4, ready_items=4) == 2


def test_throttle_stops_admission_at_one_iteration_when_max_stale_zero() -> None:
    # max_stale=0: a group submitted once a full iteration is already pending
    # would land stale=1 and be discarded at receipt, so do not generate it.
    sched = _scheduler(
        max_inflight_groups=8, capacity=100, groups_per_iteration=4, max_stale=0,
    )
    # 3 pending: predicted 0, still admits (filling the first iteration).
    assert _admit(sched, inflight=2, ready=1).admit is True
    # 4 pending == one full iteration: predicted 1 > 0 -> throttle.
    blocked = _admit(sched, inflight=2, ready=2)
    assert blocked.admit is False
    assert blocked.reason == "would_land_too_stale"


def test_throttle_window_scales_with_max_stale() -> None:
    # max_stale=1 admits up to two iterations of lookahead, then stops.
    sched = _scheduler(
        max_inflight_groups=8, capacity=100, groups_per_iteration=4, max_stale=1,
    )
    # 7 pending: predicted 1 <= 1 -> still admits.
    assert _admit(sched, inflight=4, ready=3).admit is True
    # 8 pending == two iterations: predicted 2 > 1 -> throttle.
    assert _admit(sched, inflight=4, ready=4).reason == "would_land_too_stale"


def test_throttle_never_starves_a_full_iteration() -> None:
    # The throttle must let exactly one whole iteration accumulate before it
    # binds, so the consumer can always assemble a homogeneous iteration.
    sched = _scheduler(
        max_inflight_groups=100, capacity=100, groups_per_iteration=4, max_stale=0,
    )
    # Admission stays open for the first full iteration (0..3 pending).
    for pending in range(4):
        assert _admit(sched, inflight=0, ready=pending).admit is True
    # And closes exactly at the iteration boundary.
    assert _admit(sched, inflight=0, ready=4).admit is False


# ------------------------------------------------------ precedence of reasons


def test_inflight_cap_checked_before_budgets() -> None:
    # When several constraints bind at once, the reported reason is the first in
    # the documented order (inflight -> item -> byte -> staleness).
    sched = _scheduler(
        max_inflight_groups=1, capacity=1, groups_per_iteration=1, max_stale=0,
    )
    assert _admit(sched, inflight=1, ready=1).reason == "inflight_full"


# ----------------------------------------------------- prompt-set swap update


def test_set_groups_per_iteration_updates_prediction() -> None:
    sched = _scheduler(groups_per_iteration=4)
    assert sched.predicted_landing_staleness(inflight_count=0, ready_items=2) == 0
    sched.set_groups_per_iteration(2)
    # Same backlog is now a full iteration under the smaller prompt set.
    assert sched.predicted_landing_staleness(inflight_count=0, ready_items=2) == 1
    with pytest.raises(ValueError):
        sched.set_groups_per_iteration(0)


# ---------------------------------------------- drop-stale / select on a queue
# These pin the version decisions the scheduler took over from the queue, which
# is now a pure container.


def test_select_drains_distinct_groups_at_one_version() -> None:
    queue = ContinuousRolloutQueue(max_items=8)
    queue.put(_item(group_key=0, version=1))
    queue.put(_item(group_key=1, version=1))

    selected = _scheduler(max_stale=0).select_iteration(
        queue, min_groups=2, current_version=1,
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
            queue, min_groups=2, current_version=1,
        )
        is None
    )


def test_select_never_mixes_policy_versions() -> None:
    queue = ContinuousRolloutQueue(max_items=8)
    # v1 and v2 both in-window under max_stale=1, but an iteration must stay
    # homogeneous and v2 alone cannot yet fill min_groups=2 -> None.
    queue.put(_item(group_key=0, version=1))
    queue.put(_item(group_key=1, version=2))
    sched = _scheduler(max_stale=1)
    assert sched.select_iteration(queue, min_groups=2, current_version=2) is None

    # Add the second v2 group; now v2 alone fills the iteration.
    queue.put(_item(group_key=0, version=2))
    selected = sched.select_iteration(queue, min_groups=2, current_version=2)
    assert selected is not None
    version, items = selected
    assert version == 2
    assert {item.rollout_policy_version for item in items} == {2}


def test_drop_stale_counts_and_removes() -> None:
    queue = ContinuousRolloutQueue(max_items=8)
    queue.put(_item(group_key=0, version=1))
    queue.put(_item(group_key=1, version=3))
    dropped = _scheduler(max_stale=0).drop_stale(queue, current_version=3)
    assert dropped == 1
    assert queue.size() == 1
    assert queue.dropped_stale == 1


def test_future_item_fails_fast() -> None:
    queue = ContinuousRolloutQueue(max_items=8)
    queue.put(_item(group_key=0, version=5))
    with pytest.raises(RuntimeError, match="newer than the trainer policy"):
        _scheduler(max_stale=0).drop_stale(queue, current_version=4)


def test_select_only_serves_the_requested_prompt_set() -> None:
    # A prompt swap leaves the previous set's items in the queue at the same
    # version and slot numbering; select must never serve them for the new set.
    queue = ContinuousRolloutQueue(max_items=8)
    queue.put(_item(group_key=0, version=1, prompt_set_id=0))
    queue.put(_item(group_key=1, version=1, prompt_set_id=0))
    sched = _scheduler(max_stale=0)

    # The new prompt set (id=1) has no items yet -> nothing to train, even though
    # a full version-1 iteration of the OLD set is sitting in the queue.
    assert (
        sched.select_iteration(queue, min_groups=2, current_version=1, prompt_set_id=1)
        is None
    )
    # The old set's items are untouched (not consumed, not mixed).
    assert queue.size() == 2

    # Once the new set's groups arrive, they fill the iteration; the old set's
    # items stay behind.
    queue.put(_item(group_key=0, version=1, prompt_set_id=1))
    queue.put(_item(group_key=1, version=1, prompt_set_id=1))
    selected = sched.select_iteration(
        queue, min_groups=2, current_version=1, prompt_set_id=1,
    )
    assert selected is not None
    _version, items = selected
    assert {item.prompt_set_id for item in items} == {1}
    assert {item.group_key for item in items} == {0, 1}
    assert queue.size() == 2  # the two old-set items remain


def test_drop_obsolete_prompt_sets_counts_and_removes() -> None:
    queue = ContinuousRolloutQueue(max_items=8)
    queue.put(_item(group_key=0, version=1, prompt_set_id=0))
    queue.put(_item(group_key=1, version=1, prompt_set_id=1))

    dropped = _scheduler(max_stale=1).drop_obsolete_prompt_sets(
        queue,
        prompt_set_id=1,
    )

    assert dropped == 1
    assert queue.size() == 1
    assert queue.snapshot()[0].prompt_set_id == 1
    assert queue.stats()["dropped_prompt_set"] == 1.0


def test_discard_at_receipt_matches_window() -> None:
    sched = _scheduler(max_stale=1)
    # In-window and absent/future versions are kept; only past-window drops.
    assert sched.discard_at_receipt(3, 4) is False  # staleness 1 <= 1
    assert sched.discard_at_receipt(2, 4) is True  # staleness 2 > 1
    assert sched.discard_at_receipt(None, 4) is False
    assert sched.discard_at_receipt(5, 4) is False  # future -> consumer fail-fast


def test_discard_prompt_set_at_receipt_matches_current_set() -> None:
    sched = _scheduler(max_stale=1)
    assert sched.discard_prompt_set_at_receipt(0, 1) is True
    assert sched.discard_prompt_set_at_receipt(1, 1) is False
