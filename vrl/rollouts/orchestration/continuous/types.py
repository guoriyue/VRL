"""Types for the continuous rollout producer/queue/consumer."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import torch

from vrl.rollouts.batch import RolloutBatch
from vrl.utils.stats import RolloutStats


def estimate_batch_bytes(batch: RolloutBatch) -> int:
    """Rough host-memory footprint of a queued ``RolloutBatch``.

    Used only for the queue byte-cap backpressure heuristic, so an approximate
    walk over the obvious tensor fields is enough; we deliberately avoid a deep
    traversal of arbitrary ``extras`` to keep ``put`` cheap.
    """

    total = 0
    candidates: list[Any] = [
        batch.observations,
        batch.actions,
        batch.rewards,
        batch.dones,
        batch.group_ids,
        batch.videos,
    ]
    for value in candidates:
        if isinstance(value, torch.Tensor):
            total += value.element_size() * value.nelement()
    for value in batch.extras.values():
        if isinstance(value, torch.Tensor):
            total += value.element_size() * value.nelement()
    return int(total)


@dataclass(slots=True)
class ContinuousRolloutItem:
    """One completed prompt group waiting in the ready queue.

    ``group_key`` is the prompt's slot index in the stable prompt list (not the
    prompt string) so that a prompt set with duplicate strings still yields
    ``len(prompts)`` distinct groups per iteration.
    """

    item_id: int
    group_key: int
    rollout_policy_version: int | None
    batch: RolloutBatch
    # Generation of the prompt set this group was produced for, captured at
    # submit time. ``group_key`` is only a slot index, so two different prompt
    # sets both number their groups 0..n-1; without this id a prompt swap would
    # let an iteration mix (or train) the previous set's ready items. The
    # scheduler selects only items matching the current prompt set.
    prompt_set_id: int = 0
    completed_at: float = field(default_factory=time.time)
    nbytes: int = 0
    # Per-item typed stats (collect.engine_generate / reward_score / batch_build
    # timings + reward-inference timings) owned by the producing collect call.
    # The consumer merges them into the iteration's stats; per-item ownership
    # keeps concurrent collects from overwriting one shared accumulator.
    stats: RolloutStats = field(default_factory=RolloutStats)

    @property
    def age_s(self) -> float:
        return max(0.0, time.time() - self.completed_at)


@dataclass(slots=True)
class ContinuousRolloutProducerState:
    """Observable state of the background producer for metrics/health."""

    running: bool = False
    paused_for_weight_sync: bool = False
    inflight_count: int = 0
    tick_count: int = 0
    submitted_count: int = 0
    completed_count: int = 0
    # Groups that finished generation but were already past the staleness window
    # by receipt time (current_version moved during their generation), so the
    # producer dropped them instead of enqueuing dead work. Surfaces wasted
    # generation; the consumer would drop the same items, just later/downstream.
    discarded_stale_count: int = 0
    # Groups completed for an older prompt set after the trainer has already
    # swapped prompts. They are correct rollouts, but no longer belong to the
    # requested iteration and would otherwise block admission for the new set.
    discarded_prompt_set_count: int = 0
    # Admission observability (set by the producer from the RolloutScheduler each
    # admit pass). predicted_admit_staleness is how many versions a group
    # submitted now would trail by when consumed; admit_blocked_reason is "" when
    # admitting and otherwise the binding constraint ("inflight_full" /
    # "item_budget_full" / "byte_budget_full" / "would_land_too_stale"), so a
    # serial-looking run is diagnosable without a debugger.
    predicted_admit_staleness: int = 0
    admit_blocked_reason: str = ""
    last_tick_gap_s: float = 0.0
    max_tick_gap_s: float = 0.0
    error_count: int = 0
    last_error: str | None = None


__all__ = [
    "ContinuousRolloutItem",
    "ContinuousRolloutProducerState",
    "estimate_batch_bytes",
]
