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
    submitted_at: float = 0.0
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

    @property
    def phase_times(self) -> dict[str, float]:
        """Read-only back-compat view of this item's phase timings."""

        return self.stats.as_phase_dict()


@dataclass(slots=True)
class ContinuousRolloutProducerState:
    """Observable state of the background producer for metrics/health."""

    running: bool = False
    paused_for_weight_sync: bool = False
    inflight_count: int = 0
    tick_count: int = 0
    submitted_count: int = 0
    completed_count: int = 0
    last_tick_gap_s: float = 0.0
    max_tick_gap_s: float = 0.0
    error_count: int = 0
    last_error: str | None = None


__all__ = [
    "ContinuousRolloutItem",
    "ContinuousRolloutProducerState",
    "estimate_batch_bytes",
]
