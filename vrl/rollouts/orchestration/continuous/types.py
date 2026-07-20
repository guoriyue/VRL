"""Types for the continuous rollout producer/queue/consumer."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import torch

from vrl.rollouts.batch import RolloutBatch
from vrl.utils.stats import RolloutStats


@dataclass(frozen=True, slots=True)
class ContinuousRolloutSettings:
    """The continuous rollout tuning that threads from config down to the runtime.

    One object carries the seven settings through ``build_rollout_schedule`` ->
    ``ContinuousRolloutSchedule`` -> ``ContinuousRolloutOwner`` ->
    ``_ContinuousOwnerRuntime`` so adding a knob touches one field here, not four
    repeated signatures. Deliberately has NO defaults: ``ContinuousRolloutConfig``
    (``vrl.trainers.core.types``) remains the single source of default values, and
    the rollout layer must not keep a second copy of them.

    ``max_stale_policy_versions >= 1`` is validated here so both the build factory
    and the schedule inherit one check instead of repeating it. A zero-version
    window is serial strict-on-policy execution, which is a different schedule.
    """

    max_inflight_groups: int
    max_ready_groups: int
    max_ready_bytes_mb: int
    max_stale_policy_versions: int
    wait_timeout_s: float
    queue_poll_interval_s: float
    fail_fast_errors: int

    def __post_init__(self) -> None:
        if int(self.max_stale_policy_versions) < 1:
            raise ValueError(
                "continuous rollout requires max_stale_policy_versions >= 1; "
                "use strict_on_policy for a zero-staleness serial run",
            )


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
    prompt string) so that a prompt batch with duplicate strings still yields
    ``len(prompts)`` distinct groups per iteration.
    """

    group_key: int
    rollout_policy_version: int | None
    batch: RolloutBatch
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
    # display/provenance-only: live task count included in health diagnostics.
    inflight_count: int = 0
    # display/provenance-only: owner-loop cadence health exported as metrics.
    tick_count: int = 0
    # display/provenance-only: cumulative attempts exported as diagnostics.
    submitted_count: int = 0
    completed_count: int = 0
    # display/provenance-only: cadence gaps exported as starvation diagnostics.
    last_tick_gap_s: float = 0.0
    max_tick_gap_s: float = 0.0
    error_count: int = 0
    # display/provenance-only: most recent retry cause included in wait failures.
    last_error: str | None = None
    # Behavior-consumed terminal failure from the producer control loop itself.
    # Per-slot collect failures remain retryable counters; this field means
    # cadence has stopped and the consumer must fail immediately.
    fatal_error: BaseException | None = None


__all__ = [
    "ContinuousRolloutItem",
    "ContinuousRolloutProducerState",
    "ContinuousRolloutSettings",
    "estimate_batch_bytes",
]
