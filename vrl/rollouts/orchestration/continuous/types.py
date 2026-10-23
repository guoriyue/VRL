"""Batch state and results shared by continuous rollout production and consumption."""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from vrl.rollouts.batch import RolloutBatch
from vrl.utils.stats import Stats


@dataclass(frozen=True, slots=True)
class ContinuousRolloutSettings:
    """The continuous rollout tuning that threads from config down to the runtime.

    One object carries the resolved settings through ``ContinuousRolloutSchedule.from_section``
    -> ``ContinuousRolloutThread`` ->
    ``_ContinuousRolloutController`` so adding a knob touches one field here, not four
    repeated signatures. Deliberately has NO defaults: ``ContinuousRolloutConfig``
    (``vrl.trainers.core.types``) remains the single source of default values.
    User settings are validated by ContinuousRolloutConfig before projection.
    This carrier does not repeat those checks.
    """

    max_inflight_groups: int
    max_stale_policy_versions: int
    wait_timeout_s: float
    queue_poll_interval_s: float
    fail_fast_errors: int
    # The run's decision (TrainerConfig.versioned_weight_sync): the rollout
    # workers keep each version's weights, so a sync need not drain the batch.
    versioned_weight_sync: bool


@dataclass(frozen=True, slots=True)
class ScoredRollout:
    """One completed prompt group waiting in its batch's result slot.

    Its identity is its position: ``PromptBatch.results[slot]``, where the slot
    is the prompt's index in the batch's stable prompt list (not the prompt
    string), so a prompt batch with duplicate strings still yields
    ``len(prompts)`` distinct groups per iteration. The owning ``PromptBatch``
    carries the batch id and the fixed policy version every slot ran under.
    Receipt fields are fixed at completion; the referenced batch and stats
    remain mutable payload objects.
    """

    batch: RolloutBatch
    # display/provenance-only: receipt time on this process's monotonic clock
    # (never wall time), exported as a result-age gauge. Computing age
    # requires the same monotonic clock domain; this is not a portable timestamp
    # for a consumer on another machine.
    completed_at: float = field(default_factory=time.monotonic)
    # What the producing collect call recorded (collect.engine_generate /
    # reward_score / batch_build timings + reward-inference timings + the
    # item's service gauges). Kept as data: the item outlives its collect's
    # scope on the owner thread, and the consumer absorbs it into the scope of
    # the trainer step that takes the item.
    stats: Stats = field(default_factory=Stats)

    @property
    def age_s(self) -> float:
        return max(0.0, time.monotonic() - self.completed_at)


@dataclass(slots=True)
class PromptBatch:
    """Inputs, progress, and scored results of one finite continuous batch.

    Each prompt owns one result slot. Completion order cannot change prompt
    order, and retries keep the installed policy version and collection settings.
    The producer is the only writer: it fills the slots, and its
    ``release_results`` empties them and marks the batch consumed once the
    trainer has taken the iteration, before another batch can be installed.
    """

    # Producer-assigned monotonic identity, unique per owner lifetime.
    batch_id: int
    # Fixed at install: every slot (retries included) generates at this version.
    policy_version: int
    prompts: tuple[Any, ...]
    group_size: int
    runtime_debug: bool
    pending_slots: deque[int]
    # Per-slot admission timestamps become the completed item's wait metric.
    pending_since: dict[int, float]
    results: list[ScoredRollout | None]
    failure_counts: dict[int, int] = field(default_factory=dict)
    consumed: bool = False


@dataclass(slots=True)
class ContinuousRolloutProducerState:
    """Observable state of the background producer for metrics/health."""

    running: bool = False
    paused_for_weight_sync: bool = False
    # display/provenance-only: owner-loop cadence health exported as metrics.
    tick_count: int = 0
    # display/provenance-only: cumulative attempts exported as diagnostics.
    submitted_count: int = 0
    # Behavior-consumed with error_count: the consumer's fail-fast compares
    # fresh completions against fresh errors while it waits (zero completions
    # plus a threshold of errors raises the producer's root cause), and the
    # weight-sync drain loop backs off when a harvest tick added errors.
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
    "ContinuousRolloutProducerState",
    "ContinuousRolloutSettings",
    "ScoredRollout",
]
