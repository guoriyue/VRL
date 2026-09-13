"""Types for the continuous rollout producer/queue/consumer."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from vrl.rollouts.batch import RolloutBatch
from vrl.rollouts.stats import RolloutStats


@dataclass(frozen=True, slots=True)
class ContinuousRolloutSettings:
    """The continuous rollout tuning that threads from config down to the runtime.

    One object carries the resolved settings through ``build_rollout_schedule`` ->
    ``ContinuousRolloutSchedule`` -> ``ContinuousRolloutThread`` ->
    ``_ContinuousRolloutController`` so adding a knob touches one field here, not four
    repeated signatures. Deliberately has NO defaults: ``ContinuousRolloutConfig``
    (``vrl.trainers.core.types``) remains the single source of default values.
    User settings are validated by ContinuousRolloutConfig before projection.
    This carrier does not repeat those checks.
    """

    max_inflight_groups: int
    split_generation_reward: bool
    max_unscored_groups: int
    max_unscored_bytes_mb: int
    max_generated_group_bytes_mb: int
    max_ready_bytes_mb: int
    max_stale_policy_versions: int
    wait_timeout_s: float
    queue_poll_interval_s: float
    fail_fast_errors: int


@dataclass(frozen=True, slots=True)
class ScoredRollout:
    """One completed prompt group waiting in the ready queue.

    ``batch_id + group_slot`` is the logical work identity. ``group_slot`` is
    the prompt's slot index in the batch's stable prompt list (not the prompt
    string) so that a prompt batch with duplicate strings still yields
    ``len(prompts)`` distinct groups per iteration.
    Receipt fields are fixed at completion so queue identity and charged bytes
    stay stable. The referenced batch and stats remain mutable payload objects.
    """

    # Producer-assigned monotonic identity, unique per owner lifetime. The
    # consumer selects the demanded head by this key even if the prefetched batch is ready.
    batch_id: int
    group_slot: int
    rollout_policy_version: int | None
    # 1-based collection attempt from this batch slot's failure count. Retries
    # keep the same batch_id/group_slot. The consumer exports the maximum as
    # continuous.max_attempt; this receipt field does not drive reward retries.
    attempt: int
    batch: RolloutBatch
    # display/provenance-only: receipt time on this process's monotonic clock
    # (never wall time), exported as a queue-health age gauge. Computing age
    # requires the same monotonic clock domain; this is not a portable timestamp
    # for a consumer on another machine.
    completed_at: float = field(default_factory=time.monotonic)
    nbytes: int = 0
    # Per-item typed stats (collect.engine_generate / reward_score / batch_build
    # timings + reward-inference timings) owned by the producing collect call.
    # The consumer merges them into the iteration's stats; per-item ownership
    # keeps concurrent collects from overwriting one shared accumulator.
    stats: RolloutStats = field(default_factory=RolloutStats)

    @property
    def age_s(self) -> float:
        return max(0.0, time.monotonic() - self.completed_at)


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
    # display/provenance-only: cumulative seconds the admission loop spent
    # blocked, keyed by reason. The key set derives from ``_admit()`` return
    # values plus the two loop states ("paused_for_weight_sync",
    # "no_pending_slots"); exported as continuous.backpressure_<reason>_s
    # gauges. Kept as a string-keyed map because metric reasons are a dynamic
    # namespace (same convention as RolloutStats keys), and the exporter
    # iterates instead of probing fixed keys.
    backpressure_seconds: dict[str, float] = field(default_factory=dict)
    # display/provenance-only: number of times the loop entered each blocked
    # reason (count twin of backpressure_seconds).
    backpressure_entries: dict[str, float] = field(default_factory=dict)
    # Behavior-consumed terminal failure from the producer control loop itself.
    # Per-slot collect failures remain retryable counters; this field means
    # cadence has stopped and the consumer must fail immediately.
    fatal_error: BaseException | None = None


__all__ = [
    "ContinuousRolloutProducerState",
    "ContinuousRolloutSettings",
    "ScoredRollout",
]
