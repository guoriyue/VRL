"""Types for the continuous rollout producer/queue/consumer."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from vrl.rollouts.batch import RolloutBatch
from vrl.rollouts.stats import RolloutStats
from vrl.trajectory import trajectory_tensor_bytes


@dataclass(frozen=True, slots=True)
class ContinuousRolloutSettings:
    """The continuous rollout tuning that threads from config down to the runtime.

    One object carries the six settings through ``build_rollout_schedule`` ->
    ``ContinuousRolloutSchedule`` -> ``ContinuousRolloutOwner`` ->
    ``_ContinuousOwnerRuntime`` so adding a knob touches one field here, not four
    repeated signatures. Deliberately has NO defaults: ``ContinuousRolloutConfig``
    (``vrl.trainers.core.types``) remains the single source of default values, and
    the rollout layer must not keep a second copy of them.

    Range validation lives once at the user boundary
    (``ContinuousRolloutConfig.__post_init__``); this carrier and the mechanisms
    reading it trust those values. The one check kept here is schedule
    *routing*, not a range: a zero-version window is serial strict-on-policy
    execution, which is a different schedule, and this object is the last
    shared point where both the build factory and direct constructors can be
    steered there.
    """

    max_inflight_groups: int
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

    The trajectory owns replay tensors. Shared tensor objects are deduplicated
    so queue capacity accounting does not charge the same storage twice. Arbitrary
    nested extras stay outside this cheap heuristic.
    """

    payload: dict[str, Any] = {
        "rewards": batch.rewards,
        "group_ids": batch.group_ids,
        # extras whole: trajectory_tensor_bytes recurses mappings, so nested
        # tensor payloads (reward_components) are counted. A top-level-only
        # tensor filter silently undercounted exactly what extras carries.
        "extras": batch.extras,
    }
    if batch.trajectory is not None:
        payload["trajectory"] = batch.trajectory
    return trajectory_tensor_bytes(payload)


@dataclass(slots=True)
class ContinuousRolloutItem:
    """One completed prompt group waiting in the ready queue.

    ``batch_id + group_slot`` is the logical work identity. ``group_slot`` is
    the prompt's slot index in the batch's stable prompt list (not the prompt
    string) so that a prompt batch with duplicate strings still yields
    ``len(prompts)`` distinct groups per iteration.
    """

    # Producer-assigned monotonic id of the finite prompt batch this group
    # belongs to (unique per producer lifetime, not across owner resets). The
    # consumer validates iteration homogeneity on it; batch selection consumes
    # it once the versioned lookahead lands (Sprint 2). Until then the
    # single-construction-site architecture test is its validation consumer.
    batch_id: int
    group_slot: int
    rollout_policy_version: int | None
    # 1-based attempt that produced this item. A retry of the same logical
    # work increments it without changing sample seeds or group mapping.
    # display/provenance-only in this sprint: the item is the only carrier
    # (producer failure_counts reset when the next batch installs), exported
    # as the iteration's continuous.max_attempt gauge so retried work is
    # attributable per update. The reward pump's retry identity consumes it
    # behaviorally from Sprint 1 on.
    attempt: int
    batch: RolloutBatch
    # display/provenance-only: receipt time on this process's monotonic clock
    # (never wall time), exported as a queue-health age gauge. Cross-process
    # consumers must treat it as an age/duration source, not a timestamp.
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
    "ContinuousRolloutItem",
    "ContinuousRolloutProducerState",
    "ContinuousRolloutSettings",
    "estimate_batch_bytes",
]
