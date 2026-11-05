"""Batch state and results shared by continuous rollout production and consumption."""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from vrl.rollouts.batch import RolloutBatch
from vrl.utils.stats import Stats

# Policy versions a ready prompt batch may trail the trainer by. One is the
# continuous design itself: batch N+1 generates at version N while step N
# trains, and the importance ratio absorbs that single version of lag
# (SD3.5 OCR on 4x L40S, 2026-09-14: no measurable reward difference over 11
# paired epochs). A zero window is strict_on_policy; a wider one was never run.
MAX_STALE_POLICY_VERSIONS = 1
# Ready-slot poll period of the producer tick and the consumer wait: far below
# any generation (seconds) yet coarse enough to keep the owner loop idle
# between checks.
QUEUE_POLL_INTERVAL_S = 0.05
# Failures of one prompt-batch slot tolerated before the producer raises its
# root cause instead of waiting out wait_timeout_s; the consumer applies the
# same budget when a wait sees only failures and no completion. Three tells a
# systemic generation/reward failure from a transient one; 0 disables both.
FAIL_FAST_ERRORS = 3


@dataclass(frozen=True, slots=True)
class ContinuousRolloutSettings:
    """The continuous rollout tuning that threads from config down to the runtime.

    One object carries the resolved settings through ``ContinuousRolloutSchedule.from_section``
    -> ``ContinuousRolloutThread`` ->
    ``_ContinuousRolloutController`` so adding a knob touches one field here, not four
    repeated signatures. Deliberately has NO defaults: ``ContinuousRolloutConfig``
    (``vrl.trainers.core.types``) remains the single source of default values.
    User settings are validated by ContinuousRolloutConfig before projection.
    This carrier does not repeat those checks. The fixed limits above are not
    carried: the producer and consumer read them from this module at the point
    of use, so a test narrows one by patching it here.
    """

    max_inflight_groups: int
    wait_timeout_s: float
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
