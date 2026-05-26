"""Continuous rollout schedule: producer + ready queue + consumer.

Implements the ``RolloutSchedule`` protocol. ``next_iteration`` starts the
background producer on first use and then drains one homogeneous-policy
iteration from the ready queue. ``after_train_step`` runs the weight-sync
barrier (pause admission -> drain in-flight -> sync -> resume) so no generation
request straddles a policy version bump.

With ``max_stale_policy_versions=0`` and a queue sized to one iteration, this is
behavior-equivalent to ``strict_on_policy``: every iteration trains on a full,
freshly-synced prompt set. Raising ``max_stale``/queue depth turns on real
off-policy prefetch (the cross-node throughput mode).
"""

from __future__ import annotations

from typing import Any

from vrl.rollouts.orchestration.continuous.consumer import ContinuousRolloutConsumer
from vrl.rollouts.orchestration.continuous.producer import ContinuousRolloutProducer
from vrl.rollouts.orchestration.continuous.queue import ContinuousRolloutQueue
from vrl.rollouts.orchestration.continuous.staleness import StalenessPolicy
from vrl.rollouts.orchestration.lifecycle import RolloutLifecycle, record_phase
from vrl.rollouts.orchestration.types import (
    RolloutIteration,
    RolloutScheduleMode,
    RolloutScheduleState,
)

_MB = 1024 * 1024


class ContinuousRolloutSchedule:
    """Continuously produce rollouts; train from a bounded ready queue."""

    mode = RolloutScheduleMode.CONTINUOUS

    def __init__(
        self,
        *,
        lifecycle: RolloutLifecycle,
        require_separate_gpus: bool = True,
        max_inflight_groups: int = 1,
        max_ready_groups: int = 2,
        max_ready_bytes_mb: int = 8192,
        max_stale_policy_versions: int = 0,
        drop_policy: str = "drop_oldest_stale",
        wait_timeout_s: float = 300.0,
        queue_poll_interval_s: float = 0.05,
    ) -> None:
        self.lifecycle = lifecycle
        self.require_separate_gpus = bool(require_separate_gpus)
        self.max_inflight_groups = int(max_inflight_groups)
        self.max_ready_groups = int(max_ready_groups)
        self.max_ready_bytes_mb = int(max_ready_bytes_mb)
        self.drop_policy = drop_policy
        self.wait_timeout_s = float(wait_timeout_s)
        self.queue_poll_interval_s = float(queue_poll_interval_s)

        self.staleness = StalenessPolicy(
            max_stale_policy_versions=int(max_stale_policy_versions),
        )
        self.state = RolloutScheduleState()
        self.queue: ContinuousRolloutQueue | None = None
        self.consumer: ContinuousRolloutConsumer | None = None
        self.producer: ContinuousRolloutProducer | None = None
        self._prompt_key: tuple[str, ...] | None = None

    async def next_iteration(
        self,
        prompts: list[Any],
        *,
        group_size: int,
        runtime_debug: bool = False,
    ) -> RolloutIteration:
        self._validate_allowed()
        prompt_key = _prompt_key(prompts)
        phase_times: dict[str, float] = {}

        if self.producer is None:
            phase_times.update(
                await self._start(prompts, group_size=group_size, runtime_debug=runtime_debug),
            )
            self._prompt_key = prompt_key
        elif prompt_key != self._prompt_key:
            self.producer.update_prompts(prompts)
            self._prompt_key = prompt_key

        assert self.consumer is not None
        rollout_id = self.state.rollout_id
        self.state.rollout_id += 1
        current_version = self.lifecycle.current_policy_version()

        iteration = await self.consumer.drain_for_iteration(
            rollout_id=rollout_id,
            min_groups=len(prompts),
            current_version=current_version,
            mode=self.mode,
            wait_timeout_s=self.wait_timeout_s,
            poll_interval_s=self.queue_poll_interval_s,
        )
        iteration.phase_times.update(phase_times)
        self._attach_producer_metrics(iteration)
        return iteration

    async def after_train_step(self) -> dict[str, float]:
        phase_times: dict[str, float] = {}
        if self.producer is None:
            await self.lifecycle.sync_weights_after_train(phase_times)
            return phase_times
        with record_phase(phase_times, "continuous.weight_sync_pause_s"):
            self.producer.pause_admission()
            await self.producer.drain_inflight()
            await self.lifecycle.sync_weights_after_train(phase_times)
            self.producer.resume_admission()
        return phase_times

    def reset(self) -> None:
        # Synchronous, best-effort teardown: cancel the loop + in-flight tasks
        # without awaiting.
        if self.producer is not None:
            self.producer.cancel()
        if self.queue is not None:
            self.queue.close()
        self.producer = None
        self.queue = None
        self.consumer = None
        self._prompt_key = None
        self.state.initialized = False
        self.state.pending_rollout = None

    # -- internals ------------------------------------------------------

    async def _start(
        self,
        prompts: list[Any],
        *,
        group_size: int,
        runtime_debug: bool,
    ) -> dict[str, float]:
        capacity = max(self.max_ready_groups, len(prompts))
        self.queue = ContinuousRolloutQueue(
            max_items=capacity,
            max_bytes=self.max_ready_bytes_mb * _MB,
            drop_policy=self.drop_policy,
        )
        self.consumer = ContinuousRolloutConsumer(
            queue=self.queue,
            staleness=self.staleness,
        )
        self.producer = ContinuousRolloutProducer(
            lifecycle=self.lifecycle,
            prompts=list(prompts),
            queue=self.queue,
            group_size=group_size,
            capacity=capacity,
            max_inflight_groups=self.max_inflight_groups,
            poll_interval_s=self.queue_poll_interval_s,
            runtime_debug=runtime_debug,
        )
        return await self.producer.start()

    def _attach_producer_metrics(self, iteration: RolloutIteration) -> None:
        if self.producer is None or self.queue is None:
            return
        state = self.producer.state
        stats = self.queue.stats()
        version = iteration.policy_version
        metadata = iteration.metadata
        iteration.phase_times.update(
            {
                "continuous.producer_inflight": float(state.inflight_count),
                "continuous.queue_ready_items": stats["ready_items"],
                "continuous.queue_ready_groups": stats["ready_groups"],
                "continuous.queue_ready_bytes": stats["ready_bytes"],
                "continuous.item_age_s": float(metadata.get("continuous_item_age_s", 0.0)),
                "continuous.dropped_stale": stats["dropped_stale"],
                "continuous.dropped_backpressure": stats["dropped_backpressure"],
                "continuous.producer_errors": float(state.error_count),
                "continuous.consume_policy_version": float(
                    metadata.get("consume_policy_version") or 0,
                ),
                "continuous.rollout_policy_version": float(0 if version is None else version),
                "continuous.stale_policy_versions": float(
                    metadata.get("stale_policy_versions") or 0,
                ),
            },
        )

    def _validate_allowed(self) -> None:
        if self.lifecycle.requires_driver_model_offload():
            raise RuntimeError(
                "continuous rollout is disabled when rollout runtime requires "
                "driver model offload",
            )
        if self.require_separate_gpus and self.lifecycle.runtime_is_colocated():
            raise RuntimeError(
                "continuous rollout requires separate trainer and rollout GPU "
                "ownership",
            )


def _prompt_key(prompts: list[Any]) -> tuple[str, ...]:
    return tuple(str(getattr(item, "prompt", item)) for item in prompts)


__all__ = ["ContinuousRolloutSchedule"]
