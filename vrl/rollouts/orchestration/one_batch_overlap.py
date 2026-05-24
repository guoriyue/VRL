"""One-batch-overlap rollout schedule."""

from __future__ import annotations

import asyncio
from typing import Any

from vrl.rollouts.orchestration.lifecycle import RolloutLifecycle, record_phase
from vrl.rollouts.orchestration.prompt_collection import collect_prompt_batches
from vrl.rollouts.orchestration.types import (
    RolloutIteration,
    RolloutScheduleMode,
    RolloutScheduleState,
    build_rollout_iteration,
)


class OneBatchOverlapRolloutSchedule:
    """Run rollout N+1 while the trainer consumes rollout N."""

    mode = RolloutScheduleMode.ONE_BATCH_OVERLAP

    def __init__(
        self,
        *,
        lifecycle: RolloutLifecycle,
        require_separate_gpus: bool = True,
    ) -> None:
        self.lifecycle = lifecycle
        self.require_separate_gpus = bool(require_separate_gpus)
        self.state = RolloutScheduleState()
        self._pending: asyncio.Task[RolloutIteration] | None = None
        self._ready: RolloutIteration | None = None
        self._pending_prompt_key: tuple[str, ...] | None = None
        self._ready_prompt_key: tuple[str, ...] | None = None

    async def next_iteration(
        self,
        prompts: list[Any],
        *,
        group_size: int,
        runtime_debug: bool = False,
    ) -> RolloutIteration:
        self._validate_overlap_allowed()
        prompt_key = _prompt_key(prompts)

        if self._ready is not None:
            ready = self._pop_ready(prompt_key)
            self._start_pending(
                prompts,
                group_size=group_size,
                runtime_debug=False,
            )
            await asyncio.sleep(0)
            return ready

        if self._pending is None:
            self._start_pending(
                prompts,
                group_size=group_size,
                runtime_debug=runtime_debug,
            )

        self._require_same_pending_prompts(prompt_key)
        phase_times: dict[str, float] = {}
        with record_phase(phase_times, "rollout.schedule_wait_s"):
            current = await self._require_pending()
        current.phase_times.update(phase_times)
        self._pending = None
        self._pending_prompt_key = None

        self._start_pending(
            prompts,
            group_size=group_size,
            runtime_debug=False,
        )
        await asyncio.sleep(0)
        return current

    async def after_train_step(self) -> dict[str, float]:
        phase_times: dict[str, float] = {}
        if self.lifecycle.weight_syncer is not None and self._pending is not None:
            with record_phase(phase_times, "rollout.schedule_wait_s"):
                self._ready = await self._pending
            self._ready_prompt_key = self._pending_prompt_key
            self._pending = None
            self._pending_prompt_key = None
        await self.lifecycle.sync_weights_after_train(phase_times)
        return phase_times

    def reset(self) -> None:
        if self._pending is not None and not self._pending.done():
            self._pending.cancel()
        self._pending = None
        self._ready = None
        self._pending_prompt_key = None
        self._ready_prompt_key = None
        self.state.initialized = False
        self.state.pending_rollout = None

    def _start_pending(
        self,
        prompts: list[Any],
        *,
        group_size: int,
        runtime_debug: bool,
    ) -> None:
        if self._pending is not None:
            raise RuntimeError("one_batch_overlap supports only one pending rollout")
        rollout_id = self.state.rollout_id
        self.state.rollout_id += 1
        prompt_snapshot = list(prompts)
        self._pending_prompt_key = _prompt_key(prompt_snapshot)
        self._pending = asyncio.create_task(
            self._collect_iteration(
                rollout_id=rollout_id,
                prompts=prompt_snapshot,
                group_size=group_size,
                runtime_debug=runtime_debug,
            ),
        )
        self.state.pending_rollout = self._pending

    async def _collect_iteration(
        self,
        *,
        rollout_id: int,
        prompts: list[Any],
        group_size: int,
        runtime_debug: bool,
    ) -> RolloutIteration:
        phase_times: dict[str, float] = {}
        await self.lifecycle.ensure_initial_weights(phase_times)
        policy_version = self.lifecycle.current_policy_version()
        with record_phase(phase_times, "rollout.collect_s"):
            batches = await collect_prompt_batches(
                collector=self.lifecycle.collector,
                prompts=prompts,
                group_size=group_size,
                runtime_debug=runtime_debug,
                policy_version=policy_version,
            )
        return build_rollout_iteration(
            rollout_id=rollout_id,
            policy_version=policy_version,
            mode=self.mode,
            batches=batches,
            prompt_count=len(prompts),
            phase_times=phase_times,
        )

    async def _require_pending(self) -> RolloutIteration:
        if self._pending is None:
            raise RuntimeError("one_batch_overlap has no pending rollout")
        return await self._pending

    def _pop_ready(self, prompt_key: tuple[str, ...]) -> RolloutIteration:
        ready = self._ready
        if ready is None:
            raise RuntimeError("one_batch_overlap has no ready rollout")
        if self._ready_prompt_key is not None and prompt_key != self._ready_prompt_key:
            raise ValueError(
                "one_batch_overlap requires stable prompts while a completed "
                "rollout is waiting for training",
            )
        self._ready = None
        self._ready_prompt_key = None
        return ready

    def _require_same_pending_prompts(self, prompt_key: tuple[str, ...]) -> None:
        if self._pending_prompt_key is None:
            return
        if prompt_key != self._pending_prompt_key:
            raise ValueError(
                "one_batch_overlap requires stable prompts while a rollout is "
                "already in flight",
            )

    def _validate_overlap_allowed(self) -> None:
        if self.lifecycle.requires_driver_model_offload():
            raise RuntimeError(
                "one_batch_overlap is disabled when rollout runtime requires "
                "driver model offload",
            )
        if self.require_separate_gpus and self.lifecycle.runtime_is_colocated():
            raise RuntimeError(
                "one_batch_overlap requires separate trainer and rollout GPU ownership",
            )


def _prompt_key(prompts: list[Any]) -> tuple[str, ...]:
    return tuple(str(getattr(item, "prompt", item)) for item in prompts)


__all__ = ["OneBatchOverlapRolloutSchedule"]
