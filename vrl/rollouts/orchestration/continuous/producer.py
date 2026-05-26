"""In-process continuous rollout producer.

A background ``asyncio`` task keeps a bounded number of ``RolloutCollector``
collect jobs in flight, stamps each completed group with the policy version it
was generated under, and pushes it onto the ready queue. The heavy generation
work is dispatched by the collector (e.g. to remote Ray generation actors), so
this loop only schedules and harvests — that is enough to overlap rollout with
training on a cross-node setup.

The producer never computes advantages, calls the evaluator/algorithm, or
touches the optimizer; it owns rollout *production cadence* only.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import torch

from vrl.rollouts.batch.ops import move_training_batch_to_device
from vrl.rollouts.orchestration.continuous.queue import ContinuousRolloutQueue
from vrl.rollouts.orchestration.continuous.types import (
    ContinuousRolloutItem,
    ContinuousRolloutProducerState,
    estimate_batch_bytes,
)
from vrl.rollouts.orchestration.lifecycle import RolloutLifecycle
from vrl.rollouts.orchestration.prompt_collection import collect_prompt_batches

_CPU = torch.device("cpu")


class ContinuousRolloutProducer:
    """Background producer feeding the continuous ready queue."""

    def __init__(
        self,
        *,
        lifecycle: RolloutLifecycle,
        prompts: list[Any],
        queue: ContinuousRolloutQueue,
        group_size: int,
        capacity: int,
        max_inflight_groups: int,
        poll_interval_s: float,
        runtime_debug: bool = False,
    ) -> None:
        self.lifecycle = lifecycle
        self.prompts = list(prompts)
        if not self.prompts:
            raise ValueError(
                "ContinuousRolloutProducer requires a non-empty prompt list",
            )
        self.queue = queue
        self.group_size = int(group_size)
        self.capacity = int(capacity)
        self.max_inflight_groups = max(1, int(max_inflight_groups))
        self.poll_interval_s = float(poll_interval_s)
        self.runtime_debug = bool(runtime_debug)

        self.state = ContinuousRolloutProducerState()
        self._loop_task: asyncio.Task[None] | None = None
        self._inflight: set[asyncio.Task[Any]] = set()
        self._item_counter = 0
        self._prompt_cursor = 0

    # -- lifecycle ------------------------------------------------------

    async def start(self) -> dict[str, float]:
        phase_times: dict[str, float] = {}
        await self.lifecycle.ensure_initial_weights(phase_times)
        self.state.running = True
        self._loop_task = asyncio.create_task(self._run())
        return phase_times

    async def stop(self) -> None:
        self.cancel()
        if self._loop_task is not None and not self._loop_task.done():
            await asyncio.gather(self._loop_task, return_exceptions=True)
        if self._inflight:
            await asyncio.gather(*self._inflight, return_exceptions=True)
        self._loop_task = None
        self._inflight.clear()
        self.state.inflight_count = 0

    def cancel(self) -> None:
        """Best-effort synchronous teardown (no await); see ``stop`` to drain."""

        self.state.running = False
        if self._loop_task is not None and not self._loop_task.done():
            self._loop_task.cancel()
        for task in list(self._inflight):
            task.cancel()
        self.state.inflight_count = 0

    # -- weight-sync barrier -------------------------------------------

    def pause_admission(self) -> None:
        self.state.paused_for_weight_sync = True

    async def drain_inflight(self) -> None:
        """Let every in-flight collect finish, then harvest it into the queue.

        Mutating worker weights while a generation request is running could mix
        two policies inside one request, so the barrier always drains rather
        than cancels.
        """

        if self._inflight:
            await asyncio.gather(*self._inflight, return_exceptions=True)
            self._harvest_done()

    def resume_admission(self) -> None:
        self.state.paused_for_weight_sync = False

    # -- main loop ------------------------------------------------------

    async def _run(self) -> None:
        try:
            while self.state.running:
                if self.state.paused_for_weight_sync:
                    await asyncio.sleep(self.poll_interval_s)
                    continue
                self._admit()
                self._harvest_done()
                await asyncio.sleep(self.poll_interval_s)
        except asyncio.CancelledError:  # pragma: no cover - cooperative shutdown
            raise

    def _admit(self) -> None:
        while (
            len(self._inflight) < self.max_inflight_groups
            and self.queue.size() + len(self._inflight) < self.capacity
        ):
            slot, prompt = self._next_prompt_group()
            self._submit(slot, prompt)

    def _next_prompt_group(self) -> tuple[int, Any]:
        slot = self._prompt_cursor % len(self.prompts)
        prompt = self.prompts[slot]
        self._prompt_cursor += 1
        return slot, prompt

    def _submit(self, slot: int, prompt: Any) -> None:
        version = self.lifecycle.current_policy_version()
        submitted_at = time.time()
        task = asyncio.create_task(
            self._collect_group(
                slot=slot,
                prompt=prompt,
                version=version,
                submitted_at=submitted_at,
            ),
        )
        self._inflight.add(task)
        self.state.inflight_count = len(self._inflight)

    async def _collect_group(
        self,
        *,
        slot: int,
        prompt: Any,
        version: int | None,
        submitted_at: float,
    ) -> dict[str, Any]:
        batches = await collect_prompt_batches(
            collector=self.lifecycle.collector,
            prompts=[prompt],
            group_size=self.group_size,
            runtime_debug=self.runtime_debug,
            policy_version=version,
        )
        return {
            "slot": slot,
            "version": version,
            "submitted_at": submitted_at,
            "batches": batches,
        }

    def _harvest_done(self) -> None:
        if not self._inflight:
            return
        done = [task for task in self._inflight if task.done()]
        for task in done:
            self._inflight.discard(task)
            try:
                result = task.result()
            except asyncio.CancelledError:  # pragma: no cover
                continue
            except Exception as exc:
                self.state.last_error = repr(exc)
                self.state.error_count += 1
                continue
            self._enqueue_result(result)
        self.state.inflight_count = len(self._inflight)

    def _enqueue_result(self, result: dict[str, Any]) -> None:
        completed_at = time.time()
        for batch in result["batches"]:
            stored = self._store_batch(batch)
            item = ContinuousRolloutItem(
                item_id=self._item_counter,
                group_key=int(result["slot"]),
                rollout_policy_version=result["version"],
                batch=stored,
                submitted_at=float(result["submitted_at"]),
                completed_at=completed_at,
                nbytes=estimate_batch_bytes(stored),
            )
            self._item_counter += 1
            self.queue.put(item)

    def _store_batch(self, batch: Any) -> Any:
        return move_training_batch_to_device(batch, _CPU)


__all__ = ["ContinuousRolloutProducer"]
