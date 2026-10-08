"""In-process generation runtime for tests: the rollout worker body without a Ray actor.

Test infrastructure, not a production path: it lets collector, schedule and
trainer tests run real generation (the production ``GenerationWorkerCore``,
family executor, request planning and batch merging) in one process, without
starting a Ray cluster. The Ray layer itself is covered by tests/generation/ray
and the online-recipe lifecycle tests on a real local cluster.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Callable
from dataclasses import replace
from typing import Any

from vrl.generation.execution.planner import EnginePlan
from vrl.generation.execution.types import GenerationBatchEnvelope, StaleSlotDiscard
from vrl.generation.execution.worker import GenerationWorkerCore
from vrl.generation.ray.launch_inputs import RayGenerationLaunchInputs
from vrl.generation.types import GenerationOutput, GenerationRequest


class InProcessGenerationRuntime:
    """Drive one ``GenerationWorkerCore`` in the calling process.

    The same worker body a Ray rank hosts, bound to the caller's process, for
    tests that want the real family executor, request planning and batch
    merging without a cluster. Like a Ray
    actor, the worker body executes one call at a time on a worker thread, so
    the driver's event loop stays free and a prefetched request queues behind
    the running one. Whether a weight sync may skip the drain is the launch
    contract's versioned-slot decision, exactly as for a Ray fleet: the worker
    core then installs each version into its own slot and serves every request
    from the slot of the version it was stamped with. ``activate`` loads or
    wakes the policy; ``offload`` parks it only when the launch contract asks
    for a shared-GPU lease, otherwise the model stays resident like a
    dedicated rollout worker.
    """

    def __init__(
        self,
        launch_inputs: RayGenerationLaunchInputs,
        *,
        worker_id: str = "rollout-0",
    ) -> None:
        contract = launch_inputs.launch_contract
        self._core = GenerationWorkerCore(worker_id, contract)
        self._gatherer = launch_inputs.gatherer
        self.current_policy_version: int = contract.policy_version
        self._parked = False
        # The actor's one-call-at-a-time execution.
        self._actor = threading.Lock()

    @property
    def worker(self) -> GenerationWorkerCore:
        return self._core

    async def _call(self, operation: Callable[..., Any], *args: Any) -> Any:
        def run() -> Any:
            with self._actor:
                return operation(*args)

        return await asyncio.to_thread(run)

    async def activate(self) -> None:
        await self._call(self._activate)

    def _activate(self) -> None:
        if self._parked:
            self._core.wake()
            self._parked = False
            return
        self._core.load_policy()

    async def generate(self, request: GenerationRequest) -> GenerationOutput:
        if request.policy_version is None:
            request = replace(request, policy_version=self.current_policy_version)
        return await self._call(self._generate, request)

    def _generate(self, request: GenerationRequest) -> GenerationOutput:
        plan = EnginePlan.from_request(request)
        payloads = []
        for batch in plan.sample_batches:
            result = self._core.execute_batch(
                GenerationBatchEnvelope(request=request, batch=batch)
            )
            if result.stale_slot:
                raise StaleSlotDiscard(result.error or "trainable-state slot evicted")
            if result.error is not None or result.output is None:
                raise RuntimeError(
                    f"in-process generation batch {batch.batch_key} failed: {result.error}",
                )
            payloads.append(result.output)
        return self._gatherer.merge_generation_batches(request, payloads)

    async def update_weights(self, trainable_state: Any, policy_version: int) -> None:
        self.current_policy_version = await self._call(
            self._core.update_weights, trainable_state, policy_version
        )

    async def offload(self) -> None:
        await self._call(self._offload)

    def _offload(self) -> None:
        if self._parked or self._core.executor is None:
            return
        if not self._core.launch_contract.sleep_offload:
            return
        self._core.sleep()
        self._parked = True

    async def shutdown(self) -> None:
        await self._call(self._shutdown)

    def _shutdown(self) -> None:
        self._core.release_policy()
        self._parked = False


__all__ = ["InProcessGenerationRuntime"]
