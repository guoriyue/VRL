"""Runtime facade for Ray-distributed generation."""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, replace
from typing import Any

from vrl.generation.execution.types import DistributedWorkerHandle
from vrl.generation.protocols import GenerationRuntime
from vrl.generation.ray.config import RayGenerationConfig
from vrl.generation.ray.executor import RayGenerationExecutor
from vrl.generation.ray.weight_sync import GenerationWeightSync
from vrl.generation.types import GenerationOutput, GenerationRequest
from vrl.ray.dependencies import require_ray
from vrl.ray.lifecycle import kill_actors, remove_placement_group
from vrl.ray.placement import RolePlacement


@dataclass(slots=True)
class _RuntimeLease:
    """On-demand worker lease: the inner runtime is acquired on generate and
    dropped on release, so a shared GPU can be handed back between phases."""

    config: RayGenerationConfig
    launch_contract: Any
    gatherer: Any
    placement: RolePlacement
    runtime: RayGenerationRuntime | None = None
    last_state: Any | None = None


class RayGenerationRuntime(GenerationRuntime):
    """Collector-facing Ray generation runtime.

    One public runtime covers both worker lifecycles:
    resident workers stay alive for split-GPU throughput and tiny colocated
    async debug; release-after-collect workers are recreated on demand when a
    shared GPU needs to be handed back to the trainer or reward model.
    """

    def __init__(
        self,
        executor: RayGenerationExecutor | Any,
        *,
        weight_sync: GenerationWeightSync | None = None,
        owned_workers: list[DistributedWorkerHandle] | None = None,
        owned_actors: list[Any] | None = None,
        placement_group: Any | None = None,
        colocated: bool = False,
    ) -> None:
        self.executor = executor
        self.weight_sync = weight_sync
        self._owned_workers = list(owned_workers or [])
        self._owned_actors = list(owned_actors or [])
        self._placement_group = placement_group
        self._colocated = bool(colocated)
        self._release_after_collect: _RuntimeLease | None = None
        self.requires_driver_model_offload = False
        self.current_policy_version: int | None = None

    @classmethod
    def with_release_after_collect(
        cls,
        config: RayGenerationConfig,
        launch_contract: Any,
        gatherer: Any,
        *,
        placement: RolePlacement,
    ) -> RayGenerationRuntime:
        """Build a runtime that recreates Ray workers between collect phases."""
        runtime = cls.__new__(cls)
        runtime.executor = None
        runtime.weight_sync = (
            object() if config.sync_trainable_state != "disabled" else None
        )
        runtime._owned_workers = []
        runtime._owned_actors = []
        runtime._placement_group = None
        runtime._colocated = False
        runtime._release_after_collect = _RuntimeLease(
            config=config,
            launch_contract=launch_contract,
            gatherer=gatherer,
            placement=placement,
        )
        runtime.requires_driver_model_offload = config.gpus_per_worker > 0
        runtime.current_policy_version = _launch_contract_policy_version(launch_contract)
        return runtime

    async def generate(self, request: GenerationRequest) -> GenerationOutput:
        runtime = await self._ensure_runtime()
        if runtime is not self:
            return await runtime.generate(request)
        if self.executor is None:
            raise RuntimeError("RayGenerationRuntime has no active executor")
        if request.policy_version is None and self.current_policy_version is not None:
            request = replace(request, policy_version=self.current_policy_version)
        return await self.executor.execute(request)

    def is_colocated(self) -> bool:
        state = self._release_after_collect
        if state is not None:
            if state.config.allow_driver_gpu_overlap:
                return True
            resources = state.config.resources
            return bool(resources is not None and resources.colocated)
        return self._colocated

    async def update_weights(self, state_ref: Any, policy_version: int) -> None:
        state = self._release_after_collect
        if state is not None:
            if self.weight_sync is None:
                raise RuntimeError("RayGenerationRuntime has no generation weight sync")
            state.last_state = state_ref
            self.current_policy_version = int(policy_version)
            if state.runtime is not None:
                await state.runtime.update_weights(state_ref, self.current_policy_version)
            return
        if self.weight_sync is None:
            raise RuntimeError("RayGenerationRuntime has no GenerationWeightSync")
        await self.weight_sync.push_to_rollout_workers(state_ref, policy_version)
        self.current_policy_version = int(policy_version)

    async def release(self) -> None:
        """Drop the on-demand workers (lease release); generate() reacquires them.

        No-op for a resident runtime, whose actors stay up until shutdown().
        """
        state = self._release_after_collect
        if state is None:
            return None
        runtime = state.runtime
        if runtime is None:
            return None
        state.runtime = None
        await runtime.shutdown()
        return None

    async def shutdown(self) -> None:
        if self._release_after_collect is not None:
            await self.release()
            return None
        if not self._owned_workers and not self._owned_actors and self._placement_group is None:
            return None
        ray = require_ray()
        release_refs: list[Any] = []
        for worker in self._owned_workers:
            actor = worker.actor
            if actor is None:
                continue
            with contextlib.suppress(Exception):
                release_refs.append(actor.release_policy.remote())
        if release_refs:
            with contextlib.suppress(Exception):
                ray.get(release_refs, timeout=60)
        kill_actors(
            ray,
            [worker.actor for worker in self._owned_workers if worker.actor is not None],
        )
        self._owned_workers.clear()
        kill_actors(ray, self._owned_actors)
        self._owned_actors.clear()
        remove_placement_group(self._placement_group)
        self._placement_group = None
        return None

    async def _ensure_runtime(self) -> RayGenerationRuntime:
        state = self._release_after_collect
        if state is None:
            return self
        if state.runtime is None:
            from vrl.generation.ray.launcher import RayGenerationLauncher

            runtime = RayGenerationLauncher().launch(
                state.config,
                state.launch_contract,
                state.gatherer,
                placement=state.placement,
            )
            state.runtime = runtime
            if state.last_state is not None:
                await runtime.update_weights(
                    state.last_state,
                    int(self.current_policy_version),
                )
        return state.runtime


def _launch_contract_policy_version(launch_contract: Any) -> int | None:
    from vrl.generation.launch_contract import GenerationRuntimeLaunchContract

    if isinstance(launch_contract, GenerationRuntimeLaunchContract):
        contract = launch_contract
    else:
        try:
            contract = GenerationRuntimeLaunchContract.from_value(launch_contract)
        except (TypeError, ValueError):
            return None
    if contract.policy_version is None:
        return None
    return int(contract.policy_version)


__all__ = ["RayGenerationRuntime"]
