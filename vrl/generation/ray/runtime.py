"""Runtime facade for Ray-distributed generation."""

from __future__ import annotations

import contextlib
from dataclasses import replace
from typing import Any

from vrl.generation.execution.types import DistributedWorkerHandle
from vrl.generation.protocols import GenerationRuntime
from vrl.generation.ray.config import RayGenerationConfig
from vrl.generation.ray.executor import RayGenerationExecutor
from vrl.generation.ray.weight_sync import GenerationWeightSync
from vrl.generation.types import GenerationOutput, GenerationRequest
from vrl.ray.dependencies import require_ray


class RayGenerationRuntime(GenerationRuntime):
    """Collector-facing runtime backed by Ray generation workers."""

    def __init__(
        self,
        executor: RayGenerationExecutor,
        *,
        weight_sync: GenerationWeightSync | None = None,
        owned_workers: list[DistributedWorkerHandle] | None = None,
        owned_actors: list[Any] | None = None,
        placement_group: Any | None = None,
    ) -> None:
        self.executor = executor
        self.weight_sync = weight_sync
        self._owned_workers = list(owned_workers or [])
        self._owned_actors = list(owned_actors or [])
        self._placement_group = placement_group
        self.current_policy_version: int | None = None

    async def generate(self, request: GenerationRequest) -> GenerationOutput:
        if request.policy_version is None and self.current_policy_version is not None:
            request = replace(request, policy_version=self.current_policy_version)
        return await self.executor.execute(request)

    async def update_weights(self, state_ref: Any, policy_version: int) -> None:
        if self.weight_sync is None:
            raise RuntimeError("RayGenerationRuntime has no GenerationWeightSync")
        await self.weight_sync.push_to_rollout_workers(state_ref, policy_version)
        self.current_policy_version = int(policy_version)

    async def shutdown(self) -> None:
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
        for worker in self._owned_workers:
            actor = worker.actor
            if actor is None:
                continue
            with contextlib.suppress(Exception):
                ray.kill(actor, no_restart=True)
        self._owned_workers.clear()
        for actor in self._owned_actors:
            with contextlib.suppress(Exception):
                ray.kill(actor, no_restart=True)
        self._owned_actors.clear()
        if self._placement_group is not None:
            with contextlib.suppress(Exception):
                from ray.util import remove_placement_group

                remove_placement_group(self._placement_group)
            self._placement_group = None
        return None


class ReleasableRayGenerationRuntime(GenerationRuntime):
    """Ray runtime wrapper that can drop generation actors between train phases.

    Single-GPU Ray debugging colocates the trainer and one generation worker on
    the same CUDA device. Keeping the generation worker alive after collection
    leaves the full generation pipeline resident while the trainer replays the
    batch, which is too much for large diffusion models. This wrapper keeps the
    public runtime object stable for the trainer/weight-syncer, but tears down
    and recreates the underlying Ray actors when ``release_memory()`` is called.
    """

    def __init__(
        self,
        config: RayGenerationConfig,
        launch_contract: Any,
        gatherer: Any,
    ) -> None:
        self.config = config
        self.launch_contract = launch_contract
        self.gatherer = gatherer
        self.weight_sync = object() if config.sync_trainable_state != "disabled" else None
        self.requires_driver_model_offload = config.gpus_per_worker > 0
        self.current_policy_version = _launch_contract_policy_version(launch_contract)
        self._runtime: Any | None = None
        self._last_state: Any | None = None

    async def generate(self, request: Any) -> Any:
        runtime = await self._ensure_runtime()
        return await runtime.generate(request)

    async def update_weights(self, state_ref: Any, policy_version: int) -> None:
        if self.weight_sync is None:
            raise RuntimeError("ReleasableRayGenerationRuntime has no generation weight sync")
        self._last_state = state_ref
        self.current_policy_version = int(policy_version)
        if self._runtime is not None:
            await self._runtime.update_weights(state_ref, self.current_policy_version)

    async def release_memory(self) -> None:
        runtime = self._runtime
        if runtime is None:
            return
        self._runtime = None
        await runtime.shutdown()

    async def shutdown(self) -> None:
        await self.release_memory()

    async def _ensure_runtime(self) -> Any:
        if self._runtime is None:
            from vrl.generation.ray.launcher import RayGenerationLauncher

            runtime = RayGenerationLauncher().launch(
                self.config,
                self.launch_contract,
                self.gatherer,
            )
            self._runtime = runtime
            if self._last_state is not None:
                await runtime.update_weights(
                    self._last_state,
                    int(self.current_policy_version),
                )
        return self._runtime


def _launch_contract_policy_version(launch_contract: Any) -> int | None:
    try:
        from vrl.generation.launch_contract import GenerationRuntimeLaunchContract

        contract = GenerationRuntimeLaunchContract.from_value(launch_contract)
    except Exception:
        return None
    if contract.policy_version is None:
        return None
    return int(contract.policy_version)


__all__ = ["RayGenerationRuntime", "ReleasableRayGenerationRuntime"]
