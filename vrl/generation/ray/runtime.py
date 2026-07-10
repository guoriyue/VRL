"""Runtime facade for Ray-distributed generation."""

from __future__ import annotations

import contextlib
import logging
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

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class _RuntimeLease:
    """On-demand worker lease: the inner runtime is acquired on generate and
    dropped on release, so a shared GPU can be handed back between phases.

    ``sleep_eligible`` picks the release discipline. When the only handoff is a
    colocated trainer timeshare (rollout keeps its placement bundle and just has
    to vacate the *physical* GPU between collect and train), release offloads the
    workers to host RAM and reacquire wakes them — level-1 offload-and-restore
    (cf. vLLM ``sleep_level=1``), saving the per-cycle cold reload of the frozen
    VAE / text-encoders. When a reward worker must take the bundle instead
    (``release_before_reward``), the workers are still torn down (level-2) so the
    bundle is genuinely free.
    """

    config: RayGenerationConfig
    launch_contract: Any
    gatherer: Any
    placement: RolePlacement
    runtime: RayGenerationRuntime | None = None
    last_state: Any | None = None
    sleep_eligible: bool = False
    asleep: bool = False


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
        # Set True by the launcher when every resident worker retains versioned
        # trainable-state slots, which lets the continuous schedule skip the drain
        # bubble. Default False keeps the safe draining barrier. Read as a plain
        # attribute (not a method) via RolloutLifecycle.supports_non_draining_weight_sync.
        self.supports_non_draining_weight_sync = False
        # Resolved by the startup chunk-size probe on the first request that
        # carries sampling.samples_per_chunk == "auto"; a run-level constant
        # (same shape + same contract budget), so lease sleep/relaunch cycles
        # never re-probe.
        self._probed_samples_per_chunk: int | None = None

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
        runtime.weight_sync = object() if config.sync_trainable_state else None
        runtime._owned_workers = []
        runtime._owned_actors = []
        runtime._placement_group = None
        runtime._colocated = False
        # Sleep (level-1 offload-and-restore) is safe only when the lease releases
        # purely for a colocated trainer timeshare: the rollout keeps its placement
        # bundle and just vacates the physical GPU, so offloading to host RAM frees
        # the memory the trainer needs. When a reward worker must take the bundle
        # (release_before_reward) the workers are torn down so the bundle is truly
        # free; a config without a resolved topology (test specs) also tears down.
        handoff = config.resources.lifecycle.handoff if config.resources is not None else None
        sleep_eligible = (
            handoff is not None
            and handoff.release_rollout_before_train
            and not handoff.release_rollout_before_reward
        )
        runtime._release_after_collect = _RuntimeLease(
            config=config,
            # Tell the worker to pool its model in CuMemAllocator so sleep/wake
            # release+restore GPU physical pages instead of a to(cpu)/to(gpu) round
            # trip. Only for sleep-eligible leases; teardown leases load normally.
            launch_contract=_with_sleep_offload(launch_contract) if sleep_eligible else launch_contract,
            gatherer=gatherer,
            placement=placement,
            sleep_eligible=sleep_eligible,
        )
        runtime.requires_driver_model_offload = config.gpus_per_worker > 0
        runtime.current_policy_version = _launch_contract_policy_version(launch_contract)
        # Lease mode requires driver-model offload, which already hard-fails the
        # continuous schedule, so non-draining never applies here.
        runtime.supports_non_draining_weight_sync = False
        runtime._probed_samples_per_chunk = None
        return runtime

    async def generate(self, request: GenerationRequest) -> GenerationOutput:
        runtime = await self._ensure_runtime()
        if request.sampling.get("samples_per_chunk") == "auto":
            # Resolve BEFORE delegating so the lease's inner runtime (rebuilt
            # across cycles) never sees "auto"; the facade caches the verdict.
            resolved = await self._resolve_probed_samples_per_chunk(runtime, request)
            request = replace(
                request,
                sampling={**dict(request.sampling), "samples_per_chunk": resolved},
            )
        if runtime is not self:
            return await runtime.generate(request)
        if self.executor is None:
            raise RuntimeError("RayGenerationRuntime has no active executor")
        if request.policy_version is None and self.current_policy_version is not None:
            request = replace(request, policy_version=self.current_policy_version)
        return await self.executor.execute(request)

    async def _resolve_probed_samples_per_chunk(
        self,
        runtime: RayGenerationRuntime,
        request: GenerationRequest,
    ) -> int:
        """Run the startup chunk-size probe once and cache the verdict.

        vLLM shape (EngineCore init -> determine_available_memory): sizing runs
        inside the same launch, before the first real request, user does
        nothing. Here "startup" is the first generate() because lease-mode
        workers only exist after _ensure_runtime() launches them lazily. Every
        worker is probed concurrently (seconds each, homogeneous cards give
        equal answers); the fleet answer is the min.
        """

        if self._probed_samples_per_chunk is not None:
            return self._probed_samples_per_chunk
        executor = runtime.executor if runtime is not self else self.executor
        workers = list(getattr(executor, "workers", []) or [])
        if not workers:
            raise RuntimeError(
                "samples_per_chunk: auto found no generation workers to probe",
            )
        max_samples = max(1, int(request.samples_per_prompt))
        fraction = self._rollout_memory_fraction()
        refs = []
        local_results: list[dict[str, Any]] = []
        for worker in workers:
            probe = getattr(worker.actor, "probe_chunk_size", None)
            if probe is None:
                raise RuntimeError(
                    f"worker {worker.worker_id!r} does not support the "
                    "chunk-size probe required by samples_per_chunk: auto",
                )
            remote = getattr(probe, "remote", None)
            if callable(remote):
                refs.append(
                    remote(request, max_samples=max_samples, memory_fraction=fraction),
                )
            else:
                local_results.append(
                    probe(request, max_samples=max_samples, memory_fraction=fraction),
                )
        if refs:
            ray = require_ray()
            local_results.extend(ray.get(refs, timeout=600))
        resolved = min(int(result["samples_per_chunk"]) for result in local_results)
        for result in local_results:
            logger.info(
                "chunk-size probe: n=%d (budget=%.0fMB fraction=%.2f trials=%s)",
                int(result["samples_per_chunk"]),
                result["budget_bytes"] / 2**20,
                result["memory_fraction"],
                [
                    (
                        trial["label"],
                        trial["n"],
                        "OOM"
                        if trial["oom"]
                        else f"{trial['peak_bytes'] / 2**20:.0f}MB/{trial['wall_s']:.1f}s",
                    )
                    for trial in result["trials"]
                ],
            )
        self._probed_samples_per_chunk = resolved
        return resolved

    def _rollout_memory_fraction(self) -> float | None:
        """Contract GPU share for rollout (colocated trainer timeshare), if known.

        The probe must budget against this contract, not instantaneous free
        memory: at probe time a colocated trainer has not restored its weights
        yet, so free memory overestimates what rollout may keep.
        """

        state = self._release_after_collect
        if state is None or state.config.resources is None:
            return None
        return state.config.resources.rollout_gpu_memory_fraction

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
            # A slept worker holds its model on CPU; don't push onto it now. The
            # retained ``last_state`` is replayed on wake (mirroring the cold-launch
            # path), so the awoken worker always serves the latest version.
            if state.runtime is not None and not state.asleep:
                await state.runtime.update_weights(state_ref, self.current_policy_version)
            return
        if self.weight_sync is None:
            raise RuntimeError("RayGenerationRuntime has no GenerationWeightSync")
        await self.weight_sync.push_to_rollout_workers(state_ref, policy_version)
        self.current_policy_version = int(policy_version)

    async def release(self) -> None:
        """Release the on-demand workers between phases; generate() reacquires them.

        Sleep-eligible leases (colocated trainer timeshare) offload the live
        workers to host RAM and keep them alive, so reacquire wakes them without a
        cold reload. Otherwise the workers are torn down so the placement bundle is
        free for the next role. No-op for a resident runtime (actors stay up until
        shutdown()).
        """
        state = self._release_after_collect
        if state is None:
            return None
        runtime = state.runtime
        if runtime is None:
            return None
        if state.sleep_eligible:
            await runtime.sleep_workers()
            state.asleep = True
            return None
        state.runtime = None
        await runtime.shutdown()
        return None

    async def shutdown(self) -> None:
        lease = self._release_after_collect
        if lease is not None:
            # Terminal shutdown is not a phase handoff: even a sleep-eligible
            # lease must drop its actors instead of retaining an inner runtime
            # that nothing can wake after the collector releases this facade.
            runtime = lease.runtime
            lease.runtime = None
            lease.asleep = False
            if runtime is not None:
                await runtime.shutdown()
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

    async def sleep_workers(self) -> None:
        """Offload every owned worker's model to host RAM, freeing the GPU.

        The workers stay alive (process + placement bundle retained); only their
        weights leave the GPU. Failures are not suppressed: a worker that fails to
        offload would otherwise hold the GPU a colocated trainer is about to use.
        """
        if not self._owned_workers:
            return None
        ray = require_ray()
        refs = [
            worker.actor.sleep.remote()
            for worker in self._owned_workers
            if worker.actor is not None
        ]
        if refs:
            ray.get(refs, timeout=120)
        return None

    async def wake_workers(self) -> None:
        """Restore every owned worker's model from host RAM back onto its GPU."""
        if not self._owned_workers:
            return None
        ray = require_ray()
        refs = [
            worker.actor.wake.remote()
            for worker in self._owned_workers
            if worker.actor is not None
        ]
        if refs:
            ray.get(refs, timeout=120)
        return None

    async def _ensure_runtime(self) -> RayGenerationRuntime:
        state = self._release_after_collect
        if state is None:
            return self
        if state.runtime is not None and state.asleep:
            # Wake the offloaded workers (host RAM -> GPU, no disk reload) and
            # replay the latest trainable state, the same refresh contract the
            # cold-launch path below applies after a teardown.
            await state.runtime.wake_workers()
            state.asleep = False
            if state.last_state is not None:
                await state.runtime.update_weights(
                    state.last_state,
                    int(self.current_policy_version),
                )
            return state.runtime
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


def _with_sleep_offload(launch_contract: Any) -> Any:
    """Return the contract with ``extra["sleep_offload"]`` set so the worker pools
    its model for cumem sleep/wake. Best-effort: a non-contract test spec that does
    not normalize is returned unchanged (the worker just takes the naive path)."""
    from vrl.generation.launch_contract import GenerationRuntimeLaunchContract

    try:
        contract = GenerationRuntimeLaunchContract.from_value(launch_contract)
    except (TypeError, ValueError):
        return launch_contract
    return replace(contract, extra={**contract.extra, "sleep_offload": True})


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
