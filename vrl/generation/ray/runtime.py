"""Runtime facade for Ray-distributed generation."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from dataclasses import dataclass, replace
from typing import Any

from vrl.generation.execution.types import (
    DistributedWorkerHandle,
    WorkerMemoryParkingSnapshot,
)
from vrl.generation.protocols import GenerationRuntime
from vrl.generation.ray.config import RayGenerationConfig
from vrl.generation.ray.executor import RayGenerationExecutor
from vrl.generation.ray.health_monitor import RolloutWorkerHealthMonitor
from vrl.generation.ray.launch_inputs import RayGenerationLaunchInputs
from vrl.generation.ray.lifecycle_fsm import RuntimeLifecycle, RuntimePhase
from vrl.generation.ray.weight_sync import GenerationWeightSync
from vrl.generation.types import GenerationOutput, GenerationRequest
from vrl.ray.dependencies import require_ray
from vrl.ray.operation_deadline import await_ray_refs
from vrl.ray.placement import RolePlacement
from vrl.ray.resource_cleanup import kill_and_retain
from vrl.runtime_errors import TerminalRuntimeError

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class _PolicySnapshot:
    """Latest complete trainer state desired by the rollout workers."""

    state_ref: Any
    policy_version: int


@dataclass(slots=True)
class _OnDemandRuntimeState:
    """Explicit rollout activation state across shared-GPU phase handoffs.

    The schedule activates this runtime before generation and offloads it after
    generation.  The first activation launches workers; later activations wake the
    same workers.  ``activation_task`` provides single-flight launch/wake, while
    ``desired_policy`` records weights published while workers are offloaded.
    """

    config: RayGenerationConfig
    launch_inputs: RayGenerationLaunchInputs
    placement: RolePlacement
    inner_runtime: RayGenerationRuntime | None = None
    activation_task: asyncio.Task[RayGenerationRuntime] | None = None
    desired_policy: _PolicySnapshot | None = None
    active_policy_version: int | None = None
    workers_offloaded: bool = False


class RayGenerationRuntime(GenerationRuntime):
    """Collector-facing Ray generation runtime.

    Resident workers remain GPU-active on dedicated GPUs. Shared-GPU workers
    launch lazily, remain alive across phase handoffs, and park their model in
    host RAM while the trainer or in-process reward uses the physical GPU.
    """

    def __init__(
        self,
        executor: RayGenerationExecutor | Any,
        *,
        weight_sync: GenerationWeightSync | None = None,
        owned_workers: list[DistributedWorkerHandle] | None = None,
        colocated: bool = False,
        health_check_interval_s: float = 0.0,
        health_check_timeout_s: float = 30.0,
        health_check_first_wait_s: float = 0.0,
    ) -> None:
        self.executor = executor
        self.weight_sync = weight_sync
        self._owned_workers = list(owned_workers or [])
        self._colocated = bool(colocated)
        # Operation deadlines bound active business calls; this complementary
        # monitor covers process death between calls and independently verifies
        # that the actor's health concurrency group remains reachable.
        self._health_monitor = RolloutWorkerHealthMonitor(
            self,
            interval_s=health_check_interval_s,
            timeout_s=health_check_timeout_s,
            first_wait_s=health_check_first_wait_s,
        )
        self._on_demand: _OnDemandRuntimeState | None = None
        # Rollout schedules own pause/drain. The runtime lifecycle only closes
        # terminal admission and records the first cleanup-worthy failure.
        self.lifecycle = RuntimeLifecycle()
        self._offload_task: asyncio.Task[None] | None = None
        self._shutdown_task: asyncio.Task[None] | None = None
        self._release_wait_task: asyncio.Task[Any] | None = None
        self._force_shutdown = False
        self.current_policy_version: int | None = None
        # Set True by the launcher when every resident worker retains versioned
        # trainable-state slots, which lets the continuous schedule skip the drain
        # bubble. Default False keeps the safe draining barrier. Read as a plain
        # attribute (not a method) via RolloutLifecycle.supports_non_draining_weight_sync.
        self.supports_non_draining_weight_sync = False
        # Resolved by the startup chunk-size probe on the first request that
        # carries sampling.samples_per_chunk == "auto"; a run-level constant
        # (same shape + same phase budget), so sleep/wake and recovery relaunch
        # never re-probe.
        self._probed_samples_per_chunk: int | None = None

    @classmethod
    def with_on_demand_activation(
        cls,
        config: RayGenerationConfig,
        launch_inputs: RayGenerationLaunchInputs,
        *,
        placement: RolePlacement,
    ) -> RayGenerationRuntime:
        """Build a runtime explicitly activated/offloaded by its schedule."""
        resources = config.resources
        if resources.lifecycle.rollout.mode != "on_demand":
            raise ValueError(
                "with_on_demand_activation requires a resolved on-demand rollout lifecycle plan",
            )
        contract = launch_inputs.launch_contract
        sleep_contract = contract
        if resources.rollout_gpus_per_worker > 0:
            sleep_contract = replace(
                contract,
                sleep_offload=True,
            )

        runtime = cls(executor=None)
        runtime._on_demand = _OnDemandRuntimeState(
            config=config,
            # GPU workers pool their model in CuMemAllocator so offload/activate
            # release and restore physical pages. CPU-only worker tests keep the
            # ordinary allocator because there is no CUDA memory to pool.
            launch_inputs=replace(
                launch_inputs,
                launch_contract=sleep_contract,
            ),
            placement=placement,
        )
        runtime.current_policy_version = contract.policy_version
        return runtime

    @property
    def requires_driver_model_offload(self) -> bool:
        """Whether an on-demand rollout must vacate the trainer's physical GPU."""

        state = self._on_demand
        return bool(
            state is not None
            and state.config.resources.colocated
            and state.config.resources.rollout_gpus_per_worker > 0
        )

    @property
    def supports_weight_sync(self) -> bool:
        """Whether trainer weight pushes have a configured worker consumer.

        An on-demand runtime has no inner ``GenerationWeightSync`` before first
        use, so capability cannot be inferred from the current handle. Derive it
        from the normalized launch config; resident runtimes use their live syncer.
        """

        state = self._on_demand
        if state is not None:
            return bool(state.config.worker.sync_trainable_state)
        return self.weight_sync is not None

    def start_health_monitoring(self) -> None:
        """Begin probing this runtime's own workers. Idempotent and opt-in."""

        if self._health_monitor.start():
            self._health_monitor.resume()

    async def activate(self) -> None:
        """Launch or wake on-demand workers and install the desired policy.

        Schedules call this before admitting generation. Resident runtimes are
        already active, so activation is a no-op for them.
        """

        self.lifecycle.require_running("activate")
        state = self._on_demand
        if state is None:
            return None
        task = state.activation_task
        if task is not None and task.done():
            self._activation_finished(state, task)
            task = None
        if task is None:
            task = asyncio.create_task(self._activate_once(state))
            state.activation_task = task
            task.add_done_callback(lambda done: self._activation_finished(state, done))
        try:
            await asyncio.shield(task)
        except BaseException:
            if self.lifecycle.phase is RuntimePhase.SHUTTING_DOWN:
                await self.shutdown()
            raise
        return None

    async def generate(self, request: GenerationRequest) -> GenerationOutput:
        self.lifecycle.require_running("generate")
        try:
            runtime = self._active_runtime()
            if request.sampling.get("samples_per_chunk") == "auto":
                # Resolve before delegation so every worker receives an integer. The
                # facade caches this run-level verdict across offload/onload cycles.
                resolved = await self._resolve_probed_samples_per_chunk(runtime, request)
                request = replace(
                    request,
                    sampling={**dict(request.sampling), "samples_per_chunk": resolved},
                )
            if runtime is not self:
                output = await runtime.generate(request)
                self.lifecycle.require_running("complete generation")
                return output
            if self.executor is None:
                raise RuntimeError("RayGenerationRuntime has no active executor")
            if request.policy_version is None and self.current_policy_version is not None:
                request = replace(request, policy_version=self.current_policy_version)
            output = await self.executor.execute(request)
            self.lifecycle.require_running("complete generation")
            return output
        except TerminalRuntimeError as error:
            # Cancellation cannot reliably interrupt synchronous actor code. Close
            # admission and destroy the fleet before any partial request result can
            # escape this runtime.
            await self._terminalize_after_failure(error)
            raise

    async def _resolve_probed_samples_per_chunk(
        self,
        runtime: RayGenerationRuntime,
        request: GenerationRequest,
    ) -> int:
        """Run the startup chunk-size probe once and cache the verdict.

        vLLM shape (EngineCore init -> determine_available_memory): sizing runs
        inside the same activation, before the first real request, user does
        nothing. Here "startup" is the first generate() after activate() because
        on-demand workers only exist after explicit activation. Every
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
                refs.append(remote(request, max_samples=max_samples))
            else:
                local_results.append(probe(request, max_samples=max_samples))
        if refs:
            ray = require_ray()
            local_results.extend(
                await await_ray_refs(
                    refs,
                    operation="rollout.generation.chunk_size_probe",
                    timeout_s=executor.generation_stall_timeout_s,
                    context=f"workers={len(refs)}, request_id={request.request_id}",
                    ray=ray,
                ),
            )
        resolved = min(int(result["samples_per_chunk"]) for result in local_results)
        for result in local_results:
            logger.info(
                "chunk-size probe: n=%d (budget=%.0fMB trials=%s)",
                int(result["samples_per_chunk"]),
                result["budget_bytes"] / 2**20,
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

    def is_colocated(self) -> bool:
        state = self._on_demand
        if state is not None:
            return bool(state.config.resources.colocated)
        return self._colocated

    async def update_weights(self, state_ref: Any, policy_version: int) -> None:
        # Contract/precondition failures do not make installed worker state
        # unknown and therefore must not enter terminal quarantine. In
        # particular, shutdown cannot join an activation task whose caller is
        # still waiting for this validation to return before releasing it.
        self.lifecycle.require_running("update_weights")
        state = self._on_demand
        if state is not None:
            if not state.config.worker.sync_trainable_state:
                raise RuntimeError(
                    "RayGenerationRuntime has no generation weight sync",
                )
            activation = state.activation_task
            if activation is not None and not activation.done():
                raise RuntimeError(
                    "update_weights requires rollout activation to be idle; "
                    "the rollout schedule must pause/drain before syncing",
                )
        elif self.weight_sync is None:
            raise RuntimeError("RayGenerationRuntime has no GenerationWeightSync")

        try:
            await self._install_policy(
                _PolicySnapshot(state_ref=state_ref, policy_version=int(policy_version)),
            )
        except BaseException as error:
            await self._terminalize_after_failure(error)
            raise

    async def _install_policy(self, policy: _PolicySnapshot) -> None:
        self.lifecycle.require_running("update_weights")
        state = self._on_demand
        if state is not None:
            inner_runtime = state.inner_runtime
            if inner_runtime is not None and not state.workers_offloaded:
                await inner_runtime.update_weights(
                    policy.state_ref,
                    policy.policy_version,
                )
                state.active_policy_version = policy.policy_version
            state.desired_policy = policy
            self.current_policy_version = policy.policy_version
            return

        if self.weight_sync is None:
            # Internal activation only reaches this boundary after a launcher
            # contract mismatch; public calls reject it before quarantine.
            raise RuntimeError("RayGenerationRuntime has no GenerationWeightSync")
        await self.weight_sync.push_to_rollout_workers(
            policy.state_ref,
            policy.policy_version,
        )
        self.current_policy_version = policy.policy_version

    async def _terminalize_after_failure(
        self,
        error: BaseException,
        *,
        join_control_tasks: bool = True,
    ) -> None:
        """Close admission and preserve the operation error across cleanup."""

        # Publish the operation root before upgrading an existing graceful
        # shutdown. Cancellation of its release barrier must never win the
        # first-failure slot over the terminal error that required the upgrade.
        self.lifecycle.fail(error)
        if isinstance(error, TerminalRuntimeError):
            self._force_shutdown = True
            # A terminal distributed operation proves that graceful RPC progress
            # cannot be trusted. Cancel runtime-owned local control tasks so an
            # existing shutdown join upgrades immediately to forceful teardown.
            state = self._on_demand
            control_tasks = [self._offload_task]
            if state is not None:
                control_tasks.append(state.activation_task)
            current = asyncio.current_task()
            for task in control_tasks:
                if task is not None and task is not current and not task.done():
                    task.cancel()
            # A concurrent ordinary shutdown may already be waiting on a
            # release_policy RPC. Cancel only that local barrier so the shared
            # shutdown task continues into actor destruction for every waiter.
            release_wait_task = self._release_wait_task
            if (
                release_wait_task is not None
                and release_wait_task is not current
                and not release_wait_task.done()
            ):
                release_wait_task.cancel()
        try:
            if join_control_tasks:
                await self.shutdown()
            else:
                # A runtime-owned control task cannot await shutdown(): the
                # shutdown task joins that same control task. It already owns
                # its completion, so tear down resources directly.
                await self._teardown_owned_resources()
                self.lifecycle.finish_shutdown()
        except BaseException as cleanup_error:
            # The failed install/ACK is the transaction's root cause. Cleanup is
            # retryable by the schedule owner, so it must never replace that
            # first failure at the trainer boundary.
            logger.error(
                "generation terminal cleanup failed after operation error %r",
                error,
                exc_info=(
                    type(cleanup_error),
                    cleanup_error,
                    cleanup_error.__traceback__,
                ),
            )
            error.add_note(f"generation terminal cleanup also failed: {cleanup_error!r}")

    async def offload(self) -> None:
        """Park explicitly idle on-demand workers between GPU phases.

        The actors and owner-managed placement bundle stay alive; their model
        weights move to host RAM so the trainer or in-process reward can use the
        physical GPU. The schedule must drain generation before this call; the
        next activate() wakes workers and restores the desired policy.
        Resident runtimes remain a no-op here and keep serving until shutdown().
        """
        state = self._on_demand
        if state is None or (state.inner_runtime is None and state.activation_task is None):
            return None
        if self.lifecycle.phase is RuntimePhase.TERMINATED:
            return None
        if self.lifecycle.phase is RuntimePhase.SHUTTING_DOWN:
            await self.shutdown()
            return None
        activation = state.activation_task
        if activation is not None and not activation.done():
            # A cancelled activate() waiter does not cancel its shielded
            # launch/wake task. Join that one runtime-owned task so a schedule's
            # finally block can still park workers without reviving a generic
            # operation barrier.
            try:
                await asyncio.shield(activation)
            except BaseException:
                if self.lifecycle.phase is RuntimePhase.SHUTTING_DOWN:
                    await self.shutdown()
                raise
            if self.lifecycle.phase is RuntimePhase.SHUTTING_DOWN:
                await self.shutdown()
                return None
        task = self._offload_task
        if task is not None and task.done():
            self._offload_finished(task)
            task = None
        if task is None:
            task = asyncio.create_task(self._offload_once(state))
            self._offload_task = task
            task.add_done_callback(self._offload_finished)
        try:
            await asyncio.shield(task)
        except BaseException:
            # Caller cancellation does not cancel the shared task. A real worker
            # failure closes admission inside _offload_once and triggers cleanup.
            if self.lifecycle.phase is RuntimePhase.SHUTTING_DOWN:
                await self.shutdown()
            raise
        if self.lifecycle.phase is RuntimePhase.SHUTTING_DOWN:
            await self.shutdown()
        return None

    def _offload_finished(self, task: asyncio.Task[None]) -> None:
        if self._offload_task is task:
            self._offload_task = None
        if not task.cancelled():
            task.exception()  # retrieve failures even if every waiter was cancelled

    async def _offload_once(self, state: _OnDemandRuntimeState) -> None:
        try:
            if self.lifecycle.phase is not RuntimePhase.RUNNING:
                return None
            inner_runtime = state.inner_runtime
            if inner_runtime is None:
                return None
            if not state.workers_offloaded:
                await inner_runtime.sleep_workers()
                state.workers_offloaded = True
            return None
        except BaseException as error:
            self.lifecycle.fail(error)
            raise

    async def shutdown(self) -> None:
        """Close admission and tear down owned resources exactly once.

        Shielding the shared task keeps cancellation of one caller from
        cancelling cleanup for every caller. A failed cleanup remains
        SHUTTING_DOWN and clears the task so a later shutdown call can retry
        owned resources. The schedule stops and joins generation before calling
        shutdown; this method only joins runtime-owned activation/offload tasks.
        """

        if self.lifecycle.phase is RuntimePhase.TERMINATED:
            return None
        task = self._shutdown_task
        if task is not None and task.done():
            self._shutdown_finished(task)
            task = None
        if task is None:
            self.lifecycle.begin_shutdown()
            task = asyncio.create_task(self._shutdown_once())
            self._shutdown_task = task
            task.add_done_callback(self._shutdown_finished)
        await asyncio.shield(task)
        return None

    def _shutdown_finished(self, task: asyncio.Task[None]) -> None:
        if self._shutdown_task is task:
            self._shutdown_task = None
        if not task.cancelled():
            task.exception()  # retrieve failures even if every waiter was cancelled

    async def _shutdown_once(self) -> None:
        await self._join_control_tasks()
        try:
            await self._teardown_owned_resources()
        except BaseException as error:
            root_failure = self.lifecycle.failure
            self.lifecycle.fail(error)
            if root_failure is not None and root_failure is not error:
                raise error from root_failure
            raise
        self.lifecycle.finish_shutdown()

    async def _join_control_tasks(self) -> None:
        """Join launch/wake and offload work owned below the schedule boundary."""

        state = self._on_demand
        tasks = [self._offload_task]
        if state is not None:
            tasks.append(state.activation_task)
        current = asyncio.current_task()
        pending = [task for task in tasks if task is not None and task is not current]
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

    async def _teardown_owned_resources(self) -> None:
        self._health_monitor.stop()
        state = self._on_demand
        if state is not None:
            # Terminal shutdown is not a phase handoff: parked actors must be
            # destroyed instead of retained after the facade becomes unreachable.
            inner_runtime = state.inner_runtime
            if inner_runtime is not None:
                await inner_runtime.shutdown()
            state.inner_runtime = None
            state.workers_offloaded = False
            state.active_policy_version = None
            return None
        if not self._owned_workers:
            return None
        ray = require_ray()
        doomed = [worker.actor for worker in self._owned_workers if worker.actor is not None]
        logger.info(
            "runtime shutdown: killing %d owned worker actor(s)",
            len(doomed),
        )
        # A timed-out business call may still occupy the actor's default
        # concurrency group. Waiting for release_policy on that same group would
        # add another 60 seconds before the only reliable action: ray.kill.
        if self.lifecycle.failure is None and not self._force_shutdown:
            release_refs: list[Any] = []
            for worker in self._owned_workers:
                actor = worker.actor
                if actor is None:
                    continue
                with contextlib.suppress(Exception):
                    release_refs.append(actor.release_policy.remote())
            if release_refs:
                release_wait_task = asyncio.create_task(
                    asyncio.to_thread(ray.get, release_refs, timeout=60),
                )
                self._release_wait_task = release_wait_task
                try:
                    await release_wait_task
                except asyncio.CancelledError:
                    if not self._force_shutdown:
                        raise
                except Exception:
                    pass
                finally:
                    if self._release_wait_task is release_wait_task:
                        self._release_wait_task = None
        surviving, worker_failures = kill_and_retain(
            ray,
            self._owned_workers,
            lambda worker: worker.actor,
        )
        self._owned_workers[:] = surviving

        failures = [error for _, error in worker_failures]
        if failures:
            raise RuntimeError(
                "Ray runtime cleanup incomplete: "
                f"{len(worker_failures)} worker actor kill(s) failed",
            ) from failures[0]
        return None

    async def sleep_workers(self) -> tuple[WorkerMemoryParkingSnapshot, ...]:
        """Offload every owned worker's model to host RAM, freeing the GPU.

        The workers stay alive (process + placement bundle retained); only their
        weights leave the GPU. Failures are not suppressed: a worker that fails to
        offload would otherwise hold the GPU a colocated trainer is about to use.
        """
        # Parking transitions and parked intervals are outside the active
        # serving SLA, so lifecycle policy pauses monitoring across them.
        self._health_monitor.pause()
        if not self._owned_workers:
            return ()
        missing_actor_ids = tuple(
            worker.worker_id for worker in self._owned_workers if worker.actor is None
        )
        if missing_actor_ids:
            raise RuntimeError(
                f"generation workers have no actor: {missing_actor_ids}",
            )
        active_workers = tuple(
            worker for worker in self._owned_workers if worker.actor is not None
        )
        worker_ids = tuple(worker.worker_id for worker in active_workers)
        if len(set(worker_ids)) != len(worker_ids):
            raise RuntimeError(f"duplicate generation worker ids: {worker_ids}")
        refs = [worker.actor.sleep.remote() for worker in active_workers]
        values = await asyncio.wait_for(asyncio.gather(*refs), timeout=120) if refs else []
        snapshots: list[WorkerMemoryParkingSnapshot] = []
        for worker, value in zip(active_workers, values, strict=True):
            if not isinstance(value, WorkerMemoryParkingSnapshot):
                raise TypeError(
                    f"worker {worker.worker_id!r} returned invalid memory-parking "
                    f"report {type(value).__name__}",
                )
            if value.worker_id != worker.worker_id:
                raise RuntimeError(
                    "mismatched worker memory-parking report: "
                    f"expected={worker.worker_id!r} actual={value.worker_id!r}",
                )
            value.validate()
            snapshots.append(value)
        return tuple(snapshots)

    async def wake_workers(self) -> None:
        """Restore every owned worker's model from host RAM back onto its GPU."""
        if not self._owned_workers:
            self._health_monitor.resume()
            return None
        refs = [
            worker.actor.wake.remote()
            for worker in self._owned_workers
            if worker.actor is not None
        ]
        if refs:
            await asyncio.wait_for(asyncio.gather(*refs), timeout=120)
        self._health_monitor.resume()
        return None

    def _active_runtime(self) -> RayGenerationRuntime:
        state = self._on_demand
        if state is None:
            return self
        if state.inner_runtime is None or state.workers_offloaded:
            raise RuntimeError(
                "generate requires an active rollout runtime; "
                "the rollout schedule must await activate() first",
            )
        return state.inner_runtime

    def _activation_finished(
        self,
        state: _OnDemandRuntimeState,
        task: asyncio.Task[RayGenerationRuntime],
    ) -> None:
        if state.activation_task is task:
            state.activation_task = None
        if not task.cancelled():
            task.exception()  # retrieve failures even if every waiter was cancelled

    async def _activate_once(
        self,
        state: _OnDemandRuntimeState,
    ) -> RayGenerationRuntime:
        try:
            if state.inner_runtime is not None and state.workers_offloaded:
                inner_runtime = state.inner_runtime
                await inner_runtime.wake_workers()
                # Record physical truth before applying weights so terminal
                # cleanup treats a failed restore as an awake runtime.
                state.workers_offloaded = False
                desired = state.desired_policy
                if desired is not None and desired.policy_version != state.active_policy_version:
                    await inner_runtime.update_weights(
                        desired.state_ref,
                        desired.policy_version,
                    )
                    state.active_policy_version = desired.policy_version
                return inner_runtime
            if state.inner_runtime is not None:
                return state.inner_runtime

            from vrl.generation.ray.launcher import RayGenerationLauncher

            candidate = await RayGenerationLauncher().launch_async(
                state.config,
                state.launch_inputs,
                placement=state.placement,
            )
            try:
                desired = state.desired_policy
                active_policy_version = candidate.current_policy_version
                if desired is not None:
                    await candidate.update_weights(
                        desired.state_ref,
                        desired.policy_version,
                    )
                    active_policy_version = desired.policy_version
            except BaseException as restore_error:
                # A candidate is not published until restore succeeds;
                # clean it here so a failed cold activation cannot leak actors.
                try:
                    await candidate.shutdown()
                except BaseException as cleanup_error:
                    # Retain ownership when candidate cleanup itself fails. The
                    # terminal shutdown retry path can now reach the handles.
                    state.inner_runtime = candidate
                    state.workers_offloaded = False
                    logger.error(
                        "candidate cleanup failed after restore error %r",
                        restore_error,
                        exc_info=cleanup_error,
                    )
                raise
            state.active_policy_version = active_policy_version
            state.inner_runtime = candidate
            return candidate
        except BaseException as error:
            self.lifecycle.fail(error)
            raise


__all__ = ["RayGenerationRuntime"]
