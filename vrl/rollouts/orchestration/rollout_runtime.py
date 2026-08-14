"""Runtime coordination shared by RL rollout schedules.

``RolloutRuntimeCoordinator`` gives the strict and continuous schedules one
implementation of the trainer-side lease operations: parking/restoring
training state around a shared-GPU phase, preparing weight snapshots on the
trainer thread (strategy export may run DDP/FSDP collectives) and pushing
them from any loop, and tracking the policy version across syncs. Schedules
talk to the collector only through the ``RolloutCollectorControl`` protocol,
so the scheduling layer never imports a concrete collector or strategy type.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Mapping
from contextlib import asynccontextmanager
from typing import Any, Protocol, runtime_checkable

import torch

from vrl.generation import GenerationRuntime
from vrl.rollouts.stats import RolloutStats


class RolloutPhaseCleanupError(RuntimeError):
    """A rollout phase failed and its terminal cleanup also failed."""

    def __init__(
        self,
        root_cause: BaseException,
        cleanup_error: BaseException,
    ) -> None:
        self.root_cause = root_cause
        self.cleanup_error = cleanup_error
        super().__init__(
            f"rollout phase root cause: {type(root_cause).__name__}: {root_cause}; "
            "terminal cleanup failure: "
            f"{type(cleanup_error).__name__}: {cleanup_error}",
        )


@runtime_checkable
class RolloutCollectorControl(Protocol):
    """Generation-runtime controls required for a complete phase handoff."""

    @property
    def generation_runtime(self) -> GenerationRuntime: ...

    @property
    def requires_generation_offload_before_reward(self) -> bool: ...

    @property
    def requires_driver_model_offload_for_reward(self) -> bool: ...

    @property
    def supports_reward_generation_overlap(self) -> bool: ...

    @property
    def supports_continuous_reward_execution(self) -> bool: ...

    async def activate_generation_runtime(self) -> None: ...

    async def offload_generation_runtime_memory(self) -> None: ...

    async def shutdown(self) -> None: ...


class RolloutRuntimeCoordinator:
    """Coordinate runtime operations shared by strict and overlapped schedules."""

    def __init__(
        self,
        *,
        collector: Any,
        strategy: Any,
        training_state_getter: Callable[[], Any],
        weight_syncer: Any | None,
        sync_state_getter: Callable[[], dict[str, Any]] | None,
        weights_initialized: Callable[[], bool],
        set_weights_initialized: Callable[[bool], None],
    ) -> None:
        if not isinstance(collector, RolloutCollectorControl):
            raise TypeError(
                "rollout collector must implement activate_generation_runtime(), "
                "offload_generation_runtime_memory(), shutdown(), "
                "generation_runtime, and reward handoff capabilities",
            )
        self.collector = collector
        self.strategy = strategy
        self.training_state_getter = training_state_getter
        self.weight_syncer = weight_syncer
        self.sync_state_getter = sync_state_getter
        self._weights_initialized = weights_initialized
        self._set_weights_initialized = set_weights_initialized
        self._last_policy_version = self._runtime_policy_version(default=None)

    async def ensure_initial_weights(self, stats: RolloutStats) -> None:
        prepared = self.prepare_initial_weight_sync_state()
        if prepared is not None:
            await self.push_prepared_weights(prepared, stats)

    async def sync_weights_after_train(self, stats: RolloutStats) -> None:
        if self.weight_syncer is None:
            return
        prepared = self.prepare_weight_sync_state()
        await self.push_prepared_weights(prepared, stats)

    def prepare_weight_sync_state(self) -> dict[str, Any] | None:
        """Capture an immutable CPU policy snapshot on the caller's thread.

        Strategy export may execute DDP/FSDP collectives and must never be moved
        into a background rollout-owner loop.  The returned snapshot contains no
        live/GPU tensor aliases; ``push_prepared_weights`` performs no trainer read.
        """

        if self.weight_syncer is None:
            return None
        if self.sync_state_getter is None:
            raise RuntimeError("rollout weight sync requires sync_state_getter")
        prepared = self.sync_state_getter()
        if not isinstance(prepared, dict):
            raise TypeError("rollout weight sync getter must return a dict snapshot")
        _validate_prepared_weight_snapshot(prepared)
        return prepared

    def prepare_initial_weight_sync_state(self) -> dict[str, Any] | None:
        """Prepare first-policy weights only when the runtime still needs them."""

        if self._weights_initialized() or self.weight_syncer is None:
            return None
        return self.prepare_weight_sync_state()

    async def push_prepared_weights(
        self,
        prepared: dict[str, Any] | None,
        stats: RolloutStats,
    ) -> None:
        """Push a prepared snapshot without reading trainer-owned state."""

        if self.weight_syncer is None:
            return
        if prepared is None:
            raise ValueError("weight sync requires a prepared state snapshot")
        with stats.phase("rollout.weight_sync_s"):
            await self.weight_syncer.push(prepared)
        self._set_weights_initialized(True)
        fallback_version = (
            1 if self._last_policy_version is None else int(self._last_policy_version) + 1
        )
        self._last_policy_version = self._runtime_policy_version(
            default=fallback_version,
        )

    def current_policy_version(self) -> int | None:
        return self._runtime_policy_version(default=self._last_policy_version)

    def requires_driver_model_offload(self) -> bool:
        runtime = self._collector_generation_runtime()
        if runtime is None:
            return False
        return bool(runtime.requires_driver_model_offload)

    def requires_driver_model_offload_for_reward(self) -> bool:
        """Whether reward scoring borrows trainer-owned GPU capacity."""

        return bool(self.collector.requires_driver_model_offload_for_reward)

    def requires_training_state_parking(self) -> bool:
        """Whether either rollout or reward needs the trainer's GPU lease."""

        return bool(
            self.requires_driver_model_offload() or self.requires_driver_model_offload_for_reward()
        )

    def supports_non_draining_weight_sync(self) -> bool:
        # True only when every rollout worker retains versioned trainable-state
        # slots, so the weight-sync barrier can skip draining in-flight generation
        # (old requests keep their slot). getattr-with-default keeps any runtime
        # that does not advertise the capability on the safe draining barrier.
        runtime = self._collector_generation_runtime()
        return bool(getattr(runtime, "supports_non_draining_weight_sync", False))

    def validate_training_state_parking(self) -> None:
        if not self.requires_training_state_parking():
            return
        self.strategy.validate_training_state_parking()

    def park_training_state_for_rollout(self, stats: RolloutStats) -> bool:
        if not self.requires_training_state_parking():
            return False
        with stats.phase("rollout.offload_driver_s"):
            state = self.training_state_getter()
            self.strategy.park_training_state(state)
        return True

    def restore_training_state_after_rollout(self, stats: RolloutStats) -> None:
        with stats.phase("rollout.restore_driver_s"):
            self.strategy.restore_training_state(self.training_state_getter())

    @asynccontextmanager
    async def rollout_phase(self, stats: RolloutStats) -> AsyncIterator[None]:
        """Own the whole GPU handoff around one rollout (generate + score) phase.

        The schedule announces the phase; every ordering rule lives here, next
        to the transition primitives it sequences: park the trainer when the
        topology requires it, activate generation, run the body, then release
        rollout memory (including the reward's phase-final park) and restore
        the trainer ONLY after that release succeeded — a failed release
        leaves the trainer parked so terminal shutdown reclaims the rollout
        GPU before training state returns. A body failure plus a cleanup
        failure combine into :class:`RolloutPhaseCleanupError`.
        """

        parked = self.park_training_state_for_rollout(stats)
        phase_error: BaseException | None = None
        try:
            await self.activate_rollout_runtime(stats)
            yield
        except BaseException as error:
            phase_error = error

        cleanup_error: BaseException | None = None
        rollout_memory_released = False
        try:
            await self.offload_rollout_runtime_memory(stats)
            rollout_memory_released = True
        except BaseException as error:
            cleanup_error = error
        if parked and rollout_memory_released:
            try:
                self.restore_training_state_after_rollout(stats)
            except BaseException as error:
                cleanup_error = error

        if phase_error is not None:
            if cleanup_error is not None:
                raise RolloutPhaseCleanupError(phase_error, cleanup_error) from phase_error
            raise phase_error
        if cleanup_error is not None:
            raise cleanup_error

    async def activate_rollout_runtime(self, stats: RolloutStats) -> None:
        with stats.phase("rollout.activate_generation_runtime_s"):
            await self.collector.activate_generation_runtime()

    async def offload_rollout_runtime_memory(self, stats: RolloutStats) -> None:
        with stats.phase("rollout.offload_generation_runtime_s"):
            await self.collector.offload_generation_runtime_memory()

    async def shutdown_collector_runtime(self) -> None:
        """Park shared trainer state, then release the collector/runtime.

        A shared in-process reward may be asleep in a CuMem pool; its terminal
        shutdown wakes those pages before dropping the model, so the trainer
        must yield the physical GPU first — exactly as it does for a rollout
        phase. Parking is a no-op on disjoint topologies, so every schedule
        calls this unconditionally. The top-level strategy owner restores only
        after this shutdown proves every shared rollout/reward owner was
        released.
        """

        self.validate_training_state_parking()
        # Shutdown reports no timings, but parking stays unconditional; a
        # throwaway accumulator keeps the recording sites branch-free.
        self.park_training_state_for_rollout(RolloutStats())
        await self.collector.shutdown()

    def requires_generation_offload_before_reward(self) -> bool:
        return bool(self.collector.requires_generation_offload_before_reward)

    def _collector_generation_runtime(self) -> GenerationRuntime | None:
        # The collector's `generation_runtime` property raises RuntimeError before
        # set_generation_runtime() has run (the cross-node/continuous path queries
        # the policy version during setup, before the runtime is attached). This
        # means "no collector-runtime provider yet" — fall through to the
        # weight syncer rather than crashing.
        try:
            return self.collector.generation_runtime
        except RuntimeError:
            return None

    def _runtime_policy_version(self, *, default: int | None) -> int | None:
        # Ask the collector generation runtime, then the weight syncer, through
        # the version property each concrete boundary declares.
        for provider in (self._collector_generation_runtime(), self.weight_syncer):
            if provider is None:
                continue
            value = provider.current_policy_version
            if value is not None:
                return int(value)
        return default


def _validate_prepared_weight_snapshot(value: Any) -> None:
    if isinstance(value, torch.Tensor):
        if value.device.type != "cpu" or value.requires_grad:
            raise ValueError(
                "prepared rollout weights must be detached CPU tensors; "
                f"got device={value.device}, requires_grad={value.requires_grad}",
            )
        return
    if isinstance(value, Mapping):
        for child in value.values():
            _validate_prepared_weight_snapshot(child)
        return
    if isinstance(value, (list, tuple)):
        for child in value:
            _validate_prepared_weight_snapshot(child)


__all__ = [
    "RolloutCollectorControl",
    "RolloutPhaseCleanupError",
    "RolloutRuntimeCoordinator",
]
