"""Runtime coordination shared by RL rollout schedules."""

from __future__ import annotations

import contextlib
import time
from collections.abc import Callable, Mapping
from typing import Any, Protocol, runtime_checkable

import torch


@runtime_checkable
class RolloutCollectorControl(Protocol):
    """Runtime controls a schedule must own for a complete phase handoff."""

    @property
    def runtime(self) -> Any: ...

    @property
    def requires_runtime_offload_before_reward(self) -> bool: ...

    @property
    def requires_driver_model_offload_for_reward(self) -> bool: ...

    async def activate_runtime(self) -> None: ...

    async def offload_runtime_memory(self) -> None: ...

    async def shutdown(self) -> None: ...


@contextlib.contextmanager
def record_phase(phase_times: dict[str, float], name: str):
    start = time.perf_counter()
    try:
        yield
    finally:
        phase_times[name] = phase_times.get(name, 0.0) + time.perf_counter() - start


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
                "rollout collector must implement activate_runtime(), "
                "offload_runtime_memory(), shutdown(), runtime, and reward "
                "handoff capabilities",
            )
        self.collector = collector
        self.strategy = strategy
        self.training_state_getter = training_state_getter
        self.weight_syncer = weight_syncer
        self.sync_state_getter = sync_state_getter
        self._weights_initialized = weights_initialized
        self._set_weights_initialized = set_weights_initialized
        self._last_policy_version = self._runtime_policy_version(default=None)

    async def ensure_initial_weights(self, phase_times: dict[str, float]) -> None:
        prepared = self.prepare_initial_weight_sync_state()
        if prepared is not None:
            await self.push_prepared_weights(prepared, phase_times)

    async def sync_weights_after_train(self, phase_times: dict[str, float]) -> int | None:
        if self.weight_syncer is None:
            return self.current_policy_version()
        prepared = self.prepare_weight_sync_state()
        return await self.push_prepared_weights(prepared, phase_times)

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
        phase_times: dict[str, float],
    ) -> int | None:
        """Push a prepared snapshot without reading trainer-owned state."""

        if self.weight_syncer is None:
            return self.current_policy_version()
        if prepared is None:
            raise ValueError("weight sync requires a prepared state snapshot")
        with record_phase(phase_times, "rollout.weight_sync_s"):
            await self.weight_syncer.push(prepared)
        self._set_weights_initialized(True)
        self._last_policy_version = self._runtime_policy_version(
            default=self._next_fallback_policy_version(),
        )
        return self._last_policy_version

    def current_policy_version(self) -> int | None:
        return self._runtime_policy_version(default=self._last_policy_version)

    def requires_driver_model_offload(self) -> bool:
        runtime = self._collector_runtime()
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

    def runtime_is_colocated(self) -> bool:
        # Ask the runtime (GenerationRuntime protocol) instead of probing its
        # config internals.
        runtime = self._collector_runtime()
        if runtime is None:
            return False
        return bool(runtime.is_colocated())

    def supports_non_draining_weight_sync(self) -> bool:
        # True only when every rollout worker retains versioned trainable-state
        # slots, so the weight-sync barrier can skip draining in-flight generation
        # (old requests keep their slot). getattr-with-default keeps any runtime
        # that does not advertise the capability on the safe draining barrier.
        runtime = self._collector_runtime()
        return bool(getattr(runtime, "supports_non_draining_weight_sync", False))

    def validate_training_state_parking(self) -> None:
        if not self.requires_training_state_parking():
            return
        self.strategy.validate_training_state_parking()

    def park_training_state_for_rollout(self, phase_times: dict[str, float]) -> bool:
        if not self.requires_training_state_parking():
            return False
        with record_phase(phase_times, "rollout.offload_driver_s"):
            state = self.training_state_getter()
            self.strategy.park_training_state(state)
        return True

    def restore_training_state_after_rollout(self, phase_times: dict[str, float]) -> None:
        with record_phase(phase_times, "rollout.restore_driver_s"):
            self.strategy.restore_training_state(self.training_state_getter())

    async def activate_rollout_runtime(
        self,
        phase_times: dict[str, float],
    ) -> None:
        with record_phase(phase_times, "rollout.activate_runtime_s"):
            await self.collector.activate_runtime()

    async def offload_rollout_runtime_memory(
        self,
        phase_times: dict[str, float],
    ) -> None:
        with record_phase(phase_times, "rollout.offload_runtime_s"):
            await self.collector.offload_runtime_memory()

    async def shutdown_collector_runtime(self) -> None:
        """Release the collector/runtime on the schedule's owning event loop."""

        await self.collector.shutdown()

    def requires_runtime_offload_before_reward(self) -> bool:
        return bool(self.collector.requires_runtime_offload_before_reward)

    def _collector_runtime(self) -> Any | None:
        # The collector's `runtime` property raises RuntimeError before
        # set_runtime() has run (the cross-node/continuous path queries the
        # policy version during setup, before the runtime is attached), and
        # AttributeError if the collector type exposes no runtime at all. Both
        # mean "no collector-runtime provider yet" — fall through to the weight
        # syncer rather than crashing.
        try:
            return self.collector.runtime
        except (AttributeError, RuntimeError):
            return None

    def _runtime_policy_version(self, *, default: int | None) -> int | None:
        # Ask each PolicyVersionProvider (collector runtime, then weight syncer)
        # through the protocol instead of probing their internal structure.
        for provider in (self._collector_runtime(), self.weight_syncer):
            if provider is None:
                continue
            value = provider.current_policy_version
            if value is not None:
                return int(value)
        return default

    def _next_fallback_policy_version(self) -> int:
        if self._last_policy_version is None:
            return 1
        return int(self._last_policy_version) + 1


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


__all__ = ["RolloutCollectorControl", "RolloutRuntimeCoordinator", "record_phase"]
