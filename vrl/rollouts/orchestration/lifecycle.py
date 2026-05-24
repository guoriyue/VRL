"""Runtime lifecycle helpers for RL rollout schedules."""

from __future__ import annotations

import contextlib
import time
from collections.abc import Callable
from typing import Any

import torch
import torch.nn as nn

from vrl.utils.cuda_memory import empty_cuda_cache


@contextlib.contextmanager
def record_phase(phase_times: dict[str, float], name: str):
    start = time.perf_counter()
    try:
        yield
    finally:
        phase_times[name] = phase_times.get(name, 0.0) + time.perf_counter() - start


class RolloutLifecycle:
    """Lifecycle operations shared by strict and overlapped rollout schedules."""

    def __init__(
        self,
        *,
        collector: Any,
        model: nn.Module,
        device: torch.device,
        weight_syncer: Any | None,
        sync_state_getter: Callable[[], dict[str, Any]] | None,
        weights_initialized: Callable[[], bool],
        set_weights_initialized: Callable[[bool], None],
    ) -> None:
        self.collector = collector
        self.model = model
        self.device = device
        self.weight_syncer = weight_syncer
        self.sync_state_getter = sync_state_getter
        self._weights_initialized = weights_initialized
        self._set_weights_initialized = set_weights_initialized
        self._last_policy_version = self._runtime_policy_version(default=None)

    async def ensure_initial_weights(self, phase_times: dict[str, float]) -> None:
        if self._weights_initialized() or self.weight_syncer is None:
            return
        if self.sync_state_getter is None:
            raise RuntimeError("rollout weight sync requires sync_state_getter")
        with record_phase(phase_times, "rollout.weight_sync_s"):
            await self.weight_syncer.push(self.sync_state_getter())
        self._set_weights_initialized(True)
        self._last_policy_version = self._runtime_policy_version(
            default=self._next_fallback_policy_version(),
        )

    async def sync_weights_after_train(self, phase_times: dict[str, float]) -> int | None:
        if self.weight_syncer is None:
            return self.current_policy_version()
        if self.sync_state_getter is None:
            raise RuntimeError("rollout weight sync requires sync_state_getter")
        with record_phase(phase_times, "rollout.weight_sync_s"):
            await self.weight_syncer.push(self.sync_state_getter())
        self._last_policy_version = self._runtime_policy_version(
            default=self._next_fallback_policy_version(),
        )
        return self._last_policy_version

    def current_policy_version(self) -> int | None:
        return self._runtime_policy_version(default=self._last_policy_version)

    def requires_driver_model_offload(self) -> bool:
        runtime = self._collector_runtime()
        return bool(getattr(runtime, "requires_driver_model_offload", False))

    def runtime_is_colocated(self) -> bool:
        runtime = self._collector_runtime()
        if runtime is None:
            return False
        config = getattr(runtime, "config", None)
        if bool(getattr(config, "allow_driver_gpu_overlap", False)):
            return True
        resources = getattr(config, "resources", None)
        return bool(getattr(resources, "colocated", False))

    def should_offload_driver_model_for_rollout(self) -> bool:
        return self.device.type == "cuda" and self.requires_driver_model_offload()

    def offload_driver_model_for_rollout(self, phase_times: dict[str, float]) -> bool:
        if not self.should_offload_driver_model_for_rollout():
            return False
        with record_phase(phase_times, "rollout.offload_driver_s"):
            self.model.to("cpu")
            empty_cuda_cache()
        return True

    def restore_driver_model_after_rollout(self, phase_times: dict[str, float]) -> None:
        with record_phase(phase_times, "rollout.restore_driver_s"):
            self.model.to(self.device)
            empty_cuda_cache()

    async def release_rollout_runtime_memory(
        self,
        phase_times: dict[str, float],
    ) -> None:
        with record_phase(phase_times, "rollout.release_runtime_s"):
            release = getattr(self.collector, "release_runtime_memory", None)
            if callable(release):
                await release()
            empty_cuda_cache()

    def _collector_runtime(self) -> Any | None:
        try:
            return self.collector.runtime
        except Exception:
            return getattr(self.collector, "_runtime", None)

    def _runtime_policy_version(self, *, default: int | None) -> int | None:
        runtime = self._collector_runtime()
        value = getattr(runtime, "current_policy_version", None)
        if value is None:
            sync_runtime = getattr(self.weight_syncer, "runtime", None)
            value = getattr(sync_runtime, "current_policy_version", None)
        if value is None:
            return default
        return int(value)

    def _next_fallback_policy_version(self) -> int:
        if self._last_policy_version is None:
            return 1
        return int(self._last_policy_version) + 1


__all__ = ["RolloutLifecycle", "record_phase"]
