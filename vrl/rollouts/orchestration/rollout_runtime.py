"""Runtime coordination shared by RL rollout schedules."""

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


class RolloutRuntimeCoordinator:
    """Coordinate runtime operations shared by strict and overlapped schedules."""

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

    def should_offload_driver_model_for_rollout(self) -> bool:
        return self.device.type == "cuda" and self.requires_driver_model_offload()

    def offload_driver_model_for_rollout(self, phase_times: dict[str, float]) -> bool:
        if not self.should_offload_driver_model_for_rollout():
            return False
        with record_phase(phase_times, "rollout.offload_driver_s"):
            # Park the driver model on CPU. WHERE is intentionally fixed: "cpu"
            # is the only meaningful off-GPU target, so this is not a config knob.
            # WHETHER we offload is decided upstream by GPU topology
            # (should_offload_driver_model_for_rollout -> colocation), never by a
            # model.memory setting. A future non-CPU offload target would belong
            # to distributed/topology config, not to a model backend policy.
            self.model.to("cpu")
            # nn.Module.to moves only registered submodules (the transformer);
            # diffusion families keep frozen VAE/text-encoders on the unregistered
            # pipeline, so park those too. getattr-guarded: AR families register
            # their VAE directly and expose no such hook.
            offload_frozen = getattr(self.model, "move_frozen_components", None)
            if callable(offload_frozen):
                offload_frozen("cpu")
            empty_cuda_cache()
        return True

    def restore_driver_model_after_rollout(self, phase_times: dict[str, float]) -> None:
        with record_phase(phase_times, "rollout.restore_driver_s"):
            self.model.to(self.device)
            restore_frozen = getattr(self.model, "move_frozen_components", None)
            if callable(restore_frozen):
                restore_frozen(self.device)
            empty_cuda_cache()

    async def activate_rollout_runtime(
        self,
        phase_times: dict[str, float],
    ) -> None:
        with record_phase(phase_times, "rollout.activate_runtime_s"):
            activate = getattr(self.collector, "activate_runtime", None)
            if callable(activate):
                await activate()

    async def offload_rollout_runtime_memory(
        self,
        phase_times: dict[str, float],
    ) -> None:
        with record_phase(phase_times, "rollout.offload_runtime_s"):
            offload = getattr(self.collector, "offload_runtime_memory", None)
            if callable(offload):
                await offload()
            empty_cuda_cache()

    def requires_runtime_offload_before_reward(self) -> bool:
        return bool(
            getattr(self.collector, "requires_runtime_offload_before_reward", False),
        )

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


__all__ = ["RolloutRuntimeCoordinator", "record_phase"]
