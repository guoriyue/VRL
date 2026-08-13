"""Ray generation config and driver-side validation."""

from __future__ import annotations

import math
import os
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from typing import Any

from vrl.generation.execution.types import BatchPlacementStrategy
from vrl.ray.resources import (
    ResolvedDistributedResources,
)
from vrl.utils.config import cfg_get, plain_mapping, to_builtin_deep
from vrl.utils.logging import init_logger
from vrl.utils.profiling import TorchProfilerConfig

logger = init_logger(__name__)


@dataclass(frozen=True, slots=True)
class RolloutWorkerConfig:
    """Frozen runtime projection of the public rollout-worker section."""

    cpus_per_worker: float
    health_check_interval_s: float
    health_check_timeout_s: float
    health_check_first_wait_s: float
    worker_rpc_timeout_s: float
    generation_stall_timeout_s: float
    # Opt-in single-worker pipelined rollout. Multi-worker execution is rejected
    # because per-worker request partitioning is not implemented.
    pipelined: bool
    # Batch->worker binding: "round_robin" binds at plan time (baseline);
    # "dynamic" binds at dispatch time (pull + LPT). Equivalent for 1 worker.
    # Allowed-set rejection is at the typed schema boundary (RolloutWorkerSection);
    # DistributedExecutionPlanner repeats it for direct runtime construction.
    batch_placement_strategy: BatchPlacementStrategy
    # Plain on/off. True keeps rollout workers resynced to the trained policy (the
    # syncer flattens whatever is trainable — lora or full-param); False disables it.
    # Defaults ON: online runs train the policy the rollout workers must resync, so
    # an omitted value previously meant silent stale-policy training. The syncer is
    # only built on the online launch path, so this never affects eval.
    sync_trainable_state: bool

    def __post_init__(self) -> None:
        if not math.isfinite(self.cpus_per_worker) or self.cpus_per_worker <= 0:
            raise ValueError("cpus_per_worker must be finite and > 0")
        if not math.isfinite(self.health_check_interval_s):
            raise ValueError("health_check_interval_s must be finite")
        if self.health_check_interval_s > 0 and (
            not math.isfinite(self.health_check_timeout_s) or self.health_check_timeout_s <= 0
        ):
            raise ValueError("health_check_timeout_s must be finite and > 0 when enabled")
        if not math.isfinite(self.health_check_first_wait_s) or self.health_check_first_wait_s < 0:
            raise ValueError("health_check_first_wait_s must be finite and >= 0")
        if not math.isfinite(self.worker_rpc_timeout_s) or self.worker_rpc_timeout_s <= 0:
            raise ValueError("worker_rpc_timeout_s must be finite and > 0")
        if (
            not math.isfinite(self.generation_stall_timeout_s)
            or self.generation_stall_timeout_s <= 0
        ):
            raise ValueError("generation_stall_timeout_s must be finite and > 0")

    @classmethod
    def from_public_section(cls, section: Any) -> RolloutWorkerConfig:
        """Freeze a validated public section without introducing fallback values."""

        from vrl.config.schema import RolloutWorkerSection

        if not isinstance(section, RolloutWorkerSection):
            section = RolloutWorkerSection.model_validate(to_builtin_deep(section or {}))
        return cls(**section.model_dump())


@dataclass(slots=True)
class RayGenerationConfig:
    """Ray launcher protocol composed from resources and one worker snapshot."""

    resources: ResolvedDistributedResources
    worker: RolloutWorkerConfig
    torch_profiler: TorchProfilerConfig | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.worker, RolloutWorkerConfig):
            raise TypeError(
                f"worker must be a RolloutWorkerConfig, got {type(self.worker).__name__}",
            )
        if self.torch_profiler is not None and not isinstance(
            self.torch_profiler,
            TorchProfilerConfig,
        ):
            raise TypeError(
                "torch_profiler must be a TorchProfilerConfig or None, "
                f"got {type(self.torch_profiler).__name__}",
            )
        if self.resources.rollout_num_workers < 1:
            raise ValueError("distributed.resources.rollout.num_workers must be >= 1")
        if self.resources.rollout_gpus_per_worker < 0:
            raise ValueError("distributed.resources.rollout.gpus_per_worker must be >= 0")
        if self.worker.pipelined and self.resources.rollout_num_workers != 1:
            raise ValueError(
                "distributed.rollout.pipelined=true requires exactly one rollout "
                f"worker; resolved {self.resources.rollout_num_workers}. "
                "Per-worker request pipelining "
                "is not implemented.",
            )

    @classmethod
    def from_cfg(
        cls,
        cfg: Any,
        *,
        resources: ResolvedDistributedResources,
    ) -> RayGenerationConfig:
        """Build Ray execution settings using the run's resolved resources."""
        distributed = cfg_get(cfg, "distributed", {})
        rollout = cfg_get(distributed, "rollout", {})
        public_rollout = cfg_get(cfg, "rollout", {})
        profiler_section = cfg_get(public_rollout, "torch_profiler", None)
        if profiler_section is None:
            torch_profiler = None
        elif isinstance(profiler_section, TorchProfilerConfig):
            torch_profiler = replace(profiler_section)
        else:
            torch_profiler = TorchProfilerConfig(
                **plain_mapping(
                    profiler_section,
                    field_name="rollout.torch_profiler",
                ),
            )
        if torch_profiler is not None:
            output_dir = cfg_get(cfg_get(cfg, "trainer", {}), "output_dir", None)
            if output_dir is not None:
                torch_profiler.output_dir = str(output_dir)

        return cls(
            resources=resources,
            worker=RolloutWorkerConfig.from_public_section(rollout),
            torch_profiler=torch_profiler,
        )

    def validate_driver_state(
        self,
        *,
        driver_bundle: Any,
    ) -> RayGenerationConfig:
        """Validate driver CUDA ownership before Ray rollout actors are launched."""

        self._validate_driver_cuda_ownership(_driver_cuda_devices(driver_bundle))
        self._validate_colocated_replay_memory(driver_bundle)
        return self

    def _validate_driver_cuda_ownership(self, driver_cuda_devices: set[int]) -> None:
        if not driver_cuda_devices:
            return

        resources = self.resources
        if resources.cross_node:
            # Cross-node: the driver's head-local cuda ordinal and a remote rollout
            # GPU live in different ordinal spaces, so a set-intersection overlap
            # check is meaningless. Node-level isolation is enforced by the launcher
            # preflight (head --num-gpus=0) and validate_actor_gpu_ids node check.
            return
        overlap = driver_cuda_devices & set(resources.rollout_devices)
        if not overlap:
            return

        overlap_list = sorted(overlap)
        rollout_devices = list(resources.rollout_devices)
        if not resources.colocated:
            raise ValueError(
                f"Trainer device cuda:{overlap_list[0]} overlaps rollout devices "
                f"{rollout_devices}, but resources.allow_overlap=false. "
                "Use CUDA_VISIBLE_DEVICES=0,1,2,3 with auto split for throughput, or set "
                "distributed.resources.rollout.gpu_pool=trainer for time-shared colocation.",
            )

        # Keep a runtime-boundary backstop in addition to resource resolution: an
        # overlapping driver/rollout GPU is safe only when phases hand it over.
        if resources.lifecycle.rollout_mode != "on_demand":
            raise ValueError(
                f"Trainer device cuda:{overlap_list[0]} overlaps rollout devices "
                f"{rollout_devices}, but the resolved rollout lifecycle is not on_demand. "
                "Shared trainer/rollout GPUs must hand ownership over between phases.",
            )

    def _validate_colocated_replay_memory(self, bundle: Any) -> None:
        """Warn or fail when trainer and Ray worker both own full generation state.

        This guard does not implement a family-specific minimal replay loader. It
        makes the risk explicit and provides a strict mode for CI or future recipes
        once those loaders exist.
        """

        if not (
            self.resources.colocated
            and self.resources.rollout_num_workers >= 1
            and self.resources.rollout_gpus_per_worker > 0
        ):
            return
        if not bundle.loads_full_generation_modules:
            return

        message = (
            "trainer bundle declares loads_full_generation_modules=true while "
            "colocated Ray rollout is enabled; host RAM can contain the trainer "
            "generation model plus a rollout worker generation model. Implement a "
            "family-specific minimal replay loader before enabling strict guard."
        )
        strict = os.environ.get("VRL_STRICT_REPLAY_MEMORY_GUARD", "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        if strict:
            raise ValueError(message)
        logger.warning(message)


def _driver_cuda_devices(driver_bundle: Any) -> set[int]:
    has_policy_device, device = _get_device(getattr(driver_bundle, "model", None))
    if has_policy_device:
        parsed = _cuda_device_index(device)
        return set() if parsed is None else {parsed}

    devices: set[int] = set()
    for parameter_device in _iter_parameter_devices(
        getattr(driver_bundle, "trainable_modules", None),
    ):
        parsed = _cuda_device_index(parameter_device)
        if parsed is not None:
            devices.add(parsed)
    return devices


def _get_device(obj: Any) -> tuple[bool, Any]:
    if obj is None:
        return False, None
    try:
        device = obj.device
    except Exception:
        return False, None
    if device is None:
        return False, None
    return True, device


def _iter_parameter_devices(obj: Any, seen: set[int] | None = None) -> Iterable[Any]:
    if obj is None or isinstance(obj, (str, bytes)):
        return
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return
    seen.add(obj_id)

    if isinstance(obj, Mapping):
        for value in obj.values():
            yield from _iter_parameter_devices(value, seen)
        return

    has_device, device = _get_device(obj)
    if has_device:
        yield device
        return

    parameters = getattr(obj, "parameters", None)
    if callable(parameters):
        for parameter in parameters():
            device = getattr(parameter, "device", None)
            if device is not None:
                yield device
        return

    if isinstance(obj, Iterable):
        for value in obj:
            yield from _iter_parameter_devices(value, seen)


def _cuda_device_index(device: Any) -> int | None:
    device_type = getattr(device, "type", None)
    if device_type is not None:
        if str(device_type).lower() != "cuda":
            return None
        index = getattr(device, "index", None)
        return 0 if index is None else int(index)

    text = str(device).lower()
    if not text.startswith("cuda"):
        return None
    if ":" not in text:
        return 0
    _, raw_index = text.split(":", 1)
    try:
        return int(raw_index)
    except ValueError:
        return 0


__all__ = [
    "RayGenerationConfig",
    "RolloutWorkerConfig",
]
