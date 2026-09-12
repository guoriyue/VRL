"""Ray generation config and driver-side validation."""

from __future__ import annotations

import inspect
import math
import os
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vrl.config.schema import RootConfig
    from vrl.models.interfaces.runtime import RuntimeBundle

from vrl.generation.execution.types import BatchPlacementStrategy
from vrl.ray.resources import (
    ResolvedDistributedResources,
)
from vrl.utils.config import to_builtin_deep
from vrl.utils.logging import init_logger
from vrl.utils.profiling import TorchProfilerConfig
from vrl.utils.validation import require_int

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
    # Allowed-set rejection is at the typed schema boundary (RolloutRuntimeSection);
    # DistributedExecutionPlanner repeats it for direct runtime construction.
    batch_placement_strategy: BatchPlacementStrategy
    # Plain on/off. True keeps rollout workers resynced to the trained policy (the
    # syncer flattens whatever is trainable — lora or full-param); False disables it.
    # Defaults ON: online runs train the policy the rollout workers must resync, so
    # an omitted value previously meant silent stale-policy training. The syncer is
    # only built on the online launch path, so this never affects eval.
    sync_trainable_state: bool
    update_weight_buffer_size: int | None = None

    def __post_init__(self) -> None:
        if self.update_weight_buffer_size is not None and (
            type(self.update_weight_buffer_size) is not int or self.update_weight_buffer_size < 1
        ):
            raise ValueError("update_weight_buffer_size must be a positive integer")
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

        from vrl.config.schema import RolloutRuntimeSection

        if not isinstance(section, RolloutRuntimeSection):
            section = RolloutRuntimeSection.model_validate(
                {} if section is None else to_builtin_deep(section)
            )
        return cls(**section.model_dump())


@dataclass(slots=True)
class RayGenerationConfig:
    """Ray launcher protocol composed from resources and one worker snapshot."""

    resources: ResolvedDistributedResources
    worker: RolloutWorkerConfig
    torch_profiler: TorchProfilerConfig | None = None

    def __post_init__(self) -> None:
        if self.resources.rollout_num_engines < 1:
            raise ValueError("distributed.resources.rollout.num_engines must be >= 1")
        if self.worker.pipelined and self.resources.rollout_num_engines != 1:
            raise ValueError(
                "distributed.rollout.pipelined=true requires exactly one rollout "
                f"engine; resolved {self.resources.rollout_num_engines}. "
                "Per-worker request pipelining "
                "is not implemented.",
            )

    @classmethod
    def from_root(
        cls,
        root: RootConfig,
        *,
        resources: ResolvedDistributedResources,
    ) -> RayGenerationConfig:
        """Build Ray execution settings using the run's resolved resources."""
        distributed = root.distributed
        rollout_runtime = distributed.rollout if distributed is not None else None
        profiler_section = root.rollout.torch_profiler if root.rollout is not None else None
        torch_profiler = None if profiler_section is None else replace(profiler_section)
        if torch_profiler is not None:
            output_dir = root.trainer.output_dir if root.trainer is not None else None
            if output_dir is not None:
                torch_profiler.output_dir = str(output_dir)

        return cls(
            resources=resources,
            worker=RolloutWorkerConfig.from_public_section(rollout_runtime),
            torch_profiler=torch_profiler,
        )

    def validate_driver_state(
        self,
        *,
        driver_bundle: RuntimeBundle,
    ) -> RayGenerationConfig:
        """Validate driver CUDA ownership before Ray rollout actors are launched."""

        # A model's primary device does not describe every training root.
        # Include both before checking the actual driver/rollout overlap.
        devices: set[int] = set()
        model_device = self._get_device(driver_bundle.model)
        if model_device is not None:
            index = self._cuda_device_index(model_device)
            if index is not None:
                devices.add(index)
        for device in self._iter_parameter_devices(driver_bundle.trainable_modules):
            index = self._cuda_device_index(device)
            if index is not None:
                devices.add(index)
        self._validate_driver_cuda_ownership(devices)
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
            # preflight (head --num-gpus=0) and require_actor_gpu_ids node check.
            return
        overlap = driver_cuda_devices & set(resources.rollout_devices)
        if not overlap:
            return

        overlap_list = sorted(overlap)
        rollout_devices = list(resources.rollout_devices)
        if not resources.colocated:
            raise ValueError(
                f"Trainer device cuda:{overlap_list[0]} overlaps rollout devices "
                f"{rollout_devices}, but the resolved plan expected disjoint "
                "trainer/rollout GPUs. Use CUDA_VISIBLE_DEVICES=0,1,2,3 with auto "
                "split for throughput, or set "
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

        # colocated already implies a non-empty rollout device set (it is the
        # trainer/rollout intersection), so no separate GPU-fleet check needed.
        if not (self.resources.colocated and self.resources.rollout_num_engines >= 1):
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

    @staticmethod
    def _get_device(obj: Any) -> Any | None:
        """Read an optional device without hiding errors from a declared property."""
        if obj is None:
            return None
        try:
            return obj.device
        except AttributeError:
            # A property may itself raise AttributeError. Only an absent declaration
            # permits discovery through the trainable modules instead.
            if inspect.getattr_static(obj, "device", None) is not None:
                raise
            return None

    @classmethod
    def _iter_parameter_devices(cls, obj: Any, seen: set[int] | None = None) -> Iterable[Any]:
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
                yield from cls._iter_parameter_devices(value, seen)
            return

        device = cls._get_device(obj)
        if device is not None:
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
                yield from cls._iter_parameter_devices(value, seen)

    @staticmethod
    def _cuda_device_index(device: Any) -> int | None:
        device_type = getattr(device, "type", None)
        if device_type is not None:
            if str(device_type).lower() != "cuda":
                return None
            index = getattr(device, "index", None)
            if index is not None:
                return require_int(index, path="CUDA device index", minimum=0)
        else:
            text = str(device).lower()
            if not text.startswith("cuda"):
                return None
            match = re.fullmatch(r"cuda(?::([0-9]+))?", text)
            if match is None:
                raise ValueError(
                    f"invalid CUDA device {device!r}; expected 'cuda' or 'cuda:<index>'"
                )
            if match.group(1) is not None:
                return int(match.group(1))

        # Unindexed CUDA means the current device, not ordinal zero. This function
        # runs at driver validation; parsing config still does not import Torch.
        import torch

        return torch.cuda.current_device()


__all__ = [
    "RayGenerationConfig",
    "RolloutWorkerConfig",
]
