"""Ray generation config."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vrl.config.schema import RootConfig

from vrl.ray.resources import (
    ResolvedDistributedResources,
)
from vrl.utils.config import to_builtin_deep
from vrl.utils.profiling import TorchProfilerConfig


@dataclass(frozen=True, slots=True)
class RolloutWorkerConfig:
    """Frozen runtime projection of the public rollout-worker section."""

    cpus_per_worker: float
    worker_rpc_timeout_s: float
    generation_stall_timeout_s: float
    # Opt-in per-request rollout: each engine runs its round-robin share of a
    # request's batches in one RPC and stages the payloads for a finalizer.
    pipelined: bool

    def __post_init__(self) -> None:
        if not math.isfinite(self.cpus_per_worker) or self.cpus_per_worker <= 0:
            raise ValueError("cpus_per_worker must be finite and > 0")
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


__all__ = [
    "RayGenerationConfig",
    "RolloutWorkerConfig",
]
