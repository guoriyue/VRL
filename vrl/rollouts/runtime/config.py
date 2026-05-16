"""Backend configuration for rollout collection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from vrl.distributed.resources import (
    ResolvedDistributedResources,
    resolve_distributed_resources,
)


@dataclass(slots=True)
class RolloutBackendConfig:
    """Ray rollout execution config plus resolved worker resource shape."""

    backend: str = "ray"
    num_workers: int = 1
    gpus_per_worker: float = 1.0
    cpus_per_worker: float = 1.0
    placement_strategy: str = "SPREAD"
    allow_driver_gpu_overlap: bool = False
    max_inflight_chunks_per_worker: int = 1
    sync_trainable_state: str = "disabled"
    release_after_collect: bool = False
    resources: ResolvedDistributedResources | None = None

    def __post_init__(self) -> None:
        if self.backend != "ray":
            raise ValueError("backend must be 'ray'")
        if self.num_workers < 1:
            raise ValueError("num_workers must be >= 1")
        if self.gpus_per_worker < 0:
            raise ValueError("gpus_per_worker must be >= 0")
        if self.cpus_per_worker <= 0:
            raise ValueError("cpus_per_worker must be > 0")
        if not self.placement_strategy:
            raise ValueError("placement_strategy must be non-empty")
        if self.max_inflight_chunks_per_worker < 1:
            raise ValueError("max_inflight_chunks_per_worker must be >= 1")
        if self.sync_trainable_state not in {"disabled", "lora_only"}:
            raise ValueError("sync_trainable_state must be 'disabled' or 'lora_only'")

    @classmethod
    def from_cfg(cls, cfg: Any) -> RolloutBackendConfig:
        """Build rollout backend config from a full training cfg."""
        if isinstance(cfg, cls):
            return cfg

        distributed = _config_get(cfg, "distributed", _MISSING)
        if distributed is _MISSING:
            raise ValueError("distributed.backend is required")

        backend = _config_get(distributed, "backend", _MISSING)
        if backend is _MISSING:
            raise ValueError("distributed.backend is required")

        resources_node = _config_get(distributed, "resources", _MISSING)
        if resources_node is _MISSING:
            raise ValueError("distributed.resources is required")
        resources = resolve_distributed_resources(cfg)
        rollout = _config_get(distributed, "rollout", {})

        return cls(
            backend=str(backend),
            num_workers=resources.rollout_num_workers,
            gpus_per_worker=resources.rollout_gpus_per_worker,
            cpus_per_worker=float(
                _config_get(rollout, "cpus_per_worker", 1.0),
            ),
            placement_strategy=str(
                _config_get(rollout, "placement_strategy", "SPREAD"),
            ),
            allow_driver_gpu_overlap=bool(resources.colocated),
            max_inflight_chunks_per_worker=int(
                _config_get(
                    rollout,
                    "max_inflight_chunks_per_worker",
                    1,
                ),
            ),
            sync_trainable_state=str(
                _config_get(rollout, "sync_trainable_state", "disabled"),
            ),
            release_after_collect=bool(
                _config_get(rollout, "release_after_collect", False),
            ),
            resources=resources,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the flat config shape accepted by rollout backends."""
        return {
            "backend": self.backend,
            "num_workers": self.num_workers,
            "gpus_per_worker": self.gpus_per_worker,
            "cpus_per_worker": self.cpus_per_worker,
            "placement_strategy": self.placement_strategy,
            "allow_driver_gpu_overlap": self.allow_driver_gpu_overlap,
            "max_inflight_chunks_per_worker": self.max_inflight_chunks_per_worker,
            "sync_trainable_state": self.sync_trainable_state,
            "release_after_collect": self.release_after_collect,
        }


_MISSING = object()


def _config_get(node: Any, key: str, default: Any) -> Any:
    if node is None:
        return default
    getter = getattr(node, "get", None)
    if callable(getter):
        try:
            return getter(key, default)
        except TypeError:
            pass
    try:
        return node[key]
    except (KeyError, IndexError, TypeError):
        pass
    return getattr(node, key, default)


__all__ = ["RolloutBackendConfig"]
