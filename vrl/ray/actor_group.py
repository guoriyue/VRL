"""Generic Ray actor group construction and lifecycle."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from vrl.ray.dependencies import require_ray
from vrl.ray.resource_cleanup import kill_actors
from vrl.ray.placement import actor_scheduling_strategy


@dataclass(frozen=True, slots=True)
class RayActorHandle:
    """Driver-visible metadata for one Ray actor."""

    worker_id: str
    node_ip: str
    gpu_ids: tuple[int, ...] = ()
    actor: Any | None = None


@dataclass(slots=True)
class RayActorGroup:
    """A set of homogeneous Ray actors plus driver-visible metadata."""

    handles: list[RayActorHandle]

    @classmethod
    def launch(
        cls,
        *,
        worker_cls: type[Any],
        worker_configs: Sequence[Any],
        worker_ids: Sequence[str],
        num_cpus: float,
        num_gpus: float,
        placement_group: Any | None = None,
        bundle_indices: Sequence[int] | None = None,
        startup_method: str | None = None,
        metadata_method: str = "worker_metadata",
    ) -> RayActorGroup:
        """Launch actors for ``worker_cls`` using serializable worker configs."""

        if len(worker_configs) != len(worker_ids):
            raise ValueError("worker_configs and worker_ids must have the same length")
        if bundle_indices is not None and len(bundle_indices) != len(worker_ids):
            raise ValueError("bundle_indices and worker_ids must have the same length")

        ray = require_ray()
        remote_worker = ray.remote(num_cpus=float(num_cpus), num_gpus=float(num_gpus))(worker_cls)
        actors: list[Any] = []
        try:
            for index, (worker_id, worker_config) in enumerate(zip(worker_ids, worker_configs, strict=True)):
                options: dict[str, Any] = {}
                if placement_group is not None:
                    bundle_index = None if bundle_indices is None else int(bundle_indices[index])
                    options["scheduling_strategy"] = actor_scheduling_strategy(
                        placement_group=placement_group,
                        bundle_index=bundle_index,
                        capture_child_tasks=True,
                    )
                actor = remote_worker.options(**options).remote(worker_id, worker_config)
                actors.append(actor)

            if startup_method:
                startup_refs = [
                    getattr(actor, startup_method).remote()
                    for actor in actors
                ]
                ray.get(startup_refs)

            metadata_refs = [
                getattr(actor, metadata_method).remote()
                for actor in actors
            ]
            metadata = ray.get(metadata_refs)
        except Exception:
            kill_actors(ray, actors)
            raise

        handles = [
            RayActorHandle(
                worker_id=worker_id,
                node_ip=str(_meta_get(meta, "node_ip", "unknown")),
                gpu_ids=tuple(int(gpu_id) for gpu_id in _meta_get(meta, "gpu_ids", ())),
                actor=actor,
            )
            for worker_id, actor, meta in zip(worker_ids, actors, metadata, strict=True)
        ]
        return cls(handles=handles)

    def shutdown(self) -> None:
        """Best-effort actor shutdown."""

        ray = require_ray()
        actors = [handle.actor for handle in self.handles if handle.actor is not None]
        kill_actors(ray, actors)
        self.handles.clear()


def _meta_get(meta: Any, key: str, default: Any) -> Any:
    if isinstance(meta, Mapping):
        return meta.get(key, default)
    return getattr(meta, key, default)


__all__ = ["RayActorGroup", "RayActorHandle"]
