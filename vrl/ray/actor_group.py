"""Generic Ray actor group construction and lifecycle."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from vrl.ray.dependencies import require_ray
from vrl.ray.operation_deadline import get_ray_refs, validate_ray_timeout
from vrl.ray.placement import actor_scheduling_strategy
from vrl.ray.resource_cleanup import kill_actors, kill_and_retain


@dataclass(frozen=True, slots=True)
class RayActorHandle:
    """Driver-visible metadata for one Ray actor."""

    worker_id: str
    actor: Any
    node_ip: str = "unknown"
    gpu_ids: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if not self.worker_id:
            raise ValueError("Ray actor handle worker_id must be non-empty")
        if self.actor is None:
            raise ValueError(f"Ray actor handle {self.worker_id!r} requires an actor")


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
        rpc_timeout_s: float,
        operation_prefix: str,
        placement_group: Any | None = None,
        bundle_indices: Sequence[int] | None = None,
        startup_method: str | None = None,
        concurrency_groups: Mapping[str, int] | None = None,
    ) -> RayActorGroup:
        """Launch actors for ``worker_cls`` using serializable worker configs."""

        if len(worker_configs) != len(worker_ids):
            raise ValueError("worker_configs and worker_ids must have the same length")
        if bundle_indices is not None and len(bundle_indices) != len(worker_ids):
            raise ValueError("bundle_indices and worker_ids must have the same length")
        rpc_timeout_s = validate_ray_timeout(
            rpc_timeout_s,
            name="rpc_timeout_s",
        )
        if not operation_prefix:
            raise ValueError("operation_prefix must not be empty")

        ray = require_ray()
        remote_options: dict[str, Any] = {
            "num_cpus": float(num_cpus),
            "num_gpus": float(num_gpus),
        }
        if concurrency_groups:
            remote_options["concurrency_groups"] = dict(concurrency_groups)
        remote_worker = ray.remote(**remote_options)(worker_cls)
        actors: list[Any] = []
        try:
            for index, (worker_id, worker_config) in enumerate(
                zip(worker_ids, worker_configs, strict=True)
            ):
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
                startup_refs = [getattr(actor, startup_method).remote() for actor in actors]
                get_ray_refs(
                    ray,
                    startup_refs,
                    operation=f"{operation_prefix}.startup.{startup_method}",
                    timeout_s=rpc_timeout_s,
                    context=f"workers={len(actors)}",
                )

            metadata_refs = [actor.worker_metadata.remote() for actor in actors]
            metadata = get_ray_refs(
                ray,
                metadata_refs,
                operation=f"{operation_prefix}.startup.worker_metadata",
                timeout_s=rpc_timeout_s,
                context=f"workers={len(actors)}",
            )
            handles = [
                RayActorHandle(
                    worker_id=worker_id,
                    actor=actor,
                    node_ip=str(_meta_get(meta, "node_ip", "unknown")),
                    gpu_ids=tuple(int(gpu_id) for gpu_id in _meta_get(meta, "gpu_ids", ())),
                )
                for worker_id, actor, meta in zip(
                    worker_ids,
                    actors,
                    metadata,
                    strict=True,
                )
            ]
        except BaseException as error:
            failures = kill_actors(ray, actors)
            if failures:
                error.add_note(
                    "Ray actor-group startup cleanup incomplete: "
                    f"{len(failures)} actor kill(s) failed",
                )
            raise

        return cls(handles=handles)

    def shutdown(self) -> None:
        """Best-effort actor shutdown."""

        ray = require_ray()
        surviving, failures = kill_and_retain(ray, self.handles, lambda handle: handle.actor)
        self.handles[:] = surviving
        if failures:
            raise RuntimeError(
                f"Ray actor-group cleanup incomplete: {len(failures)} actor kill(s) failed",
            ) from failures[0][1]


def _meta_get(meta: Any, key: str, default: Any) -> Any:
    if isinstance(meta, Mapping):
        return meta.get(key, default)
    return getattr(meta, key, default)


__all__ = ["RayActorGroup", "RayActorHandle"]
