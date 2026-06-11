"""Generic Ray actor-method runtime."""

from __future__ import annotations

import contextlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from vrl.ray.actor_group import RayActorGroup
from vrl.ray.dependencies import require_ray
from vrl.ray.lifecycle import kill_actors, remove_placement_group
from vrl.ray.placement import (
    actor_scheduling_strategy,
    create_placement_group,
    validate_actor_gpu_ids,
)


class _GpuReservationActor:
    def ping(self) -> bool:
        return True


@dataclass(slots=True)
class RayActorMethodRuntime:
    """Launch homogeneous Ray actors and call one actor method over payloads."""

    worker_cls: type[Any]
    worker_config: Mapping[str, Any]
    method_name: str
    worker_id_prefix: str
    num_workers: int = 1
    cpus_per_worker: float = 0.5
    gpus_per_worker: float = 0.0
    max_inflight_per_worker: int = 1
    startup_method: str | None = None
    init_ray: bool = True
    ray_init_kwargs: dict[str, Any] = field(default_factory=dict)
    release_after_call: bool = False
    placement_strategy: str = "SPREAD"
    expected_gpu_ids: tuple[int, ...] = ()
    gpu_reservation_count: int = 0
    validate_role: str = "actor"
    _actor_group: RayActorGroup | None = field(default=None, init=False, repr=False)
    _placement_group: Any | None = field(default=None, init=False, repr=False)
    _reservation_actors: list[Any] = field(default_factory=list, init=False, repr=False)

    async def map(self, payloads: Sequence[Any]) -> list[Any]:
        """Map configured actor method over payloads using the shared actor group."""

        if not payloads:
            return []
        actor_group = self._ensure_actor_group()
        try:
            return await actor_group.map_method(
                self.method_name,
                payloads,
                max_inflight_per_actor=self.max_inflight_per_worker,
            )
        finally:
            if self.release_after_call:
                await self.shutdown()

    async def shutdown(self) -> None:
        actor_group = self._actor_group
        self._actor_group = None
        if actor_group is not None:
            actor_group.shutdown()
        ray = require_ray()
        if self._reservation_actors:
            kill_actors(ray, self._reservation_actors)
            self._reservation_actors.clear()
        if self._placement_group is not None:
            remove_placement_group(self._placement_group)
            self._placement_group = None

    def _ensure_actor_group(self) -> RayActorGroup:
        if self._actor_group is not None:
            return self._actor_group
        if self.num_workers < 1:
            raise ValueError("RayActorMethodRuntime.num_workers must be >= 1")
        ray = require_ray()
        if self.init_ray and not ray.is_initialized():
            ray.init(**self.ray_init_kwargs)
        worker_ids = [f"{self.worker_id_prefix}-{index}" for index in range(self.num_workers)]
        worker_configs = [dict(self.worker_config) for _ in worker_ids]
        placement_group, bundle_indices = self._ensure_placement(ray)
        self._actor_group = RayActorGroup.launch(
            worker_cls=self.worker_cls,
            worker_configs=worker_configs,
            worker_ids=worker_ids,
            num_cpus=self.cpus_per_worker,
            num_gpus=self.gpus_per_worker,
            placement_group=placement_group,
            bundle_indices=bundle_indices,
            startup_method=self.startup_method,
        )
        if self.expected_gpu_ids and self.gpus_per_worker > 0:
            metadata = [
                {
                    "worker_id": handle.worker_id,
                    "node_ip": handle.node_ip,
                    "gpu_ids": handle.gpu_ids,
                }
                for handle in self._actor_group.handles
            ]
            validate_actor_gpu_ids(
                metadata,
                expected_gpu_ids=self.expected_gpu_ids,
                role=self.validate_role,
            )
        return self._actor_group

    def _ensure_placement(self, ray: Any) -> tuple[Any | None, list[int] | None]:
        if self.gpus_per_worker <= 0 or not self.expected_gpu_ids:
            return None, None
        if self._placement_group is not None:
            first_worker_bundle = int(self.gpu_reservation_count)
            return self._placement_group, [
                first_worker_bundle + index
                for index in range(self.num_workers)
            ]

        bundles: list[dict[str, float]] = []
        for _ in range(max(0, int(self.gpu_reservation_count))):
            bundles.append({"CPU": 0.001, "GPU": 1.0})
        worker_bundle_indices: list[int] = []
        for _ in range(self.num_workers):
            worker_bundle_indices.append(len(bundles))
            bundle = {"CPU": float(self.cpus_per_worker)}
            if self.gpus_per_worker > 0:
                bundle["GPU"] = float(self.gpus_per_worker)
            bundles.append(bundle)

        self._placement_group = create_placement_group(
            bundles,
            strategy=self.placement_strategy,
        )
        if self.gpu_reservation_count > 0:
            remote_reservation = ray.remote(num_cpus=0.001, num_gpus=1.0)(_GpuReservationActor)
            self._reservation_actors = [
                remote_reservation.options(
                    scheduling_strategy=actor_scheduling_strategy(
                        placement_group=self._placement_group,
                        bundle_index=index,
                        capture_child_tasks=True,
                    ),
                ).remote()
                for index in range(int(self.gpu_reservation_count))
            ]
            ray.get([actor.ping.remote() for actor in self._reservation_actors])
        return self._placement_group, worker_bundle_indices

    def __del__(self) -> None:
        with contextlib.suppress(Exception):
            actor_group = self._actor_group
            self._actor_group = None
            if actor_group is not None:
                actor_group.shutdown()
            ray = require_ray()
            if self._reservation_actors:
                kill_actors(ray, self._reservation_actors)
                self._reservation_actors.clear()
            if self._placement_group is not None:
                remove_placement_group(self._placement_group)
                self._placement_group = None


__all__ = ["RayActorMethodRuntime"]
