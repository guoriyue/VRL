"""Generic Ray actor-method runtime."""

from __future__ import annotations

import contextlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from vrl.ray.actor_group import RayActorGroup
from vrl.ray.dependencies import require_ray
from vrl.ray.placement import validate_actor_gpu_ids


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
    validate_role: str = "actor"
    # Owner-managed run-level placement. GPU actors schedule into these bundles;
    # the runtime never builds or removes the group, so release_after_call /
    # shutdown drop the actors only. CPU-only actors self-schedule as plain Ray
    # tasks and need no placement at all.
    placement_group: Any | None = None
    bundle_indices: tuple[int, ...] = ()
    _actor_group: RayActorGroup | None = field(default=None, init=False, repr=False)

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
                await self.release()

    async def release(self) -> None:
        """Drop the actor group (lease release); the next map() reacquires it.

        Same vocabulary as RayGenerationRuntime.release(). The owner-managed
        placement group is never touched here — only the actors are dropped.
        """
        await self.shutdown()

    async def shutdown(self) -> None:
        actor_group = self._actor_group
        self._actor_group = None
        if actor_group is not None:
            actor_group.shutdown()

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
        placement_group, bundle_indices = self._ensure_placement()
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

    def _ensure_placement(self) -> tuple[Any | None, list[int] | None]:
        if self.placement_group is not None:
            # Owner-managed run-level group: use its bundles, build nothing,
            # remove nothing.
            return self.placement_group, list(self.bundle_indices)
        if self.gpus_per_worker > 0:
            raise ValueError(
                "RayActorMethodRuntime needs placement for GPU actors: "
                "GPU workers schedule into the run-level placement group owned by "
                "a GlobalRayPlacementOwner (wired in vrl/scripts/common/online.py).",
            )
        # CPU-only actors self-schedule as plain Ray tasks; no placement group.
        return None, None

    def __del__(self) -> None:
        with contextlib.suppress(Exception):
            actor_group = self._actor_group
            self._actor_group = None
            if actor_group is not None:
                actor_group.shutdown()


__all__ = ["RayActorMethodRuntime"]
