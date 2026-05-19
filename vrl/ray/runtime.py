"""Generic Ray actor-method runtime."""

from __future__ import annotations

import contextlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from vrl.ray.actor_group import RayActorGroup
from vrl.ray.dependencies import require_ray


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
    _actor_group: RayActorGroup | None = field(default=None, init=False, repr=False)

    async def map(self, payloads: Sequence[Any]) -> list[Any]:
        """Map configured actor method over payloads using the shared actor group."""

        if not payloads:
            return []
        actor_group = self._ensure_actor_group()
        results = await actor_group.map_method(
            self.method_name,
            payloads,
            max_inflight_per_actor=self.max_inflight_per_worker,
        )
        if self.release_after_call:
            await self.shutdown()
        return results

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
        self._actor_group = RayActorGroup.launch(
            worker_cls=self.worker_cls,
            worker_configs=worker_configs,
            worker_ids=worker_ids,
            num_cpus=self.cpus_per_worker,
            num_gpus=self.gpus_per_worker,
            startup_method=self.startup_method,
        )
        return self._actor_group

    def __del__(self) -> None:
        with contextlib.suppress(Exception):
            actor_group = self._actor_group
            self._actor_group = None
            if actor_group is not None:
                actor_group.shutdown()


__all__ = ["RayActorMethodRuntime"]
