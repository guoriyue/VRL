"""Shared fakes for the Ray generation tests.

These three were character-identical copies across the session/lease/fsm/
weight-sync test modules; the parking snapshot keeps the superset signature
(the fixed-value copy is the ``residual_bytes=0`` default).
"""

from __future__ import annotations

from typing import Any

from vrl.generation.execution.types import WorkerMemoryParkingSnapshot
from vrl.generation.ray.engine import RayGenerationEngine
from vrl.ray.actor_group import RayActorHandle


def engine(worker_id: str, actor: Any) -> RayGenerationEngine:
    return RayGenerationEngine(
        worker_id,
        [RayActorHandle(worker_id=worker_id, actor=actor)],
    )


class ResolvedRef:
    """Awaitable that resolves to a value or raises it (a fake ObjectRef)."""

    def __init__(self, value: Any) -> None:
        self.value = value

    def __await__(self):
        async def resolve() -> Any:
            if isinstance(self.value, BaseException):
                raise self.value
            return self.value

        return resolve().__await__()


def parking_snapshot(
    worker_id: str = "rollout-0",
    *,
    residual_bytes: int = 0,
) -> WorkerMemoryParkingSnapshot:
    return WorkerMemoryParkingSnapshot(
        worker_id=worker_id,
        backend="cpu_offload",
        baseline_gpu_used_bytes=0,
        loaded_gpu_used_bytes=1024,
        residual_gpu_used_bytes=residual_bytes,
        residual_bytes_limit=0,
    )
