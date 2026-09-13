"""Shared fakes for the Ray generation tests.

These three were character-identical copies across the session/lease/fsm/
weight-sync test modules; the parking snapshot keeps the superset signature
(the fixed-value copy is the ``residual_bytes=0`` default).
"""

from __future__ import annotations

import asyncio
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


class NeverRef:
    """A fake ObjectRef for a call that never completes."""

    def __await__(self):
        async def wait_forever() -> None:
            await asyncio.Event().wait()

        return wait_forever().__await__()


class GatedRef:
    """A fake ObjectRef that completes only once ``gate`` is set.

    Resolves to ``value``, or raises it when it is an exception, which is the
    same convention ``ResolvedRef`` uses.
    """

    def __init__(self, gate: asyncio.Event, value: Any) -> None:
        self.gate = gate
        self.value = value

    def __await__(self):
        async def wait() -> Any:
            await self.gate.wait()
            if isinstance(self.value, BaseException):
                raise self.value
            return self.value

        return wait().__await__()


class _FakeRemoteMethod:
    """One synchronous method wearing Ray's ``.remote()`` submission face."""

    def __init__(self, call: Any) -> None:
        self._call = call

    def remote(self, *args: Any, **kwargs: Any) -> ResolvedRef:
        # A real ObjectRef surfaces the worker's exception on await, not on
        # submit, so a raising double must behave the same way here.
        try:
            return ResolvedRef(self._call(*args, **kwargs))
        except BaseException as error:  # re-raised when the ref is awaited
            return ResolvedRef(error)


class FakeRayActor:
    """A synchronous test worker wearing the Ray actor method face.

    Production submits every engine call as ``actor.<method>.remote(...)`` and
    awaits the returned ref. A plain object has no such face, which is what the
    executor's per-dispatch-site "else: call it directly" branches used to
    accommodate -- a production branch that only test doubles could reach.
    Wearing the face here instead keeps the production path single. Attributes
    that are not part of the face fall through to the worker, so tests keep
    asserting on the state it records.
    """

    def __init__(self, worker: Any, *methods: str) -> None:
        self._worker = worker
        self._remote = {name: _FakeRemoteMethod(getattr(worker, name)) for name in methods}

    def __getattr__(self, name: str) -> Any:
        remote = self.__dict__["_remote"]
        if name in remote:
            return remote[name]
        return getattr(self.__dict__["_worker"], name)


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
