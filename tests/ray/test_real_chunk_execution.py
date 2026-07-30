"""Real-Ray twins for ``RayGenerationExecutor.execute``.

``execute`` had no real-Ray coverage anywhere in the repository: its three
consumers are production (``vrl/generation/ray/launcher.py``) and two in-process
fake actors (``tests/ray/test_chunk_dispatch.py``,
``tests/generation/ray/test_oom_split.py``). A fake actor is a direct call, so
``ChunkExecutionEnvelope`` / ``GenerationRequest`` / ``SampleChunk`` /
``ChunkExecutionResult`` were never pickled by any test — a field that turned
unserializable would pass everywhere and fail on production's first chunk.

These two run the same envelope over a real cluster, one per placement strategy.
They deliberately do NOT assert which worker got which chunk under ``dynamic``
(real scheduling decides that); the deterministic distribution contract stays in
the controlled-clock tests, which point here for the wire.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from vrl.generation.execution.chunk_placement import (
    ChunkPlacementPolicy,
    DistributedExecutionPlanner,
)
from vrl.generation.execution.types import (
    ChunkExecutionEnvelope,
    ChunkExecutionResult,
    DistributedWorkerHandle,
)
from vrl.generation.ray.executor import RayGenerationExecutor
from vrl.generation.types import GenerationOutput, GenerationRequest

pytestmark = pytest.mark.slow_test


class _ChunkWorker:
    """Real Ray actor running the production worker-side signature of a chunk."""

    def __init__(self, worker_id: str) -> None:
        self.worker_id = worker_id

    def execute_chunk(self, envelope: ChunkExecutionEnvelope) -> ChunkExecutionResult:
        # Asserted inside the actor process: the envelope arrived as its own type
        # with the request and chunk intact, i.e. it really survived pickling.
        assert isinstance(envelope, ChunkExecutionEnvelope), type(envelope).__name__
        assert isinstance(envelope.request, GenerationRequest)
        chunk = envelope.chunk
        return ChunkExecutionResult(
            request_id=envelope.request.request_id,
            worker_id=self.worker_id,
            chunk=chunk,
            output={"chunk_key": chunk.chunk_key, "samples": chunk.sample_count},
        )


class _ListGatherer:
    def gather_chunks(
        self,
        request: GenerationRequest,
        sample_rows: Any,
        chunks: list[Any],
    ) -> GenerationOutput:
        return GenerationOutput(
            request_id=request.request_id,
            family=request.family,
            task=request.task,
            sample_rows=list(sample_rows),
            output=list(chunks),
        )


def _request() -> GenerationRequest:
    return GenerationRequest(
        request_id="req-real-dispatch",
        family="sd3_5",
        task="t2i",
        inputs=["a test prompt"],
        samples_per_prompt=8,
        sampling={"height": 64, "width": 64, "num_steps": 10, "samples_per_chunk": 2},
        metadata={"dataset": "unit", "_runtime_debug": True},
    )


def _executor(ray: Any, strategy: str) -> tuple[RayGenerationExecutor, list[Any]]:
    actor_cls = ray.remote(num_cpus=0)(_ChunkWorker)
    actors = [actor_cls.remote(worker_id) for worker_id in ("w0", "w1")]
    workers = [
        DistributedWorkerHandle(worker_id=worker_id, actor=actor)
        for worker_id, actor in zip(("w0", "w1"), actors, strict=True)
    ]
    executor = RayGenerationExecutor(
        DistributedExecutionPlanner(policy=ChunkPlacementPolicy(strategy=strategy)),
        workers,
        _ListGatherer(),
    )
    return executor, actors


_EXPECTED_CHUNK_KEYS = [
    "prompt:0:samples:0:2",
    "prompt:0:samples:2:4",
    "prompt:0:samples:4:6",
    "prompt:0:samples:6:8",
]


def test_real_ray_executor_round_robin_gathers_every_chunk_in_order(local_ray) -> None:
    """Plan-time binding over a real wire: each of the four chunks comes back
    from the worker it was bound to, and the gather is still ordered by chunk."""

    executor, actors = _executor(local_ray, "round_robin")
    try:
        output = asyncio.run(executor.execute(_request()))

        assert [entry["chunk_key"] for entry in output.output] == _EXPECTED_CHUNK_KEYS
        schedule = output.extra["runtime_debug"]["chunk_schedule"]
        # Round-robin binds before dispatch, so real scheduling cannot move these.
        assert [row["assigned_worker"] for row in schedule] == ["w0", "w1", "w0", "w1"]
        assert [row["chunk_key"] for row in schedule] == _EXPECTED_CHUNK_KEYS
    finally:
        for actor in actors:
            local_ray.kill(actor, no_restart=True)


def test_real_ray_executor_dynamic_gathers_every_chunk_in_order(local_ray) -> None:
    """Pull dispatch over a real wire: which worker takes which chunk is up to
    the live cluster, but every chunk is executed exactly once and the gather
    order is still the plan's."""

    executor, actors = _executor(local_ray, "dynamic")
    try:
        output = asyncio.run(executor.execute(_request()))

        assert [entry["chunk_key"] for entry in output.output] == _EXPECTED_CHUNK_KEYS
        schedule = output.extra["runtime_debug"]["chunk_schedule"]
        assert sorted(row["chunk_key"] for row in schedule) == _EXPECTED_CHUNK_KEYS
        assert all(row["assignment_strategy"] == "dynamic" for row in schedule)
        # Every chunk landed on a real worker of this fleet -- no unbound rows.
        assert {row["assigned_worker"] for row in schedule} <= {"w0", "w1"}
    finally:
        for actor in actors:
            local_ray.kill(actor, no_restart=True)
