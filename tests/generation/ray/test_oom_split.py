"""Chunk OOM degradation: split-on-OOM in the Ray generation executor."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from vrl.generation.execution.chunk_placement import (
    DeviceAssignment,
    DistributedGenerationPlan,
)
from vrl.generation.execution.chunks import SampleChunk
from vrl.generation.execution.planner import build_engine_plan
from vrl.generation.execution.types import (
    ChunkExecutionEnvelope,
    ChunkExecutionResult,
    DistributedWorkerHandle,
    PipelinedRequestOutOfMemory,
    StaleSlotDiscard,
)
from vrl.generation.ray.executor import RayGenerationExecutor, _is_oom_error
from vrl.generation.types import GenerationOutput, GenerationRequest

_OOM_MESSAGE = "CUDA out of memory. Tried to allocate 4.00 GiB"


def _key(start: int, count: int) -> str:
    """Derive a chunk_key from the source template, not a hand-copied f-string."""

    return SampleChunk(
        prompt_index=0, prompt="p", sample_start=start, sample_count=count
    ).chunk_key


@dataclass
class _CapacityWorker:
    """Synchronous worker that OOMs on chunks above ``max_samples``."""

    worker_id: str
    max_samples: int
    executed: list[str] = field(default_factory=list)
    fail_message: str = _OOM_MESSAGE
    request_id_override: str | None = None

    def execute_chunk(self, envelope: ChunkExecutionEnvelope) -> ChunkExecutionResult:
        chunk = envelope.chunk
        self.executed.append(chunk.chunk_key)
        request_id = (
            envelope.request.request_id
            if self.request_id_override is None
            else self.request_id_override
        )
        if chunk.sample_count > self.max_samples:
            return ChunkExecutionResult(
                request_id=request_id,
                worker_id=self.worker_id,
                chunk=chunk,
                output=None,
                error=self.fail_message,
            )
        return ChunkExecutionResult(
            request_id=request_id,
            worker_id=self.worker_id,
            chunk=chunk,
            output={"chunk_key": chunk.chunk_key, "samples": chunk.sample_count},
        )


@dataclass
class _StaticPlanner:
    """Plan one static assignment per provided chunk."""

    chunks: list[SampleChunk]

    @property
    def policy(self) -> Any:
        @dataclass
        class _Policy:
            strategy: str = "static"

        return _Policy()

    def plan_with_engine(
        self,
        request: GenerationRequest,
        workers: list[DistributedWorkerHandle],
    ) -> DistributedGenerationPlan:
        assignments = tuple(
            DeviceAssignment(
                worker_id=workers[index % len(workers)].worker_id,
                envelope=ChunkExecutionEnvelope(
                    request=request,
                    chunk=chunk,
                ),
            )
            for index, chunk in enumerate(self.chunks)
        )
        engine_plan = build_engine_plan(request)
        return DistributedGenerationPlan(engine_plan=engine_plan, assignments=assignments)


class _CoverageGatherer:
    """Assert sample coverage by chunk metadata, mirroring layout.ordered_chunks."""

    def gather_chunks(
        self,
        request: GenerationRequest,
        sample_rows: Any,
        chunks: list[dict[str, Any]],
    ) -> GenerationOutput:
        return GenerationOutput(
            request_id=request.request_id,
            family=request.family,
            task=request.task,
            sample_rows=list(sample_rows),
            output=list(chunks),
        )


def _request(
    num_samples: int,
    *,
    samples_per_chunk: int | None = None,
    runtime_debug: bool = False,
) -> GenerationRequest:
    return GenerationRequest(
        request_id="req-oom",
        family="test",
        task="t2i",
        inputs=["p"],
        samples_per_prompt=num_samples,
        sampling=({} if samples_per_chunk is None else {"samples_per_chunk": samples_per_chunk}),
        metadata={"_runtime_debug": True} if runtime_debug else None,
    )


def _executor(
    chunks: list[SampleChunk],
    workers: list[_CapacityWorker],
) -> tuple[RayGenerationExecutor, list[DistributedWorkerHandle]]:
    handles = [
        DistributedWorkerHandle(worker_id=worker.worker_id, actor=worker) for worker in workers
    ]
    executor = RayGenerationExecutor(
        planner=_StaticPlanner(chunks=chunks),
        workers=handles,
        gatherer=_CoverageGatherer(),
    )
    return executor, handles


@pytest.mark.asyncio
async def test_oom_chunk_splits_until_it_fits() -> None:
    """An 8-sample chunk on a 2-sample worker degrades to four 2-sample chunks."""

    chunk = SampleChunk(prompt_index=0, prompt="p", sample_start=0, sample_count=8)
    worker = _CapacityWorker(worker_id="w0", max_samples=2)
    executor, _ = _executor([chunk], [worker])

    output = await executor.execute(_request(8, runtime_debug=True))

    covered = sorted((entry["chunk_key"], entry["samples"]) for entry in output.output)
    assert covered == [
        (_key(0, 2), 2),
        (_key(2, 2), 2),
        (_key(4, 2), 2),
        (_key(6, 2), 2),
    ]
    splits = output.extra["runtime_debug"]["chunk_oom_splits"]
    # Recursion order: 8 -> [0:4] + [4:8] -> 2-sample leaves.
    assert [row["chunk_key"] for row in splits] == [_key(0, 8), _key(0, 4), _key(4, 4)]
    assert all(row["worker_id"] == "w0" for row in splits)


@pytest.mark.asyncio
async def test_single_sample_oom_still_raises() -> None:
    """A chunk that OOMs at one sample is a hard failure, not an infinite loop."""

    chunk = SampleChunk(prompt_index=0, prompt="p", sample_start=0, sample_count=4)
    worker = _CapacityWorker(worker_id="w0", max_samples=0)
    executor, _ = _executor([chunk], [worker])

    with pytest.raises(RuntimeError, match="out of memory"):
        await executor.execute(_request(4))


@pytest.mark.asyncio
async def test_non_oom_error_is_not_retried() -> None:
    """Only allocator failures degrade; other worker errors fail fast."""

    chunk = SampleChunk(prompt_index=0, prompt="p", sample_start=0, sample_count=4)
    worker = _CapacityWorker(
        worker_id="w0",
        max_samples=2,
        fail_message="ValueError: bad scheduler state",
    )
    executor, _ = _executor([chunk], [worker])

    with pytest.raises(RuntimeError, match="bad scheduler state"):
        await executor.execute(_request(4))
    assert worker.executed == [chunk.chunk_key]


@pytest.mark.asyncio
async def test_healthy_chunks_skip_degradation_path() -> None:
    """No OOM: results and telemetry are exactly the pre-split behavior."""

    chunks = [
        SampleChunk(prompt_index=0, prompt="p", sample_start=0, sample_count=2),
        SampleChunk(prompt_index=0, prompt="p", sample_start=2, sample_count=2),
    ]
    worker = _CapacityWorker(worker_id="w0", max_samples=2)
    executor, _ = _executor(chunks, [worker])

    output = await executor.execute(_request(4))

    assert len(output.output) == 2
    assert "runtime_debug" not in output.extra


@pytest.mark.asyncio
async def test_result_request_id_must_match_submitted_envelope() -> None:
    chunk = SampleChunk(prompt_index=0, prompt="p", sample_start=0, sample_count=2)
    worker = _CapacityWorker(
        worker_id="w0",
        max_samples=2,
        request_id_override="wrong-request",
    )
    executor, _ = _executor([chunk], [worker])

    with pytest.raises(RuntimeError, match="request_id mismatch"):
        await executor.execute(_request(2))

    assert worker.executed == [chunk.chunk_key]


@dataclass
class _StaleSlotWorker:
    """Worker that returns a typed stale-slot result (evicted version slot)."""

    worker_id: str
    executed: list[str] = field(default_factory=list)

    def execute_chunk(self, envelope: ChunkExecutionEnvelope) -> ChunkExecutionResult:
        chunk = envelope.chunk
        self.executed.append(chunk.chunk_key)
        version = envelope.request.policy_version
        return ChunkExecutionResult(
            request_id=envelope.request.request_id,
            worker_id=self.worker_id,
            chunk=chunk,
            output=None,
            # Slot mode stamps the REQUEST's version, so the version assert would
            # pass — only the stale_slot flag distinguishes this from success.
            policy_version=version,
            error=f"trainable-state slot evicted for policy_version={version}",
            stale_slot=True,
        )


def _versioned_request(num_samples: int, version: int) -> GenerationRequest:
    return GenerationRequest(
        request_id="req-stale",
        family="test",
        task="t2i",
        inputs=["p"],
        samples_per_prompt=num_samples,
        policy_version=version,
    )


@pytest.mark.asyncio
async def test_stale_slot_routes_to_graceful_discard_not_failure() -> None:
    """A stale-slot chunk raises StaleSlotDiscard (a typed discard), NOT a generic
    RuntimeError, so the producer counts it as a stale discard, not a collect error.
    It must also skip OOM-split retries entirely (no second execute on the chunk)."""

    chunk = SampleChunk(prompt_index=0, prompt="p", sample_start=0, sample_count=2)
    worker = _StaleSlotWorker(worker_id="w0")
    handles = [
        DistributedWorkerHandle(worker_id=worker.worker_id, actor=worker),
    ]
    executor = RayGenerationExecutor(
        planner=_StaticPlanner(chunks=[chunk]),
        workers=handles,
        gatherer=_CoverageGatherer(),
    )

    with pytest.raises(StaleSlotDiscard, match="policy_version=7"):
        await executor.execute(_versioned_request(2, version=7))

    # Routed before the OOM-degrade loop, so the chunk ran exactly once (no retry).
    assert worker.executed == [chunk.chunk_key]


def test_stale_slot_discard_is_not_runtime_error() -> None:
    """StaleSlotDiscard must be its OWN type, not a RuntimeError subclass — the
    producer's generic ``except Exception`` (error_count) must not catch it first."""

    assert not issubclass(StaleSlotDiscard, RuntimeError)
    assert issubclass(StaleSlotDiscard, Exception)


def test_is_oom_error_classifier() -> None:
    assert _is_oom_error(_OOM_MESSAGE)
    assert _is_oom_error("torch.OutOfMemoryError: HIP out of memory")
    assert not _is_oom_error("ValueError: shape mismatch")


@dataclass
class _RoutingWorker:
    """Records whether execute() took the per-request pipelined path or the
    per-chunk path."""

    worker_id: str
    chunk_calls: list[str] = field(default_factory=list)
    request_calls: list[str] = field(default_factory=list)
    max_samples: int | None = None
    pipeline_oom: bool = False
    pipeline_request_id_override: str | None = None
    pipeline_worker_id_override: str | None = None

    def execute_chunk(self, envelope: ChunkExecutionEnvelope) -> ChunkExecutionResult:
        self.chunk_calls.append(envelope.chunk.chunk_key)
        if self.max_samples is not None and envelope.chunk.sample_count > self.max_samples:
            return ChunkExecutionResult(
                request_id=envelope.request.request_id,
                worker_id=self.worker_id,
                chunk=envelope.chunk,
                output=None,
                error=_OOM_MESSAGE,
            )
        return ChunkExecutionResult(
            request_id=envelope.request.request_id,
            worker_id=self.worker_id,
            chunk=envelope.chunk,
            output={"chunk_key": envelope.chunk.chunk_key, "samples": envelope.chunk.sample_count},
        )

    def execute_request_pipelined(
        self,
        request,
        engine_plan,
        sample_rows,
    ) -> GenerationOutput | PipelinedRequestOutOfMemory:
        self.request_calls.append(request.request_id)
        request_id = self.pipeline_request_id_override or request.request_id
        if self.pipeline_oom:
            return PipelinedRequestOutOfMemory(
                request_id=request_id,
                worker_id=self.pipeline_worker_id_override or self.worker_id,
                error=_OOM_MESSAGE,
            )
        return GenerationOutput(
            request_id=request_id,
            family=request.family,
            task=request.task,
            sample_rows=list(sample_rows),
            output=[{"pipelined": True}],
        )


def _routing_executor(chunks, workers, *, pipelined):
    handles = [DistributedWorkerHandle(worker_id=w.worker_id, actor=w) for w in workers]
    return RayGenerationExecutor(
        planner=_StaticPlanner(chunks=chunks),
        workers=handles,
        gatherer=_CoverageGatherer(),
        pipelined=pipelined,
    )


@pytest.mark.asyncio
async def test_pipelined_routes_single_worker_to_per_request_path() -> None:
    """pipelined=True + one worker => the whole request runs via the per-request
    pipelined path (execute_request_pipelined), NOT per-chunk dispatch."""

    chunks = [
        SampleChunk(prompt_index=0, prompt="p", sample_start=i * 2, sample_count=2)
        for i in range(2)
    ]
    worker = _RoutingWorker(worker_id="w0")
    executor = _routing_executor(chunks, [worker], pipelined=True)

    output = await executor.execute(_request(4, samples_per_chunk=2))

    assert worker.request_calls == ["req-oom"]
    assert worker.chunk_calls == []
    assert output.output == [{"pipelined": True}]


def test_pipelined_rejects_multiple_workers_at_executor_construction() -> None:
    """Direct executor callers get the same fail-fast guard as config users."""

    chunks = [
        SampleChunk(prompt_index=0, prompt="p", sample_start=i * 2, sample_count=2)
        for i in range(2)
    ]
    workers = [_RoutingWorker(worker_id=f"w{i}") for i in range(2)]
    with pytest.raises(ValueError, match="requires exactly one rollout worker"):
        _routing_executor(chunks, workers, pipelined=True)

    assert all(not worker.request_calls and not worker.chunk_calls for worker in workers)


@pytest.mark.asyncio
async def test_pipelined_uses_per_chunk_path_for_one_chunk() -> None:
    """A one-chunk request has nothing to overlap and keeps OOM admission."""

    chunk = SampleChunk(prompt_index=0, prompt="p", sample_start=0, sample_count=4)
    worker = _RoutingWorker(worker_id="w0", max_samples=2)
    executor = _routing_executor([chunk], [worker], pipelined=True)

    output = await executor.execute(_request(4))

    assert worker.request_calls == []
    assert worker.chunk_calls == [_key(0, 4), _key(0, 2), _key(2, 2)]
    assert sorted((row["chunk_key"], row["samples"]) for row in output.output) == [
        (_key(0, 2), 2),
        (_key(2, 2), 2),
    ]


@pytest.mark.asyncio
async def test_pipelined_oom_retries_through_per_chunk_split_admission() -> None:
    chunks = [
        SampleChunk(prompt_index=0, prompt="p", sample_start=i * 4, sample_count=4)
        for i in range(2)
    ]
    worker = _RoutingWorker(
        worker_id="w0",
        max_samples=2,
        pipeline_oom=True,
    )
    executor = _routing_executor(chunks, [worker], pipelined=True)

    output = await executor.execute(_request(8, samples_per_chunk=4))

    assert worker.request_calls == ["req-oom"]
    assert worker.chunk_calls == [
        _key(0, 4),
        _key(4, 4),
        _key(0, 2),
        _key(2, 2),
        _key(4, 2),
        _key(6, 2),
    ]
    assert sorted((row["chunk_key"], row["samples"]) for row in output.output) == [
        (_key(0, 2), 2),
        (_key(2, 2), 2),
        (_key(4, 2), 2),
        (_key(6, 2), 2),
    ]


@pytest.mark.asyncio
async def test_pipelined_result_request_id_must_match_request() -> None:
    chunks = [
        SampleChunk(prompt_index=0, prompt="p", sample_start=i * 2, sample_count=2)
        for i in range(2)
    ]
    worker = _RoutingWorker(
        worker_id="w0",
        pipeline_request_id_override="wrong-request",
    )
    executor = _routing_executor(chunks, [worker], pipelined=True)

    with pytest.raises(RuntimeError, match="request_id mismatch"):
        await executor.execute(_request(4, samples_per_chunk=2))

    assert worker.request_calls == ["req-oom"]
    assert worker.chunk_calls == []


@pytest.mark.asyncio
async def test_pipelined_oom_worker_id_must_match_actor() -> None:
    chunks = [
        SampleChunk(prompt_index=0, prompt="p", sample_start=i * 2, sample_count=2)
        for i in range(2)
    ]
    worker = _RoutingWorker(
        worker_id="w0",
        pipeline_oom=True,
        pipeline_worker_id_override="wrong-worker",
    )
    executor = _routing_executor(chunks, [worker], pipelined=True)

    with pytest.raises(RuntimeError, match="worker_id mismatch"):
        await executor.execute(_request(4, samples_per_chunk=2))

    assert worker.request_calls == ["req-oom"]
    assert worker.chunk_calls == []


@pytest.mark.asyncio
async def test_default_uses_per_chunk_path() -> None:
    """Default (pipelined=False) is the unchanged per-chunk dispatch."""

    chunk = SampleChunk(prompt_index=0, prompt="p", sample_start=0, sample_count=4)
    worker = _RoutingWorker(worker_id="w0")
    executor = _routing_executor([chunk], [worker], pipelined=False)

    await executor.execute(_request(4))

    assert worker.request_calls == []
    assert worker.chunk_calls == [chunk.chunk_key]
