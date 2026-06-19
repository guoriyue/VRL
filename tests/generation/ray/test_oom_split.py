"""Chunk OOM degradation: split-on-OOM in the Ray generation executor."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from vrl.generation.capabilities import (
    AxisCapability,
    ExecutionStageCapability,
    FamilyCapability,
)
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
    StaleSlotDiscard,
)
from vrl.generation.ray.executor import RayGenerationExecutor, _is_oom_error
from vrl.generation.types import GenerationOutput, GenerationRequest

_OOM_MESSAGE = "CUDA out of memory. Tried to allocate 4.00 GiB"


@dataclass
class _CapacityWorker:
    """Synchronous worker that OOMs on chunks above ``max_samples``."""

    worker_id: str
    max_samples: int
    executed: list[str] = field(default_factory=list)
    fail_message: str = _OOM_MESSAGE

    def execute_chunk(self, envelope: ChunkExecutionEnvelope) -> ChunkExecutionResult:
        chunk = envelope.chunk
        self.executed.append(chunk.chunk_key)
        if chunk.sample_count > self.max_samples:
            return ChunkExecutionResult(
                request_id=envelope.request.request_id,
                worker_id=self.worker_id,
                chunk=chunk,
                output=None,
                error=self.fail_message,
            )
        return ChunkExecutionResult(
            request_id=envelope.request.request_id,
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
        *,
        sample_rows: Any,
    ) -> DistributedGenerationPlan:
        assignments = tuple(
            DeviceAssignment(
                worker_id=workers[index % len(workers)].worker_id,
                node_id=None,
                gpu_ids=(),
                chunk=chunk,
                envelope=ChunkExecutionEnvelope(
                    request=request,
                    chunk=chunk,
                    plan_id="plan-0",
                ),
            )
            for index, chunk in enumerate(self.chunks)
        )
        capability = FamilyCapability(
            family="test",
            task="t2i",
            trajectory_kind="diffusion",
            expected_axes=(
                AxisCapability(name="sample", kind="sample", batchable=True, chunkable=True),
            ),
            execution_stages=(ExecutionStageCapability(name="denoise"),),
        )
        engine_plan = build_engine_plan(request, sample_rows, capability=capability)
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
            prompts=list(request.prompts),
            sample_rows=list(sample_rows),
            output=list(chunks),
        )


def _request(num_samples: int) -> GenerationRequest:
    return GenerationRequest(
        request_id="req-oom",
        family="test",
        task="t2i",
        prompts=["p"],
        samples_per_prompt=num_samples,
    )


def _executor(
    chunks: list[SampleChunk],
    workers: list[_CapacityWorker],
) -> tuple[RayGenerationExecutor, list[DistributedWorkerHandle]]:
    handles = [
        DistributedWorkerHandle(worker_id=worker.worker_id, node_id="local", actor=worker)
        for worker in workers
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

    output = await executor.execute(_request(8))

    covered = sorted(
        (entry["chunk_key"], entry["samples"]) for entry in output.output
    )
    assert covered == [
        ("prompt:0:samples:0:2", 2),
        ("prompt:0:samples:2:4", 2),
        ("prompt:0:samples:4:6", 2),
        ("prompt:0:samples:6:8", 2),
    ]
    splits = output.extra["ray_chunk_oom_splits"]
    assert [row["chunk_key"] for row in splits] == [
        "prompt:0:samples:0:8",
        "prompt:0:samples:0:4",
        "prompt:0:samples:4:8",
    ]
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
    assert worker.executed == ["prompt:0:samples:0:4"]


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
    assert "ray_chunk_oom_splits" not in output.extra


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
        prompts=["p"],
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
        DistributedWorkerHandle(worker_id=worker.worker_id, node_id="local", actor=worker),
    ]
    executor = RayGenerationExecutor(
        planner=_StaticPlanner(chunks=[chunk]),
        workers=handles,
        gatherer=_CoverageGatherer(),
    )

    with pytest.raises(StaleSlotDiscard, match="policy_version=7"):
        await executor.execute(_versioned_request(2, version=7))

    # Routed before the OOM-degrade loop, so the chunk ran exactly once (no retry).
    assert worker.executed == ["prompt:0:samples:0:2"]


def test_stale_slot_discard_is_not_runtime_error() -> None:
    """StaleSlotDiscard must be its OWN type, not a RuntimeError subclass — the
    producer's generic ``except Exception`` (error_count) must not catch it first."""

    assert not issubclass(StaleSlotDiscard, RuntimeError)
    assert issubclass(StaleSlotDiscard, Exception)


def test_is_oom_error_classifier() -> None:
    assert _is_oom_error(_OOM_MESSAGE)
    assert _is_oom_error("torch.OutOfMemoryError: HIP out of memory")
    assert not _is_oom_error("ValueError: shape mismatch")
