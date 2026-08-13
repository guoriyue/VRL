"""Batch OOM degradation: split-on-OOM in the Ray generation executor."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest
import torch

from vrl.generation.execution.batch_placement import (
    DeviceAssignment,
    DistributedGenerationPlan,
)
from vrl.generation.execution.planner import EnginePlan
from vrl.generation.execution.sample_batches import GenerationSampleBatch
from vrl.generation.execution.types import (
    GenerationBatchEnvelope,
    GenerationBatchResult,
    PipelinedRequestOutOfMemory,
    StaleSlotDiscard,
)
from vrl.generation.ray.executor import RayGenerationExecutor, _is_oom_error
from vrl.generation.types import GenerationOutput, GenerationRequest
from vrl.ray.actor_group import RayActorHandle
from vrl.ray.actor_pool import RayActorDispatcher
from vrl.trajectory import TrajectoryBatch

# torch's allocator wire format, pinned against the real allocator by
# test_oom_matcher_accepts_the_real_torch_allocator_message below.
_OOM_PREFIX = "CUDA out of memory. Tried to allocate "
_OOM_MESSAGE = f"{_OOM_PREFIX}4.00 GiB"

# `_CapacityWorker` stands in for a Ray worker, not for torch; only the message
# FORMAT it hand-copies needs the gpu-lane twin, so the two tests that depend on
# that format carry the label and the rest do not.
_OOM_WIRE_FORMAT = pytest.mark.real_cover(
    "tests/generation/ray/test_oom_split.py"
    "::test_oom_matcher_accepts_the_real_torch_allocator_message",
    why=(
        "a Ray worker that OOMs on exactly the batch sizes these tests need cannot be "
        "produced in-process, so the worker itself stays a fake; what it hand-copies from "
        "torch is the allocator message format, and the gpu-lane twin pins that"
    ),
)


def _key(start: int, count: int) -> str:
    """Derive a batch_key from the source template, not a hand-copied f-string."""

    return GenerationSampleBatch(prompt_index=0, sample_start=start, sample_count=count).batch_key


@dataclass
class _CapacityWorker:
    """Synchronous worker that OOMs on batches above ``max_samples``."""

    worker_id: str
    max_samples: int
    executed: list[str] = field(default_factory=list)
    fail_message: str = _OOM_MESSAGE
    request_id_override: str | None = None

    def execute_batch(self, envelope: GenerationBatchEnvelope) -> GenerationBatchResult:
        batch = envelope.batch
        self.executed.append(batch.batch_key)
        request_id = (
            envelope.request.request_id
            if self.request_id_override is None
            else self.request_id_override
        )
        if batch.sample_count > self.max_samples:
            return GenerationBatchResult(
                request_id=request_id,
                worker_id=self.worker_id,
                batch=batch,
                output=None,
                error=self.fail_message,
            )
        return GenerationBatchResult(
            request_id=request_id,
            worker_id=self.worker_id,
            batch=batch,
            output={"batch_key": batch.batch_key, "samples": batch.sample_count},
        )


@dataclass
class _StaticPlanner:
    """Plan one static assignment per provided batch."""

    batches: list[GenerationSampleBatch]
    strategy: str = "static"

    def plan_with_engine(
        self,
        request: GenerationRequest,
        worker_ids: list[str],
    ) -> DistributedGenerationPlan:
        assignments = tuple(
            DeviceAssignment(
                worker_id=worker_ids[index % len(worker_ids)],
                envelope=GenerationBatchEnvelope(
                    request=request,
                    batch=batch,
                ),
            )
            for index, batch in enumerate(self.batches)
        )
        engine_plan = EnginePlan.from_request(request)
        return DistributedGenerationPlan(engine_plan=engine_plan, assignments=assignments)


class _CoverageGatherer:
    """Assert sample coverage by batch metadata, mirroring layout.ordered_batches."""

    def gather_batches(
        self,
        request: GenerationRequest,
        sample_rows: Any,
        batches: list[dict[str, Any]],
    ) -> GenerationOutput:
        return GenerationOutput(
            output=list(batches),
            trajectory=TrajectoryBatch(
                request_id=request.request_id,
                family=request.family,
                task=request.task,
                sample_rows=list(sample_rows),
                axes={},
                segments={},
            ),
        )


def _request(
    num_samples: int,
    *,
    samples_per_generation_batch: int | None = None,
    runtime_debug: bool = False,
) -> GenerationRequest:
    return GenerationRequest(
        request_id="req-oom",
        family="test",
        task="t2i",
        inputs=["p"],
        samples_per_prompt=num_samples,
        sampling=(
            {}
            if samples_per_generation_batch is None
            else {"samples_per_generation_batch": samples_per_generation_batch}
        ),
        runtime_debug=runtime_debug,
    )


def _executor(
    batches: list[GenerationSampleBatch],
    workers: list[_CapacityWorker],
) -> tuple[RayGenerationExecutor, list[RayActorHandle]]:
    handles = [RayActorHandle(worker_id=worker.worker_id, actor=worker) for worker in workers]
    executor = RayGenerationExecutor(
        planner=_StaticPlanner(batches=batches),
        workers=handles,
        gatherer=_CoverageGatherer(),
        actor_dispatcher=RayActorDispatcher(
            tuple(handle.worker_id for handle in handles),
        ),
        generation_stall_timeout_s=30.0,
    )
    return executor, handles


@pytest.mark.gpu
def test_oom_matcher_accepts_the_real_torch_allocator_message() -> None:
    """`_OOM_MESSAGE` is only an honest fixture while torch still emits that prefix."""

    with pytest.raises(torch.OutOfMemoryError) as caught:
        torch.empty(1024**4, dtype=torch.float32, device="cuda")

    real = str(caught.value)
    assert real.startswith(_OOM_PREFIX), real
    # The production matcher is a substring test on "out of memory"; assert it
    # against the real message, not only against our own fixture.
    assert _is_oom_error(real) is True
    assert _is_oom_error(_OOM_MESSAGE) is True


@_OOM_WIRE_FORMAT
@pytest.mark.asyncio
async def test_oom_chunk_splits_until_it_fits() -> None:
    """An 8-sample batch on a 2-sample worker degrades to four 2-sample batches."""

    batch = GenerationSampleBatch(prompt_index=0, sample_start=0, sample_count=8)
    worker = _CapacityWorker(worker_id="w0", max_samples=2)
    executor, _ = _executor([batch], [worker])

    output = await executor.execute(_request(8, runtime_debug=True))

    covered = sorted((entry["batch_key"], entry["samples"]) for entry in output.output)
    assert covered == [
        (_key(0, 2), 2),
        (_key(2, 2), 2),
        (_key(4, 2), 2),
        (_key(6, 2), 2),
    ]
    assert output.runtime_debug is not None
    splits = output.runtime_debug["batch_oom_splits"]
    # Recursion order: 8 -> [0:4] + [4:8] -> 2-sample leaves.
    assert [row["batch_key"] for row in splits] == [_key(0, 8), _key(0, 4), _key(4, 4)]
    assert all(row["worker_id"] == "w0" for row in splits)


@_OOM_WIRE_FORMAT
@pytest.mark.asyncio
async def test_single_sample_oom_still_raises() -> None:
    """A batch that OOMs at one sample is a hard failure, not an infinite loop."""

    batch = GenerationSampleBatch(prompt_index=0, sample_start=0, sample_count=4)
    worker = _CapacityWorker(worker_id="w0", max_samples=0)
    executor, _ = _executor([batch], [worker])

    with pytest.raises(RuntimeError, match="out of memory"):
        await executor.execute(_request(4))


@pytest.mark.asyncio
async def test_non_oom_error_is_not_retried() -> None:
    """Only allocator failures degrade; other worker errors fail fast."""

    batch = GenerationSampleBatch(prompt_index=0, sample_start=0, sample_count=4)
    worker = _CapacityWorker(
        worker_id="w0",
        max_samples=2,
        fail_message="ValueError: bad scheduler state",
    )
    executor, _ = _executor([batch], [worker])

    with pytest.raises(RuntimeError, match="bad scheduler state"):
        await executor.execute(_request(4))
    assert worker.executed == [batch.batch_key]


@pytest.mark.asyncio
async def test_healthy_chunks_skip_degradation_path() -> None:
    """No OOM: results and telemetry are exactly the pre-split behavior."""

    batches = [
        GenerationSampleBatch(prompt_index=0, sample_start=0, sample_count=2),
        GenerationSampleBatch(prompt_index=0, sample_start=2, sample_count=2),
    ]
    worker = _CapacityWorker(worker_id="w0", max_samples=2)
    executor, _ = _executor(batches, [worker])

    output = await executor.execute(_request(4))

    assert len(output.output) == 2
    assert output.runtime_debug is None


@pytest.mark.asyncio
async def test_result_request_id_must_match_submitted_envelope() -> None:
    batch = GenerationSampleBatch(prompt_index=0, sample_start=0, sample_count=2)
    worker = _CapacityWorker(
        worker_id="w0",
        max_samples=2,
        request_id_override="wrong-request",
    )
    executor, _ = _executor([batch], [worker])

    with pytest.raises(RuntimeError, match="request_id mismatch"):
        await executor.execute(_request(2))

    assert worker.executed == [batch.batch_key]


@dataclass
class _StaleSlotWorker:
    """Worker that returns a typed stale-slot result (evicted version slot)."""

    worker_id: str
    executed: list[str] = field(default_factory=list)

    def execute_batch(self, envelope: GenerationBatchEnvelope) -> GenerationBatchResult:
        batch = envelope.batch
        self.executed.append(batch.batch_key)
        version = envelope.request.policy_version
        return GenerationBatchResult(
            request_id=envelope.request.request_id,
            worker_id=self.worker_id,
            batch=batch,
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
    """A stale-slot batch raises StaleSlotDiscard (a typed discard), NOT a generic
    RuntimeError, so the producer counts it as a stale discard, not a collect error.
    It must also skip OOM-split retries entirely (no second execute on the batch)."""

    batch = GenerationSampleBatch(prompt_index=0, sample_start=0, sample_count=2)
    worker = _StaleSlotWorker(worker_id="w0")
    handles = [
        RayActorHandle(worker_id=worker.worker_id, actor=worker),
    ]
    executor = RayGenerationExecutor(
        planner=_StaticPlanner(batches=[batch]),
        workers=handles,
        gatherer=_CoverageGatherer(),
        actor_dispatcher=RayActorDispatcher(("w0",)),
        generation_stall_timeout_s=30.0,
    )

    with pytest.raises(StaleSlotDiscard, match="policy_version=7"):
        await executor.execute(_versioned_request(2, version=7))

    # Routed before the OOM-degrade loop, so the batch ran exactly once (no retry).
    assert worker.executed == [batch.batch_key]


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
    per-batch path."""

    worker_id: str
    batch_calls: list[str] = field(default_factory=list)
    request_calls: list[str] = field(default_factory=list)
    max_samples: int | None = None
    pipeline_oom: bool = False
    pipeline_request_id_override: str | None = None
    pipeline_worker_id_override: str | None = None

    def execute_batch(self, envelope: GenerationBatchEnvelope) -> GenerationBatchResult:
        self.batch_calls.append(envelope.batch.batch_key)
        if self.max_samples is not None and envelope.batch.sample_count > self.max_samples:
            return GenerationBatchResult(
                request_id=envelope.request.request_id,
                worker_id=self.worker_id,
                batch=envelope.batch,
                output=None,
                error=_OOM_MESSAGE,
            )
        return GenerationBatchResult(
            request_id=envelope.request.request_id,
            worker_id=self.worker_id,
            batch=envelope.batch,
            output={"batch_key": envelope.batch.batch_key, "samples": envelope.batch.sample_count},
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
            output=[{"pipelined": True}],
            trajectory=TrajectoryBatch(
                request_id=request_id,
                family=request.family,
                task=request.task,
                sample_rows=list(sample_rows),
                axes={},
                segments={},
            ),
        )


def _routing_executor(batches, workers, *, pipelined):
    handles = [RayActorHandle(worker_id=w.worker_id, actor=w) for w in workers]
    return RayGenerationExecutor(
        planner=_StaticPlanner(batches=batches),
        workers=handles,
        gatherer=_CoverageGatherer(),
        actor_dispatcher=RayActorDispatcher(
            tuple(handle.worker_id for handle in handles),
        ),
        generation_stall_timeout_s=30.0,
        pipelined=pipelined,
    )


@pytest.mark.asyncio
async def test_pipelined_routes_single_worker_to_per_request_path() -> None:
    """pipelined=True + one worker => the whole request runs via the per-request
    pipelined path (execute_request_pipelined), NOT per-batch dispatch."""

    batches = [
        GenerationSampleBatch(prompt_index=0, sample_start=i * 2, sample_count=2) for i in range(2)
    ]
    worker = _RoutingWorker(worker_id="w0")
    executor = _routing_executor(batches, [worker], pipelined=True)

    output = await executor.execute(_request(4, samples_per_generation_batch=2))

    assert worker.request_calls == ["req-oom"]
    assert worker.batch_calls == []
    assert output.output == [{"pipelined": True}]


def test_pipelined_rejects_multiple_workers_at_executor_construction() -> None:
    """Direct executor callers get the same fail-fast guard as config users."""

    batches = [
        GenerationSampleBatch(prompt_index=0, sample_start=i * 2, sample_count=2) for i in range(2)
    ]
    workers = [_RoutingWorker(worker_id=f"w{i}") for i in range(2)]
    with pytest.raises(ValueError, match="requires exactly one rollout worker"):
        _routing_executor(batches, workers, pipelined=True)

    assert all(not worker.request_calls and not worker.batch_calls for worker in workers)


@pytest.mark.asyncio
async def test_pipelined_uses_per_chunk_path_for_one_chunk() -> None:
    """A one-batch request has nothing to overlap and keeps OOM admission."""

    batch = GenerationSampleBatch(prompt_index=0, sample_start=0, sample_count=4)
    worker = _RoutingWorker(worker_id="w0", max_samples=2)
    executor = _routing_executor([batch], [worker], pipelined=True)

    output = await executor.execute(_request(4))

    assert worker.request_calls == []
    assert worker.batch_calls == [_key(0, 4), _key(0, 2), _key(2, 2)]
    assert sorted((row["batch_key"], row["samples"]) for row in output.output) == [
        (_key(0, 2), 2),
        (_key(2, 2), 2),
    ]


@pytest.mark.asyncio
async def test_pipelined_oom_retries_through_per_chunk_split_admission() -> None:
    batches = [
        GenerationSampleBatch(prompt_index=0, sample_start=i * 4, sample_count=4) for i in range(2)
    ]
    worker = _RoutingWorker(
        worker_id="w0",
        max_samples=2,
        pipeline_oom=True,
    )
    executor = _routing_executor(batches, [worker], pipelined=True)

    output = await executor.execute(_request(8, samples_per_generation_batch=4))

    assert worker.request_calls == ["req-oom"]
    assert worker.batch_calls == [
        _key(0, 4),
        _key(4, 4),
        _key(0, 2),
        _key(2, 2),
        _key(4, 2),
        _key(6, 2),
    ]
    assert sorted((row["batch_key"], row["samples"]) for row in output.output) == [
        (_key(0, 2), 2),
        (_key(2, 2), 2),
        (_key(4, 2), 2),
        (_key(6, 2), 2),
    ]


@pytest.mark.asyncio
async def test_pipelined_result_request_id_must_match_request() -> None:
    batches = [
        GenerationSampleBatch(prompt_index=0, sample_start=i * 2, sample_count=2) for i in range(2)
    ]
    worker = _RoutingWorker(
        worker_id="w0",
        pipeline_request_id_override="wrong-request",
    )
    executor = _routing_executor(batches, [worker], pipelined=True)

    with pytest.raises(RuntimeError, match="request_id mismatch"):
        await executor.execute(_request(4, samples_per_generation_batch=2))

    assert worker.request_calls == ["req-oom"]
    assert worker.batch_calls == []


@pytest.mark.asyncio
async def test_pipelined_oom_worker_id_must_match_actor() -> None:
    batches = [
        GenerationSampleBatch(prompt_index=0, sample_start=i * 2, sample_count=2) for i in range(2)
    ]
    worker = _RoutingWorker(
        worker_id="w0",
        pipeline_oom=True,
        pipeline_worker_id_override="wrong-worker",
    )
    executor = _routing_executor(batches, [worker], pipelined=True)

    with pytest.raises(RuntimeError, match="worker_id mismatch"):
        await executor.execute(_request(4, samples_per_generation_batch=2))

    assert worker.request_calls == ["req-oom"]
    assert worker.batch_calls == []


@pytest.mark.asyncio
async def test_default_uses_per_chunk_path() -> None:
    """Default (pipelined=False) is the unchanged per-batch dispatch."""

    batch = GenerationSampleBatch(prompt_index=0, sample_start=0, sample_count=4)
    worker = _RoutingWorker(worker_id="w0")
    executor = _routing_executor([batch], [worker], pipelined=False)

    await executor.execute(_request(4))

    assert worker.request_calls == []
    assert worker.batch_calls == [batch.batch_key]
