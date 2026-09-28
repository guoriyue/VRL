"""Batch memory readings: the per-batch CUDA footprint a worker reports.

Covers the reading's normalization from the binding mapping, the pre-loop
occupancy snapshot, and the worker wire contract (readings cross ungated).
"""

from __future__ import annotations

from dataclasses import asdict
from types import SimpleNamespace
from typing import Any

import pytest

from tests.generation.execution._helpers import launch_contract
from vrl.generation.execution.sample_batches import GenerationSampleBatch
from vrl.generation.execution.types import (
    BatchMemoryReading,
    GenerationBatchEnvelope,
)
from vrl.generation.execution.worker import GenerationWorkerCore
from vrl.generation.types import GenerationRequest

GB = 1024**3


def _reading(
    *,
    sample_count: int = 4,
    baseline: int = 10 * GB,
    denoise_peak: int = 18 * GB,
    decode_peak: int = 14 * GB,
    reserved_start: int = 11 * GB,
    free_start: int = 18 * GB,
    total: int = 32 * GB,
) -> BatchMemoryReading:
    return BatchMemoryReading(
        sample_count=sample_count,
        baseline_allocated_bytes=baseline,
        denoise_peak_bytes=denoise_peak,
        decode_peak_bytes=decode_peak,
        reserved_start_bytes=reserved_start,
        free_start_bytes=free_start,
        total_bytes=total,
    )


def test_reading_normalizes_binding_mapping_and_rejects_partial_data() -> None:
    reading = _reading()

    assert BatchMemoryReading.from_metrics(asdict(reading)) == reading
    partial = asdict(reading)
    del partial["decode_peak_bytes"]
    assert BatchMemoryReading.from_metrics(partial) is None
    # non-torch = device-used minus torch-reserved: (32-18) - 11 = 3GB.
    assert reading.non_torch_bytes == 3 * GB
    assert reading.budget_bytes == 29 * GB


# -- worker wire contract (readings cross ungated) -----------------------------


class _MemoryExecutor:
    family = "sd3_5"
    task = "t2i"

    def __init__(self) -> None:
        self.model = SimpleNamespace(device="cpu")

    def forward_batch(self, *args: Any, **kwargs: Any) -> Any:
        return SimpleNamespace(memory=asdict(_reading()))

    def merge_generation_batches(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError


def test_worker_forwards_batch_memory_without_runtime_debug() -> None:
    contract = launch_contract(policy_version=1)
    executor = _MemoryExecutor()
    core = GenerationWorkerCore("rollout-0", contract, executor)
    core.executor = executor
    request = GenerationRequest(
        request_id="req-1",
        family="sd3_5",
        task="t2i",
        inputs=["p"],
        samples_per_prompt=1,
        policy_version=1,
    )
    envelope = GenerationBatchEnvelope(
        request=request,
        batch=GenerationSampleBatch(prompt_index=0, sample_start=0, sample_count=1),
    )

    result = core.execute_batch(envelope)

    assert result.error is None
    assert result.memory == _reading()
    assert result.output.memory is None
    assert result.rank_metrics == {}


def test_cuda_occupancy_is_absent_only_without_cuda(monkeypatch) -> None:
    import torch

    from vrl.generation.execution.types import BatchMemoryReading

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert BatchMemoryReading.cuda_occupancy_snapshot() is None


@pytest.mark.parametrize("operation", ["mem_get_info", "memory_allocated", "memory_reserved"])
def test_cuda_occupancy_preserves_query_errors(monkeypatch, operation) -> None:
    import torch

    from vrl.generation.execution.types import BatchMemoryReading

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (8, 16))
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda: 4)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda: 6)
    failure = RuntimeError("CUDA query failed")

    def fail():
        raise failure

    monkeypatch.setattr(torch.cuda, operation, fail)
    with pytest.raises(RuntimeError, match="CUDA query failed") as caught:
        BatchMemoryReading.cuda_occupancy_snapshot()
    assert caught.value is failure
