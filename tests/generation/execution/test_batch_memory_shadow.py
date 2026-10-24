"""Batch memory readings: the per-batch CUDA footprint a worker reports.

Covers the reading's normalization from the binding mapping, the pre-loop
occupancy snapshot, and the worker wire contract (readings cross ungated).
"""

from __future__ import annotations

from dataclasses import asdict
from types import SimpleNamespace
from typing import Any

from tests.generation.execution._helpers import launch_config
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

    assert BatchMemoryReading.from_mapping(asdict(reading)) == reading
    partial = asdict(reading)
    del partial["decode_peak_bytes"]
    assert BatchMemoryReading.from_mapping(partial) is None
    # non-torch = device-used minus torch-reserved: (32-18) - 11 = 3GB.
    assert reading.non_torch_bytes == 3 * GB
    assert reading.budget_bytes == 29 * GB


def test_every_counter_of_a_reading_comes_from_the_batch_device(monkeypatch) -> None:
    """A batch on the second card of a CUDA host reads that card; a CPU batch reads none."""

    import torch

    from vrl.utils.cuda_memory import cuda_peak_allocated_bytes, reset_cuda_peak

    read_devices: list[Any] = []

    def counter(value: Any) -> Any:
        def read(device: Any = None) -> Any:
            read_devices.append(device)
            return value

        return read

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", counter(None))
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", counter(18 * GB))
    monkeypatch.setattr(torch.cuda, "memory_allocated", counter(10 * GB))
    monkeypatch.setattr(torch.cuda, "memory_reserved", counter(11 * GB))
    monkeypatch.setattr(torch.cuda, "mem_get_info", counter((18 * GB, 32 * GB)))

    for device in (torch.device("cpu"), torch.device("cuda", 1)):
        reset_cuda_peak(device)
        occupancy = BatchMemoryReading.cuda_occupancy_snapshot(device)
        peak = cuda_peak_allocated_bytes(device)
        if device.type == "cpu":
            assert (occupancy, peak, read_devices) == (None, None, [])
    assert peak == 18 * GB
    assert occupancy is not None and occupancy["baseline_allocated_bytes"] == 10 * GB
    assert read_devices == [torch.device("cuda", 1)] * 5


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
    config = launch_config(policy_version=1)
    executor = _MemoryExecutor()
    core = GenerationWorkerCore("rollout-0", config)
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
