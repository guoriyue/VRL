"""Tests for generation executor sample batch helpers."""

from __future__ import annotations

import pytest


def test_sample_batch_plan_prompt_major() -> None:
    """Checks build prompt batches prompt major."""
    from vrl.generation.execution.sample_batches import GenerationSampleBatch

    batches = GenerationSampleBatch.plan(
        2,
        samples_per_prompt=5,
        max_samples_per_batch=2,
    )

    got = [(batch.prompt_index, batch.sample_start, batch.sample_count) for batch in batches]
    assert got == [
        (0, 0, 2),
        (0, 2, 2),
        (0, 4, 1),
        (1, 0, 2),
        (1, 2, 2),
        (1, 4, 1),
    ]


def test_run_sample_batches_with_oom_retry_splits_until_success() -> None:
    """An OOM on a 5-sample batch halves recursively (2, then 3 -> 1 + 2) and the results come
    back in sample order.
    """
    from vrl.generation.execution.sample_batches import (
        GenerationSampleBatch,
        run_sample_batches_with_oom_retry,
    )

    seen: list[tuple[int, int]] = []

    def run_one(batch: GenerationSampleBatch) -> int:
        seen.append((batch.sample_start, batch.sample_count))
        if batch.sample_count > 2:
            raise RuntimeError("CUDA out of memory while allocating tensor")
        return batch.sample_count

    results = run_sample_batches_with_oom_retry(
        [
            GenerationSampleBatch(
                prompt_index=0,
                sample_start=0,
                sample_count=5,
            )
        ],
        run_one,
    )

    assert results == [2, 1, 2]
    assert seen == [(0, 5), (0, 2), (2, 3), (2, 1), (3, 2)]


@pytest.mark.parametrize("width", [0, -1, 1.5, True, "2"])
def test_engine_plan_rejects_invalid_explicit_width(width):
    from vrl.generation.execution.planner import EnginePlan
    from vrl.generation.types import GenerationRequest

    request = GenerationRequest(
        request_id="width-check",
        family="test",
        task="t2i",
        inputs=["p"],
        samples_per_prompt=2,
    )
    with pytest.raises(ValueError, match="max_samples_per_batch must be a positive integer"):
        EnginePlan.from_request(request, max_samples_per_batch=width)


@pytest.mark.parametrize("field", ["prompt_index", "sample_start", "sample_count"])
@pytest.mark.parametrize("value", [0.5, True, "1"])
def test_batch_identity_rejects_noninteger_values(field, value):
    from vrl.generation.execution.sample_batches import GenerationSampleBatch
    from vrl.generation.types import GenerationRequest

    values = {"prompt_index": 0, "sample_start": 0, "sample_count": 1}
    values[field] = value
    with pytest.raises(ValueError, match=f"{field} must be an integer"):
        GenerationSampleBatch(**values)
    request = GenerationRequest(
        request_id="identity",
        family="test",
        task="t2i",
        inputs=["p"],
        samples_per_prompt=2,
    )
    with pytest.raises(ValueError, match=f"{field} must be an integer"):
        request.validate_batch_range(**values)


@pytest.mark.parametrize("field", ["samples_per_prompt", "samples_per_generation_batch"])
@pytest.mark.parametrize("value", [1.5, True, "2"])
def test_request_rejects_noninteger_sample_counts(field, value):
    from vrl.generation.types import GenerationRequest

    values = {"samples_per_prompt": 2, field: value}
    with pytest.raises(ValueError, match=field):
        GenerationRequest(request_id="counts", family="test", task="t2i", inputs=["p"], **values)


def test_oom_retry_releases_failed_forward_locals_before_emptying_cache(monkeypatch):
    import weakref

    import torch

    from vrl.generation.execution import sample_batches

    retained = []

    def forward(batch):
        temporary = torch.ones(2)
        if batch.sample_count > 1:
            retained.append(weakref.ref(temporary))
            raise RuntimeError("CUDA out of memory")
        return batch.sample_count

    def empty_cache():
        assert retained and retained[-1]() is None

    monkeypatch.setattr(sample_batches, "empty_cuda_cache", empty_cache)
    result = sample_batches.run_sample_batches_with_oom_retry(
        [sample_batches.GenerationSampleBatch(0, 0, 2)], forward
    )
    assert result == [1, 1]


@pytest.mark.parametrize("message, count", [("shape mismatch", 2), ("CUDA out of memory", 1)])
def test_terminal_batch_failure_keeps_forward_traceback_locals(message, count):
    from vrl.generation.execution.sample_batches import (
        GenerationSampleBatch,
        run_sample_batches_with_oom_retry,
    )

    marker = object()

    def forward(batch):
        diagnostic = marker  # noqa: F841 - inspected through the preserved traceback below
        raise RuntimeError(message)

    with pytest.raises(RuntimeError, match=message) as caught:
        run_sample_batches_with_oom_retry([GenerationSampleBatch(0, 0, count)], forward)
    trace = caught.value.__traceback__
    while trace.tb_next is not None:
        trace = trace.tb_next
    assert trace.tb_frame.f_locals["diagnostic"] is marker
