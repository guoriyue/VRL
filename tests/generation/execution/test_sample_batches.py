"""Tests for generation executor sample batch helpers."""

from __future__ import annotations

from dataclasses import replace

import pytest


def test_sample_batch_plan_prompt_major() -> None:
    """Checks build prompt batches prompt major."""
    from vrl.generation.execution.sample_batches import GenerationSampleBatch

    batches = GenerationSampleBatch.plan_generation_batches(
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


@pytest.mark.parametrize("width", [0, True])
def test_sample_batches_reject_invalid_explicit_width(width):
    from vrl.generation.types import GenerationRequest

    request = GenerationRequest(
        request_id="width-check",
        family="test",
        task="t2i",
        inputs=["p"],
        samples_per_prompt=2,
    )
    with pytest.raises(ValueError, match="samples_per_generation_batch must be >= 1"):
        replace(request, samples_per_generation_batch=width).sample_batches()


@pytest.mark.parametrize("field", ["prompt_index", "sample_start", "sample_count"])
def test_batch_identity_rejects_noninteger_values(field):
    from vrl.generation.execution.sample_batches import GenerationSampleBatch
    from vrl.generation.types import GenerationRequest

    values = {"prompt_index": 0, "sample_start": 0, "sample_count": 1}
    values[field] = True
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
def test_request_rejects_noninteger_sample_counts(field):
    from vrl.generation.types import GenerationRequest

    values = {"samples_per_prompt": 2, field: True}
    with pytest.raises(ValueError, match=field):
        GenerationRequest(request_id="counts", family="test", task="t2i", inputs=["p"], **values)
