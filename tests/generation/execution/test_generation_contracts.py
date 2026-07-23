"""Tests for generation request contracts."""

from __future__ import annotations

from dataclasses import fields

import pytest

from vrl.generation import (
    GenerationOutput,
    GenerationRequest,
    GenerationSampleRow,
    build_sample_rows,
)


def _request(
    request_id: str = "req-1",
    *,
    height: int = 512,
    width: int = 512,
    num_steps: int = 10,
    seed: int | None = 7,
) -> GenerationRequest:
    sampling = {
        "height": height,
        "width": width,
        "num_steps": num_steps,
    }
    if seed is not None:
        sampling["seed"] = seed
    return GenerationRequest(
        request_id=request_id,
        family="sd3_5",
        task="t2i",
        inputs=["a test prompt"],
        samples_per_prompt=2,
        sampling=sampling,
        metadata={"dataset": "unit"},
    )


def test_generation_request_validation() -> None:
    """Checks generation request validation."""
    with pytest.raises(ValueError, match="inputs"):
        GenerationRequest(
            request_id="req",
            family="sd3_5",
            task="t2i",
            inputs=[],
            samples_per_prompt=1,
        )

    with pytest.raises(ValueError, match="samples_per_prompt"):
        GenerationRequest(
            request_id="req",
            family="sd3_5",
            task="t2i",
            inputs=["x"],
            samples_per_prompt=0,
        )

    with pytest.raises(ValueError, match="policy_version"):
        GenerationRequest(
            request_id="req",
            family="sd3_5",
            task="t2i",
            inputs=["x"],
            samples_per_prompt=1,
            policy_version=-1,
        )


@pytest.mark.parametrize(
    ("removed_name", "value"),
    [
        ("return_artifacts", {"output"}),
        ("priority", 1),
    ],
)
def test_generation_request_rejects_removed_noop_arguments(
    removed_name: str,
    value: object,
) -> None:
    """Checks removed request knobs cannot be silently accepted."""
    with pytest.raises(TypeError, match=removed_name):
        GenerationRequest(
            request_id="req",
            family="sd3_5",
            task="t2i",
            inputs=["x"],
            samples_per_prompt=1,
            **{removed_name: value},
        )


def test_generation_payloads_exclude_removed_duplicate_fields() -> None:
    """Checks sample and output payloads retain one prompt source of truth."""
    assert "prompt_id" not in {field.name for field in fields(GenerationSampleRow)}
    assert "prompts" not in {field.name for field in fields(GenerationOutput)}


def test_build_sample_rows_is_deterministic() -> None:
    """Checks build sample rows is deterministic."""
    request = _request()
    rows = build_sample_rows(request)

    assert [row.sample_id for row in rows] == [
        "req-1:prompt:0:sample:0",
        "req-1:prompt:0:sample:1",
    ]
    assert [row.seed for row in rows] == [7, 8]
    assert {row.group_id for row in rows} == {"req-1:prompt:0"}
