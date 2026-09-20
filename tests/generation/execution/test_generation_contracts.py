"""Tests for generation request contracts."""

from __future__ import annotations

import pytest

from vrl.generation import (
    GenerationRequest,
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
    )


def test_generation_request_validation() -> None:
    """``GenerationRequest`` rejects empty inputs, a non-positive ``samples_per_prompt`` and a
    negative ``policy_version`` at construction.
    """
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


def test_sample_rows_are_deterministic() -> None:
    """Checks build sample rows is deterministic."""
    request = _request()
    rows = request.sample_rows()

    assert [row.sample_id for row in rows] == [
        "req-1:prompt:0:sample:0",
        "req-1:prompt:0:sample:1",
    ]
    assert [(row.prompt_index, row.sample_index) for row in rows] == [(0, 0), (0, 1)]


@pytest.mark.parametrize("version", [True, False, 1.0, 1.5, "1", float("nan"), float("inf"), -1])
def test_request_rejects_noninteger_or_negative_policy_version(version):
    from dataclasses import replace

    with pytest.raises(ValueError, match=r"GenerationRequest\.policy_version"):
        replace(_request(), policy_version=version)


@pytest.mark.parametrize("version", [None, 0, 1, 123])
def test_request_preserves_optional_integer_policy_version(version):
    from dataclasses import replace

    assert replace(_request(), policy_version=version).policy_version is version


@pytest.mark.parametrize("seeds", [(1, 2, 3), (1,), (1, -1)])
def test_request_initial_noise_seeds_are_one_nonnegative_int_per_sample_row(seeds) -> None:
    def build(inputs: list[str], initial_noise_seeds) -> GenerationRequest:
        return GenerationRequest(
            "r", "sd3_5", "t2i", inputs, 2, initial_noise_seeds=initial_noise_seeds
        )

    with pytest.raises(ValueError, match="initial_noise_seeds"):
        build(["p0"], seeds)

    assert build(["p0", "p1"], [7, 7, 8, 8]).initial_noise_seeds == (7, 7, 8, 8)
    assert build(["p0"], None).initial_noise_seeds is None
