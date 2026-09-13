"""Doubles shared by the replay evaluator tests: one Janus-Pro request and its rows."""

from __future__ import annotations

from vrl.generation.types import GenerationRequest, GenerationSampleRow


def janus_request() -> GenerationRequest:
    return GenerationRequest(
        request_id="req",
        family="janus_pro",
        task="ar_t2i",
        inputs=["draw text"],
        samples_per_prompt=2,
    )


def janus_sample_rows() -> list[GenerationSampleRow]:
    request = janus_request()
    return [
        GenerationSampleRow(
            prompt_index=0,
            sample_index=index,
            prompt=request.prompts[0],
            sample_id=f"s{index}",
        )
        for index in range(2)
    ]
