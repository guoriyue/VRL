"""Generation sample identity helpers."""

from __future__ import annotations

from vrl.generation.types import GenerationRequest, GenerationSampleRow


def build_sample_rows(request: GenerationRequest) -> list[GenerationSampleRow]:
    """Build deterministic sample rows from a generation request."""

    rows: list[GenerationSampleRow] = []
    for prompt_index, request_input in enumerate(request.inputs):
        prompt = request_input.prompt
        prompt_id = f"{request.request_id}:prompt:{prompt_index}"
        for sample_index in range(request.samples_per_prompt):
            sample_id = f"{prompt_id}:sample:{sample_index}"
            rows.append(
                GenerationSampleRow(
                    prompt_index=prompt_index,
                    sample_index=sample_index,
                    prompt=prompt,
                    sample_id=sample_id,
                )
            )
    return rows


__all__ = ["build_sample_rows"]
