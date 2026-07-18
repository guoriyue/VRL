"""Reference conditioning follows the prompt-indexed generation input."""

from __future__ import annotations

import pytest

from vrl.generation import GenerationInput, GenerationRequest
from vrl.generation.bindings.joint_denoise.executor import ReferenceConditionedChunks
from vrl.generation.execution.chunks import SampleChunk


def _chunk(prompt_index: int, prompt: str) -> SampleChunk:
    return SampleChunk(
        prompt_index=prompt_index,
        prompt=prompt,
        sample_start=0,
        sample_count=1,
    )


def test_reference_conditioning_selects_the_chunk_prompt_input(monkeypatch) -> None:
    monkeypatch.setattr(
        "vrl.generation.bindings.joint_denoise.executor.load_reference_image",
        lambda value: value,
    )
    request = GenerationRequest(
        request_id="reference-inputs",
        family="unit-i2v",
        task="i2v",
        inputs=[
            GenerationInput(prompt="first", reference_image="first.png"),
            GenerationInput(prompt="second", reference_image="second.png"),
        ],
        samples_per_prompt=1,
    )
    executor = ReferenceConditionedChunks()

    assert executor._reference_image_for_chunk(request, _chunk(1, "second")) == "second.png"


def test_reference_conditioning_rejects_missing_prompt_reference() -> None:
    request = GenerationRequest(
        request_id="missing-reference",
        family="unit-i2v",
        task="i2v",
        inputs=[GenerationInput(prompt="prompt")],
        samples_per_prompt=1,
    )

    with pytest.raises(ValueError, match="requires reference_image for prompt index 0"):
        ReferenceConditionedChunks()._reference_image_for_chunk(
            request,
            _chunk(0, "prompt"),
        )
