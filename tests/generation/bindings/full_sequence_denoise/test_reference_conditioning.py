"""Reference conditioning follows the prompt-indexed generation input."""

from __future__ import annotations

import pytest
from PIL import Image

from vrl.generation import GenerationInput, GenerationRequest
from vrl.generation.bindings.full_sequence_denoise.executor import ReferenceConditionedBatches
from vrl.generation.execution.sample_batches import GenerationSampleBatch


def _chunk(prompt_index: int) -> GenerationSampleBatch:
    return GenerationSampleBatch(
        prompt_index=prompt_index,
        sample_start=0,
        sample_count=1,
    )


@pytest.mark.parametrize("path_object", [False, True])
def test_reference_conditioning_selects_the_chunk_prompt_input(tmp_path, path_object) -> None:
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    Image.new("RGB", (2, 2), (255, 0, 0)).save(first)
    Image.new("RGB", (2, 2), (0, 255, 0)).save(second)
    request = GenerationRequest(
        request_id="reference-inputs",
        family="unit-i2v",
        task="i2v",
        inputs=[
            GenerationInput(prompt="first", reference_image=first if path_object else str(first)),
            GenerationInput(
                prompt="second", reference_image=second if path_object else str(second)
            ),
        ],
        samples_per_prompt=1,
    )
    executor = ReferenceConditionedBatches()

    image = executor._reference_image_for_chunk(request, _chunk(1))
    assert image.mode == "RGB"
    assert image.getpixel((0, 0)) == (0, 255, 0)


def test_reference_conditioning_rejects_missing_prompt_reference() -> None:
    request = GenerationRequest(
        request_id="missing-reference",
        family="unit-i2v",
        task="i2v",
        inputs=[GenerationInput(prompt="prompt")],
        samples_per_prompt=1,
    )

    with pytest.raises(ValueError, match="requires reference_image for prompt index 0"):
        ReferenceConditionedBatches()._reference_image_for_chunk(
            request,
            _chunk(0),
        )
