"""Reference conditioning follows the prompt-indexed generation input."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from PIL import Image

from vrl.generation import GenerationInput, GenerationRequest
from vrl.generation.bindings.full_sequence_denoise.executor import ReferenceConditionedBatches
from vrl.generation.execution.sample_batches import GenerationSampleBatch


def _batch(prompt_index: int) -> GenerationSampleBatch:
    return GenerationSampleBatch(
        prompt_index=prompt_index,
        sample_start=0,
        sample_count=1,
    )


@pytest.mark.parametrize("path_object", [False, True])
def test_reference_conditioning_selects_the_batch_prompt_input(tmp_path, path_object) -> None:
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

    image = executor._reference_image_for_batch(request, _batch(1))
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
        ReferenceConditionedBatches()._reference_image_for_batch(
            request,
            _batch(0),
        )


@pytest.mark.parametrize("family", ["cosmos", "wan"])
def test_encode_and_prepare_share_the_loaded_reference(tmp_path, monkeypatch, family) -> None:
    import torch

    from vrl.models.families.cosmos.predict2.runtime import CosmosBatchExecutor
    from vrl.models.families.wan_2_1.runtime import Wan_2_1I2VBatchExecutor

    path = tmp_path / "reference.png"
    Image.new("RGB", (2, 2), (255, 0, 0)).save(path)
    opens = []
    open_image = Image.open

    def counted_open(path):
        opens.append(path)
        return open_image(path)

    monkeypatch.setattr(Image, "open", counted_open)
    model = SimpleNamespace(
        encode_prompt=lambda *args, **kwargs: {
            "prompt_embeds": torch.ones(1, 2),
            "reference_image": kwargs["reference_image"],
        },
    )
    executor_cls = CosmosBatchExecutor if family == "cosmos" else Wan_2_1I2VBatchExecutor
    executor = executor_cls(model)
    request = GenerationRequest(
        request_id="shared-reference",
        family=executor.family,
        task=executor.task,
        inputs=[GenerationInput(prompt="prompt", reference_image=path)],
        samples_per_prompt=1,
    )
    encoded = executor.encode_prompt_for_batch(
        generation_request=request,
        video_request=SimpleNamespace(negative_prompt=None),
        params=SimpleNamespace(text_encode_kwargs=lambda: {}),
        batch=_batch(0),
    )
    batch_encoded = executor.expand_conditioning_to_batch(
        encoded=encoded, generation_request=request, batch=_batch(0)
    )
    prepare = executor.build_prepare_kwargs(
        encoded=encoded, generation_request=request, batch=_batch(0)
    )

    assert opens == [path]
    assert prepare["reference_image"] is encoded["reference_image"]
    assert batch_encoded["reference_image"] is encoded["reference_image"]
