"""NextStep AR request parsing tests."""

from __future__ import annotations

import pytest

from vrl.generation import GenerationRequest
from vrl.generation.ar import ActiveSequence, ARRequestLayout


def test_nextstep_ar_sampling_params_carry_scheduler_batch_size() -> None:
    request = GenerationRequest(
        request_id="req",
        family="nextstep_1",
        task="ar_t2i",
        prompts=["draw text"],
        samples_per_prompt=2,
        sampling={
            "image_token_num": 8,
            "image_size": 256,
            "max_text_length": 16,
            "use_ar_scheduler": True,
            "ar_scheduler_batch_size": 3,
        },
    )

    params = ARRequestLayout().parse_sampling_params(request)
    sequence = ActiveSequence(
        request_id=request.request_id,
        sample_id="s0",
        family=request.family,
        task=request.task,
        tokenizer_key="nextstep_1",
        dtype="bfloat16",
        max_new_tokens=params.image_token_num,
    )

    assert params.ar_scheduler_batch_size == 3
    assert sequence.key.max_new_tokens == 8


def test_ar_layout_requires_shape_sampling_fields() -> None:
    for missing_key in ("image_token_num", "image_size", "max_text_length"):
        sampling = {
            "image_token_num": 8,
            "image_size": 256,
            "max_text_length": 16,
        }
        sampling.pop(missing_key)
        request = GenerationRequest(
            request_id="req",
            family="nextstep_1",
            task="ar_t2i",
            prompts=["draw text"],
            samples_per_prompt=1,
            sampling=sampling,
        )

        with pytest.raises(ValueError, match=f"request.sampling.{missing_key}"):
            ARRequestLayout().parse_sampling_params(request)
