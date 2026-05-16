"""NextStep AR request parsing tests."""

from __future__ import annotations

from vrl.engine import GenerationRequest
from vrl.engine.ar import ActiveSequence, ARRequestLayout


def test_nextstep_ar_spec_carries_scheduler_batch_size() -> None:
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

    spec = ARRequestLayout().parse_spec(request)
    sequence = ActiveSequence(
        request_id=request.request_id,
        sample_id="s0",
        family=request.family,
        task=request.task,
        tokenizer_key="nextstep_1",
        dtype="bfloat16",
        max_new_tokens=spec.image_token_num,
    )

    assert spec.ar_scheduler_batch_size == 3
    assert sequence.key.max_new_tokens == 8
