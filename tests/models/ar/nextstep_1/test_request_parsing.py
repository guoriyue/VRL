"""NextStep AR request parsing tests."""

from __future__ import annotations

import pytest
import torch
from omegaconf import OmegaConf

from vrl.generation import GenerationRequest
from vrl.generation.ar import ARRequestLayout
from vrl.generation.ar.decode_loop import ActiveSequence
from vrl.generation.execution.ids import build_sample_rows
from vrl.models.ar.build import extract_family_ar_runtime_spec
from vrl.models.ar.nextstep_1.runtime import (
    NextStep1ARChunkResult,
    NextStep1ChunkGatherer,
)


def test_nextstep_ar_sampling_params_carry_scheduler_batch_size() -> None:
    """Checks NextStep AR sampling params carry scheduler batch size."""
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
    """Checks AR layout requires shape sampling fields."""
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


def test_descriptor_extractor_carries_actor_gradient_checkpointing() -> None:
    cfg = OmegaConf.create(
        {
            "model": {"family": "nextstep_1", "use_lora": False},
            "precision": "fp32",
            "actor": {"gradient_checkpointing": True},
        },
    )

    spec = extract_family_ar_runtime_spec(cfg, "cpu", "float32")

    assert spec.family == "nextstep_1"
    assert spec.model_config["gradient_checkpointing"] is True


def test_nextstep_gather_derives_reward_image_from_canonical_output() -> None:
    """Decoded output is the single source for both generation and reward views."""
    request = GenerationRequest(
        request_id="req",
        family="nextstep_1",
        task="ar_t2i",
        prompts=["draw text"],
        samples_per_prompt=2,
        sampling={"image_token_num": 2},
    )
    sample_rows = build_sample_rows(request)
    images = torch.arange(24, dtype=torch.float32).reshape(2, 3, 2, 2)
    chunk = NextStep1ARChunkResult(
        prompt_index=0,
        sample_start=0,
        sample_count=2,
        output=images,
        tokens=torch.zeros(2, 2, 4),
        saved_noise=torch.zeros(2, 2, 4),
        log_probs=torch.zeros(2, 2),
        prompt_input_ids=torch.zeros(2, 3, dtype=torch.long),
        prompt_attention_mask=torch.ones(2, 3, dtype=torch.long),
        uncond_input_ids=torch.zeros(2, 3, dtype=torch.long),
        uncond_attention_mask=torch.ones(2, 3, dtype=torch.long),
        context={},
    )

    output = NextStep1ChunkGatherer().gather_chunks(
        request,
        sample_rows,
        [chunk],
    )
    reward_images = output.trajectory.segments["decoded"].tensors["images_for_reward"].value

    assert output.output is reward_images
    assert torch.equal(output.output, images)
