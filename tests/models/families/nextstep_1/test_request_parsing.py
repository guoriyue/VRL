"""NextStep AR request parsing tests."""

from __future__ import annotations

import pytest
import torch
from omegaconf import OmegaConf

from vrl.families.registry import get_model_family_entry
from vrl.generation import GenerationRequest
from vrl.generation.bindings.token_autoregressive import ARRequestLayout
from vrl.generation.execution.ids import build_sample_rows
from vrl.models.families.nextstep_1.runtime import (
    NextStep1ARChunkResult,
    NextStep1ChunkGatherer,
)


def test_nextstep_ar_sampling_params_carry_scheduler_batch_size() -> None:
    """Checks NextStep AR sampling params carry scheduler batch size."""
    request = GenerationRequest(
        request_id="req",
        family="nextstep_1",
        task="ar_t2i",
        inputs=["draw text"],
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

    assert params.ar_scheduler_batch_size == 3
    assert params.image_token_num == 8


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
            inputs=["draw text"],
            samples_per_prompt=1,
            sampling=sampling,
        )

        with pytest.raises(ValueError, match=f"request.sampling.{missing_key}"):
            ARRequestLayout().parse_sampling_params(request)


@pytest.mark.parametrize(
    ("mode", "expected"),
    [("off", False), ("full", True)],
)
def test_replay_build_resolves_gradient_checkpointing_mode(
    mode: str,
    expected: bool,
) -> None:
    cfg = OmegaConf.create(
        {
            "model": {"family": "nextstep_1", "use_lora": False},
            "precision": {
                "float32_precision": "tf32",
                "training": {"dtype": "fp32"},
                "rollout": {"dtype": "fp32"},
            },
            "actor": {"gradient_checkpointing": mode},
        },
    )

    build = get_model_family_entry("nextstep_1").resolve_model_build(
        cfg,
        "cpu",
        for_rollout=False,
    )

    assert build.family == "nextstep_1"
    assert build.model_config["gradient_checkpointing"] is expected


def test_replay_build_rejects_selective_gradient_checkpointing() -> None:
    cfg = OmegaConf.create(
        {
            "model": {"family": "nextstep_1", "use_lora": False},
            "precision": {
                "float32_precision": "tf32",
                "training": {"dtype": "fp32"},
                "rollout": {"dtype": "fp32"},
            },
            "actor": {"gradient_checkpointing": "selective"},
        },
    )

    with pytest.raises(ValueError, match="does not support selective"):
        get_model_family_entry("nextstep_1").resolve_model_build(
            cfg,
            "cpu",
            for_rollout=False,
        )


def test_rollout_build_disables_gradient_checkpointing() -> None:
    cfg = OmegaConf.create(
        {
            "model": {"family": "nextstep_1", "use_lora": False},
            "precision": {
                "float32_precision": "tf32",
                "training": {"dtype": "fp32"},
                "rollout": {"dtype": "fp32"},
            },
            "actor": {"gradient_checkpointing": "full"},
        },
    )

    build = get_model_family_entry("nextstep_1").resolve_model_build(
        cfg,
        "cpu",
        for_rollout=True,
    )

    assert build.model_config["gradient_checkpointing"] is False


def test_nextstep_gather_uses_canonical_output_as_reward_source() -> None:
    """Decoded output stays outside replay state and remains the reward source."""
    request = GenerationRequest(
        request_id="req",
        family="nextstep_1",
        task="ar_t2i",
        inputs=["draw text"],
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
    assert torch.equal(output.output, images)
    assert "decoded" not in output.trajectory.segments
    assert output.trajectory.reward_views["image"].tensor_refs == ()
    assert output.trajectory.reward_views["image"].metadata == {
        "output_ref": "GenerationOutput.output"
    }
