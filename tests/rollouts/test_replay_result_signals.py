"""Tests for ReplayResult lookup helpers and signal source ownership."""

from __future__ import annotations

import pytest
import torch

from vrl.engine import GenerationRequest, GenerationSampleSpec
from vrl.engine.trajectory import build_ar_discrete_trajectory, build_training_view
from vrl.models.interfaces import ReplayResult, ReplaySegmentResult
from vrl.rollouts.batch import RolloutBatch
from vrl.rollouts.evaluators.replay_result import (
    require_replay_segment,
    require_replay_value,
)
from vrl.rollouts.evaluators.trajectory import segment_signal_from_batch


def _request() -> GenerationRequest:
    return GenerationRequest(
        request_id="req",
        family="janus_pro",
        task="ar_t2i",
        prompts=["draw text"],
        samples_per_prompt=2,
    )


def _sample_specs() -> list[GenerationSampleSpec]:
    request = _request()
    return [
        GenerationSampleSpec(
            prompt_index=0,
            sample_index=index,
            prompt=request.prompts[0],
            prompt_id="p0",
            group_id="g0",
            sample_id=f"s{index}",
            trajectory_id=f"t{index}",
            seed=None,
        )
        for index in range(2)
    ]


def _discrete_batch() -> tuple[RolloutBatch, torch.Tensor, torch.Tensor]:
    token_ids = torch.tensor([[1, 2], [2, 3]])
    old_log_prob = torch.tensor([[-0.1, -0.2], [-0.3, -0.4]])
    token_mask = torch.tensor([[1.0, 0.0], [1.0, 1.0]])
    trajectory = build_ar_discrete_trajectory(
        request=_request(),
        sample_specs=_sample_specs(),
        token_ids=token_ids,
        token_log_probs=old_log_prob,
        token_mask=token_mask,
        prompt_input_ids=torch.ones(2, 3, dtype=torch.long),
        prompt_attention_mask=torch.ones(2, 3, dtype=torch.long),
        uncond_input_ids=torch.zeros(2, 3, dtype=torch.long),
        uncond_attention_mask=torch.ones(2, 3, dtype=torch.long),
        context={"model_family": "janus_pro"},
    )
    batch = RolloutBatch(
        observations=torch.ones(2, 1, 3, dtype=torch.long),
        actions=token_ids,
        rewards=torch.zeros(2),
        dones=torch.ones(2, dtype=torch.bool),
        group_ids=torch.tensor([0, 0]),
        trajectory=trajectory,
        training_view=build_training_view(trajectory),
    )
    return batch, old_log_prob, token_mask


def test_require_replay_segment_supports_single_segment_ar_lookup() -> None:
    logits = torch.zeros(2, 2, 8)
    output = ReplayResult(
        segments={
            "image_tokens": ReplaySegmentResult(
                segment="image_tokens",
                values={"logits": logits},
            ),
        },
    )

    segment = require_replay_segment(output, "image_tokens")

    assert require_replay_value(segment, "logits") is logits


def test_require_replay_segment_supports_single_segment_diffusion_lookup() -> None:
    noise_pred = torch.ones(2, 4, 8, 8)
    output = ReplayResult(
        segments={
            "denoise": ReplaySegmentResult(
                segment="denoise",
                values={"noise_pred": noise_pred},
            ),
        },
    )

    segment = require_replay_segment(output, "denoise")

    assert require_replay_value(segment, "noise_pred") is noise_pred


def test_require_replay_segment_supports_multi_segment_r1_lookup() -> None:
    output = ReplayResult(
        segments={
            "selfcheck_text": ReplaySegmentResult(
                segment="selfcheck_text",
                values={"logits": torch.zeros(2, 3, 10)},
            ),
            "final_image": ReplaySegmentResult(
                segment="final_image",
                values={"logits": torch.zeros(2, 4, 10)},
            ),
        },
    )

    segment = require_replay_segment(output, "final_image")

    assert segment.segment == "final_image"


def test_require_replay_segment_fails_fast_for_missing_ref_segment() -> None:
    output = ReplayResult(
        segments={
            "selfcheck_text": ReplaySegmentResult(
                segment="selfcheck_text",
                values={"logits": torch.zeros(2, 3, 10)},
            ),
        },
    )

    with pytest.raises(KeyError, match="missing segment 'final_image'"):
        require_replay_segment(output, "final_image")


def test_require_replay_value_fails_fast_with_available_keys() -> None:
    segment = ReplaySegmentResult(
        segment="image_tokens",
        values={"image_logits": torch.zeros(2, 2, 8), "token_ids": torch.ones(2, 2)},
    )

    with pytest.raises(KeyError, match="missing required key 'logits'.*image_logits"):
        require_replay_value(segment, "logits")


def test_segment_signal_reads_old_logprob_mask_and_distribution_from_trajectory() -> None:
    batch, old_log_prob, token_mask = _discrete_batch()
    output = ReplayResult(
        segments={
            "image_tokens": ReplaySegmentResult(
                segment="image_tokens",
                values={
                    "logits": torch.zeros(2, 2, 8),
                    "old_log_prob": torch.full((2, 2), 99.0),
                    "mask": torch.zeros(2, 2),
                },
            ),
        },
    )
    segment = require_replay_segment(output, "image_tokens")

    signal = segment_signal_from_batch(
        batch,
        segment_name=segment.segment,
        log_prob=torch.full((2, 2), -1.0),
        distribution="wrong",
    )

    assert signal.distribution == "categorical"
    assert torch.equal(signal.old_log_prob, old_log_prob)
    assert torch.equal(signal.mask, token_mask)
