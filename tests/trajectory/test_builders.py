"""Behavioral tests for the public trajectory-builder facades."""

from __future__ import annotations

import pytest
import torch

import vrl.trajectory as trajectory_api
from vrl.generation.execution.ids import build_sample_rows
from vrl.generation.types import GenerationRequest
from vrl.trajectory import (
    TrajectoryValidationError,
    build_ar_multisegment_trajectory,
    build_diffusion_trajectory,
)
from vrl.trajectory.ops import stack_trajectory_batches


def test_diffusion_replay_extras_only_declare_sample_axis_when_sample_aligned() -> None:
    request = GenerationRequest(
        request_id="builder-sample-alignment",
        family="test",
        task="t2i",
        inputs=["draw"],
        samples_per_prompt=2,
    )
    sample_rows = build_sample_rows(request)
    old_log_prob = torch.zeros(2, 2)

    trajectory = build_diffusion_trajectory(
        request=request,
        sample_rows=sample_rows,
        observations=torch.zeros(2, 2, 1),
        actions=torch.zeros(2, 2, 1),
        old_log_prob=old_log_prob,
        timesteps=torch.zeros(2, 2),
        kl=torch.zeros_like(old_log_prob),
        replay_tensors={
            "per_sample": torch.tensor([1.0, 2.0]),
            "scalar_tensor": torch.tensor(1.0),
            "python_scalar": 1.0,
        },
        context={},
    )

    tensors = trajectory.segments["denoise"].tensors
    assert tensors["per_sample"].axes == ("sample",)
    assert "scalar_tensor" not in tensors
    assert "python_scalar" not in tensors


def test_multisegment_primary_is_typed_and_not_mirrored_in_context() -> None:
    request = GenerationRequest(
        request_id="primary",
        family="janus_pro_r1",
        task="ar_t2i_r1",
        inputs=["draw"],
        samples_per_prompt=1,
    )

    trajectory = build_ar_multisegment_trajectory(
        request=request,
        sample_rows=build_sample_rows(request),
        segments={
            "initial_image": _segment_payload(train=True),
            "final_image": _segment_payload(train=True),
        },
        primary_segment="final_image",
        context={"temperature": 1.0},
    )

    assert trajectory.primary_segment == "final_image"
    assert trajectory.context == {"temperature": 1.0}


@pytest.mark.parametrize(
    ("primary_segment", "final_trainable", "message"),
    [
        ("missing", True, "is unknown"),
        ("final_image", False, "must reference a trainable segment"),
    ],
)
def test_multisegment_primary_rejects_unknown_or_nontrainable_segment(
    primary_segment: str,
    final_trainable: bool,
    message: str,
) -> None:
    request = GenerationRequest(
        request_id="invalid-primary",
        family="janus_pro_r1",
        task="ar_t2i_r1",
        inputs=["draw"],
        samples_per_prompt=1,
    )

    with pytest.raises(TrajectoryValidationError, match=message):
        build_ar_multisegment_trajectory(
            request=request,
            sample_rows=build_sample_rows(request),
            segments={
                "initial_image": _segment_payload(train=True),
                "final_image": _segment_payload(train=final_trainable),
            },
            primary_segment=primary_segment,
            context={},
        )


def test_stack_rejects_different_primary_segments() -> None:
    request = GenerationRequest(
        request_id="stack-primary",
        family="janus_pro_r1",
        task="ar_t2i_r1",
        inputs=["draw"],
        samples_per_prompt=1,
    )
    trajectories = [
        build_ar_multisegment_trajectory(
            request=request,
            sample_rows=build_sample_rows(request),
            segments={
                "initial_image": _segment_payload(train=True),
                "final_image": _segment_payload(train=True),
            },
            primary_segment=primary,
            context={},
        )
        for primary in ("initial_image", "final_image")
    ]

    with pytest.raises(ValueError, match="different primary segments"):
        stack_trajectory_batches(trajectories)


@pytest.mark.parametrize(
    "symbol",
    ["LossUnit", "TrainingView", "build_training_view", "replay_input_ref"],
)
def test_derived_training_view_symbols_are_not_public(symbol: str) -> None:
    assert not hasattr(trajectory_api, symbol)


def _segment_payload(*, train: bool) -> dict[str, object]:
    return {
        "token_ids": torch.ones(1, 2, dtype=torch.long),
        "token_log_probs": torch.zeros(1, 2),
        "token_mask": torch.ones(1, 2),
        "prompt_embeds": torch.zeros(1, 3, 4),
        "attention_mask": torch.ones(1, 3, dtype=torch.long),
        "train": train,
        "visual": True,
    }
