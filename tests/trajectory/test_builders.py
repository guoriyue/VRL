"""Behavioral tests for the public trajectory-builder facades."""

from __future__ import annotations

import pytest
import torch

import vrl.trajectory as trajectory_api
from vrl.generation.execution.ids import build_sample_rows
from vrl.generation.types import GenerationRequest
from vrl.trajectory import (
    TrajectoryBatch,
    TrajectoryMetrics,
    TrajectoryValidationError,
    TrajectoryValidator,
    build_ar_continuous_trajectory,
    build_ar_discrete_trajectory,
    build_ar_multisegment_trajectory,
    build_chunk_autoregressive_denoise_trajectory,
    build_chunk_autoregressive_generation_trajectory,
    build_diffusion_trajectory,
)


def _axis_lengths(trajectory: TrajectoryBatch) -> dict[str, int]:
    return {name: axis.length for name, axis in trajectory.axes.items() if axis.length is not None}


def test_all_builders_derive_structure_without_metric_copies() -> None:
    for name, trajectory, expected_axis_lengths in _structural_trajectories():
        assert len(trajectory.sample_rows) == expected_axis_lengths["sample"], name
        assert _axis_lengths(trajectory) == expected_axis_lengths, name
        assert trajectory.metrics.values == {}, name


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("num_samples", 2),
        ("axis_lengths", {"sample": 2}),
    ],
)
def test_trajectory_metrics_rejects_removed_structural_fields(
    field_name: str,
    value: object,
) -> None:
    with pytest.raises(TypeError, match=field_name):
        TrajectoryMetrics(**{field_name: value})


def test_builder_rejects_tensor_rows_that_disagree_with_sample_rows() -> None:
    request = _request()
    sample_rows = build_sample_rows(request)
    token_ids = torch.ones(1, 2, dtype=torch.long)

    with pytest.raises(
        TrajectoryValidationError,
        match=r"token_ids axis 'sample' has shape 1, expected 2",
    ):
        build_ar_discrete_trajectory(
            request=request,
            sample_rows=sample_rows,
            token_ids=token_ids,
            token_log_probs=torch.zeros(1, 2),
            token_mask=torch.ones(1, 2),
            prompt_input_ids=torch.ones(2, 3, dtype=torch.long),
            prompt_attention_mask=torch.ones(2, 3, dtype=torch.long),
            uncond_input_ids=torch.zeros(2, 3, dtype=torch.long),
            uncond_attention_mask=torch.ones(2, 3, dtype=torch.long),
            context={},
        )


def test_metrics_values_keep_provenance_validation_without_structural_ownership() -> None:
    trajectory = _structural_trajectories()[0][1]

    trajectory.metrics.values = {"source": "unit-test"}
    assert TrajectoryValidator(trajectory).validate_batch() is trajectory

    trajectory.metrics.values = {"runtime": object()}
    with pytest.raises(TrajectoryValidationError, match="runtime-only state"):
        TrajectoryValidator(trajectory).validate_batch()


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
            "initial_image": _segment_payload(),
            "final_image": _segment_payload(),
        },
        primary_segment="final_image",
        context={"temperature": 1.0},
    )

    assert trajectory.primary_segment == "final_image"
    assert trajectory.context == {"temperature": 1.0}
    assert not hasattr(trajectory.segments["initial_image"], "advantage_scope")


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
        sampling={
            "train_segments": {
                "initial_image": True,
                "final_image": final_trainable,
            },
        },
    )

    with pytest.raises(TrajectoryValidationError, match=message):
        build_ar_multisegment_trajectory(
            request=request,
            sample_rows=build_sample_rows(request),
            segments={
                "initial_image": _segment_payload(),
                "final_image": _segment_payload(),
            },
            primary_segment=primary_segment,
            context={},
        )


@pytest.mark.parametrize(
    "symbol",
    [
        "AdvantageScope",
        "LossUnit",
        "TrainingView",
        "build_training_view",
        "replay_input_ref",
    ],
)
def test_derived_training_view_symbols_are_not_public(symbol: str) -> None:
    assert not hasattr(trajectory_api, symbol)


def _segment_payload(
    *,
    batch_size: int = 1,
    token_count: int = 2,
) -> dict[str, object]:
    # Trainability is driven by ``request.sampling['train_segments']`` (or the
    # ``visual`` default when the request omits it), never by a payload key.
    return {
        "token_ids": torch.ones(batch_size, token_count, dtype=torch.long),
        "token_log_probs": torch.zeros(batch_size, token_count),
        "token_mask": torch.ones(batch_size, token_count),
        "prompt_embeds": torch.zeros(batch_size, 3, 4),
        "attention_mask": torch.ones(batch_size, 3, dtype=torch.long),
        "visual": True,
    }


def _request() -> GenerationRequest:
    return GenerationRequest(
        request_id="derived-structure",
        family="test",
        task="t2i",
        inputs=["draw"],
        samples_per_prompt=2,
    )


def _structural_trajectories() -> list[tuple[str, TrajectoryBatch, dict[str, int]]]:
    request = _request()
    sample_rows = build_sample_rows(request)
    batch_size = len(sample_rows)

    diffusion_steps = 3
    diffusion_policy_shape = (batch_size, diffusion_steps)
    diffusion = build_diffusion_trajectory(
        request=request,
        sample_rows=sample_rows,
        observations=torch.zeros(*diffusion_policy_shape, 1),
        actions=torch.ones(*diffusion_policy_shape, 1),
        old_log_prob=torch.zeros(diffusion_policy_shape),
        timesteps=torch.zeros(diffusion_policy_shape),
        kl=torch.zeros(diffusion_policy_shape),
        replay_tensors={},
        context={},
    )

    chunk_count = 2
    transition_count = 3
    chunk_policy_shape = (batch_size, chunk_count, transition_count)
    chunk_denoise = build_chunk_autoregressive_denoise_trajectory(
        request=request,
        sample_rows=sample_rows,
        observations=torch.zeros(*chunk_policy_shape, 1),
        actions=torch.ones(*chunk_policy_shape, 1),
        old_log_prob=torch.zeros(chunk_policy_shape),
        mask=torch.ones(chunk_policy_shape),
        timesteps=torch.zeros(chunk_policy_shape),
        finalized_chunk_latents=torch.zeros(batch_size, chunk_count, 1),
        replay_tensors={},
        context={},
    )

    generation_chunk_count = 4
    chunk_generation = build_chunk_autoregressive_generation_trajectory(
        request=request,
        sample_rows=sample_rows,
        output=torch.zeros(batch_size, 3, 4, 4),
        temporal_chunk_count=generation_chunk_count,
        context={},
    )

    token_count = 3
    token_ids = torch.ones(batch_size, token_count, dtype=torch.long)
    prompt_input_ids = torch.ones(batch_size, 4, dtype=torch.long)
    prompt_attention_mask = torch.ones_like(prompt_input_ids)
    uncond_input_ids = torch.zeros_like(prompt_input_ids)
    uncond_attention_mask = torch.ones_like(prompt_input_ids)
    discrete = build_ar_discrete_trajectory(
        request=request,
        sample_rows=sample_rows,
        token_ids=token_ids,
        token_log_probs=torch.zeros(batch_size, token_count),
        token_mask=torch.ones(batch_size, token_count),
        prompt_input_ids=prompt_input_ids,
        prompt_attention_mask=prompt_attention_mask,
        uncond_input_ids=uncond_input_ids,
        uncond_attention_mask=uncond_attention_mask,
        context={},
    )
    continuous = build_ar_continuous_trajectory(
        request=request,
        sample_rows=sample_rows,
        tokens=torch.ones(batch_size, token_count, 4),
        saved_noise=torch.zeros(batch_size, token_count, 4),
        token_log_probs=torch.zeros(batch_size, token_count),
        token_mask=torch.ones(batch_size, token_count),
        prompt_input_ids=prompt_input_ids,
        prompt_attention_mask=prompt_attention_mask,
        uncond_input_ids=uncond_input_ids,
        uncond_attention_mask=uncond_attention_mask,
        context={},
    )

    initial_token_count = 2
    final_token_count = 3
    multisegment = build_ar_multisegment_trajectory(
        request=request,
        sample_rows=sample_rows,
        segments={
            "initial_image": _segment_payload(
                batch_size=batch_size,
                token_count=initial_token_count,
            ),
            "final_image": _segment_payload(
                batch_size=batch_size,
                token_count=final_token_count,
            ),
        },
        primary_segment="final_image",
        context={},
    )

    return [
        (
            "diffusion",
            diffusion,
            {"sample": batch_size, "denoise": diffusion_steps},
        ),
        (
            "chunk_denoise",
            chunk_denoise,
            {
                "sample": batch_size,
                "temporal_chunk": chunk_count,
                "denoise_transition": transition_count,
            },
        ),
        (
            "chunk_generation",
            chunk_generation,
            {"sample": batch_size, "temporal_chunk": generation_chunk_count},
        ),
        (
            "ar_discrete",
            discrete,
            {"sample": batch_size, "token": token_count},
        ),
        (
            "ar_continuous",
            continuous,
            {"sample": batch_size, "token": token_count},
        ),
        (
            "ar_multisegment",
            multisegment,
            {
                "sample": batch_size,
                "initial_image_token": initial_token_count,
                "final_image_token": final_token_count,
            },
        ),
    ]
