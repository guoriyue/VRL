"""Tests for trajectory-native signal adapters."""

from __future__ import annotations

import pytest
import torch

from vrl.engine import GenerationRequest, GenerationSampleSpec
from vrl.engine.trajectory import (
    build_ar_discrete_trajectory,
    build_diffusion_trajectory,
    build_training_view,
)
from vrl.rollouts.evaluators.trajectory import (
    evaluator_output_to_trajectory_signals,
    legacy_signal_batch_to_trajectory_signals,
    to_trajectory_signals,
    trajectory_signals_to_signal_batch,
)
from vrl.rollouts.evaluators.types import SegmentSignal, SignalBatch, TrajectorySignalBatch


def test_flow_signal_adapter_preserves_kl_fields_and_axis() -> None:
    trajectory = _diffusion_trajectory()
    training_view = build_training_view(trajectory)
    old_log_prob = trajectory.segments["denoise"].tensors["old_log_prob"].value
    new_log_prob = old_log_prob + 0.1
    ref_log_prob = old_log_prob - 0.2
    prev_sample_mean = torch.ones(2, 3, 1)
    ref_prev_sample_mean = torch.zeros(2, 3, 1)
    std_dev_t = torch.full((2, 3, 1), 0.5)
    dt = torch.full((2, 3, 1), 0.25)

    legacy = SignalBatch(
        log_prob=new_log_prob,
        ref_log_prob=ref_log_prob,
        prev_sample_mean=prev_sample_mean,
        ref_prev_sample_mean=ref_prev_sample_mean,
        std_dev_t=std_dev_t,
        dt=dt,
        dist_family="flow_matching",
    )

    signals = legacy_signal_batch_to_trajectory_signals(
        legacy,
        trajectory=trajectory,
        training_view=training_view,
    )

    denoise = signals.segments["denoise"]
    assert denoise.axis == "timestep"
    assert denoise.axes == ("sample", "timestep")
    assert denoise.distribution == "flow_matching"
    assert torch.equal(denoise.old_log_prob, old_log_prob)
    assert torch.equal(denoise.mask, torch.ones_like(old_log_prob))
    assert denoise.prev_sample_mean is prev_sample_mean
    assert denoise.ref_prev_sample_mean is ref_prev_sample_mean
    assert denoise.std_dev_t is std_dev_t
    assert denoise.dt is dt

    roundtrip = trajectory_signals_to_signal_batch(signals)
    assert torch.equal(roundtrip.log_prob, legacy.log_prob)
    assert torch.equal(roundtrip.ref_log_prob, ref_log_prob)
    assert roundtrip.prev_sample_mean is prev_sample_mean
    assert roundtrip.dist_family == "flow_matching"


def test_step_signal_adapter_requires_step_old_log_probs_when_trajectory_is_full() -> None:
    trajectory = _diffusion_trajectory()
    training_view = build_training_view(trajectory)
    full_old = trajectory.segments["denoise"].tensors["old_log_prob"].value
    step_old = full_old[:, 1]
    legacy = SignalBatch(
        log_prob=step_old + 0.1,
        dist_family="flow_matching",
    )

    with pytest.raises(RuntimeError, match="step-level old_log_probs"):
        legacy_signal_batch_to_trajectory_signals(
            legacy,
            trajectory=trajectory,
            training_view=training_view,
        )

    signals = legacy_signal_batch_to_trajectory_signals(
        legacy,
        trajectory=trajectory,
        training_view=training_view,
        old_log_probs=step_old,
    )

    denoise = signals.segments["denoise"]
    assert denoise.axes == ("sample",)
    assert torch.equal(denoise.old_log_prob, step_old)
    assert torch.equal(denoise.mask, torch.ones_like(step_old))


def test_token_signal_adapter_preserves_mask_and_roundtrips() -> None:
    trajectory = _janus_trajectory()
    training_view = build_training_view(trajectory)
    token_count = trajectory.axes["token"].length
    assert token_count is not None
    new_log_prob = torch.full((2, token_count), -0.25)
    ref_log_prob = torch.full((2, token_count), -0.5)
    mask = torch.tensor([[1.0, 1.0, 0.0, 0.0], [1.0, 0.0, 1.0, 0.0]])

    legacy = SignalBatch(
        log_prob=new_log_prob,
        ref_log_prob=ref_log_prob,
        dist_family="categorical",
        aux={"token_mask": mask},
    )

    signals = legacy_signal_batch_to_trajectory_signals(
        legacy,
        trajectory=trajectory,
        training_view=training_view,
    )

    segment = signals.segments["image_tokens"]
    assert segment.axis == "token"
    assert segment.distribution == "categorical"
    assert torch.equal(segment.mask, mask)

    roundtrip = trajectory_signals_to_signal_batch(signals)
    assert torch.equal(roundtrip.log_prob, new_log_prob)
    assert torch.equal(roundtrip.aux["token_mask"], mask)


def test_multisegment_signal_adapter_lifts_legacy_aux_segments() -> None:
    initial = SignalBatch(
        log_prob=torch.full((2, 2), -0.1),
        dist_family="categorical",
        aux={"token_mask": torch.ones(2, 2), "segment_modality": "image"},
    )
    final = SignalBatch(
        log_prob=torch.full((2, 3), -0.2),
        dist_family="categorical",
        aux={"token_mask": torch.ones(2, 3), "segment_modality": "image"},
    )
    old_initial = torch.zeros(2, 2)
    old_final = torch.zeros(2, 3)
    legacy = SignalBatch(
        log_prob=final.log_prob,
        dist_family="multisegment_categorical",
        aux={
            "segments": {"initial_image": initial, "final_image": final},
            "old_log_probs": {
                "initial_image": old_initial,
                "final_image": old_final,
            },
            "primary_segment": "final_image",
        },
    )

    signals = legacy_signal_batch_to_trajectory_signals(
        legacy,
        group_ids=torch.tensor([0, 0]),
    )

    assert set(signals.segments) == {"initial_image", "final_image"}
    assert signals.primary_segment == "final_image"
    assert torch.equal(signals.segments["initial_image"].old_log_prob, old_initial)
    assert signals.segments["final_image"].axis == "token"

    roundtrip = trajectory_signals_to_signal_batch(signals)
    assert roundtrip.dist_family == "multisegment_categorical"
    assert set(roundtrip.aux["segments"]) == {"initial_image", "final_image"}
    assert torch.equal(roundtrip.aux["old_log_probs"]["final_image"], old_final)


def test_to_trajectory_signals_rejects_legacy_signal_batch() -> None:
    with pytest.raises(TypeError, match="legacy_signal_batch_to_trajectory_signals"):
        to_trajectory_signals(SignalBatch(log_prob=torch.zeros(2)))


def test_evaluator_output_normalizer_accepts_native_trajectory_signals() -> None:
    native = TrajectorySignalBatch(
        segments={
            "image_tokens": SegmentSignal(
                name="image_tokens",
                segment="image_tokens",
                axis="token",
                axes=("sample", "token"),
                distribution="categorical",
                log_prob=torch.zeros(2, 3),
                old_log_prob=torch.zeros(2, 3),
                mask=torch.ones(2, 3),
            )
        },
        group_ids=torch.tensor([0, 0]),
        primary_segment="image_tokens",
    )

    assert evaluator_output_to_trajectory_signals(native) is native


def test_evaluator_output_normalizer_converts_legacy_signal_batch() -> None:
    trajectory = _janus_trajectory()
    training_view = build_training_view(trajectory)
    legacy = SignalBatch(
        log_prob=torch.full((2, 4), -0.25),
        dist_family="categorical",
        aux={"token_mask": torch.ones(2, 4)},
    )

    signals = evaluator_output_to_trajectory_signals(
        legacy,
        trajectory=trajectory,
        training_view=training_view,
    )

    assert signals.primary is signals.segments["image_tokens"]
    assert signals.primary.distribution == "categorical"


def test_trajectory_signal_batch_validates_shapes() -> None:
    with pytest.raises(ValueError, match="log_prob/old_log_prob"):
        TrajectorySignalBatch(
            segments={
                "denoise": SegmentSignal(
                    name="denoise",
                    segment="denoise",
                    axis="timestep",
                    axes=("sample", "timestep"),
                    distribution="flow_matching",
                    log_prob=torch.zeros(2, 3),
                    old_log_prob=torch.zeros(2),
                    mask=torch.ones(2, 3),
                )
            },
            group_ids=torch.tensor([0, 0]),
        )

    with pytest.raises(ValueError, match="log_prob/mask"):
        TrajectorySignalBatch(
            segments={
                "denoise": SegmentSignal(
                    name="denoise",
                    segment="denoise",
                    axis="timestep",
                    axes=("sample", "timestep"),
                    distribution="flow_matching",
                    log_prob=torch.zeros(2, 3),
                    old_log_prob=torch.zeros(2, 3),
                    mask=torch.ones(2),
                )
            },
            group_ids=torch.tensor([0, 0]),
        )


def _diffusion_trajectory():
    request = _request("sd3", "sd3_5", "t2i")
    sample_specs = _sample_specs(request, count=2)
    return build_diffusion_trajectory(
        request=request,
        sample_specs=sample_specs,
        observations=torch.zeros(2, 3, 1),
        actions=torch.ones(2, 3, 1),
        old_log_prob=torch.full((2, 3), -0.5),
        timesteps=torch.arange(3).repeat(2, 1),
        kl=torch.zeros(2, 3),
        training_extras={},
        context={"model_family": "sd3_5"},
    )


def _janus_trajectory():
    request = _request("janus", "janus_pro", "ar_t2i")
    sample_specs = _sample_specs(request, count=2)
    return build_ar_discrete_trajectory(
        request=request,
        sample_specs=sample_specs,
        token_ids=torch.arange(8).view(2, 4),
        token_log_probs=torch.full((2, 4), -0.5),
        token_mask=torch.ones(2, 4),
        prompt_input_ids=torch.ones(2, 3, dtype=torch.long),
        prompt_attention_mask=torch.ones(2, 3, dtype=torch.long),
        uncond_input_ids=torch.zeros(2, 3, dtype=torch.long),
        uncond_attention_mask=torch.ones(2, 3, dtype=torch.long),
        context={"model_family": "janus_pro"},
    )


def _request(request_id: str, family: str, task: str) -> GenerationRequest:
    return GenerationRequest(
        request_id=request_id,
        family=family,
        task=task,
        prompts=["p"],
        samples_per_prompt=2,
    )


def _sample_specs(request: GenerationRequest, *, count: int) -> list[GenerationSampleSpec]:
    return [
        GenerationSampleSpec(
            prompt_index=0,
            sample_index=index,
            prompt=request.prompts[0],
            prompt_id=f"{request.request_id}:prompt:0",
            group_id=f"{request.request_id}:group:0",
            sample_id=f"{request.request_id}:sample:{index}",
            trajectory_id=f"{request.request_id}:trajectory:{index}",
            seed=None,
        )
        for index in range(count)
    ]
