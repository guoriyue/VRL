"""Tests for the trajectory contract introduced in Sprint A."""

from __future__ import annotations

from dataclasses import fields
from types import SimpleNamespace
from typing import get_args

import pytest
import torch

import vrl.engine.trajectory as trajectory_contract
from vrl.engine import GenerationSampleSpec
from vrl.engine.trajectory import (
    AxisSpec,
    LossUnit,
    ReplayInput,
    RewardView,
    TensorRole,
    TrainingView,
    TrajectoryBatch,
    TrajectoryMetrics,
    TrajectorySegment,
    TrajectoryTensor,
    TrajectoryValidationError,
    get_output_trajectory,
    replay_input_ref,
    tensor_ref,
    validate_training_view,
    validate_trajectory_batch,
)


def test_valid_diffusion_trajectory_and_views() -> None:
    batch = _diffusion_batch()

    validate_trajectory_batch(batch)

    view = TrainingView(
        loss_units=(
            LossUnit(
                segment="denoise",
                axis="timestep",
                axis_index=None,
                action_ref=tensor_ref("denoise", "actions"),
                old_log_prob_ref=tensor_ref("denoise", "old_log_prob"),
                mask_ref=tensor_ref("denoise", "mask"),
                replay_input_refs=(replay_input_ref("denoise", "logprob"),),
            ),
        ),
        primary_segment="denoise",
    )

    validate_training_view(batch, view)


def test_valid_ar_discrete_trajectory_contract() -> None:
    batch_size = 2
    token_count = 4
    batch = TrajectoryBatch(
        request_id="janus",
        family="janus_pro",
        task="ar_t2i",
        sample_specs=_sample_specs(batch_size),
        group_ids=torch.tensor([0, 0]),
        axes={
            "sample": AxisSpec("sample", "sample", batch_size),
            "token": AxisSpec("token", "discrete_token", token_count),
        },
        segments={
            "image_tokens": TrajectorySegment(
                name="image_tokens",
                modality="image",
                trainable=True,
                distribution="categorical",
                tensors={
                    "token_ids": TrajectoryTensor(
                        "token_ids",
                        torch.arange(batch_size * token_count).view(batch_size, token_count),
                        ("sample", "token"),
                        "action",
                    ),
                    "old_log_prob": TrajectoryTensor(
                        "old_log_prob",
                        torch.full((batch_size, token_count), -0.5),
                        ("sample", "token"),
                        "old_log_prob",
                    ),
                    "mask": TrajectoryTensor(
                        "mask",
                        torch.ones(batch_size, token_count),
                        ("sample", "token"),
                        "mask",
                    ),
                    "prompt_input_ids": TrajectoryTensor(
                        "prompt_input_ids",
                        torch.ones(batch_size, 3, dtype=torch.long),
                        ("sample",),
                        "replay_input",
                    ),
                },
            )
        },
    )

    validate_trajectory_batch(batch)


def test_multisegment_trajectory_keeps_segment_scopes() -> None:
    batch_size = 2
    batch = TrajectoryBatch(
        request_id="r1",
        family="janus_pro_r1",
        task="ar_t2i_r1",
        sample_specs=_sample_specs(batch_size),
        group_ids=torch.tensor([0, 0]),
        axes={
            "sample": AxisSpec("sample", "sample", batch_size),
            "initial_token": AxisSpec("initial_token", "discrete_token", 3),
            "text_token": AxisSpec("text_token", "text_token", 2),
            "final_token": AxisSpec("final_token", "discrete_token", 4),
        },
        segments={
            "initial_image": _token_segment("initial_image", "initial_token", batch_size, 3),
            "selfcheck_text": TrajectorySegment(
                name="selfcheck_text",
                modality="text",
                trainable=False,
                distribution="categorical",
                tensors={
                    "token_ids": TrajectoryTensor(
                        "token_ids",
                        torch.ones(batch_size, 2, dtype=torch.long),
                        ("sample", "text_token"),
                        "action",
                    ),
                    "mask": TrajectoryTensor(
                        "mask",
                        torch.ones(batch_size, 2),
                        ("sample", "text_token"),
                        "mask",
                    ),
                },
                advantage_scope="segment",
            ),
            "final_image": _token_segment("final_image", "final_token", batch_size, 4),
        },
    )

    validate_trajectory_batch(batch)
    assert batch.segments["initial_image"].advantage_scope == "segment"
    assert batch.segments["final_image"].advantage_scope == "segment"


def test_validator_rejects_unknown_axis_and_shape_mismatch() -> None:
    batch = _diffusion_batch()
    batch.segments["denoise"].tensors["mask"] = TrajectoryTensor(
        "mask",
        torch.ones(2, 4),
        ("sample", "timestep"),
        "mask",
    )

    with pytest.raises(TrajectoryValidationError, match="expected 3"):
        validate_trajectory_batch(batch)

    batch = _diffusion_batch()
    batch.segments["denoise"].tensors["mask"] = TrajectoryTensor(
        "mask",
        torch.ones(2, 3),
        ("sample", "missing_axis"),
        "mask",
    )

    with pytest.raises(TrajectoryValidationError, match="unknown axis"):
        validate_trajectory_batch(batch)


def test_trainable_segment_requires_action_old_logprob_and_mask() -> None:
    batch = _diffusion_batch()
    del batch.segments["denoise"].tensors["mask"]

    with pytest.raises(TrajectoryValidationError, match="missing required roles"):
        validate_trajectory_batch(batch)


def test_contract_does_not_export_later_sprint_types_or_duplicate_fields() -> None:
    forbidden_exports = {
        "AlgorithmAdapter",
        "AlgorithmInput",
        "EnginePlan",
        "ExecutionUnit",
        "FamilyCapability",
        "RuntimeSession",
        "SegmentSignal",
        "TrajectorySignalBatch",
    }
    assert not forbidden_exports.intersection(dir(trajectory_contract))

    forbidden_batch_fields = {
        "advantages",
        "artifacts",
        "decoded",
        "dones",
        "engine_plan",
        "micro_batches",
        "output",
        "prompts",
        "rewards",
        "runtime_caps",
        "videos",
    }
    assert not forbidden_batch_fields.intersection(
        {field.name for field in fields(TrajectoryBatch)},
    )
    assert not {"media", "reward", "aux"}.intersection(set(get_args(TensorRole)))
    assert "legacy_fields" not in {field.name for field in fields(TrainingView)}


def test_views_store_refs_not_tensor_payloads() -> None:
    with pytest.raises(ValueError, match="tensor_refs"):
        RewardView(
            name="image_reward",
            modality="image",
            tensor_refs=(torch.ones(1),),  # type: ignore[arg-type]
        )

    with pytest.raises(ValueError, match="action_ref"):
        LossUnit(
            segment="denoise",
            axis="timestep",
            axis_index=None,
            action_ref="",
            old_log_prob_ref=tensor_ref("denoise", "old_log_prob"),
            mask_ref=tensor_ref("denoise", "mask"),
        )

    batch = _diffusion_batch()
    with pytest.raises(TrajectoryValidationError, match="runtime-only state"):
        validate_training_view(
            batch,
            TrainingView(
                loss_units=(),
                metadata={"payload": torch.ones(1)},
            ),
        )

    batch.reward_views["image"] = RewardView(
        name="image",
        modality="mixed",
        tensor_refs=(tensor_ref("denoise", "actions"),),
        metadata={"payload": torch.ones(1)},
    )
    with pytest.raises(TrajectoryValidationError, match="runtime-only state"):
        validate_trajectory_batch(batch)


def test_reward_views_group_ids_and_metrics_are_not_loose_fact_sources() -> None:
    batch = _diffusion_batch()
    batch.reward_views["bad"] = "not-a-view"  # type: ignore[assignment]
    with pytest.raises(TrajectoryValidationError, match="must be a RewardView"):
        validate_trajectory_batch(batch)

    batch = _diffusion_batch()
    batch.group_ids = torch.tensor([0])
    with pytest.raises(TrajectoryValidationError, match="group_ids length"):
        validate_trajectory_batch(batch)

    batch = _diffusion_batch()
    batch.metrics = TrajectoryMetrics(
        num_samples=2,
        axis_lengths={"sample": 2, "timestep": 4},
    )
    with pytest.raises(TrajectoryValidationError, match="does not match AxisSpec"):
        validate_trajectory_batch(batch)

    batch = _diffusion_batch()
    batch.metrics = TrajectoryMetrics(values={"execution_s": 1.0})
    with pytest.raises(TrajectoryValidationError, match="engine runtime metrics"):
        validate_trajectory_batch(batch)


def test_loss_units_require_role_correct_refs_and_unique_core_roles() -> None:
    batch = _diffusion_batch()
    batch.segments["denoise"].tensors["other_actions"] = TrajectoryTensor(
        "other_actions",
        torch.ones(2, 3),
        ("sample", "timestep"),
        "action",
    )
    with pytest.raises(TrajectoryValidationError, match="multiple 'action' tensors"):
        validate_trajectory_batch(batch)

    batch = _diffusion_batch()
    view = TrainingView(
        loss_units=(
            LossUnit(
                segment="denoise",
                axis="timestep",
                axis_index=None,
                action_ref=tensor_ref("denoise", "mask"),
                old_log_prob_ref=tensor_ref("denoise", "old_log_prob"),
                mask_ref=tensor_ref("denoise", "mask"),
            ),
        ),
        primary_segment="denoise",
    )
    with pytest.raises(TrajectoryValidationError, match="must reference role 'action'"):
        validate_training_view(batch, view)


def test_runtime_state_is_rejected_from_trajectory_context_and_tensor_value() -> None:
    class FakeModule:
        def state_dict(self) -> dict[str, object]:
            return {}

        def parameters(self) -> list[object]:
            return []

    batch = _diffusion_batch()
    batch.context["model"] = FakeModule()

    with pytest.raises(TrajectoryValidationError, match="runtime-only state"):
        validate_trajectory_batch(batch)

    batch = _diffusion_batch()
    batch.segments["denoise"].tensors["actions"] = TrajectoryTensor(
        "actions",
        FakeModule(),
        ("sample", "timestep"),
        "action",
    )

    with pytest.raises(TrajectoryValidationError, match="runtime-only state"):
        validate_trajectory_batch(batch)


def test_compat_prefers_first_class_trajectory_over_extra_bridge() -> None:
    primary = _diffusion_batch(request_id="primary")
    bridge = _diffusion_batch(request_id="bridge")

    output = SimpleNamespace(trajectory=primary, extra={"trajectory": bridge})
    assert get_output_trajectory(output) is primary

    bridged_output = SimpleNamespace(extra={"trajectory": bridge})
    assert get_output_trajectory(bridged_output) is bridge


def _diffusion_batch(request_id: str = "req") -> TrajectoryBatch:
    batch_size = 2
    timesteps = 3
    return TrajectoryBatch(
        request_id=request_id,
        family="sd3_5",
        task="t2i",
        sample_specs=_sample_specs(batch_size),
        group_ids=torch.tensor([0, 0]),
        axes={
            "sample": AxisSpec("sample", "sample", batch_size),
            "timestep": AxisSpec("timestep", "denoise_step", timesteps),
        },
        segments={
            "denoise": TrajectorySegment(
                name="denoise",
                modality="latent",
                trainable=True,
                distribution="flow_matching",
                tensors={
                    "observations": TrajectoryTensor(
                        "observations",
                        torch.zeros(batch_size, timesteps, 1),
                        ("sample", "timestep"),
                        "observation",
                    ),
                    "actions": TrajectoryTensor(
                        "actions",
                        torch.ones(batch_size, timesteps, 1),
                        ("sample", "timestep"),
                        "action",
                    ),
                    "old_log_prob": TrajectoryTensor(
                        "old_log_prob",
                        torch.full((batch_size, timesteps), -0.25),
                        ("sample", "timestep"),
                        "old_log_prob",
                    ),
                    "mask": TrajectoryTensor(
                        "mask",
                        torch.ones(batch_size, timesteps),
                        ("sample", "timestep"),
                        "mask",
                    ),
                },
                reward_view="image",
                replay_inputs={
                    "logprob": ReplayInput(
                        name="logprob",
                        tensor_refs=(
                            tensor_ref("denoise", "observations"),
                            tensor_ref("denoise", "actions"),
                        ),
                    )
                },
            )
        },
        reward_views={
            "image": RewardView(
                name="image",
                modality="mixed",
                tensor_refs=(tensor_ref("denoise", "actions"),),
            )
        },
    )


def _token_segment(
    name: str,
    axis: str,
    batch_size: int,
    token_count: int,
) -> TrajectorySegment:
    return TrajectorySegment(
        name=name,
        modality="image",
        trainable=True,
        distribution="categorical",
        tensors={
            "token_ids": TrajectoryTensor(
                "token_ids",
                torch.ones(batch_size, token_count, dtype=torch.long),
                ("sample", axis),
                "action",
            ),
            "old_log_prob": TrajectoryTensor(
                "old_log_prob",
                torch.zeros(batch_size, token_count),
                ("sample", axis),
                "old_log_prob",
            ),
            "mask": TrajectoryTensor(
                "mask",
                torch.ones(batch_size, token_count),
                ("sample", axis),
                "mask",
            ),
        },
        advantage_scope="segment",
    )


def _sample_specs(count: int) -> list[GenerationSampleSpec]:
    return [
        GenerationSampleSpec(
            prompt_index=0,
            sample_index=i,
            prompt="prompt",
            prompt_id="prompt-0",
            group_id="prompt-0",
            sample_id=f"sample-{i}",
            trajectory_id=f"trajectory-{i}",
            seed=None,
        )
        for i in range(count)
    ]
