"""Shared OnlineTrainer test helpers: algorithm-input unpacking and trajectory-signal construction."""

from __future__ import annotations

from typing import Any

import torch

from vrl.config.precision import RolePrecision
from vrl.generation import GenerationRequest, GenerationSampleRow
from vrl.rollouts.batch import RolloutBatch
from vrl.trajectory import build_diffusion_trajectory

DEFAULT_PRECISION = RolePrecision(
    dtype="fp32",
    float32_precision="ieee",
    outer_autocast=False,
)


class _EvaluatorAlgorithmFake:
    """Common capabilities for test algorithms that consume evaluator signals."""

    uses_evaluator = True
    tolerates_off_policy_staleness = True
    requires_active_trust_region = False
    needs_kl_intermediates = False


def _stamp_model_precision(
    model: Any,
    *,
    precision: RolePrecision = DEFAULT_PRECISION,
) -> None:
    """Stamp the role precision required by trainer test doubles."""

    model.precision = precision


def _algorithm_inputs(inputs):
    if inputs.signals is None:
        raise AssertionError("test algorithm requires trajectory signals")
    signals = inputs.signals.primary
    return signals, inputs.advantages, signals.old_log_prob


def _diffusion_rollout_batch(
    *,
    rewards: torch.Tensor,
    group_ids: torch.Tensor,
    num_steps: int | None = None,
    observations: torch.Tensor | None = None,
    actions: torch.Tensor | None = None,
    timesteps: torch.Tensor | None = None,
    extras: dict[str, Any] | None = None,
    context: dict[str, Any] | None = None,
) -> RolloutBatch:
    """Build a canonical typed diffusion batch for trainer tests."""

    batch_size = int(rewards.shape[0])
    if tuple(group_ids.shape) != (batch_size,):
        raise ValueError("group_ids must have one value per reward")
    if observations is None:
        step_count = 2 if num_steps is None else int(num_steps)
        observations = torch.zeros(
            batch_size,
            step_count,
            1,
            device=rewards.device,
        )
    else:
        if observations.ndim < 2 or int(observations.shape[0]) != batch_size:
            raise ValueError("observations must start with [sample, denoise]")
        step_count = int(observations.shape[1])
        if num_steps is not None and int(num_steps) != step_count:
            raise ValueError("num_steps does not match observations")
    if actions is None:
        actions = torch.zeros_like(observations)
    if tuple(actions.shape[:2]) != (batch_size, step_count):
        raise ValueError("actions must start with [sample, denoise]")

    old_log_prob = torch.zeros(
        batch_size,
        step_count,
        dtype=torch.float32,
        device=observations.device,
    )
    if timesteps is None:
        timesteps = (
            torch.arange(step_count, device=observations.device)
            .unsqueeze(0)
            .expand(batch_size, step_count)
        )

    group_values = group_ids.detach().cpu().tolist()
    prompt_indices: dict[Any, int] = {}
    sample_counts: dict[Any, int] = {}
    row_specs: list[tuple[int, int]] = []
    for group_id in group_values:
        prompt_index = prompt_indices.setdefault(group_id, len(prompt_indices))
        sample_index = sample_counts.get(group_id, 0)
        sample_counts[group_id] = sample_index + 1
        row_specs.append((prompt_index, sample_index))
    prompts = [f"trainer test prompt {index}" for index in range(len(prompt_indices))]
    request = GenerationRequest(
        request_id="trainer-test",
        family="test-diffusion",
        task="t2i",
        inputs=prompts,
        samples_per_prompt=max(sample_counts.values(), default=1),
    )
    sample_rows = [
        GenerationSampleRow(
            prompt_index=prompt_index,
            sample_index=sample_index,
            prompt=prompts[prompt_index],
            sample_id=f"trainer-test:sample:{index}",
        )
        for index, (prompt_index, sample_index) in enumerate(row_specs)
    ]
    batch_context = dict(context or {})
    trajectory = build_diffusion_trajectory(
        request=request,
        sample_rows=sample_rows,
        observations=observations,
        actions=actions,
        old_log_prob=old_log_prob,
        timesteps=timesteps,
        kl=torch.zeros_like(old_log_prob),
        replay_tensors={},
        context=batch_context,
    )
    return RolloutBatch(
        rewards=rewards,
        group_ids=group_ids,
        extras=dict(extras or {}),
        context=batch_context,
        trajectory=trajectory,
    )


def _trajectory_signals(
    batch,
    log_prob,
    timestep_idx: int = 0,
    *,
    old_log_prob=None,
):
    from vrl.rollouts.evaluators.types import SegmentSignal, TrajectorySignalBatch

    if old_log_prob is None:
        # Most trainer fakes model an unchanged-policy replay. Tests that
        # deliberately exercise mismatch must provide the behavior value.
        old_log_prob = log_prob.detach().clone()
    mask = torch.ones_like(log_prob)
    return TrajectorySignalBatch(
        segments={
            "default": SegmentSignal(
                name="default",
                distribution="flow_matching",
                log_prob=log_prob,
                old_log_prob=old_log_prob,
                mask=mask,
            ),
        },
        group_ids=batch.group_ids,
        primary_segment="default",
    )
