"""Tests for trajectory-backed rollout packing."""

from __future__ import annotations

from typing import Any

import pytest
import torch

from vrl.engine import (
    GenerationRequest,
    GenerationSampleSpec,
    OutputBatch,
    forward_batch_by_merging_prompts,
)
from vrl.engine.diffusion import DiffusionChunkResult, build_diffusion_output_batch
from vrl.engine.trajectory import build_ar_discrete_trajectory
from vrl.rollouts.packers.ar.discrete import ARDiscreteRolloutPacker
from vrl.rollouts.packers.base import RolloutPackContext
from vrl.rollouts.packers.diffusion import DiffusionRolloutPacker
from vrl.rollouts.packers.trajectory import TrajectoryRolloutPacker


@pytest.mark.asyncio
async def test_trajectory_packer_matches_diffusion_packer() -> None:
    output = _diffusion_output()
    rewards = torch.tensor([3.0, 4.0])
    context = RolloutPackContext(
        metadata={"target_text": "HELLO"},
        kl_reward=0.25,
    )

    old_batch = await DiffusionRolloutPacker(error_prefix="SD3.5").pack(
        output,
        rewards_raw=rewards,
        context=context,
    )
    new_batch = await TrajectoryRolloutPacker().pack(
        output,
        rewards_raw=rewards,
        context=context,
    )

    _assert_rollout_batch_equal(new_batch, old_batch)


@pytest.mark.asyncio
async def test_trajectory_packer_matches_janus_discrete_packer() -> None:
    output = _janus_output()
    rewards = torch.tensor([1.0, 2.0])
    context = RolloutPackContext(metadata={}, rescale_to_unit=True)

    old_packer = ARDiscreteRolloutPacker()
    new_packer = TrajectoryRolloutPacker()

    assert torch.equal(
        new_packer.reward_outputs(output, context),
        old_packer.reward_outputs(output, context),
    )

    old_batch = await old_packer.pack(output, rewards_raw=rewards, context=context)
    new_batch = await new_packer.pack(output, rewards_raw=rewards, context=context)

    _assert_rollout_batch_equal(new_batch, old_batch)


@pytest.mark.asyncio
async def test_trajectory_packer_does_not_need_legacy_rollout_fields() -> None:
    diffusion_output = _diffusion_output()
    diffusion_output.rollout_trajectory_data = None
    diffusion_output.extra = {"trajectory": diffusion_output.trajectory}

    diffusion_batch = await TrajectoryRolloutPacker().pack(
        diffusion_output,
        rewards_raw=torch.tensor([1.0, 1.5]),
        context=RolloutPackContext(metadata={}),
    )
    assert diffusion_batch.observations.shape == (2, 2, 1)
    assert diffusion_batch.extras["kl"].shape == (2, 2)

    janus_output = _janus_output()
    janus_output.extra = {"trajectory": janus_output.trajectory}

    janus_batch = await TrajectoryRolloutPacker().pack(
        janus_output,
        rewards_raw=torch.tensor([2.0, 2.5]),
        context=RolloutPackContext(metadata={}),
    )
    assert janus_batch.actions.shape == (2, 4)
    assert janus_batch.extras["token_mask"].shape == (2, 4)


def test_merged_generation_slices_first_class_trajectory_and_bridge() -> None:
    executor = _MergedTrajectoryExecutor()
    requests = [
        _request("req-a", prompts=["alpha"], family="janus_pro", task="ar_t2i"),
        _request("req-b", prompts=["beta"], family="janus_pro", task="ar_t2i"),
    ]
    sample_specs_by_request = {
        request.request_id: _sample_specs(request, samples_per_prompt=2)
        for request in requests
    }

    outputs = forward_batch_by_merging_prompts(
        executor,
        requests,
        sample_specs_by_request,
    )

    first = outputs["req-a"]
    second = outputs["req-b"]

    assert first.trajectory is first.extra["trajectory"]
    assert second.trajectory is second.extra["trajectory"]
    assert first.trajectory is not None
    assert second.trajectory is not None
    assert first.trajectory.request_id == "req-a"
    assert second.trajectory.request_id == "req-b"
    assert first.trajectory.axes["sample"].length == 2
    assert second.trajectory.axes["sample"].length == 2
    assert first.trajectory.metrics.num_samples == 2
    assert second.trajectory.metrics.num_samples == 2
    assert torch.equal(
        first.trajectory.segments["image_tokens"].tensors["token_ids"].value,
        torch.tensor([[0, 1], [2, 3]]),
    )
    assert torch.equal(
        second.trajectory.segments["image_tokens"].tensors["token_ids"].value,
        torch.tensor([[4, 5], [6, 7]]),
    )


def _diffusion_output() -> OutputBatch:
    request = _request("sd3", prompts=["read HELLO"], family="sd3_5", task="t2i")
    sample_specs = _sample_specs(request, samples_per_prompt=2)
    context = {
        "guidance_scale": 4.5,
        "cfg": True,
        "model_family": "sd3_5",
    }
    chunks = [
        _diffusion_chunk(0.0, context),
        _diffusion_chunk(10.0, context),
    ]
    output = build_diffusion_output_batch(
        request=request,
        sample_specs=sample_specs,
        prompts=request.prompts,
        chunks=chunks,
        num_steps=2,
    )
    assert output.trajectory is not None
    assert output.extra["trajectory"] is output.trajectory
    return output


def _diffusion_chunk(value: float, context: dict[str, Any]) -> DiffusionChunkResult:
    return DiffusionChunkResult(
        observations=torch.full((1, 2, 1), value),
        actions=torch.full((1, 2, 1), value + 1.0),
        log_probs=torch.full((1, 2), value + 2.0),
        timesteps=torch.arange(2).view(1, 2),
        kl=torch.full((1, 2), value + 3.0),
        video=torch.full((1, 3, 4, 4), value + 4.0),
        training_extras={
            "prompt_embeds": torch.full((1, 3, 2), value + 5.0),
            "pooled_prompt_embeds": torch.full((1, 2), value + 6.0),
            "negative_prompt_embeds": None,
        },
        context=context,
    )


def _janus_output() -> OutputBatch:
    request = _request("janus", prompts=["draw a square"], family="janus_pro", task="ar_t2i")
    sample_specs = _sample_specs(request, samples_per_prompt=2)
    batch_size = len(sample_specs)
    token_count = 4
    token_ids = torch.arange(batch_size * token_count).view(batch_size, token_count)
    token_log_probs = torch.full((batch_size, token_count), -0.5)
    token_mask = torch.ones(batch_size, token_count)
    prompt_ids = torch.arange(batch_size * 3).view(batch_size, 3)
    prompt_mask = torch.ones(batch_size, 3, dtype=torch.long)
    uncond_ids = torch.zeros(batch_size, 3, dtype=torch.long)
    uncond_mask = torch.ones(batch_size, 3, dtype=torch.long)
    context = {"cfg_weight": 5.0, "model_family": "janus_pro"}
    trajectory = build_ar_discrete_trajectory(
        request=request,
        sample_specs=sample_specs,
        token_ids=token_ids,
        token_log_probs=token_log_probs,
        token_mask=token_mask,
        prompt_input_ids=prompt_ids,
        prompt_attention_mask=prompt_mask,
        uncond_input_ids=uncond_ids,
        uncond_attention_mask=uncond_mask,
        context=context,
    )
    return OutputBatch(
        request_id=request.request_id,
        family=request.family,
        task=request.task,
        prompts=list(request.prompts),
        sample_specs=sample_specs,
        output=torch.linspace(-1.0, 1.0, steps=batch_size * 3 * 4 * 4).view(
            batch_size,
            3,
            4,
            4,
        ),
        trajectory=trajectory,
        extra={
            "token_ids": token_ids,
            "token_log_probs": token_log_probs,
            "token_mask": token_mask,
            "prompt_input_ids": prompt_ids,
            "prompt_attention_mask": prompt_mask,
            "uncond_input_ids": uncond_ids,
            "uncond_attention_mask": uncond_mask,
            "context": context,
            "trajectory": trajectory,
        },
    )


class _MergedTrajectoryExecutor:
    family = "janus_pro"
    task = "ar_t2i"

    def forward(
        self,
        request: GenerationRequest,
        sample_specs: list[GenerationSampleSpec],
    ) -> OutputBatch:
        batch_size = len(sample_specs)
        token_ids = torch.arange(batch_size * 2).view(batch_size, 2)
        token_log_probs = -token_ids.float()
        token_mask = torch.ones(batch_size, 2)
        prompt_ids = torch.arange(batch_size * 3).view(batch_size, 3)
        prompt_mask = torch.ones(batch_size, 3, dtype=torch.long)
        uncond_ids = torch.zeros(batch_size, 3, dtype=torch.long)
        uncond_mask = torch.ones(batch_size, 3, dtype=torch.long)
        trajectory = build_ar_discrete_trajectory(
            request=request,
            sample_specs=sample_specs,
            token_ids=token_ids,
            token_log_probs=token_log_probs,
            token_mask=token_mask,
            prompt_input_ids=prompt_ids,
            prompt_attention_mask=prompt_mask,
            uncond_input_ids=uncond_ids,
            uncond_attention_mask=uncond_mask,
            context={"model_family": "janus_pro"},
        )
        return OutputBatch(
            request_id=request.request_id,
            family=request.family,
            task=request.task,
            prompts=list(request.prompts),
            sample_specs=list(sample_specs),
            output=torch.arange(batch_size * 2).view(batch_size, 2),
            trajectory=trajectory,
            extra={"trajectory": trajectory},
        )


def _request(
    request_id: str,
    *,
    prompts: list[str],
    family: str,
    task: str,
) -> GenerationRequest:
    return GenerationRequest(
        request_id=request_id,
        family=family,
        task=task,
        prompts=prompts,
        samples_per_prompt=2,
        sampling={"num_steps": 2, "seed": 11},
        return_artifacts={"output", "trajectory"},
    )


def _sample_specs(
    request: GenerationRequest,
    *,
    samples_per_prompt: int,
) -> list[GenerationSampleSpec]:
    specs: list[GenerationSampleSpec] = []
    for prompt_index, prompt in enumerate(request.prompts):
        for sample_index in range(samples_per_prompt):
            specs.append(
                GenerationSampleSpec(
                    prompt_index=prompt_index,
                    sample_index=sample_index,
                    prompt=prompt,
                    prompt_id=f"{request.request_id}:prompt:{prompt_index}",
                    group_id=f"{request.request_id}:prompt:{prompt_index}",
                    sample_id=f"{request.request_id}:sample:{len(specs)}",
                    trajectory_id=f"{request.request_id}:trajectory:{len(specs)}",
                    seed=11 + len(specs),
                )
            )
    return specs


def _assert_rollout_batch_equal(left: Any, right: Any) -> None:
    assert torch.equal(left.observations, right.observations)
    assert torch.equal(left.actions, right.actions)
    assert torch.equal(left.rewards, right.rewards)
    assert torch.equal(left.dones, right.dones)
    assert torch.equal(left.group_ids, right.group_ids)
    assert left.prompts == right.prompts
    assert left.context == right.context
    if left.videos is None or right.videos is None:
        assert left.videos is right.videos
    else:
        assert torch.equal(left.videos, right.videos)
    assert set(left.extras) == set(right.extras)
    for key in left.extras:
        _assert_equal_value(left.extras[key], right.extras[key])


def _assert_equal_value(left: Any, right: Any) -> None:
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        assert isinstance(left, torch.Tensor)
        assert isinstance(right, torch.Tensor)
        assert torch.equal(left, right)
    elif isinstance(left, dict) and isinstance(right, dict):
        assert set(left) == set(right)
        for key in left:
            _assert_equal_value(left[key], right[key])
    else:
        assert left == right
