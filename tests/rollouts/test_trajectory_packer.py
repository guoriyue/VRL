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
from vrl.engine.trajectory import (
    build_ar_continuous_trajectory,
    build_ar_discrete_trajectory,
    build_ar_multisegment_trajectory,
)
from vrl.rollouts.packers.base import RolloutPackContext
from vrl.rollouts.packers.trajectory import TrajectoryRolloutPacker


@pytest.mark.asyncio
async def test_trajectory_packer_packs_diffusion_trajectory() -> None:
    output = _diffusion_output()
    rewards = torch.tensor([3.0, 4.0])
    context = RolloutPackContext(
        metadata={"target_text": "HELLO"},
        kl_reward=0.25,
    )

    batch = await TrajectoryRolloutPacker().pack(
        output,
        rewards_raw=rewards,
        context=context,
    )

    assert batch.trajectory is output.trajectory
    assert batch.training_view is not None
    assert batch.training_view.primary_segment == "denoise"
    assert batch.observations.shape == (2, 2, 1)
    assert batch.actions.shape == (2, 2, 1)
    denoise = output.trajectory.segments["denoise"]
    assert torch.equal(batch.extras["log_probs"], denoise.tensors["old_log_prob"].value)
    assert torch.equal(batch.extras["timesteps"], denoise.tensors["timesteps"].value)
    assert torch.equal(batch.extras["kl"], denoise.tensors["kl"].value)
    assert torch.equal(batch.extras["reward_before_kl"], rewards)
    assert torch.equal(batch.rewards, rewards - context.kl_reward * batch.extras["kl"].sum(dim=1))
    assert torch.equal(batch.videos, output.output)
    assert batch.context["reward_metadata"] == {"target_text": "HELLO"}
    assert batch.prompts == ["read HELLO", "read HELLO"]


@pytest.mark.asyncio
async def test_trajectory_packer_packs_janus_discrete_trajectory() -> None:
    output = _janus_output()
    rewards = torch.tensor([1.0, 2.0])
    context = RolloutPackContext(metadata={}, rescale_to_unit=True)
    packer = TrajectoryRolloutPacker()

    reward_outputs = packer.reward_outputs(output, context)
    batch = await packer.pack(output, rewards_raw=rewards, context=context)
    segment = output.trajectory.segments["image_tokens"]

    assert torch.equal(
        reward_outputs,
        ((output.output + 1.0) * 0.5).clamp(0.0, 1.0),
    )
    assert torch.equal(batch.actions, segment.tensors["token_ids"].value)
    assert torch.equal(
        batch.extras["log_probs"],
        segment.tensors["old_log_prob"].value.unsqueeze(1),
    )
    assert torch.equal(batch.extras["token_mask"], segment.tensors["token_mask"].value)
    assert torch.equal(
        batch.extras["uncond_input_ids"],
        segment.tensors["uncond_input_ids"].value,
    )
    assert batch.videos.shape == (2, 3, 1, 4, 4)
    assert batch.training_view is not None
    assert batch.training_view.primary_segment == "image_tokens"


@pytest.mark.asyncio
async def test_trajectory_packer_packs_nextstep_continuous_trajectory() -> None:
    output = _nextstep_output()
    rewards = torch.tensor([1.25, 2.25])
    context = RolloutPackContext(metadata={})
    packer = TrajectoryRolloutPacker()

    reward_outputs = packer.reward_outputs(output, context)
    batch = await packer.pack(output, rewards_raw=rewards, context=context)
    segment = output.trajectory.segments["image_tokens"]
    decoded = output.trajectory.segments["decoded"]

    assert torch.equal(reward_outputs, decoded.tensors["images_for_reward"].value)
    assert torch.equal(batch.actions, segment.tensors["tokens"].value)
    assert torch.equal(batch.extras["saved_noise"], segment.tensors["saved_noise"].value)
    assert torch.equal(
        batch.extras["log_probs"],
        segment.tensors["old_log_prob"].value.unsqueeze(1),
    )
    assert torch.equal(batch.extras["token_mask"], segment.tensors["token_mask"].value)
    assert batch.videos.shape == (2, 3, 1, 4, 4)
    assert batch.training_view is not None
    assert batch.training_view.primary_segment == "image_tokens"


@pytest.mark.asyncio
async def test_trajectory_packer_packs_r1_multisegment_trajectory() -> None:
    output = _r1_output()
    output.extra = {}
    rewards = torch.tensor([1.0, 2.0])
    context = RolloutPackContext(metadata={}, rescale_to_unit=False)

    reward_outputs = TrajectoryRolloutPacker().reward_outputs(output, context)
    batch = await TrajectoryRolloutPacker().pack(
        output,
        rewards_raw=rewards,
        context=context,
    )

    decoded = output.trajectory.segments["decoded"]
    assert torch.equal(reward_outputs, decoded.tensors["final_image"].value)
    assert set(batch.extras["r1_segments"]) == {"initial_image", "selfcheck_text", "final_image"}
    assert batch.extras["r1_segments"]["initial_image"]["token_ids"].shape == (2, 3)
    assert batch.extras["r1_segments"]["selfcheck_text"]["token_ids"].shape == (2, 2)
    assert batch.extras["r1_segments"]["final_image"]["token_ids"].shape == (2, 5)
    assert batch.actions.shape == (2, 5)
    assert batch.extras["log_probs"].shape == (2, 1, 5)
    assert batch.extras["primary_segment"] == "final_image"
    assert torch.equal(batch.videos[:, :, 0], decoded.tensors["final_image"].value)
    assert batch.context["r1_segment_names"] == ("initial_image", "selfcheck_text", "final_image")
    assert batch.training_view is not None
    assert batch.training_view.primary_segment == "final_image"
    assert {unit.segment for unit in batch.training_view.loss_units} == {
        "initial_image",
        "final_image",
    }


@pytest.mark.asyncio
async def test_trajectory_packer_does_not_need_legacy_rollout_fields() -> None:
    diffusion_output = _diffusion_output()

    diffusion_batch = await TrajectoryRolloutPacker().pack(
        diffusion_output,
        rewards_raw=torch.tensor([1.0, 1.5]),
        context=RolloutPackContext(metadata={}),
    )
    assert diffusion_batch.observations.shape == (2, 2, 1)
    assert diffusion_batch.extras["kl"].shape == (2, 2)

    janus_output = _janus_output()
    janus_output.extra = {}

    janus_batch = await TrajectoryRolloutPacker().pack(
        janus_output,
        rewards_raw=torch.tensor([2.0, 2.5]),
        context=RolloutPackContext(metadata={}),
    )
    assert janus_batch.actions.shape == (2, 4)
    assert janus_batch.extras["token_mask"].shape == (2, 4)

    nextstep_output = _nextstep_output()
    nextstep_output.extra = {}
    nextstep_batch = await TrajectoryRolloutPacker().pack(
        nextstep_output,
        rewards_raw=torch.tensor([3.0, 3.5]),
        context=RolloutPackContext(metadata={}),
    )
    assert nextstep_batch.actions.shape == (2, 4, 3)
    assert nextstep_batch.extras["saved_noise"].shape == (2, 4, 3)

    r1_output = _r1_output()
    r1_output.extra = {}
    r1_batch = await TrajectoryRolloutPacker().pack(
        r1_output,
        rewards_raw=torch.tensor([4.0, 4.5]),
        context=RolloutPackContext(metadata={}),
    )
    assert r1_batch.actions.shape == (2, 5)
    assert set(r1_batch.extras["r1_segments"]) == {
        "initial_image",
        "selfcheck_text",
        "final_image",
    }


def test_merged_generation_slices_first_class_trajectory() -> None:
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
    return_artifacts = {"output", "trajectory"}
    request = _request(
        "sd3",
        prompts=["read HELLO"],
        family="sd3_5",
        task="t2i",
        return_artifacts=return_artifacts,
    )
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
    assert "trajectory" not in output.extra
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
        },
    )


def _nextstep_output() -> OutputBatch:
    request = _request("nextstep", prompts=["paint"], family="nextstep_1", task="ar_t2i")
    sample_specs = _sample_specs(request, samples_per_prompt=2)
    batch_size = len(sample_specs)
    token_count = 4
    token_dim = 3
    tokens = torch.arange(batch_size * token_count * token_dim, dtype=torch.float32).view(
        batch_size,
        token_count,
        token_dim,
    )
    saved_noise = tokens + 0.25
    log_probs = torch.full((batch_size, token_count), -0.75)
    token_mask = torch.ones_like(log_probs)
    prompt_ids = torch.arange(batch_size * 3).view(batch_size, 3)
    prompt_mask = torch.ones(batch_size, 3, dtype=torch.long)
    uncond_ids = torch.zeros(batch_size, 3, dtype=torch.long)
    uncond_mask = torch.ones(batch_size, 3, dtype=torch.long)
    images = torch.linspace(-1.0, 1.0, steps=batch_size * 3 * 4 * 4).view(
        batch_size,
        3,
        4,
        4,
    )
    images_for_reward = ((images + 1.0) * 0.5).clamp(0.0, 1.0)
    context = {
        "cfg_scale": 4.5,
        "num_flow_steps": 8,
        "noise_level": 1.0,
        "image_token_num": token_count,
        "image_size": 4,
        "rescale_to_unit": True,
    }
    trajectory = build_ar_continuous_trajectory(
        request=request,
        sample_specs=sample_specs,
        tokens=tokens,
        saved_noise=saved_noise,
        token_log_probs=log_probs,
        token_mask=token_mask,
        prompt_input_ids=prompt_ids,
        prompt_attention_mask=prompt_mask,
        uncond_input_ids=uncond_ids,
        uncond_attention_mask=uncond_mask,
        images_for_reward=images_for_reward,
        context=context,
    )
    return OutputBatch(
        request_id=request.request_id,
        family=request.family,
        task=request.task,
        prompts=list(request.prompts),
        sample_specs=sample_specs,
        output=images,
        trajectory=trajectory,
        extra={
            "tokens": tokens,
            "saved_noise": saved_noise,
            "log_probs": log_probs,
            "images_for_reward": images_for_reward,
            "prompt_input_ids": prompt_ids,
            "prompt_attention_mask": prompt_mask,
            "uncond_input_ids": uncond_ids,
            "uncond_attention_mask": uncond_mask,
            "context": context,
        },
    )


def _r1_output() -> OutputBatch:
    request = _request("r1", prompts=["refine"], family="janus_pro_r1", task="ar_t2i_r1")
    sample_specs = _sample_specs(request, samples_per_prompt=2)
    batch_size = len(sample_specs)
    initial_image = torch.zeros(batch_size, 3, 4, 4)
    final_image = torch.ones(batch_size, 3, 4, 4)
    selfcheck = torch.arange(batch_size * 2).view(batch_size, 2)
    segments = {
        "initial_image": _r1_segment("initial_image", batch_size, 3, visual=True, train=True),
        "selfcheck_text": _r1_segment("selfcheck_text", batch_size, 2, visual=False, train=False),
        "final_image": _r1_segment("final_image", batch_size, 5, visual=True, train=True),
    }
    context = {"cfg_weight": 5.0}
    trajectory = build_ar_multisegment_trajectory(
        request=request,
        sample_specs=sample_specs,
        segments=segments,
        decoded_outputs={
            "initial_image": initial_image,
            "final_image": final_image,
            "selfcheck": selfcheck,
        },
        primary_segment="final_image",
        context=context,
    )
    return OutputBatch(
        request_id=request.request_id,
        family=request.family,
        task=request.task,
        prompts=list(request.prompts),
        sample_specs=sample_specs,
        output=final_image,
        trajectory=trajectory,
        extra={
            "initial_image": initial_image,
            "final_image": final_image,
            "selfcheck": selfcheck,
            "segments": segments,
            "context": context,
        },
    )


def _r1_segment(
    name: str,
    batch_size: int,
    token_count: int,
    *,
    visual: bool,
    train: bool,
) -> dict[str, Any]:
    return {
        "name": name,
        "token_ids": torch.arange(batch_size * token_count, dtype=torch.long).view(
            batch_size,
            token_count,
        ),
        "token_log_probs": torch.full((batch_size, token_count), -0.5),
        "token_mask": torch.ones(batch_size, token_count),
        "prompt_embeds": torch.ones(batch_size, 3, 4),
        "attention_mask": torch.ones(batch_size, 3, dtype=torch.long),
        "prompt_attention_mask": torch.ones(batch_size, 3, dtype=torch.long),
        "visual": visual,
        "cfg": visual,
        "train": train,
    }


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
            extra={},
        )


def _request(
    request_id: str,
    *,
    prompts: list[str],
    family: str,
    task: str,
    return_artifacts: set[str] | None = None,
) -> GenerationRequest:
    return GenerationRequest(
        request_id=request_id,
        family=family,
        task=task,
        prompts=prompts,
        samples_per_prompt=2,
        sampling={"num_steps": 2, "seed": 11},
        return_artifacts=return_artifacts or {"output", "trajectory"},
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
