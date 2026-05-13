"""SD3.5 OCR rollout tests for the strict trajectory path."""

from __future__ import annotations

from typing import Any

import pytest
import torch

from vrl.engine import GenerationRequest, GenerationSampleSpec
from vrl.engine.diffusion import DiffusionChunkResult, build_diffusion_output_batch
from vrl.rollouts.packers.base import RolloutPackContext
from vrl.rollouts.packers.trajectory import TrajectoryRolloutPacker


@pytest.mark.asyncio
async def test_sd3_5_ocr_rollout_packer_uses_strict_trajectory() -> None:
    trajectory_output = _output_for_artifacts({"output", "trajectory"})
    rewards = torch.tensor([0.25, 0.75])
    pack_context = RolloutPackContext(
        metadata={"target_text": "HELLO"},
        kl_reward=0.1,
    )

    assert trajectory_output.rollout_trajectory_data is None
    assert trajectory_output.trajectory is not None

    trajectory_batch = await TrajectoryRolloutPacker().pack(
        trajectory_output,
        rewards_raw=rewards,
        context=pack_context,
    )

    denoise = trajectory_output.trajectory.segments["denoise"]
    assert trajectory_batch.trajectory is trajectory_output.trajectory
    assert trajectory_batch.training_view is not None
    assert trajectory_batch.training_view.primary_segment == "denoise"
    assert torch.equal(
        trajectory_batch.observations,
        denoise.tensors["observations"].value,
    )
    assert torch.equal(trajectory_batch.actions, denoise.tensors["actions"].value)
    assert torch.equal(
        trajectory_batch.extras["log_probs"],
        denoise.tensors["old_log_prob"].value,
    )
    assert torch.equal(
        trajectory_batch.extras["timesteps"],
        denoise.tensors["timesteps"].value,
    )
    assert torch.equal(trajectory_batch.extras["kl"], denoise.tensors["kl"].value)
    assert torch.equal(
        trajectory_batch.extras["reward_before_kl"],
        rewards,
    )
    assert torch.equal(
        trajectory_batch.rewards,
        rewards - pack_context.kl_reward * trajectory_batch.extras["kl"].sum(dim=1),
    )
    assert torch.equal(trajectory_batch.videos, trajectory_output.output)
    assert trajectory_batch.prompts == [
        "write HELLO in clean block letters",
        "write HELLO in clean block letters",
    ]
    assert trajectory_batch.context["reward_metadata"] == {"target_text": "HELLO"}


def _output_for_artifacts(return_artifacts: set[str]):
    request = GenerationRequest(
        request_id="sd3-ocr",
        family="sd3_5",
        task="t2i",
        prompts=["write HELLO in clean block letters"],
        samples_per_prompt=2,
        sampling={"num_steps": 2, "guidance_scale": 4.5, "cfg": True},
        return_artifacts=return_artifacts,
    )
    sample_specs = [
        GenerationSampleSpec(
            prompt_index=0,
            sample_index=index,
            prompt=request.prompts[0],
            prompt_id="sd3-ocr:prompt:0",
            group_id="sd3-ocr:prompt:0",
            sample_id=f"sd3-ocr:sample:{index}",
            trajectory_id=f"sd3-ocr:trajectory:{index}",
            seed=100 + index,
        )
        for index in range(2)
    ]
    context = {"guidance_scale": 4.5, "cfg": True, "model_family": "sd3_5"}
    return build_diffusion_output_batch(
        request=request,
        sample_specs=sample_specs,
        prompts=request.prompts,
        chunks=[_chunk(1.0, context), _chunk(2.0, context)],
        num_steps=2,
    )


def _chunk(value: float, context: dict[str, Any]) -> DiffusionChunkResult:
    return DiffusionChunkResult(
        observations=torch.full((1, 2, 1), value),
        actions=torch.full((1, 2, 1), value + 0.1),
        log_probs=torch.full((1, 2), value + 0.2),
        timesteps=torch.tensor([[0, 1]]),
        kl=torch.full((1, 2), value + 0.3),
        video=torch.full((1, 3, 4, 4), value + 0.4),
        training_extras={
            "prompt_embeds": torch.full((1, 2, 3), value + 0.5),
            "pooled_prompt_embeds": torch.full((1, 3), value + 0.6),
            "negative_prompt_embeds": None,
            "negative_pooled_prompt_embeds": None,
        },
        context=context,
    )
