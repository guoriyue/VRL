"""SD3.5 OCR rollout parity tests for the trajectory migration path."""

from __future__ import annotations

from typing import Any

import pytest
import torch

from vrl.engine import GenerationRequest, GenerationSampleSpec
from vrl.engine.diffusion import DiffusionChunkResult, build_diffusion_output_batch
from vrl.rollouts.packers.base import RolloutPackContext
from vrl.rollouts.packers.diffusion import DiffusionRolloutPacker
from vrl.rollouts.packers.trajectory import TrajectoryRolloutPacker


@pytest.mark.asyncio
async def test_sd3_5_ocr_rollout_packer_matches_trajectory_packer() -> None:
    request = GenerationRequest(
        request_id="sd3-ocr",
        family="sd3_5",
        task="t2i",
        prompts=["write HELLO in clean block letters"],
        samples_per_prompt=2,
        sampling={"num_steps": 2, "guidance_scale": 4.5, "cfg": True},
        return_artifacts={"output", "rollout_trajectory_data", "trajectory"},
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
    output = build_diffusion_output_batch(
        request=request,
        sample_specs=sample_specs,
        prompts=request.prompts,
        chunks=[_chunk(1.0, context), _chunk(2.0, context)],
        num_steps=2,
    )
    rewards = torch.tensor([0.25, 0.75])
    pack_context = RolloutPackContext(
        metadata={"target_text": "HELLO"},
        kl_reward=0.1,
    )

    legacy_batch = await DiffusionRolloutPacker(error_prefix="SD3.5 OCR").pack(
        output,
        rewards_raw=rewards,
        context=pack_context,
    )
    trajectory_batch = await TrajectoryRolloutPacker().pack(
        output,
        rewards_raw=rewards,
        context=pack_context,
    )

    assert torch.equal(trajectory_batch.observations, legacy_batch.observations)
    assert torch.equal(trajectory_batch.actions, legacy_batch.actions)
    assert torch.equal(trajectory_batch.rewards, legacy_batch.rewards)
    assert torch.equal(trajectory_batch.extras["log_probs"], legacy_batch.extras["log_probs"])
    assert torch.equal(trajectory_batch.extras["timesteps"], legacy_batch.extras["timesteps"])
    assert torch.equal(trajectory_batch.extras["kl"], legacy_batch.extras["kl"])
    assert torch.equal(
        trajectory_batch.extras["reward_before_kl"],
        legacy_batch.extras["reward_before_kl"],
    )
    assert torch.equal(trajectory_batch.videos, legacy_batch.videos)
    assert trajectory_batch.prompts == legacy_batch.prompts
    assert trajectory_batch.context == legacy_batch.context


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
