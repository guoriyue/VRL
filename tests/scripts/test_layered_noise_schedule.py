"""Piecewise exploration retains exact behavior densities at the noise switch."""

from types import SimpleNamespace

import pytest
import torch

from vrl.math.denoise.flow_matching import sde_step_with_logprob
from vrl.scripts.generation.qwen_layered_rl_probe import (
    noise_schedule,
    recorded_noise_levels,
)


def test_noise_switch_replays_exactly_and_rejects_missing_behavior_metadata():
    scheduler = SimpleNamespace(
        sigmas=torch.tensor([0.8, 0.4, 0.0]), timesteps=torch.tensor([800.0, 400.0])
    )
    plan = {"steps": 2, "noise_level": 0.05, "noise_levels": [0.5, 0.05]}
    trajectory = {**plan, "timesteps": scheduler.timesteps}
    state = torch.tensor([[[0.2, -0.1]]])
    velocity = torch.tensor([[[0.1, 0.3]]])
    generator = torch.Generator().manual_seed(17)
    levels = recorded_noise_levels(trajectory, plan)
    for index, level in enumerate(levels):
        sampled = sde_step_with_logprob(
            scheduler,
            velocity,
            scheduler.timesteps[index : index + 1],
            state,
            generator=generator,
            noise_level=level,
            step_index=index,
        )
        replay = sde_step_with_logprob(
            scheduler,
            velocity,
            scheduler.timesteps[index : index + 1],
            state,
            prev_sample=sampled.prev_sample,
            noise_level=level,
            step_index=index,
        )
        torch.testing.assert_close(sampled.log_prob, replay.log_prob, rtol=0, atol=0)
        if index == 0:
            wrong = sde_step_with_logprob(
                scheduler,
                velocity,
                scheduler.timesteps[:1],
                state,
                prev_sample=sampled.prev_sample,
                noise_level=0.05,
                step_index=0,
            )
            assert not torch.allclose(wrong.log_prob, sampled.log_prob)
        state = sampled.prev_sample
    # Restarted records store only their local levels, while plans remain absolute.
    restarted = {
        "timesteps": scheduler.timesteps[1:],
        "schedule_start_index": 1,
        "noise_level": 0.05,
        "noise_levels": [0.05],
    }
    assert recorded_noise_levels(restarted, plan) == [0.05]
    del trajectory["noise_levels"]
    with pytest.raises(ValueError, match="disagrees"):
        recorded_noise_levels(trajectory, plan)
    with pytest.raises(ValueError, match="positive"):
        noise_schedule({"noise_level": 0.05, "noise_levels": [0.5, 0]}, 2)
    with pytest.raises(ValueError, match="per step"):
        noise_schedule({"noise_level": 0.05, "noise_levels": [0.5]}, 2)
