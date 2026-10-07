"""Restarted trajectories replay with the original SDE schedule endpoints."""

from types import SimpleNamespace

import pytest
import torch

from vrl.math.denoise.flow_matching import sde_step_with_logprob
from vrl.scripts.generation.qwen_layered_rl_probe import recorded_schedule


def test_restart_mapping_preserves_density_when_sliced_schedule_does_not():
    full = SimpleNamespace(
        sigmas=torch.tensor([0.9, 0.7, 0.5, 0.3, 0.1, 0.0]),
        timesteps=torch.tensor([900.0, 700.0, 500.0, 300.0, 100.0]),
    )
    state = torch.tensor([[[0.3, -0.2]]])
    velocity = torch.tensor([[[0.1, 0.4]]])
    transition = sde_step_with_logprob(
        full,
        velocity,
        full.timesteps[2:3],
        state,
        generator=torch.Generator().manual_seed(7),
        noise_level=1.0,
        step_index=2,
    )
    trajectory = {
        "teacher_initialized": True,
        "schedule_start_index": 2,
        "full_sigmas": full.sigmas,
        "full_timesteps": full.timesteps,
        "absolute_step_indices": torch.tensor([2, 3, 4]),
        "sigmas": full.sigmas[2:],
        "timesteps": full.timesteps[2:],
    }
    sigmas, times, indices = recorded_schedule(trajectory)
    replay = sde_step_with_logprob(
        SimpleNamespace(sigmas=sigmas, timesteps=times),
        velocity,
        trajectory["timesteps"][:1],
        state,
        prev_sample=transition.prev_sample,
        noise_level=1.0,
        step_index=int(indices[0]),
    )
    torch.testing.assert_close(replay.log_prob, transition.log_prob, rtol=0, atol=0)
    incorrect = sde_step_with_logprob(
        SimpleNamespace(sigmas=trajectory["sigmas"], timesteps=trajectory["timesteps"]),
        velocity,
        trajectory["timesteps"][:1],
        state,
        prev_sample=transition.prev_sample,
        noise_level=1.0,
        step_index=0,
    )
    assert not torch.allclose(incorrect.log_prob, transition.log_prob)
    trajectory["absolute_step_indices"] = torch.tensor([1, 2, 3])
    with pytest.raises(ValueError, match="step mapping"):
        recorded_schedule(trajectory)


def test_restart_without_original_schedule_is_rejected():
    trajectory = {
        "teacher_initialized": True,
        "schedule_start_index": 45,
        "sigmas": torch.tensor([0.2, 0.1, 0.0]),
        "timesteps": torch.tensor([200.0, 100.0]),
    }
    with pytest.raises(ValueError, match="complete sampling schedule"):
        recorded_schedule(trajectory)
