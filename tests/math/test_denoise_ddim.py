"""Tests for the DDIM eta-SDE log-prob math (epsilon + v-prediction).

Parity source of truth is diffusers' own DDIMScheduler.step: with a fixed
variance noise of zero, its output IS the transition mean, so our
``prev_sample_mean`` must match it exactly — no re-derived reference formula.
"""

from __future__ import annotations

import pytest
import torch

from vrl.math.denoise.flow_matching import sde_step_with_logprob

_SHAPE = (2, 4, 8, 8)


def _scheduler(prediction_type: str):
    from diffusers import DDIMScheduler

    scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        beta_schedule="scaled_linear",
        prediction_type=prediction_type,
        set_alpha_to_one=True,
        clip_sample=False,
    )
    scheduler.set_timesteps(10)
    return scheduler


@pytest.mark.parametrize("prediction_type", ["epsilon", "v_prediction"])
@pytest.mark.parametrize("step_idx", [0, 9])
def test_ddim_mean_matches_diffusers_step(prediction_type: str, step_idx: int) -> None:
    """prev_sample_mean equals DDIMScheduler.step with eta and zero noise."""
    scheduler = _scheduler(prediction_type)
    torch.manual_seed(0)
    sample = torch.randn(_SHAPE)
    model_output = torch.randn(_SHAPE)
    t = scheduler.timesteps[step_idx]

    result = sde_step_with_logprob(
        scheduler,
        model_output,
        t.expand(_SHAPE[0]),
        sample,
        prev_sample=torch.zeros(_SHAPE),  # evaluate-only; mean is what we test
        noise_level=1.0,
        sde_type="ddim",
        step_index=step_idx,
    )

    ref = scheduler.step(
        model_output,
        t,
        sample,
        eta=1.0,
        variance_noise=torch.zeros(_SHAPE),
        return_dict=False,
    )[0]
    torch.testing.assert_close(result.prev_sample_mean, ref, atol=1e-5, rtol=1e-5)


def test_ddim_sampling_then_replay_logprob_parity() -> None:
    """The log-prob of a sampled transition equals its replay evaluation."""
    scheduler = _scheduler("epsilon")
    torch.manual_seed(1)
    sample = torch.randn(_SHAPE)
    model_output = torch.randn(_SHAPE)
    t = scheduler.timesteps[3]
    generator = torch.Generator().manual_seed(7)

    sampled = sde_step_with_logprob(
        scheduler,
        model_output,
        t.expand(_SHAPE[0]),
        sample,
        generator=generator,
        sde_type="ddim",
        step_index=3,
    )
    replayed = sde_step_with_logprob(
        scheduler,
        model_output,
        t.expand(_SHAPE[0]),
        sample,
        prev_sample=sampled.prev_sample,
        sde_type="ddim",
        step_index=3,
    )
    torch.testing.assert_close(sampled.log_prob, replayed.log_prob)
    assert torch.isfinite(sampled.log_prob).all()


def test_ddim_deterministic_is_eta_zero() -> None:
    """deterministic=True reproduces the eta=0 DDIM (ODE) step."""
    scheduler = _scheduler("epsilon")
    torch.manual_seed(2)
    sample = torch.randn(_SHAPE)
    model_output = torch.randn(_SHAPE)
    t = scheduler.timesteps[5]

    result = sde_step_with_logprob(
        scheduler,
        model_output,
        t.expand(_SHAPE[0]),
        sample,
        deterministic=True,
        sde_type="ddim",
        step_index=5,
    )
    ref = scheduler.step(model_output, t, sample, eta=0.0, return_dict=False)[0]
    torch.testing.assert_close(result.prev_sample, ref, atol=1e-5, rtol=1e-5)


def test_ddim_final_step_uses_terminal_alpha() -> None:
    """The last ladder step falls back to final_alpha_cumprod instead of indexing off the table."""
    scheduler = _scheduler("epsilon")
    torch.manual_seed(3)
    sample = torch.randn(_SHAPE)
    model_output = torch.randn(_SHAPE)
    last = len(scheduler.timesteps) - 1
    t = scheduler.timesteps[last]

    result = sde_step_with_logprob(
        scheduler,
        model_output,
        t.expand(_SHAPE[0]),
        sample,
        generator=torch.Generator().manual_seed(9),
        sde_type="ddim",
        step_index=last,
    )
    ref = scheduler.step(
        model_output,
        t,
        sample,
        eta=1.0,
        variance_noise=torch.zeros(_SHAPE),
        return_dict=False,
    )[0]
    torch.testing.assert_close(result.prev_sample_mean, ref, atol=1e-5, rtol=1e-5)
    assert torch.isfinite(result.log_prob).all()
