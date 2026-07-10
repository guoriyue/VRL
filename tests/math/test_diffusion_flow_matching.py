"""Sigma-domain contract for diffusion SDE log-prob math.

Cosmos Predict2's FlowMatch scheduler keeps sigmas in the EDM domain
(sigma_max=80) while ``sde_step_with_logprob`` is derived for rectified-flow
sigmas in [0, 1]. Feeding EDM sigmas straight into the [0, 1] formulas produced
~sigma^2-scale log-probs (observed -4635 on Predict2 GRPO) and broke the
replay ratio==1 invariant. These tests pin the internal EDM->flow conversion:
it must be the exact domain isomorphism, keep flow-native schedulers bitwise
on the old path, and restore O(1) log-prob magnitudes plus sample/replay
parity for EDM schedulers.
"""

from __future__ import annotations

import pytest
import torch

from vrl.math.diffusion.flow_matching import sde_step_with_logprob


class _FakeScheduler:
    """Minimal scheduler: a fixed descending sigma table plus timesteps."""

    def __init__(self, sigmas: torch.Tensor) -> None:
        self.sigmas = sigmas
        self.timesteps = sigmas[:-1] * 1000


# Descending EDM-domain table (max > 1 marks the domain); the matching flow
# table is its exact image under t = sigma / (1 + sigma).
_EDM_SIGMAS = torch.tensor([8.0, 4.0, 2.0, 1.0, 0.25, 0.25])
_FLOW_SIGMAS = _EDM_SIGMAS / (1 + _EDM_SIGMAS)


def _timestep(batch: int, value: float = 123.0) -> torch.Tensor:
    return torch.full((batch,), value)


@pytest.mark.parametrize("sde_type", ["flow_grpo", "cps"])
def test_edm_call_equals_flow_call_under_domain_isomorphism(sde_type: str) -> None:
    """Checks the EDM path is exactly the flow path conjugated by t=s/(1+s)."""
    step = 1
    sigma = _EDM_SIGMAS[step]
    sigma_prev = _EDM_SIGMAS[step + 1]
    sample_edm = torch.randn(2, 3, 4, generator=torch.Generator().manual_seed(1))
    noise_pred = torch.randn(2, 3, 4, generator=torch.Generator().manual_seed(2))

    res_edm = sde_step_with_logprob(
        _FakeScheduler(_EDM_SIGMAS.clone()),
        noise_pred,
        _timestep(2),
        sample_edm,
        generator=torch.Generator().manual_seed(7),
        noise_level=0.7,
        sde_type=sde_type,
        step_index=step,
    )

    # Hand-converted flow-domain inputs: x_flow = x/(1+s); the EDM model output
    # is the noise estimate n=(x-x0)/s, so the flow velocity (noise - x0) is
    # n*(1+s) - x.
    velocity = noise_pred * (1 + sigma) - sample_edm
    res_flow = sde_step_with_logprob(
        _FakeScheduler(_FLOW_SIGMAS.clone()),
        velocity,
        _timestep(2),
        sample_edm / (1 + sigma),
        generator=torch.Generator().manual_seed(7),
        noise_level=0.7,
        sde_type=sde_type,
        step_index=step,
    )

    torch.testing.assert_close(res_edm.log_prob, res_flow.log_prob)
    torch.testing.assert_close(res_edm.std_dev_t, res_flow.std_dev_t)
    torch.testing.assert_close(res_edm.prev_sample_mean, res_flow.prev_sample_mean)
    # prev_sample alone converts back to the scheduler's own domain.
    torch.testing.assert_close(
        res_edm.prev_sample,
        res_flow.prev_sample * (1 + sigma_prev),
    )


@pytest.mark.parametrize("sde_type", ["flow_grpo", "cps"])
def test_edm_sampling_then_replay_logprob_parity(sde_type: str) -> None:
    """Checks replaying the recorded prev_sample reproduces the exact log-prob.

    This is the GRPO ratio == 1 invariant that EDM-domain sigmas broke on
    Predict2: with unchanged weights, replay log-prob must equal collection
    log-prob bit-for-bit (same fp32 math, same inputs).
    """
    scheduler = _FakeScheduler(_EDM_SIGMAS.clone())
    sample = torch.randn(2, 3, 4, generator=torch.Generator().manual_seed(3))
    noise_pred = torch.randn(2, 3, 4, generator=torch.Generator().manual_seed(4))

    sampled = sde_step_with_logprob(
        scheduler,
        noise_pred,
        _timestep(2),
        sample,
        generator=torch.Generator().manual_seed(11),
        noise_level=1.0,
        sde_type=sde_type,
        step_index=1,
    )
    replayed = sde_step_with_logprob(
        scheduler,
        noise_pred,
        _timestep(2),
        sample,
        prev_sample=sampled.prev_sample,
        noise_level=1.0,
        sde_type=sde_type,
        step_index=1,
    )

    torch.testing.assert_close(replayed.log_prob, sampled.log_prob)
    # Regression pin for the observed failure mode: with EDM sigmas the cps
    # log-prob landed at ~ -sigma_prev^2 (-4635 in the smoke run). Flow-domain
    # magnitudes are O(1).
    assert sampled.log_prob.abs().max().item() < 50.0


def test_flow_scheduler_keeps_legacy_path() -> None:
    """Checks [0,1]-sigma schedulers are not converted (sd3/wan/predict2.5)."""
    scheduler = _FakeScheduler(_FLOW_SIGMAS.clone())
    step = 1
    sigma = _FLOW_SIGMAS[step]
    sigma_prev = _FLOW_SIGMAS[step + 1]
    sample = torch.randn(2, 3, generator=torch.Generator().manual_seed(5))
    velocity = torch.randn(2, 3, generator=torch.Generator().manual_seed(6))

    res = sde_step_with_logprob(
        scheduler,
        velocity,
        _timestep(2),
        sample,
        deterministic=True,
        noise_level=1.0,
        sde_type="flow_grpo",
        step_index=step,
    )

    assert scheduler._vrl_edm_sigma_domain is False
    # Deterministic Euler step in the scheduler's own (already-flow) domain —
    # the pre-fix formula, bit-for-bit.
    torch.testing.assert_close(
        res.prev_sample,
        sample + (sigma_prev - sigma) * velocity,
    )


def test_edm_domain_detection_is_cached_on_scheduler() -> None:
    """Checks the runtime-sigma domain verdict is computed once and cached."""
    scheduler = _FakeScheduler(_EDM_SIGMAS.clone())
    sample = torch.randn(1, 3)
    noise_pred = torch.randn(1, 3)

    sde_step_with_logprob(
        scheduler,
        noise_pred,
        _timestep(1),
        sample,
        generator=torch.Generator().manual_seed(0),
        step_index=1,
    )

    assert scheduler._vrl_edm_sigma_domain is True


# ── diffusion_pretraining_pair (the sft-regularizer noising construction) ────


def test_pretraining_pair_flow_matching_uses_scheduler_scale_noise() -> None:
    """Flow schedulers own x_t; the target is the rectified-flow velocity."""
    from diffusers import FlowMatchEulerDiscreteScheduler

    from vrl.math.diffusion.flow_matching import diffusion_pretraining_pair

    scheduler = FlowMatchEulerDiscreteScheduler()
    scheduler.set_timesteps(8)
    torch.manual_seed(0)
    latents = torch.randn(2, 3, 4)
    noise = torch.randn(2, 3, 4)
    t = scheduler.timesteps[3].expand(2)

    noisy, target = diffusion_pretraining_pair(scheduler, latents, noise, t)

    torch.testing.assert_close(noisy, scheduler.scale_noise(latents, t, noise))
    torch.testing.assert_close(target, noise - latents)


@pytest.mark.parametrize("prediction_type", ["epsilon", "v_prediction"])
def test_pretraining_pair_ddpm_ladder_targets(prediction_type: str) -> None:
    from diffusers import DDPMScheduler

    from vrl.math.diffusion.flow_matching import diffusion_pretraining_pair

    scheduler = DDPMScheduler(prediction_type=prediction_type)
    torch.manual_seed(0)
    latents = torch.randn(2, 3, 4)
    noise = torch.randn(2, 3, 4)
    t = torch.tensor([10, 500])

    noisy, target = diffusion_pretraining_pair(scheduler, latents, noise, t)

    torch.testing.assert_close(noisy, scheduler.add_noise(latents, noise, t))
    expected = (
        noise
        if prediction_type == "epsilon"
        else scheduler.get_velocity(latents, noise, t)
    )
    torch.testing.assert_close(target, expected)


def test_pretraining_pair_rejects_unknown_prediction_type() -> None:
    from vrl.math.diffusion.flow_matching import diffusion_pretraining_pair

    class _Sched:
        class config:
            prediction_type = "sample"

        def add_noise(self, latents, noise, t):
            return latents

    with pytest.raises(ValueError, match="prediction_type"):
        diffusion_pretraining_pair(_Sched(), torch.zeros(1), torch.zeros(1), torch.zeros(1))
