"""DDIM eta-SDE math for epsilon/v-prediction diffusion families.

The flow-matching SDE in ``flow_matching.py`` assumes rectified-flow sigmas;
DDPM-family checkpoints (PixArt-Sigma, CogVideoX) live on an
``alphas_cumprod`` ladder instead. This module provides the DDPO/DanceGRPO
step: the DDIM sampler with ``eta`` noise, whose per-step transition is the
Gaussian

    x_{t-1} ~ N( sqrt(abar_prev) * x0_hat + sqrt(1 - abar_prev - sigma_t^2) * eps_hat,
                 sigma_t^2 )
    sigma_t  = eta * sqrt((1 - abar_prev)/(1 - abar_t)) * sqrt(1 - abar_t/abar_prev)

with ``x0_hat``/``eps_hat`` derived from the model output according to
``scheduler.config.prediction_type`` (``epsilon`` or ``v_prediction``).

Dispatch: callers keep using ``sde_step_with_logprob(..., sde_type="ddim")``;
the flow-matching entry point forwards here, so the executor/evaluator call
sites stay untouched. ``noise_level`` maps onto ``eta`` (1.0 = DDPM ancestral,
the DDPO default); ``deterministic=True`` is ``eta=0``.
"""

from __future__ import annotations

import math
from typing import Any

from vrl.math.denoise.flow_matching import SDEStepResult


def ddim_step_with_logprob(
    scheduler: Any,
    model_output: Any,
    timestep: Any,
    sample: Any,
    prev_sample: Any | None = None,
    generator: Any | None = None,
    deterministic: bool = False,
    return_dt: bool = False,
    noise_level: float = 1.0,
    math_dtype: Any = None,
    step_index: Any = None,
) -> SDEStepResult:
    """Compute one DDIM eta-SDE step and its per-sample log-probability.

    Contract mirrors ``sde_step_with_logprob``: fp32 math by default,
    ``step_index`` avoids per-step host syncs, ``prev_sample`` given means
    "evaluate the log-prob of this stored transition" (replay), otherwise the
    step samples. ``sqrt_neg_dt`` is always None — the full noise scale lives
    in ``std_dev_t``, which is the branch ``compute_kl_divergence`` already
    handles.
    """
    import torch
    from diffusers.utils.torch_utils import randn_tensor

    md = torch.float32 if math_dtype is None else math_dtype
    model_output = model_output.to(md)
    sample = sample.to(md)
    if prev_sample is not None:
        prev_sample = prev_sample.to(md)

    timesteps_table = scheduler.timesteps.to(sample.device)
    if step_index is None:
        # CogVideoX's DDIM variant has no index_for_timestep; search the table.
        lookup = getattr(scheduler, "index_for_timestep", None)
        if lookup is not None:
            step_index = [lookup(t) for t in timestep]
        else:
            step_index = [int((timesteps_table == t).nonzero()[0].item()) for t in timestep]
    elif isinstance(step_index, int):
        step_index = [step_index] * len(timestep)

    alphas_cumprod = scheduler.alphas_cumprod.to(sample.device, md)
    final_alpha_cumprod = getattr(scheduler, "final_alpha_cumprod", None)
    if final_alpha_cumprod is None:
        final_alpha_cumprod = torch.ones((), dtype=md)
    final_alpha_cumprod = torch.as_tensor(final_alpha_cumprod).to(sample.device, md)

    ndim = sample.ndim
    view_shape = (-1,) + (1,) * (ndim - 1)

    t_now = timesteps_table[step_index]
    abar_t = alphas_cumprod[t_now.long()].view(*view_shape)
    # The step after the last one leaves the ladder: abar_prev falls back to
    # the scheduler's terminal value (1.0 for set_alpha_to_one).
    last = len(timesteps_table) - 1
    abar_prev_flat = torch.stack(
        [
            alphas_cumprod[timesteps_table[i + 1].long()] if i < last else final_alpha_cumprod
            for i in step_index
        ],
    )
    abar_prev = abar_prev_flat.view(*view_shape)

    beta_t = 1.0 - abar_t
    beta_prev = 1.0 - abar_prev

    if getattr(scheduler.config, "clip_sample", False) or getattr(
        scheduler.config,
        "thresholding",
        False,
    ):
        raise ValueError(
            "ddim logprob assumes an unclipped Gaussian transition; construct "
            "the rollout scheduler with clip_sample=False, thresholding=False",
        )
    prediction_type = str(getattr(scheduler.config, "prediction_type", "epsilon"))
    if prediction_type == "epsilon":
        eps_hat = model_output
        x0_hat = (sample - beta_t.sqrt() * eps_hat) / abar_t.sqrt()
    elif prediction_type == "v_prediction":
        x0_hat = abar_t.sqrt() * sample - beta_t.sqrt() * model_output
        eps_hat = abar_t.sqrt() * model_output + beta_t.sqrt() * sample
    else:
        raise ValueError(
            f"ddim logprob supports epsilon/v_prediction, got {prediction_type!r}",
        )

    eta = 0.0 if deterministic else float(noise_level)
    std_dev_t = eta * (beta_prev / beta_t).sqrt() * (1.0 - abar_t / abar_prev).sqrt()

    prev_sample_mean = (
        abar_prev.sqrt() * x0_hat + (beta_prev - std_dev_t**2).clamp_min(0.0).sqrt() * eps_hat
    )

    if prev_sample is not None and generator is not None:
        raise ValueError("Cannot pass both generator and prev_sample.")

    if prev_sample is None:
        if eta > 0.0:
            variance_noise = randn_tensor(
                model_output.shape,
                generator=generator,
                device=model_output.device,
                dtype=model_output.dtype,
            )
            prev_sample = prev_sample_mean + std_dev_t * variance_noise
        else:
            prev_sample = prev_sample_mean

    err_sq = (prev_sample.detach() - prev_sample_mean) ** 2
    # Zero-variance transitions are Dirac: eta=0 everywhere, and the terminal
    # step even at eta=1 (set_alpha_to_one makes beta_prev = 0). The pseudo
    # log-prob there mirrors the cps convention (negative squared error) so
    # those steps stay finite instead of producing log(0).
    zero_var = std_dev_t == 0
    if eta > 0.0 and not bool(zero_var.all()):
        safe_std = torch.where(zero_var, torch.ones_like(std_dev_t), std_dev_t)
        gaussian = (
            -err_sq / (2 * safe_std**2)
            - torch.log(safe_std)
            - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi)))
        )
        log_prob = torch.where(zero_var.expand_as(gaussian), -err_sq, gaussian)
    else:
        log_prob = -err_sq
    log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))

    del return_dt  # DDIM's full noise scale lives in std_dev_t.
    return SDEStepResult(
        prev_sample=prev_sample,
        log_prob=log_prob,
        prev_sample_mean=prev_sample_mean,
        std_dev_t=std_dev_t,
        sqrt_neg_dt=None,
    )


__all__ = ["ddim_step_with_logprob"]
