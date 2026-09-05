"""Gaussian re-noise transitions for chunk-autoregressive denoise policies.

CausVid predicts a clean batch ``x0`` and then corrupts that prediction to the
next schedule sigma before the next model forward.  This module owns the exact
transition distribution used by both rollout and replay; cache and temporal
orchestration remain family-owned.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class RenoiseStepResult:
    """Named outputs of one ``x0 -> x_sigma`` Gaussian transition."""

    next_sample: Any
    log_prob: Any


def renoise_step_with_logprob(
    pred_original_sample: Any,
    next_sigma: Any,
    *,
    next_sample: Any | None = None,
    noise: Any | None = None,
    math_dtype: Any = None,
) -> RenoiseStepResult:
    """Sample or score ``x_next ~ N((1-sigma)*x0, sigma^2 I)``.

    The returned log probability is the mean elementwise Gaussian log density,
    matching the scale convention used by VRL's other continuous denoise
    policies.  ``next_sample`` is detached in the density so replay gradients
    flow through the current model prediction, never through the recorded
    behavior action.

    Sampling always consumes the caller-supplied ``noise``; there is no internal
    fresh-noise draw, so rollout owns the per-sample RNG (``runner.py`` builds one
    ``torch.Generator`` per sample) and replay scores the exact stored action.
    Exactly one of ``noise`` (sample) or ``next_sample`` (score) must be given.

    Rollout quantizes the sampled action to the input latent dtype *before*
    scoring it. Replay therefore scores the exact serialized action rather than
    a higher-precision value that was never observed by the next model forward.
    """

    import torch

    if next_sample is not None and noise is not None:
        raise ValueError("next_sample cannot be combined with noise")
    if next_sample is None and noise is None:
        raise ValueError("exactly one of noise or next_sample is required")

    dtype = torch.float32 if math_dtype is None else math_dtype
    original_dtype = pred_original_sample.dtype
    x0 = pred_original_sample.to(dtype)
    sigma = torch.as_tensor(next_sigma, device=x0.device, dtype=dtype)
    if sigma.ndim > 1:
        raise ValueError("next_sigma must be scalar or sample-aligned rank-1")
    if sigma.ndim == 1 and int(sigma.shape[0]) not in (1, int(x0.shape[0])):
        raise ValueError(
            "next_sigma rank-1 length must be 1 or match the sample batch: "
            f"{int(sigma.shape[0])} != {int(x0.shape[0])}",
        )
    if bool((sigma <= 0).any()):
        raise ValueError("next_sigma must be > 0 for a stochastic re-noise transition")
    sigma_view = sigma.reshape((-1,) + (1,) * (x0.ndim - 1)) if sigma.ndim else sigma
    mean = (1 - sigma_view) * x0

    if next_sample is None:
        noise = noise.to(device=x0.device, dtype=dtype)
        if tuple(noise.shape) != tuple(x0.shape):
            raise ValueError(
                f"noise shape {tuple(noise.shape)} must match latent shape {tuple(x0.shape)}",
            )
        # This cast is part of the behavior policy: the next causal forward sees
        # the checkpoint dtype, and replay must score that same stored tensor.
        action = (mean + sigma_view * noise).to(original_dtype)
    else:
        if tuple(next_sample.shape) != tuple(x0.shape):
            raise ValueError(
                "next_sample shape must match pred_original_sample: "
                f"{tuple(next_sample.shape)} != {tuple(x0.shape)}",
            )
        action = next_sample

    scored_action = action.detach().to(dtype)
    log_prob = (
        -((scored_action - mean) ** 2) / (2 * sigma_view**2)
        - torch.log(sigma_view)
        - 0.5 * math.log(2 * math.pi)
    )
    log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))
    return RenoiseStepResult(
        next_sample=action,
        log_prob=log_prob,
    )


__all__ = ["RenoiseStepResult", "renoise_step_with_logprob"]
