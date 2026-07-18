"""Per-token flow-matching sample-with-logprob for NextStep-1.

NextStep-1 generates each image token via a small flow-matching MLP head
conditioned on the LLM's hidden state. For RL we need both the sampled
token and a Gaussian log-probability so we can form the GRPO ratio.

This mirrors ``vrl.rollouts.evaluators.denoise.sde_logprob`` but at
the per-token granularity: each AR step does its own K-step flow ODE
internally, then injects one Gaussian noise at the final step. The
log-probability is the standard isotropic-Gaussian density of that noise.

Why not full path-integral log-probability?
-------------------------------------------
NextStep-1.1's RL post-training (per the model card) treats each
generated token as Gaussian-distributed around the deterministic-flow
mean — exactly what flow_grpo's ``sde_step_with_logprob`` does for
diffusion. We follow the same convention.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import torch


def _flow_terminal_mean(
    image_head: Any,
    cond: torch.Tensor,
    x: torch.Tensor,
    *,
    num_steps: int,
    cfg_uncond: torch.Tensor | None,
    guidance_scale: float,
    velocity_fn: Callable[..., torch.Tensor] | None,
) -> torch.Tensor:
    """Run the deterministic Euler prefix and return the terminal flow mean.

    Single construction site for the velocity contract, the CFG combine, and
    the Euler loop — the sampling and replay paths below MUST walk the exact
    same trajectory, or the GRPO old/fresh log-prob ratio silently corrupts.

    Velocity contract (verified against ``stepfun-ai/NextStep-1``'s
    ``modeling_nextstep.FlowMatchingHead``): the head exposes its velocity
    predictor as ``image_head.net(x, t, c)`` (a ``SimpleMLPAdaLN``). The
    head's own ``forward(z, target, mask)`` is the training-loss path, not
    the velocity field, so we never call the head directly — always ``.net``.
    Pass ``velocity_fn`` to override.
    """

    B = cond.shape[0]
    t_grid = torch.linspace(0.0, 1.0, num_steps + 1, device=x.device, dtype=x.dtype)
    dt = 1.0 / num_steps

    def _velocity(xk: torch.Tensor, tk: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        if velocity_fn is not None:
            return velocity_fn(xk, tk, c)
        return image_head.net(xk, tk, c)

    def _guided_velocity(xk: torch.Tensor, tk: torch.Tensor) -> torch.Tensor:
        v_cond = _velocity(xk, tk, cond)
        if cfg_uncond is not None and guidance_scale > 1.0:
            v_uncond = _velocity(xk, tk, cfg_uncond)
            return v_uncond + guidance_scale * (v_cond - v_uncond)
        return v_cond

    # K-1 deterministic Euler steps, then the final step's mean.
    for k in range(num_steps - 1):
        x = x + dt * _guided_velocity(x, t_grid[k].expand(B))
    return x + dt * _guided_velocity(x, t_grid[num_steps - 1].expand(B))


def _flow_noise_std(noise_level: float, num_steps: int) -> float:
    """SDE-from-ODE final-step std: ``noise_level * sqrt(dt)``.

    Matches flow_grpo's final-step parameterisation (sigma_min ≪ sigma_max in
    a flat schedule). Single construction site so sampling and replay always
    normalize by the same scale.
    """

    return noise_level * math.sqrt(1.0 / num_steps)


def _isotropic_gaussian_logprob(delta: torch.Tensor, std_scalar: float) -> torch.Tensor:
    """``log N(delta; 0, std^2 I)`` summed over the token dim, per sample.

    SUM (not mean) over D so that ratios of fresh/old log-probs stay correctly
    scaled across batch entries.
    """

    dim = delta.shape[-1]
    sq_err = (delta**2).sum(dim=-1)  # [B]
    return (
        -sq_err / (2.0 * std_scalar**2)
        - float(dim) * math.log(max(std_scalar, 1e-12))
        - 0.5 * float(dim) * math.log(2.0 * math.pi)
    )


def flow_sample_with_logprob(
    image_head: Any,
    cond: torch.Tensor,  # [B, D_hidden] — LLM last hidden state
    *,
    num_steps: int = 20,
    noise_level: float = 1.0,
    cfg_uncond: torch.Tensor | None = None,
    guidance_scale: float = 1.0,
    generator: torch.Generator | None = None,
    initial_noise: torch.Tensor | None = None,
    velocity_fn: Callable[..., torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample one continuous token from the flow head and return its log-prob.

    Algorithm (matching the diffusion side's SDE-with-logprob convention):

        1. Initialise ``x_0 ~ N(0, I)`` (or use the head's prescribed prior).
        2. Run ``num_steps - 1`` deterministic Euler steps on the
           flow ODE: ``x_{k+1} = x_k + dt * v(x_k, t_k, cond)``.
        3. At the final step, inject Gaussian noise of scale ``std``:
           ``token = x_{K-1} + dt * v(x_{K-1}, t_{K-1}, cond) + std * eps``.
        4. ``log_prob = -0.5 * ||eps||^2 / std^2 - D_token * log(std)
                       - 0.5 * D_token * log(2 pi)``.

    Args:
        image_head: NextStep-1's flow-matching MLP (``model.image_head``).
            Its velocity predictor is called as ``image_head.net(x, t, cond)``
            (see ``_flow_terminal_mean`` for the verified upstream contract).
        cond: ``[B, D_hidden]`` LLM hidden state at this AR position.
        num_steps: Number of Euler steps inside the flow ODE.
        noise_level: Scales the final-step Gaussian std (analogue of the
            ``a`` knob in flow_grpo's SDE-from-ODE conversion). 0 → fully
            deterministic (zero log-prob mass), 1 → unit-variance noise.
        cfg_uncond: ``[B, D_hidden]`` unconditional hidden state for CFG.
            When provided, velocity is computed as
                v_guided = v(x, uncond) + s * (v(x, cond) - v(x, uncond))
            with ``s = guidance_scale``. Enabled only when guidance_scale > 1,
            matching NextStep's upstream CFG branch.
        guidance_scale: CFG strength on the velocity.
        generator: Optional torch.Generator for reproducibility.
        initial_noise: Optional explicit ``x_0`` prior. When provided, this
            exact tensor is used as the deterministic flow prefix and returned
            for replay.
        velocity_fn: Optional override for how to call image_head.

    Returns:
        ``(token, log_prob, initial_noise)`` where ``token`` is ``[B, D]``,
        ``log_prob`` is ``[B]``, and ``initial_noise`` is the replay prior.
    """
    B = cond.shape[0]
    device = cond.device
    dtype = cond.dtype

    # NextStep-1's FlowMatchingHead stores the token latent dim as
    # ``image_head.input_dim`` = num_channels * patch_size².
    token_dim = getattr(image_head, "input_dim", None)
    if token_dim is None:
        raise RuntimeError(
            "flow_sample_with_logprob: image_head has no ``input_dim`` "
            "attribute — upstream API may have changed. Inspect "
            "FlowMatchingHead.__init__ in modeling_nextstep.py."
        )
    D = int(token_dim)

    # x_0 ~ N(0, I) — flow-matching prior is standard normal. This is the
    # replay artifact saved by NextStep1Model; when supplied explicitly, do
    # not sample another prior inside this helper.
    if initial_noise is None:
        x = torch.randn(B, D, device=device, dtype=dtype, generator=generator)
    else:
        if initial_noise.shape != (B, D):
            raise ValueError(
                f"initial_noise must have shape {(B, D)}, got {tuple(initial_noise.shape)}"
            )
        x = initial_noise.to(device=device, dtype=dtype)
    x0 = x

    mean = _flow_terminal_mean(
        image_head,
        cond,
        x,
        num_steps=num_steps,
        cfg_uncond=cfg_uncond,
        guidance_scale=guidance_scale,
        velocity_fn=velocity_fn,
    )

    std_scalar = _flow_noise_std(noise_level, num_steps)
    eps = torch.randn(
        mean.shape,
        device=mean.device,
        dtype=mean.dtype,
        generator=generator,
    )
    token = mean + std_scalar * eps

    log_prob = _isotropic_gaussian_logprob(token.detach() - mean, std_scalar)

    return token, log_prob, x0


def flow_logprob_at(
    image_head: Any,
    cond: torch.Tensor,  # [B, D_hidden]
    target_token: torch.Tensor,  # [B, D_token]
    saved_noise: torch.Tensor | None = None,
    *,
    num_steps: int = 20,
    noise_level: float = 1.0,
    cfg_uncond: torch.Tensor | None = None,
    guidance_scale: float = 1.0,
    velocity_fn: Callable[..., torch.Tensor] | None = None,
) -> torch.Tensor:
    """Recompute log-prob of a previously-sampled continuous token.

    The replay assumes the deterministic flow ODE prefix is **the same
    seed-determined trajectory** as collection time — which we can only
    guarantee if we either (a) save the prior x_0 and any per-step noise
    as ``saved_noise``, or (b) accept a Monte-Carlo approximation by
    re-running with a fresh prior.

    Strategy implemented here: re-run the deterministic prefix with the
    *current* policy's velocity field, take its terminal mean ``mu``, and
    compute log p(target_token | mu, std). This matches what
    ``sde_step_with_logprob`` does in the diffusion path — the "old" and
    "fresh" log-probs differ only because the velocity field has been
    updated by SGD, which is exactly what GRPO's ratio is supposed to
    capture.

    Returns:
        ``[B]`` log-probabilities with grad flowing through ``image_head``.
    """
    B, D = target_token.shape
    device = target_token.device
    dtype = target_token.dtype

    if saved_noise is not None:
        # Reproducible mode: caller has stashed x_0 and used it everywhere.
        x = saved_noise.to(device=device, dtype=dtype)
    else:
        # Fallback mode: fresh x_0 ~ N(0, I). Biased, but matches what the
        # SD3/Wan flow-matching path does today (they also don't replay
        # collection-time noise verbatim).
        x = torch.randn(B, D, device=device, dtype=dtype)

    mean = _flow_terminal_mean(
        image_head,
        cond,
        x,
        num_steps=num_steps,
        cfg_uncond=cfg_uncond,
        guidance_scale=guidance_scale,
        velocity_fn=velocity_fn,
    )

    std_scalar = _flow_noise_std(noise_level, num_steps)
    return _isotropic_gaussian_logprob(target_token - mean, std_scalar)
