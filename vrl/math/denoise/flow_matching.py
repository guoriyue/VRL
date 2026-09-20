"""Flow-matching SDE math shared by generation and training replay."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class SDEStepResult:
    """Named outputs shared by flow-matching and DDIM denoising steps."""

    prev_sample: Any
    log_prob: Any
    prev_sample_mean: Any
    std_dev_t: Any
    sqrt_neg_dt: Any | None = None
    # Current step's sigma in [0, 1] after EDM-to-flow conversion. The standard
    # deviation and sqrt(-dt) are separate flow-domain quantities. Flash-GRPO's temporal
    # gradient rectification reads it: the log-prob gradient magnitude scales as
    # sqrt(-dt)/std + std*sqrt(-dt)*(1-sigma)/(2*sigma), and the loss weight is
    # that factor's reciprocal. None on the DDIM path (alphas ladder, no sigma).
    sigma: Any | None = None


@dataclass(frozen=True, slots=True)
class _FlowSchedule:
    """One replay step's schedule terms in the rectified-flow [0, 1] domain.

    Read once per step by ``sde_step_with_logprob`` and by the model-free
    ``flow_sde_scale_terms``, so both derive dt and the flow_grpo noise scale
    from the same numbers. ``one_plus_sigma`` / ``one_plus_sigma_prev`` are set
    only for EDM-domain sigma tables and carry the sample/velocity conversion.
    """

    sigma: Any
    sigma_prev: Any
    sigma_max: Any
    sigma_min: Any
    one_plus_sigma: Any | None
    one_plus_sigma_prev: Any | None

    @classmethod
    def read(
        cls,
        scheduler: Any,
        step_index: list[int],
        *,
        device: Any,
        view_shape: tuple[int, ...],
    ) -> _FlowSchedule:
        """Read sigma / sigma_prev / endpoints for ``step_index`` in the flow domain.

        Cosmos Predict2's FlowMatch scheduler keeps its sigma table in the EDM
        domain (sigma_max=80) while the SDE math is derived for rectified-flow
        sigmas in [0, 1] (x_t = (1-t)*x0 + t*noise). Feeding EDM sigmas through
        the [0, 1] formulas silently produces ~sigma^2-scale garbage log-probs
        (observed: logprob ~ -68^2 on Predict2 GRPO). Detect the domain from the
        RUNTIME sigma table — config keys are unreliable: Predict2.5's UniPC
        declares sigma_max=200 yet use_flow_sigmas=True already normalizes its
        runtime table to [0, 1]. Cache the verdict on the scheduler so steps
        after the first do not pay a GPU->host sync for the max().
        """

        prev_step_index = [s + 1 for s in step_index]
        scheduler.sigmas = scheduler.sigmas.to(device)
        sigma = scheduler.sigmas[step_index].view(*view_shape)
        sigma_prev = scheduler.sigmas[prev_step_index].view(*view_shape)
        # sigmas[1] / sigmas[-1] are loop-invariant schedule endpoints; keep them as
        # 0-d device tensors instead of .item() to drop two per-step host syncs.
        sigma_max = scheduler.sigmas[1]
        sigma_min = scheduler.sigmas[-1]
        edm_domain = getattr(scheduler, "_vrl_edm_sigma_domain", None)
        if edm_domain is None:
            edm_domain = bool(scheduler.sigmas.max().item() > 1.0)
            scheduler._vrl_edm_sigma_domain = edm_domain
        if not edm_domain:
            return cls(sigma, sigma_prev, sigma_max, sigma_min, None, None)
        # t = s/(1+s) maps EDM sigmas onto the flow domain; log_prob /
        # prev_sample_mean / std_dev_t stay there — the constant Jacobian offset
        # cancels in policy ratios and KL, and magnitudes match the flow families.
        one_plus_sigma = 1 + sigma
        one_plus_sigma_prev = 1 + sigma_prev
        return cls(
            sigma=sigma / one_plus_sigma,
            sigma_prev=sigma_prev / one_plus_sigma_prev,
            sigma_max=sigma_max / (1 + sigma_max),
            sigma_min=sigma_min / (1 + sigma_min),
            one_plus_sigma=one_plus_sigma,
            one_plus_sigma_prev=one_plus_sigma_prev,
        )

    @property
    def dt(self) -> Any:
        return self.sigma_prev - self.sigma

    def flow_grpo_std_dev_t(self, noise_level: float) -> Any:
        """Flow-GRPO's SDE noise scale: ``sigma_min + (sigma_max - sigma_min) * sigma``
        at ``noise_level == 1`` (the reference parameterization), its sqrt form otherwise."""

        import torch

        sigma = self.sigma
        if noise_level == 1.0:
            return self.sigma_min + (self.sigma_max - self.sigma_min) * sigma
        return (
            torch.sqrt(sigma / (1 - torch.where(sigma == 1, self.sigma_max, sigma))) * noise_level
        )


def _resolve_step_index(scheduler: Any, timestep: Any, step_index: Any) -> list[int]:
    if step_index is None:
        return [scheduler.index_for_timestep(t) for t in timestep]
    if isinstance(step_index, int):
        return [step_index] * len(timestep)
    return list(step_index)


def flow_sde_scale_terms(
    scheduler: Any,
    timestep: Any,
    *,
    noise_level: float = 1.0,
    sde_type: str = "flow_grpo",
    step_index: Any = None,
) -> tuple[Any, Any, Any]:
    """``(sigma, sqrt(-dt), std_dev_t)`` per timestep, exactly as the SDE step sees them.

    The same schedule reads and noise-scale formula ``sde_step_with_logprob``
    applies, without a model output: an objective that weights samples by the
    transition's scale (Flash-GRPO's rectification) can evaluate the whole
    update's weights from the recorded timesteps before any replay forward.
    """

    import torch

    if sde_type != "flow_grpo":
        raise ValueError(
            f"flow SDE scale terms are defined for sde_type='flow_grpo', got {sde_type!r}"
        )
    timestep = torch.as_tensor(timestep).reshape(-1)
    schedule = _FlowSchedule.read(
        scheduler,
        _resolve_step_index(scheduler, timestep, step_index),
        device=timestep.device,
        view_shape=(-1,),
    )
    return schedule.sigma, torch.sqrt(-1 * schedule.dt), schedule.flow_grpo_std_dev_t(noise_level)


def sde_step_with_logprob(
    scheduler: Any,
    model_output: Any,
    timestep: Any,
    sample: Any,
    *,
    prev_sample: Any | None = None,
    generator: Any | None = None,
    deterministic: bool = False,
    return_dt: bool = False,
    noise_level: float = 1.0,
    sde_type: str = "flow_grpo",
    math_dtype: Any = None,
    step_index: Any = None,
) -> SDEStepResult:
    """Compute one flow-matching SDE step and its per-sample log-probability.

    ``math_dtype`` is the dtype the SDE/log-prob math runs in (the ``logprob``
    precision axis). Defaults to fp32 — the numerically-protected default — so a
    bf16 ``model_output`` is upcast before the Gaussian log-density.

    ``step_index`` lets a sequential denoise caller pass the loop counter it
    already holds. When omitted, the scheduler is searched for each timestep,
    which reads a GPU tensor back to host (a per-step ``cudaStreamSynchronize``
    stall). Since ``state.timesteps`` is ``scheduler.timesteps``, the rollout
    loop index equals the scheduler index, so passing it is exact and sync-free.

    Sigma-domain contract: schedulers whose runtime sigma table exceeds 1
    (EDM domain, e.g. Cosmos Predict2) are converted to the rectified-flow
    [0, 1] domain internally. ``prev_sample`` is always returned in the
    caller's (scheduler's) domain so the denoise loop and trajectory buffers
    stay consistent; ``log_prob`` / ``prev_sample_mean`` / ``std_dev_t`` /
    ``sqrt_neg_dt`` are flow-domain quantities on every path, which keeps
    ratios, parity checks, and KL comparable across model families.
    """
    if sde_type == "ddim":
        # DDPM-family checkpoints (PixArt-Sigma, CogVideoX) live on an
        # alphas_cumprod ladder, not rectified-flow sigmas; forward to the
        # DDIM eta-SDE so every call site keeps this one entry point.
        from vrl.math.denoise.ddim import ddim_step_with_logprob

        return ddim_step_with_logprob(
            scheduler,
            model_output,
            timestep,
            sample,
            prev_sample=prev_sample,
            generator=generator,
            deterministic=deterministic,
            return_dt=return_dt,
            noise_level=noise_level,
            math_dtype=math_dtype,
            step_index=step_index,
        )

    import torch
    from diffusers.utils.torch_utils import randn_tensor

    computation_dtype = torch.float32 if math_dtype is None else math_dtype
    model_output = model_output.to(computation_dtype)
    sample = sample.to(computation_dtype)
    if prev_sample is not None:
        prev_sample = prev_sample.to(computation_dtype)

    step_index = _resolve_step_index(scheduler, timestep, step_index)
    ndim = sample.ndim
    schedule = _FlowSchedule.read(
        scheduler,
        step_index,
        device=sample.device,
        view_shape=(-1,) + (1,) * (ndim - 1),
    )
    sigma, sigma_prev = schedule.sigma, schedule.sigma_prev
    dt = schedule.dt
    one_plus_sigma_prev = schedule.one_plus_sigma_prev
    if schedule.one_plus_sigma is not None:
        # EDM-domain inputs: the model_output is the noise estimate
        # n = (x - x0)/s (see Cosmos Predict2 runner.finalize_noise_pred), so the
        # flow velocity (noise - x0) is n*(1+s) - x; x itself rescales by 1+s.
        # prev_sample converts back to the EDM domain before returning.
        model_output = model_output * schedule.one_plus_sigma - sample
        sample = sample / schedule.one_plus_sigma
        if prev_sample is not None:
            prev_sample = prev_sample / one_plus_sigma_prev

    if prev_sample is not None and generator is not None:
        raise ValueError("Cannot pass both generator and prev_sample.")

    if sde_type == "cps":
        std_dev_t = sigma_prev * math.sin(noise_level * math.pi / 2)
        pred_original_sample = sample - sigma * model_output
        noise_estimate = sample + model_output * (1 - sigma)
        prev_sample_mean = pred_original_sample * (1 - sigma_prev) + noise_estimate * torch.sqrt(
            sigma_prev**2 - std_dev_t**2
        )

        if prev_sample is None:
            variance_noise = randn_tensor(
                model_output.shape,
                generator=generator,
                device=model_output.device,
                dtype=model_output.dtype,
            )
            prev_sample = prev_sample_mean + std_dev_t * variance_noise

        if deterministic:
            prev_sample = sample + dt * model_output

        log_prob = -((prev_sample.detach() - prev_sample_mean) ** 2)
        log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))
    else:
        std_dev_t = schedule.flow_grpo_std_dev_t(noise_level)

        prev_sample_mean = (
            sample * (1 + std_dev_t**2 / (2 * sigma) * dt)
            + model_output * (1 + std_dev_t**2 * (1 - sigma) / (2 * sigma)) * dt
        )

        if prev_sample is None:
            variance_noise = randn_tensor(
                model_output.shape,
                generator=generator,
                device=model_output.device,
                dtype=model_output.dtype,
            )
            prev_sample = prev_sample_mean + std_dev_t * torch.sqrt(-1 * dt) * variance_noise

        if deterministic:
            prev_sample = sample + dt * model_output

        noise_scale = std_dev_t * torch.sqrt(-1 * dt)
        log_prob = (
            -((prev_sample.detach() - prev_sample_mean) ** 2) / (2 * noise_scale**2)
            - torch.log(noise_scale)
            - 0.5 * math.log(2 * math.pi)
        )
        log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))

    if one_plus_sigma_prev is not None:
        prev_sample = prev_sample * one_plus_sigma_prev

    sqrt_neg_dt = torch.sqrt(-1 * dt) if return_dt else None
    return SDEStepResult(
        prev_sample=prev_sample,
        log_prob=log_prob,
        prev_sample_mean=prev_sample_mean,
        std_dev_t=std_dev_t,
        sqrt_neg_dt=sqrt_neg_dt,
        sigma=sigma,
    )


def compute_kl_divergence(
    prev_sample_mean: Any,
    prev_sample_mean_ref: Any,
    std_dev_t: Any,
    sqrt_neg_dt: Any | None = None,
) -> Any:
    """KL divergence between current and reference model in latent space."""
    denom = 2 * std_dev_t**2
    if sqrt_neg_dt is not None:
        denom = 2 * (std_dev_t * sqrt_neg_dt) ** 2
    return ((prev_sample_mean - prev_sample_mean_ref) ** 2).mean(
        dim=tuple(range(1, prev_sample_mean.ndim))
    ) / denom.squeeze()


def diffusion_pretraining_pair(
    scheduler: Any,
    latents: Any,
    noise: Any,
    timesteps: Any,
) -> tuple[Any, Any]:
    """(noisy_input, prediction_target) of the family's pretraining loss.

    The forward process and the target parametrization are OWNED BY THE
    SCHEDULER — the same object rollout/replay sampled with — so the sigma
    domain is whatever the family trained in (the EDM-domain trap that broke
    predict2 parity cannot re-enter through a hand-rolled interpolation):

    - flow matching (``scale_noise``): x_t = (1-sigma)x0 + sigma*eps in the
      scheduler's own sigma table; the model predicts the velocity
      ``eps - x0``.
    - UniPC flow prediction (``add_noise`` with ``use_flow_sigmas``): the
      scheduler's shifted sigma ladder still constructs
      ``x_t = (1-sigma)x0 + sigma*eps``; the target is ``eps - x0``.
    - epsilon / v-prediction (``add_noise`` ladder): x_t from alphas_cumprod;
      the target is ``eps`` or ``get_velocity`` per
      ``scheduler.config.prediction_type``.

    Used by the online GRPO diffusion-loss regularizer (the paper's
    anti-reward-hacking term); the offline DPO trainer keeps its own
    equivalent construction.
    """

    if hasattr(scheduler, "scale_noise"):
        return scheduler.scale_noise(latents, timesteps, noise), noise - latents
    prediction_type = str(getattr(scheduler.config, "prediction_type", "epsilon"))
    noisy = scheduler.add_noise(latents, noise, timesteps)
    if prediction_type == "flow_prediction":
        if not bool(getattr(scheduler.config, "use_flow_sigmas", False)):
            raise ValueError(
                "flow_prediction pretraining requires scheduler.config.use_flow_sigmas=true",
            )
        return noisy, noise - latents
    if prediction_type == "epsilon":
        return noisy, noise
    if prediction_type == "v_prediction":
        return noisy, scheduler.get_velocity(latents, noise, timesteps)
    raise ValueError(
        f"unsupported scheduler prediction_type={prediction_type!r} for the "
        "diffusion pretraining pair (expected a flow-matching scheduler with "
        "scale_noise, UniPC flow_prediction/use_flow_sigmas, or an add_noise "
        "scheduler with epsilon/v_prediction)",
    )


__all__ = [
    "SDEStepResult",
    "compute_kl_divergence",
    "diffusion_pretraining_pair",
    "flow_sde_scale_terms",
    "sde_step_with_logprob",
]
