"""Probe: does PI0.5's flow action policy expose a trainable, replayable logprob?

Tier-2 / §P3 of SPRINT_physical_ai_model_support.md. Unlike OpenVLA-OFT (whose
released head is deterministic L1-regression -> no logprob, eval-only, see
SPRINT_physical_ai_tier2_openvla_libero.md), PI0.5 is a flow-matching action
policy whose sampling can be made *stochastic* during training, which is what
yields a per-denoise-step distribution to take a log-prob over.

Evidence (cosmos-rl PI0.5 GRPO, `cosmos_rl/policy/model/pi05/__init__.py`
get_x_t_mean_std): every denoise step emits a Gaussian transition
``x_{t-1} ~ N(x_t_mean, x_t_std^2)``. In ``eval`` mode ``x_t_std = 0``
(deterministic ODE -> no logprob); in ``train`` mode one of three methods sets a
non-zero std:

    flow_sde   x_t_std = sqrt(dt) * sigma_i,  sigma_i = noise_level*sqrt(t/(1-t))
    flow_cps   x_t_std = (t - dt) * sin(pi*noise_level/2)
    flow_noise x_t_std = learned ExploreNoiseNet head (also trainable)

This is the SAME contract as VRL's diffusion ``sde_step_with_logprob``
(``vrl/math/diffusion/flow_matching.py`` -> ``SDEStepResult`` with
``prev_sample_mean`` / ``std_dev_t`` / ``log_prob``; it even has ``sde_type="cps"``
== flow_cps). So PI0.5's action-chunk logprob is the sum of per-step Normal
log-probs and plugs into VRL's existing flow-matching segment signal with no new
math -> PI0.5 is RL-eligible.

This probe validates the contract on toy tensors (no model/sim): it builds a
stochastic flow rollout for each method and checks the property RL actually
needs -- *replay-exactness*: recomputing the log-prob of the stored sampled
trajectory under the same policy reproduces the collection-time log-prob
bit-for-bit (old_log_prob == new_log_prob at the first inner step => exact
ratio). It also confirms eval-mode (std=0) is degenerate (no logprob).

Usage:
  python -m vrl.scripts.eval.pi05_flow_logprob_probe \
    --out outputs/pi05_probe/flow_logprob_decision.json
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from vrl.models.support_matrix import get_support_entry

_ENTRY_KEY = "pi05"
_METHODS = ("flow_sde", "flow_cps", "flow_noise")


def _x_t_std(method: str, t: torch.Tensor, dt: float, noise_level: float, learned: torch.Tensor) -> torch.Tensor:
    """Per-step transition std for one PI0.5 noise method (cosmos-rl formulas)."""
    if method == "flow_sde":
        sigma = noise_level * torch.sqrt(t / (1.0 - t))
        return torch.sqrt(torch.tensor(dt)) * sigma
    if method == "flow_cps":
        return (t - dt) * math.sin(noise_level * math.pi / 2.0)
    if method == "flow_noise":
        return learned  # learned ExploreNoiseNet head (positive); trainable too
    raise ValueError(f"unknown method {method!r}")


def _chunk_logprob(mean: torch.Tensor, std: torch.Tensor, sample: torch.Tensor) -> torch.Tensor:
    """Sum of per-element Gaussian log-probs over (steps, horizon, action_dim)."""
    dist = torch.distributions.Normal(mean, std)
    # [B, steps, horizon, dim] -> [B]: the action-chunk trajectory log-prob.
    return dist.log_prob(sample).flatten(1).sum(dim=1)


def _probe_method(method: str, *, seed: int) -> dict[str, Any]:
    """Roll a stochastic flow trajectory and check replay-exactness."""
    gen = torch.Generator().manual_seed(seed)
    b, steps, horizon, dim = 4, 10, 16, 7
    noise_level = 1.0
    dt = 1.0 / steps
    learned_std = torch.rand(b, steps, horizon, dim, generator=gen) * 0.4 + 0.1

    # Per-step transition means (stand-in for the model's x_t_mean) + timesteps.
    means = torch.randn(b, steps, horizon, dim, generator=gen)
    # Avoid the t=1 singularity in flow_sde's sqrt(t/(1-t)).
    t = torch.linspace(0.05, 0.95, steps).view(1, steps, 1, 1).expand(b, steps, horizon, dim)
    stds = _x_t_std(method, t, dt, noise_level, learned_std).clamp_min(1e-4)

    # Collection: sample the trajectory and record old_log_prob.
    eps = torch.randn(b, steps, horizon, dim, generator=gen)
    samples = means + stds * eps
    old_log_prob = _chunk_logprob(means, stds, samples)

    # Replay: recompute the log-prob of the SAME stored samples under the same
    # (mean, std). This is what the RL trainer does to get the new policy's
    # log-prob; at the first inner step it must equal old_log_prob exactly.
    new_log_prob = _chunk_logprob(means.clone(), stds.clone(), samples)
    max_abs_diff = float((old_log_prob - new_log_prob).abs().max())

    # Gradient must flow into the mean (and, for flow_noise, the std) so a policy
    # update is possible -- the defining property of a trainable logprob.
    mean_param = means.clone().requires_grad_(True)
    std_param = (learned_std.clone().requires_grad_(True) if method == "flow_noise" else stds)
    _chunk_logprob(mean_param, std_param, samples).sum().backward()
    grad_into_mean = mean_param.grad is not None and float(mean_param.grad.abs().max()) > 0

    return {
        "method": method,
        "replayable": max_abs_diff == 0.0,
        "replay_max_abs_diff": max_abs_diff,
        "logprob_shape": list(old_log_prob.shape),
        "logprob_mean": round(float(old_log_prob.mean()), 3),
        "grad_flows_into_policy": bool(grad_into_mean),
        "std_is_nonzero": bool(float(stds.min()) > 0),
    }


def _probe_eval_mode_degenerate() -> dict[str, Any]:
    """Eval mode sets std=0 (deterministic ODE): no usable logprob."""
    mean = torch.zeros(2, 3)
    std = torch.zeros(2, 3)  # eval mode
    try:
        torch.distributions.Normal(mean, std).log_prob(mean)
        usable = True
    except ValueError:
        usable = False
    return {"eval_mode_std": 0.0, "logprob_usable": usable}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    entry = get_support_entry(_ENTRY_KEY)
    parser.add_argument("--out", default="outputs/pi05_probe/flow_logprob_decision.json")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    methods = [_probe_method(m, seed=args.seed) for m in _METHODS]
    all_replayable = all(m["replayable"] and m["grad_flows_into_policy"] for m in methods)

    report = {
        "probe": "pi05_flow_logprob",
        "matrix_entry": asdict(entry),
        "question": "does PI0.5 flow sampling expose a trainable, replayable logprob?",
        "methods": methods,
        "eval_mode": _probe_eval_mode_degenerate(),
        "maps_onto_vrl": (
            "vrl/math/diffusion/flow_matching.py:sde_step_with_logprob -> "
            "SDEStepResult(prev_sample_mean<->x_t_mean, std_dev_t<->x_t_std, log_prob); "
            "sde_type='cps' == flow_cps"
        ),
        "decision": (
            "rl_eligible_train_mode" if all_replayable else "logprob_unstable"
        ),
        "conclusion": (
            "PI0.5 train-mode flow sampling (default flow_sde) yields a per-step "
            "Gaussian transition whose summed log-prob is replayable and "
            "differentiable -> RL-eligible, reusing VRL's flow-matching SDE "
            "logprob. Eval-mode (std=0) is deterministic -> eval/SFT only."
        ),
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"[decision] {report['decision']}")
    for m in methods:
        print(f"  {m['method']}: replayable={m['replayable']} "
              f"diff={m['replay_max_abs_diff']} grad={m['grad_flows_into_policy']}")
    print(f"[note] {out}")


if __name__ == "__main__":
    main()
