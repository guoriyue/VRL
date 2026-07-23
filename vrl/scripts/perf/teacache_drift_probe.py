"""TeaCache rollout-vs-replay logprob drift probe (real cosmos denoise, on GPU).

WHY (gating): TeaCache skips the transformer forward on low-change denoise steps
and reuses a cached ``noise_pred``. In RL that makes the rollout (collection-time)
logprob approximate, while the trainer's replay forward is exact — the SAME
rollout-vs-replay drift fp8 introduces. fp8's drift is a property of the GEMM, so
its probe uses a synthetic head; TeaCache's drift depends on how fast the real
``noise_pred`` changes between steps and on the diverging skipped trajectory, so
it MUST be measured on the real model along the real teacache-generated path.

WHAT (two passes per threshold, through the actual code paths):
  Pass 1 (rollout WITH teacache at threshold T): generate the trajectory with the
    real ``TeaCacheState`` skip machine; record per step the input latents, the
    sampled action (``prev_sample``), and the rollout logprob (under the cached or
    real ``noise_pred`` actually used).
  Pass 2 (replay, exact): at each recorded input latent run the FULL forward
    (no skip) and score the SAME action's logprob under the exact ``noise_pred``.
  Drift = ``compute_logprob_mismatch_stats(fresh=replay, old=rollout)`` — ratio
    abs-dev (mean/max) + mismatch KL, the exact stats the drift guard / TIS read.

A ``threshold=None`` baseline (no skips) must read ~0 drift — it validates the
two-pass logprob math (rollout == replay when nothing is cached).

GO/NO-GO: a threshold is safe if its ratio drift sits well under the advantage
signal magnitude (O(1)) — the same bar fp8 passed (cosmos fp8 ~0.3% mean / 1% max).

Usage:
    HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 python -m vrl.scripts.perf.teacache_drift_probe \
        --config experiment/cosmos_predict2_5/online_nft_kling_video_reward
"""

from __future__ import annotations

import argparse

import torch

from vrl.algorithms.logprob_mismatch import compute_logprob_mismatch_stats
from vrl.config.loading import load_config
from vrl.config.precision import resolve_precision_policy
from vrl.config.schema import parse_config
from vrl.generation.steps.denoise.teacache import TeaCacheConfig, TeaCacheState
from vrl.math.denoise.flow_matching import sde_step_with_logprob
from vrl.scripts.perf.common.diffusion_runtime import (
    build_model,
    prepare_sampling_state,
)

_SDE_TYPE = "cps"


def _measure(model, cfg, device, dtype, threshold):
    """Return (rollout_logp, replay_logp, skip_ratio) for one threshold (None=off)."""

    num_steps = int(cfg.sampling.num_steps)
    state = prepare_sampling_state(model, cfg)
    teacache = (
        TeaCacheState(TeaCacheConfig.from_sampling({"threshold": threshold}), num_steps)
        if threshold is not None
        else None
    )
    gen = torch.Generator(device=device).manual_seed(0)

    latents_in: list[torch.Tensor] = []
    actions: list[torch.Tensor] = []
    rollout_logp: list[torch.Tensor] = []

    # -- Pass 1: rollout WITH teacache, generating the (diverging) trajectory ----
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
        for step_idx in range(num_steps):
            cur = state.latents.clone()
            timestep = state.timesteps[step_idx]
            if teacache is not None and not teacache.should_run(state.latents, step_idx):
                noise_pred = teacache.cached_noise_pred
            else:
                noise_pred = model.forward_step(state, step_idx)["noise_pred"]
                if teacache is not None:
                    teacache.cache_noise_pred(noise_pred)
            sde = sde_step_with_logprob(
                state.scheduler,
                noise_pred.float(),
                timestep.unsqueeze(0),
                state.latents.float(),
                generator=gen,
                deterministic=False,
                sde_type=_SDE_TYPE,
                step_index=step_idx,
            )
            latents_in.append(cur)
            actions.append(sde.prev_sample.detach())
            rollout_logp.append(sde.log_prob.detach())
            state.latents = sde.prev_sample

        # -- Pass 2: exact replay — full forward at each recorded latent, score the
        #    same action under the exact noise_pred -----------------------------
        replay_logp: list[torch.Tensor] = []
        for step_idx in range(num_steps):
            state.latents = latents_in[step_idx].clone()
            timestep = state.timesteps[step_idx]
            noise_pred = model.forward_step(state, step_idx)["noise_pred"]
            sde = sde_step_with_logprob(
                state.scheduler,
                noise_pred.float(),
                timestep.unsqueeze(0),
                latents_in[step_idx].float(),
                prev_sample=actions[step_idx].float(),
                deterministic=False,
                sde_type=_SDE_TYPE,
                step_index=step_idx,
            )
            replay_logp.append(sde.log_prob.detach())

    skip_ratio = teacache.skip_ratio if teacache is not None else 0.0
    return (
        torch.cat([t.reshape(-1) for t in rollout_logp]),
        torch.cat([t.reshape(-1) for t in replay_logp]),
        skip_ratio,
    )


def _rel(cur: torch.Tensor, prev: torch.Tensor) -> float:
    denom = prev.abs().sum()
    if float(denom) <= 0.0:
        return float("inf")
    return float((cur - prev).abs().sum().div(denom).item())


def _diagnose(model, cfg, device, dtype):
    """Per-step exact-denoise change profile = the structural TeaCache ceiling.

    Reports rel-L1 between consecutive EXACT noise_preds. A step whose noise_pred
    barely moved vs the previous step is reusable (skippable) — independent of any
    skip signal — so the fraction of small-change steps is the best TeaCache could
    ever do here. If that fraction is tiny, the 5% wall is STRUCTURAL (cosmos's
    short 20-step EDM schedule has little redundancy), not a signal problem, and a
    better signal (P0.2) cannot help.
    """

    num_steps = int(cfg.sampling.num_steps)
    state = prepare_sampling_state(model, cfg)
    gen = torch.Generator(device=device).manual_seed(0)
    preds: list[torch.Tensor] = []
    lat_rel: list[float] = []
    prev_lat = None
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
        for step_idx in range(num_steps):
            if prev_lat is not None:
                lat_rel.append(_rel(state.latents, prev_lat))
            else:
                lat_rel.append(float("nan"))
            prev_lat = state.latents.clone()
            timestep = state.timesteps[step_idx]
            np_ = model.forward_step(state, step_idx)["noise_pred"].float()
            preds.append(np_.clone())
            sde = sde_step_with_logprob(
                state.scheduler,
                np_,
                timestep.unsqueeze(0),
                state.latents.float(),
                generator=gen,
                deterministic=False,
                sde_type=_SDE_TYPE,
                step_index=step_idx,
            )
            state.latents = sde.prev_sample
    np_rel = [float("nan")] + [_rel(preds[t], preds[t - 1]) for t in range(1, num_steps)]

    print(f"\n{'step':>4} | {'latent relL1':>12} | {'noise_pred relL1 vs prev':>24}")
    print("-" * 50)
    for t in range(num_steps):
        print(f"{t:>4} | {lat_rel[t]:12.4f} | {np_rel[t]:24.4f}")
    print("\nideal skip ceiling = steps whose noise_pred barely moved vs prev:")
    valid = [v for v in np_rel if v == v]  # drop nan
    for eps in (0.01, 0.02, 0.05, 0.10):
        n = sum(1 for v in valid if v < eps)
        print(
            f"  noise_pred relL1 < {eps:.0%}: {n}/{num_steps} steps "
            f"({n / num_steps:.0%} skippable)"
        )


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument(
        "--thresholds",
        default="0.1,0.15,0.25",
        help="comma-separated teacache thresholds to probe",
    )
    p.add_argument(
        "--diagnose",
        action="store_true",
        help="report per-step exact noise_pred change = the structural "
        "skip ceiling (is the 5% wall a signal or a redundancy problem?)",
    )
    args = p.parse_args(argv)

    cfg = load_config(args.config)
    root = parse_config(cfg)
    precision = resolve_precision_policy(root)
    device = torch.device(args.device)
    dtype = torch.bfloat16
    model = build_model(root, device, dtype, precision=precision)

    if args.diagnose:
        _diagnose(model, cfg, device, dtype)
        return

    thresholds = [None] + [float(x) for x in args.thresholds.split(",") if x.strip()]
    print(
        f"\n{'config':>22} | {'skip%':>6} | {'ratio_dev_mean':>14} | "
        f"{'ratio_dev_max':>13} | {'mismatch_kl':>11}"
    )
    print("-" * 80)
    for thr in thresholds:
        rollout_lp, replay_lp, skip = _measure(model, cfg, device, dtype, thr)
        stats = compute_logprob_mismatch_stats(fresh_log_prob=replay_lp, old_log_prob=rollout_lp)
        name = "baseline(no teacache)" if thr is None else f"teacache(thr={thr})"
        print(
            f"{name:>22} | {skip * 100:5.0f}% | {stats.ratio_abs_dev_mean:14.4%} | "
            f"{stats.ratio_abs_dev_max:13.4%} | {stats.mismatch_kl:11.5f}",
            flush=True,
        )
    print(
        "\nGO/NO-GO: ratio_dev should sit well under the O(1) advantage signal "
        "(fp8 passed at ~0.3% mean / 1% max)."
    )


if __name__ == "__main__":
    main()
