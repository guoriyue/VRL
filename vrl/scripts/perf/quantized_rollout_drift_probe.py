"""Quantized-rollout precision-drift validation on real tensor cores.

WHY: SPRINT_fullparam_and_fp8_precision.md P3 asks whether a quantized rollout against
a bf16 replay produces enough logprob drift to bias GRPO, and whether the
precision machinery (drift guard + truncated importance sampling) measures and
bounds it. The hardware gate the sprint assumed (no FP8 silicon) is void on this
box -- an RTX 5090 (Blackwell, sm_120) executes ``torch._scaled_mm`` in fp8 -- so
this turns the open question into a measured number end to end.

WHAT it does, all on the GPU, through the ACTUAL codebase code paths:
  1. Build a synthetic policy head (activations @ weight) and compute per-sample
     logprobs two ways: bf16 (the replay/training forward) and the production
     FP8/NVFP4 ``torch._scaled_mm`` module. The quantized logprob is the
     behavior ``old_log_prob``; the bf16 logprob is the fresh replay ``log_prob``.
  2. Measure the drift with ``LogprobMismatchStats`` (the one shared
     stats helper used by metrics + guard) -- abs diff, ratio dev, mismatch KL.
  3. Run the exact catastrophic guard and correction defaults produced by
     ``build_precision_split_safety_configs`` for a rollout/train precision split.
  4. Feed a batch of independent denoise-step signals into the real
     continuous-GRPO loss and compare the policy-gradient norm with ``tis_mode``
     off vs truncate vs mask. This matches the trainer: it evaluates and
     backpropagates one timestep at a time, then averages across timesteps.
  5. Also print the product of all step ratios as a counterfactual stress metric.
     The trainer does not optimize that product, so it is never used as a gate.

FAITHFULNESS: both paths use the production ``Fp8Linear`` / ``Fp4Linear``, not a
cast round-trip, so the measured drift includes the real GEMM and activation
quantization. The drift -> guard -> TIS chain calls the exact functions training
uses and preserves their per-timestep loss semantics. Only the model is synthetic
(a single GEMM head stands in for the policy logit projection), so this is a
kernel/correction-path stress test rather than real-model SDE-logprob accuracy.

Usage:  python -m vrl.scripts.perf.quantized_rollout_drift_probe --scheme fp8|nvfp4
"""

from __future__ import annotations

import argparse
from typing import Any

import torch
from torch import nn

from vrl.algorithms.grpo.continuous import GRPO, GRPOConfig
from vrl.algorithms.logprob_mismatch import (
    LogprobMismatchStats,
    PrecisionCorrectionConfig,
)
from vrl.algorithms.trajectory import AlgorithmInput
from vrl.config.builders import build_precision_split_safety_configs
from vrl.nn.quantization import Fp4Linear, Fp8Linear, nvfp4_available
from vrl.rollouts.evaluators.types import SegmentSignal, TrajectorySignalBatch
from vrl.scripts.perf.common.baseline import (
    DEFAULT_BASELINE_PATH,
    BaselineRecord,
    append_baseline,
)
from vrl.trainers.online.precision_guard import (
    PrecisionDriftError,
    run_precision_drift_guard,
)


def _logprob_from_logits(logits: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    """Per-sample log-prob of `actions` under a categorical head (fp32 reduction)."""
    logp = torch.log_softmax(logits.float(), dim=-1)
    return logp.gather(-1, actions.unsqueeze(-1)).squeeze(-1)


def _signals(log_prob: torch.Tensor, old_log_prob: torch.Tensor) -> TrajectorySignalBatch:
    return TrajectorySignalBatch(
        segments={
            "denoise": SegmentSignal(
                name="denoise",
                distribution="flow_matching",
                log_prob=log_prob,
                old_log_prob=old_log_prob,
                mask=torch.ones_like(log_prob),
            ),
        },
        group_ids=torch.arange(log_prob.shape[0], device=log_prob.device),
        primary_segment="denoise",
    )


def _step_logprob(logits: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    """Categorical logprob for each independent denoise step -> [samples, steps]."""
    logp = torch.log_softmax(logits.float(), dim=-1)  # [samples, T, vocab]
    return logp.gather(-1, actions.unsqueeze(-1)).squeeze(-1)


def _policy_grad_norm(
    *,
    weight: torch.Tensor,
    activations: torch.Tensor,
    actions: torch.Tensor,
    old_log_prob: torch.Tensor,
    advantages: torch.Tensor,
    tis_mode: str = "off",
    cap: float = 2.0,
    rs_mode: str = "off",
) -> tuple[float, float, float]:
    """Backprop the GRPO loss into `weight`.

    Returns ``(grad_norm, tis_clip_fraction, rs_seq_masked_fraction)``.
    ``activations`` is [samples, T, hidden]. The real trainer computes one
    per-sample loss for every timestep and divides the accumulated gradient by T;
    flattening [samples, T] and repeating each sample advantage is algebraically
    identical. ``old_log_prob`` therefore has shape [samples, T], not a product
    or sum over the trajectory.
    """
    w = weight.detach().clone().requires_grad_(True)
    fresh_logits = activations @ w.t()  # [samples, T, vocab] bf16 replay forward
    fresh_logprob = _step_logprob(fresh_logits, actions)
    flat_fresh = fresh_logprob.reshape(-1)
    flat_old = old_log_prob.reshape(-1)
    flat_advantages = advantages[:, None].expand_as(fresh_logprob).reshape(-1)
    grpo = GRPO(GRPOConfig(kl_coef=0.0))
    grpo.precision_correction = PrecisionCorrectionConfig(
        tis_mode=tis_mode,
        tis_imp_weight_cap=cap,
        rs_mode=rs_mode,
    )
    loss, metrics = grpo.compute_loss(
        AlgorithmInput(
            signals=_signals(flat_fresh, flat_old),
            advantages=flat_advantages,
        ),
    )
    loss.backward()
    return (
        float(w.grad.norm()),
        metrics.update.tis_clip_fraction,
        metrics.update.rs_seq_masked_fraction,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scheme", choices=("fp8", "nvfp4"), default="fp8")
    parser.add_argument(
        "--record-baseline",
        nargs="?",
        const=str(DEFAULT_BASELINE_PATH),
        default=None,
        metavar="PATH",
        help=(
            "append this run's drift numbers to the perf baseline record "
            f"(default {DEFAULT_BASELINE_PATH}); see docs/perf/README.md"
        ),
    )
    return parser.parse_args()


def _require_precision_guard(
    guard_config: Any,
    *,
    scheme: str,
    replay_logprob: torch.Tensor,
    rollout_logprob: torch.Tensor,
) -> None:
    """Run the production guard and make a failed gate terminate the probe."""

    def evaluate(_timestep: int) -> TrajectorySignalBatch:
        return _signals(replay_logprob, rollout_logprob)

    print("-- production precision drift guard --")
    try:
        guard_record = run_precision_drift_guard(
            guard_config,
            training_precision="bf16",
            rollout_precision=f"bf16+{scheme}",
            math_precision="fp32",
            timestep_indices=[0],
            evaluate_fn=evaluate,
        )
    except PrecisionDriftError as exc:
        print(f"  FAILED: {str(exc)[:160]}...")
        raise SystemExit(1) from exc
    print(
        f"  PASSED: violated={guard_record['violated']} "
        f"ratio_abs_dev_max={guard_record['worst_stats']['ratio_abs_dev_max']:.4e} "
        f"limit={guard_config.max_ratio_abs_dev:.1f}"
    )


def main() -> None:
    args = _parse_args()
    scheme = args.scheme
    record_baseline = args.record_baseline
    if not torch.cuda.is_available():
        raise SystemExit("this probe needs a CUDA device with quantized GEMM support")
    if scheme == "nvfp4" and not nvfp4_available():
        raise SystemExit(
            "nvfp4 drift probing requires an NVFP4-capable CUDA build and Blackwell-class GPU",
        )
    torch.manual_seed(0)
    device = "cuda"
    dev_name = torch.cuda.get_device_name(0)
    n_samples, hidden, vocab = 512, 2048, 4096

    activations = torch.randn(n_samples, hidden, device=device, dtype=torch.bfloat16)
    weight = torch.randn(vocab, hidden, device=device, dtype=torch.bfloat16) * (hidden**-0.5)

    # The rollout side runs the production module (weight quantized once,
    # activation per call), not a hand copy. Use the FP8 module default because
    # it is also the loader's default production recipe; historical tensorwise
    # numbers are not representative of an unspecified rollout recipe.
    head_lin = nn.Linear(hidden, vocab, bias=False, device=device, dtype=torch.bfloat16)
    with torch.no_grad():
        head_lin.weight.copy_(weight)
    rollout_head = Fp8Linear(head_lin) if scheme == "fp8" else Fp4Linear(head_lin)
    correction_cfg, guard_cfg = build_precision_split_safety_configs()

    # Replay (bf16) and quantized rollout logits for the SAME head.
    logits_replay = (activations @ weight.t()).to(torch.bfloat16)
    logits_rollout = rollout_head(activations)

    # Actions are sampled by the quantized rollout policy; both sides score them.
    actions = torch.distributions.Categorical(logits=logits_rollout.float()).sample()
    replay_logprob = _logprob_from_logits(logits_replay, actions)  # fresh log_prob
    rollout_logprob = _logprob_from_logits(logits_rollout, actions)

    stats = LogprobMismatchStats.compute(replay_logprob, rollout_logprob)
    print(f"== {scheme.upper()} rollout drift probe on {dev_name} ==")
    print(f"recipe={rollout_head.recipe} profile=synthetic_full_head")
    print(f"samples={n_samples} hidden={hidden} vocab={vocab}  (rollout={scheme}, replay=bf16)")
    print("-- measured logprob drift (LogprobMismatchStats) --")
    print(
        f"  abs_diff   mean={stats.logprob_abs_diff_mean:.4e}  max={stats.logprob_abs_diff_max:.4e}"
    )
    print(f"  ratio_dev  mean={stats.ratio_abs_dev_mean:.4e}  max={stats.ratio_abs_dev_max:.4e}")
    print(
        f"  mismatch_kl={stats.mismatch_kl:.4e}  k3_kl={stats.mismatch_k3_kl:.4e}  finite={stats.finite}"
    )

    # -- precision drift guard on the real quantized-vs-bf16 split --
    _require_precision_guard(
        guard_cfg,
        scheme=scheme,
        replay_logprob=replay_logprob,
        rollout_logprob=rollout_logprob,
    )

    # -- Production-equivalent correction across independent denoise steps --
    # OnlineTrainer calls the evaluator and GRPO once per timestep, then scales
    # the accumulated loss by 1/T. It does not multiply the ratios from all
    # denoise steps into one trajectory-level importance weight. Flattening the
    # [sample, timestep] pairs below gives the same mean loss and gradient.
    n_traj, n_steps = 256, 35
    traj_acts = torch.randn(n_traj, n_steps, hidden, device=device, dtype=torch.bfloat16)
    traj_logits_replay = (traj_acts @ weight.t()).to(torch.bfloat16)
    traj_logits_quantized = rollout_head(
        traj_acts.reshape(n_traj * n_steps, hidden),
    ).reshape(n_traj, n_steps, vocab)
    traj_actions = torch.distributions.Categorical(logits=traj_logits_quantized.float()).sample()
    step_replay_logprob = _step_logprob(traj_logits_replay, traj_actions)
    step_rollout_logprob = _step_logprob(traj_logits_quantized, traj_actions)
    step_stats = LogprobMismatchStats.compute(step_replay_logprob, step_rollout_logprob)
    advantages = torch.randn(n_traj, device=device)  # mix of +/- advantages
    cap = correction_cfg.tis_imp_weight_cap
    step_ratio = torch.exp(step_replay_logprob - step_rollout_logprob)
    over_cap = float((step_ratio > cap).float().mean())
    print(f"-- production-equivalent per-timestep drift ({n_steps} independent steps) --")
    print(
        f"  ratio_dev mean={step_stats.ratio_abs_dev_mean:.4e} "
        f"max={step_stats.ratio_abs_dev_max:.4e}  finite={step_stats.finite}"
    )
    print(f"  {over_cap * 100:.1f}% of sample-steps have importance weight > cap {cap}")
    print(f"-- production-equivalent policy-gradient norm (tis cap={cap}) --")
    base_norm = None
    for mode in ("off", "truncate", "mask"):
        norm, clip_frac, _ = _policy_grad_norm(
            weight=weight,
            activations=traj_acts,
            actions=traj_actions,
            old_log_prob=step_rollout_logprob,
            advantages=advantages,
            tis_mode=mode,
            cap=cap,
        )
        if base_norm is None:
            base_norm = norm
        ratio = norm / base_norm if base_norm else float("nan")
        print(
            f"  tis_mode={mode:8s}  grad_norm={norm:.4e}  (x{ratio:.3f} vs off)  "
            f"tis_clip_fraction={clip_frac:.3f}"
        )

    # Continuous diffusion presents one scalar per sample at each evaluator
    # call, so the current RS implementation also judges each sample-step. The
    # seq_mean name is shared with token-sequence algorithms; no denoise axis is
    # present in the production call.
    band_lo = correction_cfg.rs_log_ratio_low
    band_hi = correction_cfg.rs_log_ratio_high
    print(
        "-- per-timestep reject-sampling "
        f"(seq_mean_k1, band [ln0.5, ln2.0]=[{band_lo:.3f}, {band_hi:.3f}]) --"
    )
    rs_norm, _, rs_frac = _policy_grad_norm(
        weight=weight,
        activations=traj_acts,
        actions=traj_actions,
        old_log_prob=step_rollout_logprob,
        advantages=advantages,
        rs_mode=correction_cfg.rs_mode,
    )
    ratio = rs_norm / base_norm if base_norm else float("nan")
    print(
        f"  rs_mode={correction_cfg.rs_mode}  grad_norm={rs_norm:.4e}  "
        f"(x{ratio:.3f} vs off)  "
        f"rs_seq_masked_fraction={rs_frac:.3f}  "
        f"({'OK <5%' if rs_frac < 0.05 else 'HIGH >=5% — tighten band / recompute'})"
    )
    production_norm, production_clip, production_rs = _policy_grad_norm(
        weight=weight,
        activations=traj_acts,
        actions=traj_actions,
        old_log_prob=step_rollout_logprob,
        advantages=advantages,
        tis_mode=correction_cfg.tis_mode,
        cap=cap,
        rs_mode=correction_cfg.rs_mode,
    )
    production_ratio = production_norm / base_norm if base_norm else float("nan")
    print(
        f"  production combined: tis={correction_cfg.tis_mode} "
        f"rs={correction_cfg.rs_mode} grad_norm={production_norm:.4e} "
        f"(x{production_ratio:.3f} vs off) tis_clip={production_clip:.3f} "
        f"rs_masked={production_rs:.3f}"
    )

    # Counterfactual diagnostic only: multiplying all denoise-step ratios can
    # reveal how small independent errors compound, but no production loss or
    # guard in this repository consumes this trajectory product.
    trajectory_replay_logprob = step_replay_logprob.sum(dim=-1)
    trajectory_rollout_logprob = step_rollout_logprob.sum(dim=-1)
    trajectory_stats = LogprobMismatchStats.compute(
        trajectory_replay_logprob,
        trajectory_rollout_logprob,
    )
    trajectory_ratio = torch.exp(trajectory_replay_logprob - trajectory_rollout_logprob)
    trajectory_over_cap = float((trajectory_ratio > cap).float().mean())
    print(f"-- counterfactual T={n_steps} ratio product (diagnostic only; not trainer loss) --")
    print(
        f"  ratio_dev mean={trajectory_stats.ratio_abs_dev_mean:.4e} "
        f"max={trajectory_stats.ratio_abs_dev_max:.4e}; "
        f"importance_weight>{cap}: {trajectory_over_cap * 100:.1f}%"
    )

    print("-- verdict --")
    print(f"  {scheme} GEMM runs on this GPU and produces real, measurable logprob drift.")
    print("  The guard, TIS, and RS paths are wired and measured with the trainer's")
    print("  per-timestep semantics. The trajectory product above is stress context only.")
    print("  This synthetic head cannot replace a real-model SDE-logprob calibration run.")

    if record_baseline:
        # Recorded before the gate below so a FAILING run still leaves its
        # evidence behind — the failure is the datapoint worth keeping.
        path = append_baseline(
            BaselineRecord(
                probe="quantized_rollout_drift",
                metrics={
                    "step_ratio_dev_mean": step_stats.ratio_abs_dev_mean,
                    "step_ratio_dev_max": step_stats.ratio_abs_dev_max,
                    "step_over_cap_fraction": over_cap,
                    "trajectory_ratio_dev_mean": trajectory_stats.ratio_abs_dev_mean,
                    "trajectory_ratio_dev_max": trajectory_stats.ratio_abs_dev_max,
                    "trajectory_over_cap_fraction": trajectory_over_cap,
                    "production_grad_norm_ratio": production_ratio,
                    "rs_masked_fraction": production_rs,
                },
                context={
                    "scheme": scheme,
                    "n_samples": n_samples,
                    "hidden": hidden,
                    "vocab": vocab,
                    "n_steps": n_steps,
                    "tis_cap": cap,
                    "profile": "synthetic_full_head",
                },
                notes=(
                    "synthetic head; per-timestep metrics are the trainer semantics, "
                    "trajectory_* is the counterfactual product diagnostic"
                ),
            ),
            path=record_baseline,
        )
        print(f"\nbaseline appended -> {path}")

    # NVFP4's synthetic correction-path gate follows the same sample-step ratio
    # consumed by GRPO. Real-model SDE logprobs remain a separate P2 gate.
    if scheme == "nvfp4" and (not step_stats.finite or over_cap > 0.10):
        print(
            "nvfp4 drift gate failed: "
            f"finite={step_stats.finite}, over_cap={over_cap:.3f} (limit=0.100)",
        )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
