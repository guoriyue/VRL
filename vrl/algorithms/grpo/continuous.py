"""Continuous-action GRPO for diffusion / flow-matching policies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from vrl.algorithms.advantages import group_relative_advantages
from vrl.algorithms.base import Algorithm
from vrl.algorithms.logprob_mismatch import (
    PrecisionCorrectionConfig,
    apply_rejection_sample_mask,
    apply_truncated_importance_weight,
    combine_keep_masks,
    compute_logprob_mismatch_stats,
)
from vrl.algorithms.trajectory import AlgorithmInput
from vrl.algorithms.types import TrainStepMetrics


@dataclass(slots=True)
class GRPOConfig:
    """Hyper-parameters for continuous GRPO."""

    clip_ratio: float = 0.2
    kl_coef: float = 0.0
    eps: float = 1e-4
    adv_clip_max: float = 5.0
    global_std: bool = False
    flow_kl_use_dt: bool = False


class GRPO(Algorithm):
    """Group Relative Policy Optimization for continuous rollout signals.

    Advantages are normalised within each prompt group:
        a_i = (r_i - mean(r)) / max(std(r), eps)

    Loss is the clipped surrogate objective (PPO-style) applied to
    per-sample log-probabilities produced by the evaluator.
    """

    # Signal-branch contract (AlgorithmAdapter.validate_inputs): the clipped
    # surrogate reads these from the evaluator replay. ref_log_prob is
    # conditional on kl_coef>0, so it is NOT a hard requirement here — the
    # KL branch validates it with its own detailed diagnostic.
    required_signal_keys = ("log_prob", "old_log_prob")

    def __init__(self, config: GRPOConfig | None = None) -> None:
        self.config = config or GRPOConfig()
        # Rollout->replay precision correction (TIS). Off by default; the trainer
        # injects trainer.precision_correction here at construction so the knobs
        # live at the trainer level, not in the algorithm's hyperparameters.
        self.precision_correction = PrecisionCorrectionConfig()
        # Diagnostic stash: last call's policy_loss and (kl_coef * kl_loss)
        # tensors, kept alive (not detached) for grad-split diagnostics in
        # the trainer. Set to None when not applicable.
        self._last_policy_loss_tensor: Any = None
        self._last_kl_term_tensor: Any = None

    def compute_advantages_from_tensors(
        self,
        rewards: Any,
        group_ids: Any,
    ) -> Any:
        """Per-group advantage normalization on tensors.

        Groups are identified by ``group_ids``: samples sharing the same
        group_id are normalized together (GRPO per-prompt normalization).
        """
        cfg = self.config
        return group_relative_advantages(
            rewards,
            group_ids,
            eps=cfg.eps,
            adv_clip_max=cfg.adv_clip_max,
            global_std=cfg.global_std,
        )

    def compute_loss(
        self,
        inputs: AlgorithmInput,
    ) -> tuple[Any, TrainStepMetrics]:
        """Clipped surrogate loss from trajectory-native evaluator signals.

        Handles both flow-matching latent-space KL and generic log-prob KL.
        """
        import torch

        from vrl.math.diffusion.flow_matching import compute_kl_divergence

        cfg = self.config
        # Presence of signals + required_signal_keys is enforced upstream by
        # AlgorithmAdapter.validate_inputs (one declarative gate).
        if inputs.advantages is None:
            raise RuntimeError("AlgorithmInput.advantages is required for GRPO")
        signals = inputs.signals.primary
        advantages = inputs.advantages
        old_log_probs = signals.old_log_prob

        raw_ratio = torch.exp(signals.log_prob - old_log_probs)
        # Truncated importance sampling on the rollout->replay weight before the PPO
        # clip, so quantized-rollout (fp8/fp4) drift on a few samples cannot dominate
        # the gradient via the unclipped negative-advantage branch.
        pc = self.precision_correction
        ratio, tis_keep = apply_truncated_importance_weight(raw_ratio, pc)
        # RS rejects whole samples whose rollout->replay log-ratio drift is out of
        # band — orthogonal to TIS (which clamps the per-element weight). Both feed
        # the masked-mean denominator below (true off-policy rejection, not a
        # gradient-magnitude dilution).
        rs_keep = apply_rejection_sample_mask(signals.log_prob - old_log_probs, pc)
        clipped_ratio = torch.clamp(ratio, 1.0 - cfg.clip_ratio, 1.0 + cfg.clip_ratio)
        unclipped_loss = -advantages * ratio
        clipped_loss = -advantages * clipped_ratio
        per_sample_loss = torch.maximum(unclipped_loss, clipped_loss)
        keep = combine_keep_masks(tis_keep, rs_keep)
        if keep is not None:
            policy_loss = (per_sample_loss * keep).sum() / keep.sum().clamp_min(1.0)
        else:
            policy_loss = torch.mean(per_sample_loss)
        if tis_keep is not None:
            tis_clip_fraction = (1.0 - tis_keep.mean()).item()
        else:
            tis_clip_fraction = (
                0.0
                if pc.tis_mode == "off"
                else (ratio != raw_ratio).float().mean().item()
            )
        rs_seq_masked_fraction = (
            0.0 if rs_keep is None else (1.0 - rs_keep.mean()).item()
        )

        if cfg.kl_coef > 0:
            if signals.ref_log_prob is None:
                raise RuntimeError(
                    f"GRPOConfig.kl_coef={cfg.kl_coef} > 0 but "
                    "signals.ref_log_prob is None. Check: (1) ref_model "
                    "passed to OnlineTrainer, (2) SignalRequest(need_ref=True) "
                    "in the evaluator call."
                )
            if (
                signals.distribution == "flow_matching"
                and signals.prev_sample_mean is not None
                and signals.ref_prev_sample_mean is not None
            ):
                kl = compute_kl_divergence(
                    signals.prev_sample_mean,
                    signals.ref_prev_sample_mean,
                    signals.std_dev_t,
                    sqrt_neg_dt=signals.dt if cfg.flow_kl_use_dt else None,
                )
                kl_loss = torch.mean(kl)
            else:
                kl_loss = torch.mean(signals.log_prob - signals.ref_log_prob)
            kl_term = cfg.kl_coef * kl_loss
            loss = policy_loss + kl_term
            self._last_kl_term_tensor = kl_term
        else:
            kl_loss = torch.tensor(0.0, device=signals.log_prob.device)
            loss = policy_loss
            self._last_kl_term_tensor = None

        self._last_policy_loss_tensor = policy_loss

        clip_fraction = torch.mean((torch.abs(ratio - 1.0) > cfg.clip_ratio).float()).item()
        approx_kl = 0.5 * torch.mean((signals.log_prob - old_log_probs) ** 2).item()

        # Rollout-vs-replay logprob drift: with a same-dtype on-policy first step this
        # is ~0; under rollout!=train precision it surfaces the backend mismatch.
        mismatch = compute_logprob_mismatch_stats(signals.log_prob, old_log_probs)

        metrics = TrainStepMetrics(
            loss=loss.item(),
            policy_loss=policy_loss.item(),
            kl_penalty=kl_loss.item(),
            clip_fraction=clip_fraction,
            approx_kl=approx_kl,
            tis_clip_fraction=tis_clip_fraction,
            rs_seq_masked_fraction=rs_seq_masked_fraction,
            logprob_abs_diff_mean=mismatch.logprob_abs_diff_mean,
            logprob_abs_diff_max=mismatch.logprob_abs_diff_max,
            ratio_abs_dev_mean=mismatch.ratio_abs_dev_mean,
            ratio_abs_dev_max=mismatch.ratio_abs_dev_max,
            mismatch_kl=mismatch.mismatch_kl,
            mismatch_k3_kl=mismatch.mismatch_k3_kl,
        )

        return loss, metrics
