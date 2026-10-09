"""Continuous-action GRPO for diffusion / flow-matching policies."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, ClassVar

from vrl.algorithms.advantages import GroupAdvantageConfig, GroupRelativeObjective
from vrl.algorithms.logprob_mismatch import (
    PrecisionCorrectionConfig,
    apply_rejection_sample_mask,
    apply_truncated_importance_weight,
    behavior_log_prob,
    combine_keep_masks,
)
from vrl.algorithms.requirements import AlgorithmRequirements
from vrl.algorithms.trajectory import AlgorithmInput
from vrl.algorithms.types import PolicyUpdateStats, TrainStepMetrics
from vrl.rollouts.evaluators.types import FlowSDESignal


@dataclass(slots=True)
class ImportanceRatioConfig(GroupAdvantageConfig):
    """Settings of every objective whose loss is a ratio against the rollout policy."""

    # Bridged by build_configs from ``trainer.precision_correction`` (the
    # trainer owns the knob: its replay-parity gate reads the same one), never
    # a YAML key here. Bounds rollout->replay precision drift inside the loss.
    precision_correction: PrecisionCorrectionConfig = field(
        default_factory=PrecisionCorrectionConfig, init=False
    )


@dataclass(slots=True)
class ClippedPolicyConfig(ImportanceRatioConfig):
    """Policy-ratio clipping and reference-KL knobs with real loss consumers."""

    clip_ratio: float = 0.2
    kl_coef: float = 0.0


@dataclass(slots=True)
class GRPOConfig(ClippedPolicyConfig):
    """Hyper-parameters for continuous GRPO."""

    requirements: ClassVar[AlgorithmRequirements] = AlgorithmRequirements(
        needs_sde_rollout=True,
        sft_source="latents",
    )

    flow_kl_use_dt: bool = False
    # Diffusion-loss regularizer weight (Cosmos-Predict2.5 paper 4.2.2's
    # anti-reward-hacking term): adds sft_weight * MSE(model_pred,
    # pretraining_target) on CLEAN fine-tuning latents (data.sft_latents,
    # produced by vrl/scripts/denoise/encode_targets.py). The term is
    # computed by the trainer (it needs a model forward, which algorithm
    # losses never do); this knob rides the algorithm config so recipes tune
    # it next to kl_coef.
    sft_weight: float = 0.0


class GRPO(GroupRelativeObjective):
    """Group Relative Policy Optimization for continuous rollout signals.

    Advantages are normalised within each prompt group:
        a_i = (r_i - mean(r)) / (std(r) + eps)

    Loss is the clipped surrogate objective (PPO-style) applied to
    per-sample log-probabilities produced by the evaluator.
    """

    uses_evaluator = True

    config: GRPOConfig

    @property
    def kl_coef(self) -> float:
        return float(self.config.kl_coef)

    @property
    def sft_weight(self) -> float:
        return float(self.config.sft_weight)

    # -- lifecycle entry points the trainer calls on every objective ---------

    def prepare_update(self, update_timesteps: Callable[[], Any]) -> None:
        """Called once per optimizer update, before its first replay forward.

        ``update_timesteps`` returns the recorded timestep of every (sample,
        trained step) the update will put loss on; it is only evaluated by an
        objective that normalizes over the whole update.
        """

        del update_timesteps

    def after_optimizer_step(self, global_step: int) -> None:
        """Called after every applied (not scaler-skipped) optimizer step."""

        del global_step

    def compute_loss(
        self,
        inputs: AlgorithmInput,
    ) -> tuple[Any, TrainStepMetrics]:
        """Clipped surrogate loss from trajectory-native evaluator signals.

        Handles both flow-matching latent-space KL and generic log-prob KL.
        """
        import torch

        from vrl.math.denoise.flow_matching import compute_kl_divergence

        cfg = self.config
        signals = inputs.signals.primary
        advantages = self._broadcast_sample_values(inputs.advantages, signals.log_prob)
        pc = self.config.precision_correction
        old_log_probs = behavior_log_prob(signals.log_prob, signals.old_log_prob, pc)

        raw_ratio = torch.exp(signals.log_prob - old_log_probs)
        # Truncated importance sampling on the rollout->replay weight before the PPO
        # clip, so quantized-rollout (FP8/NVFP4) drift on a few samples cannot dominate
        # the gradient via the unclipped negative-advantage branch.
        ratio, tis_keep = apply_truncated_importance_weight(raw_ratio, pc)
        # RS rejects whole samples whose rollout->replay log-ratio drift is out of
        # band — orthogonal to TIS (which clamps the per-element weight). Both feed
        # the masked-mean denominator below (true off-policy rejection, not a
        # gradient-magnitude dilution).
        rs_keep = apply_rejection_sample_mask(
            signals.log_prob - old_log_probs,
            pc,
            mask=signals.mask,
        )
        clipped_ratio = torch.clamp(ratio, 1.0 - cfg.clip_ratio, 1.0 + cfg.clip_ratio)
        unclipped_loss = -advantages * ratio
        clipped_loss = -advantages * clipped_ratio
        per_sample_loss = torch.maximum(unclipped_loss, clipped_loss)
        # Optional per-sample positive weight (Flash-GRPO's temporal gradient
        # rectification). Applied AFTER the max: for w > 0,
        # max(w*a, w*b) == w*max(a, b), so this is exactly the weighted clipped
        # surrogate without duplicating the loss body in the subclass.
        weight = self._loss_weight(signals)
        if weight is not None:
            per_sample_loss = per_sample_loss * self._broadcast_sample_values(
                weight,
                per_sample_loss,
            )
        # Unlike the historical scalar-per-denoise-step path, grouped causal
        # replay has a real transition mask. Fold it into the same denominator
        # as TIS/RS so deterministic cache-finalization passes never become
        # policy actions. Existing diffusion masks are all ones, preserving
        # their numerical behavior.
        keep = combine_keep_masks(signals.mask.to(ratio.dtype), tis_keep, rs_keep)
        policy_loss = (per_sample_loss * keep).sum() / keep.sum().clamp_min(1.0)
        active_clip_fraction = (
            ((clipped_loss > unclipped_loss).to(keep.dtype) * keep).sum()
            / keep.sum().clamp_min(1.0)
        ).item()
        if tis_keep is not None:
            tis_clip_fraction = (1.0 - tis_keep.mean()).item()
        else:
            tis_clip_fraction = (
                0.0 if pc.tis_mode == "off" else (ratio != raw_ratio).float().mean().item()
            )
        rs_seq_masked_fraction = 0.0 if rs_keep is None else (1.0 - rs_keep.mean()).item()

        if cfg.kl_coef > 0:
            # The reference forward ran (the trainer requests it whenever the KL
            # term is on). A flow-matching SDE replay carries both proposal
            # means, so the KL is the closed form in latent space.
            if isinstance(signals, FlowSDESignal):
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
        else:
            kl_loss = torch.tensor(0.0, device=signals.log_prob.device)
            kl_term = torch.tensor(0.0, device=signals.log_prob.device)
            loss = policy_loss

        clip_fraction = torch.mean((torch.abs(ratio - 1.0) > cfg.clip_ratio).float()).item()
        approx_kl = 0.5 * torch.mean((signals.log_prob - old_log_probs) ** 2).item()

        metrics = TrainStepMetrics(
            loss=loss.item(),
            policy_loss=policy_loss.item(),
            kl_penalty=kl_loss.item(),
            weighted_kl_loss=kl_term.item(),
            update=PolicyUpdateStats(
                clip_fraction=clip_fraction,
                active_clip_fraction=active_clip_fraction,
                approx_kl=approx_kl,
                tis_clip_fraction=tis_clip_fraction,
                rs_seq_masked_fraction=rs_seq_masked_fraction,
            ),
        )

        return loss, metrics

    def _loss_weight(self, signals: Any) -> Any | None:
        """Per-sample positive weight on the clipped surrogate; None = unweighted.

        The Flash-GRPO subclass overrides this with its temporal gradient
        rectification factor. Base GRPO (and the trust-region subclasses, which
        own their loss bodies) return None.
        """

        return None

    @staticmethod
    def _broadcast_sample_values(values: Any, target: Any) -> Any:
        """Expand one value per sample across grouped policy-action axes."""

        if values.shape[0] != target.shape[0]:
            raise ValueError(
                "sample values and policy signals must share their leading batch axis: "
                f"{tuple(values.shape)} vs {tuple(target.shape)}",
            )
        while values.ndim < target.ndim:
            values = values.unsqueeze(-1)
        return values


@dataclass(slots=True)
class FlashGRPOConfig(GRPOConfig):
    """Hyper-parameters for Flash-GRPO (one-step policy optimization).

    Inherits every GRPO knob; the default ``clip_ratio`` drops to the paper's
    1e-3. At ppo_epochs=1 the policy does not move within an update, so the
    ratio only deviates from 1 through rollout-vs-replay numeric drift — the
    tight clip is a drift rail, not a trust region.
    """

    requirements: ClassVar[AlgorithmRequirements] = AlgorithmRequirements(
        needs_sde_rollout=True,
        sft_source="unsupported",
    )

    clip_ratio: float = 1e-3
    # Bridged by build_configs from the rollout denoise options: the SDE the
    # rectification weight is defined over, the same one the replay evaluator
    # integrates. Never YAML keys here.
    noise_level: float = field(default=1.0, init=False)
    sde_type: str = field(default="flow_grpo", init=False)


class FlashGRPO(GRPO):
    """Flash-GRPO: GRPO with per-timestep gradient rectification.

    arXiv:2605.15980 (ICML 2026). The full recipe is three mechanisms; this
    class owns the loss-side one, the other two are rollout/trainer config:

    1. Single stochastic step (rollout): ``rollout.sde.window_size=1`` with a
       ``window_range`` over the noisy early steps — one SDE step per
       trajectory, ODE elsewhere. One action per sample -> one transition to
       replay and train.
    2. Iso-temporal grouping (rollout): the window is drawn once per generation
       request, so every sample of a prompt group shares the same timestep and
       the group advantage is never confounded by timestep difficulty.
    3. Temporal gradient rectification (THIS class): under the flow-matching
       SDE with mean
       ``mu = x*(1 + std^2/(2s)*dt) + v*(1 + std^2*(1-s)/(2s))*dt`` and noise
       scale ``std*sqrt(-dt)``, the log-prob gradient w.r.t. the velocity
       scales as

           c(t) = sqrt(-dt)/std + std*sqrt(-dt)*(1-sigma)/(2*sigma)

       which varies ~2x across the trained window (verified: 1/c reproduces
       the reference implementation's hardcoded per-timestep table
       {999: 7.4770 ... 785: 3.7754} on the Wan 20-step shift-3 schedule to
       0.1%). The loss weight is ``w_i = (1/c_i) / mean(1/c)`` with the mean
       taken over EVERY sample of the optimizer update, across ranks, so
       per-timestep gradient magnitudes equalize while the effective learning
       rate is untouched (update-mean weight = 1).

    The update-wide mean is the reference's gradient-accumulation branch
    (``value_norm_list``: one denominator per accumulation window shared by all
    its microbatches). The trainer supplies it through ``prepare_update`` from
    the recorded timesteps before the first backward, so the weight of a
    sample does not depend on how the update is split into microbatches.
    Under streaming accumulation the trainer can only see one collection at a
    time, so the denominator then spans that collection: exact parity needs
    the full-batch path (or one collection per update).
    """

    config: FlashGRPOConfig

    def __init__(self, config: FlashGRPOConfig, *, scheduler: Any) -> None:
        super().__init__(config)
        # The rollout scheduler the rectification weight is defined over.
        self._scheduler = scheduler
        self._update_coe_mean: Any | None = None

    def prepare_update(self, update_timesteps: Callable[[], Any]) -> None:
        """Fix the update's rectification denominator from its recorded timesteps.

        The timesteps hold one entry per (sample, trained step) of the whole
        optimizer update on this rank; the mean of ``1/c`` is reduced across
        ranks so every microbatch of every rank divides by the same number.
        """

        from vrl.math.denoise.flow_matching import flow_sde_scale_terms

        sigma, sqrt_neg_dt, std = flow_sde_scale_terms(
            self._scheduler,
            update_timesteps(),
            noise_level=self.config.noise_level,
            sde_type=self.config.sde_type,
        )
        coe = 1.0 / self._rectification_scale(std.float(), sqrt_neg_dt.float(), sigma.float())
        self._update_coe_mean = self._cross_rank_mean(coe).clamp_min(1e-12)

    def _loss_weight(self, signals: FlowSDESignal) -> Any:
        import torch

        if self._update_coe_mean is None:
            raise RuntimeError(
                "FlashGRPO.prepare_update must run before the update's first loss: the "
                "rectification weight is normalized over the whole optimizer update",
            )

        def _per_sample(value: Any) -> Any:
            tensor = torch.as_tensor(value)
            if tensor.ndim > 1:
                return tensor.reshape(tensor.shape[0], -1).mean(dim=1)
            return tensor

        std = _per_sample(signals.std_dev_t).float()
        sqrt_neg_dt = _per_sample(signals.dt).float()
        sigma = _per_sample(signals.sigma).float()
        coe = 1.0 / self._rectification_scale(std, sqrt_neg_dt, sigma)
        return coe / self._update_coe_mean.to(device=coe.device, dtype=coe.dtype)

    @staticmethod
    def _rectification_scale(std: Any, sqrt_neg_dt: Any, sigma: Any) -> Any:
        """Flash-GRPO's log-prob gradient magnitude ``c(t)`` for one SDE transition.

        ``c = sqrt(-dt)/std + std*sqrt(-dt)*(1-sigma)/(2*sigma)``; the loss
        weight is ``1/c``. One definition serves both the per-sample weight
        (from the evaluator's SDE intermediates) and the update-wide
        denominator (from the recorded timesteps), so the two cannot drift.
        """

        sigma = sigma.clamp_min(1e-6)
        scale = sqrt_neg_dt / std + std * sqrt_neg_dt * (1 - sigma) / (2 * sigma)
        return scale.clamp_min(1e-12)

    @staticmethod
    def _cross_rank_mean(values: Any) -> Any:
        """Mean of ``values`` over all training ranks.

        Called once per update by every rank in lockstep (the trainer's
        unanimous-work gate keeps update counts balanced), so the collective
        cannot deadlock — the same argument as ``_population_std_across_ranks``.
        An empty local tensor must NOT short-circuit before the collective:
        emptiness is rank-local, so an empty rank contributes zeros instead.
        """

        from vrl.algorithms.advantages import all_reduce_sufficient_stats

        g_sum, _g_sumsq, g_count = all_reduce_sufficient_stats(values)
        return (g_sum / g_count.clamp_min(1.0)).to(values.device)


@dataclass(slots=True)
class FlowDPPOConfig(ImportanceRatioConfig):
    """Flow-DPPO: exact-Gaussian-KL trust region instead of the PPO ratio clip."""

    # The KL mask is the objective: at strict + ppo_epochs=1 the rollout and
    # current proposal means coincide, KL==0, nothing is masked, and the loss
    # collapses to plain REINFORCE.
    requirements: ClassVar[AlgorithmRequirements] = AlgorithmRequirements(
        needs_sde_rollout=True,
        sft_source="unsupported",
        requires_active_trust_region=True,
    )

    # Per-sample latent KL above which an update that *widens* the gap from the
    # rollout policy is dropped (the trust-region boundary).
    kl_mask_threshold: float = 1.0
    # Fold the per-step diffusion coefficient sqrt(-dt) into the KL sigma
    # (sigma_t = std_dev_t * sqrt_dt) rather than std_dev_t alone.
    add_kl_coefficient: bool = True


class TrustRegionGRPO(GRPO):
    """GRPO whose loss is a trust region against the rollout policy.

    The trust region replaces the reference-KL term, and these variants carry
    no clean-target SFT term, so their configs declare neither weight.
    """

    kl_coef = 0.0
    sft_weight = 0.0


class FlowDPPO(TrustRegionGRPO):
    """Trust-region GRPO: mask high-KL, gap-widening samples (no ratio clip).

    Asymmetric by construction — only updates that *increase* the divergence from
    the rollout policy are dropped (positive advantage pushing ratio up, or
    negative advantage pushing ratio down). Updates that pull back toward the old
    policy are always kept. This is the key difference from PPO's symmetric clip.
    """

    config: FlowDPPOConfig

    def compute_loss(self, inputs: AlgorithmInput) -> tuple[Any, TrainStepMetrics]:
        import torch

        from vrl.math.denoise.flow_matching import compute_kl_divergence

        cfg = self.config
        signals = inputs.signals.primary
        old_prev_sample_mean = signals.old_prev_sample_mean
        advantages = self._broadcast_sample_values(inputs.advantages, signals.log_prob)

        pc = self.config.precision_correction
        old_log_probs = behavior_log_prob(signals.log_prob, signals.old_log_prob, pc)
        raw_ratio = torch.exp(signals.log_prob - old_log_probs)
        # Bound rollout->replay precision drift (FP8/NVFP4 rollout) before it
        # enters the trust-region loss — the same TIS/RS the base GRPO applies.
        # Without it a quantized rollout's logprob drift flows unclipped into the
        # negative-advantage branch; the trust-region KL mask below only catches
        # *policy* drift, not *precision* drift. No-op when precision is not split
        # (tis_mode/rs_mode default to "off").
        ratio, tis_keep = apply_truncated_importance_weight(raw_ratio, pc)
        rs_keep = apply_rejection_sample_mask(
            signals.log_prob - old_log_probs,
            pc,
            mask=signals.mask,
        )
        # Gaussian KL between the current and rollout proposal means (the
        # current-vs-rollout drift). With add_kl_coefficient the sigma folds in the
        # per-step diffusion coefficient (sigma_t = std_dev_t * sqrt_dt, the closed
        # form compute_kl_divergence uses); without it the trust region is
        # unit-variance — mean_diff_sq / 2, with NO std_dev_t in the denominator
        # (matches verl-omni's add_kl_coefficient=False branch).
        if cfg.add_kl_coefficient:
            kl_per_sample = compute_kl_divergence(
                signals.prev_sample_mean,
                old_prev_sample_mean,
                signals.std_dev_t,
                sqrt_neg_dt=signals.dt,
            )
        else:
            non_batch = tuple(range(1, signals.prev_sample_mean.ndim))
            kl_per_sample = (signals.prev_sample_mean - old_prev_sample_mean).pow(2).mean(
                dim=non_batch
            ) / 2.0
        high_kl = self._broadcast_sample_values(
            kl_per_sample >= cfg.kl_mask_threshold,
            ratio,
        )
        pos_rm = high_kl & (ratio > 1.0) & (advantages > 0)
        neg_rm = high_kl & (ratio < 1.0) & (advantages < 0)
        trust_keep = (~(pos_rm | neg_rm)).detach()
        # Intersect the trust-region keep with the TIS/RS precision keeps; when
        # precision is not split both are None and ``keep`` collapses to
        # ``trust_keep`` (exact legacy behavior).
        keep = combine_keep_masks(
            signals.mask.to(ratio.dtype),
            trust_keep.to(ratio.dtype),
            tis_keep,
            rs_keep,
        )
        unclipped_loss = -advantages * ratio
        # Masked mean, matching GRPO/GRPOGuard: the denominator is the
        # KEPT count, not the batch size. Dividing by the batch size would scale
        # the gradient by the keep fraction, so the effective learning rate would
        # shrink exactly as the trust region engages — a gradient-magnitude
        # dilution, not the true off-policy rejection the mask is meant to be.
        policy_loss = (unclipped_loss * keep).sum() / keep.sum().clamp_min(1.0)

        masked_fraction = (1.0 - keep.mean()).item()
        tis_clip_fraction = (1.0 - tis_keep.mean()).item() if tis_keep is not None else 0.0
        rs_seq_masked_fraction = (1.0 - rs_keep.mean()).item() if rs_keep is not None else 0.0
        approx_kl = (
            0.5
            * torch.mean(
                (signals.log_prob - signals.old_log_prob) ** 2,
            ).item()
        )
        metrics = TrainStepMetrics(
            loss=policy_loss.item(),
            policy_loss=policy_loss.item(),
            kl_penalty=kl_per_sample.mean().item(),
            update=PolicyUpdateStats(
                clip_fraction=masked_fraction,
                approx_kl=approx_kl,
                tis_clip_fraction=tis_clip_fraction,
                rs_seq_masked_fraction=rs_seq_masked_fraction,
            ),
        )
        return policy_loss, metrics


@dataclass(slots=True)
class GRPOGuardConfig(ImportanceRatioConfig):
    """GRPO-Guard: ratio-mean-bias correction + per-step magnitude normalization.

    The guard terms are derived from the per-step diffusion scale.
    """

    # The ratio-mean-bias / step-scale guard is the objective: at strict +
    # ppo_epochs=1 the current-vs-rollout drift is 0, the guard correction
    # vanishes, and the loss collapses to plain GRPO.
    requirements: ClassVar[AlgorithmRequirements] = AlgorithmRequirements(
        needs_sde_rollout=True,
        sft_source="unsupported",
        requires_active_trust_region=True,
    )

    clip_ratio: float = 0.2


class GRPOGuard(TrustRegionGRPO):
    """FlowGRPO with an additive ratio-mean-bias and 1/sqrt_dt**2 step-scale norm.

    Unlike Flow-DPPO (which *drops* high-KL samples), GRPO-Guard keeps every
    sample but folds the current-vs-rollout mean drift into the ratio exponent
    (a soft correction) and normalizes the loss magnitude across denoise steps so
    early and late timesteps contribute comparably.
    """

    config: GRPOGuardConfig

    def compute_loss(self, inputs: AlgorithmInput) -> tuple[Any, TrainStepMetrics]:
        import torch

        cfg = self.config
        signals = inputs.signals.primary
        old_prev_sample_mean = signals.old_prev_sample_mean
        advantages = self._broadcast_sample_values(inputs.advantages, signals.log_prob)

        log_ratio = signals.log_prob - signals.old_log_prob
        # Bound rollout->replay precision drift (FP8/NVFP4 rollout). GRPO-Guard keeps
        # every sample by design, so TIS-*truncate* on the raw weight does not touch
        # the soft-corrected guard ratio; RS (whole-sample band rejection on the raw
        # rollout->replay log-ratio) is the effective precision guard here, plus
        # TIS-*mask* when configured. No-op when precision is not split.
        pc = self.config.precision_correction
        _, tis_keep = apply_truncated_importance_weight(torch.exp(log_ratio), pc)
        rs_keep = apply_rejection_sample_mask(log_ratio, pc, mask=signals.mask)
        sqrt_dt_mean = signals.dt.mean()
        scale = sqrt_dt_mean * signals.std_dev_t.mean()
        non_batch = tuple(range(1, signals.prev_sample_mean.ndim))
        mean_diff_sq = (signals.prev_sample_mean - old_prev_sample_mean).pow(2).mean(dim=non_batch)
        ratio_mean_bias = mean_diff_sq / (2 * scale.pow(2))
        # Project the mean drift onto the log-ratio scale, then exponentiate.
        ratio = torch.exp((log_ratio + ratio_mean_bias) * scale)
        clipped_ratio = torch.clamp(ratio, 1.0 - cfg.clip_ratio, 1.0 + cfg.clip_ratio)
        unclipped_loss = -advantages * ratio
        clipped_loss = -advantages * clipped_ratio
        per_sample_loss = torch.maximum(unclipped_loss, clipped_loss)
        # Reject out-of-band precision-drift samples (the mask alone is all ones
        # when no precision keep is active).
        keep = combine_keep_masks(signals.mask.to(ratio.dtype), tis_keep, rs_keep)
        reduced = (per_sample_loss * keep).sum() / keep.sum().clamp_min(1.0)
        active_clip_fraction = (
            ((clipped_loss > unclipped_loss).to(keep.dtype) * keep).sum()
            / keep.sum().clamp_min(1.0)
        ).item()
        # Per-step magnitude normalization (cross-timestep consistent gradients).
        policy_loss = reduced / sqrt_dt_mean.pow(2).clamp_min(1e-12)

        clip_fraction = torch.mean(
            (torch.abs(ratio - 1.0) > cfg.clip_ratio).float(),
        ).item()
        tis_clip_fraction = (1.0 - tis_keep.mean()).item() if tis_keep is not None else 0.0
        rs_seq_masked_fraction = (1.0 - rs_keep.mean()).item() if rs_keep is not None else 0.0
        approx_kl = 0.5 * torch.mean(log_ratio**2).item()
        metrics = TrainStepMetrics(
            loss=policy_loss.item(),
            policy_loss=policy_loss.item(),
            kl_penalty=ratio_mean_bias.mean().item(),
            update=PolicyUpdateStats(
                clip_fraction=clip_fraction,
                active_clip_fraction=active_clip_fraction,
                approx_kl=approx_kl,
                tis_clip_fraction=tis_clip_fraction,
                rs_seq_masked_fraction=rs_seq_masked_fraction,
            ),
        )
        return policy_loss, metrics
