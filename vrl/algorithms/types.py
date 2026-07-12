"""RL algorithm training metrics."""

from __future__ import annotations

from dataclasses import dataclass, field, fields

from vrl.algorithms.logprob_mismatch import LogprobMismatchStats


@dataclass(slots=True)
class PolicyUpdateStats:
    """Diagnostics produced by one policy-objective evaluation.

    These values describe the update rule, not the rollout/replay numerical
    mismatch.  Keeping that distinction explicit prevents a PPO-pass aggregate
    from being mistaken for the unchanged-policy parity snapshot.
    """

    # Fraction outside the objective's trust-region band. Flow-DPPO currently
    # reports its rejected trust-region fraction here for CSV compatibility.
    clip_fraction: float = 0.0
    # Fraction for which the clipped surrogate is actually selected by max().
    active_clip_fraction: float = 0.0
    # Historical objective-specific divergence proxy: log-ratio k2 for PPO-family
    # objectives; reference-prediction MSE for DiffusionNFT. Display-only.
    approx_kl: float = 0.0
    # Precision-correction observation rates. The masks already affected the loss;
    # these values report how often that happened and do not control another branch.
    tis_clip_fraction: float = 0.0
    rs_seq_masked_fraction: float = 0.0

    @classmethod
    def weighted_mean(
        cls,
        values: list[PolicyUpdateStats],
        weights: list[float] | None = None,
    ) -> PolicyUpdateStats:
        """Reduce update diagnostics with one shared, field-derived mean."""

        if not values:
            return cls()
        resolved_weights = [1.0] * len(values) if weights is None else list(weights)
        if len(resolved_weights) != len(values):
            raise ValueError("PolicyUpdateStats values/weights length mismatch")
        total_weight = sum(resolved_weights)
        if total_weight <= 0:
            return cls()
        return cls(
            **{
                item.name: sum(
                    float(getattr(value, item.name)) * weight
                    for value, weight in zip(values, resolved_weights, strict=True)
                )
                / total_weight
                for item in fields(cls)
            },
        )


@dataclass(frozen=True, slots=True)
class InitialReplayStats:
    """Diagnostics captured before the first optimizer boundary.

    This deliberately narrow snapshot separates unchanged-policy replay parity
    from the aggregate diagnostics collected after later optimizer steps. Every
    field is either written to the stable metrics schema or consumed by the
    pre-step parity gate.
    """

    # Display-only snapshots serialized to the stable CSV diagnostics. They do
    # not choose a training branch; the max/finite pair below owns the gate.
    clip_fraction: float = 0.0
    active_clip_fraction: float = 0.0
    # Behavior-consumed by the pre-optimizer replay-parity gate.
    logprob_abs_diff_max: float = 0.0
    finite: bool = True


@dataclass(slots=True)
class TrainStepMetrics:
    """Objective metrics that the trainer promotes into one public step result.

    Algorithms fill loss decomposition + ``update``. The trainer then aggregates
    evaluations and owns rewards, mismatch, pass-zero snapshots, and grad norm.
    """

    loss: float = 0.0
    policy_loss: float = 0.0
    # Raw KL diagnostic. Some guard algorithms reuse this field for their
    # unweighted divergence/bias measure; compare loss impact with the explicit
    # weighted field below rather than inferring it from this value alone.
    kl_penalty: float = 0.0
    # Additive KL term that actually enters the optimized loss. Algorithms
    # without a coefficient-weighted KL regularizer leave it at zero.
    weighted_kl_loss: float = 0.0
    reward_mean: float = 0.0
    reward_std: float = 0.0
    # Batch-owned raw reward observations summarized by the trainer. Keeping
    # them on the metric result prevents continuous prefetch from replacing a
    # shared reward model's last-call state before this step is logged.
    reward_components: dict[str, float] = field(default_factory=dict)
    advantage_mean: float = 0.0
    # Objective diagnostics across all PPO passes, and the corresponding
    # rollout-vs-replay mismatch measured by the trainer/evaluator boundary.
    update: PolicyUpdateStats = field(default_factory=PolicyUpdateStats)
    logprob_mismatch: LogprobMismatchStats = field(default_factory=LogprobMismatchStats)
    # Trainer-owned snapshot from the evaluations immediately before the first
    # optimizer boundary. Algorithms never populate it.
    initial_replay: InitialReplayStats = field(default_factory=InitialReplayStats)
    # Weighted diffusion-loss regularizer term (algorithm.sft_weight *
    # pretraining MSE on clean fine-tuning latents); 0 when the knob is off.
    sft_loss: float = 0.0
    grad_norm: float = 0.0
    adv_saturation: float = 0.0
    adv_zero_rate: float = 0.0
    # Per-prompt grouping diagnostics, derived from the batch group_ids.
    group_size: float = 0.0  # avg samples per unique prompt in batch
    trained_prompt_num: int = 0  # unique prompts in this batch
    phase_times: dict[str, float] = field(default_factory=dict)
