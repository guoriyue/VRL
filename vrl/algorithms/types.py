"""RL algorithm training metrics."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class TrainStepMetrics:
    """Metrics produced by a single training step."""

    loss: float = 0.0
    policy_loss: float = 0.0
    kl_penalty: float = 0.0
    reward_mean: float = 0.0
    reward_std: float = 0.0
    advantage_mean: float = 0.0
    clip_fraction: float = 0.0
    approx_kl: float = 0.0
    # Rollout-vs-replay logprob mismatch (precision/backend drift diagnostics).
    logprob_abs_diff_mean: float = 0.0
    logprob_abs_diff_max: float = 0.0
    ratio_abs_dev_mean: float = 0.0
    ratio_abs_dev_max: float = 0.0
    mismatch_kl: float = 0.0
    mismatch_k3_kl: float = 0.0
    # Fraction of samples whose rollout->replay importance weight was truncated or
    # rejected by truncated importance sampling (0 when tis_mode='off'). Rises with
    # rollout-vs-replay precision drift (e.g. fp8 rollout vs bf16 replay).
    tis_clip_fraction: float = 0.0
    # Fraction of whole sequences rejected by reject-sampling on the log-ratio
    # drift (0 when rs_mode='off'). Sustained >~5% signals rollout drift too large
    # for bypass — tighten the RS band or fall back to recompute.
    rs_seq_masked_fraction: float = 0.0
    # Weighted diffusion-loss regularizer term (algorithm.sft_weight *
    # pretraining MSE on clean fine-tuning latents); 0 when the knob is off.
    sft_loss: float = 0.0
    grad_norm: float = 0.0
    adv_saturation: float = 0.0
    adv_zero_rate: float = 0.0
    # Per-prompt grouping diagnostics, derived from the batch group_ids.
    group_size: float = 0.0          # avg samples per unique prompt in batch
    trained_prompt_num: int = 0      # unique prompts in this batch
    phase_times: dict[str, float] = field(default_factory=dict)
