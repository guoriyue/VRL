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
    grad_norm: float = 0.0
    adv_saturation: float = 0.0
    adv_zero_rate: float = 0.0
    lr: float = 0.0
    # Per-prompt grouping diagnostics, derived from the batch group_ids.
    group_size: float = 0.0          # avg samples per unique prompt in batch
    trained_prompt_num: int = 0      # unique prompts in this batch
    phase_times: dict[str, float] = field(default_factory=dict)
