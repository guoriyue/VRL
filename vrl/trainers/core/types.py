"""Trainer configuration and training state."""

from __future__ import annotations

from dataclasses import dataclass, field

from vrl.utils.profiling import TorchProfilerConfig


@dataclass(slots=True)
class OptimConfig:
    """Optimizer hyper-parameters."""

    lr: float = 1e-4
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    weight_decay: float = 1e-4
    eps: float = 1e-8
    allow_tf32: bool = True


@dataclass(slots=True)
class EMAConfig:
    """Exponential moving average of model weights."""

    enable: bool = False
    decay: float = 0.9999
    update_interval: int = 1


@dataclass(slots=True)
class DebugConfig:
    """Diagnostic toggles consumed by the trainer."""

    # First-step log-prob round-trip check (collected old_lp vs fresh_lp).
    first_step: bool = False
    # Deprecated compatibility knob; the one-shot grad-split probe was removed.
    grad_split: bool = False


@dataclass(slots=True)
class ContinuousRolloutConfig:
    """Tuning for ``mode='continuous'`` (producer/ready-queue/consumer).

    Defaults are the safe, strict-equivalent Phase A profile: a single in-flight
    group, no staleness, and no off-policy training. Raising
    ``max_stale_policy_versions``/``max_ready_groups``/``max_inflight_groups``
    turns on bounded off-policy prefetch (the cross-node throughput mode).
    """

    max_inflight_groups: int = 1
    max_ready_groups: int = 2
    max_ready_bytes_mb: int = 8192
    max_stale_policy_versions: int = 0
    drop_policy: str = "drop_oldest_stale"
    wait_timeout_s: float = 300.0
    queue_poll_interval_s: float = 0.05

    def __post_init__(self) -> None:
        if int(self.max_inflight_groups) < 1:
            raise ValueError("continuous.max_inflight_groups must be >= 1")
        if int(self.max_ready_groups) < 1:
            raise ValueError("continuous.max_ready_groups must be >= 1")
        if int(self.max_stale_policy_versions) < 0:
            raise ValueError("continuous.max_stale_policy_versions must be >= 0")
        if self.drop_policy not in {"drop_oldest_stale", "drop_oldest"}:
            raise ValueError(
                "continuous.drop_policy must be 'drop_oldest_stale' or 'drop_oldest'",
            )


@dataclass(slots=True)
class RolloutOrchestrationConfig:
    """RL rollout schedule configuration."""

    mode: str = "strict_on_policy"
    max_pending_rollouts: int = 1
    require_separate_gpus: bool = True
    weight_sync_barrier: str = "before_sync"
    continuous: ContinuousRolloutConfig = field(default_factory=ContinuousRolloutConfig)

    def __post_init__(self) -> None:
        if self.mode not in {"strict_on_policy", "continuous"}:
            raise ValueError(
                "rollout_orchestration.mode must be 'strict_on_policy' or 'continuous'",
            )
        if isinstance(self.continuous, dict):
            self.continuous = ContinuousRolloutConfig(**self.continuous)
        if self.mode == "continuous":
            self._validate_continuous()
        else:
            self._validate_synchronous()

    def _validate_synchronous(self) -> None:
        if int(self.max_pending_rollouts) != 1:
            raise ValueError(
                "rollout_orchestration.max_pending_rollouts must be 1",
            )
        if self.weight_sync_barrier != "before_sync":
            raise ValueError(
                "rollout_orchestration.weight_sync_barrier must be 'before_sync'",
            )

    def _validate_continuous(self) -> None:
        if int(self.max_pending_rollouts) < 1:
            raise ValueError(
                "rollout_orchestration.max_pending_rollouts must be >= 1",
            )
        if self.weight_sync_barrier != "pause_admission_and_drain_inflight":
            raise ValueError(
                "rollout_orchestration.weight_sync_barrier must be "
                "'pause_admission_and_drain_inflight' for mode='continuous'",
            )


@dataclass(slots=True)
class TrainerConfig:
    """Configuration for the online RL training loop."""

    # --- nested groups ---
    optim: OptimConfig = field(default_factory=OptimConfig)
    ema: EMAConfig = field(default_factory=EMAConfig)
    debug: DebugConfig = field(default_factory=DebugConfig)
    rollout_orchestration: RolloutOrchestrationConfig = field(
        default_factory=RolloutOrchestrationConfig,
    )
    torch_profiler: TorchProfilerConfig = field(default_factory=TorchProfilerConfig)

    # --- gradient ---
    max_norm: float = 1.0

    # --- PPO/GRPO loop ---
    ppo_epochs: int = 1
    # 0 preserves the legacy behavior: accumulate every collected rollout
    # batch in one optimizer update. Positive values match Flow-GRPO's
    # microbatch accumulation cadence.
    gradient_accumulation_steps: int = 0
    drop_zero_advantage: bool = False

    # --- precision ---
    # Empty means "derive from bf16" for direct test construction/backward
    # compatibility. YAML configs should set this explicitly.
    mixed_precision: str = ""
    bf16: bool = True
    gradient_checkpointing: bool = True

    # --- rollout knobs the trainer drives ---
    n: int = 4
    rollout_batch_size: int = 4
    timestep_fraction: float = 1.0

    # --- lifecycle ---
    total_epochs: int = 10000
    save_freq: int = 50
    log_freq: int = 1
    output_dir: str = "outputs/"
    seed: int = 0
    resume_from: str = ""
    resume_strict: bool = True

    # --- profiling ---
    profile: bool = False


@dataclass(slots=True)
class TrainState:
    """Mutable training state tracked across steps."""

    step: int = 0
    global_step: int = 0
    total_reward: float = 0.0
    total_loss: float = 0.0
