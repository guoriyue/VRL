"""Trainer configuration and training state."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class OptimConfig:
    """Optimizer hyper-parameters."""

    lr: float = 1e-4
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    weight_decay: float = 1e-4
    eps: float = 1e-8
    use_8bit_adam: bool = False
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
    # One-shot ||grad(policy)|| vs ||grad(beta*kl)|| split.
    grad_split: bool = False


@dataclass(slots=True)
class TorchProfilerConfig:
    """PyTorch profiler settings for visual TensorBoard traces."""

    enabled: bool = False
    output_dir: str = ""
    activities: tuple[str, ...] = ("cpu", "cuda")
    record_shapes: bool = True
    profile_memory: bool = True
    with_stack: bool = False
    with_flops: bool = False
    skip_first: int = 0
    max_steps: int = 1

    def __post_init__(self) -> None:
        self.enabled = bool(self.enabled)
        self.output_dir = str(self.output_dir or "")
        self.activities = tuple(str(activity).lower() for activity in self.activities)
        self.record_shapes = bool(self.record_shapes)
        self.profile_memory = bool(self.profile_memory)
        self.with_stack = bool(self.with_stack)
        self.with_flops = bool(self.with_flops)
        self.skip_first = max(0, int(self.skip_first))
        self.max_steps = int(self.max_steps)


@dataclass(slots=True)
class TrainerConfig:
    """Configuration for the online RL training loop."""

    # --- nested groups ---
    optim: OptimConfig = field(default_factory=OptimConfig)
    ema: EMAConfig = field(default_factory=EMAConfig)
    debug: DebugConfig = field(default_factory=DebugConfig)
    torch_profiler: TorchProfilerConfig = field(default_factory=TorchProfilerConfig)

    # --- gradient ---
    max_norm: float = 1.0

    # --- PPO/GRPO loop ---
    ppo_epochs: int = 1
    # 0 preserves the legacy behavior: accumulate every collected rollout
    # batch in one optimizer update. Positive values match Flow-GRPO's
    # microbatch accumulation cadence.
    gradient_accumulation_steps: int = 0

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
