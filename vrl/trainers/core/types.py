"""Trainer configuration and training state.

Signature convention (applies to every config dataclass in this module,
recursively): every field is written with explicit ``field(...)``.
``field()`` with no default = REQUIRED — torch semantics, the experiment
config must supply it, and construction fails naming the missing field.
``field(default=...)`` / ``field(default_factory=...)`` = optional, and that
default is the single copy (base YAML must not restate it).

``metadata={"yaml": ...}`` appears only on ``TrainerConfig`` fields because it
aggregates three YAML sections (actor/trainer/rollout) and the home of each
field cannot be inferred. Nested section classes map 1:1 to the section
declared on their parent field (key == field name inside it), so their fields
carry no metadata.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from vrl.utils.profiling import TorchProfilerConfig


@dataclass(slots=True)
class OptimConfig:
    """Optimizer hyper-parameters.

    ``lr`` is required: there is no sane global learning rate, so it must come
    from the experiment config (base actor.yaml declares it ``???``).
    """

    lr: float = field()
    adam_beta1: float = field(default=0.9)
    adam_beta2: float = field(default=0.999)
    weight_decay: float = field(default=1e-4)
    eps: float = field(default=1e-8)
    allow_tf32: bool = field(default=True)


@dataclass(slots=True)
class EMAConfig:
    """Exponential moving average of model weights."""

    enable: bool = field(default=False)
    decay: float = field(default=0.9999)
    update_interval: int = field(default=1)


@dataclass(slots=True)
class DebugConfig:
    """Diagnostic toggles consumed by the trainer."""

    # First-step log-prob round-trip check (collected old_lp vs fresh_lp).
    first_step: bool = field(default=False)


@dataclass(slots=True)
class PrecisionDriftGuardConfig:
    """Rollout-vs-replay logprob parity guard (precision/backend drift).

    A correctness guard, not a debug probe: when rollout/replay forward precision
    or backend policy differs, the collection-time logprob may no longer equal the
    recomputed replay logprob, so the GRPO importance ratio drifts from 1 at the
    first step.

    ``mode``: ``"off"``/``"warn"``/``"fail"`` are explicit; ``"auto"`` enables the guard
    only when rollout!=compute precision and resolves to ``"fail"``. Use explicit
    ``"warn"``/``"fail"`` for same-forward-precision acceptance runs, such as
    SD3.5 FP16 rollout/replay parity checks.
    """

    mode: str = field(default="auto")  # "auto" | "off" | "warn" | "fail"
    max_timestep_checks: int = field(default=3)
    max_abs_log_ratio: float = field(default=1e-3)
    max_ratio_abs_dev: float = field(default=1e-3)
    fail_on_nonfinite: bool = field(default=True)

    def __post_init__(self) -> None:
        if self.mode not in ("auto", "off", "warn", "fail"):
            raise ValueError("precision_drift_guard.mode must be auto/off/warn/fail")
        if int(self.max_timestep_checks) < 0:
            raise ValueError("precision_drift_guard.max_timestep_checks must be >= 0")
        if float(self.max_abs_log_ratio) < 0:
            raise ValueError("precision_drift_guard.max_abs_log_ratio must be >= 0")
        if float(self.max_ratio_abs_dev) < 0:
            raise ValueError("precision_drift_guard.max_ratio_abs_dev must be >= 0")


@dataclass(slots=True)
class ContinuousRolloutConfig:
    """Tuning for ``mode='continuous'`` (producer/ready-queue/consumer).

    Defaults are the safe, strict-equivalent Phase A profile: a single in-flight
    group, no staleness, and no off-policy training. Raising
    ``max_stale_policy_versions``/``max_ready_groups``/``max_inflight_groups``
    turns on bounded off-policy prefetch (the cross-node throughput mode).
    """

    max_inflight_groups: int = field(default=1)
    max_ready_groups: int = field(default=2)
    max_ready_bytes_mb: int = field(default=8192)
    max_stale_policy_versions: int = field(default=0)
    drop_policy: str = field(default="drop_oldest_stale")
    wait_timeout_s: float = field(default=300.0)
    queue_poll_interval_s: float = field(default=0.05)

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

    mode: str = field(default="strict_on_policy")
    max_pending_rollouts: int = field(default=1)
    require_separate_gpus: bool = field(default=True)
    # None derives the only barrier each mode supports; an explicit value is
    # still validated so a contradictory override fails loudly.
    weight_sync_barrier: str | None = field(default=None)
    continuous: ContinuousRolloutConfig = field(default_factory=ContinuousRolloutConfig)

    def __post_init__(self) -> None:
        if self.mode not in {"strict_on_policy", "continuous"}:
            raise ValueError(
                "rollout_orchestration.mode must be 'strict_on_policy' or 'continuous'",
            )
        if self.weight_sync_barrier is None:
            self.weight_sync_barrier = (
                "pause_admission_and_drain_inflight"
                if self.mode == "continuous"
                else "before_sync"
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
    """Configuration for the online RL training loop.

    Fields without defaults are required (torch signature semantics): they are
    experiment decisions with no sane global value, and a silent default would
    design the experiment for the user. Fields with defaults are infra knobs;
    their default here is the single copy (base YAML must not restate it).

    Each field also declares its YAML home in ``metadata={"yaml": ...}``:
    a section name for scalars (the YAML key equals the field name), a dotted
    section path for nested config dataclasses, or ``"bridged"`` for values
    computed by ``build_trainer_config`` (the precision policy expands one
    ``precision:`` key into the four precision fields). The builder derives the
    whole layout from this metadata — there is no separate layout table to
    maintain, and a field without metadata fails loudly at build time.
    """

    # --- required: experiment-semantic decisions ---
    optim: OptimConfig = field(metadata={"yaml": "actor.optim"})
    # GRPO advantage group size (samples per prompt).
    n_samples_per_prompt: int = field(metadata={"yaml": "rollout"})
    rollout_batch_size: int = field(metadata={"yaml": "rollout"})
    # Fraction of denoise timesteps that receive loss (gradient estimator
    # coverage) — an experiment decision, not a tuning knob.
    timestep_fraction: float = field(metadata={"yaml": "actor"})
    total_epochs: int = field(metadata={"yaml": "trainer"})
    output_dir: str = field(metadata={"yaml": "trainer"})
    # Whether zero-advantage samples enter the loss (they still carry KL
    # weight); changes the trained sample set.
    drop_zero_advantage: bool = field(metadata={"yaml": "actor"})

    # --- nested groups ---
    ema: EMAConfig = field(default_factory=EMAConfig, metadata={"yaml": "actor.ema"})
    debug: DebugConfig = field(
        default_factory=DebugConfig,
        metadata={"yaml": "trainer.debug"},
    )
    precision_drift_guard: PrecisionDriftGuardConfig = field(
        default_factory=PrecisionDriftGuardConfig,
        metadata={"yaml": "trainer.precision_drift_guard"},
    )
    rollout_orchestration: RolloutOrchestrationConfig = field(
        default_factory=RolloutOrchestrationConfig,
        metadata={"yaml": "trainer.rollout_orchestration"},
    )
    torch_profiler: TorchProfilerConfig = field(
        default_factory=TorchProfilerConfig,
        metadata={"yaml": "trainer.torch_profiler"},
    )

    # --- gradient ---
    max_norm: float = field(default=1.0, metadata={"yaml": "actor"})

    # --- PPO/GRPO loop ---
    ppo_epochs: int = field(default=1, metadata={"yaml": "actor"})
    # 0 preserves the legacy behavior: accumulate every collected rollout
    # batch in one optimizer update. Positive values match Flow-GRPO's
    # microbatch accumulation cadence.
    gradient_accumulation_steps: int = field(default=0, metadata={"yaml": "actor"})

    # --- precision (bridged from the unified precision policy) ---
    # Empty means "derive from bf16" for direct test construction/backward
    # compatibility. YAML configs should set this explicitly.
    mixed_precision: str = field(default="", metadata={"yaml": "bridged"})
    bf16: bool = field(default=True, metadata={"yaml": "bridged"})
    gradient_checkpointing: bool = field(default=True, metadata={"yaml": "actor"})
    # Rollout (generation) compute precision. Empty -> treated as same as
    # compute. The drift guard compares this against the compute precision to
    # decide whether to enforce parity.
    rollout_precision: str = field(default="", metadata={"yaml": "bridged"})
    # Math precision used by replay log-prob arithmetic, bridged for guard reports.
    math_precision: str = field(default="fp32", metadata={"yaml": "bridged"})

    # --- lifecycle ---
    save_freq: int = field(default=50, metadata={"yaml": "trainer"})
    log_freq: int = field(default=1, metadata={"yaml": "trainer"})
    seed: int = field(default=0, metadata={"yaml": "trainer"})
    resume_from: str = field(default="", metadata={"yaml": "trainer"})
    resume_strict: bool = field(default=True, metadata={"yaml": "trainer"})

    # --- profiling ---
    profile: bool = field(default=False, metadata={"yaml": "trainer"})

    def __post_init__(self) -> None:
        # Streaming-accumulation semantics (SPRINT_streaming_rollout_accumulation):
        # gradient_accumulation_steps is the number of rollout/train microsteps the
        # optimizer-target batch (rollout_batch_size conditions) is split into. 0
        # keeps the legacy unsplit full-batch collect+train+step path.
        gas = int(self.gradient_accumulation_steps)
        if gas < 0:
            raise ValueError(
                f"actor.gradient_accumulation_steps must be >= 0 (got {gas})",
            )
        if gas > 0:
            rbs = int(self.rollout_batch_size)
            if rbs % gas != 0:
                raise ValueError(
                    "actor.gradient_accumulation_steps must evenly divide "
                    "rollout.rollout_batch_size when > 0 (it is the number of "
                    "rollout/train microsteps the optimizer target batch is split "
                    f"into): {rbs} % {gas} != 0",
                )
            if rbs // gas < 1:
                raise ValueError(
                    "rollout.rollout_batch_size // actor.gradient_accumulation_steps "
                    f"must be >= 1 (got {rbs} // {gas} = {rbs // gas})",
                )
            if int(self.ppo_epochs) != 1:
                raise ValueError(
                    "actor.ppo_epochs must be 1 when actor.gradient_accumulation_steps "
                    "> 0: streaming accumulation collects and releases each microbatch, "
                    "so the same rollout cannot be replayed across epochs "
                    f"(got ppo_epochs={self.ppo_epochs})",
                )


@dataclass(slots=True)
class TrainState:
    """Mutable training state tracked across steps."""

    step: int = 0
    global_step: int = 0
    total_reward: float = 0.0
    total_loss: float = 0.0
