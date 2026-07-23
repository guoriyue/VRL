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
from typing import Literal

from vrl.algorithms.logprob_mismatch import PrecisionCorrectionConfig
from vrl.utils.profiling import TorchProfilerConfig


def _parse_non_negative_int(value: object, *, path: str) -> int:
    """Validate a non-negative integer configuration boundary."""

    if isinstance(value, bool):
        raise ValueError(f"{path} must be a non-negative integer")
    try:
        parsed = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path} must be a non-negative integer (got {value!r})") from exc
    if parsed < 0:
        raise ValueError(f"{path} must be >= 0 (got {parsed})")
    return parsed


def _parse_samples_per_chunk(value: object, *, path: str) -> int | Literal["auto"]:
    """Validate generation chunking without resolving its runtime ``auto`` mode."""

    if value == "auto":
        return "auto"
    return _parse_non_negative_int(value, path=path)


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
    # 8-bit AdamW (bitsandbytes): keeps the optimizer momentum/variance in int8, cutting
    # Adam state from ~8 to ~2 bytes/param. This is what makes FULL-PARAMETER fine-tuning
    # of a 2B+ DiT fit on a single 32GB card (fp32 Adam state alone is ~16GB for 2B). It
    # quantizes the OPTIMIZER STATE, not the forward, so it does NOT change rollout/replay
    # logprobs — safe on the RL policy path (unlike fp8 forward). Default off (fp32 AdamW).
    optim_8bit: bool = field(default=False)


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
    max_abs_logprob_diff: float = field(default=0.01)

    def __post_init__(self) -> None:
        if float(self.max_abs_logprob_diff) < 0:
            raise ValueError("trainer.debug.max_abs_logprob_diff must be >= 0")


@dataclass(slots=True)
class PrecisionDriftGuardConfig:
    """Rollout-vs-replay logprob parity guard (precision/backend drift).

    A correctness guard, not a debug probe: when rollout/replay role precision
    differs, the collection-time logprob may no longer equal the
    recomputed replay logprob, so the GRPO importance ratio drifts from 1 at the
    first step.

    ``mode``: ``"off"``/``"warn"``/``"fail"`` are explicit; ``"auto"`` enables the guard
    only when rollout!=train precision and resolves to ``"fail"``. Use explicit
    ``"warn"``/``"fail"`` for same-role acceptance runs, such as
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

    Continuous execution is bounded off-policy prefetch by definition. A
    zero-version window is serial strict-on-policy execution and belongs to the
    ``strict_on_policy`` schedule instead of a second continuous submode.
    """

    max_inflight_groups: int = field(default=1)
    max_ready_groups: int = field(default=2)
    max_ready_bytes_mb: int = field(default=8192)
    max_stale_policy_versions: int = field(default=1)
    wait_timeout_s: float = field(default=300.0)
    queue_poll_interval_s: float = field(default=0.05)
    # Failures of one prompt-batch slot tolerated before raising the producer's
    # root cause, instead of waiting for the full wait_timeout_s. The consumer
    # also applies this budget when no request completes at all. 0 disables both
    # fast paths. This is the single source of the default; the rollout layer
    # receives the configured value explicitly.
    fail_fast_errors: int = field(default=3)

    def __post_init__(self) -> None:
        if int(self.max_inflight_groups) < 1:
            raise ValueError("continuous.max_inflight_groups must be >= 1")
        if int(self.max_ready_groups) < 1:
            raise ValueError("continuous.max_ready_groups must be >= 1")
        if int(self.max_stale_policy_versions) < 1:
            raise ValueError("continuous.max_stale_policy_versions must be >= 1")
        if int(self.fail_fast_errors) < 0:
            raise ValueError("continuous.fail_fast_errors must be >= 0")


@dataclass(slots=True)
class RolloutOrchestrationConfig:
    """RL rollout schedule configuration."""

    schedule_mode: str = field(default="strict_on_policy")
    continuous: ContinuousRolloutConfig = field(default_factory=ContinuousRolloutConfig)
    # Acceptance-measurement override for the generation/reward collection arm.
    # None = derive from the collector's overlap capability (the production
    # path). Values mirror RewardCollectionMode; this module keeps them as
    # strings because trainer config deliberately imports nothing from
    # vrl.rollouts, exactly as schedule_mode does above. The enum conversion and
    # the capability fail-closed check happen in the rollout layer.
    reward_collection_mode: str | None = field(default=None)

    def __post_init__(self) -> None:
        if self.schedule_mode not in {"strict_on_policy", "continuous"}:
            raise ValueError(
                "rollout_orchestration.schedule_mode must be 'strict_on_policy' or 'continuous'",
            )
        if isinstance(self.continuous, dict):
            self.continuous = ContinuousRolloutConfig(**self.continuous)
        if self.reward_collection_mode is not None:
            allowed = {"batched_serial", "per_group_serial", "per_group_streaming"}
            if self.reward_collection_mode not in allowed:
                raise ValueError(
                    "rollout_orchestration.reward_collection_mode must be one of "
                    f"{sorted(allowed)}",
                )
            # Continuous collects one group per call, so no arm can overlap
            # inside a collection. Accepting the key there would be a knob the
            # user sets expecting an effect that cannot exist.
            if self.schedule_mode == "continuous":
                raise ValueError(
                    "rollout_orchestration.reward_collection_mode applies to "
                    "strict_on_policy collection only; continuous collects one group "
                    "per call and has no in-collection arm to select",
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
    computed by ``build_trainer_config`` (the precision policy projects into
    the two trainer-side precision fields). The builder derives the
    whole layout from this metadata — there is no separate layout table to
    maintain, and a field without metadata fails loudly at build time.
    """

    # --- required: experiment-semantic decisions ---
    optim: OptimConfig = field(metadata={"yaml": "actor.optim"})
    # GRPO advantage group size (samples per prompt).
    n_samples_per_prompt: int = field(metadata={"yaml": "rollout"})
    prompts_per_batch: int = field(metadata={"yaml": "rollout"})
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
    # Correction counterpart to the drift guard: truncated importance sampling
    # knobs, injected into the algorithm so they live at the trainer (precision)
    # level rather than in any algorithm's hyperparameters.
    precision_correction: PrecisionCorrectionConfig = field(
        default_factory=PrecisionCorrectionConfig,
        metadata={"yaml": "trainer.precision_correction"},
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

    # How the timestep_fraction subset is chosen each update: "strided" (fixed
    # evenly-spaced steps, default) or "random" (DanceGRPO — a fresh random
    # subset per update, decorrelating denoise-step gradient coverage). No effect
    # when timestep_fraction == 1 (all steps trained either way).
    timestep_selection: str = field(default="strided", metadata={"yaml": "actor"})

    # --- PPO/GRPO loop ---
    ppo_epochs: int = field(default=1, metadata={"yaml": "actor"})
    # 0 preserves the legacy behavior: accumulate every collected rollout
    # batch in one optimizer update. Positive values match Flow-GRPO's
    # microbatch accumulation cadence.
    gradient_accumulation_steps: int = field(default=0, metadata={"yaml": "actor"})
    # Streaming slice declared as a SIZE: groups per microbatch. Inverse of
    # gradient_accumulation_steps (= prompts_per_batch // this). Set ONE of the
    # two; __post_init__ derives the other so they cannot drift. 0 = unset
    # (derive from gradient_accumulation_steps). This is the "set the slice once"
    # knob: you declare how many groups fit in one slice, the microstep count
    # falls out (SPRINT_memory_budgeted_microbatch).
    microbatch_size: int = field(default=0, metadata={"yaml": "rollout"})
    # Generation-side sample chunk size. ``auto`` is resolved by the Ray
    # generation runtime; 0 keeps the legacy full-group fallback. This field is
    # carried here only because TrainerConfig bridges the merged online recipe --
    # the trainer must not treat the generation verdict as a replay capacity.
    samples_per_chunk: int | Literal["auto"] = field(
        default=0,
        metadata={"yaml": "rollout"},
    )
    # Training-replay sample chunk size, independent of generation because
    # backward has a lower memory ceiling. Default 1 is the safe video-training
    # floor; recipes with measured headroom may explicitly raise it. 0 requests
    # one unsplit full prompt group.
    replay_samples_per_chunk: int = field(
        default=1,
        metadata={"yaml": "actor"},
    )
    # Fail-fast host-RAM guard for streaming accumulation: if, after collecting
    # one streamed microbatch, system memory used-fraction exceeds this, raise
    # immediately instead of OOMing minutes into the run. Streaming holds ~one
    # microbatch of replay tensors at a time, so one microbatch already over
    # budget means a bigger slice (or more epochs) will OOM later. 0.0 = off
    # (no measurement); e.g. 0.9 = fail when >90% of host RAM is in use
    # (SPRINT_memory_budgeted_microbatch T2).
    host_memory_budget_fraction: float = field(default=0.0, metadata={"yaml": "rollout"})

    # --- precision (bridged from the unified precision policy) ---
    # Replay/training execution signature (for example fp16+no-autocast).
    # Empty -> fp32 ("no"). Production bridges the resolved public role; legacy
    # consumers extract its base dtype instead of re-resolving execution policy.
    train_precision: str = field(default="", metadata={"yaml": "bridged"})
    # off | full | selective (or bool: true=full, false=off). Activation
    # checkpointing is a recompute tax that lowers MFU; it only pays for itself
    # when activations would otherwise OOM -- i.e. video / high-resolution x
    # high-batch. ``full`` recomputes every block (~1.3-2x slower backward).
    # ``selective`` (SAC) saves the expensive GEMM/attention outputs and recomputes
    # only cheap norm/pointwise, recovering ~2/3 of full's tax while still reaching
    # larger batches than off -- the MFU-preferred mode for large-batch runs that
    # OOM without checkpointing (measured: SPRINT_training_mfu_selective_checkpointing).
    gradient_checkpointing: bool | str = field(default=False, metadata={"yaml": "actor"})
    # Rollout execution signature (for example bf16 or bf16+fp8). Empty ->
    # treated as the training precision. The drift guard compares the two to
    # decide whether to enforce parity without adding rollout-only build fields
    # to TrainerConfig.
    rollout_precision: str = field(default="", metadata={"yaml": "bridged"})

    # --- lifecycle ---
    save_freq: int = field(default=50, metadata={"yaml": "trainer"})
    seed: int = field(default=0, metadata={"yaml": "trainer"})

    # --- profiling ---
    profile: bool = field(default=False, metadata={"yaml": "trainer"})

    def __post_init__(self) -> None:
        if self.timestep_selection not in ("strided", "random"):
            raise ValueError(
                "actor.timestep_selection must be 'strided' or 'random' "
                f"(got {self.timestep_selection!r})",
            )
        # Streaming-accumulation slice (SPRINT_streaming_rollout_accumulation +
        # SPRINT_memory_budgeted_microbatch). The optimizer-target batch
        # (prompts_per_batch groups) is collected/trained/released in
        # microbatches. The slice may be declared as a SIZE
        # (microbatch_size = groups per microbatch) OR as a COUNT
        # (gradient_accumulation_steps = number of microbatches); set ONE and the
        # other is derived here so the two views can never drift. 0/0 keeps the
        # legacy unsplit full-batch path.
        rbs = int(self.prompts_per_batch)
        gas = int(self.gradient_accumulation_steps)
        mbs = int(self.microbatch_size)
        # _parse_samples_per_chunk already enforces non-negative for the int form
        # and passes the "auto" sentinel through (resolved by the runtime, not here).
        samples_per_chunk = _parse_samples_per_chunk(
            self.samples_per_chunk,
            path="rollout.samples_per_chunk",
        )
        if gas < 0:
            raise ValueError(
                f"actor.gradient_accumulation_steps must be >= 0 (got {gas})",
            )
        if mbs < 0:
            raise ValueError(
                f"rollout.microbatch_size must be >= 0 (got {mbs})",
            )
        self.samples_per_chunk = samples_per_chunk
        replay_samples_per_chunk = _parse_non_negative_int(
            self.replay_samples_per_chunk,
            path="actor.replay_samples_per_chunk",
        )
        self.replay_samples_per_chunk = replay_samples_per_chunk
        if mbs > 0 and gas > 0:
            # Both declared: must agree (no drift). Tell the user to set one.
            if rbs != gas * mbs:
                raise ValueError(
                    "rollout.microbatch_size * actor.gradient_accumulation_steps "
                    f"must equal rollout.prompts_per_batch ({mbs} * {gas} != {rbs}); "
                    "set only one of them.",
                )
        elif mbs > 0:
            # Size declared -> derive the microstep count.
            if rbs % mbs != 0:
                raise ValueError(
                    "rollout.microbatch_size must evenly divide "
                    f"rollout.prompts_per_batch ({rbs} % {mbs} != 0)",
                )
            gas = rbs // mbs
            self.gradient_accumulation_steps = gas
        elif gas > 0:
            # Count declared (legacy) -> derive the slice size.
            if rbs % gas != 0:
                raise ValueError(
                    "actor.gradient_accumulation_steps must evenly divide "
                    "rollout.prompts_per_batch when > 0 (it is the number of "
                    "rollout/train microsteps the optimizer target batch is split "
                    f"into): {rbs} % {gas} != 0",
                )
            self.microbatch_size = rbs // gas
        # Streaming-on checks (gas>0 after reconciliation).
        if gas > 0:
            if rbs // gas < 1:
                raise ValueError(
                    "rollout.prompts_per_batch // gradient_accumulation_steps must be "
                    f">= 1 (got {rbs} // {gas} = {rbs // gas})",
                )
            if int(self.ppo_epochs) != 1:
                raise ValueError(
                    "actor.ppo_epochs must be 1 when streaming accumulation is on "
                    "(gradient_accumulation_steps>0 or microbatch_size>0): a "
                    "released microbatch cannot be replayed across epochs "
                    f"(got ppo_epochs={self.ppo_epochs})",
                )
        budget = float(self.host_memory_budget_fraction)
        if not 0.0 <= budget < 1.0:
            raise ValueError(
                "rollout.host_memory_budget_fraction must be in [0.0, 1.0) "
                f"(0.0 disables the host-RAM fail-fast guard; got {budget})",
            )
        # The guard only runs per streamed microbatch (gas>0). With no streaming
        # the legacy full-batch path holds ALL groups before backward and never
        # consults the budget, so a >0 budget here would silently do nothing.
        # Reject it: the budget guard requires streaming, not the unsplit path it
        # cannot protect.
        if budget > 0.0 and gas == 0:
            raise ValueError(
                "rollout.host_memory_budget_fraction>0 requires streaming "
                "accumulation (the guard checks host RAM per streamed microbatch); "
                "set rollout.microbatch_size (or actor.gradient_accumulation_steps) "
                "so the optimizer-target batch is streamed. Got "
                f"host_memory_budget_fraction={budget} with no streaming "
                "(gradient_accumulation_steps=0).",
            )


@dataclass(slots=True)
class TrainState:
    """Mutable training state tracked across steps."""

    step: int = 0
    global_step: int = 0
    total_reward: float = 0.0
    total_loss: float = 0.0
