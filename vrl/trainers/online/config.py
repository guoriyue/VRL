"""Configuration owned by the online RL training loop."""

from __future__ import annotations

from dataclasses import dataclass, field

from vrl.algorithms.logprob_mismatch import PrecisionCorrectionConfig
from vrl.trainers.core.types import (
    DebugConfig,
    EMAConfig,
    OptimConfig,
    PrecisionDriftGuardConfig,
    RolloutOrchestrationConfig,
)
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
        if gas < 0:
            raise ValueError(
                f"actor.gradient_accumulation_steps must be >= 0 (got {gas})",
            )
        if mbs < 0:
            raise ValueError(
                f"rollout.microbatch_size must be >= 0 (got {mbs})",
            )
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


__all__ = ["TrainerConfig"]
