"""Configuration owned by the online RL training loop."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from vrl.algorithms.logprob_mismatch import PrecisionCorrectionConfig
from vrl.trainers.core.types import (
    DebugConfig,
    EMAConfig,
    OptimConfig,
    PrecisionDriftGuardConfig,
    RolloutOrchestrationConfig,
)
from vrl.utils.config import require_exact_int
from vrl.utils.profiling import TorchProfilerConfig


@dataclass(frozen=True, slots=True)
class OnlineBatchPlan:
    """Canonical geometry and memory bounds for one online optimizer update."""

    prompts_per_batch: int = field(metadata={"yaml": "rollout"})
    n_samples_per_prompt: int = field(metadata={"yaml": "rollout"})
    gradient_accumulation_steps: int = field(default=0, metadata={"yaml": "actor"})
    replay_samples_per_batch: int = field(default=1, metadata={"yaml": "actor"})
    host_memory_budget_fraction: float = field(default=0.0, metadata={"yaml": "actor"})

    def __post_init__(self) -> None:
        prompts = require_exact_int(
            self.prompts_per_batch,
            path="rollout.prompts_per_batch",
            minimum=1,
        )
        require_exact_int(
            self.n_samples_per_prompt,
            path="rollout.n_samples_per_prompt",
            minimum=1,
        )
        accumulation_steps = require_exact_int(
            self.gradient_accumulation_steps,
            path="actor.gradient_accumulation_steps",
            minimum=0,
        )
        require_exact_int(
            self.replay_samples_per_batch,
            path="actor.replay_samples_per_batch",
            minimum=0,
        )
        if accumulation_steps > 0 and prompts % accumulation_steps != 0:
            raise ValueError(
                "actor.gradient_accumulation_steps must evenly divide "
                "rollout.prompts_per_batch when > 0 (it is the number of "
                "rollout/train microsteps the optimizer target batch is split "
                f"into): {prompts} % {accumulation_steps} != 0",
            )

        budget = self.host_memory_budget_fraction
        if isinstance(budget, bool) or not isinstance(budget, (int, float)):
            raise ValueError(
                "actor.host_memory_budget_fraction must be a finite number in [0.0, 1.0)",
            )
        budget = float(budget)
        if not math.isfinite(budget) or not 0.0 <= budget < 1.0:
            raise ValueError(
                "actor.host_memory_budget_fraction must be in [0.0, 1.0) "
                f"(0.0 disables the host-RAM fail-fast guard; got {budget})",
            )
        if budget > 0.0 and accumulation_steps == 0:
            raise ValueError(
                "actor.host_memory_budget_fraction>0 requires streaming "
                "accumulation (the guard checks host RAM per streamed microbatch); "
                "set actor.microbatch_size (or actor.gradient_accumulation_steps) "
                "so the optimizer-target batch is streamed. Got "
                f"host_memory_budget_fraction={budget} with no streaming "
                "(gradient_accumulation_steps=0).",
            )
        object.__setattr__(self, "host_memory_budget_fraction", budget)

    @property
    def microbatch_size(self) -> int:
        """Prompt groups held by one streaming slice, or the full unsplit batch."""

        if self.gradient_accumulation_steps == 0:
            return self.prompts_per_batch
        return self.prompts_per_batch // self.gradient_accumulation_steps

    @property
    def streaming(self) -> bool:
        return self.gradient_accumulation_steps > 0


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
    batch_plan: OnlineBatchPlan = field(metadata={"yaml": "bridged"})
    # Fraction of denoise timesteps that receive loss (gradient estimator
    # coverage) — an experiment decision, not a tuning knob.
    timestep_fraction: float = field(metadata={"yaml": "actor"})
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
    # --- profiling ---
    profile: bool = field(default=False, metadata={"yaml": "trainer"})

    def __post_init__(self) -> None:
        if self.timestep_selection not in ("strided", "random"):
            raise ValueError(
                "actor.timestep_selection must be 'strided' or 'random' "
                f"(got {self.timestep_selection!r})",
            )
        if self.batch_plan.streaming and int(self.ppo_epochs) != 1:
            raise ValueError(
                "actor.ppo_epochs must be 1 when streaming accumulation is on "
                "(gradient_accumulation_steps>0 or microbatch_size>0): a "
                "released microbatch cannot be replayed across epochs "
                f"(got ppo_epochs={self.ppo_epochs})",
            )


__all__ = ["OnlineBatchPlan", "TrainerConfig"]
