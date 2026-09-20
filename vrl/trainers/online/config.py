"""Configuration owned by the online RL training loop."""

from __future__ import annotations

import math
from dataclasses import MISSING, dataclass, field, fields, is_dataclass
from typing import TYPE_CHECKING, Any, get_type_hints

from vrl.algorithms.logprob_mismatch import PrecisionCorrectionConfig
from vrl.trainers.core.types import (
    DebugConfig,
    EMAConfig,
    OptimConfig,
    PrecisionDriftGuardConfig,
    ReplayParityConfig,
    RolloutOrchestrationConfig,
)
from vrl.utils.profiling import TorchProfilerConfig
from vrl.utils.validation import require_int

if TYPE_CHECKING:
    from vrl.config.precision import PrecisionPolicy
    from vrl.config.schema import RootConfig


@dataclass(frozen=True, slots=True)
class OnlineBatchPlan:
    """Canonical geometry and memory bounds for one online optimizer update."""

    prompts_per_batch: int
    n_samples_per_prompt: int
    # Zero keeps the full-batch path; positive values enable streaming collection.
    prompts_per_collection: int = 0
    # Samples per training computation within one prompt group; zero keeps it whole.
    training_microbatch_size: int = 1
    # Optimizer updates per collected batch. Advantages (and a global std) are
    # computed once over the whole batch; its samples are then shuffled and split
    # into this many disjoint updates, each stepping the optimizer (Flash-GRPO's
    # reference trains one epoch of samples as two accumulation windows).
    optimizer_steps_per_batch: int = 1
    host_memory_budget_fraction: float = 0.0

    @classmethod
    def required_public_paths(cls, root: RootConfig) -> list[str]:
        """Public inputs with no value in ``root`` (the plan cannot be built)."""

        rollout = root.rollout
        return [
            f"rollout.{name}"
            for name in ("prompts_per_batch", "n_samples_per_prompt")
            if rollout is None or getattr(rollout, name) is None
        ]

    @classmethod
    def from_root(cls, root: RootConfig) -> OnlineBatchPlan:
        """Project public batch inputs into one canonical optimizer batch plan."""

        missing = cls.required_public_paths(root)
        if missing:
            raise ValueError("config missing required key(s): " + ", ".join(missing))
        rollout = root.rollout
        actor = root.actor
        assert rollout is not None
        if actor is not None and actor.gradient_accumulation_steps is not None:
            raise ValueError(
                "actor.gradient_accumulation_steps is only supported by offline DPO; "
                "for online training set actor.prompts_per_collection instead",
            )
        payload: dict[str, Any] = {
            "prompts_per_batch": rollout.prompts_per_batch,
            "n_samples_per_prompt": rollout.n_samples_per_prompt,
        }
        for name in (
            "prompts_per_collection",
            "training_microbatch_size",
            "optimizer_steps_per_batch",
            "host_memory_budget_fraction",
        ):
            value = None if actor is None else getattr(actor, name)
            if value is not None:
                payload[name] = value
        return cls(**payload)

    def __post_init__(self) -> None:
        prompts = require_int(
            self.prompts_per_batch,
            path="rollout.prompts_per_batch",
            minimum=1,
        )
        require_int(
            self.n_samples_per_prompt,
            path="rollout.n_samples_per_prompt",
            minimum=1,
        )
        collection_prompts = require_int(
            self.prompts_per_collection,
            path="actor.prompts_per_collection",
            minimum=0,
        )
        require_int(
            self.training_microbatch_size,
            path="actor.training_microbatch_size",
            minimum=0,
        )
        if collection_prompts > 0 and prompts % collection_prompts != 0:
            raise ValueError(
                "actor.prompts_per_collection must evenly divide "
                f"rollout.prompts_per_batch ({prompts} % {collection_prompts} != 0)",
            )
        optimizer_steps = require_int(
            self.optimizer_steps_per_batch,
            path="actor.optimizer_steps_per_batch",
            minimum=1,
        )
        if optimizer_steps > 1 and collection_prompts > 0:
            raise ValueError(
                "actor.optimizer_steps_per_batch>1 splits one collected batch into "
                "several updates, which needs the full-batch path; streaming "
                "accumulation (prompts_per_collection>0) releases each collection "
                f"after one update (got optimizer_steps_per_batch={optimizer_steps})",
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
        if budget > 0.0 and collection_prompts == 0:
            raise ValueError(
                "actor.host_memory_budget_fraction>0 requires streaming "
                "accumulation (the guard checks host RAM per collection batch); "
                "set actor.prompts_per_collection "
                "so the optimizer-target batch is streamed. Got "
                f"host_memory_budget_fraction={budget} with no streaming "
                "(prompts_per_collection=0).",
            )
        object.__setattr__(self, "host_memory_budget_fraction", budget)

    @property
    def collections_per_update(self) -> int:
        """Number of prompt collections, including one for the full-batch path."""

        if not self.streaming:
            return 1
        return self.prompts_per_batch // self.prompts_per_collection

    @property
    def streaming(self) -> bool:
        return self.prompts_per_collection > 0


@dataclass(slots=True)
class TrainerConfig:
    """Configuration for the online RL training loop.

    Fields without defaults are required (torch signature semantics): they are
    experiment decisions with no sane global value, and a silent default would
    design the experiment for the user. Fields with defaults are infra knobs;
    their default here is the single copy (base YAML must not restate it).

    Every field is a projection of the public ``actor`` / ``trainer`` section
    of the same name (``vrl.config.schema.ActorSection`` / ``TrainerSection``
    own the YAML keys and types), except the three bridged fields computed by
    :meth:`from_root`: ``batch_plan`` and the two precision labels.
    """

    # --- required: experiment-semantic decisions ---
    optim: OptimConfig
    batch_plan: OnlineBatchPlan
    # Fraction of denoise timesteps that receive loss (gradient estimator
    # coverage) — an experiment decision, not a tuning knob.
    timestep_fraction: float
    output_dir: str
    # Whether zero-advantage samples enter the loss (they still carry KL
    # weight); changes the trained sample set.
    drop_zero_advantage: bool

    # --- nested groups ---
    ema: EMAConfig = field(default_factory=EMAConfig)
    debug: DebugConfig = field(default_factory=DebugConfig)
    replay_parity: ReplayParityConfig = field(default_factory=ReplayParityConfig)
    precision_drift_guard: PrecisionDriftGuardConfig = field(
        default_factory=PrecisionDriftGuardConfig,
    )
    # Correction counterpart to the drift guard: truncated importance sampling
    # knobs, injected into the algorithm so they live at the trainer (precision)
    # level rather than in any algorithm's hyperparameters.
    precision_correction: PrecisionCorrectionConfig = field(
        default_factory=PrecisionCorrectionConfig,
    )
    rollout_orchestration: RolloutOrchestrationConfig = field(
        default_factory=RolloutOrchestrationConfig,
    )
    torch_profiler: TorchProfilerConfig = field(default_factory=TorchProfilerConfig)

    # --- gradient ---
    max_norm: float = 1.0

    # How the trained denoise-step subset is chosen each update: "strided"
    # (fixed evenly-spaced steps, default), "random" (DanceGRPO — a fresh
    # random subset per update, decorrelating denoise-step gradient coverage),
    # "stratified" (V-GRPO — one random step per equal-length interval of the
    # grid, resampled per update), or "sde_window" (Flash-GRPO — exactly the steps the rollout made
    # stochastic, read from the trajectory's recorded window; requires
    # rollout.sde.window_size > 0). "strided"/"random" have no effect when
    # timestep_fraction == 1; "sde_window" ignores timestep_fraction entirely,
    # so it must be left at 1.0.
    timestep_selection: str = "strided"

    # --- PPO/GRPO loop ---
    ppo_epochs: int = 1

    # --- precision (bridged from the unified precision policy) ---
    # Replay/training execution signature (for example fp16+no-autocast).
    # Empty -> fp32 ("no"). Production bridges the resolved public role; legacy
    # consumers extract its base dtype instead of re-resolving execution policy.
    train_precision: str = ""
    # Rollout execution signature (for example bf16 or bf16+fp8). Empty ->
    # treated as the training precision. The drift guard compares the two to
    # decide whether to enforce parity without adding rollout-only build fields
    # to TrainerConfig.
    rollout_precision: str = ""

    # --- profiling ---
    profile: bool = False

    @classmethod
    def from_root(
        cls,
        root: RootConfig,
        *,
        precision: PrecisionPolicy | None = None,
    ) -> TrainerConfig:
        """Project the parsed ``actor`` / ``trainer`` sections into ``TrainerConfig``.

        Each field is read from the public section that declares its name;
        requiredness comes from the field defaults, so the dataclass is the
        single declaration of both. Missing required keys across sections are
        collected and reported together with full YAML paths.
        """

        from vrl.config.precision import PrecisionPolicy
        from vrl.config.schema import ActorSection, TrainerSection

        sections = {"actor": root.actor, "trainer": root.trainer}
        section_fields = {
            "actor": ActorSection.model_fields,
            "trainer": TrainerSection.model_fields,
        }
        bridged = {"batch_plan", "train_precision", "rollout_precision"}

        hints = get_type_hints(cls)
        payload: dict[str, Any] = {}
        missing = OnlineBatchPlan.required_public_paths(root)
        for f in fields(cls):
            if not f.init or f.name in bridged:
                continue
            owners = [
                name for name, model_fields in section_fields.items() if f.name in model_fields
            ]
            if len(owners) != 1:
                raise AssertionError(
                    f"{cls.__name__}.{f.name} must be declared by exactly one public "
                    f"section (ActorSection / TrainerSection); found {owners}",
                )
            section_name = owners[0]
            section = sections[section_name]
            path = f"{section_name}.{f.name}"
            value = None if section is None else getattr(section, f.name)
            required = f.default is MISSING and f.default_factory is MISSING
            if value is None:
                if section is not None and f.name in section.model_fields_set:
                    raise ValueError(f"config key {path} is null; delete the key or fill it")
                if required:
                    field_type = hints[f.name]
                    nested = []
                    if is_dataclass(field_type):
                        nested = [
                            f"{path}.{child.name}"
                            for child in fields(field_type)
                            if child.init
                            and child.default is MISSING
                            and child.default_factory is MISSING
                        ]
                    missing.extend(nested or [path])
                continue
            payload[f.name] = value

        if missing:
            raise ValueError("config missing required key(s): " + ", ".join(sorted(missing)))

        # Resolve the public policy once; trainer fields are its runtime projection.
        if precision is None:
            precision = PrecisionPolicy.from_section(root.precision)
        payload.update(
            batch_plan=OnlineBatchPlan.from_root(root),
            train_precision=precision.training.label,
            rollout_precision=precision.rollout.label,
        )
        # On a rollout/train precision split, the correction mechanism is an
        # implementation detail the user should not have to spell out: default to
        # TIS/RS correction plus a catastrophic-drift guard. Explicit expert
        # trainer.precision_* blocks are still respected.
        if not precision.stages_match:
            from vrl.config.builders import build_precision_split_safety_configs

            correction, guard = build_precision_split_safety_configs()
            explicit = set() if root.trainer is None else root.trainer.model_fields_set
            if "precision_correction" not in explicit:
                payload["precision_correction"] = correction
            if "precision_drift_guard" not in explicit:
                payload["precision_drift_guard"] = guard

        return cls(**payload)

    def __post_init__(self) -> None:
        if self.timestep_selection == "sde_window" and float(self.timestep_fraction) != 1.0:
            raise ValueError(
                "actor.timestep_selection='sde_window' derives the trained steps "
                "from the rollout's recorded stochastic window; "
                f"actor.timestep_fraction={self.timestep_fraction} would be "
                "silently ignored — leave it at 1.0",
            )
        if self.ema.step_per_microbatch and self.batch_plan.collections_per_update > 1:
            raise ValueError(
                "actor.ema.step_per_microbatch steps the shadow after every replay "
                "microbatch and once more after the optimizer step, which needs the "
                "update's last microbatch to be known: collect the update in one "
                "collection (prompts_per_collection=0 or =prompts_per_batch)",
            )
        if self.batch_plan.streaming and int(self.ppo_epochs) != 1:
            raise ValueError(
                "actor.ppo_epochs must be 1 when streaming accumulation is on "
                "(prompts_per_collection>0): a "
                "released microbatch cannot be replayed across epochs "
                f"(got ppo_epochs={self.ppo_epochs})",
            )


__all__ = ["OnlineBatchPlan", "TrainerConfig"]
