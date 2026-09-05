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
from vrl.utils.config import require_exact_int
from vrl.utils.profiling import TorchProfilerConfig

if TYPE_CHECKING:
    from vrl.config.precision import PrecisionPolicy


@dataclass(frozen=True, slots=True)
class OnlineBatchPlan:
    """Canonical geometry and memory bounds for one online optimizer update."""

    prompts_per_batch: int = field(metadata={"yaml": "rollout"})
    n_samples_per_prompt: int = field(metadata={"yaml": "rollout"})
    gradient_accumulation_steps: int = field(default=0, metadata={"yaml": "actor"})
    samples_per_replay_batch: int = field(default=1, metadata={"yaml": "actor"})
    host_memory_budget_fraction: float = field(default=0.0, metadata={"yaml": "actor"})

    @classmethod
    def from_cfg(cls, cfg: Any) -> OnlineBatchPlan:
        """Resolve public size/count inputs into one canonical optimizer batch plan."""

        from vrl.config.validation import path_exists, require

        base = cls(
            prompts_per_batch=require(cfg, "rollout.prompts_per_batch"),
            n_samples_per_prompt=require(cfg, "rollout.n_samples_per_prompt"),
        )
        prompts = base.prompts_per_batch

        def optional_non_negative_int(path: str) -> int | None:
            if not path_exists(cfg, path):
                return None
            value = require(cfg, path)
            parsed = require_exact_int(value, path=path, minimum=0)
            return parsed if parsed > 0 else None

        accumulation_steps = optional_non_negative_int("actor.gradient_accumulation_steps")
        microbatch_size = optional_non_negative_int("actor.microbatch_size")

        active_accumulation = int(accumulation_steps or 0)
        active_microbatch = int(microbatch_size or 0)
        if active_accumulation > 0 and active_microbatch > 0:
            if active_accumulation * active_microbatch != prompts:
                raise ValueError(
                    "actor.microbatch_size * actor.gradient_accumulation_steps "
                    f"must equal rollout.prompts_per_batch "
                    f"({active_microbatch} * {active_accumulation} != {prompts}); "
                    "set only one of them.",
                )
        elif active_microbatch > 0:
            if prompts % active_microbatch != 0:
                raise ValueError(
                    "actor.microbatch_size must evenly divide "
                    f"rollout.prompts_per_batch ({prompts} % {active_microbatch} != 0)",
                )
            active_accumulation = prompts // active_microbatch

        payload: dict[str, Any] = {
            "prompts_per_batch": prompts,
            "n_samples_per_prompt": base.n_samples_per_prompt,
        }
        if active_accumulation > 0:
            payload["gradient_accumulation_steps"] = active_accumulation
        for field_name in ("samples_per_replay_batch", "host_memory_budget_fraction"):
            path = f"actor.{field_name}"
            if path_exists(cfg, path):
                payload[field_name] = require(cfg, path)
        return cls(**payload)

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
            self.samples_per_replay_batch,
            path="actor.samples_per_replay_batch",
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
    computed by ``TrainerConfig.from_cfg`` (the precision policy projects into
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
    replay_parity: ReplayParityConfig = field(
        default_factory=ReplayParityConfig,
        metadata={"yaml": "trainer.replay_parity"},
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

    # How the trained denoise-step subset is chosen each update: "strided"
    # (fixed evenly-spaced steps, default), "random" (DanceGRPO — a fresh
    # random subset per update, decorrelating denoise-step gradient coverage),
    # or "sde_window" (Flash-GRPO — exactly the steps the rollout made
    # stochastic, read from the trajectory's recorded window; requires
    # rollout.sde.window_size > 0). "strided"/"random" have no effect when
    # timestep_fraction == 1; "sde_window" ignores timestep_fraction entirely,
    # so it must be left at 1.0.
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

    @classmethod
    def from_cfg(
        cls,
        cfg: Any,
        *,
        precision: PrecisionPolicy | None = None,
    ) -> TrainerConfig:
        """Slice merged YAML into ``TrainerConfig``.

        The layout is derived from each field's ``metadata={"yaml": ...}`` on this
        dataclass (section name for scalars, dotted path for nested sections,
        "bridged" for constructor-computed values), and requiredness from the
        field defaults — the dataclass is the single declaration of both. Missing
        required keys across sections and scalars are collected and reported
        together with full YAML paths.
        """

        from vrl.config.precision import resolve_precision_policy
        from vrl.config.validation import (
            path_exists,
            require,
            section_payload_and_missing,
            validate_yaml_home,
        )

        hints = get_type_hints(cls)
        payload: dict[str, Any] = {}
        missing: list[str] = []

        # ``batch_plan`` is bridged because its public inputs span rollout and actor.
        # Derive its required public paths from the plan fields so missing-key
        # reporting cannot drift when the typed plan changes.
        for plan_field in fields(OnlineBatchPlan):
            home = plan_field.metadata.get("yaml")
            if home is None:
                raise AssertionError(
                    f"OnlineBatchPlan.{plan_field.name} does not declare its YAML home "
                    "(field metadata {'yaml': ...})",
                )
            validate_yaml_home(plan_field.name, home, owner="OnlineBatchPlan")
            path = f"{home}.{plan_field.name}"
            if (
                plan_field.default is MISSING
                and plan_field.default_factory is MISSING
                and not path_exists(cfg, path)
            ):
                missing.append(path)

        for f in fields(cls):
            if not f.init:
                continue
            home = f.metadata.get("yaml")
            if home is None:
                raise AssertionError(
                    f"{cls.__name__}.{f.name} does not declare its YAML home "
                    "(field metadata {'yaml': ...})",
                )
            if home == "bridged":
                continue
            validate_yaml_home(f.name, home, owner=cls.__name__)
            field_cls = hints[f.name]
            if is_dataclass(field_cls):
                section_payload, section_missing = section_payload_and_missing(
                    field_cls,
                    cfg,
                    home,
                )
                if section_missing:
                    missing.extend(section_missing)
                else:
                    payload[f.name] = field_cls(**section_payload)
            else:
                path = f"{home}.{f.name}"
                if path_exists(cfg, path):
                    payload[f.name] = require(cfg, path)
                elif f.default is MISSING and f.default_factory is MISSING:
                    missing.append(path)

        if missing:
            raise ValueError("config missing required key(s): " + ", ".join(sorted(missing)))

        # Resolve the public policy once; trainer fields are its runtime projection.
        if precision is None:
            from vrl.config.schema import parse_config

            precision = resolve_precision_policy(parse_config(cfg).precision)
        payload.update(
            batch_plan=OnlineBatchPlan.from_cfg(cfg),
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
            if not path_exists(cfg, "trainer.precision_correction"):
                payload["precision_correction"] = correction
            if not path_exists(cfg, "trainer.precision_drift_guard"):
                payload["precision_drift_guard"] = guard

        return cls(**payload)

    def __post_init__(self) -> None:
        if self.timestep_selection not in ("strided", "random", "sde_window"):
            raise ValueError(
                "actor.timestep_selection must be 'strided', 'random', or "
                f"'sde_window' (got {self.timestep_selection!r})",
            )
        if self.timestep_selection == "sde_window" and float(self.timestep_fraction) != 1.0:
            raise ValueError(
                "actor.timestep_selection='sde_window' derives the trained steps "
                "from the rollout's recorded stochastic window; "
                f"actor.timestep_fraction={self.timestep_fraction} would be "
                "silently ignored — leave it at 1.0",
            )
        if self.batch_plan.streaming and int(self.ppo_epochs) != 1:
            raise ValueError(
                "actor.ppo_epochs must be 1 when streaming accumulation is on "
                "(gradient_accumulation_steps>0 or microbatch_size>0): a "
                "released microbatch cannot be replayed across epochs "
                f"(got ppo_epochs={self.ppo_epochs})",
            )


__all__ = ["OnlineBatchPlan", "TrainerConfig"]
