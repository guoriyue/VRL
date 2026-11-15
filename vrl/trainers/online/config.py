"""Configuration owned by the online RL training loop."""

from __future__ import annotations

import math
from dataclasses import MISSING, dataclass, field, fields, is_dataclass, replace
from typing import TYPE_CHECKING, Any, get_type_hints

from vrl.algorithms.logprob_mismatch import PrecisionCorrectionConfig
from vrl.trainers.core.types import (
    DebugConfig,
    EMAConfig,
    OptimConfig,
    ReplayParityConfig,
    RolloutOrchestrationConfig,
)
from vrl.utils.profiling import TorchProfilerConfig
from vrl.utils.validation import require_int

if TYPE_CHECKING:
    from vrl.config.schema import RootConfig


@dataclass(frozen=True, slots=True)
class OnlineBatchConfig:
    """Canonical geometry and memory bounds for one online optimizer update."""

    prompts_per_batch: int
    n_samples_per_prompt: int
    # Zero keeps the full-batch path; positive values enable streaming collection.
    prompts_per_collection: int = 0
    # Samples per training computation within one prompt group; zero keeps it whole.
    training_microbatch_size: int = 1
    host_memory_budget_fraction: float = 0.0

    @classmethod
    def required_public_paths(cls, root: RootConfig) -> list[str]:
        """Public inputs with no value in ``root`` (the config cannot be built)."""

        rollout = root.rollout
        return [
            f"rollout.{name}"
            for name in ("prompts_per_batch", "n_samples_per_prompt")
            if rollout is None or getattr(rollout, name) is None
        ]

    @classmethod
    def from_root(cls, root: RootConfig) -> OnlineBatchConfig:
        """Project the public batch inputs into one canonical batch config."""

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
    own the YAML keys and types), except ``batch_config`` and
    ``versioned_weight_sync``, which :meth:`from_root` computes.
    """

    # required: experiment-semantic decisions
    optim: OptimConfig
    batch_config: OnlineBatchConfig
    # Fraction of denoise timesteps that receive loss (gradient estimator
    # coverage) — an experiment decision, not a tuning knob.
    timestep_fraction: float
    output_dir: str
    # Whether zero-advantage samples enter the loss (they still carry KL
    # weight); changes the trained sample set.
    drop_zero_advantage: bool

    ema: EMAConfig = field(default_factory=EMAConfig)
    debug: DebugConfig = field(default_factory=DebugConfig)
    replay_parity: ReplayParityConfig = field(default_factory=ReplayParityConfig)
    # Rollout/replay drift correction: truncated importance sampling and
    # rejection knobs. The trainer's replay-parity gate reads them here;
    # BuiltConfigs bridges the same value into the importance-ratio
    # algorithm's config, where the loss applies it.
    precision_correction: PrecisionCorrectionConfig = field(
        default_factory=PrecisionCorrectionConfig,
    )
    rollout_orchestration: RolloutOrchestrationConfig = field(
        default_factory=RolloutOrchestrationConfig,
    )
    torch_profiler: TorchProfilerConfig = field(default_factory=TorchProfilerConfig)

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

    ppo_epochs: int = 1

    # Host wall-clock phase timings land in rollout_stats.jsonl every step;
    # profile adds a CUDA sync around each trainer phase and phase_events.jsonl.
    profile: bool = False

    # Whether a weight sync may proceed while the current prompt batch is still
    # generating against the previous version: continuous scheduling over LoRA
    # adapters, which every trainable family's runtime retains per version.
    # Full-parameter fleets and strict scheduling overwrite in place and drain.
    # Decided here once; the schedule and the rollout workers both read it.
    versioned_weight_sync: bool = False

    @classmethod
    def from_root(cls, root: RootConfig) -> TrainerConfig:
        """Project the parsed ``actor`` / ``trainer`` sections into ``TrainerConfig``.

        Each field is read from the public section that declares its name;
        requiredness comes from the field defaults, so the dataclass is the
        single declaration of both. Missing required keys across sections are
        collected and reported together with full YAML paths.
        """

        from vrl.config.schema import ActorSection, TrainerSection

        sections = {"actor": root.actor, "trainer": root.trainer}
        section_fields = {
            "actor": ActorSection.model_fields,
            "trainer": TrainerSection.model_fields,
        }
        bridged = {"batch_config", "versioned_weight_sync"}

        hints = get_type_hints(cls)
        payload: dict[str, Any] = {}
        missing = OnlineBatchConfig.required_public_paths(root)
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

        payload.update(batch_config=OnlineBatchConfig.from_root(root))
        orchestration = payload.get("rollout_orchestration") or RolloutOrchestrationConfig()
        continuous = orchestration.schedule_mode == "continuous"
        payload["versioned_weight_sync"] = continuous and bool(
            root.model is not None and root.model.use_lora
        )
        # The parity gate runs before every update unless the schedule replays
        # stale samples, whose log-ratio is a policy change, not a backend gap.
        parity = payload.get("replay_parity") or ReplayParityConfig()
        if parity.every_update is None:
            payload["replay_parity"] = replace(parity, every_update=not continuous)
        elif parity.every_update and continuous:
            raise ValueError(
                "trainer.replay_parity.every_update=true cannot combine with "
                "schedule_mode='continuous': from the second update on, the batch "
                "holds samples of older policy versions, whose replay log-ratio "
                "measures the policy change rather than rollout/replay parity.",
            )

        config = cls(**payload)
        algorithm = root.algorithm
        if algorithm is not None:
            from vrl.algorithms.forward_process import ForwardProcessAlgorithm
            from vrl.config.algorithm import algorithm_class
            from vrl.rollouts.orchestration.continuous.types import MAX_STALE_POLICY_VERSIONS

            algorithm_cls = algorithm_class(algorithm.kind)
            if config.rollout_orchestration.schedule_mode == "continuous" and issubclass(
                algorithm_cls, ForwardProcessAlgorithm
            ):
                raise ValueError(
                    "trainer.rollout_orchestration.schedule_mode='continuous' trains on "
                    f"samples up to {MAX_STALE_POLICY_VERSIONS} policy version(s) old, "
                    f"but algorithm.kind="
                    f"{algorithm.kind!r} scores against the current weights and has no "
                    "importance ratio to absorb that lag. Use schedule_mode="
                    "'strict_on_policy'.",
                )
            if algorithm_cls.replay_policy == "off_policy" and not config.replays_off_policy:
                raise ValueError(
                    f"algorithm.kind={algorithm.kind!r} is defined by its importance-ratio "
                    "trust region, but the rollout schedule permits no behavior-policy "
                    "staleness and actor.ppo_epochs=1 makes the ratio identically 1 "
                    "(behavior == target on the single replay pass), so the clip/guard "
                    "term is a no-op and the run is equivalent to plain GRPO. Set "
                    "actor.ppo_epochs>1 (which needs the full-batch path, "
                    "actor.prompts_per_collection=0, since streaming releases each "
                    "collection and cannot replay it across epochs) or use "
                    "schedule_mode='continuous' for an off-policy ratio.",
                )
            if algorithm_cls.replay_policy == "on_policy" and config.replays_off_policy:
                raise ValueError(
                    f"algorithm.kind={algorithm.kind!r} is the score of the sampled action "
                    "with no importance ratio or trust region, so the replay must train "
                    "the policy that drew the sample: actor.ppo_epochs=1 under "
                    f"schedule_mode='strict_on_policy' (got ppo_epochs={config.ppo_epochs}, "
                    f"schedule_mode={config.rollout_orchestration.schedule_mode!r})",
                )
        return config

    def __post_init__(self) -> None:
        if self.timestep_selection == "sde_window" and float(self.timestep_fraction) != 1.0:
            raise ValueError(
                "actor.timestep_selection='sde_window' derives the trained steps "
                "from the rollout's recorded stochastic window; "
                f"actor.timestep_fraction={self.timestep_fraction} would be "
                "silently ignored — leave it at 1.0",
            )
        if self.batch_config.streaming and int(self.ppo_epochs) != 1:
            raise ValueError(
                "actor.ppo_epochs must be 1 when streaming accumulation is on "
                "(prompts_per_collection>0): a "
                "released microbatch cannot be replayed across epochs "
                f"(got ppo_epochs={self.ppo_epochs})",
            )
        correction = self.precision_correction
        if correction.recompute_old_logprob == "on" and self.replays_off_policy:
            raise ValueError(
                "precision_correction.recompute_old_logprob='on' replaces the rollout "
                "log-prob with the trainer's pre-update replay log-prob, which is only "
                "the behavior policy under actor.ppo_epochs=1 and no continuous "
                f"staleness; got ppo_epochs={self.ppo_epochs}, "
                f"schedule_mode={self.rollout_orchestration.schedule_mode!r}. Use 'off' "
                "(bypass) with the TIS/RS for off-policy replay.",
            )
        if (
            correction.tis_mode != "off" or correction.rs_mode != "off"
        ) and not self.replays_off_policy:
            raise ValueError(
                "precision_correction.tis_mode / rs_mode bound an off-policy importance "
                "weight (cap 2, band [0.5, 2]); with actor.ppo_epochs=1 and strict "
                "scheduling the ratio carries only the rollout/replay numerics gap, "
                "which stays two orders of magnitude inside that band (per-step "
                "|delta logprob| at most 1e-1 under fp8) and is consumed by the 1e-4 "
                "clip instead, so neither mode ever fires. Remove the gap "
                "(model.fused_kernels, or actor.training_microbatch_size == "
                "rollout.samples_per_generation_batch, one dtype for both roles) "
                "and keep the replay-parity gate; TIS/RS need actor.ppo_epochs>1 "
                "or schedule_mode='continuous'.",
            )

    @property
    def replays_off_policy(self) -> bool:
        """Whether some replay trains weights newer than its behavior policy.

        A later PPO epoch and a continuous-schedule sample (whose window is at
        least one version) both replay under weights the rollout did not use.
        """

        return int(self.ppo_epochs) > 1 or self.rollout_orchestration.schedule_mode == "continuous"


__all__ = ["OnlineBatchConfig", "TrainerConfig"]
