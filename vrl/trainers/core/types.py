"""Trainer component configuration and mutable training state.

Signature convention (applies to every config dataclass in this module,
recursively): every field is written with explicit ``field(...)``.
``field()`` with no default = REQUIRED — torch semantics, the experiment
config must supply it, and construction fails naming the missing field.
``field(default=...)`` / ``field(default_factory=...)`` = optional, and that
default is the single copy (base YAML must not restate it).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


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

    # Rich first-step provenance for rollout/replay diagnosis and NFT probes.
    first_step: bool = field(default=False)


@dataclass(slots=True)
class ReplayParityConfig:
    """Mandatory unchanged-policy rollout/replay parity threshold."""

    max_abs_logprob_diff: float = field(default=0.01)

    def __post_init__(self) -> None:
        limit = float(self.max_abs_logprob_diff)
        if not math.isfinite(limit) or limit < 0:
            raise ValueError(
                "trainer.replay_parity.max_abs_logprob_diff must be finite and >= 0",
            )


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
        # Single user-boundary validator (vLLM's SchedulerConfig shape): every
        # range theorem is proven once here with config-key error messages;
        # the rollout mechanisms trust the carried values and do not re-check.
        if int(self.max_inflight_groups) < 1:
            raise ValueError("continuous.max_inflight_groups must be >= 1")
        if int(self.max_ready_bytes_mb) < 0:
            raise ValueError("continuous.max_ready_bytes_mb must be >= 0")
        if int(self.max_stale_policy_versions) < 1:
            raise ValueError("continuous.max_stale_policy_versions must be >= 1")
        if float(self.wait_timeout_s) <= 0:
            raise ValueError("continuous.wait_timeout_s must be > 0")
        if float(self.queue_poll_interval_s) <= 0:
            raise ValueError("continuous.queue_poll_interval_s must be > 0")
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
class TrainState:
    """Mutable training state tracked across steps."""

    step: int = 0
    global_step: int = 0
