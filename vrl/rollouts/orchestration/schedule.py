"""Rollout schedule factory and protocol."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any, Protocol

from vrl.rollouts.orchestration.continuous import (
    ContinuousRolloutSchedule,
    ContinuousRolloutSettings,
)
from vrl.rollouts.orchestration.rollout_runtime import RolloutRuntimeCoordinator
from vrl.rollouts.orchestration.strict_on_policy import StrictOnPolicyRolloutSchedule
from vrl.rollouts.orchestration.types import (
    RewardCollectionMode,
    RolloutIteration,
    RolloutScheduleMode,
)
from vrl.rollouts.stats import RolloutStats

logger = logging.getLogger(__name__)


class RolloutSchedule(Protocol):
    """Interface consumed by ``OnlineTrainer``."""

    async def next_iteration(
        self,
        prompts: list[Any],
        *,
        group_size: int,
        runtime_debug: bool = False,
        next_prompts: list[Any] | None = None,
    ) -> RolloutIteration: ...

    async def after_train_step(self) -> RolloutStats: ...

    def reset(self) -> None: ...

    async def shutdown(self) -> None: ...


def build_rollout_schedule(
    config: Any,
    *,
    collector: Any,
    strategy: Any,
    training_state_getter: Callable[[], Any],
    weight_syncer: Any | None,
    sync_state_getter: Callable[[], dict[str, Any]] | None,
    weights_initialized: Callable[[], bool],
    set_weights_initialized: Callable[[bool], None],
    algorithm_tolerates_off_policy_staleness: bool,
) -> RolloutSchedule:
    """Build the RL rollout schedule selected by trainer config.

    ``algorithm_tolerates_off_policy_staleness`` is the algorithm's soundness
    capability (a plain bool, not the algorithm object, so the rollout layer
    stays free of any ``vrl.algorithms`` import): GRPO-family algorithms carry an
    importance-sampling correction and tolerate a bounded version lag, while
    likelihood-free objectives (DiffusionNFT) must use ``strict_on_policy``. The
    staleness *mechanism* is algorithm-agnostic; only this soundness bound is
    per-algorithm, so it is validated here rather than special-cased in the
    producer/consumer.
    """

    mode = RolloutScheduleMode(config.schedule_mode)

    lifecycle = RolloutRuntimeCoordinator(
        collector=collector,
        strategy=strategy,
        training_state_getter=training_state_getter,
        weight_syncer=weight_syncer,
        sync_state_getter=sync_state_getter,
        weights_initialized=weights_initialized,
        set_weights_initialized=set_weights_initialized,
    )

    if mode is RolloutScheduleMode.STRICT_ON_POLICY:
        requested_arm = getattr(config, "reward_collection_mode", None)
        return StrictOnPolicyRolloutSchedule(
            lifecycle=lifecycle,
            reward_mode=None if requested_arm is None else RewardCollectionMode(requested_arm),
        )
    if mode is RolloutScheduleMode.CONTINUOUS:
        return _build_continuous_schedule(
            config,
            lifecycle=lifecycle,
            algorithm_tolerates_off_policy_staleness=algorithm_tolerates_off_policy_staleness,
        )
    raise AssertionError(f"unreachable rollout schedule mode: {mode}")


def validate_rollout_schedule_topology(config: Any, resources: Any) -> None:
    """Reject a schedule whose phase semantics contradict resolved GPU ownership.

    The online entrypoint calls this after resource resolution and before model or
    Ray construction. Runtime guards remain necessary for direct schedule users,
    but they are too late to be the primary configuration boundary.
    """

    mode = RolloutScheduleMode(config.schedule_mode)
    if mode is not RolloutScheduleMode.CONTINUOUS:
        return
    if bool(resources.colocated):
        raise ValueError(
            "continuous rollout requires disjoint trainer and rollout GPUs; "
            "use strict_on_policy with gpu_pool=trainer for shared-GPU phase handoff",
        )
    if bool(resources.lifecycle.release_rollout_before_reward):
        raise ValueError(
            "continuous rollout cannot hand the rollout GPU to reward scoring "
            "mid-iteration; use a dedicated reward GPU or strict_on_policy",
        )
    if bool(resources.lifecycle.release_trainer_before_reward):
        raise ValueError(
            "continuous rollout cannot run reward scoring on the trainer GPU while "
            "backward overlaps; use a CPU/dedicated reward or strict_on_policy",
        )


def _build_continuous_schedule(
    config: Any,
    *,
    lifecycle: RolloutRuntimeCoordinator,
    algorithm_tolerates_off_policy_staleness: bool,
) -> ContinuousRolloutSchedule:
    """Translate ``rollout_orchestration.continuous`` config into the schedule.

    Copies resolved fields without importing the trainer-owned config type.
    """

    # ContinuousRolloutConfig (vrl.trainers.core.types) is the single source of
    # these defaults. The rollout layer receives its already-resolved fields so it
    # needs no vrl.trainers import and keeps no second copy of the defaults.
    cont = getattr(config, "continuous", None)
    if cont is None:
        raise RuntimeError(
            "rollout_orchestration.schedule_mode='continuous' requires a continuous config "
            "block (ContinuousRolloutConfig); none was provided",
        )

    # Constructing the settings enforces max_stale_policy_versions >= 1 (its
    # __post_init__), so the fail-fast on an unsound zero-window config happens
    # here without a second copy of the check.
    settings = ContinuousRolloutSettings(
        max_inflight_groups=int(cont.max_inflight_groups),
        max_ready_bytes_mb=int(cont.max_ready_bytes_mb),
        max_stale_policy_versions=int(cont.max_stale_policy_versions),
        wait_timeout_s=float(cont.wait_timeout_s),
        queue_poll_interval_s=float(cont.queue_poll_interval_s),
        fail_fast_errors=int(cont.fail_fast_errors),
    )

    # A likelihood-free algorithm has no way to reweight off-policy samples, so
    # production continuous execution is unsound for it. Zero staleness is not a
    # continuous submode: that behavior belongs to strict_on_policy.
    if not algorithm_tolerates_off_policy_staleness:
        raise ValueError(
            "rollout_orchestration.continuous.max_stale_policy_versions="
            f"{settings.max_stale_policy_versions} is unsound for this algorithm: it is "
            "likelihood-free (no importance-sampling correction), so it can only "
            "train on strictly on-policy rollouts. Use schedule_mode='strict_on_policy', "
            "or use a GRPO-family algorithm for continuous off-policy prefetch.",
        )

    logger.info(
        "continuous async prefetch ENABLED: max_stale_policy_versions=%d, max_inflight_groups=%d",
        settings.max_stale_policy_versions,
        settings.max_inflight_groups,
    )

    return ContinuousRolloutSchedule(lifecycle=lifecycle, settings=settings)


__all__ = [
    "RolloutSchedule",
    "build_rollout_schedule",
    "validate_rollout_schedule_topology",
]
