"""Rollout schedule factory and protocol."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Protocol

from vrl.rollouts.collector.core import RewardCollectionMode
from vrl.rollouts.orchestration.continuous import (
    ContinuousRolloutSchedule,
)
from vrl.rollouts.orchestration.rollout_runtime import RolloutRuntimeCoordinator
from vrl.rollouts.orchestration.strict_on_policy import StrictOnPolicyRolloutSchedule
from vrl.rollouts.orchestration.types import (
    RolloutIteration,
    RolloutScheduleMode,
)
from vrl.rollouts.stats import RolloutStats

if TYPE_CHECKING:
    from vrl.ray.resources import ResolvedDistributedResources
    from vrl.trainers.core.types import RolloutOrchestrationConfig


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
        return ContinuousRolloutSchedule.from_config(
            config,
            lifecycle=lifecycle,
            algorithm_tolerates_off_policy_staleness=algorithm_tolerates_off_policy_staleness,
        )
    raise AssertionError(f"unreachable rollout schedule mode: {mode}")


def validate_rollout_schedule_topology(
    config: RolloutOrchestrationConfig,
    resources: ResolvedDistributedResources,
) -> None:
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


__all__ = [
    "RolloutSchedule",
    "build_rollout_schedule",
    "validate_rollout_schedule_topology",
]
