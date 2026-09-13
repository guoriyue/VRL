"""Trainer-facing facade for disaggregated continuous rollout.

The queue/producer/consumer and all asynchronous collector/runtime work live on
``ContinuousRolloutThread``'s dedicated thread.  This facade performs only
main-thread policy-state export and command/future handoff, so synchronous
training work cannot starve rollout admission or completion harvesting.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from vrl.rollouts.orchestration.continuous.thread import (
    ContinuousRolloutThread,
)
from vrl.rollouts.orchestration.continuous.types import ContinuousRolloutSettings
from vrl.rollouts.orchestration.rollout_runtime import RolloutRuntimeCoordinator
from vrl.rollouts.orchestration.types import RolloutIteration
from vrl.rollouts.stats import RolloutStats

if TYPE_CHECKING:
    from vrl.trainers.core.types import ContinuousRolloutConfig

logger = logging.getLogger(__name__)


class ContinuousRolloutSchedule:
    """Continuously produce on disjoint GPUs through one dedicated owner loop."""

    def __init__(
        self,
        *,
        lifecycle: RolloutRuntimeCoordinator,
        # No defaults: the typed config remains the single source of defaults.
        # ``settings`` already validated ``max_stale_policy_versions >= 1`` at
        # construction, so this facade only checks the runtime-topology guards.
        settings: ContinuousRolloutSettings,
    ) -> None:
        self.lifecycle = lifecycle
        self._validate_runtime_isolation()
        self._rollout_thread = ContinuousRolloutThread(lifecycle=lifecycle, settings=settings)

    @classmethod
    def from_config(
        cls,
        config: ContinuousRolloutConfig,
        *,
        lifecycle: RolloutRuntimeCoordinator,
        algorithm_tolerates_off_policy_staleness: bool,
    ) -> ContinuousRolloutSchedule:
        """Translate ``rollout_orchestration.continuous`` config into the schedule.

        Copies resolved fields without importing the trainer-owned config type.
        """

        # Constructing the settings enforces max_stale_policy_versions >= 1 (its
        # __post_init__), so the fail-fast on an unsound zero-window config happens
        # here without a second copy of the check.
        settings = ContinuousRolloutSettings(
            max_inflight_groups=config.max_inflight_groups,
            max_ready_bytes_mb=config.max_ready_bytes_mb,
            split_generation_reward=config.split_generation_reward,
            max_unscored_groups=config.max_unscored_groups,
            max_unscored_bytes_mb=config.max_unscored_bytes_mb,
            max_generated_group_bytes_mb=config.max_generated_group_bytes_mb,
            max_stale_policy_versions=config.max_stale_policy_versions,
            wait_timeout_s=config.wait_timeout_s,
            queue_poll_interval_s=config.queue_poll_interval_s,
            fail_fast_errors=config.fail_fast_errors,
        )

        # The algorithm owns the reason it cannot consume stale samples; this
        # boundary checks only its declared capability. Zero staleness belongs
        # to strict_on_policy rather than a continuous submode.
        if not algorithm_tolerates_off_policy_staleness:
            raise ValueError(
                "rollout_orchestration.continuous.max_stale_policy_versions="
                f"{settings.max_stale_policy_versions} is unsupported by this algorithm: "
                "it declares tolerates_off_policy_staleness=False. "
                "Use schedule_mode='strict_on_policy'.",
            )

        logger.info(
            "continuous async prefetch ENABLED: max_stale_policy_versions=%d, max_inflight_groups=%d",
            settings.max_stale_policy_versions,
            settings.max_inflight_groups,
        )

        return cls(lifecycle=lifecycle, settings=settings)

    async def next_iteration(
        self,
        prompts: list[Any],
        *,
        group_size: int,
        runtime_debug: bool = False,
        next_prompts: list[Any] | None = None,
    ) -> RolloutIteration:
        # Strategy/FSDP export and the CPU snapshot happen on the trainer
        # thread. The coordinator's initialized callback is the source of truth;
        # it returns None once the persistent runtime owns committed weights.
        initial_weights = self.lifecycle.prepare_initial_weight_sync_state()
        return await self._rollout_thread.next_iteration(
            prompts,
            group_size=group_size,
            runtime_debug=runtime_debug,
            initial_weights=initial_weights,
            next_prompts=next_prompts,
        )

    async def after_train_step(self) -> RolloutStats:
        prepared_weights = self.lifecycle.prepare_weight_sync_state()
        return await self._rollout_thread.commit_weights(prepared_weights)

    def reset(self) -> None:
        """Reset producer/queue state on the owner loop before a resumed rollout."""

        self._rollout_thread.reset()

    async def shutdown(self) -> None:
        """Stop the owner and its collector/runtime exactly once."""

        await self._rollout_thread.shutdown()

    def _validate_runtime_isolation(self) -> None:
        # No config escape hatch exists: shared physical capacity cannot support
        # rollout kernels and trainer backward concurrently.
        if self.lifecycle.requires_driver_model_offload():
            raise RuntimeError(
                "continuous rollout is disabled when rollout runtime requires "
                "driver model offload",
            )
        if self.lifecycle.requires_driver_model_offload_for_reward():
            raise RuntimeError(
                "continuous rollout cannot score rewards on the trainer GPU while "
                "backward overlaps; use a CPU/dedicated reward or strict_on_policy",
            )
        if self.lifecycle.requires_generation_offload_before_reward():
            raise RuntimeError(
                "continuous rollout requires reward scoring that does not offload "
                "the generation runtime mid-iteration; use a dedicated reward GPU "
                "or strict_on_policy scheduling",
            )
        if not self.lifecycle.collector.supports_continuous_reward_execution:
            # A single collect task still overlaps the trainer in continuous mode.
            # Limiting group concurrency therefore cannot make an external reward
            # service safe when its accelerator placement is unknown.
            raise RuntimeError(
                "continuous rollout requires verified reward accelerator isolation "
                "from both trainer and rollout GPUs; use a service that advertises "
                "generation_overlap_safe, or use strict_on_policy scheduling",
            )


__all__ = ["ContinuousRolloutSchedule"]
