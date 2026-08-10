"""Trainer-facing facade for disaggregated continuous rollout.

The queue/producer/consumer and all asynchronous collector/runtime work live on
``ContinuousRolloutOwner``'s dedicated thread.  This facade performs only
main-thread policy-state export and command/future handoff, so synchronous
training work cannot starve rollout admission or completion harvesting.
"""

from __future__ import annotations

from typing import Any

from vrl.rollouts.orchestration.continuous.owner import (
    ContinuousRolloutOwner,
)
from vrl.rollouts.orchestration.continuous.types import ContinuousRolloutSettings
from vrl.rollouts.orchestration.rollout_runtime import RolloutRuntimeCoordinator
from vrl.rollouts.orchestration.types import RolloutIteration
from vrl.rollouts.stats import RolloutStats


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
        self._owner = ContinuousRolloutOwner(lifecycle=lifecycle, settings=settings)

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
        return await self._owner.next_iteration(
            prompts,
            group_size=group_size,
            runtime_debug=runtime_debug,
            initial_weights=initial_weights,
            next_prompts=next_prompts,
        )

    async def after_train_step(self) -> RolloutStats:
        prepared_weights = self.lifecycle.prepare_weight_sync_state()
        return await self._owner.commit_weights(prepared_weights)

    def reset(self) -> None:
        """Reset producer/queue state on the owner loop before a resumed rollout."""

        self._owner.reset()

    async def shutdown(self) -> None:
        """Stop the owner and its collector/runtime exactly once."""

        await self._owner.shutdown()

    def _validate_runtime_isolation(self) -> None:
        # No config escape hatch exists: shared physical capacity cannot support
        # rollout kernels and trainer backward concurrently.  Check both the
        # runtime topology and the on-demand/offload capability so an incomplete
        # runtime protocol cannot accidentally bypass the guard.
        if self.lifecycle.runtime_is_colocated():
            raise RuntimeError(
                "continuous rollout requires separate trainer and rollout GPU ownership",
            )
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
        if not getattr(
            self.lifecycle.collector,
            "supports_continuous_reward_execution",
            False,
        ):
            # A single collect task still overlaps the trainer in continuous mode.
            # Limiting group concurrency therefore cannot make an external reward
            # service safe when its accelerator placement is unknown.
            raise RuntimeError(
                "continuous rollout requires verified reward accelerator isolation "
                "from both trainer and rollout GPUs; use a service that advertises "
                "generation_overlap_safe, or use strict_on_policy scheduling",
            )


__all__ = ["ContinuousRolloutSchedule"]
