"""Rollout schedule factory and protocol."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any, Protocol

import torch
import torch.nn as nn

from vrl.rollouts.orchestration.continuous import ContinuousRolloutSchedule
from vrl.rollouts.orchestration.lifecycle import RolloutLifecycle
from vrl.rollouts.orchestration.strict_on_policy import StrictOnPolicyRolloutSchedule
from vrl.rollouts.orchestration.types import RolloutIteration, RolloutScheduleMode

logger = logging.getLogger(__name__)


class RolloutSchedule(Protocol):
    """Interface consumed by ``OnlineTrainer``."""

    async def next_iteration(
        self,
        prompts: list[Any],
        *,
        group_size: int,
        runtime_debug: bool = False,
    ) -> RolloutIteration: ...

    async def after_train_step(self) -> dict[str, float]: ...

    def reset(self) -> None: ...

    async def shutdown(self) -> None: ...


def build_rollout_schedule(
    config: Any,
    *,
    collector: Any,
    model: nn.Module,
    device: torch.device,
    weight_syncer: Any | None,
    sync_state_getter: Callable[[], dict[str, Any]] | None,
    weights_initialized: Callable[[], bool],
    set_weights_initialized: Callable[[bool], None],
    algorithm_tolerates_off_policy_staleness: bool = True,
) -> RolloutSchedule:
    """Build the RL rollout schedule selected by trainer config.

    ``algorithm_tolerates_off_policy_staleness`` is the algorithm's soundness
    capability (a plain bool, not the algorithm object, so the rollout layer
    stays free of any ``vrl.algorithms`` import): GRPO-family algorithms carry an
    importance-sampling correction and tolerate a bounded version lag, while
    likelihood-free objectives (DiffusionNFT) do not and require max_stale=0. The
    staleness *mechanism* is algorithm-agnostic; only this soundness bound is
    per-algorithm, so it is validated here rather than special-cased in the
    producer/consumer.
    """

    mode = RolloutScheduleMode(
        getattr(config, "schedule_mode", RolloutScheduleMode.STRICT_ON_POLICY.value),
    )
    max_pending = int(getattr(config, "max_pending_rollouts", 1))
    if mode is not RolloutScheduleMode.CONTINUOUS and max_pending != 1:
        raise ValueError(
            "rollout_orchestration.max_pending_rollouts must be 1 unless "
            "mode='continuous'",
        )

    lifecycle = RolloutLifecycle(
        collector=collector,
        model=model,
        device=device,
        weight_syncer=weight_syncer,
        sync_state_getter=sync_state_getter,
        weights_initialized=weights_initialized,
        set_weights_initialized=set_weights_initialized,
    )

    if mode is RolloutScheduleMode.STRICT_ON_POLICY:
        return StrictOnPolicyRolloutSchedule(lifecycle=lifecycle)
    if mode is RolloutScheduleMode.CONTINUOUS:
        return _build_continuous_schedule(
            config,
            lifecycle=lifecycle,
            algorithm_tolerates_off_policy_staleness=(
                algorithm_tolerates_off_policy_staleness
            ),
        )
    raise AssertionError(f"unreachable rollout schedule mode: {mode}")


def _build_continuous_schedule(
    config: Any,
    *,
    lifecycle: RolloutLifecycle,
    algorithm_tolerates_off_policy_staleness: bool,
) -> ContinuousRolloutSchedule:
    """Translate ``rollout_orchestration.continuous`` config into the schedule.

    Reads fields via ``getattr`` to keep the rollout layer free of any
    ``vrl.trainers`` import (architecture boundary).
    """

    # ContinuousRolloutConfig (vrl.trainers.core.types) is the single source of
    # these defaults; read its already-resolved fields via getattr so the rollout
    # layer needs no vrl.trainers import (architecture boundary) and no second
    # copy of the default values.
    cont = getattr(config, "continuous", None)
    if cont is None:
        raise RuntimeError(
            "rollout_orchestration.schedule_mode='continuous' requires a continuous config "
            "block (ContinuousRolloutConfig); none was provided",
        )

    max_inflight_groups = int(cont.max_inflight_groups)
    max_ready_groups = int(cont.max_ready_groups)
    max_stale_policy_versions = int(cont.max_stale_policy_versions)

    # Per-algorithm soundness gate. A likelihood-free algorithm has no way to
    # reweight off-policy samples, so a non-zero staleness window does not just
    # add variance — it silently biases the objective. Fail fast instead of
    # producing a quietly-wrong run; GRPO-family algorithms pass through.
    if max_stale_policy_versions > 0 and not algorithm_tolerates_off_policy_staleness:
        raise ValueError(
            "rollout_orchestration.continuous.max_stale_policy_versions="
            f"{max_stale_policy_versions} is unsound for this algorithm: it is "
            "likelihood-free (no importance-sampling correction), so it can only "
            "train on strictly on-policy rollouts. Set max_stale_policy_versions=0 "
            "(serial/on-policy), or use a GRPO-family algorithm for async "
            "off-policy prefetch.",
        )

    # Make "is this actually async?" self-reporting at startup. With
    # max_stale=0 the consumer rejects every prior-version group, so continuous
    # silently degrades to strict_on_policy — no off-policy prefetch, no
    # rollout/train overlap — despite the mode flag (see module docstring of
    # continuous/schedule.py). Warn rather than let the run look async when it
    # is serial.
    if max_stale_policy_versions <= 0:
        logger.warning(
            "continuous rollout with max_stale_policy_versions=0 is "
            "behavior-equivalent to strict_on_policy: off-policy prefetch is "
            "OFF and rollout/train do not overlap. Set "
            "rollout_orchestration.continuous.max_stale_policy_versions>=1 to "
            "enable async prefetch.",
        )
    else:
        logger.info(
            "continuous async prefetch ENABLED: max_stale_policy_versions=%d, "
            "max_ready_groups=%d, max_inflight_groups=%d",
            max_stale_policy_versions,
            max_ready_groups,
            max_inflight_groups,
        )

    return ContinuousRolloutSchedule(
        lifecycle=lifecycle,
        require_separate_gpus=bool(getattr(config, "require_separate_gpus", True)),
        max_inflight_groups=max_inflight_groups,
        max_ready_groups=max_ready_groups,
        max_ready_bytes_mb=int(cont.max_ready_bytes_mb),
        max_stale_policy_versions=max_stale_policy_versions,
        wait_timeout_s=float(cont.wait_timeout_s),
        queue_poll_interval_s=float(cont.queue_poll_interval_s),
        fail_fast_errors=int(cont.fail_fast_errors),
    )


__all__ = ["RolloutSchedule", "build_rollout_schedule"]
