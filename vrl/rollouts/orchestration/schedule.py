"""Rollout schedule factory and protocol."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from pathlib import Path
from tempfile import TemporaryDirectory
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
    from vrl.trainers.distributed import ContextParallelGroups


async def collect_context_parallel_iteration(
    collect: Callable[[], Awaitable[RolloutIteration]] | None,
    *,
    groups: ContextParallelGroups,
    spool_dir: str | Path,
) -> RolloutIteration:
    """Collect once per CP group and share CPU replay data on a shared filesystem.

    All world ranks call in the same order. Only CP rank0 calls ``collect``;
    it must be rank-local and must not enter trainer/world collectives. The
    spool directory must be shared by every rank. Only a private file created
    by this invocation is deserialized, never an external checkpoint. GPU
    objects are loaded onto CPU on leaders and followers alike. Ordinary
    collection/read errors are agreed across DP as well as CP before return.
    Process death and collective transport failures require job-level recovery.
    """
    import torch
    import torch.distributed as dist

    from vrl.trainers.distributed import synchronize_context_parallel_rng

    directory = None
    message = [None]
    local_error = None
    iteration = None
    try:
        if groups.cp_rank == 0:
            try:
                if collect is None:
                    raise ValueError("CP leader requires a rollout collector")
                directory = TemporaryDirectory(prefix=".cp-rollout-", dir=spool_dir)
                iteration = await collect()
                if not isinstance(iteration, RolloutIteration):
                    raise TypeError("CP collection requires a RolloutIteration")
                path = Path(directory.name).resolve() / "iteration.pt"
                torch.save(iteration, path)
                message[0] = {"path": str(path), "error": None}
                del iteration
            except Exception as error:
                message[0] = {"path": None, "error": f"{type(error).__name__}: {error}"}
        source = dist.get_global_rank(groups.cp_group, 0)
        dist.broadcast_object_list(message, src=source, group=groups.cp_group)
        local_error = message[0]["error"]
        if local_error is None:
            try:
                iteration = torch.load(message[0]["path"], map_location="cpu", weights_only=False)
                if not isinstance(iteration, RolloutIteration):
                    raise TypeError("CP spool does not contain a RolloutIteration")
            except Exception as error:
                local_error = f"{type(error).__name__}: {error}"
        errors = [None] * (groups.dp_size * groups.cp_size)
        dist.all_gather_object(errors, local_error)
        if any(error is not None for error in errors):
            raise RuntimeError(f"CP rollout sharing failed: {errors}")
        device = (
            torch.device("cuda", torch.cuda.current_device())
            if dist.get_backend(groups.cp_group) == "nccl"
            else torch.device("cpu")
        )
        synchronize_context_parallel_rng(groups=groups, device=device)
        return iteration
    finally:
        if directory is not None:
            directory.cleanup()


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
    config: RolloutOrchestrationConfig,
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
    stays free of any ``vrl.algorithms`` import). The algorithm declares whether
    its objective supports bounded policy-version lag. Its name or use of an
    importance-sampling ratio alone cannot establish that capability. The
    continuous schedule checks this declaration; the producer/consumer implement
    the algorithm-independent staleness mechanism.
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
        reward_mode = config.reward_collection_mode
        return StrictOnPolicyRolloutSchedule(
            lifecycle=lifecycle,
            reward_mode=None if reward_mode is None else RewardCollectionMode(reward_mode),
        )
    if mode is RolloutScheduleMode.CONTINUOUS:
        return ContinuousRolloutSchedule.from_config(
            config.continuous,
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
    if resources.colocated:
        raise ValueError(
            "continuous rollout requires disjoint trainer and rollout GPUs; "
            "use strict_on_policy with gpu_pool=trainer for shared-GPU phase handoff",
        )
    if resources.lifecycle.release_rollout_before_reward:
        raise ValueError(
            "continuous rollout cannot hand the rollout GPU to reward scoring "
            "mid-iteration; use a dedicated reward GPU or strict_on_policy",
        )
    if resources.lifecycle.release_trainer_before_reward:
        raise ValueError(
            "continuous rollout cannot run reward scoring on the trainer GPU while "
            "backward overlaps; use a CPU/dedicated reward or strict_on_policy",
        )


__all__ = [
    "RolloutSchedule",
    "build_rollout_schedule",
    "collect_context_parallel_iteration",
    "validate_rollout_schedule_topology",
]
