"""Tests for RL rollout orchestration schedules."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest


def _schedule_config(mode: str) -> SimpleNamespace:
    return SimpleNamespace(
        schedule_mode=mode,
    )


def _batch(prompts: list[str], group_size: int):
    import torch

    from vrl.rollouts.batch import RolloutBatch

    batch_size = len(prompts) * group_size
    group_ids = torch.tensor(
        [prompt_idx for prompt_idx in range(len(prompts)) for _ in range(group_size)],
        dtype=torch.long,
    )
    return RolloutBatch(
        rewards=torch.arange(batch_size, dtype=torch.float32),
        group_ids=group_ids,
    )


def test_runtime_coordinator_rejects_incomplete_collector_control_protocol() -> None:
    from vrl.rollouts.orchestration.rollout_runtime import RolloutRuntimeCoordinator

    with pytest.raises(TypeError, match=r"activate_runtime\(\).*offload_runtime_memory"):
        RolloutRuntimeCoordinator(
            collector=SimpleNamespace(runtime=_Runtime()),
            strategy=SimpleNamespace(),
            training_state_getter=lambda: object(),
            weight_syncer=None,
            sync_state_getter=None,
            weights_initialized=lambda: True,
            set_weights_initialized=lambda _value: None,
        )


def test_runtime_coordinator_requires_reward_handoff_capabilities() -> None:
    from vrl.rollouts.orchestration.rollout_runtime import RolloutRuntimeCoordinator

    class _MissingRewardCapabilities:
        runtime = _Runtime()

        async def activate_runtime(self) -> None:
            return None

        async def offload_runtime_memory(self) -> None:
            return None

        async def shutdown(self) -> None:
            return None

    with pytest.raises(TypeError, match="reward handoff capabilities"):
        RolloutRuntimeCoordinator(
            collector=_MissingRewardCapabilities(),
            strategy=SimpleNamespace(),
            training_state_getter=lambda: object(),
            weight_syncer=None,
            sync_state_getter=None,
            weights_initialized=lambda: True,
            set_weights_initialized=lambda _value: None,
        )


class _Runtime:
    def __init__(self) -> None:
        self.current_policy_version = 0
        self.requires_driver_model_offload = False

    def is_colocated(self) -> bool:
        # Fakes implement the GenerationRuntime protocol method directly
        # instead of mimicking the runtime's internal config layout.
        return False


class _Syncer:
    def __init__(self, runtime: _Runtime) -> None:
        self.runtime = runtime
        self.calls: list[dict[str, Any]] = []

    async def push(self, state_dict: dict[str, Any]) -> None:
        self.calls.append(dict(state_dict))
        self.runtime.current_policy_version += 1

    async def pull(self) -> dict[str, Any]:
        return dict(self.calls[-1])

    @property
    def current_policy_version(self) -> int | None:
        # Mirrors RayRuntimeWeightSyncer's PolicyVersionProvider contract.
        return self.runtime.current_policy_version


class _Collector:
    def __init__(self, runtime: _Runtime) -> None:
        self.runtime = runtime
        self.requires_runtime_offload_before_reward = False
        self.requires_driver_model_offload_for_reward = False
        self.supports_reward_generation_overlap = False
        self.calls: list[dict[str, Any]] = []
        self.activation_calls = 0
        self.offload_calls = 0

    async def collect_unscored(self, inputs, **kwargs):
        prompts = [getattr(item, "prompt", item) for item in inputs]
        self.calls.append({"prompts": prompts, **dict(kwargs)})
        return _batch(prompts, int(kwargs["group_size"]))

    async def score_rollouts(self, pendings):
        return list(pendings)

    async def activate_runtime(self) -> None:
        self.activation_calls += 1

    async def offload_runtime_memory(self) -> None:
        self.offload_calls += 1

    async def shutdown(self) -> None:
        return None


@pytest.mark.asyncio
async def test_strict_schedule_collects_and_syncs_with_rollout_metadata() -> None:
    """Checks strict schedule collects and syncs with rollout metadata."""
    import torch
    import torch.nn as nn

    from vrl.rollouts.orchestration import RolloutScheduleMode, build_rollout_schedule
    from vrl.trainers.strategy import SingleProcessStrategy, TrainingMemoryState

    runtime = _Runtime()
    collector = _Collector(runtime)
    syncer = _Syncer(runtime)
    initialized = False

    def _set_initialized(value: bool) -> None:
        nonlocal initialized
        initialized = bool(value)

    model = nn.Linear(1, 1)
    strategy = SingleProcessStrategy()
    schedule = build_rollout_schedule(
        _schedule_config("strict_on_policy"),
        collector=collector,
        strategy=strategy,
        training_state_getter=lambda: TrainingMemoryState(
            model=model,
            ref_model=None,
            optimizer=None,
            ema=None,
            grad_scaler=None,
            device=torch.device("cpu"),
        ),
        weight_syncer=syncer,
        sync_state_getter=lambda: {"w": 1},
        weights_initialized=lambda: initialized,
        set_weights_initialized=_set_initialized,
    )

    iteration = await schedule.next_iteration(
        ["p0", "p1"],
        group_size=2,
        runtime_debug=True,
    )
    await schedule.after_train_step()

    collect_call = collector.calls[0]
    assert collect_call["prompts"] == ["p0", "p1"]
    assert collect_call["policy_version"] == 1
    assert collect_call["runtime_debug"] is True
    assert iteration.mode is RolloutScheduleMode.STRICT_ON_POLICY
    assert iteration.rollout_id == 0
    assert iteration.policy_version == 1
    assert iteration.prompt_count == 2
    assert iteration.sample_count == 4
    assert iteration.metadata == {}
    assert len(iteration.batches) == 2
    assert iteration.batches[0].context["rollout_id"] == 0
    assert iteration.batches[0].context["rollout_policy_version"] == 1
    assert iteration.batches[0].context["schedule_mode"] == "strict_on_policy"
    assert iteration.batches[0].context["prompt_count"] == 2
    assert iteration.batches[0].context["sample_count"] == 4
    assert len(syncer.calls) == 2
    assert runtime.current_policy_version == 2
    assert collector.activation_calls == 1
    assert collector.offload_calls == 1


class _ParkingStrategy:
    def __init__(self, events: list[str], *, fail_restore: bool = False) -> None:
        self.events = events
        self.fail_restore = fail_restore

    def validate_training_state_parking(self) -> None:
        self.events.append("trainer.validate")

    def park_training_state(self, _state: object) -> None:
        self.events.append("trainer.park")

    def restore_training_state(self, _state: object) -> None:
        self.events.append("trainer.restore")
        if self.fail_restore:
            raise RuntimeError("restore failed")


class _FailingPhaseCollector(_Collector):
    def __init__(
        self,
        runtime: _Runtime,
        events: list[str],
        *,
        fail_collect: bool = False,
        fail_offload: bool = False,
    ) -> None:
        super().__init__(runtime)
        self.events = events
        self.fail_collect = fail_collect
        self.fail_offload = fail_offload

    async def activate_runtime(self) -> None:
        self.events.append("rollout.activate")

    async def collect_unscored(self, prompts, **kwargs):
        self.events.append("rollout.collect")
        if self.fail_collect:
            raise RuntimeError("collect failed")
        return await super().collect_unscored(prompts, **kwargs)

    async def offload_runtime_memory(self) -> None:
        self.events.append("rollout.offload")
        if self.fail_offload:
            raise RuntimeError("offload failed")

    async def shutdown(self) -> None:
        self.events.append("collector.shutdown")


def _parking_schedule(
    *,
    fail_collect: bool = False,
    fail_offload: bool = False,
    fail_restore: bool = False,
    reward_uses_trainer: bool = False,
):
    from vrl.rollouts.orchestration.rollout_runtime import RolloutRuntimeCoordinator
    from vrl.rollouts.orchestration.strict_on_policy import StrictOnPolicyRolloutSchedule

    events: list[str] = []
    runtime = _Runtime()
    runtime.requires_driver_model_offload = not reward_uses_trainer
    collector = _FailingPhaseCollector(
        runtime,
        events,
        fail_collect=fail_collect,
        fail_offload=fail_offload,
    )
    collector.requires_driver_model_offload_for_reward = reward_uses_trainer
    strategy = _ParkingStrategy(events, fail_restore=fail_restore)

    def _state() -> object:
        events.append("trainer.get_live_state")
        return object()

    lifecycle = RolloutRuntimeCoordinator(
        collector=collector,
        strategy=strategy,
        training_state_getter=_state,
        weight_syncer=None,
        sync_state_getter=None,
        weights_initialized=lambda: True,
        set_weights_initialized=lambda _value: None,
    )
    return StrictOnPolicyRolloutSchedule(lifecycle=lifecycle), events


@pytest.mark.asyncio
async def test_strict_shared_phase_keeps_handoff_order_in_next_iteration() -> None:
    schedule, events = _parking_schedule()

    await schedule.next_iteration(["p0"], group_size=1)

    assert events == [
        "trainer.validate",
        "trainer.get_live_state",
        "trainer.park",
        "rollout.activate",
        "rollout.collect",
        "rollout.offload",
        "trainer.get_live_state",
        "trainer.restore",
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("failure", "message", "expect_restore"),
    [
        ({"fail_collect": True}, "collect failed", True),
        ({"fail_offload": True}, "offload failed", False),
        ({"fail_restore": True}, "restore failed", True),
    ],
)
async def test_strict_shared_phase_restores_only_after_rollout_offload(
    failure: dict[str, bool],
    message: str,
    expect_restore: bool,
) -> None:
    schedule, events = _parking_schedule(**failure)

    with pytest.raises(RuntimeError, match=message):
        await schedule.next_iteration(["p0"], group_size=1)

    assert ("trainer.restore" in events) is expect_restore
    if expect_restore:
        assert events.index("trainer.restore") > events.index("rollout.offload")


@pytest.mark.asyncio
async def test_strict_parks_trainer_when_only_reward_shares_its_gpu() -> None:
    schedule, events = _parking_schedule(reward_uses_trainer=True)

    await schedule.next_iteration(["p0"], group_size=1)

    assert events == [
        "trainer.validate",
        "trainer.get_live_state",
        "trainer.park",
        "rollout.activate",
        "rollout.collect",
        "rollout.offload",
        "trainer.get_live_state",
        "trainer.restore",
    ]


@pytest.mark.asyncio
async def test_strict_terminal_shutdown_parks_before_shared_pipeline_cleanup() -> None:
    schedule, events = _parking_schedule(reward_uses_trainer=True)

    await schedule.shutdown()

    assert events == [
        "trainer.validate",
        "trainer.get_live_state",
        "trainer.park",
        "collector.shutdown",
    ]


@pytest.mark.asyncio
async def test_prepared_weight_push_never_reenters_live_getter() -> None:
    import torch

    from vrl.rollouts.orchestration.rollout_runtime import RolloutRuntimeCoordinator
    from vrl.rollouts.stats import RolloutStats

    runtime = _Runtime()
    syncer = _Syncer(runtime)
    live = torch.tensor([1.0], requires_grad=True)
    getter_calls = 0
    initialized = False

    def _getter() -> dict[str, Any]:
        nonlocal getter_calls
        getter_calls += 1
        return {"weight": live.detach().to("cpu", copy=True)}

    def _set_initialized(value: bool) -> None:
        nonlocal initialized
        initialized = bool(value)

    lifecycle = RolloutRuntimeCoordinator(
        collector=_Collector(runtime),
        strategy=_ParkingStrategy([]),
        training_state_getter=lambda: object(),
        weight_syncer=syncer,
        sync_state_getter=_getter,
        weights_initialized=lambda: initialized,
        set_weights_initialized=_set_initialized,
    )

    prepared = lifecycle.prepare_initial_weight_sync_state()
    assert prepared is not None
    with torch.no_grad():
        live.fill_(9.0)
    await lifecycle.push_prepared_weights(prepared, RolloutStats())

    assert getter_calls == 1
    assert initialized is True
    assert torch.equal(syncer.calls[0]["weight"], torch.tensor([1.0]))
    assert lifecycle.prepare_initial_weight_sync_state() is None
    assert getter_calls == 1


@pytest.mark.asyncio
async def test_failed_prepared_weight_push_does_not_publish_initialized_state() -> None:
    import torch

    from vrl.rollouts.orchestration.rollout_runtime import RolloutRuntimeCoordinator
    from vrl.rollouts.stats import RolloutStats

    runtime = _Runtime()
    initialized = False

    class _FailingSyncer(_Syncer):
        async def push(self, state_dict: dict[str, Any]) -> None:
            del state_dict
            raise RuntimeError("push failed")

    def _set_initialized(value: bool) -> None:
        nonlocal initialized
        initialized = bool(value)

    lifecycle = RolloutRuntimeCoordinator(
        collector=_Collector(runtime),
        strategy=_ParkingStrategy([]),
        training_state_getter=lambda: object(),
        weight_syncer=_FailingSyncer(runtime),
        sync_state_getter=lambda: {"weight": torch.ones(1)},
        weights_initialized=lambda: initialized,
        set_weights_initialized=_set_initialized,
    )
    prepared = lifecycle.prepare_initial_weight_sync_state()

    with pytest.raises(RuntimeError, match="push failed"):
        await lifecycle.push_prepared_weights(prepared, RolloutStats())

    assert initialized is False
    assert runtime.current_policy_version == 0


@pytest.mark.asyncio
async def test_strict_schedule_forwards_the_configured_reward_collection_arm() -> None:
    """Checks the acceptance arm reaches collection instead of being ignored."""
    import torch
    import torch.nn as nn

    from vrl.rollouts.orchestration import build_rollout_schedule
    from vrl.rollouts.orchestration.types import RewardCollectionMode
    from vrl.trainers.strategy import SingleProcessStrategy, TrainingMemoryState

    runtime = _Runtime()
    collector = _Collector(runtime)
    # Capable collector: the forced arm must restrict it back to per-group serial.
    collector.supports_reward_generation_overlap = True

    config = _schedule_config("strict_on_policy")
    config.reward_collection_mode = "per_group_serial"

    schedule = build_rollout_schedule(
        config,
        collector=collector,
        strategy=SingleProcessStrategy(),
        training_state_getter=lambda: TrainingMemoryState(
            model=nn.Linear(1, 1),
            ref_model=None,
            optimizer=None,
            ema=None,
            grad_scaler=None,
            device=torch.device("cpu"),
        ),
        weight_syncer=_Syncer(runtime),
        sync_state_getter=lambda: {"w": 1},
        weights_initialized=lambda: True,
        set_weights_initialized=lambda value: None,
    )

    assert schedule.reward_mode is RewardCollectionMode.PER_GROUP_SERIAL

    iteration = await schedule.next_iteration(["p0", "p1"], group_size=1)
    assert len(iteration.batches) == 2


@pytest.mark.asyncio
async def test_strict_schedule_defaults_to_capability_derived_arm() -> None:
    """Checks omitting the override keeps the production derivation."""
    import torch
    import torch.nn as nn

    from vrl.rollouts.orchestration import build_rollout_schedule
    from vrl.trainers.strategy import SingleProcessStrategy, TrainingMemoryState

    runtime = _Runtime()
    schedule = build_rollout_schedule(
        _schedule_config("strict_on_policy"),
        collector=_Collector(runtime),
        strategy=SingleProcessStrategy(),
        training_state_getter=lambda: TrainingMemoryState(
            model=nn.Linear(1, 1),
            ref_model=None,
            optimizer=None,
            ema=None,
            grad_scaler=None,
            device=torch.device("cpu"),
        ),
        weight_syncer=_Syncer(runtime),
        sync_state_getter=lambda: {"w": 1},
        weights_initialized=lambda: True,
        set_weights_initialized=lambda value: None,
    )

    assert schedule.reward_mode is None
