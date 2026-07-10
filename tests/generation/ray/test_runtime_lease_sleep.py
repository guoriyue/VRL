"""Release-after-collect lease sleep/wake FSM (SPRINT_frozen_component_preservation, defect A).

A sleep-eligible lease (colocated trainer timeshare: rollout keeps its placement
bundle and only vacates the physical GPU) offloads its workers to host RAM on
release and wakes them on reacquire, instead of tearing them down and cold
reloading the frozen VAE / text-encoders from disk. A lease that must hand its
bundle to a reward worker still tears down. These exercise the lease state machine
directly with a fake inner runtime — no Ray cluster.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

from vrl.generation.launch_contract import GenerationRuntimeLaunchContract
from vrl.generation.ray.runtime import RayGenerationRuntime


class _FakeInner:
    """Records the lifecycle calls the lease drives on its inner runtime."""

    def __init__(self) -> None:
        self.calls: list[Any] = []
        self.current_policy_version: int | None = None

    async def sleep_workers(self) -> None:
        self.calls.append("sleep")

    async def wake_workers(self) -> None:
        self.calls.append("wake")

    async def shutdown(self) -> None:
        self.calls.append("shutdown")

    async def update_weights(self, state_ref: Any, version: int) -> None:
        self.calls.append(("update", state_ref, version))


def _lease_runtime(resources: Any = None) -> RayGenerationRuntime:
    # SimpleNamespace config: the factory only reads sync_trainable_state /
    # gpus_per_worker / resources. resources=None -> sleep_eligible derives False;
    # the FSM tests override state.sleep_eligible directly.
    config = SimpleNamespace(sync_trainable_state=False, gpus_per_worker=0.0, resources=resources)
    return RayGenerationRuntime.with_release_after_collect(
        config,
        SimpleNamespace(),  # launch_contract (never launched — runtime is injected)
        SimpleNamespace(),  # gatherer
        placement=SimpleNamespace(),
    )


# -- sleep-eligibility derivation --------------------------------------------


def _eligible(*, before_train: bool, before_reward: bool) -> bool:
    handoff = SimpleNamespace(
        release_rollout_before_train=before_train,
        release_rollout_before_reward=before_reward,
    )
    resources = SimpleNamespace(lifecycle=SimpleNamespace(handoff=handoff))
    return _lease_runtime(resources)._release_after_collect.sleep_eligible


def test_sleep_eligible_only_for_colocated_trainer_handoff() -> None:
    # Pure trainer timeshare -> sleep.
    assert _eligible(before_train=True, before_reward=False) is True
    # Reward takes the bundle -> tear down even if a trainer handoff also applies.
    assert _eligible(before_train=True, before_reward=True) is False
    assert _eligible(before_train=False, before_reward=True) is False
    # No resolved topology (hand-built specs) -> conservative teardown.
    assert _lease_runtime(None)._release_after_collect.sleep_eligible is False


def _contract_after_lease(*, before_train: bool, before_reward: bool) -> Any:
    handoff = SimpleNamespace(
        release_rollout_before_train=before_train,
        release_rollout_before_reward=before_reward,
    )
    config = SimpleNamespace(
        sync_trainable_state=False,
        gpus_per_worker=0.0,
        resources=SimpleNamespace(lifecycle=SimpleNamespace(handoff=handoff)),
    )
    contract = GenerationRuntimeLaunchContract(
        family="sd3_5",
        task="t2i",
        policy_version=1,
        runtime_builder="x:y",
        executor_cls="x:z",
    )
    runtime = RayGenerationRuntime.with_release_after_collect(
        config, contract, SimpleNamespace(), placement=SimpleNamespace(),
    )
    return runtime._release_after_collect.launch_contract


def test_sleep_eligible_lease_stamps_contract_for_cumem_offload() -> None:
    # Sleep-eligible -> the worker is told to pool its model for cumem sleep/wake.
    contract = _contract_after_lease(before_train=True, before_reward=False)
    assert contract.extra.get("sleep_offload") is True


def test_teardown_lease_leaves_contract_unstamped() -> None:
    # Reward-handoff lease tears down -> no cumem pooling requested.
    contract = _contract_after_lease(before_train=False, before_reward=True)
    assert "sleep_offload" not in contract.extra


# -- lease FSM ----------------------------------------------------------------


def test_sleep_eligible_release_offloads_and_keeps_runtime() -> None:
    runtime = _lease_runtime()
    state = runtime._release_after_collect
    assert state is not None
    state.sleep_eligible = True
    inner = _FakeInner()
    state.runtime = inner

    asyncio.run(runtime.release())

    assert inner.calls == ["sleep"]  # offloaded, not torn down
    assert state.runtime is inner  # workers stay alive
    assert state.asleep is True


def test_ensure_runtime_wakes_and_replays_latest_state() -> None:
    runtime = _lease_runtime()
    state = runtime._release_after_collect
    assert state is not None
    state.sleep_eligible = True
    state.asleep = True
    inner = _FakeInner()
    state.runtime = inner
    state.last_state = "W"
    runtime.current_policy_version = 5

    reacquired = asyncio.run(runtime._ensure_runtime())

    assert reacquired is inner  # same workers, no relaunch
    # Wake from host RAM, then replay the newest trainable state (same refresh
    # contract the cold-launch path applies after a teardown).
    assert inner.calls == ["wake", ("update", "W", 5)]
    assert state.asleep is False


def test_update_weights_while_asleep_records_but_does_not_push() -> None:
    runtime = _lease_runtime()
    state = runtime._release_after_collect
    assert state is not None
    runtime.weight_sync = object()  # lease branch needs a syncer present
    state.sleep_eligible = True
    state.asleep = True
    inner = _FakeInner()
    state.runtime = inner

    asyncio.run(runtime.update_weights("W2", 7))

    # Recorded for replay-on-wake; NOT pushed onto the slept (CPU-resident) worker.
    assert state.last_state == "W2"
    assert runtime.current_policy_version == 7
    assert inner.calls == []


def test_non_sleep_lease_release_still_tears_down() -> None:
    """Regression: a reward-handoff lease keeps the level-2 teardown path."""
    runtime = _lease_runtime()
    state = runtime._release_after_collect
    assert state is not None
    state.sleep_eligible = False
    inner = _FakeInner()
    state.runtime = inner

    asyncio.run(runtime.release())

    assert inner.calls == ["shutdown"]
    assert state.runtime is None
    assert state.asleep is False


def test_sleep_eligible_shutdown_terminates_active_runtime() -> None:
    runtime = _lease_runtime()
    state = runtime._release_after_collect
    assert state is not None
    state.sleep_eligible = True
    inner = _FakeInner()
    state.runtime = inner

    asyncio.run(runtime.shutdown())

    assert inner.calls == ["shutdown"]
    assert state.runtime is None
    assert state.asleep is False


def test_sleep_eligible_shutdown_terminates_asleep_runtime_once() -> None:
    runtime = _lease_runtime()
    state = runtime._release_after_collect
    assert state is not None
    state.sleep_eligible = True
    state.asleep = True
    inner = _FakeInner()
    state.runtime = inner

    asyncio.run(runtime.shutdown())
    asyncio.run(runtime.shutdown())

    assert inner.calls == ["shutdown"]
    assert state.runtime is None
    assert state.asleep is False
