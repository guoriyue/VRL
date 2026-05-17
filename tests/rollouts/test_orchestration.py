"""Tests for RL rollout orchestration schedules."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest


def _schedule_config(mode: str) -> SimpleNamespace:
    return SimpleNamespace(
        mode=mode,
        max_pending_rollouts=1,
        require_separate_gpus=True,
        weight_sync_barrier="before_sync",
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
        observations=torch.zeros(batch_size, 1, 1),
        actions=torch.zeros(batch_size, 1, 1),
        rewards=torch.arange(batch_size, dtype=torch.float32),
        dones=torch.ones(batch_size, dtype=torch.bool),
        group_ids=group_ids,
        prompts=[prompt for prompt in prompts for _ in range(group_size)],
    )


class _Runtime:
    def __init__(self) -> None:
        self.current_policy_version = 0
        self.requires_driver_model_offload = False
        self.config = SimpleNamespace(
            allow_driver_gpu_overlap=False,
            resources=SimpleNamespace(colocated=False),
        )


class _Syncer:
    def __init__(self, runtime: _Runtime) -> None:
        self.runtime = runtime
        self.calls: list[dict[str, Any]] = []

    async def push(self, state_dict: dict[str, Any]) -> None:
        self.calls.append(dict(state_dict))
        self.runtime.current_policy_version += 1

    async def pull(self) -> dict[str, Any]:
        return dict(self.calls[-1])


class _Collector:
    def __init__(self, runtime: _Runtime) -> None:
        self.runtime = runtime
        self.calls: list[dict[str, Any]] = []

    async def collect(self, prompts, **kwargs):
        prompts = list(prompts)
        self.calls.append({"prompts": prompts, **dict(kwargs)})
        return _batch(prompts, int(kwargs.get("group_size", 1)))

    async def release_runtime_memory(self) -> None:
        self.calls.append({"release_runtime_memory": True})


@pytest.mark.asyncio
async def test_strict_schedule_collects_and_syncs_with_rollout_metadata() -> None:
    import torch.nn as nn

    from vrl.rollouts.orchestration import RolloutScheduleMode, build_rollout_schedule

    runtime = _Runtime()
    collector = _Collector(runtime)
    syncer = _Syncer(runtime)
    initialized = False

    def _set_initialized(value: bool) -> None:
        nonlocal initialized
        initialized = bool(value)

    schedule = build_rollout_schedule(
        _schedule_config("strict_on_policy"),
        collector=collector,
        model=nn.Linear(1, 1),
        device=__import__("torch").device("cpu"),
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
    assert len(iteration.batches) == 2
    assert iteration.batches[0].context["rollout_id"] == 0
    assert iteration.batches[0].context["rollout_policy_version"] == 1
    assert iteration.batches[0].context["schedule_mode"] == "strict_on_policy"
    assert len(syncer.calls) == 2
    assert runtime.current_policy_version == 2


@pytest.mark.asyncio
async def test_one_batch_overlap_keeps_completed_pending_rollout_for_next_step() -> None:
    import asyncio

    import torch.nn as nn

    from vrl.rollouts.orchestration import build_rollout_schedule

    runtime = _Runtime()
    syncer = _Syncer(runtime)
    starts: list[tuple[int, int]] = []

    class _SlowCollector(_Collector):
        async def collect(self, prompts, **kwargs):
            starts.append((len(starts), int(kwargs["policy_version"])))
            await asyncio.sleep(0.01)
            return await super().collect(prompts, **kwargs)

    collector = _SlowCollector(runtime)
    initialized = False

    def _set_initialized(value: bool) -> None:
        nonlocal initialized
        initialized = bool(value)

    schedule = build_rollout_schedule(
        _schedule_config("one_batch_overlap"),
        collector=collector,
        model=nn.Linear(1, 1),
        device=__import__("torch").device("cpu"),
        weight_syncer=syncer,
        sync_state_getter=lambda: {"w": 1},
        weights_initialized=lambda: initialized,
        set_weights_initialized=_set_initialized,
    )

    first = await schedule.next_iteration(["p0"], group_size=2)

    assert first.rollout_id == 0
    assert first.policy_version == 1
    assert starts[:2] == [(0, 1), (1, 1)]

    await schedule.after_train_step()
    second = await schedule.next_iteration(["p0"], group_size=2)

    assert second.rollout_id == 1
    assert second.policy_version == 1
    assert second.batches[0].context["rollout_id"] == 1
    assert second.batches[0].context["schedule_mode"] == "one_batch_overlap"
    assert starts[2] == (2, 2)


@pytest.mark.asyncio
async def test_one_batch_overlap_rejects_colocated_runtime() -> None:
    import torch.nn as nn

    from vrl.rollouts.orchestration import build_rollout_schedule

    runtime = _Runtime()
    runtime.config.resources.colocated = True
    collector = _Collector(runtime)

    schedule = build_rollout_schedule(
        _schedule_config("one_batch_overlap"),
        collector=collector,
        model=nn.Linear(1, 1),
        device=__import__("torch").device("cpu"),
        weight_syncer=None,
        sync_state_getter=None,
        weights_initialized=lambda: True,
        set_weights_initialized=lambda value: None,
    )

    with pytest.raises(RuntimeError, match="separate trainer and rollout GPU"):
        await schedule.next_iteration(["p0"], group_size=1)
