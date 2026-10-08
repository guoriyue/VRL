"""The in-process generation runtime drives the real family executor end to end."""

from __future__ import annotations

import asyncio

import torch

from tests.scripts.eval.fixtures import tiny_sana_stack
from vrl.generation.protocols import GenerationRuntime
from vrl.rewards.functions.registry import MultiReward
from vrl.rewards.runtime import RewardFunctionRuntime
from vrl.rollouts.collector import RolloutCollector
from vrl.rollouts.stats import RolloutStats
from vrl.trainers.data.prompts import PromptExample


def test_runtime_satisfies_the_generation_protocol(monkeypatch, tmp_path) -> None:
    runtime = tiny_sana_stack(monkeypatch, tmp_path).runtime

    assert isinstance(runtime, GenerationRuntime)


def test_collector_generates_and_scores_through_the_real_family_executor(
    monkeypatch, tmp_path
) -> None:
    """A real RolloutCollector over the tiny SANA family: request building, the
    generic denoise executor, batch merging and a CPU reward, no doubles."""

    stack = tiny_sana_stack(monkeypatch, tmp_path)
    collector = RolloutCollector.from_family(
        stack.family,
        reward_runtime=RewardFunctionRuntime(
            MultiReward.from_dict({"image_sharpness": 1.0}, device="cpu"),
        ),
        config=stack.collector_config(),
        generation_runtime=stack.runtime,
    )
    stats = RolloutStats()

    async def collect():
        await collector.activate_generation_runtime()
        try:
            return await collector.prepare_training_batches(
                prompts=[PromptExample(prompt="a cat")],
                group_size=2,
                runtime_debug=False,
                policy_version=0,
                stats=stats,
            )
        finally:
            await collector.shutdown()

    batches = asyncio.run(collect())

    assert len(batches) == 1
    batch = batches[0]
    assert batch.rewards.shape == (2,)
    assert torch.isfinite(batch.rewards).all()
    assert batch.group_ids.tolist() == [0, 0]
    assert batch.trajectory is not None
    assert [row.prompt for row in batch.trajectory.sample_rows] == ["a cat", "a cat"]
    assert stats.counters["collect.sample_count"] == 2.0
