"""Streaming must retain optimizer-batch normalization before clipping/filtering.

Both sides are the online recipe's real trainer on tiny SANA (``real_trainer``):
one takes the full-batch step, the other streams the same prompts through the
global-std advantage spool. A prompt-keyed ``RewardFunction`` fixes every
reward, so the two updates see the same rollouts and the same scores.
"""

from __future__ import annotations

import asyncio
import random
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
import torch

from tests.trainers.online._helpers import TrainerBench, real_trainer
from vrl.rewards import RewardOutput, RewardSample
from vrl.rewards.base import RewardFunction
from vrl.scripts.common.online import _run_streaming_optimizer_update
from vrl.trainers.data.prompts import PromptExample

# Prompt "c" has equal rewards, so its group advantage is zero (filtered when
# drop_zero_advantage), and "b" saturates the 0.5 advantage clip under global std.
_REWARDS = {"a": [0.0, 1.0], "b": [0.0, 100.0], "c": [3.0, 3.0], "d": [-2.0, 5.0]}
_GLOBAL_STD = ("actor.ppo_epochs=1", "algorithm.global_std=true", "algorithm.adv_clip_max=0.5")
_TWO_COMPONENTS = (
    "reward.components={image_sharpness: 0.3, ocr: 0.7}",
    "algorithm.advantage_combine=normalized_sum",
)


class _PromptReward(RewardFunction):
    """Scores sample k of prompt p as ``_REWARDS[p][k]``.

    With ``components`` it also reports the raw component observations the
    normalized-sum estimator combines: ``ocr`` = r and ``image_sharpness`` = r^2.
    """

    def __init__(self, *, components: bool) -> None:
        self.components = components

    async def score_batch(self, samples: Sequence[RewardSample]) -> RewardOutput:
        seen: Counter[str] = Counter()
        scores = []
        for sample in samples:
            scores.append(_REWARDS[sample.prompt][seen[sample.prompt]])
            seen[sample.prompt] += 1
        if not self.components:
            return RewardOutput(scores=tuple(scores))
        return RewardOutput(
            scores=tuple(scores),
            components={
                "ocr": tuple(scores),
                "image_sharpness": tuple(value * value for value in scores),
            },
        )


def _record_advantages(monkeypatch, bench: TrainerBench) -> list[tuple[float, ...]]:
    """The advantages every real ``compute_loss`` call trains on."""

    seen: list[tuple[float, ...]] = []
    real = bench.trainer.algorithm.compute_loss

    def compute_loss(inputs):
        seen.append(tuple(inputs.advantages.detach().cpu().tolist()))
        return real(inputs)

    monkeypatch.setattr(bench.trainer.algorithm, "compute_loss", compute_loss)
    return seen


def _pushes(bench: TrainerBench) -> list[tuple[Any, int]]:
    return [args for event, args in bench.collector.trace.calls if event == "update_weights"]


def _output(bench: TrainerBench) -> Path:
    return Path(bench.trainer.config.output_dir)


# The three axes are independent in production (advantage filter, streaming
# split width, component combiner): a base row plus one flip per axis.
@pytest.mark.parametrize(
    ("drop_zero", "micro", "normalized_components"),
    [(False, 1, False), (True, 1, False), (False, 2, False), (False, 1, True)],
)
def test_streaming_matches_full_batch_advantages_gradients_and_adam(
    monkeypatch, tmp_path, drop_zero, micro, normalized_components
):
    prompts = [PromptExample(prompt=prompt) for prompt in _REWARDS]
    overrides = (
        *_GLOBAL_STD,
        f"actor.drop_zero_advantage={str(drop_zero).lower()}",
        "rollout.prompts_per_batch=4",
        *(_TWO_COMPONENTS if normalized_components else ()),
    )
    full = real_trainer(
        monkeypatch,
        tmp_path / "full",
        reward=_PromptReward(components=normalized_components),
        overrides=overrides,
    )
    # Each stack serves one snapshot: load the first policy before the second
    # stack installs its pipeline.
    asyncio.run(full.collector.collector.activate_generation_runtime())
    streamed = real_trainer(
        monkeypatch,
        tmp_path / "streamed",
        reward=_PromptReward(components=normalized_components),
        overrides=(*overrides, f"actor.prompts_per_collection={micro}"),
    )
    full_advantages = _record_advantages(monkeypatch, full)
    streamed_advantages = _record_advantages(monkeypatch, streamed)

    random.seed(0)
    full_metrics = asyncio.run(full.trainer.step(list(prompts)))
    random.seed(0)
    streamed_metrics = asyncio.run(
        _run_streaming_optimizer_update(
            streamed.trainer, list(prompts), batch_plan=streamed.trainer.config.batch_plan
        )
    )

    # Same request seeds, so both updates train on the same real rollouts.
    assert [r.sampling["seed"] for r in full.collector.trace.requests] == [
        r.sampling["seed"] for r in streamed.collector.trace.requests
    ]
    # Global normalization over the whole optimizer batch: identical advantages.
    assert full_advantages
    assert sorted(full_advantages) == sorted(streamed_advantages)
    assert streamed_metrics.reward_std == pytest.approx(full_metrics.reward_std)
    assert full_metrics.grad_norm > 0
    assert streamed_metrics.grad_norm == pytest.approx(full_metrics.grad_norm, rel=1e-5)
    # One update each: the initial push plus one post-train push of equal weights.
    full_pushes, streamed_pushes = _pushes(full), _pushes(streamed)
    assert [version for _, version in full_pushes] == [1, 2]
    assert [version for _, version in streamed_pushes] == [1, 2]
    for name, value in full_pushes[1][0].items():
        torch.testing.assert_close(streamed_pushes[1][0][name], value, rtol=1e-5, atol=1e-7)
    full_params, streamed_params = full.trainable_parameters(), streamed.trainable_parameters()
    for name, value in full_params.items():
        torch.testing.assert_close(streamed_params[name], value, rtol=1e-5, atol=1e-7)
    left = full.trainer._optimizer.state_dict()
    right = streamed.trainer._optimizer.state_dict()
    assert left["param_groups"] == right["param_groups"]
    assert left["state"].keys() == right["state"].keys()
    for key, state in left["state"].items():
        for name, value in state.items():
            torch.testing.assert_close(right["state"][key][name], value, rtol=1e-5, atol=1e-9)
    assert full.trainer.state.global_step == streamed.trainer.state.global_step == 1
    for bench in (full, streamed):
        assert not list(_output(bench).glob(".advantage-spool-*"))


@pytest.mark.parametrize("stage", ["collect", "replay"])
def test_spool_cleanup_on_failure(monkeypatch, tmp_path, stage):
    bench = real_trainer(
        monkeypatch,
        tmp_path,
        overrides=(
            *_GLOBAL_STD,
            "rollout.prompts_per_batch=2",
            "actor.prompts_per_collection=1",
        ),
    )
    trainer = bench.trainer
    output = _output(bench)
    calls: list[list[Any]] = []
    real_next_iteration = trainer.rollout_schedule.next_iteration

    async def next_iteration(prompts, **kwargs):
        calls.append(list(prompts))
        if len(calls) == 2 and stage == "collect":
            assert list(output.glob(".advantage-spool-*/*.pt"))
            raise RuntimeError("injected collection failure")
        return await real_next_iteration(prompts, **kwargs)

    def backward(*args, **kwargs):
        assert len(calls) == 2
        raise RuntimeError("injected replay failure")

    monkeypatch.setattr(trainer.rollout_schedule, "next_iteration", next_iteration)
    monkeypatch.setattr(trainer, "backward_on_training_batch", backward)

    with pytest.raises(RuntimeError, match="injected"):
        asyncio.run(
            _run_streaming_optimizer_update(
                trainer, ["a", "b"], batch_plan=trainer.config.batch_plan
            )
        )

    assert not list(output.glob(".advantage-spool-*"))
    assert trainer.state.global_step == 0
