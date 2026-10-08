"""Real four-rank FSDP2 global-std streaming over a native BF16 base and FP32 adapters.

Every process builds the online recipe's trainer on tiny SANA: a LoRA policy
whose frozen base trains in BF16 and whose adapters stay FP32, the real GRPO /
SDE evaluator pair, the real in-process rollout and a prompt-keyed reward. The
reference process trains all eight prompt groups with ``SingleProcessStrategy``;
each of four gloo ranks trains two of them under a real ``FSDPStrategy`` that
shards only the trainable adapters. Two streamed updates, each followed by a
trainer park/restore, must leave every rank with the reference's gradient
norms, adapter weights and Adam moments.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.tensor import DTensor

from tests.rollouts.collector._helpers import real_collector
from tests.trainers._strategy_policies import free_port
from vrl.rewards import RewardOutput, RewardSample
from vrl.rewards.base import RewardFunction
from vrl.scripts.common.factory import AlgorithmEvaluatorPair
from vrl.scripts.common.online import _run_streaming_optimizer_update
from vrl.trainers.distributed import DistributedTrainingContext
from vrl.trainers.online.trainer import OnlineTrainer
from vrl.trainers.strategy import FSDPStrategy, SingleProcessStrategy
from vrl.trainers.weight_sync import RayRuntimeWeightSyncer

_WORLD = 4
_PROMPTS = [str(index) for index in range(2 * _WORLD)]
# LoRA adapters over a BF16 base, global-std streaming, and one fixed request
# seed so a prompt's rollout is the same in whichever process generates it.
_OVERRIDES = (
    "model.use_lora=true",
    "precision.training.dtype=bf16",
    "actor.ppo_epochs=1",
    "actor.drop_zero_advantage=true",
    "algorithm.global_std=true",
    "algorithm.adv_clip_max=0.5",
    "rollout.n_samples_per_prompt=4",
    "rollout.samples_per_generation_batch=4",
    "actor.training_microbatch_size=4",
    "sampling.seed=7",
)


class _PromptReward(RewardFunction):
    """Prompt ``p`` scores ``[0, 1, 3, 7] * (p + 1)``; ``uneven`` flattens prompt 0."""

    def __init__(self, *, uneven: bool) -> None:
        self.uneven = uneven

    async def score_batch(self, samples: Sequence[RewardSample]) -> RewardOutput:
        seen: dict[str, int] = {}
        scores = []
        for sample in samples:
            index = seen.get(sample.prompt, 0)
            seen[sample.prompt] = index + 1
            prompt = int(sample.prompt)
            scores.append(
                2.0 if self.uneven and prompt == 0 else (0, 1, 3, 7)[index] * (prompt + 1)
            )
        return RewardOutput(scores=tuple(float(score) for score in scores))


def _trainer(
    monkeypatch: pytest.MonkeyPatch,
    root: Path,
    *,
    strategy: Any,
    prompts: int,
    uneven: bool,
) -> OnlineTrainer:
    """The online recipe's trainer wiring on tiny SANA under ``strategy``."""

    bench = real_collector(
        monkeypatch,
        root,
        reward=_PromptReward(uneven=uneven),
        overrides=(
            *_OVERRIDES,
            f"rollout.prompts_per_batch={prompts}",
            f"actor.prompts_per_collection={max(prompts // 4, 1)}",
        ),
    )
    stack = bench.stack
    built = stack.resolved.built
    # Same adapter initialization in every process; a rank that does not
    # materialize weights receives the primary rank's through the strategy.
    torch.manual_seed(0)
    bundle = stack.replay.materialize(
        context="fsdp streaming equivalence",
        materialize_weights=strategy.materialize_weights,
    )
    pair = AlgorithmEvaluatorPair.from_configs(
        family_entry=stack.family,
        built=built,
        collector_config=stack.collector_config(),
        scheduler=getattr(bundle, "scheduler", None),
    )
    return OnlineTrainer(
        algorithm=pair.algorithm,
        collector=bench.collector,
        evaluator=pair.evaluator,
        model=bundle.model,
        ref_model=bundle.model,
        weight_syncer=RayRuntimeWeightSyncer(bench.runtime),
        sync_state_getter=lambda: strategy.export_rollout_state(bundle),
        config=built.trainer,
        device=torch.device("cpu"),
        strategy=strategy,
    )


def _adapters(trainer: OnlineTrainer) -> dict[str, torch.nn.Parameter]:
    transformer = trainer.model.trainable_modules["transformer"]
    return {
        name: parameter
        for name, parameter in transformer.named_parameters()
        if parameter.requires_grad
    }


def _full(value: Any) -> torch.Tensor:
    return value.full_tensor() if isinstance(value, DTensor) else value


async def _updates(trainer: OnlineTrainer, prompts: list[str]) -> list[float]:
    norms = []
    for _ in range(2):
        metric = await _run_streaming_optimizer_update(
            trainer, prompts, batch_plan=trainer.config.batch_plan
        )
        norms.append(metric.grad_norm)
        state = trainer._training_memory_state()
        trainer._strategy.park_training_state(state)
        trainer._strategy.restore_training_state(state)
    return norms


def _result(trainer: OnlineTrainer, norms: list[float], output: Path) -> dict[str, Any]:
    assert trainer.state.global_step == 2
    assert not list(output.glob(".advantage-spool-*"))
    adapters = _adapters(trainer)
    moments = {}
    for name, parameter in adapters.items():
        for key, value in trainer._optimizer.state[parameter].items():
            if isinstance(value, torch.Tensor) and value.ndim > 0:
                moments[f"{name}.{key}"] = _full(value).detach().tolist()
    # Plain lists: a tensor queued from a spawned rank dies with its process.
    return {
        "norms": norms,
        "weights": {name: _full(value).detach().tolist() for name, value in adapters.items()},
        "moments": moments,
    }


def _rank(rank: int, port: int, root: str, uneven: bool, queue: Any) -> None:
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    torch.set_num_threads(1)
    dist.init_process_group("gloo", rank=rank, world_size=_WORLD)
    monkeypatch = pytest.MonkeyPatch()
    try:
        # A spawned rank escapes the conftest CUDA pin; this test is CPU-only.
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)
        strategy = FSDPStrategy(
            DistributedTrainingContext("fsdp", rank, _WORLD, torch.device("cpu")),
            mesh_dims=["dp_shard"],
            precision_policy="none",
            reshard_after_forward=True,
            cpu_offload=False,
            shard_trainable_only=True,
        )
        output = Path(root) / f"rank-{rank}"
        trainer = _trainer(monkeypatch, output, strategy=strategy, prompts=2, uneven=uneven)
        # Only the FP32 adapters are sharded; the frozen BF16 base stays local.
        transformer = trainer.model.trainable_modules["transformer"]
        dtypes = {
            (p.requires_grad, p.dtype, isinstance(p, DTensor)) for p in transformer.parameters()
        }
        assert dtypes == {(True, torch.float32, True), (False, torch.bfloat16, False)}
        prompts = _PROMPTS[2 * rank : 2 * rank + 2]
        run_output = Path(trainer.config.output_dir)
        if uneven:
            with pytest.raises(ValueError, match="uneven filtering"):
                asyncio.run(_updates(trainer, prompts))
            assert not list(run_output.glob(".advantage-spool-*"))
            queue.put((rank, True))
        else:
            norms = asyncio.run(_updates(trainer, prompts))
            queue.put((rank, _result(trainer, norms, run_output)))
    except BaseException as error:
        queue.put((rank, f"{type(error).__name__}: {error}"))
        raise
    finally:
        monkeypatch.undo()
        dist.destroy_process_group()


def _reference(monkeypatch: pytest.MonkeyPatch, root: Path) -> dict[str, Any]:
    trainer = _trainer(
        monkeypatch,
        root,
        strategy=SingleProcessStrategy(),
        prompts=len(_PROMPTS),
        uneven=False,
    )
    norms = asyncio.run(_updates(trainer, list(_PROMPTS)))
    return _result(trainer, norms, Path(trainer.config.output_dir))


@pytest.mark.parametrize("uneven", [False, True])
def test_four_rank_streaming_semantics(monkeypatch, tmp_path, uneven):
    reference = None if uneven else _reference(monkeypatch, tmp_path / "reference")
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    port = free_port()
    processes = [
        ctx.Process(target=_rank, args=(rank, port, str(tmp_path), uneven, queue))
        for rank in range(_WORLD)
    ]
    try:
        for process in processes:
            process.start()
        results = dict(queue.get(timeout=300) for _ in range(_WORLD))
        for rank, result in results.items():
            assert not isinstance(result, str), (rank, result)
        for process in processes:
            process.join(timeout=30)
            assert process.exitcode == 0
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=10)
    for result in results.values():
        if uneven:
            assert result is True
            continue
        assert all(norm > 0 for norm in result["norms"])
        torch.testing.assert_close(
            torch.tensor(result["norms"]),
            torch.tensor(reference["norms"]),
            rtol=1e-4,
            atol=1e-6,
        )
        # Rank gradients are summed in a different order than the reference's
        # sequential accumulation; Adam's per-element normalization amplifies that
        # fp32 roundoff on near-zero gradients. A tenth of one Adam step (lr 3e-4)
        # bounds it, far below any sharding or normalization error.
        assert result["weights"].keys() == reference["weights"].keys()
        for name, value in reference["weights"].items():
            torch.testing.assert_close(
                torch.tensor(result["weights"][name]), torch.tensor(value), rtol=0, atol=3e-5
            )
        # The second update's rollouts come from the first update's weights, so
        # BF16-base roundoff reaches the second gradient (measured ~1e-3 relative).
        # A sharding, rank-averaging or global-std error is O(1) relative.
        assert result["moments"].keys() == reference["moments"].keys()
        for name, value in reference["moments"].items():
            expected = torch.tensor(value)
            actual = torch.tensor(result["moments"][name])
            error = (actual - expected).norm() / expected.norm().clamp_min(1e-30)
            assert float(error) < 1e-2, (name, float(error))
