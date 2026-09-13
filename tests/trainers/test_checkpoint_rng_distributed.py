"""Rank-local random streams survive primary-only checkpoint publication."""

from __future__ import annotations

import os
import random
from datetime import timedelta

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from vrl.trainers.checkpointing import (
    TrainingCheckpoint,
    capture_rng_state,
    restore_rng_state,
    save_training_checkpoint,
)
from vrl.trainers.distributed import (
    DistributedTrainingContext,
    init_training_process_group,
    shutdown_training_process_group,
)
from vrl.trainers.strategy import DDPStrategy


def _draw(generator, *, cuda=False):
    return (
        torch.rand(8),
        torch.rand(8, generator=generator),
        random.random(),
        np.random.rand(8),
        [
            torch.rand(8, device=f"cuda:{device}").cpu()
            for device in range(torch.cuda.device_count())
        ]
        if cuda
        else [],
    )


def _rng_rank(rank, rendezvous, output, backend="gloo", world_size=2):
    from tests.trainers.test_checkpointing import UNIT_IDENTITY, _Bundle, _Trainer

    context = DistributedTrainingContext(
        strategy="ddp",
        rank=rank,
        world_size=world_size,
        device=torch.device(f"cuda:{rank}" if backend == "nccl" else "cpu"),
    )
    if backend == "nccl":
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(rendezvous)
        init_training_process_group(context, backend=backend)
    else:
        dist.init_process_group(
            "gloo",
            init_method=rendezvous,
            rank=rank,
            world_size=world_size,
            timeout=timedelta(seconds=60),
        )
    try:
        strategy = DDPStrategy(
            context,
            find_unused_parameters=False,
        )
        bundle = _Bundle()
        torch.manual_seed(100 + rank)
        random.seed(200 + rank)
        np.random.seed(300 + rank)
        generator = torch.Generator().manual_seed(400 + rank)
        local = capture_rng_state(prompt_generator=generator)
        expected = _draw(generator, cuda=backend == "nccl")
        save_training_checkpoint(
            output,
            trainer=_Trainer(),
            bundle=bundle,
            family="unit",
            progress={"next_epoch": 1},
            model_identity=UNIT_IDENTITY,
            rng_state=local,
            strategy=strategy,
        )
        strategy.barrier()
        checkpoint = TrainingCheckpoint.load(output)
        assert checkpoint.rng_state["world_size"] == world_size
        rank_states = checkpoint.rng_state["by_rank"]
        assert not torch.equal(rank_states[0]["torch"], rank_states[1]["torch"])
        if backend == "nccl":
            assert len(rank_states[rank]["cuda"]) == torch.cuda.device_count()
            assert not torch.equal(rank_states[0]["cuda"][rank], rank_states[1]["cuda"][rank])
        restore_rng_state(
            checkpoint.rng_state, rank=rank, world_size=world_size, prompt_generator=generator
        )
        actual = _draw(generator, cuda=backend == "nccl")
        assert torch.equal(expected[0], actual[0])
        assert torch.equal(expected[1], actual[1])
        assert expected[2] == actual[2]
        np.testing.assert_array_equal(expected[3], actual[3])
        assert all(
            torch.equal(left, right) for left, right in zip(expected[4], actual[4], strict=True)
        )
    finally:
        shutdown_training_process_group()


def test_two_rank_rng_checkpoint_round_trip(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    mp.spawn(
        _rng_rank,
        args=((tmp_path / "rendezvous").as_uri(), str(tmp_path / "checkpoint")),
        nprocs=2,
        join=True,
    )


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA devices")
def test_two_rank_cuda_rng_checkpoint_round_trip(tmp_path):
    from tests.trainers._strategy_policies import free_port

    mp.spawn(
        _rng_rank, args=(free_port(), str(tmp_path / "checkpoint"), "nccl"), nprocs=2, join=True
    )


@pytest.mark.skipif(torch.cuda.device_count() < 4, reason="requires four CUDA devices")
def test_four_rank_cuda_rng_checkpoint_round_trip(tmp_path):
    from tests.trainers._strategy_policies import free_port

    mp.spawn(
        _rng_rank, args=(free_port(), str(tmp_path / "checkpoint"), "nccl", 4), nprocs=4, join=True
    )


@pytest.mark.parametrize("state", [None, {}, {"generators": {}}])
def test_strict_multi_rank_restore_rejects_legacy_rng(state):
    with pytest.raises(ValueError, match="no per-rank RNG"):
        restore_rng_state(state, rank=1, world_size=2)


@pytest.mark.parametrize("named", [None, {}, {"probe": torch.Generator().get_state()}])
def test_strict_restore_rejects_missing_requested_generator_before_mutation(named):
    before = torch.get_rng_state().clone()
    prompt_generator = torch.Generator().manual_seed(999)
    prompt_before = prompt_generator.get_state().clone()
    saved = {"torch": torch.Generator().manual_seed(12).get_state(), "generators": named}

    with pytest.raises(ValueError, match="missing requested generators: prompt_generator"):
        restore_rng_state(saved, prompt_generator=prompt_generator)

    assert torch.equal(torch.get_rng_state(), before)
    assert torch.equal(prompt_generator.get_state(), prompt_before)


@pytest.mark.parametrize("state", [None, {}])
def test_strict_restore_requires_requested_generator_even_without_rng_tree(state):
    with pytest.raises(ValueError, match="missing requested generators: prompt_generator"):
        restore_rng_state(state, prompt_generator=torch.Generator())


@pytest.mark.parametrize("rank", range(4))
def test_strict_four_rank_restore_checks_selected_rank_generator(rank):
    states = [
        capture_rng_state(prompt_generator=torch.Generator().manual_seed(20)) for _ in range(4)
    ]
    states[rank]["generators"] = {"probe": torch.Generator().get_state()}
    with pytest.raises(ValueError, match="missing requested generators: prompt_generator"):
        restore_rng_state(
            {"world_size": 4, "by_rank": states},
            rank=rank,
            world_size=4,
            prompt_generator=torch.Generator(),
        )


def test_nonstrict_missing_generator_warns_and_preserves_missing_stream(caplog):
    prompt_generator = torch.Generator().manual_seed(99)
    before = prompt_generator.get_state().clone()
    other = torch.Generator().manual_seed(44)
    saved_other = torch.Generator().manual_seed(22).get_state()
    restore_rng_state(
        {"generators": {"other": saved_other}},
        strict=False,
        prompt_generator=prompt_generator,
        other=other,
    )
    assert torch.equal(prompt_generator.get_state(), before)
    assert torch.equal(other.get_state(), saved_other)
    assert "missing requested generators: prompt_generator" in caplog.text


def test_nonstrict_legacy_rng_restore_warns(monkeypatch):
    from vrl.trainers import checkpointing

    messages = []
    monkeypatch.setattr(checkpointing.logger, "warning", messages.append)
    restore_rng_state({}, rank=1, world_size=2, strict=False)
    assert len(messages) == 1
    assert "not equivalent" in messages[0]


@pytest.mark.parametrize(
    "state",
    [
        {"world_size": 1, "by_rank": [{"generators": {}}]},
        {"world_size": 2, "by_rank": [{"generators": {}}]},
        {"world_size": 2, "by_rank": [{}, {}]},
        {"world_size": True, "by_rank": [{"generators": {}}, {"generators": {}}]},
    ],
)
def test_invalid_rank_rng_rejected_even_nonstrict(state):
    before = torch.get_rng_state()
    with pytest.raises(ValueError, match="world_size"):
        restore_rng_state(state, rank=0, world_size=2, strict=False)
    assert torch.equal(before, torch.get_rng_state())


@pytest.mark.parametrize(("rank", "world_size"), [(-1, 2), (2, 2), (0, 0), (True, 2)])
def test_invalid_rng_restore_rank(rank, world_size):
    with pytest.raises(ValueError, match="valid rank"):
        restore_rng_state({}, rank=rank, world_size=world_size)
