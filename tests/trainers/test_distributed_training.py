"""Tests for the distributed training context (vrl/trainers/distributed.py)."""

from __future__ import annotations

import pytest
import torch
from omegaconf import OmegaConf

from vrl.config.schema import parse_config
from vrl.trainers.distributed import (
    DistributedTrainingContext,
)


def _cfg(training: dict | None = None):
    body: dict = {} if training is None else {"distributed": {"training": training}}
    return OmegaConf.create(body)


# ── single_process ────────────────────────────────────────────────────────────


def test_single_process_is_rank0_world1_and_keeps_device() -> None:
    ctx = DistributedTrainingContext.from_root(
        parse_config(_cfg({"strategy": "single_process"})),
        device=torch.device("cpu"),
        env={"RANK": "3", "WORLD_SIZE": "8"},  # env ignored for single_process
    )
    assert ctx.strategy == "single_process"
    assert (ctx.distributed, ctx.rank, ctx.world_size) == (False, 0, 1)
    assert ctx.is_primary is True
    assert ctx.device == torch.device("cpu")


def test_single_process_is_default_when_training_absent() -> None:
    ctx = DistributedTrainingContext.from_root(
        parse_config(_cfg()), device=torch.device("cpu"), env={}
    )
    assert ctx.strategy == "single_process"
    assert ctx.is_primary is True


# ── fsdp env parsing ──────────────────────────────────────────────────────────


def test_fsdp_parses_torchrun_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    ctx = DistributedTrainingContext.from_root(
        parse_config(_cfg({"strategy": "fsdp", "num_nodes": 1, "gpus_per_node": 2})),
        device=torch.device("cpu"),
        env={"RANK": "1", "LOCAL_RANK": "1", "WORLD_SIZE": "2"},
    )
    assert ctx.distributed is True
    assert (ctx.rank, ctx.world_size) == (1, 2)
    assert ctx.is_primary is False
    assert ctx.device == torch.device("cuda:1")


def test_fsdp_rank0_is_primary() -> None:
    ctx = DistributedTrainingContext.from_root(
        parse_config(_cfg({"strategy": "fsdp", "gpus_per_node": 2})),
        device=torch.device("cpu"),
        env={"RANK": "0", "LOCAL_RANK": "0", "WORLD_SIZE": "2"},
    )
    assert ctx.is_primary is True


def test_fsdp_missing_env_fails_fast_listing_keys() -> None:
    with pytest.raises(ValueError, match=r"RANK.*LOCAL_RANK.*WORLD_SIZE"):
        DistributedTrainingContext.from_root(
            parse_config(_cfg({"strategy": "fsdp"})), device=torch.device("cpu"), env={}
        )


def test_fsdp_world_size_must_match_topology() -> None:
    with pytest.raises(ValueError, match=r"WORLD_SIZE=4 must equal"):
        DistributedTrainingContext.from_root(
            parse_config(_cfg({"strategy": "fsdp", "num_nodes": 1, "gpus_per_node": 2})),
            device=torch.device("cpu"),
            env={"RANK": "0", "LOCAL_RANK": "0", "WORLD_SIZE": "4"},
        )


def test_fsdp_local_rank_must_be_in_range() -> None:
    with pytest.raises(ValueError, match=r"LOCAL_RANK=2 is out of range"):
        DistributedTrainingContext.from_root(
            parse_config(_cfg({"strategy": "fsdp", "gpus_per_node": 2})),
            device=torch.device("cpu"),
            env={"RANK": "0", "LOCAL_RANK": "2", "WORLD_SIZE": "2"},
        )


# Execution wiring lives outside this resolver: build_strategy constructs the
# FSDP/DDP adapter, while the online lifecycle tests cover the supported
# symmetric-colocated multi-rank orchestration. This module only resolves and
# validates process identity and the rank-local device.
