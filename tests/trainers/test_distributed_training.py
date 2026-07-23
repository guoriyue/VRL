"""Tests for the distributed training context (vrl/trainers/distributed.py)."""

from __future__ import annotations

import pytest
import torch
from omegaconf import OmegaConf

from vrl.trainers.distributed import (
    DistributedTrainingContext,
    resolve_training_context,
)


def _cfg(training: dict | None = None):
    body: dict = {} if training is None else {"distributed": {"training": training}}
    return OmegaConf.create(body)


# ── single_process ────────────────────────────────────────────────────────────


def test_single_process_is_rank0_world1_and_keeps_device() -> None:
    ctx = resolve_training_context(
        _cfg({"strategy": "single_process"}),
        device=torch.device("cpu"),
        env={"RANK": "3", "WORLD_SIZE": "8"},  # env ignored for single_process
    )
    assert ctx.strategy == "single_process"
    assert (ctx.distributed, ctx.rank, ctx.local_rank, ctx.world_size) == (False, 0, 0, 1)
    assert ctx.is_primary is True
    assert ctx.device == torch.device("cpu")


def test_single_process_is_default_when_training_absent() -> None:
    ctx = resolve_training_context(_cfg(), device=torch.device("cpu"), env={})
    assert ctx.strategy == "single_process"
    assert ctx.is_primary is True


# ── fsdp env parsing ──────────────────────────────────────────────────────────


def test_fsdp_parses_torchrun_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    ctx = resolve_training_context(
        _cfg({"strategy": "fsdp", "num_nodes": 1, "gpus_per_node": 2}),
        device=torch.device("cpu"),
        env={"RANK": "1", "LOCAL_RANK": "1", "WORLD_SIZE": "2"},
    )
    assert ctx.distributed is True
    assert (ctx.rank, ctx.local_rank, ctx.world_size) == (1, 1, 2)
    assert ctx.is_primary is False
    assert ctx.device == torch.device("cuda:1")


@pytest.mark.parametrize(
    ("field", "value"),
    [("distributed", False), ("is_primary", False)],
)
def test_context_rejects_stored_derived_flags(field: str, value: bool) -> None:
    kwargs = {
        "strategy": "ddp",
        "rank": 1,
        "local_rank": 0,
        "world_size": 2,
        "device": torch.device("cpu"),
        field: value,
    }

    with pytest.raises(TypeError, match=field):
        DistributedTrainingContext(**kwargs)


def test_fsdp_rank0_is_primary() -> None:
    ctx = resolve_training_context(
        _cfg({"strategy": "fsdp", "gpus_per_node": 2}),
        device=torch.device("cpu"),
        env={"RANK": "0", "LOCAL_RANK": "0", "WORLD_SIZE": "2"},
    )
    assert ctx.is_primary is True


def test_fsdp_missing_env_fails_fast_listing_keys() -> None:
    with pytest.raises(ValueError, match=r"RANK.*LOCAL_RANK.*WORLD_SIZE"):
        resolve_training_context(_cfg({"strategy": "fsdp"}), device=torch.device("cpu"), env={})


def test_fsdp_world_size_must_match_topology() -> None:
    with pytest.raises(ValueError, match=r"WORLD_SIZE=4 must equal"):
        resolve_training_context(
            _cfg({"strategy": "fsdp", "num_nodes": 1, "gpus_per_node": 2}),
            device=torch.device("cpu"),
            env={"RANK": "0", "LOCAL_RANK": "0", "WORLD_SIZE": "4"},
        )


def test_fsdp_local_rank_must_be_in_range() -> None:
    with pytest.raises(ValueError, match=r"LOCAL_RANK=2 is out of range"):
        resolve_training_context(
            _cfg({"strategy": "fsdp", "gpus_per_node": 2}),
            device=torch.device("cpu"),
            env={"RANK": "0", "LOCAL_RANK": "2", "WORLD_SIZE": "2"},
        )


# The execution gate moved out of this module: build_strategy
# (tests/trainers/test_strategy.py) constructs the FSDP2 strategy + runs the §10
# config gates, and the online recipe's _require_supported_online_strategy
# (tests/scripts/test_online_lifecycle.py) gates the not-yet-wired multi-rank
# orchestration. distributed.py now only resolves/validates the context.
