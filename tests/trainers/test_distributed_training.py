"""Tests for the distributed training context (vrl/trainers/distributed.py)."""

from __future__ import annotations

import pytest
import torch
from omegaconf import OmegaConf

from vrl.config.schema import parse_config
from vrl.trainers.distributed import (
    DistributedTrainingContext,
    TrainerProcessGroup,
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


def test_distributed_missing_env_fails_fast_listing_keys() -> None:
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


@pytest.mark.parametrize("rank", [-1, 2])
def test_global_rank_must_be_within_world_size(rank) -> None:
    with pytest.raises(ValueError, match=rf"RANK={rank} is out of range.*WORLD_SIZE=2"):
        DistributedTrainingContext.from_root(
            parse_config(_cfg({"strategy": "fsdp", "gpus_per_node": 2})),
            device=torch.device("cpu"),
            env={"RANK": str(rank), "LOCAL_RANK": "0", "WORLD_SIZE": "2"},
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


# ── TrainerProcessGroup: the process owns its groups ────────────────────────


class _FakeTorchDistributed:
    """torch.distributed's group lifecycle, recorded instead of performed."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, *, fail_subgroup: bool = False):
        from types import SimpleNamespace

        import torch.distributed as dist

        self.live = False
        self.events: list[str] = []
        self._fail_subgroup = fail_subgroup
        self.group = SimpleNamespace(WORLD=None)
        monkeypatch.setattr(dist, "is_initialized", lambda: self.live)
        monkeypatch.setattr(dist, "init_process_group", self._init)
        monkeypatch.setattr(dist, "new_group", self._new_group)
        monkeypatch.setattr(dist, "destroy_process_group", self._destroy)
        monkeypatch.setattr(dist, "group", self.group)
        monkeypatch.setattr(dist, "get_rank", lambda: 1)
        monkeypatch.setattr(dist, "get_world_size", lambda: 2)
        monkeypatch.setattr(dist, "get_backend", lambda: "nccl")
        monkeypatch.setattr(torch.cuda, "set_device", lambda device: None)

    def _init(self, **kwargs):
        self.events.append(f"init:{kwargs['backend']}")
        self.live = True
        self.group.WORLD = "world-group"

    def _new_group(self, **kwargs):
        if self._fail_subgroup:
            raise RuntimeError("gloo subgroup failed")
        self.events.append(f"subgroup:{kwargs['backend']}")
        return "cpu-group"

    def _destroy(self, group=None):
        self.events.append("destroy")
        self.live = False
        self.group.WORLD = None


def _rank(device: str = "cuda:0") -> DistributedTrainingContext:
    return DistributedTrainingContext(
        strategy="fsdp", rank=1, world_size=2, device=torch.device(device)
    )


def test_process_group_lives_until_its_owner_shuts_it_down(monkeypatch) -> None:
    fake = _FakeTorchDistributed(monkeypatch)

    group = TrainerProcessGroup.open(_rank())
    assert fake.events == ["init:nccl", "subgroup:gloo"]
    assert TrainerProcessGroup.cpu_group() == "cpu-group"
    TrainerProcessGroup.require(_rank())

    group.shutdown()
    group.shutdown()  # repeatable

    assert fake.events == ["init:nccl", "subgroup:gloo", "destroy"]
    assert TrainerProcessGroup.cpu_group() is None


def test_a_failed_run_still_destroys_the_process_group(monkeypatch) -> None:
    fake = _FakeTorchDistributed(monkeypatch)

    with pytest.raises(ValueError, match="run failed"), TrainerProcessGroup.open(_rank()):
        raise ValueError("run failed")

    assert fake.events[-1] == "destroy"
    assert not fake.live


def test_a_half_built_process_group_is_destroyed(monkeypatch) -> None:
    """The default group exists when the CPU subgroup fails; open tears it down."""

    fake = _FakeTorchDistributed(monkeypatch, fail_subgroup=True)

    with pytest.raises(RuntimeError, match="gloo subgroup failed"):
        TrainerProcessGroup.open(_rank())

    assert fake.events == ["init:nccl", "destroy"]
    assert not fake.live
    assert TrainerProcessGroup.cpu_group() is None


def test_an_existing_process_group_is_refused_not_adopted(monkeypatch) -> None:
    fake = _FakeTorchDistributed(monkeypatch)
    fake.live = True

    with pytest.raises(RuntimeError, match="already initialized"):
        TrainerProcessGroup.open(_rank())

    # Someone else's group is left alone.
    assert fake.events == []
    assert fake.live


def test_single_process_owns_no_process_group(monkeypatch) -> None:
    fake = _FakeTorchDistributed(monkeypatch)
    single = DistributedTrainingContext(
        strategy="single_process", rank=0, world_size=1, device=torch.device("cpu")
    )

    with TrainerProcessGroup.open(single):
        TrainerProcessGroup.require(single)

    assert fake.events == []


def test_a_strategy_refuses_a_group_with_another_identity(monkeypatch) -> None:
    _FakeTorchDistributed(monkeypatch)

    with (
        TrainerProcessGroup.open(_rank()),
        pytest.raises(RuntimeError, match="does not match"),
    ):
        TrainerProcessGroup.require(
            DistributedTrainingContext(
                strategy="fsdp", rank=0, world_size=2, device=torch.device("cuda:0")
            )
        )
