"""Context-parallel training seam: identity, config, meshes, CP hooks, group sharing.

The rank layout is contiguous CP groups: ranks ``0..cp-1`` replay one sample
set, ``cp..2cp-1`` the next. Tests without a process group check the pure
identity/config rules; the two-rank gloo spawn checks the collective pieces
(mesh shapes, group creation, leader-to-peer batch sharing, loss scaling); the
four-rank NCCL spawn (gpu lane) checks that cp=2 x dp=2 groups form under a
device-bound default group.
"""

from __future__ import annotations

import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from omegaconf import OmegaConf
from torch import nn

from vrl.config.schema import FSDPConfig, parse_config
from vrl.rollouts.batch import RolloutBatch
from vrl.rollouts.orchestration.context_parallel import (
    ContextParallelRolloutSchedule,
    batch_fingerprint,
)
from vrl.rollouts.orchestration.types import RolloutIteration
from vrl.trainers.context_parallel import (
    autocast_safe_attention_forward_op,
    enable_context_parallel,
    install_autocast_safe_attention_ops,
)
from vrl.trainers.distributed import (
    ContextParallelPeerGroup,
    DistributedTrainingContext,
    TrainerProcessGroup,
    collective_timeout,
)
from vrl.trainers.fsdp import build_context_parallel_mesh, build_fsdp_mesh

from ._strategy_policies import run_ranks

# ── identity and config ──────────────────────────────────────────────────────


def test_context_splits_the_world_into_contiguous_cp_groups() -> None:
    ctx = DistributedTrainingContext(
        strategy="fsdp", rank=3, world_size=4, device=torch.device("cpu"), cp_size=2
    )
    assert (ctx.dp_size, ctx.dp_rank, ctx.cp_rank) == (2, 1, 1)
    first = DistributedTrainingContext(
        strategy="fsdp", rank=2, world_size=4, device=torch.device("cpu"), cp_size=2
    )
    assert (first.dp_rank, first.cp_rank) == (1, 0)


def test_cp_followers_get_a_rollout_length_collective_timeout() -> None:
    """A follower waits in the batch broadcast for the leader's whole rollout."""
    from datetime import timedelta

    from torch.distributed import default_pg_timeout

    plain = DistributedTrainingContext(
        strategy="fsdp", rank=0, world_size=2, device=torch.device("cpu")
    )
    assert collective_timeout(plain) == default_pg_timeout
    cp = DistributedTrainingContext(
        strategy="fsdp", rank=0, world_size=2, device=torch.device("cpu"), cp_size=2
    )
    assert collective_timeout(cp) >= timedelta(hours=6)


def test_context_rejects_cp_size_that_does_not_divide_the_world() -> None:
    with pytest.raises(ValueError, match="divide"):
        DistributedTrainingContext(
            strategy="fsdp", rank=0, world_size=4, device=torch.device("cpu"), cp_size=3
        )


def test_fsdp_config_ties_the_cp_mesh_axis_to_the_degrees() -> None:
    assert FSDPConfig().context_parallel.size == 1
    cfg = FSDPConfig(mesh=["dp_shard", "cp"], context_parallel={"ulysses_degree": 2})
    assert cfg.context_parallel.size == 2
    with pytest.raises(ValueError, match="includes 'cp' exactly when"):
        FSDPConfig(context_parallel={"ring_degree": 2})
    with pytest.raises(ValueError, match="includes 'cp' exactly when"):
        FSDPConfig(mesh=["dp_shard", "cp"])
    with pytest.raises(ValueError, match="must be"):
        FSDPConfig(mesh=["dp_replicate", "dp_shard"])


def test_from_root_reads_the_cp_size_and_checks_divisibility(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    training = {
        "strategy": "fsdp",
        "num_nodes": 1,
        "gpus_per_node": 4,
        "fsdp": {"mesh": ["dp_shard", "cp"], "context_parallel": {"ulysses_degree": 2}},
    }
    root = parse_config(OmegaConf.create({"distributed": {"training": training}}))
    ctx = DistributedTrainingContext.from_root(
        root,
        device=torch.device("cpu"),
        env={"RANK": "2", "LOCAL_RANK": "2", "WORLD_SIZE": "4"},
    )
    assert (ctx.cp_size, ctx.dp_rank, ctx.cp_rank) == (2, 1, 0)

    training["gpus_per_node"] = 3
    root = parse_config(OmegaConf.create({"distributed": {"training": training}}))
    with pytest.raises(ValueError, match="must divide WORLD_SIZE=3"):
        DistributedTrainingContext.from_root(
            root,
            device=torch.device("cpu"),
            env={"RANK": "0", "LOCAL_RANK": "0", "WORLD_SIZE": "3"},
        )


def test_context_parallel_mesh_rejects_degrees_that_disagree_with_the_context() -> None:
    ctx = DistributedTrainingContext(
        strategy="fsdp", rank=0, world_size=1, device=torch.device("cpu"), cp_size=1
    )
    with pytest.raises(ValueError, match="disagree"):
        build_context_parallel_mesh(ctx, ulysses_degree=2, ring_degree=1)


def test_backward_scales_the_loss_by_the_cp_size() -> None:
    """CP peers backpropagate partial sums of one loss and FSDP averages over
    dp*cp ranks; the strategy restores the sum by scaling the loss."""
    from vrl.trainers.strategy import FSDPStrategy

    ctx = DistributedTrainingContext(
        strategy="fsdp", rank=0, world_size=2, device=torch.device("cpu"), cp_size=2
    )
    strategy = FSDPStrategy(
        ctx,
        precision_policy="none",
        reshard_after_forward=True,
        cpu_offload=False,
        ulysses_degree=2,
    )
    weight = torch.tensor(3.0, requires_grad=True)
    strategy.backward(weight * 5.0)
    assert weight.grad is not None and float(weight.grad) == pytest.approx(10.0)
    assert strategy.context_parallel is True


# ── the diffusers hook guard ─────────────────────────────────────────────────


def test_enable_context_parallel_refuses_models_without_a_plan() -> None:
    from diffusers.models.modeling_utils import ModelMixin

    class _NoPlan(ModelMixin):
        def __init__(self) -> None:
            super().__init__()
            self.linear = nn.Linear(2, 2)

    with pytest.raises(TypeError, match="ModelMixin"):
        enable_context_parallel(nn.Linear(2, 2), mesh=None, ulysses_degree=2, ring_degree=1)
    with pytest.raises(ValueError, match="declares no _cp_plan"):
        enable_context_parallel(_NoPlan(), mesh=None, ulysses_degree=2, ring_degree=1)


def test_batch_fingerprint_separates_reordered_and_rescored_batches() -> None:
    def _batch(rewards, groups):
        return RolloutBatch(rewards=torch.tensor(rewards), group_ids=torch.tensor(groups))

    same = batch_fingerprint([_batch([1.0, 2.0], [0, 0]), _batch([3.0], [1])])
    assert same == batch_fingerprint([_batch([1.0, 2.0], [0, 0]), _batch([3.0], [1])])
    assert same != batch_fingerprint([_batch([3.0], [1]), _batch([1.0, 2.0], [0, 0])])
    assert same != batch_fingerprint([_batch([1.0, 2.5], [0, 0]), _batch([3.0], [1])])
    assert same != batch_fingerprint([_batch([1.0, 2.0], [0, 1]), _batch([3.0], [1])])


def test_attention_forward_op_saves_one_dtype_for_the_backward_recompute() -> None:
    """Autocast leaves q/k in fp32 and v in bf16 at the kernel boundary; the
    wrapped op casts them to one dtype before the original saves them."""
    seen: dict[str, tuple] = {}

    def original(ctx, query, key, value, *args, **kwargs):
        seen["dtypes"] = (query.dtype, key.dtype, value.dtype)
        return value

    wrapped = autocast_safe_attention_forward_op(original)
    q = torch.zeros(1, 2, 2, 4)
    k = torch.zeros(1, 2, 2, 4)
    v = torch.zeros(1, 2, 2, 4, dtype=torch.bfloat16)
    wrapped(None, q, k, v)
    assert seen["dtypes"] == (torch.bfloat16,) * 3  # lowest precision without autocast
    with torch.autocast("cpu", dtype=torch.bfloat16):
        wrapped(None, q, k, v.to(torch.float16))
    assert seen["dtypes"] == (torch.bfloat16,) * 3  # the autocast dtype wins under autocast
    wrapped(None, q, k, v.float())
    assert seen["dtypes"] == (torch.float32,) * 3  # already uniform: untouched


def test_install_is_idempotent_and_targets_the_native_op() -> None:
    from diffusers.models import attention_dispatch

    install_autocast_safe_attention_ops()
    first = attention_dispatch._native_attention_forward_op
    install_autocast_safe_attention_ops()
    assert attention_dispatch._native_attention_forward_op is first
    assert getattr(first, "_vrl_autocast_safe", False)


# ── two-rank gloo: groups, batch sharing ─────────────────────────────────────


class _LeaderOnlyInner:
    """Fake schedule: returns one batch per prompt, records what it was asked."""

    def __init__(self) -> None:
        self.prompts_seen: list[list[str]] = []

    async def next_iteration(self, prompts, *, group_size, runtime_debug=False, next_prompts=None):
        self.prompts_seen.append(list(prompts))
        batches = [
            RolloutBatch(
                rewards=torch.full((group_size,), float(i)),
                group_ids=torch.full((group_size,), i, dtype=torch.long),
                context={"prompt": prompt},
            )
            for i, prompt in enumerate(prompts)
        ]
        return RolloutIteration(batches=batches)

    async def after_train_step(self) -> None:
        pass

    def reset(self) -> None:
        pass

    async def shutdown(self) -> None:
        pass


def _run_rank(rank: int, world_size: int, port: int, q: mp.Queue) -> None:
    import asyncio

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    try:
        ctx = DistributedTrainingContext(
            strategy="fsdp",
            rank=rank,
            world_size=world_size,
            device=torch.device("cpu"),
            cp_size=2,
        )
        groups = ContextParallelPeerGroup.create(ctx)
        cp_mesh = build_context_parallel_mesh(ctx, ulysses_degree=2, ring_degree=1)
        fsdp_mesh = build_fsdp_mesh(ctx)
        # FSDP shards over the whole world; the CP mesh is its own 3D object.
        mesh_ok = (
            tuple(cp_mesh.mesh_dim_names) == ("dp_shard", "ring", "ulysses")
            and tuple(cp_mesh.shape) == (1, 1, 2)
            and tuple(fsdp_mesh.shape) == (2,)
        )

        inner = _LeaderOnlyInner()
        schedule = ContextParallelRolloutSchedule(inner, groups=groups)
        iteration = asyncio.run(schedule.next_iteration(["a", "b", "c"], group_size=2))
        prompts = [batch.context["prompt"] for batch in iteration.batches]
        q.put((rank, mesh_ok, inner.prompts_seen, prompts))
    finally:
        dist.destroy_process_group()


def test_two_rank_cp_group_generates_slices_and_gathers_the_union() -> None:
    results = run_ranks(_run_rank, 2, timeout=120)

    for rank in (0, 1):
        mesh_ok, _seen, prompts = results[rank]
        assert mesh_ok, f"rank{rank} mesh shape/names"
        assert prompts == ["a", "b", "c"], f"rank{rank} does not hold the gathered union in order"
    assert results[0][1] == [["a"]]  # peer 0 generated its slice
    assert results[1][1] == [["b", "c"]]  # the last peer took the remainder


def _nccl_cp_dp_rank(rank: int, world_size: int, port: int, q: mp.Queue, output_dir: str) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    ctx = DistributedTrainingContext(
        strategy="fsdp",
        rank=rank,
        world_size=world_size,
        device=torch.device("cuda", rank),
        cp_size=2,
    )
    # The default group binds ctx.device, so every subgroup below is split from
    # its communicator; a CP gloo group's non-members still split (nocolor).
    # Creation order follows FSDPStrategy.prepare_model.
    with TrainerProcessGroup.open(ctx, output_dir=output_dir):
        fsdp_mesh = build_fsdp_mesh(ctx)
        groups = ContextParallelPeerGroup.create(ctx)
        cp_mesh = build_context_parallel_mesh(ctx, ulysses_degree=2, ring_degree=1)
        shard_sum = torch.tensor([float(rank)], device=ctx.device)
        dist.all_reduce(shard_sum, group=fsdp_mesh.get_group("dp_shard"))
        ulysses_sum = torch.tensor([float(rank)], device=ctx.device)
        dist.all_reduce(ulysses_sum, group=cp_mesh.get_group("ulysses"))
        peers: list[int | None] = [None] * groups.cp_size
        dist.all_gather_object(peers, rank, group=groups.object_group)
        q.put((rank, shard_sum.item(), ulysses_sum.item(), peers))


@pytest.mark.gpu
@pytest.mark.skipif(torch.cuda.device_count() < 4, reason="requires four CUDA devices")
def test_four_rank_nccl_cp_and_dp_groups_split_from_the_bound_default_group(tmp_path) -> None:
    results = run_ranks(_nccl_cp_dp_rank, 4, str(tmp_path), timeout=300)

    for rank in range(4):
        shard_sum, ulysses_sum, peers = results[rank]
        first_peer = rank - rank % 2
        assert shard_sum == 6.0, f"rank{rank} dp_shard all-reduce"
        assert ulysses_sum == 2 * first_peer + 1, f"rank{rank} ulysses all-reduce"
        assert peers == [first_peer, first_peer + 1], f"rank{rank} CP gloo group"
