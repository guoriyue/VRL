"""FSDP replay weights come from the primary rank, not from every rank's loader.

``FSDPStrategy.materialize_weights`` is true only on rank 0. The other ranks
build the replay transformer as a meta skeleton (``load_diffusers_transformer``
with ``materialize_weights=False``) and ``prepare_model`` fills it after
sharding from rank 0's tensors, so a checkpoint is read into host memory once
per node. Two real gloo ranks on a tiny real ``SD3Transformer2DModel``: every
gathered parameter and every buffer must equal rank 0's pre-shard values and no
meta storage may survive, for both the fully sharded and the adapter-only
(replicated frozen base) layouts.
"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

pytest.importorskip("diffusers")
pytest.importorskip("peft")

from tests.models.steps.denoise.fixtures import build_tiny_sd3_transformer
from tests.trainers._strategy_policies import free_port
from vrl.config.precision import RolePrecision
from vrl.config.schema import FSDPConfig
from vrl.models.families.sd3_5.model import SD3_5ReplayModel
from vrl.models.interfaces.runtime import ModelBuild
from vrl.trainers.distributed import DistributedTrainingContext
from vrl.trainers.strategy import FSDPStrategy

_LORA = {"rank": 2, "alpha": 4, "target_modules": ["to_q", "to_v"]}


def _build(repo: str) -> ModelBuild:
    return ModelBuild(
        model_name_or_path=repo,
        revision=None,
        device=torch.device("cpu"),
        parameter_dtype=torch.float32,
        family="sd3_5",
        precision=RolePrecision("fp32", "ieee", outer_autocast=False),
        model_config={"use_lora": True, "lora": _LORA},
    )


def _policy(repo: str, *, materialize_weights: bool) -> SD3_5ReplayModel:
    """The replay model exactly as the family builder assembles it, LoRA attached."""
    from tests.models.steps.denoise.fixtures import stamp_model_precision
    from vrl.models.loader import load_diffusers_transformer

    build = _build(repo)
    policy = SD3_5ReplayModel(
        transformer=load_diffusers_transformer(
            build, "SD3Transformer2DModel", materialize_weights=materialize_weights
        ),
        scheduler=None,
        device=torch.device("cpu"),
    )
    policy.apply_lora(build)
    stamp_model_precision(policy)
    return policy


def _run_rank(
    rank: int,
    world_size: int,
    port: int,
    repo: str,
    shard_trainable_only: bool,
    queue: mp.Queue,
) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    try:
        context = DistributedTrainingContext(
            strategy="fsdp", rank=rank, world_size=world_size, device=torch.device("cpu")
        )
        config = FSDPConfig.model_validate(
            {"precision_policy": "none", "shard_trainable_only": shard_trainable_only}
        )
        strategy = FSDPStrategy(
            context,
            mesh_dims=config.mesh,
            precision_policy=config.precision_policy,
            reshard_after_forward=config.reshard_after_forward,
            cpu_offload=config.cpu_offload,
            shard_trainable_only=config.shard_trainable_only,
        )
        policy = _policy(repo, materialize_weights=strategy.materialize_weights)
        skeleton_before = any(p.is_meta for p in policy.transformer.parameters())
        # Rank 0's authoritative values, captured before sharding touches them.
        reference = (
            {k: v.detach().clone() for k, v in policy.transformer.state_dict().items()}
            if rank == 0
            else None
        )

        strategy.prepare_model(policy)

        transformer = policy.transformer
        meta_left = any(t.is_meta for t in (*transformer.parameters(), *transformer.buffers()))
        # Compare every tensor with rank 0's reference: gather DTensor shards to full
        # values, then broadcast rank 0's reference for the equality check.
        mismatched: list[str] = []
        for name, tensor in (*transformer.named_parameters(), *transformer.named_buffers()):
            full = tensor.full_tensor() if hasattr(tensor, "full_tensor") else tensor.detach()
            expected = torch.zeros_like(full)
            if rank == 0:
                expected.copy_(reference[name])
            dist.broadcast(expected, src=0)
            if not torch.equal(full, expected):
                mismatched.append(name)
        trainable = [n for n, p in transformer.named_parameters() if p.requires_grad]
        queue.put((rank, skeleton_before, meta_left, mismatched, trainable))
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("shard_trainable_only", [False, True])
def test_non_primary_ranks_are_filled_from_rank_zero(
    tmp_path: Path, shard_trainable_only: bool
) -> None:
    repo = tmp_path / "tiny-sd3"
    build_tiny_sd3_transformer(seed=123).save_pretrained(repo / "transformer")

    context = mp.get_context("spawn")
    queue: mp.Queue = context.Queue()
    processes = [
        context.Process(
            target=_run_rank, args=(rank, 2, free_port(), str(repo), shard_trainable_only, queue)
        )
        for rank in range(2)
    ]
    port = free_port()
    processes = [
        context.Process(
            target=_run_rank, args=(rank, 2, port, str(repo), shard_trainable_only, queue)
        )
        for rank in range(2)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=300)
    assert all(process.exitcode == 0 for process in processes)

    results = {}
    while not queue.empty():
        rank, skeleton_before, meta_left, mismatched, trainable = queue.get()
        results[rank] = SimpleNamespace(
            skeleton_before=skeleton_before,
            meta_left=meta_left,
            mismatched=mismatched,
            trainable=trainable,
        )
    assert set(results) == {0, 1}
    # Only the non-primary rank built a skeleton; nothing stays on meta afterwards.
    assert results[0].skeleton_before is False
    assert results[1].skeleton_before is True
    assert not results[0].meta_left and not results[1].meta_left
    # Every parameter and buffer on both ranks equals rank 0's pre-shard values.
    assert results[0].mismatched == [] and results[1].mismatched == []
    # The adapter is still the trainable set on the filled rank.
    assert results[1].trainable and all("lora_" in name for name in results[1].trainable)
    assert results[1].trainable == results[0].trainable
