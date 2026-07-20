"""Wan I2V LoRA replay through a real multi-rank FSDP2 lifecycle.

The production checkpoint is too large for ordinary CI, so this uses the real
diffusers ``WanTransformer3DModel`` with a tiny config. It still exercises the
family wrapper, image-conditioning branch, PEFT adapter, FSDP2 collectives,
optimizer state, and serialized resume contract instead of substituting a fake
linear model.
"""

from __future__ import annotations

import os
import socket
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

pytest.importorskip("diffusers")
pytest.importorskip("peft")

from tests.models.steps.denoise.fixtures import (
    TINY_WAN_LATENT_SHAPE,
    TINY_WAN_TEXT_DIM,
    TINY_WAN_TEXT_LEN,
    build_tiny_wan_i2v_transformer,
    stamp_model_precision,
)
from vrl.config.precision import RolePrecision
from vrl.models.families.wan_2_1.model import (
    WanI2VReplayModel,
    WanI2VSamplingState,
)
from vrl.models.interfaces.runtime import ModelBuild
from vrl.trainers.distributed import DistributedTrainingContext
from vrl.trainers.strategy import FSDPStrategy


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def _build_policy(seed: int = 0) -> WanI2VReplayModel:
    policy = WanI2VReplayModel(
        transformer=build_tiny_wan_i2v_transformer(seed=seed),
        scheduler=None,
        device=torch.device("cpu"),
    )
    build = ModelBuild(
        model_name_or_path="tiny-wan-i2v",
        device=torch.device("cpu"),
        parameter_dtype=torch.float32,
        family="wan_2_1_i2v",
        precision=RolePrecision("fp32", "ieee", outer_autocast=False),
        model_config={
            "use_lora": True,
            "lora": {
                "rank": 2,
                "alpha": 4,
                "target_modules": ["to_q", "to_v"],
            },
        },
        defer_trainable_device_move=True,
    )
    policy.apply_lora(build)
    stamp_model_precision(policy)
    return policy


def _input_state(
    seed: int = 7,
    *,
    device: torch.device | str = "cpu",
) -> WanI2VSamplingState:
    generator = torch.Generator().manual_seed(seed)
    state = WanI2VSamplingState(
        latents=torch.randn(TINY_WAN_LATENT_SHAPE, generator=generator),
        timesteps=torch.tensor([5.0]),
        scheduler=None,
        prompt_embeds=torch.randn(
            1,
            TINY_WAN_TEXT_LEN,
            TINY_WAN_TEXT_DIM,
            generator=generator,
        ),
        negative_prompt_embeds=None,
        image_embeds=torch.randn(1, 2, TINY_WAN_TEXT_DIM, generator=generator),
        condition=torch.randn(TINY_WAN_LATENT_SHAPE, generator=generator),
        guidance_scale=1.0,
        do_cfg=False,
    )
    target = torch.device(device)
    if target.type == "cpu":
        return state
    state.latents = state.latents.to(target)
    state.timesteps = state.timesteps.to(target)
    state.prompt_embeds = state.prompt_embeds.to(target)
    state.image_embeds = state.image_embeds.to(target)
    state.condition = state.condition.to(target)
    return state


def _bundle(policy: WanI2VReplayModel) -> SimpleNamespace:
    return SimpleNamespace(trainable_modules=policy.trainable_modules)


def _train_once(
    policy: WanI2VReplayModel,
    strategy: FSDPStrategy,
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device | str = "cpu",
) -> float:
    optimizer.zero_grad()
    noise_pred = policy.forward_step(_input_state(device=device), 0)["noise_pred"]
    strategy.backward(noise_pred.float().square().mean())
    trainable = [parameter for parameter in policy.parameters() if parameter.requires_grad]
    grad_norm = strategy.clip_grad_norm(trainable, max_norm=10.0)
    optimizer.step()
    return grad_norm


def _run_rank(
    rank: int,
    world_size: int,
    port: int,
    checkpoint_path: str,
    queue: mp.Queue,
) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    try:
        context = DistributedTrainingContext(
            strategy="fsdp",
            distributed=True,
            rank=rank,
            local_rank=0,
            world_size=world_size,
            is_primary=rank == 0,
            device=torch.device("cpu"),
        )
        strategy = FSDPStrategy(context, precision_policy="none")
        policy = strategy.prepare_model(_build_policy())
        optimizer = torch.optim.AdamW(
            [parameter for parameter in policy.parameters() if parameter.requires_grad],
            lr=1e-2,
        )

        before = strategy.export_trainable_state(_bundle(policy))
        grad_norm = _train_once(policy, strategy, optimizer)
        after = strategy.export_trainable_state(_bundle(policy))
        optimizer_state = strategy.export_optimizer_state(policy, optimizer)

        adapter_state = after["transformer"]
        adapter_only = bool(adapter_state) and all("lora_" in key for key in adapter_state)
        changed = any(
            not torch.equal(value, after["transformer"][key])
            for key, value in before["transformer"].items()
        )

        if rank == 0:
            torch.save(
                {"trainable": after, "optimizer": optimizer_state},
                checkpoint_path,
            )
        dist.barrier()
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        resumed = strategy.prepare_model(_build_policy())
        resumed_optimizer = torch.optim.AdamW(
            [parameter for parameter in resumed.parameters() if parameter.requires_grad],
            lr=1e-2,
        )
        strategy.load_trainable_state(_bundle(resumed), checkpoint["trainable"], strict=True)
        strategy.load_optimizer_state(resumed, resumed_optimizer, checkpoint["optimizer"])
        restored = strategy.export_trainable_state(_bundle(resumed))
        resume_matches = all(
            torch.equal(value, restored["transformer"][key])
            for key, value in after["transformer"].items()
        )

        resumed_before = restored
        resumed_grad_norm = _train_once(resumed, strategy, resumed_optimizer)
        resumed_after = strategy.export_trainable_state(_bundle(resumed))
        continued = any(
            not torch.equal(value, resumed_after["transformer"][key])
            for key, value in resumed_before["transformer"].items()
        )
        queue.put(
            (
                rank,
                grad_norm,
                adapter_only,
                changed,
                resume_matches,
                resumed_grad_norm,
                continued,
            ),
        )
    finally:
        dist.destroy_process_group()


def test_wan_i2v_fsdp_step_checkpoint_resume_and_continue(tmp_path: Path) -> None:
    context = mp.get_context("spawn")
    queue: mp.Queue = context.Queue()
    port = _free_port()
    checkpoint_path = tmp_path / "checkpoint.pt"
    processes = [
        context.Process(
            target=_run_rank,
            args=(rank, 2, port, str(checkpoint_path), queue),
        )
        for rank in range(2)
    ]
    for process in processes:
        process.start()

    results = {}
    for _ in range(2):
        rank, *values = queue.get(timeout=120)
        results[rank] = values
    for process in processes:
        process.join(timeout=10)
        assert process.exitcode == 0

    assert checkpoint_path.is_file()
    for rank in (0, 1):
        (
            grad_norm,
            adapter_only,
            changed,
            resume_matches,
            resumed_grad_norm,
            continued,
        ) = results[rank]
        assert grad_norm > 0, f"rank{rank} produced a zero first-step gradient"
        assert adapter_only, f"rank{rank} checkpoint materialized frozen Wan weights"
        assert changed, f"rank{rank} optimizer did not change a LoRA parameter"
        assert resume_matches, f"rank{rank} did not restore the serialized LoRA state"
        assert resumed_grad_norm > 0, f"rank{rank} produced a zero resumed gradient"
        assert continued, f"rank{rank} did not update after resume"


def _run_cuda_rank(rank: int, world_size: int, port: int, queue: mp.Queue) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    try:
        device = torch.device("cuda", rank)
        context = DistributedTrainingContext(
            strategy="fsdp",
            distributed=True,
            rank=rank,
            local_rank=rank,
            world_size=world_size,
            is_primary=rank == 0,
            device=device,
        )
        strategy = FSDPStrategy(context, precision_policy="none")
        policy = strategy.prepare_model(_build_policy())

        from torch.distributed.tensor import DTensor

        cuda_sharded = all(
            isinstance(parameter, DTensor)
            and parameter.to_local().device == device
            and parameter.numel() >= parameter.to_local().numel()
            for parameter in policy.parameters()
        )
        optimizer = torch.optim.AdamW(
            [parameter for parameter in policy.parameters() if parameter.requires_grad],
            lr=1e-2,
        )
        before = strategy.export_trainable_state(_bundle(policy))
        grad_norm = _train_once(policy, strategy, optimizer, device=device)
        after = strategy.export_trainable_state(_bundle(policy))
        changed = any(
            not torch.equal(value, after["transformer"][key])
            for key, value in before["transformer"].items()
        )
        adapter_only = all("lora_" in key for key in after["transformer"])
        peak_bytes = int(torch.cuda.max_memory_allocated(device))
        queue.put((rank, cuda_sharded, grad_norm, changed, adapter_only, peak_bytes))
    finally:
        dist.destroy_process_group()
        torch.cuda.empty_cache()


@pytest.mark.gpu
@pytest.mark.distributed
def test_wan_i2v_fsdp_four_rank_cuda_step() -> None:
    if torch.cuda.device_count() < 4:
        pytest.skip("requires four CUDA devices")

    context = mp.get_context("spawn")
    queue: mp.Queue = context.Queue()
    port = _free_port()
    processes = [
        context.Process(target=_run_cuda_rank, args=(rank, 4, port, queue)) for rank in range(4)
    ]
    for process in processes:
        process.start()

    results = {}
    for _ in range(4):
        rank, *values = queue.get(timeout=180)
        results[rank] = values
    for process in processes:
        process.join(timeout=10)
        assert process.exitcode == 0

    for rank in range(4):
        cuda_sharded, grad_norm, changed, adapter_only, peak_bytes = results[rank]
        assert cuda_sharded, f"rank{rank} did not own CUDA DTensor shards"
        assert grad_norm > 0, f"rank{rank} produced a zero CUDA gradient"
        assert changed, f"rank{rank} did not update a CUDA LoRA shard"
        assert adapter_only, f"rank{rank} gathered frozen Wan weights"
        assert peak_bytes > 0, f"rank{rank} reported no CUDA allocation"
