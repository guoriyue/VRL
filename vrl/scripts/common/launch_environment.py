"""Prepare rank-local CUDA visibility before importing training runtimes."""

from __future__ import annotations

import os
from collections.abc import MutableMapping
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vrl.config.schema import RootConfig


def narrow_rank_local_cuda_visibility(
    root: RootConfig,
    *,
    environ: MutableMapping[str, str] | None = None,
) -> str | None:
    """Give each symmetric-colocated torchrun rank one logical CUDA device.

    Every rank owns a separate local Ray cluster. If all ranks retain the host's
    full CUDA view, Ray may place a rank's single GPU bundle on another rank's
    card. Narrow before importing the trainer so Torch, NCCL, and Ray all agree
    that this rank's physical card is logical ``cuda:0``.
    """

    distributed = root.distributed
    training = None if distributed is None else distributed.training
    resources = None if distributed is None else distributed.resources
    strategy = "single_process" if training is None else str(training.strategy)
    rollout_pool = "auto" if resources is None else str(resources.rollout.gpu_pool)
    if strategy not in {"ddp", "fsdp"} or rollout_pool != "trainer":
        return None

    environment = os.environ if environ is None else environ
    local_rank_raw = environment.get("LOCAL_RANK")
    world_size_raw = environment.get("WORLD_SIZE")
    if local_rank_raw is None or world_size_raw is None:
        # DistributedTrainingContext.from_root owns the complete missing torchrun-env error.
        return None
    try:
        local_rank = int(local_rank_raw)
        world_size = int(world_size_raw)
        configured_local_world_size = 1 if training is None else int(training.gpus_per_node)
        local_world_size = int(
            environment.get("LOCAL_WORLD_SIZE", str(configured_local_world_size)),
        )
    except ValueError as exc:
        raise ValueError(
            "torchrun rank sizes must be integers before rank-local CUDA selection: "
            f"LOCAL_RANK={local_rank_raw!r}, WORLD_SIZE={world_size_raw!r}, "
            f"LOCAL_WORLD_SIZE={environment.get('LOCAL_WORLD_SIZE')!r}",
        ) from exc
    if (
        local_rank < 0
        or local_world_size <= 0
        or world_size < local_world_size
        or local_rank >= local_world_size
    ):
        raise ValueError(
            "invalid torchrun rank identity before rank-local CUDA selection: "
            f"LOCAL_RANK={local_rank}, LOCAL_WORLD_SIZE={local_world_size}, "
            f"WORLD_SIZE={world_size}",
        )
    if local_world_size != configured_local_world_size:
        raise ValueError(
            "LOCAL_WORLD_SIZE must match distributed.training.gpus_per_node before "
            f"rank-local CUDA selection: LOCAL_WORLD_SIZE={local_world_size}, "
            f"gpus_per_node={configured_local_world_size}",
        )

    if "CUDA_VISIBLE_DEVICES" not in environment:
        selected = str(local_rank)
    else:
        raw_visible = environment["CUDA_VISIBLE_DEVICES"].strip()
        if not raw_visible:
            raise ValueError(
                "CUDA_VISIBLE_DEVICES is empty for a GPU-distributed rank-local launch",
            )
        tokens = [token.strip() for token in raw_visible.split(",")]
        if any(not token or not token.isascii() or not token.isdecimal() for token in tokens):
            raise ValueError(
                "CUDA_VISIBLE_DEVICES must contain non-negative integer device ordinals "
                f"without empty entries: {raw_visible!r}",
            )
        devices = [int(token) for token in tokens]
        if len(set(devices)) != len(devices):
            raise ValueError(
                "CUDA_VISIBLE_DEVICES contains duplicate devices before rank-local "
                f"launch: {raw_visible!r}",
            )
        if len(devices) < local_world_size:
            raise ValueError(
                "CUDA_VISIBLE_DEVICES cannot supply every local torchrun rank: "
                f"LOCAL_WORLD_SIZE={local_world_size}, "
                f"CUDA_VISIBLE_DEVICES={raw_visible!r}",
            )
        selected = str(devices[local_rank])

    environment["CUDA_VISIBLE_DEVICES"] = selected
    return selected
