"""Process-group runtime for the ranks of one multi-GPU generation engine.

The ranks of one engine cooperate through torch.distributed collectives
(sequence-parallel attention exchanges activations, weight installs broadcast
per rank). This module owns the rendezvous spec that crosses the Ray launch
boundary and the init/destroy pair the rank program calls around its model
lifetime. A single-rank engine carries no spec and never touches
torch.distributed — the entire module is inert until a multi-GPU engine
backend lands (docs/sprints/SPRINT_engine_worker_vocabulary.md, P4/P5).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class RankGroupSpec:
    """Rendezvous contract for one rank of one engine's process group.

    ``master_addr`` is reachable by every rank of the engine; engine groups are
    single-node (PACK placement), so the launcher hands out a loopback address.
    ``backend`` is ``nccl`` for GPU ranks and ``gloo`` for the CPU test path.
    """

    master_addr: str
    master_port: int
    group_rank: int
    group_world_size: int
    backend: str = "nccl"

    def __post_init__(self) -> None:
        if not self.master_addr:
            raise ValueError("rank group master_addr must be non-empty")
        if not 0 < self.master_port < 65536:
            raise ValueError(f"rank group master_port out of range: {self.master_port}")
        if self.group_world_size < 2:
            raise ValueError(
                "a rank group exists only for multi-rank engines; "
                f"got world size {self.group_world_size}",
            )
        if not 0 <= self.group_rank < self.group_world_size:
            raise ValueError(
                f"rank {self.group_rank} outside world of {self.group_world_size}",
            )
        if self.backend not in {"nccl", "gloo"}:
            raise ValueError(f"unsupported rank group backend: {self.backend!r}")


def init_rank_process_group(spec: RankGroupSpec) -> None:
    """Join this rank into its engine's process group (idempotence rejected).

    Called by the rank program before model build so collectives are available
    for the whole model lifetime. Double init means two owners disagree about
    the engine lifecycle — fail loud instead of silently reusing the group.
    """

    import torch.distributed as dist

    if dist.is_initialized():
        raise RuntimeError(
            "rank process group already initialized; the engine lifecycle owns "
            "exactly one init/destroy pair per policy load",
        )
    dist.init_process_group(
        backend=spec.backend,
        init_method=f"tcp://{spec.master_addr}:{spec.master_port}",
        rank=spec.group_rank,
        world_size=spec.group_world_size,
    )
    logger.info(
        "rank process group up: backend=%s rank=%d/%d rendezvous=%s:%d",
        spec.backend,
        spec.group_rank,
        spec.group_world_size,
        spec.master_addr,
        spec.master_port,
    )


def destroy_rank_process_group() -> None:
    """Leave the engine's process group; safe to call when none exists."""

    import torch.distributed as dist

    if dist.is_initialized():
        dist.destroy_process_group()


__all__ = [
    "RankGroupSpec",
    "destroy_rank_process_group",
    "init_rank_process_group",
]
