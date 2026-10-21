"""Process-group runtime for the ranks of one multi-GPU generation engine.

The ranks of one engine cooperate through torch.distributed collectives
(sequence-parallel attention exchanges activations, weight installs broadcast
per rank). This module owns the process group config that crosses the Ray launch
boundary and the engine's ``EngineProcessGroup``, which the rank program opens
before model build and shuts down when unloading the policy. A single-rank
engine carries no process group config and never touches torch.distributed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from vrl.utils.process_group import DefaultProcessGroup
from vrl.utils.validation import require_int

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ProcessGroupConfig:
    """Where, and as which rank, one process joins its engine's process group.

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
        if not isinstance(self.master_addr, str) or not self.master_addr:
            raise ValueError("rank group master_addr must be a non-empty string")
        require_int(self.master_port, path="rank group.master_port")
        require_int(self.group_rank, path="rank group.group_rank")
        require_int(self.group_world_size, path="rank group.group_world_size")
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


class EngineProcessGroup(DefaultProcessGroup):
    """The process group of one multi-GPU generation engine's ranks.

    The rank program owns it for one policy load: opened before model build
    so collectives are available for the whole model lifetime, shut down
    when the policy is unloaded.
    """

    @classmethod
    def open(cls, config: ProcessGroupConfig) -> EngineProcessGroup:
        device_kwargs = {}
        if config.backend == "nccl":
            import torch

            # An nccl rank is a Ray actor granted one whole GPU, so its only
            # visible card is the current device, the one its model is built
            # on. Binding it connects NCCL eagerly; torch 2.15's torchcomms
            # default requires a bound device.
            device_kwargs["device_id"] = torch.device("cuda", torch.cuda.current_device())
        owner = cls._create(
            backend=config.backend,
            init_method=f"tcp://{config.master_addr}:{config.master_port}",
            rank=config.group_rank,
            world_size=config.group_world_size,
            **device_kwargs,
        )
        logger.info(
            "rank process group up: backend=%s rank=%d/%d rendezvous=%s:%d",
            config.backend,
            config.group_rank,
            config.group_world_size,
            config.master_addr,
            config.master_port,
        )
        return owner


__all__ = [
    "EngineProcessGroup",
    "ProcessGroupConfig",
]
