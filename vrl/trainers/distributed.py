"""Distributed training process identity for the single-process / FSDP trainer.

This module answers "which training process am I, and on what device" without
touching Ray, creating a process group, or wrapping the model. Ray placement and
actor lifecycle stay in ``vrl/ray/``. The strategy seam (backward / clip / state
export) lives in ``vrl/trainers/strategy.py`` and consumes the context produced
here.

This module contains the context (``DistributedTrainingContext``) and the
process-group lifecycle every multi-rank strategy shares -- ``ddp`` and ``fsdp``
both create the group from the context, tear it down on shutdown, and exchange
park/wake coordination messages over the CPU-capable group. The FSDP2 strategy
layer (``fully_shard`` wrapping + DTensor full-state export) lives in
``vrl/trainers/fsdp.py`` + ``FSDPStrategy``, built from this context by
``vrl/trainers/strategy.py`` build_strategy. The online recipe supports the
symmetric colocated torchrun path for ``ddp`` and ``fsdp``: each rank owns its
local rollout/training device and the strategy layer handles cross-rank gradient
coordination. TrainingCollectives provides communication over those groups.

Training-only differentiable token/head exchanges also live here. They consume
an existing process group and do not change process identity or model wrapping.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from vrl.config.schema import RootConfig

# torchrun / env-launcher contract. Source of truth for the keys the fsdp context
# parses; the missing-env error lists exactly these.
_TORCHRUN_ENV_KEYS = ("RANK", "LOCAL_RANK", "WORLD_SIZE")


def _context_parallel_layout(tensor: torch.Tensor, group: Any) -> tuple[int, int]:
    import torch.distributed as dist

    if not dist.is_initialized():
        raise RuntimeError("context parallel exchange requires an initialized process group")
    world, rank = dist.get_world_size(group), dist.get_rank(group)
    if world < 2 or rank < 0:
        raise ValueError("context parallel exchange requires membership in a group of >= 2")
    if tensor.ndim != 4 or any(size == 0 for size in tensor.shape):
        raise ValueError(
            "context parallel exchange expects nonempty [batch, heads, tokens, width]"
        )
    return world, rank


def _context_parallel_gather(tensor: torch.Tensor, dim: int, group: Any) -> torch.Tensor:
    from torch.distributed._functional_collectives import all_gather_tensor_autograd

    # Keep the collective's gather dimension zero; preserve head/token ordering.
    full = all_gather_tensor_autograd(tensor.movedim(dim, 0).contiguous(), 0, group)
    return full.movedim(0, dim)


def context_parallel_gather_tokens(tensor: torch.Tensor, *, group: Any) -> torch.Tensor:
    """Gather [B,S/P,D] outputs with summed consumer gradients.

    A full-output objective replicated on every CP rank must be divided by the
    group size before backward. Parameter-gradient reduction remains separate.
    """
    if tensor.ndim != 3:
        raise ValueError("context parallel output gather expects [batch, tokens, width]")
    _context_parallel_layout(tensor.unsqueeze(1), group)
    return _context_parallel_gather(tensor, 1, group)


def context_parallel_tokens_to_heads(tensor: torch.Tensor, *, group: Any) -> torch.Tensor:
    """[B,H,S/P,D] -> [B,H/P,S,D], with gradients across equal token shards.

    All group members must call with matching shapes/dtypes in the same order.
    This gather-based baseline materializes full Q/K/V temporarily; it is not
    the bandwidth-optimal all-to-all implementation.
    """
    world, rank = _context_parallel_layout(tensor, group)
    if tensor.shape[1] % world:
        raise ValueError("attention heads must be divisible by context parallel group size")
    full = _context_parallel_gather(tensor, 2, group)
    return full.chunk(world, dim=1)[rank].contiguous()


def context_parallel_heads_to_tokens(tensor: torch.Tensor, *, group: Any) -> torch.Tensor:
    """[B,H/P,S,D] -> [B,H,S/P,D], the differentiable inverse exchange.

    Backward sums contributions from each consumer rank; callers must not
    divide local-token losses by the CP group size unless their objective is
    replicated. Parameter-gradient reduction remains the strategy's job.
    """
    world, rank = _context_parallel_layout(tensor, group)
    if tensor.shape[2] % world:
        raise ValueError("token count must be divisible by context parallel group size")
    full = _context_parallel_gather(tensor, 1, group)
    return full.chunk(world, dim=2)[rank].contiguous()


@dataclass(frozen=True, slots=True)
class DistributedTrainingContext:
    """Identity of the current training process — pure description.

    Creates no process group and wraps no model; it only records who this process
    is (rank / world_size / primary) and which device it owns, so the
    trainer and rank0-only output paths can branch without reading env directly.
    """

    strategy: str
    rank: int
    world_size: int
    device: torch.device

    @property
    def distributed(self) -> bool:
        return self.strategy != "single_process"

    @property
    def is_primary(self) -> bool:
        return self.rank == 0

    @staticmethod
    def _require_env_int(env: Mapping[str, str], key: str) -> int:
        raw = env.get(key)
        if not raw:
            missing = [k for k in _TORCHRUN_ENV_KEYS if not env.get(k)]
            raise ValueError(
                "distributed training requires torchrun env vars "
                f"{list(_TORCHRUN_ENV_KEYS)}; missing {missing}. Launch with "
                "`torchrun --nproc-per-node=<N>` or set them explicitly before the run "
                "(this fails here, not later at CUDA-device or Ray-launch time)."
            )
        try:
            return int(raw)
        except ValueError as exc:
            raise ValueError(f"distributed.training: {key}={raw!r} is not an integer") from exc

    @classmethod
    def from_root(
        cls,
        root: RootConfig,
        *,
        device: torch.device,
        env: Mapping[str, str] | None = None,
    ) -> DistributedTrainingContext:
        """Build this process's training identity from the root config and the torchrun env.

        ``single_process`` always returns rank0 / world1 / primary and keeps the
        resource-resolved ``device``; it ignores env entirely. ``fsdp`` parses and
        validates ``RANK`` / ``LOCAL_RANK`` / ``WORLD_SIZE`` (fail-fast on missing or
        inconsistent values) and derives a per-process ``cuda:<local_rank>`` device. It
        does NOT create a process group; ``vrl/trainers/strategy.py`` build_strategy
        turns this context into the matching strategy, whose ``prepare_model`` calls
        ``init_training_process_group`` below.
        """

        env = os.environ if env is None else env
        distributed = root.distributed
        training = None if distributed is None else distributed.training
        strategy = "single_process" if training is None else str(training.strategy)

        if strategy == "single_process":
            return cls(
                strategy=strategy,
                rank=0,
                world_size=1,
                device=device,
            )

        if strategy in {"fsdp", "ddp"}:
            # Both are torchrun multi-rank strategies: one process per GPU, identity +
            # per-process cuda:<local_rank> device derived from the launcher env. No
            # process group is created here (build_strategy's strategy does that).
            rank = cls._require_env_int(env, "RANK")
            local_rank = cls._require_env_int(env, "LOCAL_RANK")
            world_size = cls._require_env_int(env, "WORLD_SIZE")
            assert training is not None  # strategy came from it
            num_nodes = int(training.num_nodes)
            gpus_per_node = int(training.gpus_per_node)
            expected = num_nodes * gpus_per_node
            if world_size != expected:
                raise ValueError(
                    f"distributed.training: WORLD_SIZE={world_size} must equal "
                    f"num_nodes*gpus_per_node={expected} "
                    f"(num_nodes={num_nodes}, gpus_per_node={gpus_per_node})."
                )
            if not 0 <= rank < world_size:
                raise ValueError(
                    f"distributed.training: RANK={rank} is out of range for "
                    f"WORLD_SIZE={world_size} (expected 0..{world_size - 1})."
                )
            if not 0 <= local_rank < gpus_per_node:
                raise ValueError(
                    f"distributed.training: LOCAL_RANK={local_rank} is out of range for "
                    f"gpus_per_node={gpus_per_node} (expected 0..{gpus_per_node - 1})."
                )
            # The rank's device is an index into the devices THIS PROCESS can see,
            # which is not always the local rank. Symmetric-colocated placement is
            # resolved per rank as "my one local GPU" (vrl/ray/resources.py), so a
            # single-node launch narrows each rank to its own card with
            # CUDA_VISIBLE_DEVICES; that rank then sees exactly one device and
            # cuda:<local_rank> would be out of range for every rank but 0. Only the
            # two known shapes map implicitly; a partial mask (more ranks than
            # visible devices, but not exactly one) must fail here instead of
            # silently double-mapping ranks onto one card and dying later in NCCL.
            visible_device_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
            if visible_device_count == 0 or local_rank < visible_device_count:
                device_index = local_rank
            elif visible_device_count == 1:
                device_index = 0
            else:
                raise ValueError(
                    f"distributed.training: LOCAL_RANK={local_rank} but this process sees "
                    f"only {visible_device_count} CUDA devices; either expose every GPU "
                    "(unset/expand CUDA_VISIBLE_DEVICES) or narrow each rank to exactly "
                    "its own single device.",
                )
            return cls(
                strategy=strategy,
                rank=rank,
                world_size=world_size,
                device=torch.device(f"cuda:{device_index}"),
            )

        # Schema (TrainingSection.strategy Literal) rejects other values before we get
        # here; this guards direct callers that bypass schema validation.
        raise ValueError(
            f"unknown distributed.training.strategy={strategy!r}; "
            "expected 'single_process', 'fsdp', or 'ddp'"
        )


_CPU_COORDINATION_GROUP: Any = None


def cpu_coordination_group() -> Any:
    """The CPU-capable group for phase-boundary coordination, or ``None``.

    Park/wake windows unmap multi-GB cumem pools; coordination messages inside
    those windows (quiesce barriers, park-success flags) must therefore issue
    ZERO GPU kernels — a NCCL all-reduce right after this rank's unmap runs a
    kernel on the just-parked card while slower peers are still unmapping
    theirs (the load pattern in the 2026-08-16 Xid 79 postmortem). NCCL runs
    get a dedicated gloo subgroup; a gloo default group is already CPU-capable
    and is returned as-is.
    """

    import torch.distributed as dist

    if not dist.is_initialized():
        return None
    if _CPU_COORDINATION_GROUP is not None:
        return _CPU_COORDINATION_GROUP
    if dist.get_backend() == "gloo":
        return dist.group.WORLD
    return None


def init_training_process_group(
    context: DistributedTrainingContext,
    *,
    backend: str = "nccl",
) -> None:
    """Create the torch.distributed process group for a ddp/fsdp rank.

    No-op for ``single_process`` and when a group already exists. The owning
    ``Strategy.shutdown`` calls the matching ``shutdown_training_process_group``.
    ``DistributedTrainingContext.from_root`` validates rank and world-size
    identity. Torch's environment rendezvous validates ``MASTER_ADDR`` and
    ``MASTER_PORT`` when creating the group.
    """

    import torch.distributed as dist

    global _CPU_COORDINATION_GROUP
    if not context.distributed or dist.is_initialized():
        return
    if context.device.type == "cuda":
        # ``context.device`` is the CUDA ordinal inside this rank's masked view.
        torch.cuda.set_device(context.device)
    dist.init_process_group(
        backend=backend,
        rank=context.rank,
        world_size=context.world_size,
    )
    if backend == "nccl":
        # Collective creation: every rank reaches this line inside the same
        # init call, so the subgroup handshake cannot mismatch.
        _CPU_COORDINATION_GROUP = dist.new_group(backend="gloo")


def shutdown_training_process_group() -> None:
    """Tear down the process group if one is live (safe to call unconditionally)."""

    import torch.distributed as dist

    global _CPU_COORDINATION_GROUP
    _CPU_COORDINATION_GROUP = None
    if dist.is_initialized():
        dist.destroy_process_group()


def run_on_primary_rank(
    context: DistributedTrainingContext,
    operation: Callable[[], None],
    *,
    description: str,
) -> None:
    """Run ``operation`` on rank 0 only; if it fails, every rank raises.

    Rank 0 owns the output directory, so file writes happen there. The other
    ranks wait on a broadcast of the outcome, so a failed write cannot leave
    rank 0 dead while its peers enter the next collective and hang.
    """

    if not context.distributed:
        operation()
        return
    failure: BaseException | None = None
    if context.is_primary:
        try:
            operation()
        except BaseException as error:
            failure = error
    failure_message = [None if failure is None else f"{type(failure).__name__}: {failure}"]
    torch.distributed.broadcast_object_list(failure_message, src=0, device=context.device)
    if failure_message[0] is not None:
        raise RuntimeError(f"{description} failed on rank 0: {failure_message[0]}") from failure


class TrainingCollectives:
    """Rank communication shared by one trainer and its training strategy.

    The strategy initializes and closes the process groups. Resolve those live
    groups at each call because construction precedes model preparation. A
    single-process context never joins another runtime's distributed group.
    """

    def __init__(self, context: DistributedTrainingContext) -> None:
        self.context = context

    def _reduce_values(
        self,
        values: list[int] | list[float],
        *,
        dtype: torch.dtype,
        op: Any,
        group: Any = None,
    ) -> list[Any]:
        """Reduce on the selected group's backend, using this rank's device."""
        dist = torch.distributed
        if not self.context.distributed or not (
            dist.is_available() and dist.is_initialized() and dist.get_world_size(group) > 1
        ):
            return values
        device = self.context.device if dist.get_backend(group) == "nccl" else "cpu"
        tensor = torch.tensor(values, dtype=dtype, device=device)
        dist.all_reduce(tensor, op=op, group=group)
        return tensor.tolist()

    def max_int(self, value: int) -> int:
        return int(
            self._reduce_values([value], dtype=torch.int64, op=torch.distributed.ReduceOp.MAX)[0]
        )

    def max_float(self, value: float) -> float:
        return float(
            self._reduce_values([value], dtype=torch.float64, op=torch.distributed.ReduceOp.MAX)[0]
        )

    def sum(self, values: list[float]) -> list[float]:
        return self._reduce_values(values, dtype=torch.float64, op=torch.distributed.ReduceOp.SUM)

    def all_true(self, value: bool) -> bool:
        return bool(
            self._reduce_values(
                [int(value)], dtype=torch.int32, op=torch.distributed.ReduceOp.MIN
            )[0]
        )

    def succeeded(self, succeeded: bool) -> bool:
        # Parking may unmap GPU pools on slower peers. Use the CPU coordination
        # group so this agreement cannot launch a NCCL kernel during that window.
        return bool(
            self._reduce_values(
                [int(succeeded)],
                dtype=torch.int64,
                op=torch.distributed.ReduceOp.MIN,
                group=cpu_coordination_group(),
            )[0]
        )

    def barrier(self) -> None:
        import torch.distributed as dist

        if self.context.distributed and dist.is_initialized():
            dist.barrier()

    def coordination_barrier(self) -> None:
        """Wait on the CPU coordination group without launching GPU kernels."""
        if not self.context.distributed:
            return
        group = cpu_coordination_group()
        if group is not None:
            torch.distributed.barrier(group=group)
