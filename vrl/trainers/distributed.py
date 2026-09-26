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
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import timedelta
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
    if full.requires_grad:
        # PyTorch's reduce-scatter backward requires contiguous consumer gradients.
        full.register_hook(lambda gradient: gradient.contiguous())
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
class ContextParallelGroups:
    """A complete world arranged as contiguous CP groups and strided DP groups."""

    cp_group: Any
    dp_group: Any
    cp_size: int
    dp_size: int
    cp_rank: int
    dp_rank: int


def create_context_parallel_groups(cp_size: int) -> ContextParallelGroups:
    """Collectively create DP x CP groups in identical order on every world rank.

    Does not initialize the default group. The process-group owner is also
    responsible for shutdown. CP peers must receive identical replay inputs;
    sampler identity is dp_rank/dp_size, not physical rank/world size.
    """
    import torch.distributed as dist

    if not dist.is_initialized():
        raise RuntimeError("CP groups require an initialized process group")
    world, rank = dist.get_world_size(), dist.get_rank()
    if isinstance(cp_size, bool) or not isinstance(cp_size, int) or cp_size < 2 or world % cp_size:
        raise ValueError("CP size must be an integer >= 2 dividing world size")
    dp_size = world // cp_size
    cp_group = dp_group = None
    for dp_rank in range(dp_size):
        members = list(range(dp_rank * cp_size, (dp_rank + 1) * cp_size))
        group = dist.new_group(members)
        if rank in members:
            cp_group = group
    for cp_rank in range(cp_size):
        members = list(range(cp_rank, world, cp_size))
        group = dist.new_group(members)
        if rank in members:
            dp_group = group
    return ContextParallelGroups(
        cp_group, dp_group, cp_size, dp_size, rank % cp_size, rank // cp_size
    )


def synchronize_context_parallel_rng(
    *, groups: ContextParallelGroups, device: torch.device
) -> None:
    """Copy a CP leader's process RNGs without touching other CUDA devices.

    Call at the strict replay boundary after leader-only collection. Does not
    alter explicit torch.Generator objects such as the prompt sampler. Matching
    RNG state does not make differently shaped dropout/noise draws equivalent;
    stochastic model operations need a separate sharding contract.
    """
    import random

    import numpy as np
    import torch.distributed as dist

    device = torch.device(device)
    if device.type == "cuda" and (
        device.index is None or device.index != torch.cuda.current_device()
    ):
        raise ValueError("CP RNG synchronization requires the current rank-local CUDA device")
    payload = [None]
    if groups.cp_rank == 0:
        payload[0] = {
            "torch": torch.get_rng_state(),
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
        }
    dist.broadcast_object_list(
        payload, src=dist.get_global_rank(groups.cp_group, 0), group=groups.cp_group, device=device
    )
    state = payload[0]
    torch.set_rng_state(state["torch"])
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    if device.type == "cuda":
        torch.cuda.set_rng_state(state["cuda"], device)


def reduce_context_parallel_gradients(
    parameters: Iterable[torch.nn.Parameter], *, groups: ContextParallelGroups
) -> None:
    """SUM CP contributions, then average independent DP replicas in place.

    Call exactly once after local gradient accumulation, before clipping/step;
    do not also use a DDP reducer. Replicated full-output losses must already
    be divided by CP size, and accumulation normalized by the caller. Every
    world rank supplies the same ordered dense parameter list on one device.
    Globally unused gradients remain None (preserving optimizer semantics).
    """
    import torch.distributed as dist

    parameters = list(parameters)
    if not parameters:
        return
    device = parameters[0].device
    if any(parameter.device != device for parameter in parameters):
        raise ValueError("CP gradient reduction requires one parameter device")
    flags = torch.tensor(
        [0 if p.grad is None else (2 if p.grad.is_sparse else 1) for p in parameters],
        dtype=torch.int32,
        device=device,
    )
    dist.all_reduce(flags, op=dist.ReduceOp.MAX)
    active = flags.tolist()
    if 2 in active:
        raise ValueError("CP gradient reduction requires dense gradients")
    for parameter, present in zip(parameters, active, strict=True):
        if not present:
            continue
        if parameter.grad is None:
            parameter.grad = torch.zeros_like(parameter)
        dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM, group=groups.cp_group)
        if groups.dp_size > 1:
            dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM, group=groups.dp_group)
            parameter.grad.div_(groups.dp_size)


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
    # Ranks per context-parallel group (fsdp mesh ["dp_shard", "cp"]). Groups are
    # contiguous: ranks 0..cp_size-1 share one sample's sequence, and so on.
    cp_size: int = 1

    def __post_init__(self) -> None:
        if self.cp_size < 1 or self.world_size % self.cp_size:
            raise ValueError(
                f"cp_size={self.cp_size} must be >= 1 and divide world_size={self.world_size}",
            )

    @property
    def distributed(self) -> bool:
        return self.strategy != "single_process"

    @property
    def is_primary(self) -> bool:
        return self.rank == 0

    @property
    def dp_size(self) -> int:
        """Number of data-parallel replicas: sampler identity is dp, not rank."""
        return self.world_size // self.cp_size

    @property
    def dp_rank(self) -> int:
        return self.rank // self.cp_size

    @property
    def cp_rank(self) -> int:
        return self.rank % self.cp_size

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
            cp_size = 1
            if strategy == "fsdp" and training.fsdp is not None:
                cp_size = int(training.fsdp.context_parallel.size)
                if world_size % cp_size:
                    raise ValueError(
                        f"distributed.training.fsdp.context_parallel size {cp_size} must "
                        f"divide WORLD_SIZE={world_size}",
                    )
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
                cp_size=cp_size,
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
    # Symmetric runs keep torch's default timeout and its exact call shape;
    # only context-parallel runs lengthen it (see collective_timeout).
    timeout_kwargs = {"timeout": collective_timeout(context)} if context.cp_size > 1 else {}
    dist.init_process_group(
        backend=backend,
        rank=context.rank,
        world_size=context.world_size,
        **timeout_kwargs,
    )
    if backend == "nccl":
        # Collective creation: every rank reaches this line inside the same
        # init call, so the subgroup handshake cannot mismatch.
        _CPU_COORDINATION_GROUP = dist.new_group(backend="gloo", **timeout_kwargs)


def collective_timeout(context: DistributedTrainingContext) -> timedelta:
    """How long a rank may wait in a collective before the group gives up.

    Symmetric ranks reach every collective together, so torch's default is
    fine. Context-parallel followers wait in the batch broadcast for the whole
    of the leader's rollout collection (Wan 1.3B: ~35 min for six groups plus
    reward scoring), which exceeds the 30-minute default and killed the first
    cp=2 run; a run-length timeout keeps a genuinely dead peer detectable while
    never racing a healthy rollout.
    """

    if context.cp_size > 1:
        return timedelta(hours=12)
    from torch.distributed import default_pg_timeout

    return default_pg_timeout


def shutdown_training_process_group() -> None:
    """Tear down the process group if one is live (safe to call unconditionally)."""

    import torch.distributed as dist

    global _CPU_COORDINATION_GROUP
    _CPU_COORDINATION_GROUP = None
    if dist.is_initialized():
        dist.destroy_process_group()


@dataclass(frozen=True, slots=True)
class ContextParallelPeerGroup:
    """This rank's context-parallel group for CPU-side coordination.

    ``object_group`` is CPU-capable (gloo): it carries the peers' pickled
    rollout batches without a GPU kernel, so the exchange is safe while a peer
    may still be inside a park/wake window.
    Gradients need no CP collective: parameters shard over the whole world and
    FSDP's reduce-scatter already sums the CP peers (see ``FSDPStrategy.backward``).
    """

    object_group: Any
    cp_size: int
    cp_rank: int


def create_context_parallel_peer_group(
    context: DistributedTrainingContext,
) -> ContextParallelPeerGroup:
    """Collectively create every CP group; return this rank's.

    Every rank must call this in the same order (``dist.new_group`` is
    collective). CP groups are contiguous, matching the row-major rank layout of
    the ``("dp_shard", "ring", "ulysses")`` device mesh.
    """

    import torch.distributed as dist

    if context.cp_size < 2:
        raise ValueError("context-parallel groups require cp_size >= 2")
    if not dist.is_initialized():
        raise RuntimeError("context-parallel groups require an initialized process group")
    mine: Any = None
    for dp_rank in range(context.dp_size):
        members = list(range(dp_rank * context.cp_size, (dp_rank + 1) * context.cp_size))
        group = dist.new_group(members, backend="gloo", timeout=collective_timeout(context))
        if context.rank in members:
            mine = group
    assert mine is not None
    return ContextParallelPeerGroup(
        object_group=mine,
        cp_size=context.cp_size,
        cp_rank=context.cp_rank,
    )


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
