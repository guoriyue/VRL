"""FSDP2 applier: turn a trainable handle into a sharded module and back.

This is the torch-native FSDP2 path (``torch.distributed.fsdp.fully_shard`` +
DTensor), not FSDP1. The design is the standard diffusion ZeRO-3 parallelizer:
shard each transformer *block* with ``fully_shard`` and then the root, deriving
the block boundaries from the model's ``_no_split_modules``. There is no
tensor/pipeline parallel —
diffusion + LoRA is a pure ZeRO-3 (sharded-params/grads/optim) workload (see
``SPRINT_multi_gpu_training.md`` §10.5 for why FSDP2, not Megatron).

Everything here is collective code: it needs an initialized process group. It is
exercised on a single CPU rank (``world_size=1`` + gloo) in
``tests/trainers/test_fsdp.py``, which is enough to prove wrapping, forward /
backward, full-state gather, and load round-trip without real multi-GPU.

Boundaries this module does NOT cross (they belong to later phases of
``SPRINT_multi_gpu_training.md``): the online GRPO rank-split collect/train loop,
the torchrun↔Ray rollout coordination, and optimizer/EMA state sharding.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import torch
from torch import nn

from vrl.trainers.distributed import DistributedTrainingContext


def init_training_process_group(
    context: DistributedTrainingContext,
    *,
    backend: str = "nccl",
) -> None:
    """Create the torch.distributed process group for an fsdp rank.

    No-op for ``single_process`` and when a group already exists. The owning
    ``Strategy.shutdown`` calls the matching ``shutdown_training_process_group``.
    ``torchrun`` has already exported ``RANK`` / ``WORLD_SIZE`` / ``MASTER_ADDR``;
    ``resolve_training_context`` validated them, so ``init_method='env'`` is the
    only contract we rely on here.
    """

    import torch.distributed as dist

    if not context.distributed or dist.is_initialized():
        return
    if context.device.type == "cuda":
        torch.cuda.set_device(context.local_rank)
    dist.init_process_group(
        backend=backend,
        rank=context.rank,
        world_size=context.world_size,
    )


def shutdown_training_process_group() -> None:
    """Tear down the process group if one is live (safe to call unconditionally)."""

    import torch.distributed as dist

    if dist.is_initialized():
        dist.destroy_process_group()


def build_fsdp_mesh(context: DistributedTrainingContext, mesh_dims: list[str]) -> Any:
    """Build the 1D DeviceMesh FSDP2 shards over.

    ``["dp_shard"]`` is plain ZeRO-3 across the whole world — the single-node
    start point (sprint §3). 2D HSDP (``["dp_replicate", "dp_shard"]``, replicate
    across nodes / shard within one) needs ``num_nodes`` * ``gpus_per_node`` from
    config, which the single-process-shaped context here does not carry; it is the
    multi-node follow-on and fail-fasts rather than half-working.
    """

    from torch.distributed.device_mesh import init_device_mesh

    if mesh_dims != ["dp_shard"]:
        raise ValueError(
            f"distributed.training.fsdp.mesh only supports 1D ['dp_shard'] for now, "
            f"got {mesh_dims!r}; 2D HSDP is the multi-node follow-on "
            "(SPRINT_multi_gpu_training.md §3).",
        )
    return init_device_mesh(
        context.device.type,
        (context.world_size,),
        mesh_dim_names=("dp_shard",),
    )


def mixed_precision_policy(
    name: str,
    *,
    parameter_dtype: torch.dtype | None = None,
) -> Any:
    """Map the config knob to an FSDP2 ``MixedPrecisionPolicy``.

    ``actor`` keeps the model builder's resolved parameter dtype and performs
    gradient reduction in fp32. FSDP must not introduce a second parameter-dtype
    owner: most families follow top-level precision, while SANA deliberately
    stores fp16 parameters under bf16 forward autocast. ``none`` leaves both
    parameters and reduction in their native dtype.
    """

    from torch.distributed.fsdp import MixedPrecisionPolicy

    if name == "none":
        return MixedPrecisionPolicy()
    if name == "actor":
        if parameter_dtype is None:
            raise ValueError("FSDP actor precision requires the resolved parameter dtype")
        return MixedPrecisionPolicy(
            param_dtype=parameter_dtype,
            reduce_dtype=torch.float32,
        )
    raise ValueError(
        f"distributed.training.fsdp.precision_policy must be 'actor' or 'none', got {name!r}",
    )


def unwrap_module(module: Any) -> nn.Module:
    """Peel torch.compile and PEFT wrappers off a trainable handle.

    The handle the policy trains may be ``compile(PeftModel(transformer))``;
    block discovery and root identification need the underlying library model
    (the diffusers DiT / Llama trunk that actually owns ``_no_split_modules`` and
    the transformer blocks). torch.compile exposes the inner module as
    ``_orig_mod``; PEFT exposes the base model via ``get_base_model()`` (falling
    back to ``base_model.model``). Loop because wrapper order/nesting varies.
    """

    seen: set[int] = set()
    while id(module) not in seen:
        seen.add(id(module))
        inner = getattr(module, "_orig_mod", module)
        if inner is not module:
            module = inner
            continue
        get_base = getattr(module, "get_base_model", None)
        if callable(get_base):
            module = get_base()
            continue
        base_model = getattr(module, "base_model", None)
        inner_model = getattr(base_model, "model", None)
        if inner_model is not None:
            module = inner_model
            continue
        break
    return module


def iter_blocks(base: nn.Module) -> Iterator[nn.Module]:
    """Yield the transformer blocks to shard individually.

    diffusers DiTs and Llama-style trunks list their per-layer block *class
    names* in ``_no_split_modules`` (the same list ``device_map='auto'`` uses to
    keep a layer un-split); we shard every submodule whose class name is in it.
    Deriving block boundaries from ``getattr(model, "_no_split_modules", None)``
    keeps the applier family-agnostic — change the mesh, not per-family hooks.
    """

    no_split = getattr(base, "_no_split_modules", None)
    if not no_split:
        raise ValueError(
            f"{type(base).__name__} exposes no `_no_split_modules`; FSDP2 needs the "
            "transformer block boundaries to shard per layer. Set them on the model "
            "or extend the applier with an explicit block list for this family.",
        )
    block_names = set(no_split)
    for submodule in base.modules():
        if type(submodule).__name__ in block_names:
            yield submodule


def apply_fsdp(
    handle: nn.Module,
    *,
    mesh: Any,
    mp_policy: Any,
    reshard_after_forward: bool = True,
) -> nn.Module:
    """Shard ``handle`` in place with FSDP2 and return it.

    Per-block then root: each block becomes its own FSDP module (so its
    params/grads reshard independently) and the root call wraps whatever is left.
    ``reshard_after_forward=True`` is ZeRO-3 (re-gather params each forward, lowest
    memory); ``False`` trades memory for fewer all-gathers. After this the handle's
    parameters are DTensors sharded over ``mesh``; its ``forward`` is unchanged.
    """

    from torch.distributed.fsdp import fully_shard

    base = unwrap_module(handle)
    for block in iter_blocks(base):
        fully_shard(
            block, mesh=mesh, mp_policy=mp_policy, reshard_after_forward=reshard_after_forward
        )
    fully_shard(
        handle, mesh=mesh, mp_policy=mp_policy, reshard_after_forward=reshard_after_forward
    )
    return handle


def gather_full_state_dict(module: nn.Module) -> dict[str, Any]:
    """Gather a sharded module's params into a full CPU state dict ON EVERY RANK.

    FSDP2 holds each parameter as a DTensor shard; checkpoint and rollout sync
    both need ordinary full tensors in the unwrapped, policy-facing key space.
    ``get_model_state_dict(full_state_dict=True)`` all-gathers + materializes each
    param to a full tensor, and (verified on a single CPU rank) returns plain
    ``torch.Tensor`` leaves with no ``module.`` / ``_orig_mod.`` / FSDP shard-key
    leakage.

    ``cpu_offload`` MUST stay False: with ``cpu_offload=True`` DCP returns the full
    state ONLY on rank0 and an EMPTY dict on every other rank (a rank0-only
    checkpoint optimization). That is wrong for the symmetric colocated model, where
    EACH rank pushes its gathered weights to its own colocated rollout — non-rank0
    would get nothing and ``select_trainable_state`` then reports every LoRA param
    "missing" (reproduced on a real 2x1 NCCL run; the world_size=1 CPU test never
    hit it because the gather is a no-op there). We move the full tensors to CPU
    ourselves below, so each rank still ends with plain CPU tensors. The first
    version is rank-replicated full state, not sharded checkpoints
    (``SPRINT_multi_gpu_training.md`` §8).
    """

    from torch.distributed.checkpoint.state_dict import (
        StateDictOptions,
        get_model_state_dict,
    )
    from torch.distributed.tensor import DTensor

    state = get_model_state_dict(
        module,
        options=StateDictOptions(full_state_dict=True, cpu_offload=False),
    )
    # full_state_dict=True already all-gathers each param to a full tensor;
    # defensively materialize any DTensor leaf (no-op gather in a real single-mesh
    # run) so the contract — plain full CPU tensors — holds regardless of how many
    # meshes the process has built.
    return {
        key: (value.full_tensor() if isinstance(value, DTensor) else value).detach().cpu()
        for key, value in state.items()
    }


def load_full_state_dict(module: nn.Module, state: dict[str, Any]) -> None:
    """Load a full (unsharded) state dict back into a sharded module on all ranks.

    ``broadcast_from_rank0=True`` lets rank0 hold the only full copy (the others
    pass an empty/placeholder dict) and FSDP scatters the shards — the resume
    contract from ``SPRINT_multi_gpu_training.md`` §8 (all ranks end up with the
    same trainable state). A no-op broadcast at ``world_size=1``.
    """

    from torch.distributed.checkpoint.state_dict import (
        StateDictOptions,
        set_model_state_dict,
    )

    set_model_state_dict(
        module,
        state,
        options=StateDictOptions(full_state_dict=True, broadcast_from_rank0=True),
    )


def _materialize_full_cpu(value: Any) -> Any:
    """Recursively gather DTensor leaves to full CPU tensors.

    Optimizer state nests tensors inside dicts/lists (per-param exp_avg /
    exp_avg_sq plus scalar step counters); everything non-tensor passes
    through untouched.
    """

    from torch.distributed.tensor import DTensor

    if isinstance(value, DTensor):
        return value.full_tensor().detach().cpu()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _materialize_full_cpu(inner) for key, inner in value.items()}
    if isinstance(value, (list, tuple)):
        materialized = [_materialize_full_cpu(inner) for inner in value]
        return tuple(materialized) if isinstance(value, tuple) else materialized
    return value


def gather_full_optimizer_state_dict(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> dict[str, Any]:
    """Gather a sharded model's optimizer state into a full CPU dict ON EVERY RANK.

    FSDP2 optimizer state (Adam moments) lives as DTensor shards keyed by the
    optimizer's positional param ids; ``get_optimizer_state_dict`` re-keys by
    parameter FQN (checkpoint-stable across runs) and, with
    ``full_state_dict=True``, all-gathers each moment to a full tensor. Same
    ``cpu_offload=False`` rationale as ``gather_full_state_dict``: with offload
    DCP returns the state only on rank0 and empties elsewhere, which breaks
    every-rank symmetric callers; we move to CPU ourselves.
    """

    from torch.distributed.checkpoint.state_dict import (
        StateDictOptions,
        get_optimizer_state_dict,
    )

    state = get_optimizer_state_dict(
        model,
        optimizer,
        options=StateDictOptions(full_state_dict=True, cpu_offload=False),
    )
    return _materialize_full_cpu(state)


def load_full_optimizer_state_dict(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    state: dict[str, Any],
) -> None:
    """Load a full (FQN-keyed) optimizer state back into a sharded model's optimizer.

    Mirrors ``load_full_state_dict``: every resuming rank read the checkpoint
    file itself, ``broadcast_from_rank0=True`` makes rank0's copy authoritative
    and DCP re-shards each moment onto the local DTensor layout. A no-op
    broadcast at ``world_size=1``.
    """

    from torch.distributed.checkpoint.state_dict import (
        StateDictOptions,
        _init_optim_state,
        set_optimizer_state_dict,
    )

    # A freshly-built optimizer has NO state yet; DCP needs materialized local
    # state to locate devices/layouts before it can re-shard the checkpoint
    # onto it. _init_optim_state is DCP's own zero-grad-step initializer (a
    # no-op when state already exists). It early-returns if gradients are
    # pending — resume runs before any training step, so none are.
    _init_optim_state(optimizer)
    set_optimizer_state_dict(
        model,
        optimizer,
        state,
        options=StateDictOptions(full_state_dict=True, broadcast_from_rank0=True),
    )
