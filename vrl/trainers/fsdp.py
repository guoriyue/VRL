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

import logging
from collections.abc import Iterable, Iterator, Mapping
from typing import Any

import torch
from torch import nn

from vrl.models.interfaces.runtime import checkpoint_owned_state_names
from vrl.trainers.distributed import DistributedTrainingContext

logger = logging.getLogger(__name__)


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
    stores native fp16 parameters with outer autocast disabled. ``none`` leaves
    both parameters and reduction in their native dtype.
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
    cpu_offload: bool = False,
) -> nn.Module:
    """Shard ``handle`` in place with FSDP2 and return it.

    Per-block then root: each block becomes its own FSDP module (so its
    params/grads reshard independently) and the root call wraps whatever is left.
    ``reshard_after_forward=True`` is ZeRO-3 (re-gather params each forward, lowest
    memory); ``False`` trades memory for fewer all-gathers. After this the handle's
    parameters are DTensors sharded over ``mesh``; its ``forward`` is unchanged.
    """

    from dataclasses import replace

    from torch.distributed.fsdp import CPUOffloadPolicy, fully_shard

    offload_kwargs = {"offload_policy": CPUOffloadPolicy()} if cpu_offload else {}

    # Only the root module casts forward inputs to the compute dtype; inner blocks
    # receive already-cast activations, so re-casting them is wasted work.
    block_mp_policy = replace(mp_policy, cast_forward_inputs=False)
    base = unwrap_module(handle)
    for block in iter_blocks(base):
        fully_shard(
            block,
            mesh=mesh,
            mp_policy=block_mp_policy,
            reshard_after_forward=reshard_after_forward,
            **offload_kwargs,
        )
    fully_shard(
        handle,
        mesh=mesh,
        mp_policy=mp_policy,
        reshard_after_forward=reshard_after_forward,
        **offload_kwargs,
    )
    return handle


def gather_trainable_state_dict(
    module: nn.Module,
    *,
    rank0_only: bool = False,
) -> dict[str, Any]:
    """Gather only trainable parameters as full CPU tensors on every rank.

    This is both the rollout weight-sync gather (``rank0_only=False``: every
    rank pushes its own gathered weights to its colocated rollout) and the
    rank0-only checkpoint gather.

    Asking DCP for a full state before filtering materializes the frozen base on
    every rank, which defeats LoRA's memory scaling. Keep the state sharded while
    selecting the ``requires_grad`` parameter keys, then all-gather only those
    DTensor leaves. Buffers and frozen checkpoint-owned state are deliberately
    excluded: this is the mutable trainable-state contract, not a standalone
    model checkpoint.

    ``cpu_offload`` MUST stay False: with ``cpu_offload=True`` DCP returns the
    gathered state ONLY on rank0 and an EMPTY dict on every other rank (a
    rank0-only checkpoint optimization). That is wrong for the symmetric colocated
    model, where EACH rank pushes its gathered weights to its own colocated
    rollout — non-rank0 would get nothing and ``select_trainable_state`` then
    reports every LoRA param "missing" (reproduced on a real 2x1 NCCL run; the
    world_size=1 CPU test never hit it because the gather is a no-op there). We
    move the selected tensors to CPU ourselves, so each rank keeps plain CPU tensors.
    """

    from torch.distributed.checkpoint.state_dict import (
        StateDictOptions,
        get_model_state_dict,
    )

    trainable_names = {
        str(name) for name, parameter in module.named_parameters() if parameter.requires_grad
    }
    if not trainable_names:
        raise ValueError(f"{type(module).__name__} has no trainable parameters")

    sharded_state = get_model_state_dict(
        module,
        options=StateDictOptions(
            full_state_dict=False,
            cpu_offload=False,
            ignore_frozen_params=True,
        ),
    )

    import torch.distributed as dist

    keep = not rank0_only or not dist.is_initialized() or dist.get_rank() == 0
    return _gather_named_full_cpu(
        sharded_state,
        trainable_names,
        keep=keep,
        what="trainable parameters",
    )


def gather_checkpoint_state_dict(module: nn.Module) -> dict[str, Any]:
    """Gather exactly trainable plus registered checkpoint-owned state.

    DCP exposes the sharded state mapping without materializing full tensors.
    Selection happens before ``DTensor.full_tensor()``, so frozen base weights
    never enter an all-gather while exceptional frozen mutable state (for
    example DiffusionNFT's ``previous`` adapter) is still checkpointed.
    """

    from torch.distributed.checkpoint.state_dict import (
        StateDictOptions,
        get_model_state_dict,
    )

    owned_names = checkpoint_owned_state_names(module)
    if not owned_names:
        raise ValueError(f"{type(module).__name__} has no checkpoint-owned state")
    trainable_names = frozenset(
        str(name) for name, parameter in module.named_parameters() if parameter.requires_grad
    )
    has_registered_frozen_state = owned_names != trainable_names
    sharded_state = get_model_state_dict(
        module,
        options=StateDictOptions(
            full_state_dict=False,
            cpu_offload=False,
            # The ordinary LoRA path lets DCP omit the frozen base entirely.
            # Opt out only when the model registered a frozen mutable exception.
            ignore_frozen_params=not has_registered_frozen_state,
        ),
    )
    return _gather_named_full_cpu(
        sharded_state,
        owned_names,
        keep=True,
        what="checkpoint-owned state",
    )


def _full_cpu_tensor(value: torch.Tensor, *, keep: bool) -> torch.Tensor | None:
    """Gather one (D)Tensor to a full CPU clone; ``keep=False`` releases it at once.

    Every rank must call this for a DTensor (the all-gather is collective) even
    when only some ranks keep the result. ``Tensor.cpu()`` aliases an already-CPU
    tensor such as Adam's step counter; a checkpoint snapshot must not change
    when the live optimizer takes its next step, so plain tensors are cloned too.
    """

    from torch.distributed.tensor import DTensor

    if isinstance(value, DTensor):
        value = value.full_tensor()
    return value.detach().cpu().clone() if keep else None


def _gather_named_full_cpu(
    sharded_state: Mapping[str, Any],
    names: Iterable[str],
    *,
    keep: bool,
    what: str,
) -> dict[str, Any]:
    """Gather ``names`` out of a sharded model state as full CPU tensors.

    ``what`` is the error domain ("trainable parameters", "checkpoint-owned
    state"); the caller has already decided which names the DCP options expose.
    """

    names = sorted(names)
    missing = sorted(set(names) - set(sharded_state))
    if missing:
        preview = ", ".join(missing[:5])
        suffix = " ..." if len(missing) > 5 else ""
        raise ValueError(f"sharded state is missing {what}: {preview}{suffix}")

    gathered: dict[str, Any] = {}
    # All ranks must enter DTensor collectives in the same order.
    for name in names:
        value = sharded_state[name]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{what} entry {name!r} must be a tensor")
        full = _full_cpu_tensor(value, keep=keep)
        if keep:
            gathered[name] = full
    return gathered


def load_checkpoint_state_dict(
    module: nn.Module,
    state: Mapping[str, Any],
    *,
    strict: bool = True,
) -> None:
    """Load exact checkpoint-owned full tensors into local DTensor shards."""

    from torch.distributed.checkpoint.state_dict import (
        StateDictOptions,
        get_model_state_dict,
        set_model_state_dict,
    )

    if not isinstance(state, Mapping):
        raise TypeError("checkpoint state must be a mapping")

    local_state = get_model_state_dict(
        module,
        options=StateDictOptions(full_state_dict=False, cpu_offload=False),
    )
    owned_names = checkpoint_owned_state_names(module)
    missing_local = sorted(owned_names - set(local_state))
    if missing_local:
        raise ValueError(
            f"FSDP local state is missing checkpoint-owned state before load: {missing_local}",
        )
    missing = sorted(owned_names - set(state))
    unexpected = sorted(set(state) - owned_names)
    if strict and (missing or unexpected):
        raise ValueError(
            f"checkpoint owned-state keys mismatch: missing={missing}, unexpected={unexpected}",
        )

    compatible = {
        name: value for name, value in state.items() if name in owned_names and name in local_state
    }
    if not compatible:
        return
    # DCP performs the layout-aware full-tensor -> DTensor scatter. Its own
    # strict=False is intentional: strictness above applies to exact owned state,
    # while absent immutable base keys are valid in schema v2.
    set_model_state_dict(
        module,
        compatible,
        options=StateDictOptions(
            full_state_dict=True,
            broadcast_from_rank0=True,
            strict=False,
        ),
    )


def load_full_state_dict(
    module: nn.Module,
    state: dict[str, Any],
    *,
    strict: bool = True,
) -> None:
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
        options=StateDictOptions(
            full_state_dict=True,
            broadcast_from_rank0=True,
            strict=strict,
        ),
    )


def _materialize_full_cpu(value: Any, *, keep: bool = True) -> Any:
    """Recursively gather DTensor leaves, retaining them only when requested.

    Optimizer state nests tensors inside dicts/lists (per-param exp_avg /
    exp_avg_sq plus scalar step counters); everything non-tensor passes
    through untouched. ``keep=False`` still enters every DTensor collective but
    releases each full tensor immediately. Checkpoint export uses that path on
    non-primary ranks so only the file-writing rank retains the full CPU tree.
    """

    if isinstance(value, torch.Tensor):
        return _full_cpu_tensor(value, keep=keep)
    if isinstance(value, dict):
        if keep:
            return {key: _materialize_full_cpu(inner, keep=True) for key, inner in value.items()}
        for inner in value.values():
            _materialize_full_cpu(inner, keep=False)
        return {}
    if isinstance(value, (list, tuple)):
        materialized = [_materialize_full_cpu(inner, keep=keep) for inner in value]
        if not keep:
            return () if isinstance(value, tuple) else []
        return tuple(materialized) if isinstance(value, tuple) else materialized
    return value if keep else None


def gather_full_optimizer_state_dict(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    rank0_only: bool = False,
) -> dict[str, Any]:
    """Gather a sharded model's optimizer state into a full CPU dict ON EVERY RANK.

    FSDP2 optimizer state (Adam moments) lives as DTensor shards keyed by the
    optimizer's positional param ids; ``get_optimizer_state_dict`` re-keys by
    parameter FQN (checkpoint-stable across runs) and, with
    ``full_state_dict=True``, all-gathers each moment to a full tensor. The
    every-rank ``cpu_offload=False`` rationale applies here too: with offload
    DCP returns the state only on rank0 and empties elsewhere, which breaks
    every-rank symmetric callers; we move to CPU ourselves. Checkpoint export
    passes ``rank0_only=True`` so every rank joins the collectives while only
    rank0 retains the full CPU tree needed by the sole file writer.
    """

    import torch.distributed as dist

    keep = not rank0_only or not dist.is_initialized() or dist.get_rank() == 0

    from vrl.trainers.optimizer import FP32MasterWeightOptimizer

    if isinstance(optimizer, FP32MasterWeightOptimizer):
        # DCP maps optimizer parameters back to model FQNs by object identity.
        # FP32 masters are deliberately distinct from the BF16 model DTensors,
        # so that mapping cannot apply. The wrapper's state_dict is positional,
        # and OnlineTrainer checkpoints an independent ordered FQN manifest that
        # rejects any parameter-order drift before restore. Non-primary ranks
        # gather one DTensor leaf at a time and discard it immediately.
        return _materialize_full_cpu(optimizer.state_dict(), keep=keep)

    from torch.distributed.checkpoint.state_dict import (
        StateDictOptions,
        get_optimizer_state_dict,
    )

    state = get_optimizer_state_dict(
        model,
        optimizer,
        options=StateDictOptions(
            full_state_dict=True,
            cpu_offload=rank0_only,
        ),
    )
    return _materialize_full_cpu(state, keep=keep)


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

    An ``FP32MasterWeightOptimizer`` holds masters that are DTensors distinct
    from the model's, which DCP cannot re-shard by FQN identity, so its state is
    re-distributed onto the live master layout explicitly instead.
    """

    from vrl.trainers.optimizer import FP32MasterWeightOptimizer

    if isinstance(optimizer, FP32MasterWeightOptimizer):
        optimizer.load_state_dict(
            _distribute_fp32_master_optimizer_state(optimizer, state),
        )
        return

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


def _distribute_fp32_master_optimizer_state(
    optimizer: Any,
    state: Mapping[str, Any],
) -> dict[str, Any]:
    """Restore full positional optimizer tensors onto the masters' DTensor shards.

    ``FP32MasterWeightOptimizer.state_dict`` uses ordinary optimizer parameter
    ids plus an ordered master list. Checkpoint validation separately proves the
    source-parameter FQN order, so this function only owns tensor layout: full
    CPU moment/master tensors become DTensors with the same mesh/placements as
    the corresponding live FP32 master. Scalar step counters remain ordinary
    tensors on the master's local device.
    """

    from torch.distributed.tensor import DTensor, distribute_tensor

    if not isinstance(state, Mapping):
        raise TypeError("distributed FP32-master optimizer state must be a mapping")
    if "state" not in state or "param_groups" not in state:
        raise ValueError("distributed FP32-master optimizer state is incomplete")

    masters = tuple(optimizer.parameters())
    current = optimizer.state_dict()
    current_groups = current.get("param_groups")
    saved_groups = state.get("param_groups")
    if not isinstance(current_groups, list) or not isinstance(saved_groups, list):
        raise ValueError("distributed optimizer param_groups must be lists")
    current_ids = [item for group in current_groups for item in group.get("params", [])]
    saved_ids = [item for group in saved_groups for item in group.get("params", [])]
    if len(current_ids) != len(masters) or len(saved_ids) != len(masters):
        raise ValueError(
            "distributed FP32-master checkpoint parameter count does not match runtime",
        )
    master_by_saved_id = dict(zip(saved_ids, masters, strict=True))

    def distribute_like(value: Any, master: torch.Tensor) -> Any:
        if isinstance(value, Mapping):
            return {key: distribute_like(inner, master) for key, inner in value.items()}
        if isinstance(value, list):
            return [distribute_like(inner, master) for inner in value]
        if isinstance(value, tuple):
            return tuple(distribute_like(inner, master) for inner in value)
        if not isinstance(value, torch.Tensor):
            return value
        if isinstance(master, DTensor) and tuple(value.shape) == tuple(master.shape):
            full = value.to(device=master.device, dtype=value.dtype)
            return distribute_tensor(
                full,
                device_mesh=master.device_mesh,
                placements=master.placements,
                src_data_rank=0,
            )
        return value.to(device=master.device)

    saved_entries = state.get("state")
    if not isinstance(saved_entries, Mapping):
        raise ValueError("distributed optimizer state entries must be a mapping")
    local_entries = {
        saved_id: distribute_like(entry, master_by_saved_id[saved_id])
        for saved_id, entry in saved_entries.items()
    }

    master_state = state.get("fp32_master_weights")
    if not isinstance(master_state, Mapping):
        raise ValueError("optimizer checkpoint is missing fp32_master_weights")
    saved_masters = master_state.get("parameters")
    if not isinstance(saved_masters, list) or len(saved_masters) != len(masters):
        raise ValueError("distributed FP32-master checkpoint master list is invalid")
    local_master_state = dict(master_state)
    local_master_state["parameters"] = [
        None if value is None else distribute_like(value, master)
        for value, master in zip(saved_masters, masters, strict=True)
    ]

    local = dict(state)
    local["state"] = local_entries
    local["fp32_master_weights"] = local_master_state
    return local


def normalize_fsdp_parameter_dtype(
    module: nn.Module,
    target_dtype: torch.dtype,
    *,
    allow_cast: bool,
) -> None:
    """Make an FSDP parameter group uniform without hiding dtype provenance.

    FSDP2 validates original parameter dtypes before its mixed-precision cast.
    Diffusers deliberately leaves a small set of normalization/conditioning
    parameters in FP32 even when the resolved model dtype is BF16. The actor
    policy stores all model parameters in its resolved low precision and relies
    on the FP32-master optimizer for update precision, so normalize those source
    tensors before sharding. A ``none`` policy promises native dtype semantics
    and therefore fails instead of silently changing them.
    """

    if not target_dtype.is_floating_point:
        raise TypeError(f"FSDP target parameter dtype must be floating, got {target_dtype}")
    mismatched = [
        (name, parameter)
        for name, parameter in module.named_parameters()
        if parameter.dtype != target_dtype
    ]
    if not mismatched:
        return
    unsupported = [
        (name, parameter.dtype)
        for name, parameter in mismatched
        if not parameter.is_floating_point()
    ]
    preview = ", ".join(f"{name}:{parameter.dtype}" for name, parameter in mismatched[:8])
    if unsupported or not allow_cast:
        reason = "non-floating parameters" if unsupported else "precision_policy='none'"
        raise ValueError(
            f"FSDP parameter dtype normalization is disabled for {reason}; "
            f"target={target_dtype}, mismatched={preview}",
        )

    source_counts: dict[torch.dtype, int] = {}
    cast_numel = 0
    with torch.no_grad():
        for _, parameter in mismatched:
            source_counts[parameter.dtype] = source_counts.get(parameter.dtype, 0) + 1
            cast_numel += parameter.numel()
            parameter.data = parameter.data.to(dtype=target_dtype)
    logger.info(
        "FSDP parameter dtype normalization: target=%s tensors=%d numel=%d sources=%s examples=%s",
        target_dtype,
        len(mismatched),
        cast_numel,
        {str(dtype): count for dtype, count in source_counts.items()},
        preview,
    )
