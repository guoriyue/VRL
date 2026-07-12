"""Training strategy seam: the boundary between the trainer and how it runs.

The trainer drives the GRPO loop; *how* a step executes on the hardware —
backward, grad clipping, and trainable-state export/load — goes through a
``Strategy`` so the trainer never hard-codes single-process vs FSDP2.

This readiness sprint ships only ``SingleProcessStrategy`` (current behavior
moved behind the protocol, byte-for-byte). The FSDP2 strategy — DTensor-aware
clip and full-state export, a real ``barrier`` — lands in
``SPRINT_multi_gpu_training.md`` and slots in here without touching the trainer.
"""

from __future__ import annotations

import gc
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol

import torch
from torch import nn

from vrl.trainers.distributed import DistributedTrainingContext
from vrl.utils.config import cfg_path
from vrl.utils.cuda_memory import empty_cuda_cache


@dataclass(frozen=True, slots=True)
class TrainingMemoryState:
    """Live trainer-owned state that must leave a shared GPU for rollout.

    The trainer builds this value at the start of every rollout phase.  Keeping
    object discovery there is important: optimizer and EMA state are lazy, and
    streaming accumulation can add live gradients between two rollout phases.
    The strategy owns *how* those objects move; rollout orchestration must never
    inspect their internals.
    """

    model: nn.Module
    ref_model: nn.Module | None
    optimizer: torch.optim.Optimizer | None
    ema: Any | None
    grad_scaler: Any | None
    device: torch.device


@dataclass(slots=True)
class _ModuleRestore:
    module: Any
    device: torch.device


@dataclass(slots=True)
class _TensorRestore:
    tensor: torch.Tensor
    device: torch.device


@dataclass(slots=True)
class _ParkedTrainingState:
    state: TrainingMemoryState
    key: tuple[int, int, int, int, int, str]
    modules: list[_ModuleRestore]
    tensors: list[_TensorRestore]
    ema_device: Any | None


class Strategy(Protocol):
    """How one training step executes; the only seam the trainer depends on."""

    context: DistributedTrainingContext

    def prepare_model(self, model: Any) -> Any:
        """Return the model the trainer should train (wrapped if the backend needs it).

        Identity for single process; FSDP2 shards the trainable handle here. The
        trainer routes its model through this once at construction so it never
        hard-codes the wrapping.
        """
        ...

    def backward(self, loss: torch.Tensor, *, grad_scaler: Any | None = None) -> None:
        """Run the backward pass (scaled when an fp16 GradScaler is active)."""
        ...

    def clip_grad_norm(
        self,
        parameters: Iterable[nn.Parameter],
        max_norm: float,
    ) -> float:
        """Clip gradients in place and return the pre-clip total norm."""
        ...

    def export_trainable_state(self, bundle: Any) -> dict[str, dict[str, Any]]:
        """Checkpoint-facing trainable state (nested by module name, CPU tensors)."""
        ...

    def export_rollout_state(self, bundle: Any) -> dict[str, Any]:
        """Rollout-facing flat trainable state (unwrapped, policy-facing keys)."""
        ...

    def load_trainable_state(self, bundle: Any, state: dict[str, Any]) -> None:
        """Load a checkpoint-facing trainable state back into the bundle."""
        ...

    def export_optimizer_state(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
    ) -> dict[str, Any]:
        """Checkpoint-facing optimizer state (full tensors under sharding)."""
        ...

    def load_optimizer_state(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        state: dict[str, Any],
    ) -> None:
        """Load a checkpoint-facing optimizer state back (re-shard under FSDP)."""
        ...

    def validate_training_state_parking(self) -> None:
        """Fail before collection when this strategy cannot park shared-GPU state."""
        ...

    def park_training_state(self, state: TrainingMemoryState) -> None:
        """Move all live trainer-owned CUDA state out of the rollout GPU."""
        ...

    def restore_training_state(self, state: TrainingMemoryState) -> None:
        """Restore the state previously parked by ``park_training_state``."""
        ...

    def barrier(self) -> None:
        """Synchronize all training ranks (no-op for single process)."""
        ...

    def shutdown(self, *, restore_parked: bool = True) -> None:
        """Release resources, restoring parked GPU state only when ownership is safe."""
        ...


class SingleProcessStrategy(Strategy):
    """The current single-GPU behavior, moved behind the strategy protocol.

    Every method here is the existing trainer / checkpoint / weight-sync logic
    verbatim; this installs the seam without changing what a single-process run
    does. ``context`` defaults to a rank0/world1 identity.
    """

    def __init__(self, context: DistributedTrainingContext | None = None) -> None:
        self.context = context or _single_process_context()
        self._parked_training_state: _ParkedTrainingState | None = None

    def prepare_model(self, model: Any) -> Any:
        # Single process trains the model as-is; the seam exists so FSDP2 can wrap
        # without the trainer changing.
        return model

    def backward(self, loss: torch.Tensor, *, grad_scaler: Any | None = None) -> None:
        if grad_scaler is not None:
            grad_scaler.scale(loss).backward()
        else:
            loss.backward()

    def clip_grad_norm(
        self,
        parameters: Iterable[nn.Parameter],
        max_norm: float,
    ) -> float:
        return float(nn.utils.clip_grad_norm_(parameters, max_norm))

    def export_trainable_state(self, bundle: Any) -> dict[str, dict[str, Any]]:
        from vrl.trainers.checkpointing import export_trainable_state

        return export_trainable_state(bundle)

    def export_rollout_state(self, bundle: Any) -> dict[str, Any]:
        from vrl.trainers.weight_sync import build_trainable_state_sync_getter, to_cpu_snapshot

        return to_cpu_snapshot(build_trainable_state_sync_getter(bundle)())

    def load_trainable_state(self, bundle: Any, state: dict[str, Any]) -> None:
        from vrl.trainers.checkpointing import load_trainable_state

        load_trainable_state(bundle, state)

    def export_optimizer_state(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
    ) -> dict[str, Any]:
        del model  # plain tensors: torch's native (positional-id) format suffices
        return optimizer.state_dict()

    def load_optimizer_state(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        state: dict[str, Any],
    ) -> None:
        del model
        optimizer.load_state_dict(state)

    def validate_training_state_parking(self) -> None:
        return None

    def park_training_state(self, state: TrainingMemoryState) -> None:
        """Park model, optimizer, EMA, scaler, and live gradients on CPU.

        Tensor objects are moved in place so optimizer/EMA/scaler aliases remain
        valid.  A single identity set spans every owner, preventing a tensor that
        appears through two live structures from being moved twice.
        """

        self.validate_training_state_parking()
        if not isinstance(state, TrainingMemoryState):
            raise TypeError("training state parking requires TrainingMemoryState")
        key = _training_state_key(state)
        if self._parked_training_state is not None:
            if self._parked_training_state.key == key:
                return
            raise RuntimeError(
                "cannot park different training state before restoring the current phase"
            )

        parked = _ParkedTrainingState(
            state=state,
            key=key,
            modules=[],
            tensors=[],
            ema_device=getattr(state.ema, "device", None),
        )
        self._parked_training_state = parked
        seen_objects: set[int] = set()
        seen_tensors: set[int] = set()
        try:
            for module in (state.model, state.ref_model):
                if module is None or id(module) in seen_objects:
                    continue
                seen_objects.add(id(module))
                parked.modules.append(
                    _ModuleRestore(module=module, device=_module_device(module, state.device)),
                )
                seen_tensors.update(_module_tensor_ids(module))
                _move_module(module, torch.device("cpu"))

            if state.optimizer is not None:
                _move_tensor_tree_in_place(
                    state.optimizer.state,
                    torch.device("cpu"),
                    seen=seen_tensors,
                    restores=parked.tensors,
                )
            if state.ema is not None:
                _move_tensor_tree_in_place(
                    getattr(state.ema, "ema_parameters", ()),
                    torch.device("cpu"),
                    seen=seen_tensors,
                    restores=parked.tensors,
                )
                _move_tensor_tree_in_place(
                    getattr(state.ema, "temp_stored_parameters", ()),
                    torch.device("cpu"),
                    seen=seen_tensors,
                    restores=parked.tensors,
                )
                if hasattr(state.ema, "device"):
                    state.ema.device = torch.device("cpu")
            if state.grad_scaler is not None:
                for attr in ("_scale", "_growth_tracker", "_per_optimizer_states"):
                    _move_tensor_tree_in_place(
                        getattr(state.grad_scaler, attr, None),
                        torch.device("cpu"),
                        seen=seen_tensors,
                        restores=parked.tensors,
                    )
            _release_training_cuda_memory()
        except BaseException:
            try:
                self._restore_parked_training_state(parked)
            except BaseException as rollback_error:
                raise RuntimeError(
                    "training-state parking failed and rollback could not restore the trainer",
                ) from rollback_error
            self._parked_training_state = None
            raise

    def restore_training_state(self, state: TrainingMemoryState) -> None:
        parked = self._parked_training_state
        if parked is None:
            return
        same_state = parked.key == _training_state_key(state)
        self._restore_parked_training_state(parked)
        self._parked_training_state = None
        if not same_state:
            raise RuntimeError("trainer-owned state changed while its GPU memory was parked")

    def _restore_parked_training_state(self, parked: _ParkedTrainingState) -> None:
        for module in parked.modules:
            _move_module(module.module, module.device)
        for restore in reversed(parked.tensors):
            _move_tensor_data(restore.tensor, restore.device)
        ema = parked.state.ema
        if ema is not None and hasattr(ema, "device"):
            ema.device = parked.ema_device
        empty_cuda_cache()

    def barrier(self) -> None:
        return None

    def shutdown(self, *, restore_parked: bool = True) -> None:
        if self._parked_training_state is not None and restore_parked:
            self.restore_training_state(self._parked_training_state.state)
        elif self._parked_training_state is not None:
            # Terminal role cleanup could not prove the shared GPU was released.
            # Drop only this adapter's restore ticket; the live objects deliberately
            # remain on CPU until process exit instead of racing another GPU owner.
            self._parked_training_state = None


def _training_state_key(state: TrainingMemoryState) -> tuple[int, int, int, int, int, str]:
    return (
        id(state.model),
        id(state.ref_model),
        id(state.optimizer),
        id(state.ema),
        id(state.grad_scaler),
        str(state.device),
    )


def _module_device(module: Any, fallback: torch.device) -> torch.device:
    for tensor in _module_tensors(module):
        return torch.device(tensor.device)
    return torch.device(fallback)


def _module_tensors(module: Any) -> Iterable[torch.Tensor]:
    parameters = getattr(module, "parameters", None)
    if callable(parameters):
        for parameter in parameters():
            if isinstance(parameter, torch.Tensor):
                yield parameter
                if parameter.grad is not None:
                    yield parameter.grad
    buffers = getattr(module, "buffers", None)
    if callable(buffers):
        yield from (buffer for buffer in buffers() if isinstance(buffer, torch.Tensor))


def _module_tensor_ids(module: Any) -> set[int]:
    return {id(tensor) for tensor in _module_tensors(module)}


def _move_module(module: Any, device: torch.device) -> None:
    move = getattr(module, "to", None)
    if not callable(move):
        raise TypeError(f"training module {type(module).__name__} does not expose to(device)")
    move(device)
    move_frozen = getattr(module, "move_frozen_components", None)
    if callable(move_frozen):
        move_frozen(device)


def _move_tensor_tree_in_place(
    value: Any,
    device: torch.device,
    *,
    seen: set[int],
    restores: list[_TensorRestore],
) -> None:
    if isinstance(value, torch.Tensor):
        tensor_id = id(value)
        if tensor_id in seen:
            return
        seen.add(tensor_id)
        original_device = torch.device(value.device)
        if original_device != device:
            restores.append(_TensorRestore(tensor=value, device=original_device))
            _move_tensor_data(value, device)
        return
    if isinstance(value, Mapping):
        for child in value.values():
            _move_tensor_tree_in_place(child, device, seen=seen, restores=restores)
        return
    if isinstance(value, (list, tuple, set)):
        for child in value:
            _move_tensor_tree_in_place(child, device, seen=seen, restores=restores)


def _move_tensor_data(tensor: torch.Tensor, device: torch.device) -> None:
    if torch.device(tensor.device) == device:
        return
    tensor.data = tensor.data.to(device=device)


def _release_training_cuda_memory() -> None:
    """Release trainer allocator pages; any CUDA failure invalidates the handoff."""

    gc.collect()
    if not torch.cuda.is_available():
        return
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def _single_transformer_handle(model: Any) -> tuple[Any, Any]:
    """The trainable ``transformer`` handle + its writer, shared by FSDP2 and DDP.

    Diffusion policies expose the trainable root as a ``transformer`` handle plus
    ``_set_transformer`` to write the wrapped module back (so the pipeline and
    attention processors keep pointing at it). AR families do not yet expose
    explicit trainable roots, and dual-stage Wan exposes a second trainable root
    (whose DEFAULT trainable set is ``("transformer_2",)``), so both fail-fast here
    — blindly wrapping ``transformer`` could wrap a frozen module and leave the real
    trainable one unmanaged (SPRINT_multi_gpu_training.md §5).
    """

    handle = getattr(model, "transformer", None)
    set_transformer = getattr(model, "_set_transformer", None)
    if handle is None or not callable(set_transformer):
        raise NotImplementedError(
            "multi-GPU model wrapping needs a diffusion policy exposing a `transformer` "
            f"handle and `_set_transformer`; {type(model).__name__} exposes neither. "
            "AR families (janus_pro / nextstep_1) need explicit trainable roots first "
            "(SPRINT_multi_gpu_training.md §5).",
        )
    trainable = getattr(model, "trainable_modules", None)
    if isinstance(trainable, Mapping) and set(trainable) != {"transformer"}:
        raise NotImplementedError(
            f"multi-GPU wrapping wraps only the single 'transformer' handle, but "
            f"{type(model).__name__}.trainable_modules = {sorted(trainable)}. "
            "Multi-transformer wrapping (e.g. dual-stage Wan transformer_2) needs "
            "per-handle writers and is not wired yet (SPRINT_multi_gpu_training.md §5).",
        )
    return handle, set_transformer


class FSDPStrategy(Strategy):
    """FSDP2 (``fully_shard`` + DTensor) training behind the same seam.

    The model wraps once in ``prepare_model``; thereafter params/grads/optimizer
    state live as DTensor shards over the mesh. Checkpoint and rollout export both
    gather a full, unwrapped, policy-facing state on rank0 — the trainer and the
    Ray rollout workers never see a shard or a wrapper key. The collective work
    lives in ``vrl/trainers/fsdp.py``; this class is the trainer-facing adapter.

    This is the strategy *layer* of ``SPRINT_multi_gpu_training.md``. The online
    recipe drives it through the same per-rank-local symmetric-colocated path as
    DDP: every torchrun rank owns a local rollout/training device, while FSDP
    handles DTensor sharding and collectives behind this adapter.
    """

    def __init__(
        self,
        context: DistributedTrainingContext,
        *,
        mesh_dims: list[str] | None = None,
        precision_policy: str = "actor",
        reshard_after_forward: bool = True,
    ) -> None:
        self.context = context
        self._mesh_dims = mesh_dims or ["dp_shard"]
        self._precision_policy = precision_policy
        self._reshard_after_forward = reshard_after_forward
        self._mesh: Any | None = None  # built on first prepare_model (needs a live PG)

    def _ensure_mesh(self) -> Any:
        if self._mesh is None:
            from vrl.trainers.fsdp import build_fsdp_mesh

            self._mesh = build_fsdp_mesh(self.context, self._mesh_dims)
        return self._mesh

    def prepare_model(self, model: Any) -> Any:
        """Shard the policy's trainable transformer in place and return the policy."""
        from vrl.trainers.fsdp import (
            apply_fsdp,
            init_training_process_group,
            mixed_precision_policy,
        )

        # Validate the trainable handle BEFORE touching the process group so a bad
        # model fails fast (mirrors DDPStrategy; the guard tests need no live PG).
        handle, set_transformer = _single_transformer_handle(model)
        # Create the process group + bind this rank's cuda device up front, exactly
        # like DDPStrategy. init_device_mesh would lazily auto-init a default group,
        # but it would NOT call torch.cuda.set_device(local_rank) first, so the NCCL
        # group and the per-block fully_shard could bind the wrong card on a
        # single-node multi-GPU box. Doing it here keeps the two strategies
        # symmetric and the device choice explicit. No-op for single_process and
        # when a group already exists (the CPU gloo test fixture pre-inits one).
        backend = "gloo" if self.context.device.type == "cpu" else "nccl"
        init_training_process_group(self.context, backend=backend)
        wrapped = apply_fsdp(
            handle,
            mesh=self._ensure_mesh(),
            mp_policy=mixed_precision_policy(self._precision_policy),
            reshard_after_forward=self._reshard_after_forward,
        )
        set_transformer(wrapped)
        return model

    def backward(self, loss: torch.Tensor, *, grad_scaler: Any | None = None) -> None:
        # FSDP2 reduce-scatters gradients inside the backward hooks; the bf16 actor
        # recipe runs without a GradScaler, but keep the seam identical to single
        # process so the trainer loop is backend-agnostic.
        if grad_scaler is not None:
            grad_scaler.scale(loss).backward()
        else:
            loss.backward()

    def clip_grad_norm(self, parameters: Iterable[nn.Parameter], max_norm: float) -> float:
        # torch.nn.utils.clip_grad_norm_ is DTensor-aware: it reduces the global
        # norm across the mesh and clips the local shards. float() collapses the
        # replicated norm scalar to a Python float for logging.
        return float(nn.utils.clip_grad_norm_(parameters, max_norm))

    def _gather_unwrapped(self, module: Any) -> tuple[Any, dict[str, Any]]:
        """Peel compile/DDP, then gather the sharded module to a full state.

        Mirrors single-process: ``get_model_state_dict`` strips the
        ``_orig_mod.`` compile prefix while ``named_parameters()`` keeps it, so a
        sharded gather + trainable-key select must run on the SAME uncompiled
        module or the two key sets disagree (the select would drop everything).
        Returns the unwrapped module (for trainable-name selection) and its full
        state. PEFT is kept so LoRA keys stay policy-facing.
        """
        from vrl.trainers.fsdp import gather_full_state_dict
        from vrl.trainers.weight_sync import unwrap_compile_and_ddp

        inner = unwrap_compile_and_ddp(module)
        return inner, gather_full_state_dict(inner)

    def export_trainable_state(self, bundle: Any) -> dict[str, dict[str, Any]]:
        from vrl.trainers.weight_sync import require_trainable_modules, to_cpu

        modules = require_trainable_modules(bundle)
        return {
            name: to_cpu(self._gather_unwrapped(module)[1]) for name, module in modules.items()
        }

    def export_rollout_state(self, bundle: Any) -> dict[str, Any]:
        from vrl.trainers.weight_sync import (
            require_trainable_modules,
            select_trainable_state,
            to_cpu_snapshot,
        )

        modules = require_trainable_modules(bundle)
        state: dict[str, Any] = {}
        # Gather each sharded module to a full state, then pick the trainable keys
        # the same way single process does — so the rollout-facing key space is
        # identical whether or not the trainer was sharded.
        for module_name, module in modules.items():
            inner, full = self._gather_unwrapped(module)
            state.update(select_trainable_state(inner, str(module_name), full))
        if not state:
            raise ValueError("trainable module state is empty")
        return to_cpu_snapshot(state)

    def load_trainable_state(self, bundle: Any, state: dict[str, Any]) -> None:
        from vrl.trainers.fsdp import load_full_state_dict
        from vrl.trainers.weight_sync import require_trainable_modules, unwrap_compile_and_ddp

        modules = require_trainable_modules(bundle)
        # `state` is the checkpoint-facing shape: nested by trainable module name
        # (what export_trainable_state produced). Load into the same uncompiled
        # namespace the export gathered from.
        for name, module in modules.items():
            if name in state:
                load_full_state_dict(unwrap_compile_and_ddp(module), state[name])

    def export_optimizer_state(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
    ) -> dict[str, Any]:
        # COLLECTIVE (all-gathers each Adam moment): run on every rank; FQN-keyed
        # so the checkpoint does not depend on optimizer param ordering.
        from vrl.trainers.fsdp import gather_full_optimizer_state_dict

        return gather_full_optimizer_state_dict(model, optimizer)

    def load_optimizer_state(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        state: dict[str, Any],
    ) -> None:
        from vrl.trainers.fsdp import load_full_optimizer_state_dict

        load_full_optimizer_state_dict(model, optimizer, state)

    def validate_training_state_parking(self) -> None:
        raise NotImplementedError(
            "shared-GPU on-demand rollout is not supported with FSDP: DTensor model, "
            "optimizer, EMA, GradScaler, and live-gradient parking has not been "
            "implemented collectively. Use disjoint rollout GPUs.",
        )

    def park_training_state(self, state: TrainingMemoryState) -> None:
        self.validate_training_state_parking()

    def restore_training_state(self, state: TrainingMemoryState) -> None:
        self.validate_training_state_parking()

    def barrier(self) -> None:
        import torch.distributed as dist

        if dist.is_initialized():
            dist.barrier()

    def shutdown(self, *, restore_parked: bool = True) -> None:
        del restore_parked
        from vrl.trainers.fsdp import shutdown_training_process_group

        shutdown_training_process_group()


class DDPStrategy(Strategy):
    """DistributedDataParallel training behind the same seam.

    For a model that fits on one card (a 2B diffusion transformer + LoRA does), DDP
    replicates the full module on every rank and all-reduces gradients in the
    backward hooks — simpler and cheaper than FSDP2's shard/all-gather, which only
    earns its keep when the model does NOT fit. Because every rank keeps FULL
    params, the checkpoint/rollout export is the same full-state path FSDP uses on
    the unwrapped module (``unwrap_compile_and_ddp`` peels DDP's ``.module``; the
    "gather" is a no-op at the plain-tensor level). Only ``prepare_model`` (wrap)
    diverges from FSDPStrategy.

    Symmetric half of ``SPRINT_symmetric_colocated_ddp.md``. The online multi-rank
    loop that drives N ranks is a separate, not-yet-wired phase, so the online
    recipe still gates non-single_process strategies (see ``run_online_recipe``);
    the strategy itself is exercised on a single CPU rank in
    ``tests/trainers/test_ddp.py``.
    """

    def __init__(
        self,
        context: DistributedTrainingContext,
        *,
        find_unused_parameters: bool = False,
    ) -> None:
        self.context = context
        self._find_unused_parameters = find_unused_parameters

    def prepare_model(self, model: Any) -> Any:
        """Replicate the policy's trainable transformer with DDP and return the policy.

        Wrap only the trainable ``transformer`` handle, not the whole policy: the
        frozen base inside still has ``requires_grad=False`` so it stays out of
        DDP's reducer buckets, whereas wrapping the top-level model would drag the
        frozen VAE/text-encoder into DDP and force ``find_unused_parameters=True``.
        """
        from torch.nn.parallel import DistributedDataParallel

        from vrl.trainers.fsdp import init_training_process_group

        # Validate the trainable handle BEFORE touching the process group so a bad
        # model fails fast (and the guard tests need no live PG).
        handle, set_transformer = _single_transformer_handle(model)
        backend = "gloo" if self.context.device.type == "cpu" else "nccl"
        init_training_process_group(self.context, backend=backend)
        device_ids = [self.context.local_rank] if self.context.device.type == "cuda" else None
        wrapped = DistributedDataParallel(
            handle,
            device_ids=device_ids,
            find_unused_parameters=self._find_unused_parameters,
        )
        set_transformer(wrapped)
        return model

    def backward(self, loss: torch.Tensor, *, grad_scaler: Any | None = None) -> None:
        # DDP all-reduces gradients inside the backward hooks (this IS the
        # synchronized step); the seam stays identical to single-process/FSDP.
        if grad_scaler is not None:
            grad_scaler.scale(loss).backward()
        else:
            loss.backward()

    def clip_grad_norm(self, parameters: Iterable[nn.Parameter], max_norm: float) -> float:
        # Grads are already all-reduced (identical on every rank), so a local clip
        # is globally correct.
        return float(nn.utils.clip_grad_norm_(parameters, max_norm))

    def _unwrapped_full_state(self, module: Any) -> tuple[Any, dict[str, Any]]:
        from vrl.trainers.weight_sync import unwrap_compile_and_ddp

        inner = unwrap_compile_and_ddp(module)
        # DDP replicates the full module on every rank (no sharding), so the plain
        # unwrapped state_dict() IS the full policy-facing state — same key space as
        # inner.named_parameters(), which select_trainable_state() checks against.
        #
        # Do NOT route this through the FSDP gather_full_state_dict (DCP
        # get_model_state_dict full_state_dict=True): at world_size>1 its distributed
        # all-gather path drops the PEFT LoRA keys for a *replicated* (non-sharded)
        # module, so select_trainable_state() then reports every lora_A/lora_B param
        # "missing" and the first weight sync raises. The ws=1 CPU test never hit that
        # path (gather is a no-op at ws=1); the real 2x1 NCCL run did. Callers perform
        # their own CPU conversion: checkpoint export uses ``to_cpu`` and rollout
        # export uses the stronger non-aliasing ``to_cpu_snapshot`` contract.
        full = {key: value.detach() for key, value in inner.state_dict().items()}
        return inner, full

    def export_trainable_state(self, bundle: Any) -> dict[str, dict[str, Any]]:
        from vrl.trainers.weight_sync import require_trainable_modules, to_cpu

        modules = require_trainable_modules(bundle)
        return {
            name: to_cpu(self._unwrapped_full_state(module)[1]) for name, module in modules.items()
        }

    def export_rollout_state(self, bundle: Any) -> dict[str, Any]:
        from vrl.trainers.weight_sync import (
            require_trainable_modules,
            select_trainable_state,
            to_cpu_snapshot,
        )

        modules = require_trainable_modules(bundle)
        state: dict[str, Any] = {}
        for module_name, module in modules.items():
            inner, full = self._unwrapped_full_state(module)
            state.update(select_trainable_state(inner, str(module_name), full))
        if not state:
            raise ValueError("trainable module state is empty")
        return to_cpu_snapshot(state)

    def load_trainable_state(self, bundle: Any, state: dict[str, Any]) -> None:
        from vrl.trainers.fsdp import load_full_state_dict
        from vrl.trainers.weight_sync import require_trainable_modules, unwrap_compile_and_ddp

        modules = require_trainable_modules(bundle)
        for name, module in modules.items():
            if name in state:
                load_full_state_dict(unwrap_compile_and_ddp(module), state[name])

    def export_optimizer_state(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
    ) -> dict[str, Any]:
        # DDP replicates params (no sharding), so optimizer state is already
        # full plain tensors on every rank — torch's native format suffices.
        del model
        return optimizer.state_dict()

    def load_optimizer_state(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        state: dict[str, Any],
    ) -> None:
        del model
        optimizer.load_state_dict(state)

    def validate_training_state_parking(self) -> None:
        raise NotImplementedError(
            "shared-GPU on-demand rollout is not supported with DDP: reducer buckets, "
            "optimizer, EMA, GradScaler, and live-gradient parking has not been "
            "implemented collectively. Use disjoint rollout GPUs.",
        )

    def park_training_state(self, state: TrainingMemoryState) -> None:
        self.validate_training_state_parking()

    def restore_training_state(self, state: TrainingMemoryState) -> None:
        self.validate_training_state_parking()

    def barrier(self) -> None:
        import torch.distributed as dist

        if dist.is_initialized():
            dist.barrier()

    def shutdown(self, *, restore_parked: bool = True) -> None:
        del restore_parked
        from vrl.trainers.fsdp import shutdown_training_process_group

        shutdown_training_process_group()


def build_strategy(cfg: Any, context: DistributedTrainingContext) -> Strategy:
    """Construct the training strategy named by the resolved context.

    The single dispatch point from config/context to a concrete strategy.
    ``fsdp`` reads its FSDP2 knobs from ``distributed.training.fsdp`` and runs the
    §10 readiness gates (combinations that need DTensor-aware state handling not
    yet built) before constructing ``FSDPStrategy``.
    """

    if context.strategy == "single_process":
        return SingleProcessStrategy(context)
    if context.strategy == "fsdp":
        _assert_fsdp_config_supported(cfg)
        return FSDPStrategy(
            context,
            mesh_dims=list(
                cfg_path(cfg, "distributed.training.fsdp.mesh", ["dp_shard"]) or ["dp_shard"]
            ),
            precision_policy=str(
                cfg_path(cfg, "distributed.training.fsdp.precision_policy", "actor")
            ),
            reshard_after_forward=bool(
                cfg_path(cfg, "distributed.training.fsdp.reshard_after_forward", True),
            ),
        )
    if context.strategy == "ddp":
        return DDPStrategy(
            context,
            find_unused_parameters=bool(
                cfg_path(cfg, "distributed.training.ddp.find_unused_parameters", False),
            ),
        )
    # resolve_training_context / the schema Literal reject other values upstream;
    # this guards direct callers.
    raise ValueError(
        f"unknown distributed.training.strategy={context.strategy!r}; "
        "expected 'single_process', 'fsdp', or 'ddp'",
    )


def _assert_fsdp_config_supported(cfg: Any) -> None:
    """Fail-fast on fsdp + a feature whose DTensor handling is not implemented yet.

    Two of the original ``SPRINT_multi_gpu_training.md`` §10 gates are now
    lifted: EMA shadows update through DTensor local-shard views and
    gather/re-shard at checkpoint boundaries (EMAModuleWrapper), and optimizer
    resume goes through the strategy's full-state export/load
    (gather_full_optimizer_state_dict / load_full_optimizer_state_dict).
    torch.compile remains gated.
    """

    if bool(cfg_path(cfg, "model.torch_compile.enable", False)):
        raise NotImplementedError(
            "distributed.training.strategy=fsdp with model.torch_compile.enable=true "
            "is not supported: torch.compile (inductor graph capture) is unsound with "
            "FSDP2 fully_shard's reshard-after-forward all-gathers. Set "
            "model.torch_compile.enable=false to run FSDP2.",
        )


def _single_process_context() -> DistributedTrainingContext:
    return DistributedTrainingContext(
        strategy="single_process",
        distributed=False,
        rank=0,
        local_rank=0,
        world_size=1,
        is_primary=True,
        device=torch.device("cpu"),
    )


__all__ = [
    "DDPStrategy",
    "FSDPStrategy",
    "SingleProcessStrategy",
    "Strategy",
    "TrainingMemoryState",
    "build_strategy",
]
