"""Training strategy seam: the boundary between the trainer and how it runs.

The trainer drives the GRPO loop; *how* a step executes on the hardware —
backward, grad clipping, and checkpoint-state export/load — goes through a
``Strategy`` so the trainer never hard-codes single-process vs FSDP2.

``SingleProcessStrategy``, ``DDPStrategy``, and the DTensor-aware
``FSDPStrategy`` share this boundary. Concrete strategies own wrapping,
collectives, state parking, checkpoint/rollout export, and shutdown without
teaching the trainer which distributed mechanism is active.
"""

from __future__ import annotations

import gc
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Any, Protocol

import torch
from torch import nn

from vrl.trainers.distributed import (
    DistributedTrainingContext,
    cpu_coordination_group,
    init_training_process_group,
    shutdown_training_process_group,
)
from vrl.trainers.weight_sync import require_trainable_modules
from vrl.utils.cuda_memory import empty_cuda_cache

if TYPE_CHECKING:
    from vrl.config.schema import RootConfig


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
    """How one training step executes; the structural seam the trainer consumes.

    Concrete strategies intentionally do not inherit this protocol. Their real
    implementation bases and mixins own behavior; keeping this consumer contract
    out of the MRO prevents an unimplemented ``...`` stub from silently
    shadowing that behavior.
    """

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

    def export_checkpoint_state(self, bundle: Any) -> dict[str, dict[str, Any]]:
        """Checkpoint-owned state (nested by module name, detached CPU tensors)."""
        ...

    def export_rollout_state(self, bundle: Any) -> dict[str, Any]:
        """Rollout-facing flat trainable state (unwrapped, policy-facing keys)."""
        ...

    def load_checkpoint_state(
        self,
        bundle: Any,
        state: dict[str, Any],
        *,
        strict: bool = True,
    ) -> None:
        """Load checkpoint-owned state back into the bundle."""
        ...

    def load_full_checkpoint_state(
        self,
        bundle: Any,
        state: dict[str, Any],
        *,
        strict: bool = True,
    ) -> None:
        """Load a schema-v1 full-state root through the strategy boundary."""
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

    def all_ranks_succeeded(self, succeeded: bool) -> bool:
        """Agree whether rank-local work succeeded before a collective stage."""
        ...

    def shutdown(self, *, restore_parked: bool = True) -> None:
        """Release resources, restoring parked GPU state only when ownership is safe."""
        ...


class _TrainingStateParking:
    """Move live trainer state off a shared GPU for a rollout phase, and back.

    Shared by single-process and FSDP2. Once ``_move_tensor_data`` understands
    DTensor the traversal is identical for both, because parking only ever
    touches the shard a rank already owns: it issues no collective, so ranks
    cannot desynchronize by parking. What a distributed strategy adds on top is
    *failure* agreement -- see ``FSDPStrategy.park_training_state``.
    """

    # Class-level default so a strategy that inherits this cannot forget to
    # initialize it; the first park assigns a per-instance value.
    _parked_training_state: _ParkedTrainingState | None = None

    def validate_training_state_parking(self) -> None:
        return None

    def park_training_state(self, state: TrainingMemoryState) -> None:
        """Park model, optimizer, EMA, scaler, and live gradients on CPU.

        Tensor objects are moved in place so optimizer/EMA/scaler aliases remain
        valid.  A single identity set spans every owner, preventing a tensor that
        appears through two live structures from being moved twice.
        """

        self._park_training_state_locally(state)

    def _park_training_state_locally(self, state: TrainingMemoryState) -> None:
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
                # Optimizers normally point at model parameters already moved
                # above. A low-precision policy instead exposes independent FP32
                # master parameters through param_groups; move those (and any
                # live master grads) explicitly before walking moment state.
                for group in state.optimizer.param_groups:
                    for parameter in group.get("params", ()):
                        _move_tensor_tree_in_place(
                            parameter,
                            torch.device("cpu"),
                            seen=seen_tensors,
                            restores=parked.tensors,
                        )
                        _move_tensor_tree_in_place(
                            getattr(parameter, "grad", None),
                            torch.device("cpu"),
                            seen=seen_tensors,
                            restores=parked.tensors,
                        )
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


class _ProcessGroupStrategy:
    """Shared behavior of strategies that own a torch process group.

    barrier / cross-rank success reduction / process-group shutdown were
    character-identical between FSDP and DDP; one home, next to the other
    implementation mixins in this file.
    """

    context: DistributedTrainingContext

    def all_ranks_succeeded(self, succeeded: bool) -> bool:
        """Reduce a per-rank success flag to one answer shared by every rank."""

        return _distributed_all_ranks_succeeded(self.context, succeeded)

    def barrier(self) -> None:
        import torch.distributed as dist

        if dist.is_initialized():
            dist.barrier()

    def shutdown(self, *, restore_parked: bool = True) -> None:
        del restore_parked
        shutdown_training_process_group()


class _UnshardedStateStrategy:
    """Checkpoint and optimizer state for backends where no tensor is sharded.

    The shared precondition is "every rank already holds the full unsharded
    tensor": single process trivially, DDP because it *replicates* the module
    instead of splitting it. Under that precondition a rank's own
    ``state_dict()`` already is the full policy-facing state, so these five
    methods can call the plain checkpoint helpers with no collective at all.

    ``FSDPStrategy`` is the counterexample and overrides all five: its
    parameters, gradients, and optimizer moments live as DTensor shards, so
    every one of these operations becomes an all-gather (or a re-scatter on
    load) through ``vrl/trainers/fsdp.py``.

    Concrete strategies inherit this implementation mixin directly. ``Strategy``
    stays outside their MRO as the structural contract consumed by the trainer.
    """

    def export_checkpoint_state(self, bundle: Any) -> dict[str, dict[str, Any]]:
        from vrl.trainers.checkpointing import export_checkpoint_state

        return export_checkpoint_state(bundle)

    def load_checkpoint_state(
        self,
        bundle: Any,
        state: dict[str, Any],
        *,
        strict: bool = True,
    ) -> None:
        from vrl.trainers.checkpointing import load_checkpoint_state

        load_checkpoint_state(bundle, state, strict=strict)

    def load_full_checkpoint_state(
        self,
        bundle: Any,
        state: dict[str, Any],
        *,
        strict: bool = True,
    ) -> None:
        from vrl.trainers.checkpointing import load_full_checkpoint_state

        load_full_checkpoint_state(bundle, state, strict=strict)

    def export_optimizer_state(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
    ) -> dict[str, Any]:
        # Unsharded parameters mean the moments are already full plain tensors on
        # every rank, so torch's native (positional-id) format suffices.
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


class SingleProcessStrategy(_TrainingStateParking, _UnshardedStateStrategy):
    """The current single-GPU behavior, moved behind the strategy protocol.

    Every method here is the existing trainer / checkpoint / weight-sync logic
    verbatim; this installs the seam without changing what a single-process run
    does. ``context`` defaults to a rank0/world1 identity.
    """

    def __init__(self, context: DistributedTrainingContext | None = None) -> None:
        self.context = context or _single_process_context()

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

    def export_rollout_state(self, bundle: Any) -> dict[str, Any]:
        from vrl.trainers.weight_sync import build_trainable_state_sync_getter, to_cpu_snapshot

        return to_cpu_snapshot(build_trainable_state_sync_getter(bundle)())

    def barrier(self) -> None:
        return None

    def all_ranks_succeeded(self, succeeded: bool) -> bool:
        return succeeded

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
        original_device = _tensor_device(value)
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


def _tensor_device(tensor: torch.Tensor) -> torch.device:
    """Where this tensor's storage actually lives (local shard for a DTensor)."""

    local = getattr(tensor, "_local_tensor", None)
    return torch.device(local.device if local is not None else tensor.device)


def _move_tensor_data(tensor: torch.Tensor, device: torch.device) -> None:
    # A DTensor (FSDP2 parameter, its gradient, or an Adam moment) must move by
    # its LOCAL shard: assigning `.data` re-points the storage and torch rejects
    # a cross-device storage swap ("Attempted to set the storage of a tensor on
    # device cuda:0 to a storage on different device cpu"). Moving the local
    # shard keeps the DTensor wrapper -- device mesh and placements -- intact, so
    # the parameter still participates in later collectives, and it issues no
    # collective itself: every rank touches only the shard it already owns.
    local = getattr(tensor, "_local_tensor", None)
    if local is not None:
        if torch.device(local.device) != device:
            tensor._local_tensor = local.to(device=device)
        return
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


def _trainable_module_handles(model: Any) -> list[tuple[str, Any, Any]]:
    """Return every explicitly named trainable root and its writer.

    Diffusion policies already expose the source-of-truth mapping through
    ``trainable_modules``. The writer is the model's own
    ``set_module_root(name, module)``, which is responsible for updating every
    alias the model or its pipeline holds. This supports ordinary ``transformer``
    policies and Wan's timestep-routed ``transformer`` / ``transformer_2``
    without teaching the strategy about either family.
    """

    trainable = getattr(model, "trainable_modules", None)
    if not isinstance(trainable, Mapping) or not trainable:
        raise NotImplementedError(
            "multi-GPU model wrapping needs a non-empty `trainable_modules` mapping "
            f"and a `set_module_root` writer; {type(model).__name__} exposes no "
            "explicit trainable roots. AR families (janus_pro / nextstep_1) need "
            "explicit trainable roots first (SPRINT_multi_gpu_training.md §5).",
        )
    set_root = getattr(model, "set_module_root", None)
    if not callable(set_root):
        raise NotImplementedError(
            f"multi-GPU model wrapping needs {type(model).__name__} to implement "
            "`set_module_root(name, module)` so the distributed wrapper reaches "
            "every alias the model holds for that root.",
        )

    handles: list[tuple[str, Any, Any]] = []
    for raw_name, handle in trainable.items():
        name = str(raw_name)
        if handle is None:
            raise NotImplementedError(
                f"multi-GPU trainable root {name!r} on {type(model).__name__} "
                "requires a non-null handle",
            )
        # Bind the name so each writer targets its own root; the strategies call
        # these positionally with just the wrapped module.
        handles.append((name, handle, partial(set_root, name)))
    return handles


class FSDPStrategy(_ProcessGroupStrategy, _TrainingStateParking):
    """FSDP2 (``fully_shard`` + DTensor) training behind the same seam.

    The model wraps once in ``prepare_model``; thereafter params/grads/optimizer
    state live as DTensor shards over the mesh. Checkpoint and rollout export both
    gather full trainable tensors in the unwrapped, policy-facing key space on
    every rank; frozen base tensors remain sharded. The trainer and Ray rollout
    workers never see a shard or wrapper key. The collective work lives in
    ``vrl/trainers/fsdp.py``; this class is the trainer-facing adapter.

    This is the strategy *layer* of ``SPRINT_multi_gpu_training.md``. The online
    recipe drives it through the same per-rank-local symmetric-colocated path as
    DDP: every torchrun rank owns a local rollout/training device, while FSDP
    handles DTensor sharding and collectives behind this adapter.
    """

    def __init__(
        self,
        context: DistributedTrainingContext,
        *,
        mesh_dims: list[str],
        precision_policy: str,
        reshard_after_forward: bool,
        cpu_offload: bool,
    ) -> None:
        self.context = context
        self._mesh_dims = list(mesh_dims)
        self._precision_policy = precision_policy
        self._reshard_after_forward = reshard_after_forward
        self._cpu_offload = cpu_offload
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
            mixed_precision_policy,
            normalize_fsdp_parameter_dtype,
        )

        # Validate the trainable handles BEFORE touching the process group so a bad
        # model fails fast (mirrors DDPStrategy; the guard tests need no live PG).
        handles = _trainable_module_handles(model)
        prepared_handles: list[tuple[str, nn.Module, Any, torch.dtype]] = []
        for name, handle, writer in handles:
            parameter_dtype = getattr(handle, "dtype", None)
            if not isinstance(parameter_dtype, torch.dtype):
                try:
                    parameter_dtype = next(handle.parameters()).dtype
                except StopIteration as exc:
                    raise ValueError(f"FSDP trainable handle {name!r} has no parameters") from exc
            normalize_fsdp_parameter_dtype(
                handle,
                parameter_dtype,
                allow_cast=self._precision_policy == "actor",
            )
            prepared_handles.append((name, handle, writer, parameter_dtype))
        # Create the process group + bind this rank's cuda device up front, exactly
        # like DDPStrategy. init_device_mesh would lazily auto-init a default group,
        # but it would NOT bind the process's resolved rank-local CUDA device first,
        # so the NCCL group and per-block fully_shard could bind the wrong card on a
        # single-node multi-GPU box. Doing it here keeps the two strategies symmetric
        # and the device choice explicit. No-op for single_process and when a group
        # already exists (the CPU gloo test fixture pre-inits one).
        backend = "gloo" if self.context.device.type == "cpu" else "nccl"
        init_training_process_group(self.context, backend=backend)
        mesh = self._ensure_mesh()
        for _name, handle, writer, parameter_dtype in prepared_handles:
            wrapped = apply_fsdp(
                handle,
                mesh=mesh,
                mp_policy=mixed_precision_policy(
                    self._precision_policy,
                    parameter_dtype=parameter_dtype,
                ),
                reshard_after_forward=self._reshard_after_forward,
                cpu_offload=self._cpu_offload,
            )
            writer(wrapped)
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
        parameter_list = list(parameters)
        if self.context.device.type == "cuda" and any(
            _tensor_device(parameter.grad).type == "cpu"
            for parameter in parameter_list
            if parameter.grad is not None
        ):
            return self._clip_cpu_offloaded_grad_norm(parameter_list, max_norm)
        return float(nn.utils.clip_grad_norm_(parameter_list, max_norm))

    def _clip_cpu_offloaded_grad_norm(
        self,
        parameters: list[nn.Parameter],
        max_norm: float,
    ) -> float:
        """Clip CPU-offloaded FSDP shards with a CUDA scalar collective.

        NCCL cannot all-reduce the CPU DTensor produced by ``CPUOffloadPolicy``.
        The gradient payload itself stays on host: each rank computes its local
        shard's squared norm, copies one scalar to the mesh device, all-reduces
        that scalar, then applies the global clip coefficient to its CPU shards.
        """

        local_squared = torch.zeros((), dtype=torch.float64)
        local_gradients: list[torch.Tensor] = []
        for parameter in parameters:
            gradient = parameter.grad
            if gradient is None:
                continue
            local = getattr(gradient, "_local_tensor", gradient)
            if local.device.type != "cpu":
                raise RuntimeError(
                    "FSDP CPU-offload gradient clipping requires every active "
                    f"gradient shard on CPU; got {local.device}",
                )
            local_gradients.append(local)
            local_squared.add_(local.detach().double().square().sum())

        reduced = local_squared.to(self.context.device)
        import torch.distributed as dist

        if dist.is_initialized():
            dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
        total_norm = float(reduced.sqrt().item())
        clip_coefficient = min(float(max_norm) / (total_norm + 1e-6), 1.0)
        for gradient in local_gradients:
            gradient.mul_(clip_coefficient)
        return total_norm

    def export_checkpoint_state(self, bundle: Any) -> dict[str, dict[str, Any]]:
        from vrl.models.weight_utils import unwrap_compile_and_ddp
        from vrl.trainers.fsdp import gather_checkpoint_state_dict

        modules = require_trainable_modules(bundle)
        return {
            name: gather_checkpoint_state_dict(unwrap_compile_and_ddp(module))
            for name, module in modules.items()
        }

    def export_rollout_state(self, bundle: Any) -> dict[str, Any]:
        from vrl.models.weight_utils import unwrap_compile_and_ddp
        from vrl.trainers.fsdp import gather_trainable_state_dict

        modules = require_trainable_modules(bundle)
        state: dict[str, Any] = {}
        for module_name, module in modules.items():
            gathered = gather_trainable_state_dict(unwrap_compile_and_ddp(module))
            state.update({f"{module_name}.{name}": value for name, value in gathered.items()})
        if not state:
            raise ValueError("trainable module state is empty")
        return state

    def load_checkpoint_state(
        self,
        bundle: Any,
        state: dict[str, Any],
        *,
        strict: bool = True,
    ) -> None:
        from vrl.trainers.fsdp import load_checkpoint_state_dict

        self._load_module_states(bundle, state, strict=strict, load_one=load_checkpoint_state_dict)

    def load_full_checkpoint_state(
        self,
        bundle: Any,
        state: dict[str, Any],
        *,
        strict: bool = True,
    ) -> None:
        from vrl.trainers.fsdp import load_full_state_dict

        self._load_module_states(bundle, state, strict=strict, load_one=load_full_state_dict)

    def _load_module_states(
        self,
        bundle: Any,
        state: dict[str, Any],
        *,
        strict: bool,
        load_one: Callable[..., None],
    ) -> None:
        """Check the module roots, then scatter each module's state with ``load_one``."""

        from vrl.models.weight_utils import unwrap_compile_and_ddp

        modules = require_trainable_modules(bundle)
        missing = sorted(set(modules) - set(state))
        extra = sorted(set(state) - set(modules))
        if strict and (missing or extra):
            raise ValueError(
                f"checkpoint module roots mismatch: missing={missing}, unexpected={extra}",
            )
        for name, module in modules.items():
            if name in state:
                load_one(unwrap_compile_and_ddp(module), state[name], strict=strict)

    def export_optimizer_state(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
    ) -> dict[str, Any]:
        # COLLECTIVE (all-gathers each Adam moment): run on every rank; FQN-keyed
        # so the checkpoint does not depend on optimizer param ordering.
        from vrl.trainers.fsdp import gather_full_optimizer_state_dict

        return gather_full_optimizer_state_dict(model, optimizer)

    def export_checkpoint_optimizer_state(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
    ) -> dict[str, Any]:
        """Gather resume state without rank-replicated full host snapshots."""

        from vrl.trainers.fsdp import gather_full_optimizer_state_dict

        return gather_full_optimizer_state_dict(
            model,
            optimizer,
            rank0_only=True,
        )

    def load_optimizer_state(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        state: dict[str, Any],
    ) -> None:
        from vrl.trainers.fsdp import load_full_optimizer_state_dict

        load_full_optimizer_state_dict(model, optimizer, state)

    def park_training_state(self, state: TrainingMemoryState) -> None:
        """Park this rank's shards, then agree with every peer before returning.

        The move itself is rank-local: ``_move_tensor_data`` relocates a DTensor's
        local shard and leaves the mesh and placements alone, so no collective is
        issued and ranks cannot drift apart by parking.

        Failure is the part that needs coordination. If one rank raised (CUDA
        OOM while staging to host, a module without ``to()``) and rolled itself
        back while the others stayed parked, the ranks would hold different
        residency and the next all-gather would hang instead of reporting the
        real error. So every rank reports its own outcome, takes the minimum, and
        a single failure rolls everyone back to the same resident state.
        """

        # Quiesce before the first unmap: drain this rank's streams, then align
        # every rank on the CPU coordination group so no peer still has the
        # previous phase's collectives in flight when cuMemUnmap starts. This is
        # vLLM's sleep contract (sleep only from a fully idle engine) applied to
        # the phase lease; see the 2026-08-16 Xid 79 postmortem.
        if self.context.device.type == "cuda":
            torch.cuda.synchronize(self.context.device)
        _cpu_coordination_barrier()

        failure: BaseException | None = None
        try:
            self._park_training_state_locally(state)
        except BaseException as error:
            failure = error
        if not self.all_ranks_succeeded(failure is None):
            if failure is None:
                # A peer failed while this rank parked cleanly. Undo locally so
                # the whole world is resident again, and say why.
                self.restore_training_state(state)
                raise RuntimeError(
                    "training-state parking was rolled back: a peer rank failed to park, "
                    "so every rank restored to keep shard residency identical",
                )
            raise failure
        if failure is not None:
            raise failure


class DDPStrategy(_ProcessGroupStrategy, _UnshardedStateStrategy):
    """DistributedDataParallel training behind the same seam.

    For a model that fits on one card (a 2B diffusion transformer + LoRA does), DDP
    replicates the full module on every rank and all-reduces gradients in the
    backward hooks — simpler and cheaper than FSDP2's shard/all-gather, which only
    earns its keep when the model does NOT fit. Because every rank keeps FULL
    params, checkpoint and optimizer state need no gather at all: they come from
    ``_UnshardedStateStrategy``, shared with single process, NOT from FSDP's
    DTensor path. Rollout export is the one state seam DDP writes itself, because
    it must peel DDP's ``.module`` and must NOT route through the DCP full-state
    gather (see ``_unwrapped_full_state``).

    The online recipe drives this through the symmetric-colocated multi-rank path:
    each torchrun rank owns one local trainer/rollout GPU and draws a disjoint
    prompt slice, while DDP synchronizes gradients across ranks. The strategy's
    wrapping and export seams are also exercised on a single CPU rank in
    ``tests/trainers/test_ddp.py``.
    """

    def __init__(
        self,
        context: DistributedTrainingContext,
        *,
        find_unused_parameters: bool,
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

        # Validate the trainable handles BEFORE touching the process group so a bad
        # model fails fast (and the guard tests need no live PG).
        handles = _trainable_module_handles(model)
        backend = "gloo" if self.context.device.type == "cpu" else "nccl"
        init_training_process_group(self.context, backend=backend)
        device_ids = None
        if self.context.device.type == "cuda":
            if self.context.device.index is None:
                raise ValueError("DDP requires an explicit rank-local CUDA device index")
            device_ids = [self.context.device.index]
        for _name, handle, writer in handles:
            wrapped = DistributedDataParallel(
                handle,
                device_ids=device_ids,
                find_unused_parameters=self._find_unused_parameters,
            )
            writer(wrapped)
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
        from vrl.models.weight_utils import unwrap_compile_and_ddp

        inner = unwrap_compile_and_ddp(module)
        # DDP replicates the full module on every rank (no sharding), so the plain
        # unwrapped state_dict() IS the full policy-facing state — same key space as
        # inner.named_parameters(), which select_trainable_state() checks against.
        #
        # Do NOT route this through the FSDP DCP full-state gather
        # (get_model_state_dict full_state_dict=True): at world_size>1 its distributed
        # all-gather path drops the PEFT LoRA keys for a *replicated* (non-sharded)
        # module, so select_trainable_state() then reports every lora_A/lora_B param
        # "missing" and the first weight sync raises. The ws=1 CPU test never hit that
        # path (gather is a no-op at ws=1); the real 2x1 NCCL run did. Callers perform
        # Rollout export performs its own non-aliasing CPU snapshot after selecting
        # the requires-grad policy keys.
        full = {key: value.detach() for key, value in inner.state_dict().items()}
        return inner, full

    def export_rollout_state(self, bundle: Any) -> dict[str, Any]:
        from vrl.trainers.weight_sync import (
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


def build_strategy(config: RootConfig, context: DistributedTrainingContext) -> Strategy:
    """Construct the training strategy named by the resolved context.

    The single dispatch point from config/context to a concrete strategy.
    ``fsdp`` reads its FSDP2 knobs from ``distributed.training.fsdp`` and runs the
    §10 readiness gates (combinations that need DTensor-aware state handling not
    yet built) before constructing ``FSDPStrategy``.
    """

    from vrl.config.schema import TrainingSection

    training = (
        config.distributed.training
        if config.distributed is not None and config.distributed.training is not None
        else TrainingSection()
    )
    configured_strategy = training.strategy
    if configured_strategy != context.strategy:
        raise ValueError(
            "distributed training strategy mismatch: "
            f"config={configured_strategy!r}, context={context.strategy!r}",
        )

    if configured_strategy == "single_process":
        return SingleProcessStrategy(context)
    if configured_strategy == "fsdp":
        _assert_fsdp_config_supported(config)
        if training.fsdp is None:
            raise AssertionError("typed fsdp config was not resolved")
        fsdp = training.fsdp
        return FSDPStrategy(
            context,
            mesh_dims=fsdp.mesh,
            precision_policy=fsdp.precision_policy,
            reshard_after_forward=fsdp.reshard_after_forward,
            cpu_offload=fsdp.cpu_offload,
        )
    if configured_strategy == "ddp":
        if training.ddp is None:
            raise AssertionError("typed ddp config was not resolved")
        return DDPStrategy(
            context,
            find_unused_parameters=training.ddp.find_unused_parameters,
        )
    raise AssertionError(
        f"typed config admitted unknown training strategy {configured_strategy!r}"
    )


def _assert_fsdp_config_supported(config: RootConfig) -> None:
    """Fail-fast on fsdp + a feature whose DTensor handling is not implemented yet.

    Two of the original ``SPRINT_multi_gpu_training.md`` §10 gates are now
    lifted: EMA shadows update through DTensor local-shard views and
    gather/re-shard at checkpoint boundaries (EMAModuleWrapper), and optimizer
    resume goes through the strategy's full-state export/load
    (gather_full_optimizer_state_dict / load_full_optimizer_state_dict).
    torch.compile remains gated.
    """

    from vrl.models.interfaces.runtime import torch_compile_for_role

    compile_block = config.model.torch_compile if config.model is not None else None
    if torch_compile_for_role(compile_block, "replay"):
        raise NotImplementedError(
            "distributed.training.strategy=fsdp cannot compile the replay policy: "
            "torch.compile (inductor graph capture) is unsound with FSDP2 "
            "fully_shard's reshard-after-forward all-gathers. Set "
            "model.torch_compile.enable=false, or model.torch_compile.scope=rollout "
            "to keep the FSDP2 replay policy eager while the rollout policy compiles.",
        )


def _single_process_context() -> DistributedTrainingContext:
    return DistributedTrainingContext(
        strategy="single_process",
        rank=0,
        world_size=1,
        device=torch.device("cpu"),
    )


def _cpu_coordination_barrier() -> None:
    """Barrier on the CPU coordination group; no-op without one."""

    import torch.distributed as dist

    group = cpu_coordination_group()
    if group is not None:
        dist.barrier(group=group)


def _distributed_all_ranks_succeeded(
    context: DistributedTrainingContext,
    succeeded: bool,
) -> bool:
    """Reduce one rank-local outcome through the strategy process group.

    Prefers the CPU coordination group: this flag is exchanged inside park/wake
    windows, where a NCCL all-reduce would run a GPU kernel on a card whose
    cumem pools were just unmapped while slower peers are still unmapping.
    """

    import torch.distributed as dist

    if not dist.is_initialized():
        return succeeded
    group = cpu_coordination_group()
    if group is not None:
        flag = torch.tensor([1 if succeeded else 0])
        dist.all_reduce(flag, op=dist.ReduceOp.MIN, group=group)
        return bool(flag.item())
    flag = torch.tensor(
        [1 if succeeded else 0],
        device=context.device if context.device.type == "cuda" else None,
    )
    dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    return bool(flag.item())


__all__ = [
    "DDPStrategy",
    "FSDPStrategy",
    "SingleProcessStrategy",
    "Strategy",
    "TrainingMemoryState",
    "build_strategy",
]
