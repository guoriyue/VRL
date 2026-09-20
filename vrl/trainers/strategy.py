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
from contextlib import ExitStack
from functools import partial
from typing import TYPE_CHECKING, Any, Protocol

import torch
from torch import nn

from vrl.models.parking import (
    ModelParking,
    TrainingMemoryState,
    TrainingStateParking,
    module_on_host,
)
from vrl.trainers.distributed import (
    ContextParallelPeerGroup,
    DistributedTrainingContext,
    TrainingCollectives,
    cpu_coordination_group,
    create_context_parallel_peer_group,
    init_training_process_group,
    shutdown_training_process_group,
)
from vrl.trainers.weight_sync import require_trainable_modules

if TYPE_CHECKING:
    from vrl.config.schema import RootConfig


class Strategy(Protocol):
    """How one training step executes; the structural seam the trainer consumes.

    Concrete strategies intentionally do not inherit this protocol. Their real
    implementation bases and mixins own behavior; keeping this consumer contract
    out of the MRO prevents an unimplemented ``...`` stub from silently
    shadowing that behavior.
    """

    context: DistributedTrainingContext

    @property
    def materialize_weights(self) -> bool:
        """Whether this process must load the replay model with real weights.

        Unsharded backends replicate the policy, so every rank loads it. FSDP
        loads on the primary rank only; ``prepare_model`` fills the other
        ranks' meta skeletons from it, so a checkpoint is read into host memory
        once per node instead of once per rank.
        """
        ...

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

    def export_checkpoint_optimizer_state(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
    ) -> dict[str, Any]:
        """The same state, for the primary-only checkpoint writer.

        Separate from ``export_optimizer_state`` because a sharded backend can
        let every rank join the gather while only the writing rank retains the
        full host tree. Where nothing is sharded there is nothing to save, and
        the two are the same call.
        """
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

    collectives: TrainingCollectives

    def gather_rng_states(self, state: dict[str, Any]) -> list[dict[str, Any]]:
        """Collect checkpoint RNG trees in training-rank order."""
        ...

    def shutdown(self, *, restore_parked: bool = True) -> None:
        """Release resources, restoring parked GPU state only when ownership is safe."""
        ...


class _TrainingParkingStrategy:
    """Move live trainer state off a shared GPU for a rollout phase, and back.

    Shared by single-process and FSDP2. The shared parking ledger handles
    DTensor storage locally, because parking only ever
    touches the shard a rank already owns: it issues no collective, so ranks
    cannot desynchronize by parking. What a distributed strategy adds on top is
    *failure* agreement -- see ``FSDPStrategy.park_training_state``.
    """

    # Class-level default so a strategy that inherits this cannot forget to
    # initialize it; the first park assigns a per-instance value.
    _parked_training_state: TrainingStateParking | None = None
    # distributed.resources.trainer_parking_directory; None parks into host RAM.
    _parking_directory: str | None = None

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
        if self._parked_training_state is not None:
            if self._parked_training_state.state.identity_key == state.identity_key:
                return
            raise RuntimeError(
                "cannot park different training state before restoring the current phase"
            )

        parked = TrainingStateParking(state, parking_directory=self._parking_directory)
        self._parked_training_state = parked
        try:
            parked.park_training_state()
            _release_training_cuda_memory()
        except BaseException:
            try:
                parked.restore()
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
        same_state = parked.state.identity_key == state.identity_key
        parked.restore()
        self._parked_training_state = None
        if not same_state:
            raise RuntimeError("trainer-owned state changed while its GPU memory was parked")

    def shutdown(self, *, restore_parked: bool = True) -> None:
        if self._parked_training_state is not None and restore_parked:
            self.restore_training_state(self._parked_training_state.state)
        elif self._parked_training_state is not None:
            # Terminal role cleanup could not prove the shared GPU was released.
            # Drop only this adapter's restore ticket; the live objects deliberately
            # remain on CPU until process exit instead of racing another GPU owner.
            self._parked_training_state = None


class _ProcessGroupStrategy:
    """Shared behavior of strategies that own a torch process group.

    Keep DDP/FSDP process-group teardown identical. Communication operations
    belong to the shared TrainingCollectives instance.
    """

    context: DistributedTrainingContext
    collectives: TrainingCollectives

    def gather_rng_states(self, state: dict[str, Any]) -> list[dict[str, Any]]:
        import torch.distributed as dist

        states = [None] * self.context.world_size
        group = cpu_coordination_group()
        if group is None:
            raise RuntimeError("checkpoint RNG gather requires the CPU coordination group")
        dist.all_gather_object(states, state, group=group)
        return states

    def shutdown(self, *, restore_parked: bool = True) -> None:
        del restore_parked
        shutdown_training_process_group()


class _UnshardedStateStrategy:
    """Checkpoint, optimizer state and placement for backends that never shard.

    The shared precondition is "every rank already holds the full unsharded
    tensor": single process trivially, DDP because it *replicates* the module
    instead of splitting it. Under that precondition a rank's own
    ``state_dict()`` already is the full policy-facing state, so these seven
    methods can call the plain checkpoint helpers with no collective at all.

    ``FSDPStrategy`` is the counterexample and overrides all seven: its
    parameters, gradients, and optimizer moments live as DTensor shards, so
    every one of these operations becomes an all-gather (or a re-scatter on
    load) through ``vrl/trainers/fsdp.py``.

    Concrete strategies inherit this implementation mixin directly. ``Strategy``
    stays outside their MRO as the structural contract consumed by the trainer.
    """

    @property
    def materialize_weights(self) -> bool:
        # Every rank holds the full unsharded tensors, so every rank loads them.
        return True

    def place_trainable_roots(self, model: Any) -> None:
        """Move the policy's trainable roots onto this process's device.

        Replay builds leave the trainable roots where the loader put them (CPU);
        the strategy owns placement. FSDP shards them onto the mesh block by
        block; the unsharded backends move each root whole. Plain ``nn.Module``
        policies (tests, non-diffusion trainers) expose no root mapping and are
        trained where they are; a root a loader already dispatched across
        devices (block-partitioned H3) is not on the host and is left alone.
        """

        trainable = getattr(model, "trainable_modules", None)
        if not isinstance(trainable, Mapping):
            return
        for handle in trainable.values():
            if module_on_host(handle):
                handle.to(self.context.device)

    def export_rollout_state(self, bundle: Any) -> dict[str, Any]:
        """Flat trainable state for the rollout policy, with no collective.

        Do NOT route this through the FSDP DCP full-state gather
        (``get_model_state_dict(full_state_dict=True)``). At world_size>1 its
        distributed all-gather path drops the PEFT LoRA keys for a *replicated*
        (non-sharded) module, so ``select_trainable_state`` then reports every
        lora_A/lora_B parameter "missing" and the first weight sync raises. The
        ws=1 CPU test never hit that path, because the gather is a no-op at
        ws=1; the real 2x1 NCCL run did. ``to_cpu_snapshot`` takes the
        non-aliasing copy after the requires-grad keys are selected.
        """

        from vrl.trainers.weight_sync import flatten_trainable_module_state, to_cpu_snapshot

        return to_cpu_snapshot(flatten_trainable_module_state(require_trainable_modules(bundle)))

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

    def export_checkpoint_optimizer_state(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
    ) -> dict[str, Any]:
        # Nothing is sharded, so there is no gather to join and nothing for a
        # non-writing rank to skip retaining.
        return self.export_optimizer_state(model, optimizer)

    def load_optimizer_state(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        state: dict[str, Any],
    ) -> None:
        del model
        optimizer.load_state_dict(state)


class SingleProcessStrategy(_TrainingParkingStrategy, _UnshardedStateStrategy):
    """The current single-GPU behavior, moved behind the strategy protocol.

    Every method here is the existing trainer / checkpoint / weight-sync logic
    verbatim; this installs the seam without changing what a single-process run
    does. ``context`` defaults to a rank0/world1 identity.
    """

    def __init__(
        self,
        context: DistributedTrainingContext | None = None,
        *,
        collectives: TrainingCollectives | None = None,
    ) -> None:
        self.context = context or DistributedTrainingContext(
            strategy="single_process",
            rank=0,
            world_size=1,
            device=torch.device("cpu"),
        )

        self.collectives = (
            collectives if collectives is not None else TrainingCollectives(self.context)
        )

    def prepare_model(self, model: Any) -> Any:
        # Single process trains the model as-is once placed; the seam exists so
        # FSDP2 can wrap without the trainer changing.
        self.place_trainable_roots(model)
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

    def gather_rng_states(self, state: dict[str, Any]) -> list[dict[str, Any]]:
        return [state]


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
            "explicit trainable roots. Define the model's trainable_modules mapping "
            "and set_module_root(name, module) before distributed wrapping.",
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


class FSDPStrategy(_ProcessGroupStrategy, _TrainingParkingStrategy):
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
        collectives: TrainingCollectives | None = None,
        mesh_dims: list[str],
        precision_policy: str,
        reshard_after_forward: bool,
        cpu_offload: bool,
        shard_trainable_only: bool = False,
        ulysses_degree: int = 1,
        ring_degree: int = 1,
        parking_directory: str | None = None,
    ) -> None:
        self.context = context
        self.collectives = collectives if collectives is not None else TrainingCollectives(context)
        self._mesh_dims = list(mesh_dims)
        self._precision_policy = precision_policy
        self._reshard_after_forward = reshard_after_forward
        self._cpu_offload = cpu_offload
        self._parking_directory = parking_directory
        self._shard_trainable_only = shard_trainable_only
        if shard_trainable_only and precision_policy != "none":
            raise ValueError("shard_trainable_only requires precision_policy='none'")
        self._ulysses_degree = int(ulysses_degree)
        self._ring_degree = int(ring_degree)
        self._mesh: Any | None = None  # built on first prepare_model (needs a live PG)
        self._context_parallel_groups: ContextParallelPeerGroup | None = None

    @property
    def context_parallel(self) -> bool:
        return self._ulysses_degree * self._ring_degree > 1

    @property
    def context_parallel_groups(self) -> ContextParallelPeerGroup | None:
        """This rank's CP groups once ``prepare_model`` has created them."""
        return self._context_parallel_groups

    def _ensure_mesh(self) -> Any:
        if self._mesh is None:
            from vrl.trainers.fsdp import build_fsdp_mesh

            self._mesh = build_fsdp_mesh(self.context, self._mesh_dims)
        return self._mesh

    @property
    def materialize_weights(self) -> bool:
        # The primary rank reads the checkpoint; ``prepare_model`` fills every
        # other rank's meta skeleton from it after sharding.
        return self.context.is_primary

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

        # A rank that built the replay model without weights (``materialize_weights``
        # False) carries meta parameters; the primary rank's tensors are the source
        # for every rank. Decided collectively so the fill is a matched collective.
        skeleton = any(
            p.is_meta for _name, handle, _writer in handles for p in handle.parameters()
        )
        fill_from_primary = bool(self.collectives.max_int(int(skeleton)))
        sources: dict[str, dict[str, Any]] = {}
        if fill_from_primary:
            for name, handle, _writer in handles:
                self._match_primary_dtypes(handle)
                # Plain host tensors on the primary; fully_shard swaps the module's
                # parameters for DTensors but leaves these references intact.
                sources[name] = dict(handle.state_dict()) if self.context.is_primary else {}

        prepared_handles: list[tuple[str, nn.Module, Any, torch.dtype]] = []
        for name, handle, writer in handles:
            parameter_dtype = getattr(handle, "dtype", None)
            if self._shard_trainable_only:
                dtypes = {p.dtype for p in handle.parameters() if p.requires_grad}
                if len(dtypes) != 1:
                    raise ValueError("trainable-only FSDP requires one trainable parameter dtype")
                parameter_dtype = next(iter(dtypes))
            else:
                if self._precision_policy == "none":
                    # A leading frozen FP32 norm does not determine the LoRA
                    # gradient group's dtype; native FSDP preserves both.
                    dtypes = {p.dtype for p in handle.parameters() if p.requires_grad}
                    if len(dtypes) == 1:
                        parameter_dtype = next(iter(dtypes))
                if parameter_dtype is None:
                    parameter_dtypes = {parameter.dtype for parameter in handle.parameters()}
                    if not parameter_dtypes:
                        raise ValueError(f"FSDP trainable handle {name!r} has no parameters")
                    if len(parameter_dtypes) != 1:
                        raise ValueError(
                            f"FSDP trainable handle {name!r} has mixed parameter dtypes; "
                            "declare its target dtype explicitly before preparation",
                        )
                    parameter_dtype = parameter_dtypes.pop()
                elif not isinstance(parameter_dtype, torch.dtype):
                    raise TypeError(f"FSDP trainable handle {name!r} dtype must be a torch.dtype")
                normalize_fsdp_parameter_dtype(
                    handle,
                    parameter_dtype,
                    allow_cast=self._precision_policy == "actor",
                )
            prepared_handles.append((name, handle, writer, parameter_dtype))
        if self.context_parallel:
            from vrl.trainers.context_parallel import enable_context_parallel
            from vrl.trainers.fsdp import build_context_parallel_mesh, unwrap_module

            self._context_parallel_groups = create_context_parallel_peer_group(self.context)
            cp_mesh = build_context_parallel_mesh(
                self.context,
                ulysses_degree=self._ulysses_degree,
                ring_degree=self._ring_degree,
            )
            for _name, handle, _writer, _dtype in prepared_handles:
                enable_context_parallel(
                    unwrap_module(handle),
                    mesh=cp_mesh,
                    ulysses_degree=self._ulysses_degree,
                    ring_degree=self._ring_degree,
                )
        for name, handle, writer, parameter_dtype in prepared_handles:
            if self._shard_trainable_only and not any(p.is_meta for p in handle.parameters()):
                # fully_shard does not place ignored frozen parameters. The
                # replay loader may have staged the entire transformer on CPU.
                # A skeleton has no storage to move; ``_fill_from_primary``
                # allocates it in place after sharding.
                handle.to(device=self.context.device)
            wrapped = apply_fsdp(
                handle,
                mesh=mesh,
                mp_policy=mixed_precision_policy(
                    self._precision_policy,
                    parameter_dtype=parameter_dtype,
                ),
                reshard_after_forward=self._reshard_after_forward,
                cpu_offload=self._cpu_offload,
                ignored_params=(
                    {p for p in handle.parameters() if not p.requires_grad}
                    if self._shard_trainable_only
                    else None
                ),
            )
            if fill_from_primary:
                self._fill_from_primary(wrapped, sources.pop(name))
            writer(wrapped)
        return model

    def _match_primary_dtypes(self, handle: nn.Module) -> None:
        """Give a skeleton the primary rank's per-tensor dtypes before it is filled.

        A skeleton is cast uniformly to the build's parameter dtype; the loader
        that read the checkpoint may have kept exceptions (diffusers' fp32 pins
        under fp16). The fill copies values into the local dtype, so every rank
        must agree on it first.
        """

        import torch.distributed as dist

        names_and_dtypes = [
            (name, tensor.dtype)
            for name, tensor in (*handle.named_parameters(), *handle.named_buffers())
        ]
        payload = [names_and_dtypes if self.context.is_primary else None]
        dist.broadcast_object_list(payload, src=0, group=cpu_coordination_group())
        expected = dict(payload[0])
        local = dict(names_and_dtypes)
        if set(expected) != set(local):
            raise ValueError(
                "replay skeleton does not match the primary rank's model: "
                f"only_on_primary={sorted(set(expected) - set(local))[:5]}, "
                f"only_here={sorted(set(local) - set(expected))[:5]}",
            )
        if not self.context.is_primary:
            for name, tensor in (*handle.named_parameters(), *handle.named_buffers()):
                if tensor.dtype != expected[name]:
                    tensor.data = tensor.data.to(dtype=expected[name])

    def _fill_from_primary(self, wrapped: nn.Module, source: dict[str, Any]) -> None:
        """Materialize a sharded skeleton and load the primary rank's full state.

        Ranks that still hold meta tensors allocate real storage first
        (``to_empty``: uninitialized parameters AND buffers). The parameters
        then arrive through the same rank-0 broadcast checkpoint resume uses;
        buffers are outside any state dict FSDP reshards, so they are broadcast
        one by one over the CPU coordination group, which keeps this free of
        backend and placement assumptions (offloaded shards, resident buffers).
        """

        import torch.distributed as dist

        from vrl.trainers.fsdp import load_full_state_dict

        if any(t.is_meta for t in (*wrapped.parameters(), *wrapped.buffers())):
            wrapped.to_empty(device="cpu" if self._cpu_offload else self.context.device)
            if self._cpu_offload:
                # CPUOffloadPolicy manages parameters only; buffers must live
                # on the compute device for forward.
                for buffer in wrapped.buffers():
                    buffer.data = buffer.data.to(self.context.device)
        load_full_state_dict(wrapped, source, strict=True)
        group = cpu_coordination_group()
        for buffer in wrapped.buffers():
            staged = buffer.detach().to("cpu")
            dist.broadcast(staged, src=0, group=group)
            with torch.no_grad():
                buffer.copy_(staged.to(buffer.device))

    def backward(self, loss: torch.Tensor, *, grad_scaler: Any | None = None) -> None:
        # FSDP2 reduce-scatters gradients inside the backward hooks; the bf16 actor
        # recipe runs without a GradScaler, but keep the seam identical to single
        # process so the trainer loop is backend-agnostic.
        if self.context.cp_size > 1:
            # Every CP peer computes the same full-sequence loss but backpropagates
            # only its own token shard, so the peers' gradients are partial sums
            # of ONE loss. FSDP shards over the whole world and averages over
            # dp*cp ranks; scaling by cp turns that into the (1/dp)-mean of the
            # summed shards, i.e. the gradient a single rank would have computed.
            loss = loss * float(self.context.cp_size)
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
            ModelParking.tensor_device(parameter.grad).type == "cpu"
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

        The move itself is rank-local: the parking ledger relocates a DTensor's
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
        self.collectives.coordination_barrier()

        failure: BaseException | None = None
        # Disk parking first creates a transient anonymous CPU copy. Avoid
        # multiplying that peak by world size while rollout actors are asleep
        # in host RAM. The move is local; all peers still participate in every
        # barrier, including after a local failure.
        serialized = self._parking_directory is not None
        for owner in range(self.context.world_size if serialized else 1):
            if not serialized or self.context.rank == owner:
                try:
                    self._park_training_state_locally(state)
                except BaseException as error:
                    failure = error
            if serialized:
                self.collectives.coordination_barrier()
        if not self.collectives.succeeded(failure is None):
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

    def shutdown(self, *, restore_parked: bool = True) -> None:
        try:
            _TrainingParkingStrategy.shutdown(self, restore_parked=restore_parked)
        finally:
            super().shutdown(restore_parked=restore_parked)


class ContextParallelStrategy(_ProcessGroupStrategy, _UnshardedStateStrategy):
    """Explicit Cosmos CP candidate; deliberately absent from config dispatch.

    Caller owns group-consistent replay inputs and DP-aware sampling. Full
    outputs/losses are replicated inside CP. Parameters are replicated across
    all ranks, so checkpoint state is unsharded. Rollout must independently use
    the same fixed-row/FP32-LoRA compute contract without replay CP hooks.
    """

    def __init__(self, context: DistributedTrainingContext, *, cp_size: int):
        self.context = context
        self.collectives = TrainingCollectives(context)
        self.cp_size = cp_size
        self.groups = None
        self._execution = ExitStack()
        self._parameters: list[nn.Parameter] = []
        self._pending_backward = False

    def prepare_model(self, model: Any) -> Any:
        import os

        import torch.distributed as dist
        from diffusers import CosmosTransformer3DModel

        from vrl.models.families.cosmos.context_parallel import cosmos_context_parallel
        from vrl.models.precision import fixed_row_linear_compute, float32_precision_state
        from vrl.trainers.distributed import create_context_parallel_groups

        if self.groups is not None:
            raise RuntimeError("CP strategy has already prepared a model")
        handles = _trainable_module_handles(model)
        if len(handles) != 1:
            raise ValueError("CP strategy requires one Cosmos transformer")
        self.place_trainable_roots(model)
        handle = handles[0][1]
        base = handle.get_base_model() if hasattr(handle, "get_base_model") else handle
        if not isinstance(base, CosmosTransformer3DModel):
            raise ValueError("CP strategy requires a Cosmos transformer")
        parameters = [p for p in handle.parameters() if p.requires_grad]
        if not parameters or any(p.dtype != torch.float32 for p in parameters):
            raise ValueError("CP strategy requires FP32 trainable parameters")
        if self.context.device.type == "cuda" and (
            not torch.are_deterministic_algorithms_enabled()
            or torch.is_deterministic_algorithms_warn_only_enabled()
            or os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in (":4096:8", ":16:8")
            or float32_precision_state()["matmul"] != "ieee"
        ):
            raise ValueError("CP CUDA strategy requires strict deterministic IEEE compute")
        init_training_process_group(
            self.context, backend="nccl" if self.context.device.type == "cuda" else "gloo"
        )
        if (
            dist.get_rank() != self.context.rank
            or dist.get_world_size() != self.context.world_size
        ):
            raise ValueError("CP process group does not match training context")
        self.groups = create_context_parallel_groups(self.cp_size)
        for tensor in (*handle.parameters(), *handle.buffers()):
            dist.broadcast(tensor.detach(), src=0)
        branches = [
            module
            for name, module in handle.named_modules()
            if isinstance(module, nn.Linear) and (".lora_A." in name or ".lora_B." in name)
        ]
        try:
            if self.context.device.type == "cuda":
                from torch.nn.attention import SDPBackend, sdpa_kernel

                self._execution.enter_context(sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION))
            self._execution.enter_context(fixed_row_linear_compute(handle, fp32_modules=branches))
            self._execution.enter_context(
                cosmos_context_parallel(
                    handle, group=self.groups.cp_group, shard_cross_attention=True
                )
            )
        except BaseException:
            self._execution.close()
            raise
        self._parameters = parameters
        return model

    def backward(self, loss: torch.Tensor, *, grad_scaler: Any | None = None) -> None:
        if self.groups is None:
            raise RuntimeError("CP strategy requires prepare_model before backward")
        if grad_scaler is not None:
            raise NotImplementedError("CP GradScaler synchronization is not implemented")
        (loss / self.groups.cp_size).backward()
        self._pending_backward = True

    def clip_grad_norm(self, parameters: Iterable[nn.Parameter], max_norm: float) -> float:
        from vrl.trainers.distributed import reduce_context_parallel_gradients

        parameters = [p for p in parameters if p.requires_grad]
        if self.groups is None or [id(p) for p in parameters] != [id(p) for p in self._parameters]:
            raise ValueError("CP clipping requires the prepared trainable parameters")
        if self._pending_backward:
            reduce_context_parallel_gradients(parameters, groups=self.groups)
            self._pending_backward = False
        return float(nn.utils.clip_grad_norm_(parameters, max_norm))

    def validate_training_state_parking(self) -> None:
        raise NotImplementedError("CP shared-GPU training-state parking is not implemented")

    def park_training_state(self, state: TrainingMemoryState) -> None:
        self.validate_training_state_parking()

    def restore_training_state(self, state: TrainingMemoryState) -> None:
        self.validate_training_state_parking()

    def shutdown(self, *, restore_parked: bool = True) -> None:
        self._execution.close()
        self.groups = None
        self._parameters = []
        self._pending_backward = False
        super().shutdown(restore_parked=restore_parked)


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
        collectives: TrainingCollectives | None = None,
        find_unused_parameters: bool,
    ) -> None:
        self.context = context
        self.collectives = collectives if collectives is not None else TrainingCollectives(context)
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
        self.place_trainable_roots(model)
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
    replay-compile compatibility check before constructing ``FSDPStrategy``.
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

    collectives = TrainingCollectives(context)
    if configured_strategy == "single_process":
        return SingleProcessStrategy(context, collectives=collectives)
    if configured_strategy == "fsdp":
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
        if training.fsdp is None:
            raise AssertionError("typed fsdp config was not resolved")
        fsdp = training.fsdp
        return FSDPStrategy(
            context,
            collectives=collectives,
            mesh_dims=fsdp.mesh,
            ulysses_degree=fsdp.context_parallel.ulysses_degree,
            ring_degree=fsdp.context_parallel.ring_degree,
            precision_policy=fsdp.precision_policy,
            reshard_after_forward=fsdp.reshard_after_forward,
            cpu_offload=fsdp.cpu_offload,
            shard_trainable_only=fsdp.shard_trainable_only,
            parking_directory=(
                config.distributed.resources.trainer_parking_directory
                if config.distributed is not None and config.distributed.resources is not None
                else None
            ),
        )
    if configured_strategy == "ddp":
        if training.ddp is None:
            raise AssertionError("typed ddp config was not resolved")
        return DDPStrategy(
            context,
            collectives=collectives,
            find_unused_parameters=training.ddp.find_unused_parameters,
        )
    raise AssertionError(
        f"typed config admitted unknown training strategy {configured_strategy!r}"
    )


__all__ = [
    "ContextParallelStrategy",
    "DDPStrategy",
    "FSDPStrategy",
    "SingleProcessStrategy",
    "Strategy",
    "build_strategy",
]
