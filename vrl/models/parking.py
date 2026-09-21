"""Release GPU storage for a phase, then restore it without losing model state.

Every role parks through one :class:`ParkingSession`: build the object (inside
a CuMem pool when the role must prove release), park it, restore it, release
it, and measure the physical footprint in between. Two mechanisms sit under
the session. ``move`` (:class:`ModelParking`) relocates tensors to CPU and
records their original devices; with a ``parking_directory`` its frozen shards
continue to file mappings (:class:`FrozenParameterFileStore`), which owns disk
validation and file lifetime, not device moves. ``cumem`` (:class:`CumemPool`)
requires model allocations to be created inside its pool, which it backs up to
pinned RAM and unmaps while preserving virtual addresses; it has no disk
destination. :class:`TrainingStateParking` extends ``move`` to optimizer,
gradient, EMA and scaler state.

The session raises typed failures and keeps no phase policy. Whether a failure
quarantines the owner, rolls back every rank, or is retried belongs to the
owners (``WorkerMemoryParking``, ``_TrainingParkingStrategy``,
``InProcessRewardScorer``), and whether a role yields its GPU at all is
decided by distributed.resources.offload in vrl/ray/resources.py.
"""

from __future__ import annotations

import gc
import itertools
import traceback
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from vrl.models.frozen_parameter_storage import FrozenParameterFileStore
from vrl.utils.cuda_memory import (
    empty_cuda_cache,
    gpu_process_used_bytes,
    release_cuda_memory,
    release_cuda_memory_for_parking,
)

if TYPE_CHECKING:
    import torch


def module_on_host(module: Any) -> bool:
    """Whether every parameter of ``module`` still lives on the CPU.

    The placement rule shared by the rollout builder and the training
    strategies: a loader that leaves a trainable root on the host expects the
    build path (rollout) or ``prepare_model`` (replay) to place it; a loader
    that already dispatched it -- block-partitioned H3 on several CUDA devices,
    a native package loading straight to its card -- is left alone.
    """

    return all(parameter.device.type == "cpu" for parameter in module.parameters())


class ModelParking:
    """Mechanism ``move``: relocate storage to CPU and remember where it was.

    Retains original devices before moving storage, including partial moves.
    With a ``parking_directory`` the frozen shards of every parked module go on
    to file mappings under it, so Linux can reclaim them while another role
    owns the GPU; trainable storage stays in host RAM.
    """

    def __init__(self, *, parking_directory: str | None = None) -> None:
        self._modules: list[tuple[Any, Any]] = []
        self._tensors: list[tuple[torch.Tensor, torch.device]] = []
        self._seen_modules: set[int] = set()
        self._seen_tensors: set[int] = set()
        self._module_tensor_devices: dict[int, dict[str, Any]] = {}
        self._frozen_file_store = (
            FrozenParameterFileStore(parking_directory) if parking_directory is not None else None
        )

    @property
    def restore_device(self) -> Any | None:
        return self._modules[0][1] if self._modules else None

    @staticmethod
    def module_tensors(module: Any) -> Iterator[torch.Tensor]:
        import torch

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

    def park(
        self, model: Any, *, restore_device: Any, preserve_tensor_devices: bool = False
    ) -> None:
        if id(model) in self._seen_modules:
            return
        self._seen_modules.add(id(model))
        self._modules.append((model, restore_device))
        if preserve_tensor_devices:
            import torch

            if isinstance(model, torch.nn.Module):
                self._module_tensor_devices[id(model)] = {
                    name: self.tensor_device(tensor)
                    for name, tensor in (*model.named_parameters(), *model.named_buffers())
                }
        self._seen_tensors.update(id(tensor) for tensor in self.module_tensors(model))
        self._move_module(model, "cpu")
        move_frozen = getattr(model, "move_frozen_components", None)
        if callable(move_frozen):
            move_frozen("cpu")
        if self._frozen_file_store is not None and callable(getattr(model, "parameters", None)):
            self._frozen_file_store.store(model.parameters())
            # Storage replacement requires FSDP to refresh its local shard views.
            self._move_module(model, "cpu")
            self._frozen_file_store.release_unused_host_memory()

    def park_tensors(self, value: Any) -> None:
        """Move extra state in place, preserving aliases with parked parameters."""
        import torch

        if isinstance(value, torch.Tensor):
            if id(value) in self._seen_tensors:
                return
            self._seen_tensors.add(id(value))
            device = self.tensor_device(value)
            if device.type != "cpu":
                self._tensors.append((value, device))
                self._move_tensor(value, torch.device("cpu"))
        elif isinstance(value, Mapping):
            for child in value.values():
                self.park_tensors(child)
        elif isinstance(value, (list, tuple, set)):
            for child in value:
                self.park_tensors(child)

    @staticmethod
    def _move_module(model: Any, device: Any) -> None:
        import torch
        from torch.distributed.fsdp import FSDPModule

        fsdp_modules = (
            [child for child in model.modules() if isinstance(child, FSDPModule)]
            if isinstance(model, torch.nn.Module)
            else []
        )
        if not fsdp_modules:
            model.to(device)
            return
        # Keep DTensor wrappers and move only owned storage across phase handoffs.
        for child in fsdp_modules:
            child.reshard()
        for tensor in ModelParking.module_tensors(model):
            ModelParking._move_tensor(tensor, torch.device(device))
        for child in fsdp_modules:
            state = child._get_fsdp_state()
            groups = getattr(state, "_fsdp_param_groups", None)
            if groups is None:
                groups = (state._fsdp_param_group,) if state._fsdp_param_group else ()
            for group in groups:
                for parameter in group.fsdp_params:
                    parameter.reset_sharded_param()

    @staticmethod
    def tensor_device(tensor: torch.Tensor) -> torch.device:
        """Return the local storage device, including for an FSDP shard."""
        import torch

        return torch.device(getattr(tensor, "_local_tensor", tensor).device)

    @staticmethod
    def _move_tensor(tensor: torch.Tensor, device: torch.device) -> None:
        # Move only the owned DTensor shard: replacing the wrapper or performing
        # a collective here would break optimizer aliases or rank-local parking.
        local = getattr(tensor, "_local_tensor", None)
        if local is not None:
            if local.device != device:
                tensor._local_tensor = local.to(device=device)
        elif tensor.device != device:
            tensor.data = tensor.data.to(device=device)

    def restore(self) -> None:
        """Attempt every recorded restore; keep the ledger if any move fails."""
        failures: list[BaseException] = []
        for model, device in self._modules:
            try:
                self._move_module(model, device)
            except BaseException as error:
                failures.append(error)
            move_frozen = getattr(model, "move_frozen_components", None)
            if callable(move_frozen):
                try:
                    move_frozen(device)
                except BaseException as error:
                    failures.append(error)
            targets = self._module_tensor_devices.get(id(model), {})
            if targets:
                # Module.to may replace buffers; resolve the current objects by name.
                tensors = dict((*model.named_parameters(), *model.named_buffers()))
                for name, target in targets.items():
                    try:
                        self._move_tensor(tensors[name], target)
                    except BaseException as error:
                        failures.append(error)
        for tensor, device in reversed(self._tensors):
            try:
                self._move_tensor(tensor, device)
            except BaseException as error:
                failures.append(error)
        if failures:
            for error in failures[1:]:
                failures[0].add_note(f"additional parking restore failure: {error!r}")
            raise failures[0]
        self.discard()

    def discard(self) -> None:
        """Drop restore references without bringing storage back onto the GPU."""
        self._modules.clear()
        self._tensors.clear()
        self._seen_modules.clear()
        self._seen_tensors.clear()
        self._module_tensor_devices.clear()
        if self._frozen_file_store is not None:
            self._frozen_file_store.cleanup()


@dataclass(frozen=True, slots=True)
class TrainingMemoryState:
    """Live trainer-owned state that must leave a shared GPU for rollout.

    The trainer builds this value at the start of every rollout phase.  Keeping
    object discovery there is important: optimizer and EMA state are lazy, and
    streaming accumulation can add live gradients between two rollout phases.
    The strategy owns *how* those objects move; rollout orchestration must never
    inspect their internals.
    """

    model: torch.nn.Module
    ref_model: torch.nn.Module | None
    optimizer: torch.optim.Optimizer | None
    ema: Any | None
    grad_scaler: Any | None
    device: torch.device

    @property
    def identity_key(self) -> tuple[int, int, int, int, int, str]:
        """Identify live owners and device without comparing tensor contents."""
        return (
            id(self.model),
            id(self.ref_model),
            id(self.optimizer),
            id(self.ema),
            id(self.grad_scaler),
            str(self.device),
        )


class TrainingStateParking(ModelParking):
    """The trainer's ``move`` ledger: model parking plus optimizer, gradient,
    EMA and scaler storage."""

    def __init__(
        self,
        state: TrainingMemoryState,
        *,
        parking_directory: str | None = None,
    ) -> None:
        super().__init__(parking_directory=parking_directory)
        self.state = state
        self.ema_device = getattr(state.ema, "device", None)

    def park_training_state(self) -> None:
        import torch

        state = self.state
        for model in (state.model, state.ref_model):
            if model is None:
                continue
            tensor = next(self.module_tensors(model), None)
            device = state.device if tensor is None else torch.device(tensor.device)
            self.park(model, restore_device=device, preserve_tensor_devices=True)
        if state.optimizer is not None:
            # Independent FP32 master parameters and live grads may not belong
            # to the model. The shared ledger deduplicates ordinary parameters.
            for group in state.optimizer.param_groups:
                for parameter in group.get("params", ()):
                    self.park_tensors(parameter)
                    self.park_tensors(getattr(parameter, "grad", None))
            self.park_tensors(state.optimizer.state)
        if state.ema is not None:
            self.park_tensors(getattr(state.ema, "ema_parameters", ()))
            self.park_tensors(getattr(state.ema, "temp_stored_parameters", ()))
            if hasattr(state.ema, "device"):
                state.ema.device = torch.device("cpu")
        if state.grad_scaler is not None:
            for attr in ("_scale", "_growth_tracker", "_per_optimizer_states"):
                self.park_tensors(getattr(state.grad_scaler, attr, None))

    def restore(self) -> None:
        super().restore()
        if self.state.ema is not None and hasattr(self.state.ema, "device"):
            self.state.ema.device = self.ema_device
        empty_cuda_cache()


class ParkingBroken(RuntimeError):
    """Parking left the owner's residency unknowable.

    A move failed and its rollback failed too; the owner must not trust the
    state it holds.
    """


class CumemBroken(ParkingBroken):
    """A CuMem allocator operation failed midway.

    vLLM mutates mappings one by one without rollback, so only process
    termination is safe afterwards.
    """


ParkingMechanism = Literal["cumem", "move"]


class ParkingSession:
    """One role's parking of one built object: the steps every role shares.

    ``build`` claims the mechanism (a CuMem pool wrapped around construction,
    or a ``move`` ledger), ``park`` releases the card, ``restore`` brings it
    back and ``release_scope`` tears everything down. ``baseline_gpu_used_bytes``,
    ``release_gpu`` and ``gpu_used_bytes`` supply the physical evidence a
    parking-required role must produce. ``device`` scopes every pool operation
    to a configured CUDA device (a reward pinned to ``cuda:1`` while the driver
    sits on ``cuda:0``); ``None`` uses the current device.

    Failures are typed and carry no policy: :class:`ParkingBroken` when a move
    and its rollback both failed, :class:`CumemBroken` when a pool operation
    failed midway. Any other error propagates unchanged and leaves the session
    retryable (a failed move was rolled back, a failed pool sleep changed no
    state).
    """

    def __init__(
        self,
        owner: str,
        *,
        required: bool,
        device: str | None = None,
        ledger: ModelParking | None = None,
    ) -> None:
        self.owner = owner
        # Whether this owner must prove physical release (baseline captured
        # before build, residual measured after park).
        self.required = required
        self._device = device
        self.backend: ModelParking | CumemPool | None = ledger
        self.baseline_gpu_used_bytes: int | None = None
        self.parked = False

    @property
    def pool(self) -> CumemPool | None:
        return self.backend if isinstance(self.backend, CumemPool) else None

    @property
    def mechanism(self) -> ParkingMechanism | None:
        if self.backend is None:
            return None
        return "cumem" if isinstance(self.backend, CumemPool) else "move"

    def _device_scope(self) -> Any:
        if self._device is None:
            return nullcontext()
        import torch

        if not torch.cuda.is_available():
            return nullcontext()
        target = torch.device(self._device)
        return torch.cuda.device(target) if target.type == "cuda" else nullcontext()

    def gpu_used_bytes(self) -> int:
        """This process's physical footprint on the session's device."""

        return gpu_process_used_bytes(self._device)

    def build(
        self,
        factory: Callable[[], Any],
        *,
        cumem: bool,
        tag: str | None = None,
        resident: Callable[[Any], bool] | None = None,
    ) -> Any:
        """Build once and commit the mechanism.

        ``cumem`` claims the pool before building, so a misconfigured box fails
        in milliseconds rather than after loading GiB of weights it could not
        release; a build failure closes the pool again. ``resident`` is read
        after the build: an object it reports off CUDA allocated nothing in the
        pool, so the pool is closed and the session parks by moving instead.
        """

        if self.backend is not None:
            raise RuntimeError(f"{self.owner} already owns a parking backend")
        if self.required:
            # Compare this process against its own preload physical baseline.
            self.baseline_gpu_used_bytes = self.gpu_used_bytes()
        if not cumem:
            with self._device_scope():
                result = factory()
            self.backend = ModelParking()
            return result
        pool = CumemPool.require(tag=tag)
        try:
            with self._device_scope(), pool.building():
                result = factory()
        except BaseException as build_error:
            # Drop traceback-held builder locals before touching vLLM's retained
            # MemPool registry while preserving the diagnostic stack itself.
            if build_error.__traceback__ is not None:
                traceback.clear_frames(build_error.__traceback__)
            release_cuda_memory(ipc_collect=True)
            self._close_pool(pool, after=f"a failed build ({build_error!r})")
            raise
        if resident is not None and not resident(result):
            self._close_pool(pool, after="a build that left the model off CUDA")
            self.backend = ModelParking()
            return result
        self.backend = pool
        return result

    def park(self, *, move: Callable[[], None] | None = None) -> None:
        """Release the card; idempotent once parked.

        ``cumem`` sleeps the pool. ``move`` runs the owner's move (the ledger
        knows what to relocate) and rolls it back through the ledger if it
        fails, so a retry starts from a resident model.
        """

        backend = self._require_backend("park")
        if self.parked:
            return
        if isinstance(backend, CumemPool):
            try:
                with self._device_scope():
                    backend.sleep()
            except BaseException as error:
                raise CumemBroken(
                    f"{self.owner}: CuMem sleep failed and may have partially unmapped "
                    f"allocations: {error!r}",
                ) from error
            self.parked = True
            return
        if move is None:
            raise TypeError(f"{self.owner}: parking by moving needs the move to perform")
        try:
            move()
        except BaseException as move_error:
            try:
                backend.restore()
            except BaseException as rollback_error:
                raise ParkingBroken(
                    f"{self.owner} parking and rollback both failed: "
                    f"move={move_error!r}; rollback={rollback_error!r}",
                ) from rollback_error
            raise
        self.parked = True

    def restore(self) -> None:
        """Bring the card back; a failed ``move`` restore keeps the ledger for retry."""

        backend = self._require_backend("restore")
        if not self.parked:
            return
        if isinstance(backend, CumemPool):
            try:
                with self._device_scope():
                    backend.wake()
            except BaseException as error:
                raise CumemBroken(
                    f"{self.owner}: CuMem wake failed and may have partially remapped "
                    f"allocations: {error!r}",
                ) from error
        else:
            backend.restore()
        self.parked = False

    def release_gpu(self) -> None:
        """Strict CUDA cleanup after a park, before the residual is measured.

        A ``move`` park invalidates device residency, so idle BLAS workspaces
        that pin allocator segments are cleared too; ``cumem`` keeps its pools.
        """

        release_cuda_memory_for_parking(
            self._device,
            clear_blas_workspaces=isinstance(self.backend, ModelParking),
        )

    @contextmanager
    def release_scope(self) -> Iterator[None]:
        """Wake a slept pool, let the owner drop its references, then free everything.

        A slept pool holds pinned host buffers for its pages; waking before the
        owner drops the object makes freeing the tensors return the pool's
        memory instead of leaking offloaded copies.
        """

        backend = self.backend
        pool = self.pool
        if pool is not None:
            self.restore()
        yield
        if isinstance(backend, ModelParking):
            # The ledger also owns model/tensor references; drop them before
            # collecting allocator pages.
            backend.discard()
        release_cuda_memory(ipc_collect=True)
        if pool is not None:
            self._close_pool(pool, after="release")
            release_cuda_memory(ipc_collect=True)
        self.backend = None
        self.parked = False

    def _require_backend(self, operation: str) -> ModelParking | CumemPool:
        if self.backend is None:
            raise RuntimeError(
                f"{self.owner} cannot {operation} before its build committed a backend"
            )
        return self.backend

    def _close_pool(self, pool: CumemPool, *, after: str) -> None:
        try:
            with self._device_scope():
                pool.close()
        except BaseException as close_error:
            self.backend = pool
            raise CumemBroken(
                f"{self.owner}: CuMem pool close failed after {after}: {close_error!r}",
            ) from close_error


def cumem_allocator() -> Any | None:
    """Return the process-wide vLLM CuMemAllocator, or None when unavailable.

    None on a CPU box or when vLLM is not importable. The allocator is a
    per-process singleton; callers hold a :class:`CumemPool` tagged handle
    rather than the raw allocator. Tags control backup/discard during one
    process-wide sleep; they do not isolate sleep operations. This is the
    public vLLM boundary seam: tests replace this function to hand
    :class:`CumemPool` a fake allocator instead of touching CUDA.
    """

    import torch

    if not torch.cuda.is_available():
        return None
    try:
        from vllm.device_allocator.cumem import CuMemAllocator
    except Exception:
        return None
    try:
        return CuMemAllocator.get_instance()
    except Exception:
        return None


class CumemPool:
    """Mechanism ``cumem``, destination ``ram`` (see the module docstring).

    Tagged handle over the process-wide vLLM CuMemAllocator.

    A tag selects which pages receive a CPU backup; it is not an independently
    sleepable allocator slice. ``CuMemAllocator.sleep`` walks and unmaps every
    registered pointer, discarding pages outside ``offload_tags``. Therefore a
    process may have only one independently parked owner unless a higher-level
    coordinator performs one process-wide sleep with all backup tags together.
    Build the owner's model inside :meth:`building` so its CUDA allocations get
    the backup tag; :meth:`sleep` then copies those pages to pinned host RAM and
    unmaps the process-wide allocator while virtual addresses stay valid.

    Note: pooled physical pages come from CUDA virtual memory, NOT torch's
    caching allocator — a co-resident phase must ``empty_cache`` before this
    pool wakes, or its cached-but-free blocks starve the remap.
    """

    _tags = itertools.count()

    def __init__(self, allocator: Any, tag: str) -> None:
        self._allocator = allocator
        self.tag = tag
        self.asleep = False
        self._building_claimed = False
        self._closed = False

    @classmethod
    def try_create(cls, tag: str | None = None) -> CumemPool | None:
        """Availability probe: pool handle, or None on a CPU box / without vLLM.

        Callers that need a pool must use :meth:`require`. Branching on this
        None to build unpooled is how the deleted CPU-parking fallback came
        back once already: it turns a misconfigured box into a silent 6x
        slowdown that only appears to release GPU memory.
        """

        allocator = cumem_allocator()
        if allocator is None:
            return None
        return cls(allocator, tag if tag else f"cumem-{next(cls._tags)}")

    @classmethod
    def require(cls, tag: str | None = None) -> CumemPool:
        """Pool handle; raise when cumem is unavailable (no silent fallback)."""

        pool = cls.try_create(tag)
        if pool is None:
            raise RuntimeError(
                "vLLM's CuMemAllocator is required here but unavailable — "
                "install vLLM and run on a CUDA device.",
            )
        return pool

    def building(self) -> Any:
        """Return the tag's one-shot model-construction allocation scope.

        vLLM creates a new ``torch.cuda.MemPool`` on every
        ``use_memory_pool`` call. Re-entering the same tag while tensors from its
        first pool are alive can abort the process inside PyTorch rather than
        raise Python. Keep the dependency's supported shape: one scope for model
        construction, then normal execution plus a physical residual check.
        """

        if self._closed:
            raise RuntimeError(f"CuMemPool tag {self.tag!r} is closed")
        if self._building_claimed:
            raise RuntimeError(
                f"CuMemPool tag {self.tag!r} model-building scope is one-shot",
            )
        self._building_claimed = True
        return self._allocator.use_memory_pool(tag=self.tag)

    def sleep(self) -> None:
        """Sleep the process-wide allocator, backing up only this handle's tag."""

        if self._closed:
            raise RuntimeError(f"CuMemPool tag {self.tag!r} is closed")
        self._allocator.sleep(offload_tags=(self.tag,))
        self.asleep = True

    def wake(self) -> None:
        """Remap this handle's backed-up tag (no-op when not asleep)."""

        if self._closed:
            return
        if not self.asleep:
            return
        self._allocator.wake_up(tags=[self.tag])
        self.asleep = False

    def close(self) -> None:
        """Drop vLLM's retained MemPool after all tagged tensors are gone.

        This is a terminal-only adapter for the installed vLLM API. Its public
        ``use_memory_pool`` context retains the created ``torch.cuda.MemPool`` in
        ``allocator_and_pools`` but exposes no close operation; leaving that
        registry entry alive keeps freed model pages mapped indefinitely. Guard
        the one required internal seam explicitly so a vLLM layout change fails
        closed instead of silently leaking GPU ownership.
        """

        if self._closed:
            return
        if self.asleep:
            raise RuntimeError(
                f"CuMemPool tag {self.tag!r} must be awake before terminal close",
            )
        retained_pools = getattr(self._allocator, "allocator_and_pools", None)
        if not isinstance(retained_pools, dict):
            raise RuntimeError(
                "installed vLLM CuMemAllocator does not expose the retained-pool "
                "registry required for terminal release",
            )
        retained = retained_pools.pop(self.tag, None)
        if retained is None:
            raise RuntimeError(
                f"vLLM retained no CuMem pool for terminal tag {self.tag!r}",
            )
        self._closed = True
        del retained
        gc.collect()
