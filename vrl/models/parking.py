"""Model, training-state and CuMem parking backends.

Callers own phase transitions and failure recovery. CPU relocation records
original devices; CuMem preserves virtual addresses through allocator mappings.
"""

from __future__ import annotations

import gc
import itertools
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from vrl.utils.cuda_memory import empty_cuda_cache

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
    """Retain original devices before moving storage, including partial moves."""

    def __init__(self) -> None:
        self._modules: list[tuple[Any, Any]] = []
        self._tensors: list[tuple[torch.Tensor, torch.device]] = []
        self._seen_modules: set[int] = set()
        self._seen_tensors: set[int] = set()
        self._module_tensor_devices: dict[int, dict[str, Any]] = {}

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
    """Extend model parking with optimizer, gradient, EMA and scaler storage."""

    # A tmpfs mount (``/tmp`` on many hosts, ``/dev/shm``) keeps mapped files
    # in RAM, which defeats disk parking silently.
    _RAM_BACKED_FILESYSTEMS = frozenset({"tmpfs", "ramfs", "devtmpfs"})
    _MOUNTS = "/proc/mounts"

    def __init__(
        self,
        state: TrainingMemoryState,
        *,
        parking_directory: str | None = None,
    ) -> None:
        super().__init__()
        self.state = state
        self.ema_device = getattr(state.ema, "device", None)
        # distributed.resources.parking_directory: frozen shards go to shared
        # file mappings under it instead of anonymous host RAM.
        self._parking_directory = parking_directory
        self._host_storage_directories = []

    def park_training_state(self) -> None:
        import torch

        state = self.state
        for model in (state.model, state.ref_model):
            if model is None:
                continue
            tensor = next(self.module_tensors(model), None)
            device = state.device if tensor is None else torch.device(tensor.device)
            self.park(model, restore_device=device, preserve_tensor_devices=True)
            self._map_frozen_host_storage(model)
        if self._host_storage_directories:
            # Return freed anonymous CPU copies to the OS after replacing them
            # with reclaimable file mappings. glibc may otherwise retain GiBs.
            import ctypes
            import gc

            gc.collect()
            trim = getattr(ctypes.CDLL(None), "malloc_trim", None)
            if trim is not None:
                trim.argtypes = [ctypes.c_size_t]
                trim.restype = ctypes.c_int
                trim(0)
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

    def _map_frozen_host_storage(self, model: Any) -> None:
        """Disk-backed frozen shards; bytes and parameter aliases stay intact.

        Anonymous CPU copies of every parked role can exceed host RAM. Shared
        file mappings let Linux reclaim inactive frozen shards while generation
        owns the GPUs. This is storage relocation only.
        """
        import tempfile
        from pathlib import Path

        import torch

        root = self._parking_directory
        if root is None:
            return
        self._require_disk_backed_directory()
        directory = tempfile.TemporaryDirectory(prefix="trainer-", dir=root)
        self._host_storage_directories.append(directory)
        with torch.no_grad():
            for index, parameter in enumerate(model.parameters()):
                if parameter.requires_grad:
                    continue
                local = getattr(parameter, "_local_tensor", parameter)
                if local.device.type != "cpu" or not local.is_contiguous():
                    raise RuntimeError("disk parking requires contiguous CPU frozen shards")
                if local.numel() == 0:
                    continue
                mapped = torch.from_file(
                    str(Path(directory.name) / f"{index}.bin"),
                    shared=True,
                    size=local.numel(),
                    dtype=local.dtype,
                ).reshape(local.shape)
                mapped.copy_(local)
                if hasattr(parameter, "_local_tensor"):
                    parameter._local_tensor = mapped
                else:
                    parameter.data = mapped
        # Refresh FSDP's local shard views after replacing only owned storage.
        ModelParking._move_module(model, torch.device("cpu"))

    def _require_disk_backed_directory(self) -> None:
        """Refuse a parking directory that is missing or not a real disk.

        The longest mount point that prefixes the directory decides; hosts
        without ``/proc/mounts`` are not checked.
        """
        import os

        directory = self._parking_directory
        assert directory is not None
        if not os.path.isdir(directory):
            raise ValueError(f"parking directory does not exist: {directory}")
        try:
            with open(self._MOUNTS, encoding="utf-8") as handle:
                entries = [line.split() for line in handle]
        except OSError:
            return
        resolved = os.path.realpath(directory)
        best: tuple[str, str] | None = None
        for entry in entries:
            if len(entry) < 3:
                continue
            mount_point, fstype = entry[1], entry[2]
            covers = resolved == mount_point or resolved.startswith(mount_point.rstrip("/") + "/")
            if covers and (best is None or len(mount_point) > len(best[0])):
                best = (mount_point, fstype)
        if best is not None and best[1] in self._RAM_BACKED_FILESYSTEMS:
            raise ValueError(
                f"parking directory {directory} is on a {best[1]} mount ({best[0]}); "
                "disk parking needs node-local disk such as NVMe, or the files stay in RAM",
            )

    def restore(self) -> None:
        super().restore()
        for directory in self._host_storage_directories:
            directory.cleanup()
        self._host_storage_directories.clear()
        if self.state.ema is not None and hasattr(self.state.ema, "device"):
            self.state.ema.device = self.ema_device
        empty_cuda_cache()


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
    """Tagged handle over the process-wide vLLM CuMemAllocator.

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
