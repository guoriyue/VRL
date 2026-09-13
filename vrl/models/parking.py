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


class ModelParking:
    """Retain original devices before moving storage, including partial moves."""

    def __init__(self) -> None:
        self._modules: list[tuple[Any, Any]] = []
        self._tensors: list[tuple[torch.Tensor, torch.device]] = []
        self._seen_modules: set[int] = set()
        self._seen_tensors: set[int] = set()

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

    def park(self, model: Any, *, restore_device: Any) -> None:
        if id(model) in self._seen_modules:
            return
        self._seen_modules.add(id(model))
        self._modules.append((model, restore_device))
        self._seen_tensors.update(id(tensor) for tensor in self.module_tensors(model))
        model.to("cpu")
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
                model.to(device)
            except BaseException as error:
                failures.append(error)
            move_frozen = getattr(model, "move_frozen_components", None)
            if callable(move_frozen):
                try:
                    move_frozen(device)
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

    def __init__(self, state: TrainingMemoryState) -> None:
        super().__init__()
        self.state = state
        self.ema_device = getattr(state.ema, "device", None)

    def park_training_state(self) -> None:
        import torch

        state = self.state
        for model in (state.model, state.ref_model):
            if model is None:
                continue
            # Whole modules restore to their original device; heterogeneous
            # pipeline offload is owned by generation's hook backend instead.
            tensor = next(self.module_tensors(model), None)
            device = state.device if tensor is None else torch.device(tensor.device)
            self.park(model, restore_device=device)
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


def cumem_allocator() -> Any | None:
    """Return the process-wide vLLM CuMemAllocator, or None when unavailable.

    None on a CPU box or when vLLM is not importable. The allocator is a
    per-process singleton; callers hold a :class:`CumemPool` tagged handle
    rather than the raw allocator. Tags control backup/discard during one
    process-wide sleep; they do not isolate sleep operations. This is the
    public vLLM boundary seam: tests replace this function to hand
    :class:`CumemPool` a fake allocator instead of touching CUDA.
    """

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
