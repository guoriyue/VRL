"""Model and tensor relocation with one shared restore ledger.

Callers own phase transitions and decide whether a failed restore is retryable
or terminal. CuMem mappings and Accelerate hooks do not use this CPU mover.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import TYPE_CHECKING, Any

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
