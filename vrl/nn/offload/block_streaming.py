"""Train a policy whose frozen blocks do not fit the device: stream them per call.

A bf16 ~20B transformer is ~40 GB; a 32 GB card holds its LoRA leaves, its stem
and head, and one block at a time. :class:`HostStreaming` leaves the frozen
parameters of every repeated block on the host and makes each block's forward
run ``torch.func.functional_call`` over device copies of them. The copies are
forward temporaries: non-reentrant activation checkpointing recreates them in
backward, and the host masters are never swapped (``Parameter.data`` is not
touched after setup), so parameter names, ``state_dict`` and checkpoints are
those of the plain policy.

One update of a 640x640, 5-layer Qwen-Image-Layered scene takes about 12.5 s
on an RTX 5090 this way, against the ~3 s a 48 GB card spends with the blocks
resident. Either role selects it through its ``offload`` section.
"""

from __future__ import annotations

from typing import Any, ClassVar

import torch
import torch.nn as nn

from vrl.nn.optimization.regional_compile import repeated_block_lists


class HostStreaming:
    """The streamed blocks of one root."""

    def __init__(self, blocks: tuple[nn.Module, ...]) -> None:
        self.blocks = blocks

    @classmethod
    def install(cls, root: nn.Module, *, device: torch.device | str) -> HostStreaming:
        """Place ``root`` on ``device`` except the frozen weights of its repeated blocks.

        Everything outside the repeated block lists moves whole; inside them
        only the trainable leaves (LoRA) move, the frozen weights stay on the
        host and each block streams them per call.
        """

        device = torch.device(device)
        lists = repeated_block_lists(root)
        if not lists:
            raise ValueError(
                f"{type(root).__name__} has no repeated block list (an nn.ModuleList of two "
                "or more same-class modules); there is nothing to stream from the host",
            )
        blocks = tuple(block for group in lists.values() for block in group)
        block_modules = {id(module) for block in blocks for module in block.modules()}
        with torch.no_grad():
            for module in root.modules():
                if id(module) in block_modules:
                    continue
                for parameter in module.parameters(recurse=False):
                    parameter.data = parameter.data.to(device)
                for buffer in module.buffers(recurse=False):
                    buffer.data = buffer.data.to(device)
            for block in blocks:
                for parameter in block.parameters():
                    if parameter.requires_grad:
                        parameter.data = parameter.data.to(device)
        streaming = cls(blocks)
        for block in blocks:
            _StreamedBlock.install(block, streaming, device)
        return streaming

    @classmethod
    def of(cls, module: nn.Module) -> HostStreaming | None:
        """The streaming installed under ``module``, if any block of it streams."""

        for child in module.modules():
            if isinstance(child, _StreamedBlock):
                return child._streaming
        return None


class _StreamedBlock(nn.Module):
    """Mixin swapped in front of a block's class: its forward sees device weights."""

    _streaming: HostStreaming
    _device: torch.device
    _streaming_active: bool = False
    _subclasses: ClassVar[dict[type[nn.Module], type[nn.Module]]] = {}

    @classmethod
    def install(cls, block: nn.Module, streaming: HostStreaming, device: torch.device) -> None:
        block_cls = type(block)
        if not issubclass(block_cls, cls):
            if block_cls not in cls._subclasses:
                name = f"HostStreamed{block_cls.__name__}"
                cls._subclasses[block_cls] = type(name, (cls, block_cls), {})
            block.__class__ = cls._subclasses[block_cls]
        block._streaming = streaming
        block._device = device

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        if self._streaming_active:
            return super().forward(*args, **kwargs)  # type: ignore[misc]
        device = self._device
        state = {
            name: parameter if parameter.requires_grad else parameter.to(device)
            for name, parameter in self.named_parameters()
        }
        state.update({name: buffer.to(device) for name, buffer in self.named_buffers()})
        self._streaming_active = True
        try:
            return torch.func.functional_call(self, state, args, kwargs)
        finally:
            self._streaming_active = False


__all__ = ["HostStreaming"]
