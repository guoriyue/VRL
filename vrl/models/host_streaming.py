"""Train a policy whose frozen blocks do not fit the device: stream them per call.

A bf16 ~20B transformer is ~40 GB; a 32 GB card holds its LoRA leaves, its stem
and head, and one block at a time. ``stream_frozen_blocks`` leaves the frozen
parameters of every repeated block on the host and makes each block's forward
run ``torch.func.functional_call`` over device copies of them. The copies are
forward temporaries: non-reentrant activation checkpointing recreates them in
backward, and the host masters are never swapped (``Parameter.data`` is not
touched after setup), so parameter names, ``state_dict`` and checkpoints are
those of the plain policy.

One update of a 640x640, 5-layer Qwen-Image-Layered scene takes about 12.5 s
on an RTX 5090 this way, against the ~3 s a 48 GB card spends with the blocks
resident. The single-process offline recipes select it through
``actor.stream_frozen_blocks``.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from vrl.nn.optimization.regional_compile import repeated_block_lists


class _HostStreamedBlock(nn.Module):
    """Mixin swapped in front of a block's class: its forward sees device weights."""

    _streaming_device: torch.device
    _streaming_active: bool = False

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        if self._streaming_active:
            return super().forward(*args, **kwargs)  # type: ignore[misc]
        device = self._streaming_device
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


def _streamed_class(block_cls: type[nn.Module]) -> type[nn.Module]:
    if issubclass(block_cls, _HostStreamedBlock):
        return block_cls
    return type(f"HostStreamed{block_cls.__name__}", (_HostStreamedBlock, block_cls), {})


def stream_frozen_blocks(root: nn.Module, *, device: torch.device | str) -> int:
    """Place ``root`` on ``device`` except the frozen weights of its repeated blocks.

    Everything outside the repeated block lists moves whole; inside them only
    the trainable leaves (LoRA) move, the frozen weights stay on the host and
    each block streams them per call. Returns the number of streamed blocks.
    """

    device = torch.device(device)
    lists = repeated_block_lists(root)
    if not lists:
        raise ValueError(
            f"{type(root).__name__} has no repeated block list (an nn.ModuleList of two "
            "or more same-class modules); there is nothing to stream from the host",
        )
    streamed: list[nn.Module] = [block for blocks in lists.values() for block in blocks]
    inside = {id(module) for block in streamed for module in block.modules()}
    with torch.no_grad():
        for module in root.modules():
            if id(module) in inside:
                continue
            for parameter in module.parameters(recurse=False):
                parameter.data = parameter.data.to(device)
            for buffer in module.buffers(recurse=False):
                buffer.data = buffer.data.to(device)
        for block in streamed:
            for parameter in block.parameters():
                if parameter.requires_grad:
                    parameter.data = parameter.data.to(device)
    for block in streamed:
        block.__class__ = _streamed_class(type(block))
        block._streaming_device = device
    return len(streamed)


def is_host_streamed(module: nn.Module) -> bool:
    """Whether any block under ``module`` streams its frozen weights from the host."""

    return any(isinstance(child, _HostStreamedBlock) for child in module.modules())


__all__ = ["is_host_streamed", "stream_frozen_blocks"]
