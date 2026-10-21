"""Train a policy whose frozen blocks do not fit the device: stream them per call.

A bf16 ~20B transformer is ~40 GB; a 32 GB card holds its LoRA leaves, its stem
and head, and one block at a time. :class:`HostStreaming` leaves the frozen
weights of every repeated block on the host and makes each block's call run
``torch.func.functional_call`` over device copies of them, fetched one block
ahead by a :class:`~vrl.nn.offload.prefetch.Prefetcher`. The copies are call
temporaries (non-reentrant activation checkpointing recreates them in
backward) and the host masters keep their identity, so parameter names,
``state_dict`` and checkpoints are those of the plain policy.
"""

from __future__ import annotations

from typing import Any, ClassVar

import torch
import torch.nn as nn

from vrl.nn.offload.prefetch import Prefetcher
from vrl.nn.optimization.regional_compile import repeated_block_lists


class HostStreaming:
    """The streamed blocks of one root and the prefetcher that feeds them."""

    def __init__(self, blocks: tuple[nn.Module, ...], prefetcher: Prefetcher) -> None:
        self.blocks = blocks
        self.prefetcher = prefetcher

    @classmethod
    def install(cls, root: nn.Module, *, device: torch.device | str) -> HostStreaming:
        """Place ``root`` on ``device`` except the frozen weights of its repeated blocks.

        Everything outside the repeated block lists moves whole; inside them
        only the trainable leaves move, the frozen weights stay on the host
        and each block streams them per call.
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
        # A block's unit is what still lives on the host: its frozen parameters
        # and buffers. Trainable leaves and anything already on the device stay
        # resident and pass through each call untouched.
        units = []
        for block in blocks:
            state = {name: p for name, p in block.named_parameters() if not p.requires_grad}
            state.update(block.named_buffers())
            units.append({name: t for name, t in state.items() if t.device.type == "cpu"})
        streaming = cls(blocks, Prefetcher(units, device))
        for index, (block, unit) in enumerate(zip(blocks, units, strict=True)):
            resident = {
                name: tensor
                for name, tensor in (*block.named_parameters(), *block.named_buffers())
                if name not in unit
            }
            _StreamedBlock.install(block, streaming, index, resident)
        return streaming

    @classmethod
    def of(cls, module: nn.Module) -> HostStreaming | None:
        """The streaming installed under ``module``, if any block of it streams."""

        for child in module.modules():
            if isinstance(child, _StreamedBlock):
                return child._streaming
        return None

    @property
    def counters(self) -> dict[str, int]:
        """Prefetch hits and misses so far."""

        return {"prefetch_hits": self.prefetcher.hits, "prefetch_misses": self.prefetcher.misses}


class _StreamedBlock(nn.Module):
    """Mixin swapped in front of a block's class: its call runs over device copies.

    The wrapper sits on ``__call__``, not ``forward``, so a block compiled in
    place (regional compile) keeps the streaming outside the compiled frame
    whichever of the two was applied first; swapping the class is what puts a
    method there. One streamed subclass per block class, not per block: dynamo
    guards a compiled frame on ``type(self)``, and distinct classes would cost
    one trace per block instead of the one trace regional compile exists for.
    """

    _streaming: HostStreaming
    _unit: int
    _resident: dict[str, torch.Tensor]
    _streaming_active: bool = False
    _subclasses: ClassVar[dict[type[nn.Module], type[nn.Module]]] = {}

    @classmethod
    def install(
        cls,
        block: nn.Module,
        streaming: HostStreaming,
        unit: int,
        resident: dict[str, torch.Tensor],
    ) -> None:
        block_cls = type(block)
        if not issubclass(block_cls, cls):
            if block_cls not in cls._subclasses:
                name = f"HostStreamed{block_cls.__name__}"
                cls._subclasses[block_cls] = type(name, (cls, block_cls), {})
            block.__class__ = cls._subclasses[block_cls]
        block._streaming = streaming
        block._unit = unit
        block._resident = resident

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if self._streaming_active:
            return super().__call__(*args, **kwargs)
        state = {**self._resident, **self._streaming.prefetcher.acquire(self._unit)}
        self._streaming_active = True
        try:
            return torch.func.functional_call(self, state, args, kwargs)
        finally:
            self._streaming_active = False


__all__ = ["HostStreaming"]
