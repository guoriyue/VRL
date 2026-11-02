"""Swapping a model's stock blocks for their fused subclasses, in place.

A family's fused block is a subclass of the stock diffusers block with only
``forward`` (and its attention processor) replaced, so installing it is a
``__class__`` swap that keeps every module, parameter and state-dict name.
:func:`swap_block_class` is that walk, shared by the families' installers.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch.nn as nn


def swap_block_class(
    root: nn.Module,
    stock: type[nn.Module],
    fused: type[nn.Module],
    *,
    prepare: Callable[[Any], None],
) -> int:
    """Swap every ``stock`` block under ``root`` to ``fused``; return how many are fused.

    ``prepare`` runs on each block before its swap (checks that the fused
    forward covers it, processor swaps). Blocks already fused count and stay
    as they are; a block some other wrapper has already subclassed (host
    streaming, compile) is refused, since its swap would drop that wrapper.
    """

    count = 0
    for module in root.modules():
        if not isinstance(module, stock):
            continue
        if type(module) is fused:
            count += 1
            continue
        if type(module) is not stock:
            raise TypeError(
                f"cannot fuse {type(module).__name__}: install fused blocks before "
                "anything else subclasses them (host streaming, compile)"
            )
        prepare(module)
        module.__class__ = fused
        count += 1
    return count


def swap_processor(attention: Any, processor_class: type) -> None:
    """Give ``attention`` a ``processor_class`` on its current backend and parallel config."""

    processor = attention.processor
    replacement = processor_class()
    replacement._attention_backend = processor._attention_backend
    replacement._parallel_config = processor._parallel_config
    attention.set_processor(replacement)


__all__ = ["swap_block_class", "swap_processor"]
