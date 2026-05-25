"""Collector/runtime helpers for online training."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from vrl.utils.cuda_memory import empty_cuda_cache


async def _release_collector_runtime_memory(collector: Any) -> None:
    release = getattr(collector, "release_runtime_memory", None)
    if callable(release):
        await release()
    empty_cuda_cache()


def _collector_runtime_requires_driver_model_offload(collector: Any) -> bool:
    try:
        runtime = collector.runtime
    except AttributeError:
        runtime = None
    return bool(getattr(runtime, "requires_driver_model_offload", False))


def _move_model_to_device(model: nn.Module, device: torch.device | str) -> None:
    model.to(device)
    empty_cuda_cache()


__all__ = [
    "_collector_runtime_requires_driver_model_offload",
    "_move_model_to_device",
    "_release_collector_runtime_memory",
]
