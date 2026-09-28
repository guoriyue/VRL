"""A frozen copy of the trainable parameters that can stand in for them.

A full fine-tune's KL reference (DiffusionNFT, the GRPO family) is taken from
here; with a LoRA adapter the reference is the base model with the adapter
disabled and nothing is copied. The snapshot copies whatever is trainable, so
the full case holds one transformer.

Shadows are non-persistent buffers: they follow the model across devices and
into memory parking, and stay out of every state dict (checkpoints, weight
sync) — a snapshot is rebuilt from the live policy, never restored.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterable, Iterator

import torch
from torch import nn


class TrainableWeightsSnapshot(nn.Module):
    """Shadow copies of ``parameters`` plus the swap that makes them the live weights."""

    def __init__(self, parameters: Iterable[torch.Tensor]) -> None:
        super().__init__()
        self._live = list(parameters)
        if not self._live:
            raise ValueError("TrainableWeightsSnapshot needs at least one parameter")
        for index, parameter in enumerate(self._live):
            self.register_buffer(f"shadow_{index}", parameter.detach().clone(), persistent=False)

    @property
    def shadows(self) -> list[torch.Tensor]:
        return [buffer for _, buffer in self.named_buffers(recurse=False)]

    @contextlib.contextmanager
    def active(self) -> Iterator[None]:
        """Run with the shadows as the live weights.

        The swap is in place, one tensor at a time, and bumps the parameters'
        version counters: enter it only while no autograd graph through the
        live parameters is still waiting for its backward — every frozen-policy
        forward runs before the trainable one.
        """

        self._swap()
        try:
            yield
        finally:
            self._swap()

    @torch.no_grad()
    def _swap(self) -> None:
        for parameter, shadow in zip(self._live, self.shadows, strict=True):
            held = parameter.detach().clone()
            parameter.copy_(shadow)
            shadow.copy_(held)


__all__ = ["TrainableWeightsSnapshot"]
