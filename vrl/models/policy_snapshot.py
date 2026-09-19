"""A frozen copy of the trainable parameters that can stand in for them.

The objectives that need a policy other than the one being optimized — the
previous-step behaviour policy of DiffusionNFT / V-GRPO, the KL reference of
the GRPO family — take it from here. The snapshot copies whatever is trainable,
so a LoRA adapter and a fully fine-tuned transformer go through the same code:
the adapter case copies a few MB, the full case one transformer.

Shadows are non-persistent buffers: they follow the model across devices and
into memory parking, and stay out of every state dict (checkpoints, weight
sync) — a snapshot is rebuilt from the live policy, never restored.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterable, Iterator

import torch
from torch import nn


class PolicySnapshot(nn.Module):
    """Shadow copies of ``parameters`` plus the swap that makes them the live weights."""

    def __init__(self, parameters: Iterable[torch.Tensor]) -> None:
        super().__init__()
        self._live = list(parameters)
        if not self._live:
            raise ValueError("PolicySnapshot needs at least one parameter")
        for index, parameter in enumerate(self._live):
            self.register_buffer(f"shadow_{index}", parameter.detach().clone(), persistent=False)

    @property
    def shadows(self) -> list[torch.Tensor]:
        return [buffer for _, buffer in self.named_buffers(recurse=False)]

    @torch.no_grad()
    def update(self, decay: float = 0.0) -> None:
        """``shadow <- decay * shadow + (1 - decay) * live``; ``decay=0`` is an exact copy."""

        decay = float(decay)
        if not 0.0 <= decay <= 1.0:
            raise ValueError(f"snapshot decay must be in [0, 1], got {decay}")
        live = [parameter.detach() for parameter in self._live]
        if decay == 0.0:
            # Per tensor: DTensor (FSDP2) has no sharding rule for the fused
            # _foreach_copy_, while copy_ and _foreach_lerp_ both have one.
            for shadow, source in zip(self.shadows, live, strict=True):
                shadow.copy_(source)
        else:
            # Each shadow is a clone of its parameter, so the shardings match.
            torch._foreach_lerp_(self.shadows, live, 1.0 - decay)

    @contextlib.contextmanager
    def active(self) -> Iterator[None]:
        """Run with the shadows as the live weights; the swap is in place, one tensor at a time."""

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


__all__ = ["PolicySnapshot"]
