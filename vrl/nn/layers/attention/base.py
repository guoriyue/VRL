"""Base contracts for reusable attention-like NN layers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass(frozen=True, slots=True)
class AttentionCacheView:
    """Backend-neutral view of cache metadata consumed by attention layers."""

    block_table: torch.Tensor | None = None
    slot_mapping: torch.Tensor | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class AttentionLayerBase(ABC):
    """Base interface for attention-like layers used by model modules."""

    @abstractmethod
    def debug_info(self) -> dict[str, Any]:
        """Return backend and cache details without exposing backend objects."""


__all__ = ["AttentionCacheView", "AttentionLayerBase"]

