"""Base contracts for reusable attention-like NN layers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass(frozen=True, slots=True)
class AttentionCacheView:
    """Backend-neutral view of cache metadata consumed by attention layers."""

    block_table: torch.Tensor | None = None
    slot_mapping: torch.Tensor | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


__all__ = ["AttentionCacheView"]
