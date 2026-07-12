"""Reward scoring data containers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class RewardRollout:
    """A single generated output paired with reward-scoring metadata."""

    prompt: str
    output: Any  # Final generated media (frames / latents)
    metadata: dict[str, Any] = field(default_factory=dict)


__all__ = ["RewardRollout"]
