"""Reward scoring data containers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class RewardRollout:
    """One generated sample with its canonical generation lineage."""

    prompt: str
    output: Any  # Final generated media (frames / latents)
    source_request_id: str
    sample_id: str
    group_id: str
    trajectory_id: str
    policy_version: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in (
            "source_request_id",
            "sample_id",
            "group_id",
            "trajectory_id",
        ):
            if not str(getattr(self, name)):
                raise ValueError(f"RewardRollout.{name} must be non-empty")
        if self.policy_version is not None and int(self.policy_version) < 0:
            raise ValueError("RewardRollout.policy_version must be >= 0")


__all__ = ["RewardRollout"]
