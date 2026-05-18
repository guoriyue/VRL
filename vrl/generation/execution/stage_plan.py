"""Generation stage placement contracts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class StagePlacement:
    """Placement for one generation runtime stage."""

    stage: str
    role: str = "local"
    worker: str | None = None
    device: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.stage:
            raise ValueError("StagePlacement.stage must be non-empty")
        if not self.role:
            raise ValueError("StagePlacement.role must be non-empty")


@dataclass(frozen=True, slots=True)
class GenerationStagePlan:
    """Ordered stage plan for one generation request or chunk."""

    request_id: str
    stages: tuple[str, ...]
    placements: Mapping[str, StagePlacement] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ValueError("GenerationStagePlan.request_id must be non-empty")
        if not self.stages:
            raise ValueError("GenerationStagePlan.stages must be non-empty")
        missing = [stage for stage in self.stages if stage not in self.placements]
        if missing:
            raise ValueError(f"missing stage placements: {missing}")

    @classmethod
    def local(
        cls,
        *,
        request_id: str,
        stages: Sequence[str],
        metadata: Mapping[str, Any] | None = None,
    ) -> GenerationStagePlan:
        ordered = tuple(str(stage) for stage in stages)
        return cls(
            request_id=request_id,
            stages=ordered,
            placements={
                stage: StagePlacement(stage=stage, role="local") for stage in ordered
            },
            metadata=dict(metadata or {}),
        )

    def placement_for(self, stage: str) -> StagePlacement:
        try:
            return self.placements[stage]
        except KeyError as exc:
            raise KeyError(f"unknown generation stage: {stage!r}") from exc


__all__ = ["GenerationStagePlan", "StagePlacement"]
