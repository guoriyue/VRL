"""Payload envelopes for physical generation pipeline stages."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class PipelineStagePayload:
    """Data owned by one pipeline stage invocation."""

    request_id: str
    stage: str
    data: Any
    policy_version: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def for_stage(
        self,
        stage: str,
        *,
        data: Any = None,
        metadata: dict[str, Any] | None = None,
    ) -> PipelineStagePayload:
        """Return this payload routed to another stage."""

        next_metadata = dict(self.metadata)
        if metadata:
            next_metadata.update(metadata)
        return PipelineStagePayload(
            request_id=self.request_id,
            stage=stage,
            data=self.data if data is None else data,
            policy_version=self.policy_version,
            metadata=next_metadata,
        )


@dataclass(slots=True)
class PipelineStageResult:
    """Result of one stage actor invocation."""

    payload: PipelineStagePayload | None
    metrics: dict[str, Any] = field(default_factory=dict)
    terminal: bool = False
    error: str | None = None


__all__ = [
    "PipelineStagePayload",
    "PipelineStageResult",
]
