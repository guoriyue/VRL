"""Ray-backed physical pipeline runner for generation stages."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from vrl.generation.pipeline import (
    PipelineStagePayload,
    PipelineStageResult,
    PipelineTopology,
)
from vrl.ray.dependencies import require_ray


@dataclass(frozen=True, slots=True)
class RayPipelineStageHandle:
    """Driver-side handle for one physical stage actor."""

    stage: str
    actor: Any


class RayPipelineRunner:
    """Route stage payloads across Ray actors using the pipeline contract."""

    def __init__(
        self,
        topology: PipelineTopology,
        stage_handles: list[RayPipelineStageHandle],
        *,
        max_hops: int = 64,
    ) -> None:
        if max_hops < 1:
            raise ValueError("max_hops must be >= 1")
        self.topology = topology
        self.handles = {handle.stage: handle for handle in stage_handles}
        missing = sorted({stage.name for stage in topology.stages} - set(self.handles))
        if missing:
            raise ValueError(f"missing Ray pipeline stage actor(s): {', '.join(missing)}")
        self.max_hops = int(max_hops)

    async def execute(self, payload: PipelineStagePayload) -> PipelineStageResult:
        """Execute one payload through the declared actor pipeline."""

        current = payload
        stage_metrics: list[dict[str, Any]] = []
        for _ in range(self.max_hops):
            stage = self.topology.stage_for(current.stage)
            handle = self.handles[stage.name]
            result = await _execute_actor_stage(handle.actor, current)
            stage_metrics.append(dict(result.metrics))
            if result.error:
                result.metrics["stages"] = stage_metrics
                return result
            if result.payload is None:
                raise RuntimeError(f"stage {stage.name!r} returned no payload")
            if stage.terminal:
                result.terminal = True
                result.metrics["stages"] = stage_metrics
                return result
            if len(stage.next_stages) != 1:
                raise NotImplementedError(
                    "RayPipelineRunner currently supports one next stage per "
                    f"stage; {stage.name!r} declares {len(stage.next_stages)}",
                )
            if result.payload.stage in stage.next_stages:
                current = result.payload
            else:
                current = result.payload.for_stage(stage.next_stages[0])
        raise RuntimeError(f"pipeline exceeded max_hops={self.max_hops}")


async def _execute_actor_stage(
    actor: Any,
    payload: PipelineStagePayload,
) -> PipelineStageResult:
    execute_stage = actor.execute_stage
    remote = getattr(execute_stage, "remote", None)
    if callable(remote):
        ray = require_ray()
        return await asyncio.to_thread(ray.get, remote(payload))
    return execute_stage(payload)


__all__ = [
    "RayPipelineRunner",
    "RayPipelineStageHandle",
]
