"""In-process pipeline runner and stage worker core."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from typing import Any

from vrl.generation.pipeline.payload import PipelineStagePayload, PipelineStageResult
from vrl.generation.pipeline.topology import PipelineStage, PipelineTopology

StageHandler = Callable[[PipelineStagePayload], Any]


class PipelineStageWorkerCore:
    """Execute one physical stage handler behind an actor-friendly contract."""

    def __init__(self, stage: PipelineStage, handler: StageHandler) -> None:
        self.stage = stage
        self.handler = handler

    def execute_stage(self, payload: PipelineStagePayload) -> PipelineStageResult:
        """Execute the stage and return an actor-safe result envelope."""

        if payload.stage != self.stage.name:
            raise ValueError(
                f"stage worker {self.stage.name!r} received payload for "
                f"{payload.stage!r}",
            )
        started = time.perf_counter()
        try:
            output = self.handler(payload)
            result = _normalize_stage_result(
                output,
                input_payload=payload,
                terminal=self.stage.terminal,
            )
        except Exception as exc:
            return PipelineStageResult(
                payload=None,
                terminal=self.stage.terminal,
                error=str(exc),
                metrics={
                    "stage": self.stage.name,
                    "wall_time_s": time.perf_counter() - started,
                },
            )
        result.metrics.setdefault("stage", self.stage.name)
        result.metrics.setdefault("wall_time_s", time.perf_counter() - started)
        return result


class SerialPipelineRunner:
    """Run a pipeline graph in-process using the same payload contract as actors."""

    def __init__(
        self,
        topology: PipelineTopology,
        handlers: Mapping[str, StageHandler],
    ) -> None:
        self.topology = topology
        self.handlers = dict(handlers)
        missing = sorted({stage.name for stage in topology.stages} - set(self.handlers))
        if missing:
            raise ValueError(f"missing pipeline stage handler(s): {', '.join(missing)}")

    def run(self, payload: PipelineStagePayload) -> PipelineStageResult:
        """Run from the payload's stage through a terminal stage."""

        current = payload
        stage_metrics: list[dict[str, Any]] = []
        while True:
            stage = self.topology.stage_for(current.stage)
            worker = PipelineStageWorkerCore(stage, self.handlers[stage.name])
            result = worker.execute_stage(current)
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
                    "SerialPipelineRunner currently supports one next stage per "
                    f"stage; {stage.name!r} declares {len(stage.next_stages)}",
                )
            if result.payload.stage in stage.next_stages:
                current = result.payload
            else:
                current = result.payload.for_stage(stage.next_stages[0])


def _normalize_stage_result(
    output: Any,
    *,
    input_payload: PipelineStagePayload,
    terminal: bool,
) -> PipelineStageResult:
    if output is None:
        # A handler that forgot to return would otherwise silently forward
        # the INPUT data downstream (for_stage(data=None) keeps prior data).
        raise ValueError(
            f"stage {input_payload.stage!r} handler returned None; return a "
            "payload, a result, or the stage's output data",
        )
    if isinstance(output, PipelineStageResult):
        output.terminal = output.terminal or terminal
        return output
    if isinstance(output, PipelineStagePayload):
        return PipelineStageResult(payload=output, terminal=terminal)
    return PipelineStageResult(
        payload=input_payload.for_stage(input_payload.stage, data=output),
        terminal=terminal,
    )


__all__ = [
    "PipelineStageWorkerCore",
    "SerialPipelineRunner",
    "StageHandler",
]
