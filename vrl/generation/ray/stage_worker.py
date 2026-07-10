"""Ray actor wrapper for physical pipeline stage workers."""

from __future__ import annotations

import contextlib
from collections.abc import Mapping
from typing import Any

from vrl.generation.pipeline import (
    PipelineStage,
    PipelineStagePayload,
    PipelineStageWorkerCore,
)
from vrl.utils.config import import_from_path


class RayPipelineStageWorker:
    """Thin Ray actor adapter around ``PipelineStageWorkerCore``."""

    def __init__(
        self,
        stage: PipelineStage | Mapping[str, Any],
        handler_factory: str | None = None,
        handler_kwargs: Mapping[str, Any] | None = None,
        *,
        worker_id: str | None = None,
    ) -> None:
        pipeline_stage = PipelineStage.from_value(stage)
        factory_path = handler_factory or pipeline_stage.handler
        if not factory_path:
            raise ValueError(
                f"stage {pipeline_stage.name!r} requires a handler factory path",
            )
        factory = import_from_path(factory_path)
        resolved_handler_kwargs = dict(pipeline_stage.handler_kwargs)
        resolved_handler_kwargs.update(dict(handler_kwargs or {}))
        handler = factory(**resolved_handler_kwargs)
        self.core = PipelineStageWorkerCore(pipeline_stage, handler)
        self.worker_id = worker_id or pipeline_stage.name

    @property
    def stage_name(self) -> str:
        return self.core.stage.name

    def execute_stage(self, payload: PipelineStagePayload) -> Any:
        return self.core.execute_stage(payload)

    def worker_metadata(self) -> dict[str, Any]:
        node_ip = "unknown"
        gpu_ids: list[int] = []
        with contextlib.suppress(Exception):
            from vrl.ray.dependencies import current_gpu_ids, current_node_ip

            node_ip = current_node_ip()
            gpu_ids = current_gpu_ids()
        return {
            "worker_id": self.worker_id,
            "stage": self.stage_name,
            "node_ip": node_ip,
            "gpu_ids": gpu_ids,
        }


__all__ = ["RayPipelineStageWorker"]
