"""Request batch for compatible generation requests."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

from vrl.generation.types import (
    GenerationRequest,
    GenerationSampleRow,
    OutputBatch,
)
from vrl.trajectory.ops import slice_trajectory_batch


@dataclass(slots=True)
class RequestBatch:
    """Compatible generation requests grouped for one executor call."""

    requests: list[GenerationRequest]
    sample_rows_by_request: dict[str, list[GenerationSampleRow]]

    def run(
        self,
        forward: Callable[[GenerationRequest, list[GenerationSampleRow]], OutputBatch],
    ) -> dict[str, OutputBatch]:
        """Run the batched forward path and split output back per request."""

        if not self.requests:
            return {}
        if len(self.requests) == 1:
            request = self.requests[0]
            return {
                request.request_id: forward(
                    request,
                    self.sample_rows_by_request[request.request_id],
                )
            }

        self._validate_batchable()
        batched_request, batched_sample_rows, request_sample_counts = self._batch_requests()
        batched_output = forward(batched_request, batched_sample_rows)

        outputs: dict[str, OutputBatch] = {}
        offset = 0
        for request in self.requests:
            count = request_sample_counts[request.request_id]
            output = self._slice_output_batch(
                batched_output,
                request=request,
                sample_rows=self.sample_rows_by_request[request.request_id],
                offset=offset,
                count=count,
                total=len(batched_sample_rows),
            )
            outputs[request.request_id] = output
            offset += count
        return outputs

    def _batch_requests(
        self,
    ) -> tuple[GenerationRequest, list[GenerationSampleRow], dict[str, int]]:
        first = self.requests[0]
        prompts: list[str] = []
        batched_sample_rows: list[GenerationSampleRow] = []
        request_sample_counts: dict[str, int] = {}
        for request in self.requests:
            prompts.extend(request.prompts)
            sample_rows = self.sample_rows_by_request[request.request_id]
            batched_sample_rows.extend(sample_rows)
            request_sample_counts[request.request_id] = len(sample_rows)

        return (
            GenerationRequest(
                request_id=f"batch:{first.request_id}",
                family=first.family,
                task=first.task,
                prompts=prompts,
                samples_per_prompt=first.samples_per_prompt,
                sampling=dict(first.sampling),
                return_artifacts=set(first.return_artifacts),
                metadata={"batched_request_ids": [r.request_id for r in self.requests]},
                priority=min(r.priority for r in self.requests),
                policy_version=first.policy_version,
            ),
            batched_sample_rows,
            request_sample_counts,
        )

    def _validate_batchable(self) -> None:
        first = self.requests[0]
        for request in self.requests[1:]:
            if request.family != first.family or request.task != first.task:
                raise ValueError("Cannot batch requests from different family/task")
            if request.samples_per_prompt != first.samples_per_prompt:
                raise ValueError("Cannot batch requests with different sample counts")
            if request.sampling != first.sampling:
                raise ValueError("Cannot batch requests with different sampling config")
            if request.return_artifacts != first.return_artifacts:
                raise ValueError("Cannot batch requests with different artifact modes")
            if request.policy_version != first.policy_version:
                raise ValueError("Cannot batch requests with different policy versions")

    def _slice_output_batch(
        self,
        output: OutputBatch,
        *,
        request: GenerationRequest,
        sample_rows: list[GenerationSampleRow],
        offset: int,
        count: int,
        total: int,
    ) -> OutputBatch:
        trajectory = self._slice_trajectory(
            output.trajectory,
            request=request,
            sample_rows=sample_rows,
            offset=offset,
            count=count,
            total=total,
        )
        extra = self._slice_value(output.extra, offset, count, total)
        return OutputBatch(
            request_id=request.request_id,
            family=request.family,
            task=request.task,
            prompts=list(request.prompts),
            sample_rows=sample_rows,
            output=self._slice_value(output.output, offset, count, total),
            trajectory=trajectory,
            extra=extra,
            metrics=replace(
                output.metrics,
                num_prompts=len(request.prompts),
                num_samples=count,
            )
            if output.metrics is not None
            else None,
            peak_memory_mb=output.peak_memory_mb,
            error=output.error,
        )

    @staticmethod
    def _slice_trajectory(
        data: Any,
        *,
        request: GenerationRequest,
        sample_rows: list[GenerationSampleRow],
        offset: int,
        count: int,
        total: int,
    ) -> Any:
        return slice_trajectory_batch(
            data,
            request=request,
            sample_rows=sample_rows,
            offset=offset,
            count=count,
            total=total,
        )

    @classmethod
    def _slice_value(cls, value: Any, offset: int, count: int, total: int) -> Any:
        if value is None:
            return None
        shape = getattr(value, "shape", None)
        if shape is not None and len(shape) > 0 and int(shape[0]) == total:
            return value[offset : offset + count]
        if isinstance(value, list) and len(value) == total:
            return value[offset : offset + count]
        if isinstance(value, tuple) and len(value) == total:
            return value[offset : offset + count]
        if isinstance(value, dict):
            return {
                key: cls._slice_value(inner, offset, count, total)
                for key, inner in value.items()
            }
        return value


__all__ = ["RequestBatch"]
