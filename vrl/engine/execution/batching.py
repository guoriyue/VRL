"""Shared helpers for batching multiple generation requests."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Protocol

from vrl.engine.core.types import (
    GenerationRequest,
    GenerationSampleSpec,
    OutputBatch,
)
from vrl.engine.trajectory.ops import slice_trajectory_batch


class _PlanAwareRequestExecutor(Protocol):
    family: str
    task: str

    def forward_plan(
        self,
        request: GenerationRequest,
        sample_specs: list[GenerationSampleSpec],
        plan: Any,
    ) -> OutputBatch: ...


def forward_batch_by_merging_prompts(
    executor: _PlanAwareRequestExecutor,
    requests: list[GenerationRequest],
    sample_specs_by_request: dict[str, list[GenerationSampleSpec]],
    *,
    engine_plans_by_request: dict[str, Any],
    merged_plan: Any | None = None,
) -> dict[str, OutputBatch]:
    """Run same-config requests as one prompt-major executor call.

    This is intentionally stricter than ``WorkloadSignature``. The planner may
    select same-shape requests, but the executor only fuses requests whose full
    sampling config and artifact contract are identical. If seeds or CFG differ,
    the worker should fall back to per-request execution.
    """

    if not requests:
        return {}
    if len(requests) == 1:
        request = requests[0]
        plan = engine_plans_by_request[request.request_id]
        output = _forward_request(
            executor,
            request,
            sample_specs_by_request[request.request_id],
            plan=plan,
        )
        output = _attach_plan(output, plan)
        return {
            request.request_id: output
        }

    first = requests[0]
    _validate_mergeable(requests)

    prompts: list[str] = []
    merged_specs: list[GenerationSampleSpec] = []
    request_sample_counts: dict[str, int] = {}
    for request in requests:
        prompts.extend(request.prompts)
        specs = sample_specs_by_request[request.request_id]
        merged_specs.extend(specs)
        request_sample_counts[request.request_id] = len(specs)

    merged_request = GenerationRequest(
        request_id=f"batch:{first.request_id}",
        family=first.family,
        task=first.task,
        prompts=prompts,
        samples_per_prompt=first.samples_per_prompt,
        sampling=dict(first.sampling),
        return_artifacts=set(first.return_artifacts),
        metadata={"batched_request_ids": [r.request_id for r in requests]},
        priority=min(r.priority for r in requests),
        policy_version=first.policy_version,
    )
    if merged_plan is None:
        merged_plan = _build_plan(executor, merged_request, merged_specs)
    merged_output = _forward_request(
        executor,
        merged_request,
        merged_specs,
        plan=merged_plan,
    )

    outputs: dict[str, OutputBatch] = {}
    offset = 0
    for request in requests:
        count = request_sample_counts[request.request_id]
        output = _slice_output_batch(
            merged_output,
            request=request,
            sample_specs=sample_specs_by_request[request.request_id],
            offset=offset,
            count=count,
            total=len(merged_specs),
        )
        output = _attach_plan(output, engine_plans_by_request[request.request_id])
        outputs[request.request_id] = output
        offset += count
    return outputs


def _forward_request(
    executor: _PlanAwareRequestExecutor,
    request: GenerationRequest,
    sample_specs: list[GenerationSampleSpec],
    *,
    plan: Any,
) -> OutputBatch:
    forward_plan = getattr(executor, "forward_plan", None)
    if not callable(forward_plan):
        raise TypeError(
            f"{type(executor).__name__} must implement forward_plan(...) "
            "for plan-aware local batching",
        )
    output = forward_plan(request, sample_specs, plan)
    execution_extra = output.extra.setdefault("engine_execution", {})
    if isinstance(execution_extra, dict):
        execution_extra["plan_aware_forward"] = True
        execution_extra["forward_plan_id"] = plan.request_id
    return output


def _build_plan(
    executor: _PlanAwareRequestExecutor,
    request: GenerationRequest,
    sample_specs: list[GenerationSampleSpec],
) -> Any:
    plan_method = getattr(executor, "plan", None)
    if callable(plan_method):
        return plan_method(request, sample_specs)
    from vrl.engine.execution.planner import build_engine_plan, resolve_executor_capability

    return build_engine_plan(
        request,
        sample_specs,
        capability=resolve_executor_capability(executor, request),
    )


def _attach_plan(output: OutputBatch, plan: Any) -> OutputBatch:
    from vrl.engine.execution.planner import attach_engine_plan

    return attach_engine_plan(output, plan)


def _validate_mergeable(requests: list[GenerationRequest]) -> None:
    first = requests[0]
    for request in requests[1:]:
        if request.family != first.family or request.task != first.task:
            raise ValueError("Cannot merge requests from different family/task")
        if request.samples_per_prompt != first.samples_per_prompt:
            raise ValueError("Cannot merge requests with different sample counts")
        if request.sampling != first.sampling:
            raise ValueError("Cannot merge requests with different sampling config")
        if request.return_artifacts != first.return_artifacts:
            raise ValueError("Cannot merge requests with different artifact modes")
        if request.policy_version != first.policy_version:
            raise ValueError("Cannot merge requests with different policy versions")


def _slice_output_batch(
    output: OutputBatch,
    *,
    request: GenerationRequest,
    sample_specs: list[GenerationSampleSpec],
    offset: int,
    count: int,
    total: int,
) -> OutputBatch:
    trajectory = _slice_trajectory(
        output.trajectory,
        request=request,
        sample_specs=sample_specs,
        offset=offset,
        count=count,
        total=total,
    )
    extra = _slice_value(output.extra, offset, count, total)
    return OutputBatch(
        request_id=request.request_id,
        family=request.family,
        task=request.task,
        prompts=list(request.prompts),
        sample_specs=sample_specs,
        output=_slice_value(output.output, offset, count, total),
        trajectory=trajectory,
        extra=extra,
        metrics=replace(output.metrics, num_prompts=len(request.prompts), num_samples=count)
        if output.metrics is not None
        else None,
        peak_memory_mb=output.peak_memory_mb,
        error=output.error,
    )


def _slice_trajectory(
    data: Any,
    *,
    request: GenerationRequest,
    sample_specs: list[GenerationSampleSpec],
    offset: int,
    count: int,
    total: int,
) -> Any:
    return slice_trajectory_batch(
        data,
        request=request,
        sample_specs=sample_specs,
        offset=offset,
        count=count,
        total=total,
    )


def _slice_value(value: Any, offset: int, count: int, total: int) -> Any:
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
        return {key: _slice_value(inner, offset, count, total) for key, inner in value.items()}
    return value
