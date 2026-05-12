"""Generation worker that executes family pipeline executors."""

from __future__ import annotations

import logging
from typing import Any

from vrl.engine.core.capabilities import FamilyCapability, generic_family_capability
from vrl.engine.core.planner import (
    EnginePlan,
    attach_engine_plan,
    build_engine_plan,
    resolve_executor_capability,
)
from vrl.engine.core.registry import FamilyPipelineRegistry
from vrl.engine.core.types import (
    GenerationMetrics,
    GenerationRequest,
    GenerationSampleSpec,
    OutputBatch,
)

logger = logging.getLogger(__name__)


class GenerationIdFactory:
    """Build deterministic sample specs from a generation request."""

    def build_sample_specs(
        self,
        request: GenerationRequest,
    ) -> list[GenerationSampleSpec]:
        base_seed = request.sampling.get("seed")
        seed_int = int(base_seed) if base_seed is not None else None
        specs: list[GenerationSampleSpec] = []
        for prompt_index, prompt in enumerate(request.prompts):
            prompt_id = f"{request.request_id}:prompt:{prompt_index}"
            group_id = prompt_id
            for sample_index in range(request.samples_per_prompt):
                flat_index = len(specs)
                sample_id = f"{prompt_id}:sample:{sample_index}"
                metadata = dict(request.metadata)
                metadata.update(
                    {
                        "request_id": request.request_id,
                        "prompt_index": prompt_index,
                        "sample_index": sample_index,
                        "flat_sample_index": flat_index,
                        "policy_version": request.policy_version,
                    }
                )
                specs.append(
                    GenerationSampleSpec(
                        prompt_index=prompt_index,
                        sample_index=sample_index,
                        prompt=prompt,
                        prompt_id=prompt_id,
                        group_id=group_id,
                        sample_id=sample_id,
                        trajectory_id=sample_id,
                        seed=None if seed_int is None else seed_int + flat_index,
                        metadata=metadata,
                    )
                )
        return specs


class GenerationWorker:
    """Execute generation requests on one local worker."""

    def __init__(
        self,
        registry: FamilyPipelineRegistry,
        *,
        id_factory: GenerationIdFactory | None = None,
        device: Any | None = None,
    ) -> None:
        self.registry = registry
        self.id_factory = id_factory or GenerationIdFactory()
        self.device = device

    def execute(
        self,
        requests: list[GenerationRequest],
    ) -> dict[str, OutputBatch]:
        outputs: dict[str, OutputBatch] = {}
        for grouped_requests in self._group_batchable_requests(requests):
            if len(grouped_requests) == 1:
                request = grouped_requests[0]
                outputs[request.request_id] = self._execute_one(request)
            else:
                outputs.update(self._execute_group(grouped_requests))
        return outputs

    def _execute_one(self, request: GenerationRequest) -> OutputBatch:
        sample_specs = self.id_factory.build_sample_specs(request)
        plan: EnginePlan | None = None
        try:
            executor = self.registry.resolve(request.family, request.task)
            plan = self._plan_request(executor, request, sample_specs)
            output = executor.forward(request, sample_specs)
            if output.request_id != request.request_id:
                raise ValueError(
                    f"Executor returned request_id={output.request_id!r} for "
                    f"request_id={request.request_id!r}"
                )
            return attach_engine_plan(output, plan)
        except Exception as exc:
            logger.exception("Generation request %s failed", request.request_id)
            return OutputBatch(
                request_id=request.request_id,
                family=request.family,
                task=request.task,
                prompts=list(request.prompts),
                sample_specs=sample_specs,
                output=None,
                metrics=GenerationMetrics(
                    num_prompts=len(request.prompts),
                    num_samples=len(sample_specs),
                    trajectory_kind=None if plan is None else plan.trajectory_kind,
                    execution_units=() if plan is None else plan.profiler_labels,
                    engine_plan_id=None if plan is None else plan.request_id,
                ),
                engine_plan=plan,
                error=str(exc),
            )

    def _execute_group(
        self,
        requests: list[GenerationRequest],
    ) -> dict[str, OutputBatch]:
        sample_specs_by_request = {
            request.request_id: self.id_factory.build_sample_specs(request) for request in requests
        }
        plans: dict[str, EnginePlan] = {}
        try:
            executor = self.registry.resolve(requests[0].family, requests[0].task)
            for request in requests:
                plans[request.request_id] = self._plan_request(
                    executor,
                    request,
                    sample_specs_by_request[request.request_id],
                )
            forward_batch = getattr(executor, "forward_batch", None)
            if forward_batch is None:
                return {request.request_id: self._execute_one(request) for request in requests}
            outputs = forward_batch(requests, sample_specs_by_request)
            completed: dict[str, OutputBatch] = {}
            for request in requests:
                output = outputs.get(request.request_id)
                plan = plans[request.request_id]
                if output is None:
                    output = _error_output(
                        request,
                        sample_specs_by_request[request.request_id],
                        "Batched executor did not return an output for this request",
                        engine_plan=plan,
                    )
                completed[request.request_id] = attach_engine_plan(output, plan)
            return completed
        except Exception as exc:
            logger.exception(
                "Generation request batch failed: %s",
                [request.request_id for request in requests],
            )
            return {
                request.request_id: _error_output(
                    request,
                    sample_specs_by_request[request.request_id],
                    str(exc),
                    engine_plan=plans.get(request.request_id),
                )
                for request in requests
            }

    def _group_batchable_requests(
        self,
        requests: list[GenerationRequest],
    ) -> list[list[GenerationRequest]]:
        groups: dict[Any, list[GenerationRequest]] = {}
        ordered_keys: list[Any] = []
        for request in requests:
            key = _strict_batch_key(request, self._capability_for_group_key(request))
            if key not in groups:
                groups[key] = []
                ordered_keys.append(key)
            groups[key].append(request)
        return [groups[key] for key in ordered_keys]

    def _plan_request(
        self,
        executor: Any,
        request: GenerationRequest,
        sample_specs: list[GenerationSampleSpec],
    ) -> EnginePlan:
        from vrl.trainers.profiling import record_function

        with record_function("engine.plan"):
            plan_method = getattr(executor, "plan", None)
            if callable(plan_method):
                return plan_method(request, sample_specs)
            return build_engine_plan(
                request,
                sample_specs,
                capability=resolve_executor_capability(executor, request),
            )

    def _capability_for_group_key(self, request: GenerationRequest) -> FamilyCapability:
        try:
            executor = self.registry.resolve(request.family, request.task)
            return resolve_executor_capability(executor, request)
        except Exception:
            return generic_family_capability(request.family, request.task)


def _strict_batch_key(
    request: GenerationRequest,
    capability: FamilyCapability,
) -> tuple[Any, ...]:
    if not _safe_to_batch(request, capability):
        return (request.request_id,)
    return (
        request.family,
        request.task,
        capability.batch_signature(),
        request.samples_per_prompt,
        request.policy_version,
        tuple(sorted(request.return_artifacts)),
        _freeze(request.sampling),
    )


def _safe_to_batch(request: GenerationRequest, capability: FamilyCapability) -> bool:
    if not capability.supports_batched_requests:
        return False
    sampling = request.sampling
    if "seed" in sampling:
        return False
    return int(sampling.get("sde_window_size", 0)) <= 0


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return tuple((key, _freeze(inner)) for key, inner in sorted(value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(inner) for inner in value)
    if isinstance(value, set):
        return tuple(sorted(_freeze(inner) for inner in value))
    try:
        hash(value)
    except TypeError:
        return repr(value)
    return value


def _error_output(
    request: GenerationRequest,
    sample_specs: list[Any],
    error: str,
    *,
    engine_plan: EnginePlan | None = None,
) -> OutputBatch:
    return OutputBatch(
        request_id=request.request_id,
        family=request.family,
        task=request.task,
        prompts=list(request.prompts),
        sample_specs=sample_specs,
        output=None,
        metrics=GenerationMetrics(
            num_prompts=len(request.prompts),
            num_samples=len(sample_specs),
            trajectory_kind=None if engine_plan is None else engine_plan.trajectory_kind,
            execution_units=() if engine_plan is None else engine_plan.profiler_labels,
            engine_plan_id=None if engine_plan is None else engine_plan.request_id,
        ),
        engine_plan=engine_plan,
        error=error,
    )
