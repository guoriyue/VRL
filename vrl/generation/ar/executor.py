"""Shared scaffolding for autoregressive generation executors."""

from __future__ import annotations

from typing import Any

from vrl.generation.ar.layout import ARRequestLayout
from vrl.generation.execution.planner import attach_engine_plan
from vrl.generation.execution.request_batch import RequestBatch
from vrl.generation.protocols import PipelineExecutor
from vrl.generation.types import (
    GenerationOutput,
    GenerationRequest,
    GenerationSampleRow,
)


class ARPipelineExecutorBase(
    PipelineExecutor,
):
    """Base helpers for AR family executors.

    Subclasses still own tokenization details, sampling math, decoding, and
    family-specific output packing.
    """

    family: str
    task: str
    default_image_token_num: int | None = None
    default_image_size: int | None = None
    default_max_text_length: int | None = None

    @property
    def layout(self) -> ARRequestLayout:
        return ARRequestLayout(
            default_image_token_num=self.default_image_token_num,
            default_image_size=self.default_image_size,
            default_max_text_length=self.default_max_text_length,
        )

    def require_native_ar_engine(self, request: GenerationRequest) -> str:
        """Reject unsupported full-engine AR selectors before native parity runs."""

        engine = str(request.sampling.get("ar_engine", "native"))
        if engine == "native":
            return engine
        if engine == "vllm":
            raise ValueError(
                "request.sampling.ar_engine='vllm' is not a supported full-engine "
                "backend. AR paged attention is wired inside family runners, not "
                "through a vLLM LLMEngine adapter.",
            )
        raise ValueError("request.sampling.ar_engine must be 'native' if set")

    def forward_batch_plan(
        self,
        requests: list[GenerationRequest],
        sample_rows_by_request: dict[str, list[GenerationSampleRow]],
        engine_plans_by_request: dict[str, Any],
    ) -> dict[str, GenerationOutput]:
        def forward(
            request: GenerationRequest,
            sample_rows: list[GenerationSampleRow],
        ) -> GenerationOutput:
            plan = engine_plans_by_request.get(request.request_id)
            if plan is None:
                plan_method = getattr(self, "plan", None)
                if not callable(plan_method):
                    raise TypeError(f"{type(self).__name__} must implement plan(...)")
                plan = plan_method(request, sample_rows)
            forward_plan = getattr(self, "forward_plan", None)
            if not callable(forward_plan):
                raise TypeError(f"{type(self).__name__} must implement forward_plan(...)")
            output = forward_plan(request, sample_rows, plan)
            execution_extra = output.extra.setdefault("engine_execution", {})
            if isinstance(execution_extra, dict):
                execution_extra["plan_aware_forward"] = True
                execution_extra["forward_plan_id"] = plan.request_id
            return output

        outputs = RequestBatch(
            requests=requests,
            sample_rows_by_request=sample_rows_by_request,
        ).run(forward)
        return {
            request_id: attach_engine_plan(output, engine_plans_by_request[request_id])
            for request_id, output in outputs.items()
        }


__all__ = [
    "ARPipelineExecutorBase",
]
