"""Tests for engine capability and planning contracts."""

from __future__ import annotations

from typing import Any

import torch

from vrl.engine import (
    FamilyPipelineRegistry,
    GenerationIdFactory,
    GenerationRequest,
    GenerationWorker,
    OutputBatch,
    WorkloadSignature,
    ar_discrete_family_capability,
    build_engine_plan,
    diffusion_family_capability,
    forward_batch_by_merging_prompts,
)
from vrl.rollouts.family_registry import get_rollout_family_entry


def test_diffusion_engine_plan_exposes_axes_chunks_and_profiler_labels() -> None:
    request = GenerationRequest(
        request_id="sd3",
        family="sd3_5",
        task="t2i",
        prompts=["a", "b"],
        samples_per_prompt=3,
        sampling={
            "num_steps": 4,
            "height": 512,
            "width": 512,
            "guidance_scale": 7.0,
            "sample_batch_size": 2,
        },
    )
    specs = GenerationIdFactory().build_sample_specs(request)

    plan = build_engine_plan(
        request,
        specs,
        capability=diffusion_family_capability("sd3_5", "t2i"),
    )

    assert plan.trajectory_kind == "diffusion"
    assert plan.expected_axes["sample"].length == 6
    assert plan.expected_axes["timestep"].kind == "denoise_step"
    assert [
        (chunk.prompt_index, chunk.sample_start, chunk.sample_count)
        for chunk in plan.micro_batches
    ] == [(0, 0, 2), (0, 2, 1), (1, 0, 2), (1, 2, 1)]
    assert "engine.plan" in plan.profiler_labels
    assert "engine.forward_chunk" in plan.profiler_labels
    assert "engine.denoise_step" in plan.profiler_labels
    assert "engine.vq_decode" in plan.profiler_labels
    assert "engine.cache_read" in plan.profiler_labels
    assert "engine.cache_write" in plan.profiler_labels
    assert plan.capability.execution_units[0].cache_read is True
    assert plan.capability.execution_units[0].cache_write is True
    assert plan.workload.trajectory_kind == "diffusion"
    assert ("timestep", "denoise_step") in plan.workload.axis_kinds


def test_ar_engine_plan_exposes_decode_cache_and_vq_units() -> None:
    request = GenerationRequest(
        request_id="janus",
        family="janus_pro",
        task="ar_t2i",
        prompts=["draw a cube"],
        samples_per_prompt=2,
        sampling={"image_token_num": 8, "sample_batch_size": 2},
    )
    specs = GenerationIdFactory().build_sample_specs(request)

    plan = build_engine_plan(
        request,
        specs,
        capability=ar_discrete_family_capability("janus_pro", "ar_t2i"),
    )

    assert plan.trajectory_kind == "ar_discrete"
    assert plan.expected_axes["token"].length == 8
    assert plan.primary_chunk_unit.name == "forward_chunk"
    assert "engine.forward_chunk" in plan.profiler_labels
    assert "engine.prefill" in plan.profiler_labels
    assert "engine.decode_step" in plan.profiler_labels
    assert "engine.vq_decode" in plan.profiler_labels
    assert "engine.cache_read" in plan.profiler_labels
    assert "engine.cache_write" in plan.profiler_labels


def test_rollout_family_registry_carries_capability_metadata() -> None:
    sd3 = get_rollout_family_entry("sd3_5").capability
    janus_r1 = get_rollout_family_entry("janus_pro_r1").capability

    assert sd3.trajectory_kind == "diffusion"
    assert sd3.trainable_segments == ("denoise",)
    assert sd3.supports_kv_decode is False
    assert janus_r1.trajectory_kind == "multisegment"
    assert janus_r1.trainable_segments == (
        "initial_image",
        "selfcheck_text",
        "final_image",
    )


def test_generation_worker_attaches_engine_plan_to_output() -> None:
    class _Executor:
        family = "janus_pro"
        task = "ar_t2i"
        family_capability = ar_discrete_family_capability("janus_pro", "ar_t2i")

        def workload_signature(self, request: GenerationRequest) -> WorkloadSignature:
            return WorkloadSignature.from_request_and_capability(
                request,
                self.family_capability,
            )

        def forward(self, request: GenerationRequest, sample_specs: list[Any]) -> OutputBatch:
            return OutputBatch(
                request_id=request.request_id,
                family=request.family,
                task=request.task,
                prompts=list(request.prompts),
                sample_specs=sample_specs,
                output=torch.zeros(len(sample_specs), 1),
            )

    registry = FamilyPipelineRegistry()
    registry.register(_Executor())
    request = GenerationRequest(
        request_id="req",
        family="janus_pro",
        task="ar_t2i",
        prompts=["p"],
        samples_per_prompt=2,
        sampling={"image_token_num": 4},
    )

    output = GenerationWorker(registry).execute([request])["req"]

    assert output.engine_plan is not None
    assert output.engine_plan.trajectory_kind == "ar_discrete"
    assert output.metrics is not None
    assert output.metrics.execution_units == output.engine_plan.profiler_labels
    assert output.extra["engine_plan"]["trajectory_kind"] == "ar_discrete"


def test_generation_worker_overwrites_stale_batched_engine_plan_extra() -> None:
    class _Executor:
        family = "janus_pro"
        task = "ar_t2i"
        family_capability = ar_discrete_family_capability("janus_pro", "ar_t2i")

        def workload_signature(self, request: GenerationRequest) -> WorkloadSignature:
            return WorkloadSignature.from_request_and_capability(
                request,
                self.family_capability,
            )

        def forward(self, request: GenerationRequest, sample_specs: list[Any]) -> OutputBatch:
            merged_plan = build_engine_plan(
                request,
                sample_specs,
                capability=self.family_capability,
            )
            return OutputBatch(
                request_id=request.request_id,
                family=request.family,
                task=request.task,
                prompts=list(request.prompts),
                sample_specs=sample_specs,
                output=torch.arange(len(sample_specs)).float().unsqueeze(1),
                extra={"engine_plan": merged_plan.summary()},
            )

        def forward_batch(
            self,
            requests: list[GenerationRequest],
            sample_specs_by_request: dict[str, list[Any]],
        ) -> dict[str, OutputBatch]:
            return forward_batch_by_merging_prompts(
                self,
                requests,
                sample_specs_by_request,
            )

    registry = FamilyPipelineRegistry()
    registry.register(_Executor())
    requests = [
        GenerationRequest(
            request_id="r1",
            family="janus_pro",
            task="ar_t2i",
            prompts=["p1"],
            samples_per_prompt=1,
            sampling={"image_token_num": 4},
        ),
        GenerationRequest(
            request_id="r2",
            family="janus_pro",
            task="ar_t2i",
            prompts=["p2"],
            samples_per_prompt=1,
            sampling={"image_token_num": 4},
        ),
    ]

    outputs = GenerationWorker(registry).execute(requests)

    assert outputs["r1"].engine_plan is not None
    assert outputs["r2"].engine_plan is not None
    assert outputs["r1"].engine_plan.request_id == "r1"
    assert outputs["r2"].engine_plan.request_id == "r2"
    assert outputs["r1"].extra["engine_plan"]["request_id"] == "r1"
    assert outputs["r2"].extra["engine_plan"]["request_id"] == "r2"
