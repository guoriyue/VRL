"""Tests for physical generation pipeline contracts."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from vrl.generation.pipeline import (
    PipelineStage,
    PipelineStagePayload,
    PipelineStageResult,
    PipelineStageWorkerCore,
    PipelineTopology,
    SerialPipelineRunner,
)
from vrl.generation.ray import (
    RayPipelineRunner,
    RayPipelineStageHandle,
    RayPipelineStageWorker,
)


def test_pipeline_topology_normalizes_stage_configs() -> None:
    """Checks stage mappings normalize: scalar next/gpu wrap into tuples."""
    topology = PipelineTopology.from_value(
        {
            "entry_stage": "denoise",
            "stages": [
                {
                    "name": "denoise",
                    "handler": "tests.generation.pipeline.test_pipeline_contracts:make_append_handler",
                    "handler_kwargs": {"suffix": "d"},
                    "next_stages": "decode_latents",
                    "gpu_ids": 0,
                    "runtime": {
                        "batch_size": 16,
                        "max_inflight": 2,
                    },
                    "metadata": {"note": "deliberate annotations live here"},
                },
                {
                    "name": "decode_latents",
                    "terminal": True,
                    "runtime": {"batch_size": 1},
                },
            ],
        }
    )

    denoise = topology.stage_for("denoise")
    decode = topology.stage_for("decode_latents")

    assert topology.resolved_entry_stage == "denoise"
    assert denoise.handler == (
        "tests.generation.pipeline.test_pipeline_contracts:make_append_handler"
    )
    assert denoise.handler_kwargs == {"suffix": "d"}
    assert denoise.next_stages == ("decode_latents",)
    assert denoise.gpu_ids == (0,)
    assert denoise.runtime.batch_size == 16
    assert denoise.runtime.max_inflight == 2
    assert denoise.metadata == {"note": "deliberate annotations live here"}
    assert decode.terminal is True
    assert decode.runtime.batch_size == 1


@pytest.mark.parametrize(
    "stage_value",
    [
        {"name": "denoise", "terminal": True, "route_fn": "pkg:route"},
        {"name": "denoise", "terminal": True, "next": "decode"},
        {"name": "denoise", "terminal": True, "factory": "pkg:make"},
        {"name": "denoise", "terminal": True, "runtime": {"max_seq_len": 4096}},
    ],
)
def test_unknown_stage_keys_fail_construction(stage_value: dict[str, Any]) -> None:
    """Checks keys outside our own field vocabulary fail loudly.

    This covers SGLang-Omni vocabulary too (``next``/``factory``/``route_fn``):
    we learn from that design but do not carry its schema.
    """
    with pytest.raises(TypeError):
        PipelineStage.from_value(stage_value)


def test_pipeline_topology_rejects_cycles() -> None:
    """Checks cyclic stage topologies fail validation instead of looping forever.

    SerialPipelineRunner follows next_stages hops without a hop cap; this
    DAG validation is what bounds it.
    """
    with pytest.raises(ValueError, match="cycle"):
        PipelineTopology.from_value(
            {
                "stages": [
                    {"name": "denoise", "next_stages": "decode_latents"},
                    {"name": "decode_latents", "next_stages": "denoise"},
                    {"name": "reward", "terminal": True},
                ],
            }
        )


def test_stage_handler_returning_none_fails_loudly() -> None:
    """Checks a handler that forgets to return does not silently forward input."""
    worker = PipelineStageWorkerCore(
        PipelineStage(name="decode_latents", terminal=True),
        lambda payload: None,
    )

    result = worker.execute_stage(
        PipelineStagePayload(request_id="req-1", stage="decode_latents", data=["x"])
    )

    assert result.payload is None
    assert "returned None" in str(result.error)


def test_pipeline_topology_rejects_missing_next_stage() -> None:
    """Checks topology validation catches dangling stage edges."""
    with pytest.raises(ValueError, match="missing next stage"):
        PipelineTopology.from_value(
            {
                "stages": [
                    {"name": "encode", "next_stages": "missing"},
                    {"name": "decode", "terminal": True},
                ],
            }
        )


def test_serial_pipeline_runner_preserves_actor_payload_contract() -> None:
    """Checks in-process routing uses the same payload contract as stage actors."""
    topology = PipelineTopology.from_value(
        {
            "stages": [
                {"name": "denoise", "next_stages": "decode_latents"},
                {"name": "decode_latents", "terminal": True},
            ],
        }
    )

    runner = SerialPipelineRunner(
        topology,
        {
            "denoise": make_append_handler("denoise"),
            "decode_latents": make_append_handler("decode"),
        },
    )

    result = runner.run(
        PipelineStagePayload(
            request_id="req-1",
            stage="denoise",
            data=[],
            policy_version=7,
        )
    )

    assert result.error is None
    assert result.terminal is True
    assert result.payload is not None
    assert result.payload.stage == "decode_latents"
    assert result.payload.data == ["denoise", "decode"]
    assert result.payload.policy_version == 7
    assert [stage["stage"] for stage in result.metrics["stages"]] == [
        "denoise",
        "decode_latents",
    ]


def test_stage_worker_core_returns_error_result() -> None:
    """Checks stage exceptions stay inside the actor result envelope."""
    worker = PipelineStageWorkerCore(
        PipelineStage(name="decode_latents", terminal=True),
        _raising_handler,
    )

    result = worker.execute_stage(
        PipelineStagePayload(request_id="req-1", stage="decode_latents", data={})
    )

    assert result.payload is None
    assert result.terminal is True
    assert "decode failed" in str(result.error)
    assert result.metrics["stage"] == "decode_latents"
    assert result.metrics["wall_time_s"] >= 0.0


def test_ray_stage_worker_loads_handler_from_stage_config() -> None:
    """Checks the Ray actor adapter can build a handler from the stage spec."""
    worker = RayPipelineStageWorker(
        {
            "name": "decode_latents",
            "terminal": True,
            "handler": "tests.generation.pipeline.test_pipeline_contracts:make_append_handler",
            "handler_kwargs": {"suffix": "decode"},
        },
        worker_id="decode-0",
    )

    result = worker.execute_stage(
        PipelineStagePayload(request_id="req-1", stage="decode_latents", data=[])
    )

    assert result.error is None
    assert result.payload is not None
    assert result.payload.data == ["decode"]
    assert worker.worker_metadata()["worker_id"] == "decode-0"
    assert worker.worker_metadata()["stage"] == "decode_latents"


def test_ray_pipeline_runner_routes_across_actor_handles() -> None:
    """Checks the Ray-side pipeline runner routes across stage actor handles."""
    topology = PipelineTopology.from_value(
        {
            "stages": [
                {"name": "denoise", "next_stages": "decode_latents"},
                {"name": "decode_latents", "terminal": True},
            ],
        }
    )
    runner = RayPipelineRunner(
        topology,
        [
            RayPipelineStageHandle(
                stage="denoise",
                actor=_LocalStageActor("denoise", make_append_handler("denoise")),
            ),
            RayPipelineStageHandle(
                stage="decode_latents",
                actor=_LocalStageActor(
                    "decode_latents",
                    make_append_handler("decode"),
                ),
            ),
        ],
    )

    result = asyncio.run(
        runner.execute(PipelineStagePayload(request_id="req-1", stage="denoise", data=[]))
    )

    assert result.error is None
    assert result.terminal is True
    assert result.payload is not None
    assert result.payload.data == ["denoise", "decode"]
    assert [stage["stage"] for stage in result.metrics["stages"]] == [
        "denoise",
        "decode_latents",
    ]


def make_append_handler(suffix: str):
    """Builds a test stage handler that appends one marker."""

    def _handler(payload: PipelineStagePayload) -> PipelineStagePayload:
        data = list(payload.data)
        data.append(suffix)
        return payload.for_stage(payload.stage, data=data)

    return _handler


def _raising_handler(payload: PipelineStagePayload) -> None:
    del payload
    raise RuntimeError("decode failed")


class _LocalStageActor:
    def __init__(self, name: str, handler: Any) -> None:
        self._core = PipelineStageWorkerCore(
            PipelineStage(name=name, terminal=name == "decode_latents"),
            handler,
        )

    def execute_stage(self, payload: PipelineStagePayload) -> PipelineStageResult:
        return self._core.execute_stage(payload)
