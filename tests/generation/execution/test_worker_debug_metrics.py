"""Tests for generation worker runtime debug metrics."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from vrl.generation.execution.worker import GenerationWorkerCore


def test_batch_output_debug_metrics_includes_stage_memory_and_counters() -> None:
    """Checks runtime debug keeps batch timing and memory counters."""

    output = SimpleNamespace(
        stage_durations={"denoise": 1.25, "decode": 0.5},
        engine_counters={
            "diffusion_samples_per_generation_batch": 8,
            "nested": {"scalar": torch.tensor(3)},
            "sequence": (None, {7: torch.tensor(4)}, [True, "ready"]),
            "non_scalar": torch.tensor([1, 2]),
        },
        peak_memory_mb=1234.5,
    )

    worker = object.__new__(GenerationWorkerCore)
    worker.worker_id = "rollout-0"
    metrics = worker._rank_metrics(runtime_debug=True, batch_output=output)

    assert metrics["rollout-0"] == {
        "stage_durations_s": {"denoise": 1.25, "decode": 0.5},
        "engine_counters": {
            "diffusion_samples_per_generation_batch": 8,
            "nested": {"scalar": 3},
            "sequence": [None, {"7": 4}, [True, "ready"]],
            "non_scalar": "tensor([1, 2])",
        },
        "peak_memory_mb": 1234.5,
    }


def test_disabled_debug_does_not_read_batch_properties() -> None:
    class UnreadableOutput:
        @property
        def stage_durations(self):
            raise AssertionError("disabled debug must not inspect batch metrics")

    worker = object.__new__(GenerationWorkerCore)
    assert worker._rank_metrics(runtime_debug=False, batch_output=UnreadableOutput()) == {}
    assert worker._rank_metrics(runtime_debug=True, batch_output=None) == {}
