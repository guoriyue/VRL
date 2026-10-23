"""Tests for generation worker runtime debug metrics."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from tests.generation.execution._helpers import launch_config
from vrl.generation.execution.worker import GenerationWorkerCore


def test_batch_output_debug_metrics_includes_memory_and_counters() -> None:
    """Checks runtime debug keeps batch memory and engine counters."""

    output = SimpleNamespace(
        engine_counters={
            "diffusion_samples_per_generation_batch": 8,
            "nested": {"scalar": torch.tensor(3)},
            "sequence": (None, {7: torch.tensor(4)}, [True, "ready"]),
            "non_scalar": torch.tensor([1, 2]),
        },
        peak_memory_mb=1234.5,
    )

    worker = GenerationWorkerCore("rollout-0", launch_config())
    metrics = worker._rank_metrics(runtime_debug=True, batch_output=output)

    assert metrics["rollout-0"] == {
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
        def engine_counters(self):
            raise AssertionError("disabled debug must not inspect batch metrics")

    worker = GenerationWorkerCore("rollout-0", launch_config())
    assert worker._rank_metrics(runtime_debug=False, batch_output=UnreadableOutput()) == {}
