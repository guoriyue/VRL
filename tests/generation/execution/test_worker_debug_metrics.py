"""Tests for generation worker runtime debug metrics."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from vrl.generation.execution.worker import _batch_output_debug_metrics


def test_batch_output_debug_metrics_includes_stage_memory_and_counters() -> None:
    """Checks runtime debug keeps batch timing and memory counters."""

    output = SimpleNamespace(
        stage_durations={"denoise": 1.25, "decode": 0.5},
        engine_counters={
            "diffusion_samples_per_generation_batch": 8,
            "nested": {"scalar": torch.tensor(3)},
        },
        peak_memory_mb=1234.5,
    )

    metrics = _batch_output_debug_metrics(output)

    assert metrics == {
        "stage_durations_s": {"denoise": 1.25, "decode": 0.5},
        "engine_counters": {
            "diffusion_samples_per_generation_batch": 8,
            "nested": {"scalar": 3},
        },
        "peak_memory_mb": 1234.5,
    }
