"""Tests for optional torch profiler helpers."""

from __future__ import annotations

import torch

from vrl.trainers.profiling import torch_profiler_step
from vrl.trainers.types import TorchProfilerConfig


def test_torch_profiler_step_writes_tensorboard_trace(tmp_path) -> None:
    config = TorchProfilerConfig(
        enabled=True,
        output_dir=str(tmp_path),
        activities=("cpu",),
        max_steps=1,
    )

    with torch_profiler_step(
        config,
        output_dir="unused",
        step=0,
        device=torch.device("cpu"),
        worker_name="unit/trainer",
    ):
        x = torch.ones(4)
        _ = (x + 1).sum().item()

    trace_dir = tmp_path / "trainer"
    assert trace_dir.exists()
    assert any(path.name.endswith(".pt.trace.json") for path in trace_dir.rglob("*"))
    assert any(path.name.endswith(".summary.txt") for path in trace_dir.rglob("*"))
