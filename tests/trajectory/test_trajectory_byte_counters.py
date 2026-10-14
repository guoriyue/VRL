"""Tests for trajectory tensor byte counters."""

from __future__ import annotations

import torch

from vrl.trajectory.storage import trajectory_tensor_bytes


def test_byte_counter_deduplicates_shared_references() -> None:
    """The same tensor referenced twice is counted once."""
    tensor = torch.zeros(4, dtype=torch.float32)

    assert trajectory_tensor_bytes({"a": tensor, "b": tensor}) == 16
