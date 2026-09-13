"""Tests for host-memory guard helpers."""

from __future__ import annotations

from vrl.utils.cuda_memory import is_cuda_out_of_memory
from vrl.utils.memory import HostMemorySnapshot


def test_format_host_memory_omits_unknown_fields() -> None:
    """``str(snapshot)`` prints only the fields the snapshot knows: ``rss`` alone when
    available/total are unknown.
    """
    snapshot = HostMemorySnapshot(rss_mb=10.0, available_mb=None, total_mb=None)

    assert str(snapshot) == "rss=10.0MiB"


def test_cuda_oom_detection_prefers_the_typed_exception() -> None:
    """A typed CUDA OOM remains detectable even if its message format changes."""
    import torch

    assert is_cuda_out_of_memory(torch.cuda.OutOfMemoryError("allocation failed"))
