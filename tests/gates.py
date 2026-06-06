"""Shared skip gates for tests that need resources plain CI lacks.

A "gate" is a ``pytest.mark.skipif`` evaluated once per session: the test runs
for real when the resource is present and is skipped (not failed) otherwise. So
``pytest`` on a CPU box stays green while the same test runs as a real
integration check on a GPU/checkpoint-equipped machine.
"""

from __future__ import annotations

import pytest

try:  # torch may be importable without a usable CUDA device
    import torch

    _HAS_CUDA = bool(torch.cuda.is_available())
except Exception:  # pragma: no cover - torch import/driver failure
    _HAS_CUDA = False

#: Skip the decorated test unless a CUDA GPU is available.
requires_cuda = pytest.mark.skipif(not _HAS_CUDA, reason="requires a CUDA GPU")
