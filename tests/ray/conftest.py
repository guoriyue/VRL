"""One real Ray cluster for the whole ``tests/ray`` package.

Every real-Ray test here used to hand-roll its own ``ray.init()``, so the four of
them paid four full cluster start/stop cycles — 17.6s of call time for 0.8s of
actual work. They now share one cluster: the package-scoped shell below overrides
the function-scoped ``local_ray`` in ``tests/conftest.py``, and the directory went
from ~19s to ~6s (measured over 3 runs each) while gaining three more real-Ray
tests — the two executor twins plus the probe-kill test CRD-02 converted.

Read ``real_local_ray``'s docstring before adding a test here — a shared cluster
has no per-test actor namespace, so every test must kill the actors it creates.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest

from tests.conftest import real_local_ray


@pytest.fixture(scope="package")
def local_ray() -> Iterator[Any]:
    with real_local_ray() as ray:
        yield ray
