"""One real Ray cluster for the whole ``tests/generation/ray`` package.

The package-scoped shell below overrides the function-scoped ``local_ray`` in
``tests/conftest.py``, so every test here pays one cluster start/stop between
them instead of one each.

The cluster's workers serve the tiny SANA snapshot: a Ray worker is a separate
process, so the test's own loader patch never reaches it, and
``tiny_sana_ray_cluster`` installs the same ``TinySanaPipeline`` loader
double at worker start (the Hub load is the one model double). Everything else
a worker runs -- the family registry, ``build_rollout``, the SANA executor,
the checkpoint identity -- is production code over real tiny weights.

Read ``real_local_ray``'s docstring before adding a test here: a shared cluster
has no per-test actor namespace, so every test must kill the actors it creates.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from tests.scripts.eval.fixtures import tiny_sana_ray_cluster, write_tiny_sana_snapshot


@pytest.fixture(scope="package")
def ray_sana_snapshot(tmp_path_factory) -> Path:
    """The one tiny SANA snapshot every Ray worker in this package serves."""

    return write_tiny_sana_snapshot(tmp_path_factory.mktemp("ray-sana") / "sana-snapshot")


@pytest.fixture(scope="package")
def local_ray(ray_sana_snapshot) -> Iterator[Any]:
    with tiny_sana_ray_cluster(ray_sana_snapshot) as ray:
        yield ray
