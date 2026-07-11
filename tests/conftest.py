"""Shared pytest configuration.

Two collection-time gates live here:

- ``gpu``: tests marked ``@pytest.mark.gpu`` auto-skip when no CUDA device is
  present, and run for real when one is. The marker is the single source of truth
  — it both selects (``-m gpu``) and skips, so GPU tests carry one decorator, not
  a separate skip gate. (This deviates from vLLM, which keeps selection and skip
  separate and routes via a CI fleet; a single-machine setup is better served by
  graceful auto-skip.)
- ``distributed``: skipped unless ``--distributed`` is passed or
  ``VRL_RUN_DISTRIBUTED_TESTS=1`` is set. Distributed tests need an explicit lane
  because local Ray smoke tests and multi-node/GPU distributed tests have very
  different resource profiles.
- ``optional``: skipped unless ``--optional`` is passed (verbatim vLLM behavior).
"""

from __future__ import annotations

import logging

import pytest

from tests import ci_envs

try:  # torch may be importable without a usable CUDA device
    import torch

    _HAS_CUDA = bool(torch.cuda.is_available())
except Exception:  # pragma: no cover - torch import/driver failure
    _HAS_CUDA = False


def pytest_addoption(parser):
    parser.addoption(
        "--optional", action="store_true", default=False, help="run optional test"
    )
    parser.addoption(
        "--distributed",
        action="store_true",
        default=False,
        help="run distributed test",
    )


def pytest_collection_modifyitems(config, items):
    # GPU tests skip on machines without CUDA.
    if not _HAS_CUDA:
        skip_gpu = pytest.mark.skip(reason="requires a CUDA GPU")
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(skip_gpu)
    # NOTE: the `distributed` and `optional` gating branches below are reserved
    # vLLM-parity lanes (commit "vLLM-style marker gating") with no current
    # members — `@pytest.mark.distributed`/`@pytest.mark.optional` is unused
    # repo-wide. Keep them: they preserve structural parity with vLLM's marker
    # scaffold and are the opt-in points for future real multi-node/optional
    # suites. Do not delete as dead code.
    #
    # Distributed tests need an explicit lane; slow local Ray tests should use
    # slow_test instead of distributed.
    if not (config.getoption("--distributed") or ci_envs.VRL_RUN_DISTRIBUTED_TESTS):
        skip_distributed = pytest.mark.skip(
            reason="needs --distributed or VRL_RUN_DISTRIBUTED_TESTS=1",
        )
        for item in items:
            if "distributed" in item.keywords:
                item.add_marker(skip_distributed)
    # optional tests are skipped unless --optional is given on the cli.
    if not config.getoption("--optional"):
        skip_optional = pytest.mark.skip(reason="need --optional option to run")
        for item in items:
            if "optional" in item.keywords:
                item.add_marker(skip_optional)


@pytest.fixture()
def local_ray():
    """Real local Ray cluster (small, CPU-only) for real-Ray unit tests.

    ``address="local"`` always starts a fresh cluster, so an operator cluster
    already running on the host is never hijacked; teardown disconnects and
    stops only the processes this driver spawned (never ``ray stop``).
    """
    ray = pytest.importorskip("ray")
    ray.shutdown()
    ray.init(
        address="local",
        num_cpus=2,
        include_dashboard=False,
        log_to_driver=False,
    )
    yield ray
    ray.shutdown()


@pytest.fixture(autouse=True)
def _propagate_vrl_logs():
    """vrl.utils.logging sets propagate=False on the "vrl" logger so production
    output is emitted exactly once; caplog relies on propagation to the root
    logger, so re-enable it for the duration of each test."""
    vrl_logger = logging.getLogger("vrl")
    previous = vrl_logger.propagate
    vrl_logger.propagate = True
    yield
    vrl_logger.propagate = previous
