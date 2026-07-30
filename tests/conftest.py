"""Shared pytest configuration.

Opt-in test lanes live here:

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
- ``rollout_preview``: test-owned few-shot preview; it skips unless an exact
  experiment config and fresh output directory are supplied on the command line.

``--real-cover-report`` prints the ``real_cover`` register: every test that
labels a double it cannot make real in-process, the real counterpart it names,
and the lane that counterpart lives in. The lane is the payload — a ``gpu``
counterpart can still be skipped on this very machine — so the register never
claims "covered", only "covered over there".
"""

from __future__ import annotations

import logging
from collections.abc import Callable

import pytest

from tests import ci_envs, real_cover

try:  # torch may be importable without a usable CUDA device
    import torch

    _HAS_CUDA = bool(torch.cuda.is_available())
except Exception:  # pragma: no cover - torch import/driver failure
    _HAS_CUDA = False


_REAL_COVER_REGISTER = pytest.StashKey[list[tuple[str, pytest.Mark]]]()


def pytest_addoption(parser):
    parser.addoption("--optional", action="store_true", default=False, help="run optional test")
    parser.addoption(
        "--distributed",
        action="store_true",
        default=False,
        help="run distributed test",
    )
    parser.addoption(
        "--real-cover-report",
        action="store_true",
        default=False,
        help="print the real_cover register: labelled double -> real counterpart + its lane",
    )
    preview = parser.getgroup("rollout preview")
    preview.addoption(
        "--rollout-preview-config",
        default=None,
        help="bundled experiment config or absolute YAML path to preview",
    )
    preview.addoption(
        "--rollout-preview-dir",
        default=None,
        help="new directory where the preview writes individual images",
    )


def pytest_collection_modifyitems(config, items):
    # GPU tests skip on machines without CUDA.
    if not _HAS_CUDA:
        skip_gpu = pytest.mark.skip(reason="requires a CUDA GPU")
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(skip_gpu)
    # NOTE: the `optional` branch below is a reserved vLLM-parity lane (commit
    # "vLLM-style marker gating") with no current members — `@pytest.mark.optional`
    # is unused repo-wide. Keep it: it preserves structural parity with vLLM's
    # marker scaffold and is the opt-in point for future optional suites. Do not
    # delete as dead code.
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
    if config.getoption("--real-cover-report"):
        config.stash[_REAL_COVER_REGISTER] = [
            (item.nodeid, mark) for item in items for mark in item.iter_markers(real_cover.MARKER)
        ]


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Print the real_cover register, resolving each target's lane on disk."""

    del exitstatus
    register = config.stash.get(_REAL_COVER_REGISTER, None)
    if register is None:
        return
    write = terminalreporter.write_line
    write("")
    # Counts collected tests, not label call sites: one module-level label or one
    # reused constant covers many tests, and the register lists them per test.
    write(f"real_cover register  ({len(register)} labelled tests: double -> counterpart / gap)")
    for nodeid, mark in sorted(register):
        target = mark.args[0] if mark.args else None
        lane = real_cover.resolve_target(target).lane_label if target else "-"
        write(f"  {nodeid}")
        write(f"      -> {target or 'NO REAL COUNTERPART'}   [lane: {lane}]")
        if mark.kwargs.get("why"):
            write(f"      why: {mark.kwargs['why']}")
        if mark.kwargs.get("tracked_in"):
            write(f"      tracked_in: {mark.kwargs['tracked_in']}")


@pytest.fixture()
def local_ray():
    """Real local Ray cluster (small, CPU-only) for real-Ray unit tests.

    ``address="local"`` always starts a fresh cluster, so an operator cluster
    already running on the host is never hijacked; teardown disconnects and
    stops only the processes this driver spawned (never ``ray stop``).
    """
    ray = pytest.importorskip("ray")
    from ray._private import ray_constants

    ray.shutdown()
    # Ray's uv hook packages the entire current working directory and launches
    # workers through a newly resolved project environment. That is useful for a
    # remote cluster, but it breaks this local fixture's contract: test actors then
    # lose the driver's dev dependencies (including pytest) and a small unit test
    # uploads hundreds of MiB. A local cluster must inherit the already-active test
    # interpreter instead.
    previous_uv_hook = ray_constants.RAY_ENABLE_UV_RUN_RUNTIME_ENV
    ray_constants.RAY_ENABLE_UV_RUN_RUNTIME_ENV = False
    try:
        ray.init(
            address="local",
            num_cpus=2,
            include_dashboard=False,
            log_to_driver=False,
            _skip_env_hook=True,
        )
        yield ray
    finally:
        ray.shutdown()
        ray_constants.RAY_ENABLE_UV_RUN_RUNTIME_ENV = previous_uv_hook


@pytest.fixture()
def cuda_devices(monkeypatch) -> Callable[[int], None]:
    """Pin the host GPU topology at the torch layer for the length of one test.

    Patch torch, never our own resolvers: replacing
    ``vrl.ray.resources._auto_visible_cuda_devices`` with a lambda made the
    asserted device tuple a value the test itself wrote, leaving the wrapper's
    ``is_available`` short circuit, ``int()`` coercion and exception fallback
    unexecuted. Pinning ``torch.cuda`` runs the real wrapper on a topology this
    single-GPU host does not have.
    """

    def pin(count: int) -> None:
        torch = pytest.importorskip("torch")
        monkeypatch.setattr(torch.cuda, "is_available", lambda: count > 0)
        monkeypatch.setattr(torch.cuda, "device_count", lambda: count)

    return pin


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
