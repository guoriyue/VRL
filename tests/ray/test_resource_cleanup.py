"""Failure observability for handle-scoped Ray cleanup helpers."""

from __future__ import annotations

import logging
import sys
from types import SimpleNamespace

import pytest

from vrl.ray.resource_cleanup import kill_actors, remove_placement_group

# Both tests below inject the same thing for the same reason, so the label is a
# module constant instead of the same string twice (the `requires_fp8` precedent
# in tests/nn/quantization/test_fp8.py).
_REAL_RAY_CLEANUP = pytest.mark.real_cover(
    "tests/ray/test_global_placement.py"
    "::test_owner_reserves_trainer_gpu_and_binds_roles_on_simulated_gpus",
    why=(
        "a live Ray cluster cannot be told to fail ray.kill / remove_placement_group on "
        "demand, and these tests assert the WARNING text emitted on that failure; the "
        "cleanup path against a real cluster is exercised by the slow_test twin's shutdown"
    ),
)


@_REAL_RAY_CLEANUP
def test_actor_cleanup_failure_is_logged(caplog) -> None:
    class _Ray:
        @staticmethod
        def kill(actor, *, no_restart: bool) -> None:
            del actor, no_restart
            raise RuntimeError("kill failed")

    with caplog.at_level(logging.WARNING, logger="vrl.ray.resource_cleanup"):
        failures = kill_actors(_Ray(), ["actor-1"])

    assert len(failures) == 1
    assert failures[0][0] == "actor-1"
    assert isinstance(failures[0][1], RuntimeError)
    assert "Failed to kill owned Ray actor 'actor-1'" in caplog.text
    assert "kill failed" in caplog.text


@_REAL_RAY_CLEANUP
def test_placement_cleanup_failure_is_logged(monkeypatch, caplog) -> None:
    def _remove(placement_group) -> None:
        del placement_group
        raise RuntimeError("remove failed")

    monkeypatch.setitem(
        sys.modules,
        "ray.util",
        SimpleNamespace(remove_placement_group=_remove),
    )

    with caplog.at_level(logging.WARNING, logger="vrl.ray.resource_cleanup"):
        failure = remove_placement_group("pg-1")

    assert isinstance(failure, RuntimeError)
    assert "Failed to remove owned Ray placement group 'pg-1'" in caplog.text
    assert "remove failed" in caplog.text
