"""Ray cluster ownership tests for the online recipe (fake Ray only)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from vrl.scripts.common import online


class _FakeRay:
    __version__ = "test-ray"

    def __init__(self, *, initialized: bool = False) -> None:
        self.initialized = initialized
        self.init_calls: list[dict[str, str]] = []
        self.shutdown_calls = 0

    def is_initialized(self) -> bool:
        return self.initialized

    def init(self, **kwargs):
        self.init_calls.append(dict(kwargs))
        self.initialized = True
        return SimpleNamespace(
            address_info={
                "gcs_address": "127.0.0.1:10001",
                "session_dir": "/tmp/ray/test-session",
            },
        )

    def shutdown(self) -> None:
        self.shutdown_calls += 1
        self.initialized = False


def _resources(*, cross_node: bool) -> SimpleNamespace:
    return SimpleNamespace(cross_node=cross_node)


def test_local_run_starts_owned_cluster_even_when_environment_has_address() -> None:
    ray = _FakeRay()

    session = online._initialize_ray_cluster(
        _resources(cross_node=False),
        ray,
        environ={"RAY_ADDRESS": "10.0.0.9:6379"},
    )
    session.shutdown()
    session.shutdown()

    assert ray.init_calls == [{"address": "local"}]
    assert ray.shutdown_calls == 1


@pytest.mark.parametrize("address", [None, "", "auto", "local"])
def test_cross_node_requires_concrete_operator_address(address: str | None) -> None:
    ray = _FakeRay()
    environ = {} if address is None else {"RAY_ADDRESS": address}

    with pytest.raises(ValueError, match="requires a concrete RAY_ADDRESS"):
        online._initialize_ray_cluster(
            _resources(cross_node=True),
            ray,
            environ=environ,
        )

    assert ray.init_calls == []
    assert ray.shutdown_calls == 0


def test_cross_node_attaches_only_to_explicit_address() -> None:
    ray = _FakeRay()

    session = online._initialize_ray_cluster(
        _resources(cross_node=True),
        ray,
        environ={"RAY_ADDRESS": "10.0.0.1:6379"},
    )
    session.shutdown()

    assert ray.init_calls == [{"address": "10.0.0.1:6379"}]
    # Ray defines shutdown on an attached driver as disconnect-only.
    assert ray.shutdown_calls == 1


def test_preinitialized_connection_remains_owned_by_embedding_caller() -> None:
    ray = _FakeRay(initialized=True)

    session = online._initialize_ray_cluster(
        _resources(cross_node=False),
        ray,
        environ={},
    )
    session.shutdown()

    assert ray.init_calls == []
    assert ray.shutdown_calls == 0
