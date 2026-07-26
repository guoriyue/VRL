"""Ray cluster ownership tests for the online recipe, against real Ray.

Every test drives ``online._RayClusterSession.connect`` with the real ``ray``
module; ownership is observed through ``ray.is_initialized()`` and the
connected cluster's GCS address instead of recorded fake calls. Error-path
tests raise before ``ray.init`` and stay in the fast lane; tests that start a
real cluster are ``slow_test`` (nightly lane). The attach test starts its
operator cluster in a subprocess and tears it down with the subprocess's own
``ray.shutdown()`` — never ``ray stop``, which would kill unrelated clusters
on a shared host.
"""

from __future__ import annotations

import signal
import subprocess
import sys

import pytest

from vrl.scripts.common import online

ray = pytest.importorskip("ray")


@pytest.fixture()
def isolated_ray():
    """Real ray with no driver connection before or after the test."""
    ray.shutdown()
    yield ray
    ray.shutdown()


@pytest.mark.slow_test
def test_local_run_starts_owned_cluster_even_when_environment_has_address(
    isolated_ray, monkeypatch
) -> None:
    """A stale RAY_ADDRESS (another user's or a dead cluster's) must not hijack
    a single-node run: 10.0.0.9 is unroutable, so accidentally attaching would
    fail instead of starting the owned local cluster asserted below."""
    monkeypatch.setenv("RAY_ADDRESS", "10.0.0.9:6379")

    session = online._RayClusterSession.connect(
        isolated_ray,
        cross_node=False,
        environ={"RAY_ADDRESS": "10.0.0.9:6379"},
        local_num_cpus=3,
    )

    assert isolated_ray.is_initialized()
    assert isolated_ray.cluster_resources()["CPU"] == 3
    session.shutdown()
    session.shutdown()  # idempotent: the second call must be a no-op
    assert not isolated_ray.is_initialized()


@pytest.mark.parametrize("address", [None, "", "auto", "local"])
def test_cross_node_requires_concrete_operator_address(isolated_ray, address: str | None) -> None:
    environ = {} if address is None else {"RAY_ADDRESS": address}

    with pytest.raises(ValueError, match="requires a concrete RAY_ADDRESS"):
        online._RayClusterSession.connect(
            isolated_ray,
            cross_node=True,
            environ=environ,
        )

    assert not isolated_ray.is_initialized()


_OPERATOR_HEAD_SCRIPT = """\
import sys

import ray

context = ray.init(
    address="local", num_cpus=1, include_dashboard=False, log_to_driver=False
)
print(context.address_info["gcs_address"], flush=True)
sys.stdin.readline()
ray.shutdown()
"""


@pytest.mark.slow_test
def test_cross_node_attaches_only_to_explicit_address(isolated_ray) -> None:
    """Cross-node runs join the operator-owned cluster at the explicit address
    (never a second local cluster), and session shutdown is disconnect-only:
    the operator cluster must survive the recipe driver."""
    head = subprocess.Popen(
        [sys.executable, "-c", _OPERATOR_HEAD_SCRIPT],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        address = head.stdout.readline().strip()
        assert address, "operator head subprocess failed to start a cluster"

        session = online._RayClusterSession.connect(
            isolated_ray,
            cross_node=True,
            environ={"RAY_ADDRESS": address},
            # Attached clusters own their resources; this local-only capacity
            # must not be forwarded to ray.init(address=<existing>).
            local_num_cpus=5,
        )

        assert isolated_ray.is_initialized()
        assert isolated_ray.get_runtime_context().gcs_address == address

        session.shutdown()
        assert not isolated_ray.is_initialized()

        # The operator cluster outlived our disconnect: its own driver can
        # still shut it down cleanly.
        head.stdin.write("stop\n")
        head.stdin.flush()
        assert head.wait(timeout=60) == 0
    finally:
        if head.poll() is None:
            head.kill()
            head.wait(timeout=30)


@pytest.mark.slow_test
def test_preinitialized_connection_remains_owned_by_embedding_caller(
    isolated_ray,
) -> None:
    isolated_ray.init(address="local", num_cpus=1, include_dashboard=False, log_to_driver=False)

    session = online._RayClusterSession.connect(
        isolated_ray,
        cross_node=False,
        environ={},
        local_num_cpus=5,
    )
    session.shutdown()

    # The embedding caller's connection must survive the recipe session.
    assert isolated_ray.is_initialized()


@pytest.mark.slow_test
def test_ray_init_cannot_steal_cli_signal_handlers(isolated_ray) -> None:
    """Real ray.init overwrites the driver SIGTERM handler (ray._private.worker
    documents this); the recipe must restore the launcher-owned handlers
    around init."""

    def cli_handler(signum, frame):  # pragma: no cover - never invoked
        del signum, frame

    previous = {
        signal.SIGTERM: signal.getsignal(signal.SIGTERM),
        signal.SIGINT: signal.getsignal(signal.SIGINT),
    }
    try:
        signal.signal(signal.SIGTERM, cli_handler)
        signal.signal(signal.SIGINT, cli_handler)

        online._RayClusterSession.connect(
            isolated_ray,
            cross_node=False,
            environ={},
        )

        assert signal.getsignal(signal.SIGTERM) is cli_handler
        assert signal.getsignal(signal.SIGINT) is cli_handler
    finally:
        for signum, handler in previous.items():
            if handler is not None:
                signal.signal(signum, handler)
