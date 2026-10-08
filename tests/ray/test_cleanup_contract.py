"""Failure observability for handle-scoped Ray cleanup helpers, on the real cluster.

The failures are the real ``ray`` API's own: ``ray.kill`` refuses a handle that
is not an actor, and ``remove_placement_group`` fails on one that is not a
placement group. The sweep around a failure still releases the real handles.
"""

from __future__ import annotations

import logging
import time

from vrl.ray.dependencies import kill_actors
from vrl.ray.placement import remove_placement_group


class _Pinger:
    def ping(self) -> bool:
        return True


def test_actor_cleanup_failure_is_logged_and_the_sweep_continues(local_ray, caplog) -> None:
    ray = local_ray
    actor = ray.remote(num_cpus=0)(_Pinger).remote()
    assert ray.get(actor.ping.remote(), timeout=30) is True

    with caplog.at_level(logging.WARNING, logger="vrl.ray.dependencies"):
        failures = kill_actors(ray, ["actor-1", actor])

    assert len(failures) == 1
    assert failures[0][0] == "actor-1"
    assert isinstance(failures[0][1], ValueError)
    assert "Failed to kill owned Ray actor 'actor-1'" in caplog.text
    assert "ray.kill() only supported for actors" in caplog.text
    # The real actor after the failed one was still killed.
    deadline = time.monotonic() + 30
    while True:
        try:
            ray.get(actor.ping.remote(), timeout=5)
        except ray.exceptions.RayActorError:
            break
        assert time.monotonic() < deadline, "the sweep left the real actor alive"
        time.sleep(0.05)


def test_placement_cleanup_failure_is_logged(local_ray, caplog) -> None:
    from ray.util.placement_group import placement_group, placement_group_table

    with caplog.at_level(logging.WARNING, logger="vrl.ray.placement"):
        failure = remove_placement_group("pg-1")

    assert isinstance(failure, AttributeError)
    assert "Failed to remove owned Ray placement group 'pg-1'" in caplog.text

    # A real handle is removed and reports no failure.
    pg = placement_group([{"CPU": 0.1}])
    local_ray.get(pg.ready(), timeout=30)
    assert remove_placement_group(pg) is None
    assert placement_group_table(pg)["state"] == "REMOVED"
