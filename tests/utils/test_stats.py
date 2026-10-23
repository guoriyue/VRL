"""The stats accumulator, and how measurement sites reach the bound one."""

from __future__ import annotations

import asyncio
import threading

import pytest

from vrl.utils.stats import (
    Recorded,
    Stats,
    absorb,
    count,
    elapsed,
    gauge,
    recorded,
    recording,
    timed,
    unwrap,
)


def test_counters_and_phases_sum_and_gauges_keep_their_peak_across_merges() -> None:
    first, second = Stats(), Stats()
    with recording(first):
        elapsed("collect.engine_generate", 2.0)
        count("collect.sample_count", 3)
        gauge("continuous.producer_inflight", 4)
        gauge("continuous.producer_inflight", 1)
    with recording(second):
        elapsed("collect.engine_generate", 3.0)
        count("collect.sample_count", 2)
        gauge("continuous.producer_inflight", 3)

    first.merge(second)

    assert first.phase_seconds == {"collect.engine_generate": 5.0}
    assert first.counters == {"collect.sample_count": 5.0}
    assert first.gauges == {"continuous.producer_inflight": 4.0}
    assert first.as_metrics_dict() == {
        "collect.engine_generate": 5.0,
        "collect.sample_count": 5.0,
        "continuous.producer_inflight": 4.0,
    }


def test_non_finite_counts_and_gauges_are_rejected() -> None:
    with recording(), pytest.raises(ValueError, match="finite"):
        count("collect.sample_count", float("nan"))
    with recording(), pytest.raises(ValueError, match="finite"):
        gauge("continuous.item_age_s", float("inf"))


def test_profiled_phase_records_its_wall_clock_interval_and_merge_carries_it() -> None:
    with recording() as stats:
        with timed("evaluate"):
            pass
        with timed("backward", profile=True):
            pass
    merged = Stats()
    merged.merge(stats)

    assert set(stats.phase_seconds) == {"evaluate", "backward"}
    assert [name for name, _, _ in stats.phase_events] == ["backward"]
    assert all(start <= end for _, start, end in stats.phase_events)
    assert merged.phase_events == stats.phase_events
    assert merged.take_phase_events() == stats.phase_events and merged.phase_events == []


def test_measurements_reach_the_bound_accumulator_and_nothing_when_unbound() -> None:
    with timed("unbound"):
        pass
    count("unbound")
    gauge("unbound", 1.0)
    elapsed("unbound", 1.0)

    with recording() as stats:
        with timed("collect.put_s"):
            pass
        with timed("collect.put_s"):
            pass
        count("collect.put_bytes", 3)
        count("collect.put_bytes", 4)

    assert set(stats.phase_seconds) == {"collect.put_s"}
    assert stats.counters == {"collect.put_bytes": 7.0}
    with timed("after"):
        pass
    assert "after" not in stats.phase_seconds


def test_a_failing_block_still_reports_its_time() -> None:
    with recording() as stats, pytest.raises(RuntimeError), timed("evaluate"):
        raise RuntimeError("boom")

    assert "evaluate" in stats.phase_seconds


def test_timed_decorates_functions_and_methods_per_call() -> None:
    @timed("reward.latency_s")
    def score(value: int) -> int:
        return value * 2

    class Runtime:
        @timed("rollout.activate_generation_runtime_s")
        def activate(self) -> str:
            return "active"

    assert score(1) == 2  # unbound: the call still runs
    with recording() as stats:
        assert score(2) == 4
        assert score(3) == 6
        assert Runtime().activate() == "active"

    assert set(stats.phase_seconds) == {
        "reward.latency_s",
        "rollout.activate_generation_runtime_s",
    }
    assert score.__name__ == "score"


def test_timed_coroutine_times_the_awaited_body_not_its_creation() -> None:
    @timed("rollout.weight_sync_s")
    async def push() -> str:
        await asyncio.sleep(0.02)
        return "pushed"

    async def run() -> tuple[Stats, Stats]:
        with recording() as at_creation:
            pending = push()
        with recording() as at_await:
            assert await pending == "pushed"
        return at_creation, at_await

    at_creation, at_await = asyncio.run(run())

    assert at_creation.phase_seconds == {}
    assert at_await.phase_seconds["rollout.weight_sync_s"] >= 0.02


def test_an_inner_binding_shadows_the_outer_until_it_closes() -> None:
    with recording() as outer:
        with recording() as inner:
            count("calls")
        count("calls")
        count("calls")

    assert inner.counters == {"calls": 1.0}
    assert outer.counters == {"calls": 2.0}


def test_the_binding_follows_tasks_and_to_thread_but_not_a_plain_thread() -> None:
    def plain_thread() -> None:
        count("plain_thread")

    async def child_task() -> None:
        count("task")

    async def collect() -> Stats:
        with recording() as stats:
            await asyncio.create_task(child_task())
            await asyncio.to_thread(count, "to_thread")
            thread = threading.Thread(target=plain_thread)
            thread.start()
            thread.join()
        return stats

    stats = asyncio.run(collect())

    assert stats.counters == {"task": 1.0, "to_thread": 1.0}


def test_concurrent_tasks_keep_their_own_bindings() -> None:
    first, second = Stats(), Stats()

    async def request(stats: Stats, value: float) -> None:
        with recording(stats):
            await asyncio.sleep(0)
            count("samples", value)

    async def both() -> None:
        await asyncio.gather(request(first, 1), request(second, 5))

    asyncio.run(both())

    assert first.counters == {"samples": 1.0}
    assert second.counters == {"samples": 5.0}


def test_absorb_folds_an_accumulator_into_the_bound_one_and_nothing_when_unbound() -> None:
    with recording() as remote:
        count("collect.put_bytes", 8)
    absorb(remote)

    with recording() as local:
        absorb(remote)

    assert local.counters == {"collect.put_bytes": 8.0}


def test_recorded_returns_the_call_scope_and_unwrap_folds_it_into_the_caller() -> None:
    class Worker:
        @recorded
        def execute(self, value: int) -> int:
            count("generation.calls")
            return value + 1

        @recorded
        async def stage(self, value: int) -> int:
            await asyncio.sleep(0)
            count("generation.staged")
            return value * 10

    worker = Worker()
    with recording() as caller:
        result = worker.execute(1)
        # The call opened its own scope: nothing reached the caller yet.
        assert isinstance(result, Recorded) and caller.counters == {}
        assert unwrap(result) == 2
        assert unwrap(asyncio.run(worker.stage(2))) == 20

    assert result.stats.counters == {"generation.calls": 1.0}
    assert caller.counters == {"generation.calls": 1.0, "generation.staged": 1.0}


def test_a_raising_recorded_call_drops_its_stats() -> None:
    @recorded
    def fail() -> None:
        count("generation.calls")
        raise RuntimeError("boom")

    with recording() as caller, pytest.raises(RuntimeError):
        fail()

    assert caller.counters == {}
