"""Tests for the per-request rollout stats accumulator."""

from __future__ import annotations

from vrl.rollouts.stats import RolloutStats


def test_counter_sums_without_entering_phase_percentage_base() -> None:
    s = RolloutStats()
    s.add_phase("denoise", 2.0)
    s.add_counter("collect.sample_count", 3)
    s.add_counter("collect.sample_count", 2)

    assert s.counters == {"collect.sample_count": 5.0}
    assert s.as_metrics_dict()["collect.sample_count"] == 5.0


def test_gauge_merge_retains_peak_without_summing_snapshots() -> None:
    first = RolloutStats()
    first.observe_gauge("continuous.stale_policy_versions", 1)
    first.observe_gauge("continuous.producer_inflight", 4)
    second = RolloutStats()
    second.observe_gauge("continuous.stale_policy_versions", 0)
    second.observe_gauge("continuous.producer_inflight", 3)

    first.merge(second)

    assert first.gauges == {
        "continuous.stale_policy_versions": 1.0,
        "continuous.producer_inflight": 4.0,
    }
    assert first.as_metrics_dict()["continuous.stale_policy_versions"] == 1.0


def test_merge_sums_phases_and_all_reward_calls() -> None:
    a = RolloutStats()
    a.add_phase("collect.engine_generate", 2.0)
    a.fold_reward_timing({"inference_ms": 10.0})
    b = RolloutStats()
    b.add_phase("collect.engine_generate", 3.0)
    b.fold_reward_timing({"inference_ms": 20.0})
    a.merge(b)
    assert a.phase_seconds["collect.engine_generate"] == 5.0
    assert a.phase_seconds["reward.inference_s"] == 0.03
    assert a.counters["reward.call_count"] == 2


def test_reward_timings_become_second_phases_and_count_the_call() -> None:
    s = RolloutStats()
    s.fold_reward_timing(
        {"latency_ms": 5.0, "queue_wait_ms": 1.0, "artifact_validation_ms": 2.0, "scores": 3}
    )
    s.fold_reward_timing({})

    metrics = s.as_metrics_dict()
    assert metrics["reward.latency_s"] == 0.005
    assert metrics["reward.queue_wait_s"] == 0.001
    assert metrics["reward.artifact_validation_s"] == 0.002
    assert metrics["reward.call_count"] == 1.0
    assert "scores" not in metrics and "reward.scores_s" not in metrics


def test_profiled_phase_records_its_wall_clock_interval_and_merge_carries_it() -> None:
    s = RolloutStats()
    with s.phase("evaluate"):
        pass
    with s.phase("backward", profile=True):
        pass
    merged = RolloutStats()
    merged.merge(s)

    assert set(s.phase_seconds) == {"evaluate", "backward"}
    assert [name for name, _, _ in s.phase_events] == ["backward"]
    assert all(start <= end for _, start, end in s.phase_events)
    assert merged.phase_events == s.phase_events
    assert merged.take_phase_events() == s.phase_events and merged.phase_events == []
