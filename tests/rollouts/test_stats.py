"""Tests for the per-request rollout stats accumulator and sinks."""

from __future__ import annotations

import json
import logging

from vrl.rollouts.stats import LoggingStatsSink, RolloutStats


def test_add_phase_sums_on_repeat() -> None:
    s = RolloutStats()
    s.add_phase("denoise", 1.0)
    s.add_phase("denoise", 0.5)
    assert s.phase_seconds["denoise"] == 1.5


def test_counter_sums_without_entering_phase_percentage_base() -> None:
    s = RolloutStats()
    s.add_phase("denoise", 2.0)
    s.add_counter("collect.sample_count", 3)
    s.add_counter("collect.sample_count", 2)

    assert s.counters == {"collect.sample_count": 5.0}
    assert s.as_phase_dict()["collect.sample_count"] == 5.0


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
    assert first.as_phase_dict()["continuous.stale_policy_versions"] == 1.0


def test_merge_sums_phases_and_all_reward_calls() -> None:
    a = RolloutStats()
    a.add_phase("collect.engine_generate", 2.0)
    a.fold_reward_timing(inference_ms=10.0)
    b = RolloutStats()
    b.add_phase("collect.engine_generate", 3.0)
    b.fold_reward_timing(inference_ms=20.0)
    a.merge(b)
    assert a.phase_seconds["collect.engine_generate"] == 5.0
    assert a.reward_inference_ms == 30.0
    assert a.counters["reward.call_count"] == 2


def test_fold_reward_timing_records_typed_fields() -> None:
    s = RolloutStats()
    s.fold_reward_timing(latency_ms=5.0, queue_wait_ms=1.0, inference_ms=4.0)
    assert s.reward_queue_wait_ms == 1.0
    assert s.reward_inference_ms == 4.0
    assert s.counters["reward.call_count"] == 1
    assert s.as_phase_dict()["reward.latency_s"] == 0.005


def test_reward_timing_aggregates_calls_percentiles_and_extra_phases() -> None:
    s = RolloutStats()
    for latency in (10.0, 30.0, 20.0):
        s.fold_reward_timing(
            latency_ms=latency,
            extra_ms={"artifact_validation_ms": 2.0},
        )

    metrics = s.as_phase_dict()
    assert metrics["reward.latency_s"] == 0.06
    assert metrics["reward.call_count"] == 3.0
    assert metrics["reward.latency_p50_s"] == 0.02
    assert metrics["reward.latency_p95_s"] == 0.03
    assert metrics["reward.artifact_validation_s"] == 0.006


def test_reward_extra_timing_rejects_non_timing_names() -> None:
    s = RolloutStats()
    try:
        s.fold_reward_timing(extra_ms={"artifact_validation": 1.0})
    except ValueError as error:
        assert "end with '_ms'" in str(error)
    else:  # pragma: no cover - assertion aid
        raise AssertionError("invalid reward timing name was accepted")


def test_as_phase_dict_surfaces_reward_as_seconds() -> None:
    s = RolloutStats()
    s.add_phase("denoise", 1.0)
    s.fold_reward_timing(inference_ms=2000.0)
    d = s.as_phase_dict()
    assert d["denoise"] == 1.0
    assert d["reward.inference_s"] == 2.0


def test_add_phases_accumulates_mapping() -> None:
    s = RolloutStats()
    s.add_phases({"a": 1.0, "b": 2.0})
    assert s.phase_seconds == {"a": 1.0, "b": 2.0}
    assert RolloutStats().phase_seconds == {}


def test_logging_sink_excludes_collect_from_percent_base(caplog) -> None:
    s = RolloutStats()
    s.add_phase("denoise", 3.0)
    s.add_phase("collect.engine_generate", 7.0)  # excluded from total base
    sink = LoggingStatsSink(logging.getLogger("vrl.stats.test"))
    with caplog.at_level(logging.INFO, logger="vrl.stats.test"):
        sink.record(5, s)
    msg = caplog.records[-1].getMessage()
    assert "total=3.000s" in msg  # collect.* not in the base
    assert "denoise=3.000s (100.0%)" in msg


def test_logging_sink_noop_on_empty(caplog) -> None:
    sink = LoggingStatsSink(logging.getLogger("vrl.stats.empty"))
    with caplog.at_level(logging.INFO, logger="vrl.stats.empty"):
        sink.record(0, RolloutStats())
    assert caplog.records == []


def test_jsonl_stats_sink_writes_one_row_per_step(tmp_path) -> None:
    """Checks collect.* phases reach a machine-readable file, not just the log."""
    from vrl.rollouts.stats import JsonlStatsSink

    path = tmp_path / "nested" / "rollout_stats.jsonl"
    sink = JsonlStatsSink(path)

    first = RolloutStats()
    first.add_phases({"collect.wall": 2.0, "collect.generation_reward_overlap": 0.5})
    first.add_counter("collect.group_count", 2)
    sink.record(3, first)

    second = RolloutStats()
    second.add_phases({"collect.wall": 1.5})
    sink.record(4, second)

    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert [row["step"] for row in rows] == [3, 4]
    assert rows[0]["collect.wall"] == 2.0
    assert rows[0]["collect.generation_reward_overlap"] == 0.5
    assert rows[0]["collect.group_count"] == 2
    assert rows[1]["collect.wall"] == 1.5


def test_jsonl_stats_sink_skips_empty_stats(tmp_path) -> None:
    """Checks an empty record does not create a file or a blank row."""
    from vrl.rollouts.stats import JsonlStatsSink

    path = tmp_path / "rollout_stats.jsonl"
    JsonlStatsSink(path).record(0, RolloutStats())

    assert not path.exists()


def test_multi_stats_sink_fans_out_in_order() -> None:
    """Checks one record reaches every sink, so the log line survives the jsonl."""
    from vrl.rollouts.stats import MultiStatsSink

    seen: list[tuple[str, int]] = []

    class _Recorder:
        def __init__(self, name: str) -> None:
            self.name = name

        def record(self, step: int, stats: RolloutStats) -> None:
            del stats
            seen.append((self.name, step))

    MultiStatsSink(_Recorder("a"), _Recorder("b")).record(7, RolloutStats())

    assert seen == [("a", 7), ("b", 7)]
