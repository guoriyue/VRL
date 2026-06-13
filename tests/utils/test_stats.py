"""Tests for the per-request rollout stats accumulator and sinks."""

from __future__ import annotations

import logging

from vrl.utils.stats import LoggingStatsSink, RolloutStats


def test_add_phase_sums_on_repeat() -> None:
    s = RolloutStats()
    s.add_phase("denoise", 1.0)
    s.add_phase("denoise", 0.5)
    assert s.phase_seconds["denoise"] == 1.5


def test_merge_sums_phases_and_last_wins_reward() -> None:
    a = RolloutStats()
    a.add_phase("collect.engine_generate", 2.0)
    a.fold_reward_timing(inference_ms=10.0)
    b = RolloutStats()
    b.add_phase("collect.engine_generate", 3.0)
    b.fold_reward_timing(inference_ms=20.0)
    a.merge(b)
    assert a.phase_seconds["collect.engine_generate"] == 5.0
    # reward timings are call-level -> last non-None wins, not summed
    assert a.reward_inference_ms == 20.0


def test_fold_reward_timing_records_typed_fields() -> None:
    s = RolloutStats()
    s.fold_reward_timing(latency_ms=5.0, queue_wait_ms=1.0, inference_ms=4.0)
    assert s.reward_latency_ms == 5.0
    assert s.reward_queue_wait_ms == 1.0
    assert s.reward_inference_ms == 4.0


def test_as_phase_dict_surfaces_reward_as_seconds() -> None:
    s = RolloutStats()
    s.add_phase("denoise", 1.0)
    s.fold_reward_timing(inference_ms=2000.0)
    d = s.as_phase_dict()
    assert d["denoise"] == 1.0
    assert d["reward.inference_s"] == 2.0


def test_from_phase_dict_round_trips() -> None:
    s = RolloutStats.from_phase_dict({"a": 1.0, "b": 2.0})
    assert s.phase_seconds == {"a": 1.0, "b": 2.0}
    assert RolloutStats.from_phase_dict(None).phase_seconds == {}


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
