"""Stable online metrics CSV protocol tests."""

from __future__ import annotations

import math

import pytest

from vrl.algorithms.logprob_mismatch import LogprobMismatchStats
from vrl.algorithms.types import (
    InitialReplayStats,
    PolicyUpdateStats,
    TrainStepMetrics,
)
from vrl.trainers.metrics_io import (
    build_online_metric_row,
    format_online_metric_row,
    online_metric_columns,
    prepare_metrics_csv,
)

_EXPECTED_FIXED_COLUMNS = (
    "epoch",
    "loss",
    "policy_loss",
    "sft_loss",
    "kl_penalty",
    "weighted_kl_loss",
    "reward_mean",
    "reward_std",
    "clip_fraction",
    "active_clip_fraction",
    "pre_update_clip_fraction",
    "pre_update_active_clip_fraction",
    "pre_update_logprob_abs_diff_max",
    "tis_clip_fraction",
    "rs_seq_masked_fraction",
    "approx_kl",
    "logprob_abs_diff_mean",
    "logprob_abs_diff_max",
    "ratio_abs_dev_mean",
    "ratio_abs_dev_max",
    "mismatch_kl",
    "mismatch_k3_kl",
    "advantage_mean",
    "grad_norm",
    "adv_saturation",
    "adv_zero_rate",
    "group_size",
    "trained_prompt_num",
    "continuous_stale_versions",
    "continuous_ready_groups_at_demand",
    "continuous_queue_wait_s",
    "continuous_item_age_s",
    "continuous_lookahead_requested",
    "continuous_weight_sync_pause_s",
    "continuous_producer_max_gap_s",
    "continuous_producer_submitted",
    "continuous_producer_completed",
    "continuous_producer_errors",
    "continuous_weight_sync_barrier_mode",
)


def test_online_metric_columns_preserve_the_existing_csv_contract() -> None:
    columns = online_metric_columns(("aesthetic", "pickscore"))

    assert columns == (*_EXPECTED_FIXED_COLUMNS, "r_aesthetic", "r_pickscore")


def test_online_metric_row_uses_the_same_order_and_formats_as_header() -> None:
    metrics = TrainStepMetrics(
        loss=1.1,
        policy_loss=2.1,
        sft_loss=3.1,
        kl_penalty=4.1,
        weighted_kl_loss=5.1,
        reward_mean=6.1,
        reward_std=7.1,
        update=PolicyUpdateStats(
            clip_fraction=8.1,
            active_clip_fraction=9.1,
            tis_clip_fraction=13.1,
            rs_seq_masked_fraction=14.1,
            approx_kl=15.1,
        ),
        initial_replay=InitialReplayStats(
            clip_fraction=10.1,
            active_clip_fraction=11.1,
            logprob_abs_diff_max=12.1,
        ),
        logprob_mismatch=LogprobMismatchStats(
            logprob_abs_diff_mean=16.1,
            logprob_abs_diff_max=17.1,
            ratio_abs_dev_mean=18.1,
            ratio_abs_dev_max=19.1,
            mismatch_kl=20.1,
            mismatch_k3_kl=21.1,
        ),
        advantage_mean=22.1,
        grad_norm=23.1,
        adv_saturation=24.1,
        adv_zero_rate=25.1,
        group_size=26.1,
        trained_prompt_num=27,
        reward_components={"aesthetic": 40.1},
        phase_times={
            "continuous.stale_policy_versions": 28.1,
            "continuous.ready_groups_at_demand": 30.1,
            "continuous.queue_wait_s": 31.1,
            "continuous.item_age_s": 32.1,
            "continuous.lookahead_requested": 33.1,
            "continuous.weight_sync_pause_s": 34.1,
            "continuous.producer_max_tick_gap_s": 35.1,
            "continuous.producer_submitted": 36.1,
            "continuous.producer_completed": 37.1,
            "continuous.producer_errors": 38.1,
            "continuous.weight_sync_barrier_mode": 39.1,
        },
    )
    row = build_online_metric_row(
        0,
        metrics,
        ("aesthetic", "pickscore"),
    )
    values = format_online_metric_row(row).strip().split(",")

    assert values[:-1] == [
        "0",
        "1.100000",
        "2.100000",
        "3.100000",
        "4.100000",
        "5.100000",
        "6.1000",
        "7.1000",
        "8.1000",
        "9.1000",
        "10.1000",
        "11.1000",
        "12.100000",
        "13.1000",
        "14.1000",
        "15.100000",
        "16.100000",
        "17.100000",
        "18.100000",
        "19.100000",
        "20.100000",
        "21.100000",
        "22.100000",
        "23.100000",
        "24.1000",
        "25.1000",
        "26.10",
        "27",
        "28.1",
        "30.1",
        "31.1000",
        "32.1000",
        "33.1",
        "34.1000",
        "35.1000",
        "36.1",
        "37.1",
        "38.1",
        "39.1",
        "40.1000",
    ]
    assert math.isnan(float(values[-1]))


def test_continuous_phase_metrics_are_mapped_at_the_io_boundary() -> None:
    metrics = TrainStepMetrics(
        trained_prompt_num=1,
        phase_times={
            "continuous.ready_groups_at_demand": 3.0,
            "continuous.lookahead_requested": 1.0,
            "continuous.producer_errors": 2.0,
            "continuous.weight_sync_barrier_mode": 1.0,
        },
    )

    row = build_online_metric_row(0, metrics)

    assert row.continuous_ready_groups_at_demand == pytest.approx(3.0)
    assert row.continuous_lookahead_requested == pytest.approx(1.0)
    assert row.continuous_producer_errors == pytest.approx(2.0)
    assert row.continuous_weight_sync_barrier_mode == pytest.approx(1.0)


def test_strict_online_metric_row_defaults_all_continuous_columns_to_zero() -> None:
    row = build_online_metric_row(0, TrainStepMetrics())
    values = dict(
        zip(
            online_metric_columns(),
            format_online_metric_row(row).strip().split(","),
            strict=True,
        ),
    )

    continuous_values = {
        name: float(value) for name, value in values.items() if name.startswith("continuous_")
    }
    assert continuous_values
    assert set(continuous_values.values()) == {0.0}


def test_online_metric_row_preserves_non_finite_values_for_health_consumers() -> None:
    row = build_online_metric_row(
        0,
        TrainStepMetrics(loss=float("nan"), reward_mean=float("inf")),
    )
    values = dict(
        zip(
            online_metric_columns(),
            format_online_metric_row(row).strip().split(","),
            strict=True,
        ),
    )

    assert math.isnan(float(values["loss"]))
    assert math.isinf(float(values["reward_mean"]))


@pytest.mark.parametrize(
    "component_names",
    [
        "aesthetic",
        b"aesthetic",
        ("",),
        ("aesthetic", "aesthetic"),
        ("bad,name",),
        ("bad\nname",),
    ],
)
def test_online_metric_columns_reject_invalid_dynamic_components(
    component_names: object,
) -> None:
    with pytest.raises(ValueError, match="component"):
        online_metric_columns(component_names)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("epoch", "trained_prompt_num", "field_name"),
    [
        (-1, 1, "epoch"),
        (True, 1, "epoch"),
        (0, -1, "trained_prompt_num"),
        (0, 1.5, "trained_prompt_num"),
    ],
)
def test_online_metric_row_rejects_invalid_integer_fields(
    epoch: object,
    trained_prompt_num: object,
    field_name: str,
) -> None:
    metrics = TrainStepMetrics()
    metrics.trained_prompt_num = trained_prompt_num  # type: ignore[assignment]

    with pytest.raises(ValueError, match=field_name):
        build_online_metric_row(epoch, metrics)  # type: ignore[arg-type]


def test_prepare_metrics_csv_rejects_resume_across_schema_change(tmp_path) -> None:
    """Appending new-schema rows under an old header silently shifts columns."""

    path = tmp_path / "metrics.csv"
    columns = ("epoch", "loss", "approx_kl")

    prepare_metrics_csv(path, columns, resume_at=None)
    prepare_metrics_csv(path, columns, resume_at=("epoch", 0))

    with pytest.raises(ValueError, match="different metrics schema"):
        prepare_metrics_csv(
            path,
            ("epoch", "loss", "active_clip_fraction", "approx_kl"),
            resume_at=("epoch", 0),
        )


@pytest.mark.parametrize(
    "columns",
    [
        "epoch",
        b"epoch",
        (),
        ("epoch", "epoch"),
        ("", "loss"),
        ("bad,column", "loss"),
        ("bad\ncolumn", "loss"),
        (1, "loss"),
    ],
)
def test_prepare_metrics_csv_rejects_invalid_columns(tmp_path, columns) -> None:
    with pytest.raises(ValueError, match="columns"):
        prepare_metrics_csv(tmp_path / "metrics.csv", columns, resume_at=None)


def test_prepare_metrics_csv_discards_rows_not_covered_by_checkpoint(tmp_path) -> None:
    path = tmp_path / "metrics.csv"
    columns = ("epoch", "loss")
    header = "epoch,loss\n"
    path.write_text(header + "38,1.0\n39,0.9\n40,0.8\n41,0.7\n42,0")

    prepare_metrics_csv(path, columns, resume_at=("epoch", 40))

    assert path.read_text() == header + "38,1.0\n39,0.9\n"


def test_prepare_metrics_csv_rejects_wrong_row_width(tmp_path) -> None:
    path = tmp_path / "metrics.csv"
    path.write_text("epoch,loss\n0\n")

    with pytest.raises(ValueError, match="has 1 columns; expected 2"):
        prepare_metrics_csv(path, ("epoch", "loss"), resume_at=("epoch", 1))


@pytest.mark.parametrize(
    "rows",
    [
        "0,1.0\n0,0.9\n",
        "1,1.0\n0,0.9\n",
        "0.5,1.0\n",
        "-1,1.0\n",
    ],
)
def test_prepare_metrics_csv_rejects_invalid_resume_positions(tmp_path, rows) -> None:
    path = tmp_path / "metrics.csv"
    columns = ("epoch", "loss")
    header = "epoch,loss\n"
    path.write_text(header + rows)

    with pytest.raises(ValueError, match=r"integer|strictly increasing"):
        prepare_metrics_csv(path, columns, resume_at=("epoch", 2))
