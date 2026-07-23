"""Stable CSV protocol for online training metrics."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, fields
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vrl.algorithms.types import TrainStepMetrics


def _csv_field(format_spec: str | None = None) -> Any:
    metadata = {} if format_spec is None else {"csv_format": format_spec}
    return field(metadata=metadata)


@dataclass(frozen=True, slots=True)
class OnlineMetricRow:
    """One fixed-schema online metrics row before CSV serialization."""

    epoch: int = _csv_field("d")
    loss: float = _csv_field()
    policy_loss: float = _csv_field()
    sft_loss: float = _csv_field()
    kl_penalty: float = _csv_field()
    weighted_kl_loss: float = _csv_field()
    reward_mean: float = _csv_field(".4f")
    reward_std: float = _csv_field(".4f")
    clip_fraction: float = _csv_field(".4f")
    active_clip_fraction: float = _csv_field(".4f")
    pre_update_clip_fraction: float = _csv_field(".4f")
    pre_update_active_clip_fraction: float = _csv_field(".4f")
    pre_update_logprob_abs_diff_max: float = _csv_field()
    tis_clip_fraction: float = _csv_field(".4f")
    rs_seq_masked_fraction: float = _csv_field(".4f")
    approx_kl: float = _csv_field()
    logprob_abs_diff_mean: float = _csv_field()
    logprob_abs_diff_max: float = _csv_field()
    ratio_abs_dev_mean: float = _csv_field()
    ratio_abs_dev_max: float = _csv_field()
    mismatch_kl: float = _csv_field()
    mismatch_k3_kl: float = _csv_field()
    advantage_mean: float = _csv_field()
    grad_norm: float = _csv_field()
    adv_saturation: float = _csv_field(".4f")
    adv_zero_rate: float = _csv_field(".4f")
    group_size: float = _csv_field(".2f")
    trained_prompt_num: int = _csv_field("d")
    # Strict on-policy runs leave these at zero. Continuous runs populate them
    # from TrainStepMetrics.phase_times at this IO boundary.
    continuous_stale_versions: float = _csv_field(".1f")
    continuous_ready_groups: float = _csv_field(".1f")
    continuous_ready_groups_at_demand: float = _csv_field(".1f")
    continuous_queue_wait_s: float = _csv_field(".4f")
    continuous_item_age_s: float = _csv_field(".4f")
    continuous_lookahead_requested: float = _csv_field(".1f")
    continuous_weight_sync_pause_s: float = _csv_field(".4f")
    continuous_producer_max_gap_s: float = _csv_field(".4f")
    continuous_producer_submitted: float = _csv_field(".1f")
    continuous_producer_completed: float = _csv_field(".1f")
    continuous_producer_errors: float = _csv_field(".1f")
    # 0 = draining weight-sync barrier; 1 = non-draining versioned slots.
    continuous_weight_sync_barrier_mode: float = _csv_field(".1f")
    component_names: tuple[str, ...] = field(
        default=(),
        metadata={"csv_extension": True},
    )
    component_values: tuple[float, ...] = field(
        default=(),
        metadata={"csv_extension": True},
    )

    def __post_init__(self) -> None:
        for name in ("epoch", "trained_prompt_num"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"online metric {name} must be a non-negative integer")
        _component_columns(self.component_names)
        if len(self.component_values) != len(self.component_names):
            raise ValueError(
                "online metric component names/values length mismatch",
            )


def build_online_metric_row(
    epoch: int,
    metrics: TrainStepMetrics,
    component_names: Sequence[str] = (),
) -> OnlineMetricRow:
    """Flatten one nested trainer metric result exactly once at the IO boundary."""

    phases = getattr(metrics, "phase_times", None) or {}
    names = tuple(component_names)
    _component_columns(names)
    current = getattr(metrics, "reward_components", None) or {}
    component_values = tuple(
        float(current[name]) if name in current else float("nan") for name in names
    )
    return OnlineMetricRow(
        epoch=epoch,
        loss=metrics.loss,
        policy_loss=metrics.policy_loss,
        sft_loss=metrics.sft_loss,
        kl_penalty=metrics.kl_penalty,
        weighted_kl_loss=metrics.weighted_kl_loss,
        reward_mean=metrics.reward_mean,
        reward_std=metrics.reward_std,
        clip_fraction=metrics.update.clip_fraction,
        active_clip_fraction=metrics.update.active_clip_fraction,
        pre_update_clip_fraction=metrics.initial_replay.clip_fraction,
        pre_update_active_clip_fraction=metrics.initial_replay.active_clip_fraction,
        pre_update_logprob_abs_diff_max=metrics.initial_replay.logprob_abs_diff_max,
        tis_clip_fraction=metrics.update.tis_clip_fraction,
        rs_seq_masked_fraction=metrics.update.rs_seq_masked_fraction,
        approx_kl=metrics.update.approx_kl,
        logprob_abs_diff_mean=metrics.logprob_mismatch.logprob_abs_diff_mean,
        logprob_abs_diff_max=metrics.logprob_mismatch.logprob_abs_diff_max,
        ratio_abs_dev_mean=metrics.logprob_mismatch.ratio_abs_dev_mean,
        ratio_abs_dev_max=metrics.logprob_mismatch.ratio_abs_dev_max,
        mismatch_kl=metrics.logprob_mismatch.mismatch_kl,
        mismatch_k3_kl=metrics.logprob_mismatch.mismatch_k3_kl,
        advantage_mean=metrics.advantage_mean,
        grad_norm=metrics.grad_norm,
        adv_saturation=metrics.adv_saturation,
        adv_zero_rate=metrics.adv_zero_rate,
        group_size=metrics.group_size,
        trained_prompt_num=metrics.trained_prompt_num,
        continuous_stale_versions=phases.get(
            "continuous.stale_policy_versions",
            0.0,
        ),
        continuous_ready_groups=phases.get(
            "continuous.queue_ready_groups",
            0.0,
        ),
        continuous_ready_groups_at_demand=phases.get(
            "continuous.ready_groups_at_demand",
            0.0,
        ),
        continuous_queue_wait_s=phases.get("continuous.queue_wait_s", 0.0),
        continuous_item_age_s=phases.get("continuous.item_age_s", 0.0),
        continuous_lookahead_requested=phases.get(
            "continuous.lookahead_requested",
            0.0,
        ),
        continuous_weight_sync_pause_s=phases.get(
            "continuous.weight_sync_pause_s",
            0.0,
        ),
        continuous_producer_max_gap_s=phases.get(
            "continuous.producer_max_tick_gap_s",
            0.0,
        ),
        continuous_producer_submitted=phases.get(
            "continuous.producer_submitted",
            0.0,
        ),
        continuous_producer_completed=phases.get(
            "continuous.producer_completed",
            0.0,
        ),
        continuous_producer_errors=phases.get(
            "continuous.producer_errors",
            0.0,
        ),
        continuous_weight_sync_barrier_mode=phases.get(
            "continuous.weight_sync_barrier_mode",
            0.0,
        ),
        component_names=names,
        component_values=component_values,
    )


def _fixed_fields() -> tuple[Any, ...]:
    return tuple(
        item for item in fields(OnlineMetricRow) if not item.metadata.get("csv_extension", False)
    )


def _fixed_columns() -> tuple[str, ...]:
    return tuple(item.name for item in _fixed_fields())


def _component_columns(component_names: Sequence[str]) -> tuple[str, ...]:
    if isinstance(component_names, (str, bytes)):
        raise ValueError("online metric component names must be a sequence of names")
    columns: list[str] = []
    for name in component_names:
        if not isinstance(name, str) or not name or any(char in name for char in ",\r\n"):
            raise ValueError(
                "online metric component names must be non-empty CSV-safe strings",
            )
        columns.append(f"r_{name}")
    if len(columns) != len(set(columns)):
        raise ValueError("online metric component names must be unique")
    collision = sorted(set(columns) & set(_fixed_columns()))
    if collision:
        raise ValueError(
            "online metric component columns collide with fixed columns: " + ", ".join(collision),
        )
    return tuple(columns)


def online_metric_columns(component_names: Sequence[str] = ()) -> tuple[str, ...]:
    """Return the frozen CSV column order for one run."""

    return (*_fixed_columns(), *_component_columns(component_names))


def format_online_metric_row(
    row: OnlineMetricRow,
) -> str:
    """Serialize a row in the same field-derived order as its header."""

    fixed_values = [
        format(getattr(row, item.name), str(item.metadata.get("csv_format", ".6f")))
        for item in _fixed_fields()
    ]
    component_values = [format(value, ".4f") for value in row.component_values]
    values = [*fixed_values, *component_values]
    return ",".join(values) + "\n"


__all__ = [
    "OnlineMetricRow",
    "build_online_metric_row",
    "format_online_metric_row",
    "online_metric_columns",
]
