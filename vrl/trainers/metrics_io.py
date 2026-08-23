"""Stable CSV protocol for online training metrics."""

from __future__ import annotations

import csv
import logging
import os
from collections.abc import Sequence
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vrl.algorithms.types import TrainStepMetrics

logger = logging.getLogger(__name__)


def _csv_field(format_spec: str | None = None, *, phase_key: str | None = None) -> Any:
    """Declare one CSV column; ``phase_key`` names the ``phase_times`` entry it
    is populated from so ``from_step_metrics`` derives the mapping instead of
    hand-maintaining a field-name -> phases.get(...) twin per column."""

    metadata: dict[str, Any] = {}
    if format_spec is not None:
        metadata["csv_format"] = format_spec
    if phase_key is not None:
        metadata["phase_key"] = phase_key
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
    # from TrainStepMetrics.phase_times at this IO boundary; the phase_key
    # metadata IS the mapping (from_step_metrics iterates it).
    continuous_stale_versions: float = _csv_field(
        ".1f", phase_key="continuous.stale_policy_versions"
    )
    continuous_ready_groups_at_demand: float = _csv_field(
        ".1f", phase_key="continuous.ready_groups_at_demand"
    )
    continuous_queue_wait_s: float = _csv_field(".4f", phase_key="continuous.queue_wait_s")
    continuous_item_age_s: float = _csv_field(".4f", phase_key="continuous.item_age_s")
    continuous_lookahead_requested: float = _csv_field(
        ".1f", phase_key="continuous.lookahead_requested"
    )
    continuous_weight_sync_pause_s: float = _csv_field(
        ".4f", phase_key="continuous.weight_sync_pause_s"
    )
    continuous_producer_max_gap_s: float = _csv_field(
        ".4f", phase_key="continuous.producer_max_tick_gap_s"
    )
    continuous_producer_submitted: float = _csv_field(
        ".1f", phase_key="continuous.producer_submitted"
    )
    continuous_producer_completed: float = _csv_field(
        ".1f", phase_key="continuous.producer_completed"
    )
    continuous_producer_errors: float = _csv_field(".1f", phase_key="continuous.producer_errors")
    # 0 = draining weight-sync barrier; 1 = non-draining versioned slots.
    continuous_weight_sync_barrier_mode: float = _csv_field(
        ".1f", phase_key="continuous.weight_sync_barrier_mode"
    )
    # Stage-contract columns (SPRINT_continuous_stage_contracts_and_baseline).
    # Identity of the consumed iteration plus per-stage worst-observed
    # intervals (gauge reduction; busy totals stay in collect.*_wall phases)
    # and cumulative backpressure reason durations/entries.
    continuous_batch_id: float = _csv_field(".1f", phase_key="continuous.batch_id")
    continuous_generation_queue_wait_s: float = _csv_field(
        ".4f", phase_key="continuous.generation_queue_wait_s"
    )
    continuous_generation_service_s: float = _csv_field(
        ".4f", phase_key="continuous.generation_service_s"
    )
    continuous_reward_queue_wait_s: float = _csv_field(
        ".4f", phase_key="continuous.reward_queue_wait_s"
    )
    continuous_reward_service_s: float = _csv_field(".4f", phase_key="continuous.reward_service_s")
    continuous_backpressure_inflight_full_s: float = _csv_field(
        ".4f", phase_key="continuous.backpressure_inflight_full_s"
    )
    continuous_backpressure_inflight_full_count: float = _csv_field(
        ".1f", phase_key="continuous.backpressure_inflight_full_count"
    )
    continuous_backpressure_paused_s: float = _csv_field(
        ".4f", phase_key="continuous.backpressure_paused_for_weight_sync_s"
    )
    continuous_backpressure_paused_count: float = _csv_field(
        ".1f", phase_key="continuous.backpressure_paused_for_weight_sync_count"
    )
    continuous_backpressure_no_pending_s: float = _csv_field(
        ".4f", phase_key="continuous.backpressure_no_pending_slots_s"
    )
    continuous_backpressure_no_pending_count: float = _csv_field(
        ".1f", phase_key="continuous.backpressure_no_pending_slots_count"
    )
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

    @classmethod
    def from_step_metrics(
        cls,
        epoch: int,
        metrics: TrainStepMetrics,
        component_names: Sequence[str] = (),
    ) -> OnlineMetricRow:
        """Flatten one nested trainer metric result exactly once at the IO boundary."""

        phases = getattr(metrics, "phase_times", None) or {}
        phase_kwargs = {
            item.name: float(phases.get(item.metadata["phase_key"], 0.0))
            for item in fields(cls)
            if "phase_key" in item.metadata
        }
        names = tuple(component_names)
        _component_columns(names)
        current = getattr(metrics, "reward_components", None) or {}
        component_values = tuple(
            float(current[name]) if name in current else float("nan") for name in names
        )
        return cls(
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
            component_names=names,
            component_values=component_values,
            **phase_kwargs,
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


def prepare_metrics_csv(
    csv_path: str | Path,
    columns: Sequence[str],
    *,
    resume_at: tuple[str, int] | None,
) -> None:
    """Create a metrics CSV or align it atomically with a resume checkpoint.

    A checkpoint at position N cannot support metrics already written for N or
    later: those updates were not captured in the checkpoint and will be
    recomputed. Keeping them would create duplicate positions after resume.
    """

    path = Path(csv_path)
    if isinstance(columns, (str, bytes)):
        raise ValueError("metrics columns must be a sequence of column names")
    column_names = tuple(columns)
    if (
        not column_names
        or any(
            not isinstance(name, str) or not name or any(char in name for char in ",\r\n")
            for name in column_names
        )
        or len(column_names) != len(set(column_names))
    ):
        raise ValueError("metrics columns must be unique non-empty CSV-safe strings")
    normalized_header = ",".join(column_names) + "\n"
    if resume_at is None:
        path.write_text(normalized_header)
        return

    position_column, resume_position = resume_at
    if resume_position < 0:
        raise ValueError(f"metrics resume position must be >= 0, got {resume_position}")
    if not path.exists():
        logger.warning("Resume requested but metrics file does not exist; creating %s", path)
        path.write_text(normalized_header)
        return

    text = path.read_text(encoding="utf-8")
    complete_text = text if text.endswith("\n") else text.rpartition("\n")[0] + "\n"
    lines = complete_text.splitlines(keepends=True)
    existing_header = lines[0] if lines else ""
    if existing_header.rstrip("\r\n") != normalized_header.rstrip("\n"):
        raise ValueError(
            f"{path} was written by a different metrics schema; appending "
            "would silently misalign columns. Move the old file aside or "
            "start a fresh output_dir.",
        )

    if position_column not in column_names:
        raise ValueError(f"metrics CSV is missing resume column {position_column!r}: {path}")
    position_index = column_names.index(position_column)
    retained_lines: list[str] = []
    previous_position = -1
    truncated_rows = 0
    for line_number, line in enumerate(lines[1:], start=2):
        values = next(csv.reader([line]))
        if len(values) != len(column_names):
            raise ValueError(
                f"metrics CSV row {line_number} has {len(values)} columns; "
                f"expected {len(column_names)}: {path}",
            )
        raw_position = values[position_index]
        try:
            position = int(raw_position)
        except ValueError as exc:
            raise ValueError(
                f"metrics CSV row {line_number} has a non-integer "
                f"{position_column}: {raw_position!r}",
            ) from exc
        if position < 0 or position <= previous_position:
            raise ValueError(
                f"metrics CSV {position_column} must be strictly increasing "
                f"non-negative integers; row {line_number} has {position}",
            )
        previous_position = position
        if position < resume_position:
            retained_lines.append(line)
        else:
            truncated_rows += 1

    aligned_text = normalized_header + "".join(retained_lines)
    if aligned_text != text:
        temporary = path.with_suffix(f"{path.suffix}.tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(aligned_text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    if truncated_rows:
        logger.warning(
            "Discarded %d metrics row(s) at %s >= %d before checkpoint resume",
            truncated_rows,
            position_column,
            resume_position,
        )
    elif previous_position + 1 < resume_position:
        logger.warning(
            "Metrics end at %s=%d before checkpoint resume position %d",
            position_column,
            previous_position,
            resume_position,
        )


__all__ = [
    "OnlineMetricRow",
    "format_online_metric_row",
    "online_metric_columns",
    "prepare_metrics_csv",
]
