"""Engine-owned envelope for AR token-loop scheduling."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from vrl.generation.ar.token_loop.row_cache import ARCacheRows
from vrl.generation.ar.token_loop.sequence import ActiveSequence


@dataclass(slots=True)
class ARStepResult:
    """One scheduled AR token step.

    ``sequence_ids`` and every tensor-like value must follow the same order as
    the input active sequence list. ``replay_extras`` is reserved for per-step
    tensors needed by training replay; for NextStep-1 it must include
    ``saved_noise``.
    """

    sequence_ids: list[str]
    positions: list[int]
    token: Any
    log_prob: Any
    replay_extras: dict[str, Any] = field(default_factory=dict)
    debug_counters: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ARTokenLoopInit:
    """Model-provided payload that the engine turns into a scheduled envelope."""

    state: Any
    cache_lanes: Mapping[str, Any] = field(default_factory=dict)
    row_lanes: Mapping[str, Any] = field(default_factory=dict)
    cache_lane_owners: Mapping[str, str] = field(default_factory=dict)
    row_lane_owners: Mapping[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class ARStepBatch:
    """One scheduled token step built from an engine-owned envelope."""

    sequences: list[ActiveSequence]
    row_indices: list[int]
    positions: list[int]
    position: int
    cache_lanes: dict[str, Any]
    row_lanes: dict[str, Any]

    @property
    def sequence_ids(self) -> list[str]:
        return [str(sequence.sample_id) for sequence in self.sequences]


@dataclass(slots=True)
class ARStepOutput:
    """Model output for one scheduled AR token step."""

    result: ARStepResult
    updated_cache_lanes: Mapping[str, Any] = field(default_factory=dict)
    updated_row_lanes: Mapping[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ARTokenLoopEnvelope:
    """Engine-owned KV/row-state envelope scheduled by the AR token loop."""

    state: Any
    cache_lanes: dict[str, ARCacheRows] = field(default_factory=dict)
    row_lanes: dict[str, ARCacheRows] = field(default_factory=dict)

    @classmethod
    def from_init(
        cls,
        init: ARTokenLoopInit,
        *,
        batch_size: int,
    ) -> ARTokenLoopEnvelope:
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        return cls(
            state=init.state,
            cache_lanes={
                name: ARCacheRows.from_batched(
                    value,
                    batch_size,
                    owner=init.cache_lane_owners.get(name, f"ar.cache_lanes.{name}"),
                )
                for name, value in init.cache_lanes.items()
            },
            row_lanes={
                name: ARCacheRows.from_batched(
                    value,
                    batch_size,
                    owner=init.row_lane_owners.get(name, f"ar.row_lanes.{name}"),
                )
                for name, value in init.row_lanes.items()
            },
        )

    def build_step_batch(self, sequences: Sequence[ActiveSequence]) -> ARStepBatch:
        row_indices = _row_indices(sequences, size=self.row_count)
        positions = [int(sequence.position) for sequence in sequences]
        if len(set(positions)) != 1:
            raise ValueError("scheduled AR sequences must share one token position")
        return ARStepBatch(
            sequences=list(sequences),
            row_indices=row_indices,
            positions=positions,
            position=positions[0],
            cache_lanes={
                name: lane.gather(row_indices)
                for name, lane in self.cache_lanes.items()
            },
            row_lanes={
                name: lane.gather(row_indices)
                for name, lane in self.row_lanes.items()
            },
        )

    def apply_step_output(self, batch: ARStepBatch, output: ARStepOutput) -> None:
        for name, value in output.updated_cache_lanes.items():
            self._require_cache_lane(name).scatter(batch.row_indices, value)
        for name, value in output.updated_row_lanes.items():
            self._require_row_lane(name).scatter(batch.row_indices, value)

    @property
    def row_count(self) -> int:
        for lanes in (self.cache_lanes, self.row_lanes):
            for lane in lanes.values():
                return len(lane)
        raise ValueError("ARTokenLoopEnvelope requires at least one row/cache lane")

    def _require_cache_lane(self, name: str) -> ARCacheRows:
        try:
            return self.cache_lanes[name]
        except KeyError as exc:
            raise KeyError(f"unknown AR cache lane: {name!r}") from exc

    def _require_row_lane(self, name: str) -> ARCacheRows:
        try:
            return self.row_lanes[name]
        except KeyError as exc:
            raise KeyError(f"unknown AR row lane: {name!r}") from exc


def _row_indices(sequences: Sequence[ActiveSequence], *, size: int) -> list[int]:
    row_indices = [int(sequence.metadata.get("row_index", -1)) for sequence in sequences]
    if not row_indices:
        raise ValueError("scheduled AR step requires at least one row")
    invalid = [index for index in row_indices if index < 0 or index >= size]
    if invalid:
        raise ValueError(f"scheduled AR row indices out of range for {size} rows: {invalid}")
    return row_indices


__all__ = [
    "ARStepBatch",
    "ARStepOutput",
    "ARStepResult",
    "ARTokenLoopEnvelope",
    "ARTokenLoopInit",
]
