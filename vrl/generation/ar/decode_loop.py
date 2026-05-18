"""Scheduled AR decode loop used by AR family runtimes.

The loop owns token-level scheduling and row-wise state routing for the current
family runners. It has no public rollout API surface; generation callers only
see the family runtime's ``GenerationOutput``.
"""

from __future__ import annotations

import inspect
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
from transformers.cache_utils import Cache, DynamicCache

from vrl.generation.types import GenerationRequest, GenerationSampleRow


@dataclass(frozen=True, slots=True)
class ARSequenceKey:
    """Grouping key for token-level AR batching."""

    family: str
    task: str
    tokenizer_key: str
    dtype: str
    max_new_tokens: int


@dataclass(slots=True)
class ActiveSequence:
    """One in-flight AR image sequence inside the scheduled decode loop."""

    request_id: str
    sample_id: str
    family: str
    task: str
    tokenizer_key: str
    dtype: str
    max_new_tokens: int
    position: int = 0
    finished: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ValueError("ActiveSequence.request_id must be non-empty")
        if not self.sample_id:
            raise ValueError("ActiveSequence.sample_id must be non-empty")
        if self.max_new_tokens < 1:
            raise ValueError("ActiveSequence.max_new_tokens must be >= 1")
        if self.position < 0:
            raise ValueError("ActiveSequence.position must be >= 0")
        if self.position >= self.max_new_tokens:
            self.finished = True

    @property
    def key(self) -> ARSequenceKey:
        return ARSequenceKey(
            family=self.family,
            task=self.task,
            tokenizer_key=self.tokenizer_key,
            dtype=self.dtype,
            max_new_tokens=self.max_new_tokens,
        )

    @property
    def remaining_tokens(self) -> int:
        return max(0, self.max_new_tokens - self.position)

    def advance(self, steps: int = 1) -> None:
        if steps < 1:
            raise ValueError("steps must be >= 1")
        if self.finished:
            return
        self.position += steps
        if self.position >= self.max_new_tokens:
            self.position = self.max_new_tokens
            self.finished = True

    def mark_finished(self) -> None:
        self.position = min(self.position, self.max_new_tokens)
        self.finished = True


@dataclass(slots=True)
class TokenBatch:
    """One token-forward batch inside the scheduled decode loop."""

    key: ARSequenceKey
    sequences: list[ActiveSequence]

    def __post_init__(self) -> None:
        if not self.sequences:
            raise ValueError("TokenBatch.sequences must be non-empty")
        for sequence in self.sequences:
            if sequence.key != self.key:
                raise ValueError("TokenBatch sequences must share the same key")

    @property
    def request_ids(self) -> list[str]:
        return [sequence.request_id for sequence in self.sequences]

    @property
    def sample_ids(self) -> list[str]:
        return [sequence.sample_id for sequence in self.sequences]


class TokenScheduler:
    """Group active AR sequences into same-shape token batches."""

    def __init__(self, max_batch_size: int) -> None:
        if max_batch_size < 1:
            raise ValueError("max_batch_size must be >= 1")
        self.max_batch_size = max_batch_size
        self._pending: list[ActiveSequence] = []

    def __len__(self) -> int:
        return sum(not sequence.finished for sequence in self._pending)

    def add(self, sequence: ActiveSequence) -> None:
        if not sequence.finished:
            self._pending.append(sequence)

    def add_many(self, sequences: list[ActiveSequence]) -> None:
        for sequence in sequences:
            self.add(sequence)

    def pop_batch(self) -> TokenBatch | None:
        groups: dict[tuple[ARSequenceKey, int], list[ActiveSequence]] = {}
        ordered_keys: list[tuple[ARSequenceKey, int]] = []
        retained: list[ActiveSequence] = []

        for sequence in self._pending:
            if sequence.finished:
                continue
            group_key = (sequence.key, sequence.position)
            if group_key not in groups:
                groups[group_key] = []
                ordered_keys.append(group_key)
            groups[group_key].append(sequence)
            retained.append(sequence)

        self._pending = retained
        if not groups:
            return None

        key, _position = ordered_keys[0]
        selected_group = groups[ordered_keys[0]]
        selected = selected_group[: self.max_batch_size]
        selected_ids = {id(sequence) for sequence in selected}
        self._pending = [
            sequence for sequence in self._pending if id(sequence) not in selected_ids
        ]
        return TokenBatch(key=key, sequences=selected)

    def push_back_unfinished(self, batch: TokenBatch) -> None:
        for sequence in batch.sequences:
            if not sequence.finished:
                self._pending.append(sequence)


def ar_split_rows(value: Any, batch_size: int) -> list[Any]:
    """Split a batched AR cache/value into one-row values."""

    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if isinstance(value, Cache):
        return _split_hf_cache_rows(value, batch_size)
    return _split_plain_rows(value, batch_size)


@dataclass(slots=True)
class ARCacheRows:
    """Mutable per-row AR cache store for the scheduled decode loop."""

    rows: list[Any]
    owner: str = "ar_cache"

    def __post_init__(self) -> None:
        self.rows = list(self.rows)
        if not self.rows:
            raise ValueError(f"{self.owner} requires at least one cache row")

    @classmethod
    def from_batched(
        cls,
        value: Any,
        batch_size: int,
        *,
        owner: str = "ar_cache",
    ) -> ARCacheRows:
        """Create a row cache store from one batched cache value."""

        return cls(ar_split_rows(value, batch_size), owner=owner)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Any:
        return self.rows[index]

    def gather(self, indices: Sequence[int]) -> Any:
        """Return selected rows as one batched cache value."""

        return ar_concat_rows(self.select_rows(indices))

    def scatter(self, indices: Sequence[int], value: Any) -> None:
        """Overwrite selected rows from one batched cache value."""

        row_indices = self._validate_indices(indices)
        self.scatter_rows(row_indices, ar_split_rows(value, len(row_indices)))

    def select_rows(self, indices: Sequence[int]) -> list[Any]:
        """Return selected row cache objects without batching them."""

        row_indices = self._validate_indices(indices)
        return [self.rows[index] for index in row_indices]

    def scatter_rows(self, indices: Sequence[int], rows: Sequence[Any]) -> None:
        """Overwrite selected rows from already-split row cache objects."""

        row_indices = self._validate_indices(indices)
        row_values = list(rows)
        if len(row_values) != len(row_indices):
            raise ValueError(
                f"{self.owner} received {len(row_values)} rows for "
                f"{len(row_indices)} row indices",
            )
        for index, row_value in zip(row_indices, row_values, strict=True):
            self.rows[index] = row_value

    def _validate_indices(self, indices: Sequence[int]) -> list[int]:
        row_indices = [int(index) for index in indices]
        if not row_indices:
            raise ValueError(f"{self.owner} requires at least one row index")
        size = len(self.rows)
        invalid = [index for index in row_indices if index < 0 or index >= size]
        if invalid:
            raise IndexError(
                f"{self.owner} row indices out of range for {size} rows: {invalid}",
            )
        return row_indices


def ar_concat_rows(values: Sequence[Any]) -> Any:
    """Concatenate one-row AR cache/value objects along batch dim 0."""

    if not values:
        raise ValueError("values must be non-empty")
    first = values[0]
    if isinstance(first, Cache):
        return _concat_hf_cache_rows(values)
    if any(isinstance(value, Cache) for value in values[1:]):
        raise TypeError("cannot concatenate mixed HF cache and non-cache rows")
    return _concat_plain_rows(values)


@dataclass(slots=True)
class ARStepResult:
    """One scheduled AR token step."""

    sequence_ids: list[str]
    positions: list[int]
    token: Any
    log_prob: Any
    replay_extras: dict[str, Any] = field(default_factory=dict)
    debug_counters: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ARTokenLoopInit:
    """Model-provided payload for the scheduled AR decode loop."""

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
    """Generation-owned KV/row-state envelope scheduled by the decode loop."""

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


@dataclass(slots=True)
class ARDecodeResult:
    """Result of one scheduled AR decode loop."""

    finalized: Any
    scheduler_batches: int
    engine_counters: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ARDecodeLoop:
    """Family-neutral scheduled AR decode loop driver."""

    request: GenerationRequest
    sample_rows: Sequence[GenerationSampleRow]
    runner: Any
    max_new_tokens: int
    tokenizer_key: str
    dtype: str
    scheduler_batch_size: int | None = None
    init_args: Sequence[Any] = ()
    init_kwargs: Mapping[str, Any] | None = None
    step_kwargs: Mapping[str, Any] | None = None

    def run(self) -> ARDecodeResult:
        rows = list(self.sample_rows)
        if not rows:
            raise ValueError("sample_rows must be non-empty")
        if self.max_new_tokens < 1:
            raise ValueError("max_new_tokens must be >= 1")

        self._require_hooks(("init_ar", "step_ar", "finalize_ar"))

        init_state = self._call_with_supported_kwargs(
            self.runner.init_ar,
            *self.init_args,
            **dict(self.init_kwargs or {}),
        )
        if not isinstance(init_state, ARTokenLoopInit):
            raise TypeError(
                "AR model runner init_ar must return ARTokenLoopInit; "
                f"got {type(init_state).__name__}",
            )
        envelope = ARTokenLoopEnvelope.from_init(init_state, batch_size=len(rows))
        state = envelope.state
        sequences = self._build_sequences(rows)
        scheduler = TokenScheduler(max_batch_size=self._max_batch_size(sequences))
        scheduler.add_many(sequences)

        scheduler_batches = 0
        decode_tokens = 0
        engine_counters: dict[str, Any] = {
            "ar_scheduled_decode_loop_enabled": True,
            "ar_scheduler_batch_size": self.scheduler_batch_size,
        }
        call_step_kwargs = dict(self.step_kwargs or {})

        while True:
            batch = scheduler.pop_batch()
            if batch is None:
                break
            scheduler_batches += 1
            decode_tokens += len(batch.sequences)
            step_batch = envelope.build_step_batch(batch.sequences)
            step_output = self._call_with_supported_kwargs(
                self.runner.step_ar,
                state,
                step_batch,
                **call_step_kwargs,
            )
            if not isinstance(step_output, ARStepOutput):
                raise TypeError(
                    "AR model runner step_ar must return ARStepOutput; "
                    f"got {type(step_output).__name__}",
                )
            envelope.apply_step_output(step_batch, step_output)
            result = step_output.result
            debug_counters = getattr(result, "debug_counters", None)
            if debug_counters:
                engine_counters.update(dict(debug_counters))
            for sequence in batch.sequences:
                sequence.advance()
            scheduler.push_back_unfinished(batch)

        finalized = self.runner.finalize_ar(state)
        engine_counters.setdefault("ar_decode_tokens", decode_tokens)
        engine_counters["ar_scheduler_batches"] = scheduler_batches
        return ARDecodeResult(
            finalized=finalized,
            scheduler_batches=scheduler_batches,
            engine_counters=engine_counters,
        )

    def _max_batch_size(self, sequences: Sequence[ActiveSequence]) -> int:
        max_batch_size = int(self.scheduler_batch_size or len(sequences))
        return max(1, max_batch_size)

    def _require_hooks(self, names: Sequence[str]) -> None:
        missing = [name for name in names if not hasattr(self.runner, name)]
        if missing:
            raise TypeError(
                "AR decode loop requires AR model runner hooks: "
                + ", ".join(missing),
            )

    def _build_sequences(
        self,
        sample_rows: Sequence[GenerationSampleRow],
    ) -> list[ActiveSequence]:
        return [
            ActiveSequence(
                request_id=self.request.request_id,
                sample_id=row.sample_id,
                family=self.request.family,
                task=self.request.task,
                tokenizer_key=self.tokenizer_key,
                dtype=self.dtype,
                max_new_tokens=self.max_new_tokens,
                metadata={
                    **dict(row.metadata),
                    "row_index": row_index,
                    "prompt_index": row.prompt_index,
                    "sample_index": row.sample_index,
                },
            )
            for row_index, row in enumerate(sample_rows)
        ]

    def _call_with_supported_kwargs(self, fn: Any, *args: Any, **kwargs: Any) -> Any:
        try:
            signature = inspect.signature(fn)
        except (TypeError, ValueError):
            return fn(*args, **kwargs)
        parameters = signature.parameters
        if any(
            param.kind is inspect.Parameter.VAR_KEYWORD
            for param in parameters.values()
        ):
            return fn(*args, **kwargs)
        supported = {key: value for key, value in kwargs.items() if key in parameters}
        return fn(*args, **supported)


def _split_plain_rows(value: Any, batch_size: int) -> list[Any]:
    if isinstance(value, torch.Tensor):
        if value.shape[0] != batch_size:
            raise ValueError(
                f"cannot split tensor with batch={value.shape[0]} into "
                f"{batch_size} rows",
            )
        return [value[row : row + 1] for row in range(batch_size)]
    if isinstance(value, Mapping):
        split_items = {
            key: ar_split_rows(inner, batch_size) for key, inner in value.items()
        }
        return [
            type(value)((key, parts[row]) for key, parts in split_items.items())
            for row in range(batch_size)
        ]
    if isinstance(value, tuple):
        split_items = [ar_split_rows(inner, batch_size) for inner in value]
        return [tuple(parts[row] for parts in split_items) for row in range(batch_size)]
    if isinstance(value, list):
        split_items = [ar_split_rows(inner, batch_size) for inner in value]
        return [[parts[row] for parts in split_items] for row in range(batch_size)]
    return [value for _ in range(batch_size)]


def _concat_plain_rows(values: Sequence[Any]) -> Any:
    first = values[0]
    if isinstance(first, torch.Tensor):
        return torch.cat(list(values), dim=0)
    if isinstance(first, Mapping):
        return type(first)(
            (key, ar_concat_rows([value[key] for value in values])) for key in first
        )
    if isinstance(first, tuple):
        return tuple(
            ar_concat_rows([value[index] for value in values])
            for index in range(len(first))
        )
    if isinstance(first, list):
        return [
            ar_concat_rows([value[index] for value in values])
            for index in range(len(first))
        ]
    if any(value != first for value in values[1:]):
        raise ValueError("cannot concatenate non-tensor AR values that differ")
    return first


def _split_hf_cache_rows(value: Cache, batch_size: int) -> list[Cache]:
    if not isinstance(value, DynamicCache):
        raise TypeError(
            "AR KV row scheduling currently supports transformers DynamicCache; "
            f"got {type(value).__name__}",
        )
    legacy_layers = value.to_legacy_cache()
    rows: list[DynamicCache] = []
    for row in range(batch_size):
        row_layers = tuple(
            (key[row : row + 1], val[row : row + 1])
            for key, val in legacy_layers
        )
        rows.append(DynamicCache.from_legacy_cache(row_layers))
    return rows


def _concat_hf_cache_rows(values: Sequence[Any]) -> DynamicCache:
    if not all(isinstance(value, DynamicCache) for value in values):
        got = ", ".join(type(value).__name__ for value in values)
        raise TypeError(f"cannot concatenate mixed HF cache row types: {got}")
    legacy_rows = [value.to_legacy_cache() for value in values]
    layer_count = len(legacy_rows[0])
    if any(len(row) != layer_count for row in legacy_rows[1:]):
        raise ValueError("cannot concatenate DynamicCache rows with different layer counts")
    layers = tuple(
        (
            torch.cat([row[layer_idx][0] for row in legacy_rows], dim=0),
            torch.cat([row[layer_idx][1] for row in legacy_rows], dim=0),
        )
        for layer_idx in range(layer_count)
    )
    return DynamicCache.from_legacy_cache(layers)


def _row_indices(sequences: Sequence[ActiveSequence], *, size: int) -> list[int]:
    row_indices = [int(sequence.metadata.get("row_index", -1)) for sequence in sequences]
    if not row_indices:
        raise ValueError("scheduled AR step requires at least one row")
    invalid = [index for index in row_indices if index < 0 or index >= size]
    if invalid:
        raise ValueError(f"scheduled AR row indices out of range for {size} rows: {invalid}")
    return row_indices


__all__ = [
    "ARCacheRows",
    "ARDecodeLoop",
    "ARDecodeResult",
    "ARSequenceKey",
    "ARStepBatch",
    "ARStepOutput",
    "ARStepResult",
    "ARTokenLoopEnvelope",
    "ARTokenLoopInit",
    "ActiveSequence",
    "TokenBatch",
    "TokenScheduler",
    "ar_concat_rows",
    "ar_split_rows",
]
