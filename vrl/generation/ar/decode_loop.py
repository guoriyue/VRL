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

from vrl.generation.types import GenerationRequest, GenerationSampleRow
from vrl.nn.layers.attention.cache_rows import ARCacheRows, ar_concat_rows, ar_split_rows


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


@dataclass(slots=True)
class ARStepResult:
    """One scheduled AR token step."""

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

        init_state = call_with_supported_kwargs(
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
            step_output = call_with_supported_kwargs(
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

def call_with_supported_kwargs(fn: Any, *args: Any, **kwargs: Any) -> Any:
    """Call hooks while dropping optional kwargs unsupported by older signatures."""

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
    "call_with_supported_kwargs",
]
