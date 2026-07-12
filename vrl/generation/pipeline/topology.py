"""Physical pipeline stage topology.

This module owns runtime stage topology and routing. The chunk planner has no
stage model; physical pipeline stages remain a separate actor/placement boundary.

Field vocabulary is ours alone: every accepted key maps onto implemented
behavior, and unknown keys fail construction. Deliberate annotations belong
under ``metadata``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class PipelineStageRuntimePolicy:
    """Scheduler and resource intent for one physical pipeline stage."""

    batch_size: int | None = None
    max_inflight: int = 1
    max_batch_wait_ms: int = 0
    max_batch_cost: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.batch_size is not None and self.batch_size < 1:
            raise ValueError("stage batch_size must be >= 1 when set")
        if self.max_inflight < 1:
            raise ValueError("stage max_inflight must be >= 1")
        if self.max_batch_wait_ms < 0:
            raise ValueError("stage max_batch_wait_ms must be >= 0")
        if self.max_batch_cost is not None and self.max_batch_cost < 1:
            raise ValueError("stage max_batch_cost must be >= 1 when set")

    @classmethod
    def from_value(
        cls,
        value: PipelineStageRuntimePolicy | Mapping[str, Any] | None,
    ) -> PipelineStageRuntimePolicy:
        """Normalize a runtime policy mapping; unknown keys fail construction."""

        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise TypeError("stage runtime policy must be a mapping")
        raw = dict(value)
        metadata = dict(raw.pop("metadata", {}))
        return cls(metadata=metadata, **raw)


@dataclass(frozen=True, slots=True)
class PipelineStage:
    """One stage node in a physical generation pipeline topology."""

    name: str
    next_stages: tuple[str, ...] = ()
    terminal: bool = False
    handler: str | None = None
    handler_kwargs: dict[str, Any] = field(default_factory=dict)
    gpu_ids: tuple[int, ...] = ()
    runtime: PipelineStageRuntimePolicy = field(
        default_factory=PipelineStageRuntimePolicy,
    )
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("pipeline stage name must be non-empty")
        if self.terminal and self.next_stages:
            raise ValueError(f"terminal stage {self.name!r} must not declare next stages")
        if any(not stage for stage in self.next_stages):
            raise ValueError(f"stage {self.name!r} has an empty next stage")
        if any(gpu < 0 for gpu in self.gpu_ids):
            raise ValueError(f"stage {self.name!r} gpu_ids must be non-negative")

    @classmethod
    def from_value(
        cls,
        value: PipelineStage | Mapping[str, Any],
    ) -> PipelineStage:
        """Normalize a stage mapping; unknown keys fail construction.

        Scalar conveniences only: a single string ``next_stages`` and a single
        int ``gpu_ids`` are wrapped into tuples.
        """

        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise TypeError("pipeline stage must be a mapping")
        raw = dict(value)
        next_value = raw.pop("next_stages", ())
        if next_value is None:
            next_stages: tuple[str, ...] = ()
        elif isinstance(next_value, str):
            next_stages = (next_value,)
        else:
            next_stages = tuple(str(stage) for stage in next_value)
        gpu_value = raw.pop("gpu_ids", ())
        if gpu_value is None:
            gpu_ids: tuple[int, ...] = ()
        elif isinstance(gpu_value, int):
            gpu_ids = (gpu_value,)
        else:
            gpu_ids = tuple(int(gpu) for gpu in gpu_value)
        handler_kwargs = raw.pop("handler_kwargs", None)
        runtime = PipelineStageRuntimePolicy.from_value(raw.pop("runtime", None))
        metadata = dict(raw.pop("metadata", {}))
        return cls(
            next_stages=next_stages,
            gpu_ids=gpu_ids,
            handler_kwargs=dict(handler_kwargs or {}),
            runtime=runtime,
            metadata=metadata,
            **raw,
        )


@dataclass(frozen=True, slots=True)
class PipelineTopology:
    """Validated directed stage topology for physical generation pipelines."""

    stages: tuple[PipelineStage, ...]
    entry_stage: str | None = None

    def __post_init__(self) -> None:
        if not self.stages:
            raise ValueError("pipeline topology requires at least one stage")
        names = [stage.name for stage in self.stages]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(f"duplicate pipeline stage(s): {', '.join(duplicates)}")
        known = set(names)
        entry = self.entry_stage or self.stages[0].name
        if entry not in known:
            raise ValueError(f"pipeline entry stage {entry!r} is not declared")
        for stage in self.stages:
            missing = sorted(set(stage.next_stages) - known)
            if missing:
                raise ValueError(
                    f"stage {stage.name!r} references missing next stage(s): {', '.join(missing)}",
                )
        if not any(stage.terminal for stage in self.stages):
            raise ValueError("pipeline topology requires at least one terminal stage")
        # A pipeline must be a DAG: runners follow next_stages edges hop by
        # hop, so a cycle would loop forever (SerialPipelineRunner has no hop
        # cap by design; this validation is what bounds it.
        edges = {stage.name: stage.next_stages for stage in self.stages}
        in_progress: set[str] = set()
        done: set[str] = set()

        def _visit(name: str, path: tuple[str, ...]) -> None:
            if name in done:
                return
            if name in in_progress:
                cycle = (*path[path.index(name) :], name)
                raise ValueError(
                    f"pipeline topology has a cycle: {' -> '.join(cycle)}",
                )
            in_progress.add(name)
            for successor in edges[name]:
                _visit(successor, (*path, name))
            in_progress.discard(name)
            done.add(name)

        for stage in self.stages:
            _visit(stage.name, ())

    @property
    def resolved_entry_stage(self) -> str:
        return self.entry_stage or self.stages[0].name

    @property
    def terminal_stages(self) -> tuple[str, ...]:
        return tuple(stage.name for stage in self.stages if stage.terminal)

    def stage_for(self, name: str) -> PipelineStage:
        for stage in self.stages:
            if stage.name == name:
                return stage
        raise KeyError(f"unknown pipeline stage {name!r}")

    @classmethod
    def from_value(
        cls,
        value: PipelineTopology | Mapping[str, Any],
    ) -> PipelineTopology:
        """Normalize a topology mapping."""

        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise TypeError("pipeline topology must be a mapping")
        stages = tuple(PipelineStage.from_value(stage) for stage in value["stages"])
        return cls(
            stages=stages,
            entry_stage=value.get("entry_stage"),
        )


__all__ = [
    "PipelineStage",
    "PipelineStageRuntimePolicy",
    "PipelineTopology",
]
