"""Scheduled token-autoregressive loop used by token-family runtimes.

The loop owns token-level scheduling and row-wise state routing for the current
family runners. It has no public rollout API surface; generation callers only
see the family runtime's ``GenerationOutput``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from vrl.generation.steps.token import TokenLoopInit, TokenStepBatch, TokenStepOutput
from vrl.nn.layers.attention.cache_rows import ARCacheRows
from vrl.utils.validation import require_int


@dataclass(slots=True)
class TokenAutoregressiveEnvelope:
    """Generation-owned row-state envelope scheduled by the decode loop."""

    row_lanes: dict[str, ARCacheRows] = field(default_factory=dict)

    @classmethod
    def from_init(
        cls,
        init: TokenLoopInit,
    ) -> TokenAutoregressiveEnvelope:
        if not init.row_lanes:
            raise ValueError("TokenLoopInit.row_lanes must be non-empty")
        return cls(
            row_lanes={
                name: ARCacheRows.from_batched(
                    value,
                    init.row_count,
                    owner=f"ar.row_lanes.{name}",
                )
                for name, value in init.row_lanes.items()
            },
        )

    def build_step_batch(
        self,
        row_indices: Sequence[int],
        *,
        position: int,
    ) -> TokenStepBatch:
        rows = list(row_indices)
        return TokenStepBatch(
            row_indices=rows,
            position=position,
            row_lanes={name: lane.gather(rows) for name, lane in self.row_lanes.items()},
        )

    def apply_step_output(self, batch: TokenStepBatch, output: TokenStepOutput) -> None:
        # Reject an invalid output schema before changing any scheduled row.
        for name in output.updated_row_lanes:
            if name not in self.row_lanes:
                raise KeyError(f"unknown AR row lane: {name!r}")
        for name, value in output.updated_row_lanes.items():
            self.row_lanes[name].scatter(batch.row_indices, value)


class TokenAutoregressiveLoop:
    """Family-neutral token-autoregressive composition over token policy steps."""

    __slots__ = ("init_args", "init_kwargs", "runner", "scheduler_batch_size")

    def __init__(
        self,
        *,
        runner: Any,
        scheduler_batch_size: int | None = None,
        init_args: Sequence[Any] = (),
        init_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        self.runner = runner
        self.scheduler_batch_size = scheduler_batch_size
        self.init_args = init_args
        self.init_kwargs = init_kwargs
        if scheduler_batch_size is not None:
            require_int(scheduler_batch_size, path="scheduler_batch_size", minimum=1)

    def run(self) -> Any:
        init = self.runner.init_token(
            *self.init_args,
            **dict(self.init_kwargs or {}),
        )
        envelope = TokenAutoregressiveEnvelope.from_init(init)
        batch_size = self.scheduler_batch_size or init.row_count

        for position in range(init.step_count):
            for start in range(0, init.row_count, batch_size):
                step_batch = envelope.build_step_batch(
                    range(start, min(start + batch_size, init.row_count)),
                    position=position,
                )
                step_output = self.runner.step_token(init.state, step_batch)
                envelope.apply_step_output(step_batch, step_output)

        return self.runner.finalize_token(init.state)


__all__ = [
    "TokenAutoregressiveEnvelope",
    "TokenAutoregressiveLoop",
]
