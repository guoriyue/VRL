"""Scheduled token-autoregressive loop used by token-family runtimes.

The loop owns token-level scheduling and row-wise state routing for the current
family runners. It has no public rollout API surface; generation callers only
see the family runtime's ``GenerationOutput``.
"""

from __future__ import annotations

import inspect
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from vrl.generation.steps.token import TokenLoopInit, TokenStepBatch, TokenStepOutput
from vrl.nn.layers.attention.cache_rows import ARCacheRows, ar_concat_rows, ar_split_rows


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
        rows = [int(index) for index in row_indices]
        return TokenStepBatch(
            row_indices=rows,
            position=position,
            row_lanes={name: lane.gather(rows) for name, lane in self.row_lanes.items()},
        )

    def apply_step_output(self, batch: TokenStepBatch, output: TokenStepOutput) -> None:
        for name, value in output.updated_row_lanes.items():
            self._require_row_lane(name).scatter(batch.row_indices, value)

    def _require_row_lane(self, name: str) -> ARCacheRows:
        try:
            return self.row_lanes[name]
        except KeyError as exc:
            raise KeyError(f"unknown AR row lane: {name!r}") from exc


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
        if self.scheduler_batch_size is None:
            return
        if (
            isinstance(self.scheduler_batch_size, bool)
            or not isinstance(self.scheduler_batch_size, int)
            or self.scheduler_batch_size < 1
        ):
            raise ValueError("scheduler_batch_size must be a positive integer")

    def run(self) -> Any:
        self._require_hooks(("init_token", "step_token", "finalize_token"))

        init = call_with_supported_kwargs(
            self.runner.init_token,
            *self.init_args,
            **dict(self.init_kwargs or {}),
        )
        if not isinstance(init, TokenLoopInit):
            raise TypeError(
                "token model runner init_token must return TokenLoopInit; "
                f"got {type(init).__name__}",
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
                if not isinstance(step_output, TokenStepOutput):
                    raise TypeError(
                        "token model runner step_token must return TokenStepOutput; "
                        f"got {type(step_output).__name__}",
                    )
                envelope.apply_step_output(step_batch, step_output)

        return self.runner.finalize_token(init.state)

    def _require_hooks(self, names: Sequence[str]) -> None:
        missing = [name for name in names if not hasattr(self.runner, name)]
        if missing:
            raise TypeError(
                "token-autoregressive loop requires model runner hooks: " + ", ".join(missing),
            )


def call_with_supported_kwargs(fn: Any, *args: Any, **kwargs: Any) -> Any:
    """Call hooks while dropping optional kwargs unsupported by older signatures."""

    try:
        signature = inspect.signature(fn)
    except (TypeError, ValueError):
        return fn(*args, **kwargs)
    parameters = signature.parameters
    if any(param.kind is inspect.Parameter.VAR_KEYWORD for param in parameters.values()):
        return fn(*args, **kwargs)
    supported = {key: value for key, value in kwargs.items() if key in parameters}
    return fn(*args, **supported)


__all__ = [
    "ARCacheRows",
    "TokenAutoregressiveEnvelope",
    "TokenAutoregressiveLoop",
    "ar_concat_rows",
    "ar_split_rows",
    "call_with_supported_kwargs",
]
