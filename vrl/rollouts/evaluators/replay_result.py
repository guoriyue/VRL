"""Helpers for typed replay-forward results."""

from __future__ import annotations

from typing import Any

from vrl.models.interfaces import ReplayResult, ReplaySegmentResult


def require_replay_segment(
    output: ReplayResult,
    segment_name: str,
) -> ReplaySegmentResult:
    """Return a replay segment result or fail with a helpful segment list."""

    if not isinstance(output, ReplayResult):
        raise TypeError(f"replay output must be ReplayResult, got {type(output).__name__}")
    try:
        return output.segments[segment_name]
    except KeyError as err:
        raise KeyError(
            f"ReplayResult missing segment {segment_name!r}; "
            f"got {sorted(output.segments)}",
        ) from err


def require_replay_value(result: ReplaySegmentResult, key: str) -> Any:
    """Return a named replay payload or fail with the available payload keys."""

    if key not in result.values:
        raise KeyError(
            f"ReplaySegmentResult for segment {result.segment!r} "
            f"missing required key {key!r}; got {sorted(result.values)}",
        )
    return result.values[key]


__all__ = ["require_replay_segment", "require_replay_value"]
