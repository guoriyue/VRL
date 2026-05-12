"""Compatibility helpers for the migration period."""

from __future__ import annotations

from typing import Any

from vrl.engine.trajectory.types import TrajectoryBatch

TRAJECTORY_EXTRA_KEY = "trajectory"


def get_output_trajectory(output: Any) -> TrajectoryBatch | None:
    """Return a trajectory from ``OutputBatch`` or its short-term bridge."""

    trajectory = getattr(output, "trajectory", None)
    if trajectory is not None:
        return trajectory
    extra = getattr(output, "extra", None)
    if isinstance(extra, dict):
        bridge = extra.get(TRAJECTORY_EXTRA_KEY)
        if isinstance(bridge, TrajectoryBatch):
            return bridge
    return None


def require_output_trajectory(output: Any) -> TrajectoryBatch:
    """Return the trajectory or raise a migration-friendly error."""

    trajectory = get_output_trajectory(output)
    if trajectory is None:
        request_id = getattr(output, "request_id", "<unknown>")
        raise RuntimeError(f"OutputBatch {request_id!r} is missing TrajectoryBatch")
    return trajectory


__all__ = [
    "TRAJECTORY_EXTRA_KEY",
    "get_output_trajectory",
    "require_output_trajectory",
]
