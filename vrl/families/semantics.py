"""Orthogonal semantics for the trainable policy exposed by a family entry.

These fields classify the executable policy variant selected by the registry,
not every component stored in a checkpoint.  A hybrid checkpoint may contain a
causal reasoner, a token prior, and a frozen denoise renderer while exposing
only one of those stages as the RL policy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, get_args

TemporalOrganization = Literal["joint", "causal", "causal_chunked"]
PolicyStepKind = Literal["denoise", "token"]
ActionDistribution = Literal["categorical", "continuous"]
TrajectoryLayout = Literal["denoise", "token", "multisegment_token"]


@dataclass(frozen=True, slots=True)
class PolicySemantics:
    """Generation semantics of the trainable, action-producing policy stage.

    ``joint`` means one model step updates the whole output field without an
    output-prefix cache. ``causal`` advances ordered positions from a prefix;
    ``causal_chunked`` does the same at temporal-chunk granularity.
    """

    temporal_organization: TemporalOrganization
    step_kind: PolicyStepKind
    action_distribution: ActionDistribution
    trajectory_layout: TrajectoryLayout

    def __post_init__(self) -> None:
        for name, value, literal_type in (
            (
                "temporal_organization",
                self.temporal_organization,
                TemporalOrganization,
            ),
            ("step_kind", self.step_kind, PolicyStepKind),
            ("action_distribution", self.action_distribution, ActionDistribution),
            ("trajectory_layout", self.trajectory_layout, TrajectoryLayout),
        ):
            if value not in get_args(literal_type):
                raise ValueError(f"unsupported policy {name}: {value!r}")

        if self.step_kind == "denoise" and self.trajectory_layout != "denoise":
            raise ValueError("denoise policy steps require a denoise trajectory layout")
        if self.step_kind == "token" and self.trajectory_layout == "denoise":
            raise ValueError("token policy steps require a token trajectory layout")
        if self.action_distribution == "categorical" and self.step_kind != "token":
            raise ValueError("categorical policy actions require token steps")
        if self.step_kind == "denoise" and self.action_distribution != "continuous":
            raise ValueError("denoise policy steps require continuous actions")
        if self.trajectory_layout == "multisegment_token" and self.step_kind != "token":
            raise ValueError("multisegment_token trajectories require token steps")


__all__ = [
    "ActionDistribution",
    "PolicySemantics",
    "PolicyStepKind",
    "TemporalOrganization",
    "TrajectoryLayout",
]
