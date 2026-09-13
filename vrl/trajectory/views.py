"""Derived views over trajectory records."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from vrl.trajectory.types import validate_string_tuple

RewardValueRange = Literal["unit", "tanh"]


@dataclass(frozen=True, slots=True)
class RewardInputSpec:
    """Declare the output references and pixel range consumed by reward scoring."""

    name: str
    tensor_refs: tuple[str, ...] = ()
    # Declared pixel value range of the reward media. The collector normalizes
    # to "unit" [0, 1] for scoring; "tanh" [-1, 1] sources (VQ decode) get
    # rescaled. This is a model fact owned by the producer, not an operator knob.
    value_range: RewardValueRange = "unit"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("RewardInputSpec.name must be non-empty")
        if self.value_range not in {"unit", "tanh"}:
            raise ValueError(
                f"RewardInputSpec.value_range must be 'unit' or 'tanh', got {self.value_range!r}",
            )
        validate_string_tuple("RewardInputSpec.tensor_refs", self.tensor_refs)


__all__ = [
    "RewardInputSpec",
    "RewardValueRange",
]
