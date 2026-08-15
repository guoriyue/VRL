"""Typed inputs shared by Ray generation launch and on-demand activation."""

from __future__ import annotations

import pickle
from dataclasses import dataclass

from vrl.generation.execution.rank_group import RankGroupSpec
from vrl.generation.launch_contract import GenerationRuntimeLaunchContract
from vrl.generation.protocols import GenerationBatchGatherer


@dataclass(frozen=True, slots=True)
class RayGenerationLaunchInputs:
    """Serializable rank build contract plus registry-owned batch gatherer.

    The driver executor and single-engine pipelined path receive the same
    binding; neither side re-resolves family identity after composition.
    ``rank_group`` is None for single-rank engines; the launcher stamps a
    per-rank rendezvous spec when an engine spans multiple GPUs.
    """

    launch_contract: GenerationRuntimeLaunchContract
    gatherer: GenerationBatchGatherer
    rank_group: RankGroupSpec | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.launch_contract, GenerationRuntimeLaunchContract):
            raise TypeError(
                "launch_contract must be a GenerationRuntimeLaunchContract, "
                f"got {type(self.launch_contract).__name__}",
            )
        if self.rank_group is not None and not isinstance(self.rank_group, RankGroupSpec):
            raise TypeError(
                f"rank_group must be a RankGroupSpec or None, got {type(self.rank_group).__name__}",
            )
        if not isinstance(self.gatherer, GenerationBatchGatherer) or not callable(
            getattr(self.gatherer, "gather_batches", None),
        ):
            raise TypeError(
                f"gatherer must implement GenerationBatchGatherer, got {type(self.gatherer).__name__}",
            )
        try:
            pickle.dumps(self)
        except Exception as error:
            raise TypeError(
                "RayGenerationLaunchInputs must be pickle-serializable",
            ) from error


__all__ = ["RayGenerationLaunchInputs"]
