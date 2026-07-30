"""Typed inputs shared by Ray generation launch and on-demand activation."""

from __future__ import annotations

import pickle
from dataclasses import dataclass

from vrl.generation.launch_contract import GenerationRuntimeLaunchContract
from vrl.generation.protocols import ChunkGatherer


@dataclass(frozen=True, slots=True)
class RayGenerationLaunchInputs:
    """Serializable worker build contract plus registry-owned chunk gatherer.

    The driver executor and single-worker pipelined path receive the same
    binding; neither side re-resolves family identity after composition.
    """

    launch_contract: GenerationRuntimeLaunchContract
    gatherer: ChunkGatherer

    def __post_init__(self) -> None:
        if not isinstance(self.launch_contract, GenerationRuntimeLaunchContract):
            raise TypeError(
                "launch_contract must be a GenerationRuntimeLaunchContract, "
                f"got {type(self.launch_contract).__name__}",
            )
        if not isinstance(self.gatherer, ChunkGatherer) or not callable(
            getattr(self.gatherer, "gather_chunks", None),
        ):
            raise TypeError(
                f"gatherer must implement ChunkGatherer, got {type(self.gatherer).__name__}",
            )
        try:
            pickle.dumps(self)
        except Exception as error:
            raise TypeError(
                "RayGenerationLaunchInputs must be pickle-serializable",
            ) from error


__all__ = ["RayGenerationLaunchInputs"]
