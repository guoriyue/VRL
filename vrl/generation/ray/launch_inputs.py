"""Typed inputs shared by Ray generation launch and on-demand activation."""

from __future__ import annotations

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


__all__ = ["RayGenerationLaunchInputs"]
