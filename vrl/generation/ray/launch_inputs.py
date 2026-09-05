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
        try:
            pickle.dumps(self)
        except Exception as error:
            raise TypeError(
                "RayGenerationLaunchInputs must be pickle-serializable",
            ) from error


__all__ = ["RayGenerationLaunchInputs"]
