"""SD 3.5 family runtime.

Registry-descriptor family: no builder functions live here — the generic
functions in ``vrl.models.diffusion.build`` construct the bundles from the
``DiffusionFamilyBuild`` recipe on the sd3_5 registry entry.
"""

from __future__ import annotations

from typing import Any

from vrl.generation.diffusion import DiffusionChunkExecutorBase
from vrl.models.diffusion.capabilities import diffusion_family_capability

SD3_5_FAMILY_CAPABILITY = diffusion_family_capability("sd3_5", "t2i")


class SD3_5ChunkExecutor(DiffusionChunkExecutorBase):
    """Diffusion executor for SD3.5-M text-to-image rollouts."""

    family: str = "sd3_5"
    task: str = "t2i"
    family_capability = SD3_5_FAMILY_CAPABILITY
    default_num_frames: int = 1
    default_max_sequence_length: int = 128

    def __init__(
        self,
        model: Any,  # SD3_5Model
        *,
        samples_per_chunk: int = 8,
    ) -> None:
        self.model = model
        self.default_samples_per_chunk = max(1, int(samples_per_chunk))

__all__ = [
    "SD3_5ChunkExecutor",
    "SD3_5_FAMILY_CAPABILITY",
]
