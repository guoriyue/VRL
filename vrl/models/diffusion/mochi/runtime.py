"""Mochi-1 family runtime.

Registry-descriptor family: no builder functions live here — the generic
functions in ``vrl.models.diffusion.build`` construct the bundles from the
``DiffusionFamilyBuild`` recipe on the mochi registry entry — this module
ships only the chunk executor.
"""

from __future__ import annotations

from typing import Any

from vrl.generation.diffusion import (
    DiffusionChunkExecutorBase,
)


class MochiChunkExecutor(DiffusionChunkExecutorBase):
    """Diffusion executor for Mochi-1 text-to-video rollouts.

    All encoded values (prompt/negative embeds + bool masks) are batched
    tensors or None, so the base ``build_chunk_encoded`` repeat path needs no
    override.
    """

    family: str = "mochi"
    task: str = "t2v"
    # (F-1) must divide by the VAE's 6x temporal compression; 19 frames is the
    # reference default.
    default_num_frames: int = 19
    default_max_sequence_length: int = 256

    def __init__(
        self,
        model: Any,  # MochiModel
        *,
        samples_per_chunk: int = 2,
    ) -> None:
        self.model = model
        self.default_samples_per_chunk = max(1, int(samples_per_chunk))


__all__ = [
    "MochiChunkExecutor",
]
