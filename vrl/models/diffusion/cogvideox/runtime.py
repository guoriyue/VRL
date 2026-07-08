"""CogVideoX family runtime.

Registry-descriptor family: no builder functions live here — the generic
functions in ``vrl.models.diffusion.build`` construct the bundles from the
``DiffusionFamilyBuild`` recipe on the cogvideox registry entry — this module
ships only the chunk executor.

The replay scheduler needs the family's own class: the generic loader's
``scheduler_classname`` field covers it (``CogVideoXDDIMScheduler``).
"""

from __future__ import annotations

from typing import Any

from vrl.generation.diffusion import (
    DiffusionChunkExecutorBase,
)


class CogVideoXChunkExecutor(DiffusionChunkExecutorBase):
    """Diffusion executor for CogVideoX text-to-video rollouts.

    All encoded values (prompt/negative embeds, fixed-length padded) are
    batched tensors or None, so the base ``build_chunk_encoded`` repeat path
    needs no override.
    """

    family: str = "cogvideox"
    task: str = "t2v"
    # (F-1) must divide by the VAE's 4x temporal compression; 49 frames is the
    # reference default.
    default_num_frames: int = 49
    default_max_sequence_length: int = 226

    def __init__(
        self,
        model: Any,  # CogVideoXModel
        *,
        samples_per_chunk: int = 2,
    ) -> None:
        self.model = model
        self.default_samples_per_chunk = max(1, int(samples_per_chunk))


__all__ = [
    "CogVideoXChunkExecutor",
]
