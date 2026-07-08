"""Lumina-Image-2.0 family runtime.

Registry-descriptor family: no builder functions live here — the generic
functions in ``vrl.models.diffusion.build`` construct the bundles from the
``DiffusionFamilyBuild`` recipe on the lumina2 registry entry, and the
capability is declared once on the registry entry (the worker injects it into
the executor) — this module ships only the chunk executor.
"""

from __future__ import annotations

from typing import Any

from vrl.generation.diffusion import (
    DiffusionChunkExecutorBase,
)


class Lumina2ChunkExecutor(DiffusionChunkExecutorBase):
    """Diffusion executor for Lumina-Image-2.0 text-to-image rollouts.

    All encoded values (prompt/negative embeds + attention masks) are batched
    tensors or None, so the base ``build_chunk_encoded`` repeat path needs no
    override.
    """

    family: str = "lumina2"
    task: str = "t2i"
    default_num_frames: int = 1
    default_max_sequence_length: int = 256

    def __init__(
        self,
        model: Any,  # Lumina2Model
        *,
        samples_per_chunk: int = 8,
    ) -> None:
        self.model = model
        self.default_samples_per_chunk = max(1, int(samples_per_chunk))


__all__ = [
    "Lumina2ChunkExecutor",
]
