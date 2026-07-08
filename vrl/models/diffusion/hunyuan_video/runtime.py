"""HunyuanVideo family runtime.

Registry-descriptor family: no builder functions live here — the generic
functions in ``vrl.models.diffusion.build`` construct the bundles from the
``DiffusionFamilyBuild`` recipe on the hunyuan_video registry entry — this
module ships only the chunk executor.
"""

from __future__ import annotations

from typing import Any

from vrl.generation.diffusion import (
    DiffusionChunkExecutorBase,
)


class HunyuanVideoChunkExecutor(DiffusionChunkExecutorBase):
    """Diffusion executor for HunyuanVideo text-to-video rollouts.

    All encoded values (prompt embeds + mask + pooled) are batched tensors or
    None, so the base ``build_chunk_encoded`` repeat path needs no override.
    """

    family: str = "hunyuan_video"
    task: str = "t2v"
    # (F-1) must divide by the VAE's 4x temporal compression; 17 frames is the
    # small validation config from the landing sprint.
    default_num_frames: int = 17
    default_max_sequence_length: int = 256

    def __init__(
        self,
        model: Any,  # HunyuanVideoModel
        *,
        samples_per_chunk: int = 2,
    ) -> None:
        self.model = model
        self.default_samples_per_chunk = max(1, int(samples_per_chunk))


__all__ = [
    "HunyuanVideoChunkExecutor",
]
