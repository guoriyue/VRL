"""Qwen-Image family runtime.

Registry-descriptor family: no builder functions live here — the generic
functions in ``vrl.models.diffusion.build`` construct the bundles from the
``DiffusionFamilyBuild`` recipe on the qwen_image registry entry.
"""

from __future__ import annotations

from typing import Any

from vrl.generation.diffusion import (
    DiffusionChunkExecutorBase,
)

# Registry-descriptor family: build/extract/replay all come from the generic
# functions in ``vrl.models.diffusion.build``, and the capability is declared
# once on the registry entry (the worker injects it into the executor) — this
# module ships only the chunk executor.


class QwenImageChunkExecutor(DiffusionChunkExecutorBase):
    """Diffusion executor for Qwen-Image text-to-image rollouts.

    All encoded values (prompt/negative embeds + masks) are batched tensors or
    None, so the base ``build_chunk_encoded`` repeat path needs no override
    (unlike FLUX's batch-shared ``text_ids``).
    """

    family: str = "qwen_image"
    task: str = "t2i"
    default_num_frames: int = 1
    default_max_sequence_length: int = 1024

    def __init__(
        self,
        model: Any,  # QwenImageModel
        *,
        samples_per_chunk: int = 8,
    ) -> None:
        self.model = model
        self.default_samples_per_chunk = max(1, int(samples_per_chunk))


__all__ = [
    "QwenImageChunkExecutor",
]
