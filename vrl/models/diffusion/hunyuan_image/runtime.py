"""HunyuanImage family runtime.

Registry-descriptor family: no builder functions live here — the generic
functions in ``vrl.models.diffusion.build`` construct the bundles from the
``DiffusionFamilyBuild`` recipe on the hunyuan_image registry entry — this
module ships only the chunk executor.
"""

from __future__ import annotations

from typing import Any

from vrl.generation.diffusion import (
    DiffusionChunkExecutorBase,
)


class HunyuanImageChunkExecutor(DiffusionChunkExecutorBase):
    """Diffusion executor for HunyuanImage-2.1 text-to-image rollouts.

    All encoded values (both prompt/negative embed pairs + masks) are batched
    tensors or None, so the base ``build_chunk_encoded`` repeat path needs no
    override.
    """

    family: str = "hunyuan_image"
    task: str = "t2i"
    default_num_frames: int = 1
    # The pipeline pads the Qwen2.5-VL stream to a FIXED tokenizer_max_length
    # of 1000 (no encode-time knob); this default only labels the request — the
    # family encode_prompt ignores max_sequence_length (see model.py).
    default_max_sequence_length: int = 1000

    def __init__(
        self,
        model: Any,  # HunyuanImageModel
        *,
        samples_per_chunk: int = 2,
    ) -> None:
        self.model = model
        # 17B transformer + separate-CFG double forward: keep chunks small by
        # default; recipes raise it explicitly when VRAM allows.
        self.default_samples_per_chunk = max(1, int(samples_per_chunk))


__all__ = [
    "HunyuanImageChunkExecutor",
]
