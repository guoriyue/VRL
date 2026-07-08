"""FLUX.1 family runtime.

Registry-descriptor family: no builder functions live here — the generic
functions in ``vrl.models.diffusion.build`` construct the bundles from the
``DiffusionFamilyBuild`` recipe on the flux registry entry. The NFT
``previous`` adapter is config-driven (``model.nft_previous_adapter``) inside
``FluxModel.apply_lora``; the dynamic-shift replay timesteps live in
``FluxReplayModel.prepare_replay``.
"""

from __future__ import annotations

from typing import Any

from vrl.generation.diffusion import DiffusionChunkExecutorBase
from vrl.utils.logging import init_logger

logger = init_logger(__name__)
class FluxChunkExecutor(DiffusionChunkExecutorBase):
    """Diffusion executor for FLUX.1 text-to-image rollouts."""

    family: str = "flux"
    task: str = "t2i"
    default_num_frames: int = 1
    default_max_sequence_length: int = 512
    # ``text_ids`` is batch-shared (shape [seq, 3], no batch dim) — repeating
    # it would corrupt its leading dim, so the base repeat path skips it.
    chunk_passthrough_keys: tuple[str, ...] = ("text_ids",)

    def __init__(
        self,
        model: Any,  # FluxModel
        *,
        samples_per_chunk: int = 8,
    ) -> None:
        self.model = model
        self.default_samples_per_chunk = max(1, int(samples_per_chunk))

__all__ = [
    "FluxChunkExecutor",
]
