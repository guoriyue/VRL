"""Wan 2.1 family runtime.

Registry-descriptor family on both sides: the generic builders construct the
rollout and replay bundles from the two per-variant registry entries (the
dual-stage ``transformer_2`` late-load lives in the replay model's
``prepare_replay``). This module ships only the i2v chunk executor — t2v uses
the shared generic executor; i2v keeps a subclass for its
reference-conditioning chunk logic.
"""

from __future__ import annotations

from vrl.generation.bindings.full_sequence_denoise import (
    DiffusionChunkExecutorBase,
    ReferenceConditionedChunks,
)


class Wan_2_1I2VChunkExecutor(ReferenceConditionedChunks, DiffusionChunkExecutorBase):
    """Diffusion executor for Wan 2.1 image-to-video rollouts."""

    family: str = "wan_2_1_i2v"
    task: str = "i2v"
    default_num_frames: int = 81
    default_max_sequence_length: int = 512


__all__ = [
    "Wan_2_1I2VChunkExecutor",
]
