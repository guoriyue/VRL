"""Hunyuan3D family runtime: the reference-image batch executor.

The shape DiT is conditioned on one RGBA cutout per prompt and emits rendered
views, so the executor is the shared reference-conditioned base with the
family's identity, the RGBA load mode upstream's recentering needs (the alpha
channel is the object mask) and the released four-view render as the default
frame count.
"""

from __future__ import annotations

from vrl.generation.bindings.full_sequence import (
    DenoiseBatchExecutorBase,
    ReferenceConditionedBatches,
)


class Hunyuan3DBatchExecutor(ReferenceConditionedBatches, DenoiseBatchExecutorBase):
    """Diffusion executor for Hunyuan3D image-to-shape rollouts."""

    family: str = "hunyuan3d"
    task: str = "i23d"
    # ``sampling.num_frames`` is the number of rendered views.
    default_num_frames: int = 4
    reference_image_mode = "RGBA"


__all__ = ["Hunyuan3DBatchExecutor"]
