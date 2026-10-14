"""VDN-H3 family runtime: the batch-one executor.

It is MiniMax-H3's, differing only in the launch identity it publishes. Replay
construction lives on ``VDNH3Model.from_build``, which applies the same graft
as the rollout loader: a replay policy without it would recompute log-probs for
a different model.
"""

from __future__ import annotations

from vrl.models.families.minimax_h3.runtime import (
    DEFAULT_FPS,
    DEFAULT_MAX_SEQUENCE_LENGTH,
    DEFAULT_NUM_FRAMES,
    MiniMaxH3BatchExecutor,
)


class VDNH3BatchExecutor(MiniMaxH3BatchExecutor):
    """Diffusion executor for VDN-H3 text-to-video rollouts.

    Inherits MiniMax-H3's batch-one contract unchanged: one packed layout per
    forward, one prompt per generation batch, no negative prompt (the backbone
    is guidance-distilled).
    """

    family: str = "vdn_h3"
    task: str = "t2v"


__all__ = [
    "DEFAULT_FPS",
    "DEFAULT_MAX_SEQUENCE_LENGTH",
    "DEFAULT_NUM_FRAMES",
    "VDNH3BatchExecutor",
]
