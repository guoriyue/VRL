"""VDN-H3 family runtime: the batch-one executor and the replay bundle builder.

Both are the MiniMax-H3 ones plus the graft. The executor differs only in the
launch identity it publishes and in its default step count, since VDN-H3 ships
a 50-NFE and an 8-NFE (DMD-distilled) artifact where the dense base runs 50.
The replay builder must apply the same transform and weights as the rollout
builder: a replay policy without the graft would recompute log-probs for a
different model.
"""

from __future__ import annotations

from vrl.models.families.minimax_h3.runtime import (
    DEFAULT_FPS,
    DEFAULT_MAX_SEQUENCE_LENGTH,
    DEFAULT_NUM_FRAMES,
    MiniMaxH3BatchExecutor,
    load_h3_replay_components,
)
from vrl.models.interfaces.runtime import ModelBuild, RuntimeBundle
from vrl.utils.logging import init_logger

logger = init_logger(__name__)


def build_vdn_h3_replay_runtime_bundle(build: ModelBuild) -> RuntimeBundle:
    """Transformer + both H3 schedulers, with the hybrid graft applied."""

    from vrl.models.families.vdn_h3.model import VDNH3ReplayModel
    from vrl.models.steps.denoise.build import assemble_replay_bundle

    logger.info("Building vdn_h3 replay runtime bundle from %s", build.model_name_or_path)
    model = VDNH3ReplayModel(**load_h3_replay_components(build))
    # Same graft as the rollout policy, before LoRA attach or FSDP wrapping:
    # the transform replaces every block's attention module, so anything that
    # holds a reference to the old one must be built after it.
    model.install_hybrid_attention(build)
    num_steps = build.num_steps
    if num_steps is not None:
        model.set_num_steps(num_steps)
    return assemble_replay_bundle(model, build)


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
    "build_vdn_h3_replay_runtime_bundle",
]
