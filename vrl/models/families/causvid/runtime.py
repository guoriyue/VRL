"""Runtime assembly and generation executor for CausVid."""

from __future__ import annotations

from vrl.generation.bindings.chunk_autoregressive_denoise import (
    ChunkAutoregressiveDenoiseExecutorBase,
)
from vrl.models.interfaces.runtime import ModelBuild, RuntimeBundle
from vrl.utils.logging import init_logger

logger = init_logger(__name__)


class CausVidChunkExecutor(ChunkAutoregressiveDenoiseExecutorBase):
    """Prompt/sample transport around CausVid's family-owned causal loop."""

    family: str = "causvid"
    task: str = "t2v"


def build_causvid_replay_runtime_bundle(build: ModelBuild) -> RuntimeBundle:
    """Build the minimal transformer-only CausVid trainer runtime."""

    from vrl.models.families.causvid.model import CausVidReplayModel
    from vrl.models.steps.denoise.build import assemble_replay_bundle

    build.require_replay()
    logger.info("Building CausVid replay runtime from %s", build.model_name_or_path)
    return assemble_replay_bundle(CausVidReplayModel.from_build(build), build)


__all__ = [
    "CausVidChunkExecutor",
    "build_causvid_replay_runtime_bundle",
]
