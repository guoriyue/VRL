"""Runtime assembly and generation executor for CausVid."""

from __future__ import annotations

from typing import Any

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

    def __init__(self, model: Any, *, samples_per_chunk: int = 1) -> None:
        self.model = model
        self.default_samples_per_chunk = max(1, int(samples_per_chunk))


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
