"""Generation executor for CausVid."""

from __future__ import annotations

from vrl.generation.bindings.chunk_autoregressive import (
    ChunkAutoregressiveDenoiseExecutorBase,
)


class CausVidBatchExecutor(ChunkAutoregressiveDenoiseExecutorBase):
    """Prompt/sample transport around CausVid's family-owned causal loop."""

    family: str = "causvid"
    task: str = "t2v"


__all__ = [
    "CausVidBatchExecutor",
]
