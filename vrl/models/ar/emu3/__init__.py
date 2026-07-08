"""Emu3 family — BAAI's autoregressive multimodal model (HF-native).

Currently supports the HF-format ``BAAI/Emu3-Gen-hf`` checkpoint for
text-to-image generation under the visual-rl GRPO pipeline. Unlike janus_pro
(which needs the out-of-tree ``janus`` package), Emu3 ships inside
``transformers`` (>= 4.48), so no extra dependency is required.
"""

from __future__ import annotations

from vrl.models.ar.emu3.model import (
    Emu3Config,
    Emu3Model,
    Emu3ReplayModel,
    emu3_allowed_token_mask,
    emu3_forced_token_schedule,
    emu3_grid_token_num,
)
from vrl.models.ar.emu3.runtime import (
    Emu3ChunkExecutor,
    Emu3ChunkGatherer,
    build_emu3_replay_runtime_bundle,
    build_emu3_runtime_bundle,
    extract_emu3_runtime_spec,
)

__all__ = [
    "Emu3ChunkExecutor",
    "Emu3ChunkGatherer",
    "Emu3Config",
    "Emu3Model",
    "Emu3ReplayModel",
    "build_emu3_replay_runtime_bundle",
    "build_emu3_runtime_bundle",
    "emu3_allowed_token_mask",
    "emu3_forced_token_schedule",
    "emu3_grid_token_num",
    "extract_emu3_runtime_spec",
]
