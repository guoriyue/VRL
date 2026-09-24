"""Model-free line-art sharpness reward: normalized Laplacian energy.

Measures high-frequency image content. A blurred edge can score lower than a
crisp edge, but this is not a semantic quality measure: noise, checkerboards,
oversharpening and text overlays can also increase it. Pairing it with a learned
reward does not guarantee rejection of these shortcuts. Validate the combined
reward on task-specific corruptions and held-out preferences before training.
Never use it as the sole quality signal.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any


class ImageSharpnessRewardModel:
    """Per-artifact Laplacian-energy score in ``[0, 1]`` (higher = crisper)."""

    def __init__(self, worker_config: Mapping[str, Any]) -> None:
        cfg = dict(worker_config)
        # Divisor mapping base-quality crisp anime (~0.013 raw Laplacian
        # variance on [0,1] luma) to ~1.0; softened/collapsed frames fall below.
        self._scale = float(cfg.get("scale", 0.013))
        if not math.isfinite(self._scale) or self._scale <= 0.0:
            raise ValueError("image_sharpness scale must be finite and > 0")
        self._num_frames = int(cfg.get("num_frames", 1))
        if self._num_frames <= 0:
            raise ValueError("image_sharpness num_frames must be > 0")

    def score_batch(self, artifacts: Sequence[Any]) -> list[dict[str, float]]:
        return [{"image_sharpness": self._score_one(artifact)} for artifact in artifacts]

    def __call__(self, artifact: Any) -> dict[str, float]:
        return {"image_sharpness": self._score_one(artifact)}

    def _score_one(self, artifact: Any) -> float:
        import numpy as np

        from vrl.rewards.models.media import decode_artifact_frames

        frames = decode_artifact_frames(artifact, self._num_frames)
        arr = np.asarray(frames.detach().cpu().numpy(), dtype=np.float32)
        frame = arr[arr.shape[0] // 2]  # [H, W, 3] in [0, 1]
        gray = 0.299 * frame[..., 0] + 0.587 * frame[..., 1] + 0.114 * frame[..., 2]
        # 3x3 Laplacian; its variance is the high-frequency (edge) energy.
        lap = (
            -4.0 * gray[1:-1, 1:-1]
            + gray[:-2, 1:-1]
            + gray[2:, 1:-1]
            + gray[1:-1, :-2]
            + gray[1:-1, 2:]
        )
        energy = float(lap.var()) if lap.size else 0.0
        return min(1.0, energy / self._scale)


__all__ = ["ImageSharpnessRewardModel"]
