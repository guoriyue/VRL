"""Model-free line-art sharpness reward: normalized Laplacian energy.

Rewards crisp cel-shaded line art. Every learned quality/aesthetic model tried
on this model's outputs — AnimeReward, PickScore, LAION aesthetic,
aesthetic-shadow-v2, even a Qwen2.5-VL-7B judge — is blind to cel-shading
softening (verified 2026-08-20: they score softened frames >= crisp ones, or
rank them inconsistently) because they score "prettiness", not line-art
fidelity. Laplacian energy measures the high-frequency edge content that IS the
crisp linework; it tracks the degradation monotonically (crisp ~13.4 > soft
~9.9 > softer ~7.7 >> grid ~0.28, in 1e-3 units) and cannot be faked by
softening — clean outlines require real high-frequency structure.

The one thing it CAN be fooled by is high-frequency noise, which also has high
Laplacian energy. It is therefore designed to ride alongside a coherence reward
(PickScore): PickScore rejects noise as low quality, so the only remaining way
up is genuine line art. Never use it as the sole quality signal.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


class ImageSharpnessRewardModel:
    """Per-artifact Laplacian-energy score in ``[0, 1]`` (higher = crisper)."""

    def __init__(self, worker_config: Mapping[str, Any]) -> None:
        cfg = dict(worker_config)
        # Divisor mapping base-quality crisp anime (~0.013 raw Laplacian
        # variance on [0,1] luma) to ~1.0; softened/collapsed frames fall below.
        self._scale = float(cfg.get("scale", 0.013))
        if self._scale <= 0.0:
            raise ValueError("image_sharpness scale must be > 0")
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
