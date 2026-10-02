"""Model-free tonal anchor: penalize drift in exposure and saturation from a reference image.

A preference judge trained on "which image is better" is indifferent to a global brightening,
a saturation boost or a glow (the pairwise Qwen3-VL-32B judge rates those transforms better than
the original 33-47% of the time), so a policy optimizing it drifts brighter and glossier without
being stopped. This guard measures the drift explicitly against the base-model image of the same
prompt (``metadata[reference_key]``): mean luma and mean HSV saturation, each in ``[0, 1]``.

Score = -(relu(|dL| - tol) + relu(|dS| - tol)), so it is 0 inside the tolerance band and
negative outside; it never rewards anything, it only anchors. Pair it with a learned judge;
it is not a quality measure.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any


def tonal_stats(frame: Any) -> tuple[float, float]:
    """Mean luma and mean HSV saturation of an ``[H, W, 3]`` float array in ``[0, 1]``."""
    import numpy as np

    frame = np.asarray(frame, dtype=np.float32)
    luma = 0.299 * frame[..., 0] + 0.587 * frame[..., 1] + 0.114 * frame[..., 2]
    mx = frame.max(axis=-1)
    mn = frame.min(axis=-1)
    sat = np.where(mx > 1e-6, (mx - mn) / np.maximum(mx, 1e-6), 0.0)
    return float(luma.mean()), float(sat.mean())


class TonalAnchorRewardModel:
    """Per-artifact exposure/saturation drift penalty against a reference image."""

    def __init__(self, worker_config: Mapping[str, Any]) -> None:
        cfg = dict(worker_config)
        self._reference_key = str(cfg.get("reference_key") or "judge_reference")
        self._tolerance = float(cfg.get("tolerance", 0.02))
        if not math.isfinite(self._tolerance) or self._tolerance < 0.0:
            raise ValueError("tonal_anchor tolerance must be finite and >= 0")
        self.data_root = str(cfg.get("data_root") or "")
        self._cache: dict[str, tuple[float, float]] = {}

    def score_batch(self, artifacts: Sequence[Any]) -> list[dict[str, float]]:
        return [self(artifact) for artifact in artifacts]

    def __call__(self, artifact: Any) -> dict[str, float]:
        from vrl.rewards.models.media import decode_artifact_frames

        frames = decode_artifact_frames(artifact, 1)
        luma, sat = tonal_stats(frames.detach().cpu().numpy()[0])
        ref_luma, ref_sat = self._reference_stats(artifact)
        d_luma, d_sat = luma - ref_luma, sat - ref_sat
        penalty = max(0.0, abs(d_luma) - self._tolerance) + max(0.0, abs(d_sat) - self._tolerance)
        return {
            "tonal_anchor": -penalty,
            "tonal_anchor/luma_delta": d_luma,
            "tonal_anchor/saturation_delta": d_sat,
        }

    def _reference_stats(self, artifact: Any) -> tuple[float, float]:
        import numpy as np
        from PIL import Image

        from vrl.utils.artifacts import default_data_root, resolve_artifact_path

        raw = artifact.metadata.get(self._reference_key)
        if not isinstance(raw, str) or not raw.strip():
            raise ValueError(
                f"tonal_anchor needs metadata.{self._reference_key} (a reference image path)"
            )
        path = str(
            resolve_artifact_path(
                raw, data_root=self.data_root or default_data_root(), allow_absolute=True
            )
        )
        if path not in self._cache:
            with Image.open(path) as image:
                arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
            self._cache[path] = tonal_stats(arr)
        return self._cache[path]


__all__ = ["TonalAnchorRewardModel", "tonal_stats"]
