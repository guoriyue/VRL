"""Model-free image statistics as a controllable GRPO probe reward.

This exists to answer "is the training pipeline working?" separately from "is the
reward model any good?" -- two questions that were entangled for ten runs of
SPRINT_anima_rl_5090 because every reward on the shelf ranks partly on style
within a group of equally-competent images.

A probe reward must satisfy three properties that no preference model does:

1. **Known ground truth.** The target IS the statistic, so there is no question
   of whether the reward "means" what we think. Success is checkable with numpy
   and with the eye at the same time.
2. **Dense within-group signal.** Measured on a real 16-sample rollout group,
   saturation gives mean|advantage| = 0.803 with zero degenerate samples --
   the same scale a healthy preference reward produces, unlike a thresholded
   classifier which is either all-zero or binary.
3. **Provably controllable.** Run 9's policy moved mean saturation +6.3% while
   chasing AnimeReward, so the policy demonstrably CAN steer this quantity.
   A probe the policy cannot control would test nothing.

``direction: -1`` inverts the objective, which is the point: a pipeline that
genuinely learns must be able to drive the statistic BOTH ways. One direction
moving could be drift; both directions following the sign is proof.

NOT a quality reward, and not a step toward one. Optimizing saturation produces a
garish model -- that is expected and is why this is a probe, to be deleted once
the pipeline question is settled.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from vrl.rewards.models.base import TorchRewardModel
from vrl.utils.media import to_uint8

_STATISTICS = ("saturation", "edge_energy", "brightness")


class ImageStatisticRewardModel(TorchRewardModel):
    """Score an image by a plain pixel statistic. No network, no weights."""

    def __init__(self, worker_config: Mapping[str, Any]) -> None:
        super().__init__(worker_config)
        statistic = str(self.worker_config.get("statistic", "saturation"))
        if statistic not in _STATISTICS:
            raise ValueError(
                f"image_statistic.statistic must be one of {_STATISTICS}; got {statistic!r}",
            )
        self.statistic = statistic
        direction = float(self.worker_config.get("direction", 1.0))
        if direction not in (1.0, -1.0):
            raise ValueError(
                f"image_statistic.direction must be +1 (maximize) or -1 (minimize); "
                f"got {direction!r}",
            )
        self.direction = direction

    def _load_module(self) -> Any:
        # No weights to load; the "module" is a marker so the shared lazy-load
        # plumbing in TorchRewardModel has something to hold.
        return object()

    def score_media(self, *, media: Any, prompt: str) -> Mapping[str, float]:
        del prompt  # a pixel statistic is prompt-independent by construction
        import numpy as np
        import torch
        from PIL import Image

        if isinstance(media, torch.Tensor):
            arr = to_uint8(media).cpu().numpy()
            if arr.ndim == 3:
                arr = arr[None]
            if arr.ndim == 4:
                arr = arr.transpose(0, 2, 3, 1)  # NCHW -> NHWC
            elif arr.ndim == 5:
                mid = arr.shape[1] // 2
                arr = arr[:, mid].transpose(0, 2, 3, 1)
            frames = [a for a in arr]
        elif isinstance(media, np.ndarray):
            frames = [media] if media.ndim == 3 else list(media)
        elif isinstance(media, Image.Image):
            frames = [np.asarray(media.convert("RGB"), dtype=np.uint8)]
        else:
            return {"image_statistic": 0.0}

        values = [self._statistic_value(np.asarray(f, dtype=np.float32) / 255.0) for f in frames]
        value = float(sum(values) / max(len(values), 1))
        return {"image_statistic": self.direction * value}

    def _statistic_value(self, image: Any) -> float:
        import numpy as np

        if self.statistic == "saturation":
            return float((image.max(axis=2) - image.min(axis=2)).mean())
        gray = image.mean(axis=2)
        if self.statistic == "brightness":
            return float(gray.mean())
        # edge_energy: mean absolute first difference along both axes, a plain
        # proxy for line/detail density.
        return float(np.abs(np.diff(gray, axis=1)).mean() + np.abs(np.diff(gray, axis=0)).mean())


__all__ = ["ImageStatisticRewardModel"]
