"""Line-art sharpness reward (model-free, scored in-process on CPU).

Thin function-layer binding registered as ``image_sharpness``. The scoring is a
normalized Laplacian energy over frame luma — no network, no GPU — and lives in
``vrl.rewards.models.image_sharpness``. It defends crisp cel-shaded line art,
which every learned quality model tried on this policy failed to capture; pair
it with a coherence reward (PickScore) so high-frequency noise cannot game it.
"""

from __future__ import annotations

from typing import Any

from vrl.rewards.base import InferenceRewardFunction
from vrl.rewards.models.image_sharpness import ImageSharpnessRewardModel
from vrl.rewards.runtime import InProcessRewardScorer


class ImageSharpnessReward(InferenceRewardFunction):
    """Crisp line-art score in ``[0, 1]`` from Laplacian edge energy."""

    def __init__(self, **kwargs: Any) -> None:
        # Model-free CPU compute; ``device`` is accepted for a uniform factory
        # signature but never used.
        kwargs.pop("device", None)
        model = ImageSharpnessRewardModel(kwargs)
        super().__init__(
            reward_name="image_sharpness",
            score_key="image_sharpness",
            scorer=InProcessRewardScorer(model=model),
        )


__all__ = ["ImageSharpnessReward"]
