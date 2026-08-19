"""Model-free image-statistic probe reward (in-process transport).

See ``vrl.rewards.models.image_statistic`` for why this exists: a controllable
ground-truth signal that separates "is the training pipeline working?" from
"is the reward model any good?".
"""

from __future__ import annotations

from typing import Any

from vrl.rewards.base import InferenceRewardFunction
from vrl.rewards.models.image_statistic import ImageStatisticRewardModel
from vrl.rewards.runtime import InProcessRewardScorer


class ImageStatisticReward(InferenceRewardFunction):
    """Pixel-statistic probe; ``statistic`` and ``direction`` ride the kwargs."""

    def __init__(
        self,
        device: str = "cuda",
        **kwargs: Any,
    ) -> None:
        # Build eagerly so config validation (statistic/direction) fires now.
        model = ImageStatisticRewardModel({"device": device, **kwargs})
        super().__init__(
            reward_name="image_statistic",
            score_key="image_statistic",
            scorer=InProcessRewardScorer(model=model),
        )


__all__ = ["ImageStatisticReward"]
