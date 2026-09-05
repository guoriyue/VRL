"""WD tagger reward binding for requested general-tag recall, on CPU."""

from __future__ import annotations

from typing import Any

from vrl.rewards.base import InferenceRewardFunction
from vrl.rewards.models.wd_tagger import WDTaggerRewardModel
from vrl.rewards.runtime import InProcessRewardScorer


class WDTaggerReward(InferenceRewardFunction):
    """Recall of requested general tags in ``[0, 1]`` from the WD tagger."""

    def __init__(self, **kwargs: Any) -> None:
        # onnxruntime CPU compute; ``device`` is accepted for a uniform factory
        # signature but never used.
        kwargs.pop("device", None)
        # Build eagerly so config validation (threshold/metadata_key) fires now.
        model = WDTaggerRewardModel(kwargs)
        super().__init__(
            reward_name="wd_tagger",
            score_key="wd_tagger",
            scorer=InProcessRewardScorer(model=model),
        )


__all__ = ["WDTaggerReward"]
