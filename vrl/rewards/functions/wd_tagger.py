"""WD tagger reward binding for requested general-tag adherence, on CPU.

``score_key`` selects the reading that drives training. ``wd_tagger_dense`` is
the default: the thresholded ``wd_tagger_recall`` takes at most a handful of
values inside a prompt group, which measured as unable to move this policy,
while the dense reading moved it (SPRINT_anima_geneval_spatial_rl 7.5, 8.2).
"""

from __future__ import annotations

from typing import Any

from vrl.rewards.base import InferenceRewardFunction
from vrl.rewards.models.wd_tagger import WDTaggerRewardModel
from vrl.rewards.runtime import InProcessRewardScorer

WD_TAGGER_SCORE_KEYS = ("wd_tagger_dense", "wd_tagger_recall")


class WDTaggerReward(InferenceRewardFunction):
    """Requested general-tag adherence in ``[0, 1]`` from the WD tagger."""

    def __init__(self, score_key: str = "wd_tagger_dense", **kwargs: Any) -> None:
        if score_key not in WD_TAGGER_SCORE_KEYS:
            raise ValueError(
                f"wd_tagger score_key must be one of {list(WD_TAGGER_SCORE_KEYS)}, got {score_key!r}"
            )
        # onnxruntime CPU compute; ``device`` is accepted for a uniform factory
        # signature but never used.
        kwargs.pop("device", None)
        # Build eagerly so config validation (threshold/metadata_key) fires now.
        model = WDTaggerRewardModel(kwargs)
        super().__init__(
            reward_name="wd_tagger",
            score_key=score_key,
            scorer=InProcessRewardScorer(model=model),
        )


__all__ = ["WD_TAGGER_SCORE_KEYS", "WDTaggerReward"]
