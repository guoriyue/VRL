"""Tag adherence reward (WD14 tagger recall, scored in-process on CPU).

Thin function-layer binding registered as ``tag_adherence``. The scoring is
deterministic tagger recall over the prompt's own ``metadata.adherence_tags``
and lives in ``vrl.rewards.models.tag_adherence``. It replaces noisy judge
rewards with a zero-noise, verifiable target on the prompt-following axis.
"""

from __future__ import annotations

from typing import Any

from vrl.rewards.base import InferenceRewardFunction
from vrl.rewards.models.tag_adherence import TagAdherenceRewardModel
from vrl.rewards.runtime import InProcessRewardScorer


class TagAdherenceReward(InferenceRewardFunction):
    """Recall of requested danbooru tags in ``[0, 1]`` from the WD14 tagger."""

    def __init__(self, **kwargs: Any) -> None:
        # onnxruntime CPU compute; ``device`` is accepted for a uniform factory
        # signature but never used.
        kwargs.pop("device", None)
        # Build eagerly so config validation (threshold/metadata_key) fires now.
        model = TagAdherenceRewardModel(kwargs)
        super().__init__(
            reward_name="tag_adherence",
            score_key="tag_adherence",
            scorer=InProcessRewardScorer(model=model),
        )


__all__ = ["TagAdherenceReward"]
