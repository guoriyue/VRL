"""IDM action-following reward entry point for world-model RL.

``ActionFollowingReward`` writes each sample's video to disk and scores it with
the trained DROID inverse-dynamics model. ``DiskArtifactRewardFunction`` is the
transport capability boundary; this file only pins the IDM model factory and
its defaults. Why the signal exists — and why it must pass the discrimination
gate before driving GRPO — is documented in
``vrl.rewards.models.action_following``.
"""

from __future__ import annotations

from typing import Any

from vrl.rewards.base import DiskArtifactRewardFunction

_ACTION_FOLLOWING_MODEL = "vrl.rewards.models.action_following:ActionFollowingIDMModel"


class ActionFollowingReward(DiskArtifactRewardFunction):
    """Commanded-action agreement scored from disk video artifacts."""

    def __init__(
        self,
        *,
        reward_name: str = "idm_action_following",
        score_key: str = "action_match",
        artifact_format: str = "mp4",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model_factory=_ACTION_FOLLOWING_MODEL,
            request_prefix="idm-action-following",
            debug_basename="idm_action_following",
            artifact_format=artifact_format,
            reward_name=reward_name,
            score_key=score_key,
            **kwargs,
        )


__all__ = ["ActionFollowingReward"]
