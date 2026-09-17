"""IDM action-following reward entry point for world-model RL.

``ActionFollowingReward`` forwards sample video tensors for scoring with
the trained DROID inverse-dynamics model. ``ModelRewardFunction`` is the
transport capability boundary; this file only pins the IDM model factory and
its defaults. Why the signal exists — and why it must pass the discrimination
gate before driving GRPO — is documented in
``vrl.rewards.models.idm_action_following``.
"""

from __future__ import annotations

from vrl.rewards.base import ModelRewardFunction


class ActionFollowingReward(ModelRewardFunction):
    """Commanded-action agreement scored from video media."""

    model_factory = "vrl.rewards.models.idm_action_following:ActionFollowingIDMModel"
    name = "idm_action_following"
    default_score_key = "action_match"
    default_artifact_format = "mp4"
    default_media_type = "video"


__all__ = ["ActionFollowingReward"]
