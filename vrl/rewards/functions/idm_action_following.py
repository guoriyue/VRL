"""IDM action-following reward entry point for world-model RL.

``ActionFollowingReward`` writes each sample's video to disk and scores it with
the trained DROID inverse-dynamics model. ``DiskArtifactRewardFunction`` is the
transport capability boundary; this file only pins the IDM model factory and
its defaults. Why the signal exists — and why it must pass the discrimination
gate before driving GRPO — is documented in
``vrl.rewards.models.idm_action_following``.
"""

from __future__ import annotations

from vrl.rewards.base import DiskArtifactRewardFunction


class ActionFollowingReward(DiskArtifactRewardFunction):
    """Commanded-action agreement scored from disk video artifacts."""

    model_factory = "vrl.rewards.models.idm_action_following:ActionFollowingIDMModel"
    request_prefix = "idm-action-following"
    debug_basename = "idm_action_following"
    default_reward_name = "idm_action_following"
    default_score_key = "action_match"
    default_artifact_format = "mp4"
    default_media_type = "video"


__all__ = ["ActionFollowingReward"]
