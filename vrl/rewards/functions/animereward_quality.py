"""AnimeReward visual-quality reward entry point for anime image RL.

``AnimeRewardQualityReward`` writes each sample's media to disk and scores it
through the configured in-process or HTTP runtime. ``DiskArtifactRewardFunction``
is the transport capability boundary; this file only pins the AnimeReward
quality model factory and its defaults.

The head is Mantis's Idefics2 fork, which targets transformers 4.x while this
repo runs 5.x, so in practice it is driven over HTTP against the standalone
reward service (``/reward/animereward_quality_http``) rather than colocated.
"""

from __future__ import annotations

from vrl.rewards.base import DiskArtifactRewardFunction


class AnimeRewardQualityReward(DiskArtifactRewardFunction):
    """AnimeReward visual quality (Idefics2-8B regression head), roughly [0, 1]."""

    model_factory = "vrl.rewards.models.animereward_quality:AnimeRewardQualityModel"
    request_prefix = "animereward-quality"
    debug_basename = "animereward_quality"
    default_reward_name = "animereward_quality"
    default_score_key = "animereward_quality"
    default_artifact_format = "tensor"
    default_media_type = "image"


__all__ = ["AnimeRewardQualityReward"]
