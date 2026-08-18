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

from typing import Any

from vrl.rewards.base import DiskArtifactRewardFunction

_ANIMEREWARD_QUALITY_MODEL = "vrl.rewards.models.animereward_quality:AnimeRewardQualityModel"


class AnimeRewardQualityReward(DiskArtifactRewardFunction):
    """AnimeReward visual quality (Idefics2-8B regression head), roughly [0, 1]."""

    def __init__(
        self,
        *,
        reward_name: str = "animereward_quality",
        score_key: str = "animereward_quality",
        # Anima is an image model: the head's frame window is filled by holding
        # the still, so the artifact is a tensor rather than a video container.
        artifact_format: str = "tensor",
        media_type: str = "image",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model_factory=_ANIMEREWARD_QUALITY_MODEL,
            request_prefix="animereward-quality",
            debug_basename="animereward_quality",
            artifact_format=artifact_format,
            media_type=media_type,
            reward_name=reward_name,
            score_key=score_key,
            **kwargs,
        )


__all__ = ["AnimeRewardQualityReward"]
