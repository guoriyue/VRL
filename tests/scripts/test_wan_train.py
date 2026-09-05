from __future__ import annotations

import pytest

from vrl.config.schema import RootConfig
from vrl.scripts.common.online import _preflight_production_video_reward


def test_production_preflight_fails_when_inference_code_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Checks production preflight fails when inference code missing."""

    def _raise() -> None:
        raise ImportError("missing Kling inference backend")

    monkeypatch.setattr(
        "vrl.rewards.models.kling_video_reward.preflight_kling_video_reward_backend",
        _raise,
    )
    root = RootConfig.model_validate({"production": {"kling_video_reward": {"enabled": True}}})

    with pytest.raises(RuntimeError, match="repo-owned Kling VideoReward inference backend"):
        _preflight_production_video_reward(root)


def test_production_preflight_skipped_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Checks production preflight skipped when disabled."""

    def _raise() -> None:
        raise ImportError("missing Kling inference backend")

    monkeypatch.setattr(
        "vrl.rewards.models.kling_video_reward.preflight_kling_video_reward_backend",
        _raise,
    )
    root = RootConfig.model_validate({"production": {"kling_video_reward": {"enabled": False}}})

    _preflight_production_video_reward(root)
