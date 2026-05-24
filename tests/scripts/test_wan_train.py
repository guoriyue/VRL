from __future__ import annotations

import pytest
from omegaconf import OmegaConf

from vrl.scripts.common.online import _preflight_production_video_reward


def test_production_preflight_fails_when_inference_code_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _raise() -> None:
        raise ImportError("missing VideoAlign")

    monkeypatch.setattr(
        "vrl.rewards.models.kling_video_reward.preflight_kling_video_reward_backend",
        _raise,
    )
    cfg = OmegaConf.create(
        {
            "production": {"video_reward": {"enabled": True}},
        },
    )

    with pytest.raises(RuntimeError, match="VideoAlign inference code"):
        _preflight_production_video_reward(cfg)


def test_production_preflight_skipped_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _raise() -> None:
        raise ImportError("missing VideoAlign")

    monkeypatch.setattr(
        "vrl.rewards.models.kling_video_reward.preflight_kling_video_reward_backend",
        _raise,
    )
    cfg = OmegaConf.create({"production": {"video_reward": {"enabled": False}}})

    _preflight_production_video_reward(cfg)
