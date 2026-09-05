"""Reward inference deployment config tests."""

from __future__ import annotations

import pytest

from vrl.config.builders import RewardRuntimeConfig
from vrl.config.reward_inference import (
    RewardInferenceConfig,
    parse_reward_inference_config,
)
from vrl.config.schema import RewardConfig


def test_in_process_is_the_default() -> None:
    assert parse_reward_inference_config(None, context="reward.inference.x") == (
        RewardInferenceConfig()
    )


def test_http_requires_endpoint_and_expected_model() -> None:
    with pytest.raises(ValueError, match="absolute http"):
        parse_reward_inference_config(
            {"kind": "http", "expected_model": "judge-v1"},
            context="reward.inference.x",
        )
    with pytest.raises(ValueError, match="expected_model"):
        parse_reward_inference_config(
            {"kind": "http", "endpoint": "http://reward:8300"},
            context="reward.inference.x",
        )


@pytest.mark.parametrize(
    "endpoint",
    [
        "http://user:secret@reward:8300",
        "http://reward:8300/score",
        "http://reward:8300?token=secret",
        "http://reward:8300#fragment",
    ],
)
def test_http_endpoint_is_an_origin_without_embedded_credentials(
    endpoint: str,
) -> None:
    with pytest.raises(ValueError, match="endpoint"):
        parse_reward_inference_config(
            {
                "kind": "http",
                "endpoint": endpoint,
                "expected_model": "unit",
            },
            context="reward.inference.x",
        )


def test_unknown_inference_key_is_rejected_from_typed_source() -> None:
    with pytest.raises(ValueError, match=r"unsupported .* keys"):
        parse_reward_inference_config(
            {"kind": "in_process", "service_url": "http://legacy"},
            context="reward.inference.x",
        )


def test_component_inference_configs_resolve_independently() -> None:
    cfg = {
        "reward": {
            "components": {"ocr": 0.25, "videoscore2": 0.75},
            "inference": {
                "videoscore2": {
                    "kind": "http",
                    "endpoint": "http://reward:8300",
                    "timeout_s": 90,
                    "expected_model": "videoscore2-v1",
                    "expected_model_version": "VideoScore2@unit-revision",
                },
            },
        },
    }

    resolved = RewardRuntimeConfig.from_cfg(
        RewardConfig.model_validate(cfg["reward"])
    ).inference_configs

    assert resolved["ocr"].kind == "in_process"
    assert resolved["videoscore2"] == RewardInferenceConfig(
        kind="http",
        endpoint="http://reward:8300",
        timeout_s=90,
        expected_model="videoscore2-v1",
        expected_model_version="VideoScore2@unit-revision",
    )
