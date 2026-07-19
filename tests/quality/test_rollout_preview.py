"""Opt-in few-shot generation from a real RL experiment config."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.quality.preview import generate_rollout_preview
from vrl.config.loading import load_config


@pytest.mark.gpu
@pytest.mark.rollout_preview
def test_generate_rollout_preview(request: pytest.FixtureRequest) -> None:
    config_name = request.config.getoption("--rollout-preview-config")
    output_dir = request.config.getoption("--rollout-preview-dir")
    if not config_name or not output_dir:
        pytest.skip(
            "pass --rollout-preview-config and --rollout-preview-dir to generate images",
        )

    preview = generate_rollout_preview(
        load_config(config_name),
        Path(output_dir),
        config_name=str(config_name),
    )

    assert preview["items"]
    assert all(
        (Path(output_dir).expanduser().resolve() / item["file"]).is_file()
        for item in preview["items"]
    )
