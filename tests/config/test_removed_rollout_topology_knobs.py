"""Migration guards for removed rollout-topology knobs."""

from __future__ import annotations

import pytest

from vrl.config.builders import build_configs
from vrl.config.loading import load_config


def test_require_separate_gpus_points_to_topology_derived_modes() -> None:
    cfg = load_config(
        "experiment/sd3_5/online_grpo_ocr",
        overrides=["trainer.rollout_orchestration.require_separate_gpus=false"],
    )

    with pytest.raises(
        ValueError,
        match=r"unknown config keys.*require_separate_gpus",
    ):
        build_configs(cfg)
