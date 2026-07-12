"""Migration guard for the removed inline fixed-evaluation surface."""

from __future__ import annotations

import pytest

from vrl.config.builders import build_configs
from vrl.config.loading import load_config


def test_trainer_eval_points_to_standalone_checkpoint_evaluation() -> None:
    cfg = load_config(
        "experiment/diffusion/sd3_5/online_grpo_ocr",
        overrides=["trainer.eval.enabled=true"],
    )

    with pytest.raises(ValueError, match=r"trainer\.eval was removed"):
        build_configs(cfg)
