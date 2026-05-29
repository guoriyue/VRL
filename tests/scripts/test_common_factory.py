from __future__ import annotations

import torch

from vrl.config.builders import build_configs
from vrl.config.loading import load_config
from vrl.models.diffusion.wan_2_1.runtime import extract_wan_2_1_runtime_spec
from vrl.scripts.common.factory import (
    build_algorithm_and_evaluator_from_cfg,
    build_rollout_config_from_cfg,
)


def test_diffusion_grpo_evaluator_uses_resolved_rollout_sde_config() -> None:
    cfg = load_config(
        "experiment/diffusion/wan_2_1/online_grpo_ocr",
        overrides=[
            "rollout.noise_level=0.37",
            "rollout.sde.type=cps",
        ],
    )
    collector_config = build_rollout_config_from_cfg(cfg, "wan_2_1")

    pair = build_algorithm_and_evaluator_from_cfg(
        cfg,
        family="wan_2_1",
        built=build_configs(cfg),
        collector_config=collector_config,
        scheduler=object(),
    )

    assert pair.evaluator.noise_level == 0.37
    assert pair.evaluator.sde_type == "cps"
    assert collector_config.values["denoise_mode"] == "native"


def test_wan_empty_lora_preserves_base_policy_initially() -> None:
    cfg = load_config("experiment/diffusion/wan_2_1/online_grpo_physics")

    spec = extract_wan_2_1_runtime_spec(cfg, torch.device("cpu"), torch.bfloat16)

    assert spec.use_lora is True
    assert spec.lora_config is not None
    assert spec.lora_config["init_lora_weights"] is True
