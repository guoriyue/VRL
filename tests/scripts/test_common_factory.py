from __future__ import annotations

import torch
from omegaconf import OmegaConf

from vrl.config.builders import build_configs
from vrl.config.loading import load_config
from vrl.models.diffusion.wan_2_1.runtime import extract_wan_2_1_runtime_spec
from vrl.scripts.common.factory import (
    _with_resolved_reward_runtime_kwargs,
    build_algorithm_and_evaluator_from_cfg,
    build_rollout_config_from_cfg,
)


def test_diffusion_grpo_evaluator_uses_resolved_rollout_sde_config() -> None:
    """Checks diffusion GRPO evaluator uses resolved rollout SDE config."""
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
    """Checks Wan empty LoRA preserves base policy initially."""
    cfg = load_config("experiment/diffusion/wan_2_1/online_grpo_physics")

    spec = extract_wan_2_1_runtime_spec(cfg, torch.device("cpu"), torch.bfloat16)

    assert spec.use_lora is True
    lora_config = spec.lora
    assert lora_config is not None
    # Wan's apply_lora reads ``init_lora_weights`` with a True default, so an
    # empty training adapter still initially preserves base Wan output.
    assert lora_config.get("init_lora_weights", True) is True


def test_reward_runtime_kwargs_apply_to_any_active_ray_reward() -> None:
    """Checks generic Ray reward kwargs are derived from distributed resources."""
    cfg = OmegaConf.create(
        {
            "distributed": {
                "resources": {
                    "visible_devices": [0],
                    "trainer": {"num_gpus": 0},
                    "rollout": {
                        "num_gpus": 0,
                        "gpus_per_worker": 0,
                        "num_workers": 1,
                    },
                    "reward": {
                        "num_gpus": 1,
                        "gpus_per_worker": 1,
                        "num_workers": 1,
                    },
                },
                # release_after_score derives to true for a multi-reward pool.
                "reward": {
                    "cpus_per_worker": 0.5,
                    "max_inflight_batches": 2,
                    "placement_strategy": "STRICT_PACK",
                },
            },
            "reward": {
                "components": {
                    "custom_gpu_reward": 1.0,
                    "second_gpu_reward": 1.0,
                    "ocr": 1.0,
                },
                "kwargs": {
                    "custom_gpu_reward": {"execution": "pool"},
                    "second_gpu_reward": {"execution": "pool"},
                    "ocr": {"debug_dir": "ocr-debug"},
                },
            },
        },
    )
    placement = object()

    reward_kwargs = _with_resolved_reward_runtime_kwargs(
        cfg,
        {"custom_gpu_reward": 1.0, "second_gpu_reward": 1.0, "ocr": 1.0},
        {
            "custom_gpu_reward": {"execution": "pool"},
            "second_gpu_reward": {"execution": "pool"},
            "ocr": {"debug_dir": "ocr-debug"},
        },
        reward_placement=placement,
    )

    for reward_key in ("custom_gpu_reward", "second_gpu_reward"):
        custom = reward_kwargs[reward_key]
        assert custom["num_workers"] == 1
        assert custom["gpus_per_worker"] == 1.0
        assert custom["expected_gpu_ids"] == (0,)
        assert custom["release_after_score"] is True
        assert custom["max_inflight_batches"] == 2
        assert custom["placement_strategy"] == "STRICT_PACK"
        assert custom["placement"] is placement
    assert reward_kwargs["ocr"] == {"debug_dir": "ocr-debug"}
