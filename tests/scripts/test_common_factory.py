from __future__ import annotations

import pytest
import torch
from omegaconf import OmegaConf

from vrl.algorithms.grpo.continuous import (
    GRPO,
    FlowDPPO,
    FlowDPPOConfig,
    GRPOConfig,
    GRPOGuard,
)
from vrl.config.builders import build_configs
from vrl.config.loading import load_config
from vrl.models.diffusion.build import extract_family_runtime_spec
from vrl.scripts.common.factory import (
    build_algorithm_and_evaluator_from_cfg,
    build_reward_from_cfg,
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


@pytest.mark.parametrize(
    ("recipe", "expected_algorithm"),
    [
        ("flow_matching_grpo", GRPO),
        ("flow_matching_dppo", FlowDPPO),
        ("flow_matching_grpo_guard", GRPOGuard),
    ],
)
def test_diffusion_factory_accepts_each_kind_exact_config_type(
    recipe: str,
    expected_algorithm: type,
) -> None:
    cfg = load_config(
        "experiment/diffusion/sd3_5/online_grpo_ocr",
        overrides=[f"/recipe/online={recipe}"],
    )

    pair = build_algorithm_and_evaluator_from_cfg(
        cfg,
        family="sd3_5",
        built=build_configs(cfg),
        collector_config=build_rollout_config_from_cfg(cfg, "sd3_5"),
        scheduler=object(),
    )

    assert type(pair.algorithm) is expected_algorithm


@pytest.mark.parametrize(
    ("recipe", "wrong_config", "expected_name"),
    [
        ("flow_matching_dppo", GRPOConfig(), "FlowDPPOConfig"),
        ("flow_matching_grpo_guard", GRPOConfig(), "GRPOGuardConfig"),
        ("flow_matching_grpo", FlowDPPOConfig(), "GRPOConfig"),
    ],
)
def test_diffusion_factory_rejects_a_sibling_config_type(
    recipe: str,
    wrong_config: object,
    expected_name: str,
) -> None:
    cfg = load_config(
        "experiment/diffusion/sd3_5/online_grpo_ocr",
        overrides=[f"/recipe/online={recipe}"],
    )

    with pytest.raises(TypeError, match=expected_name):
        build_algorithm_and_evaluator_from_cfg(
            cfg,
            family="sd3_5",
            built={"algorithm": wrong_config},
            scheduler=object(),
        )


def test_wan_empty_lora_preserves_base_policy_initially() -> None:
    """Checks Wan empty LoRA preserves base policy initially."""
    cfg = load_config("experiment/diffusion/wan_2_1/online_grpo_physics")

    spec = extract_family_runtime_spec(cfg, torch.device("cpu"), torch.bfloat16)

    assert spec.use_lora is True
    lora_config = spec.lora
    assert lora_config is not None
    # Wan's apply_lora reads ``init_lora_weights`` with a True default, so an
    # empty training adapter still initially preserves base Wan output.
    assert lora_config.get("init_lora_weights", True) is True


def test_sana_aesthetic_builds_zero_weight_pickscore_observer() -> None:
    """Checks the SANA objective scores PickScore without optimizing it."""
    cfg = load_config("experiment/diffusion/sana/online_grpo_aesthetic")
    built = build_configs(cfg)

    reward = build_reward_from_cfg(cfg, built=built, device="cpu")

    assert [(name, weight) for name, weight, _ in reward.rewards] == [
        ("aesthetic", 1.0),
        ("pickscore", 0.0),
    ]


def test_reward_factory_passes_the_selected_local_device(monkeypatch) -> None:
    """The resolved reward device reaches every component factory unchanged."""
    from vrl.rewards.functions.registry import MultiReward

    captured: dict[str, object] = {}
    sentinel = object()

    def fake_from_dict(
        cls,
        score_dict,
        device="cuda",
        reward_kwargs=None,
    ):
        del cls
        captured.update(
            score_dict=dict(score_dict),
            device=device,
            reward_kwargs=dict(reward_kwargs or {}),
        )
        return sentinel

    monkeypatch.setattr(MultiReward, "from_dict", classmethod(fake_from_dict))

    reward = build_reward_from_cfg(
        OmegaConf.create({}),
        built={"reward": ({"fake": 1.0}, {"fake": {"marker": True}})},
        device="cuda:2",
    )

    assert reward is sentinel
    assert captured == {
        "score_dict": {"fake": 1.0},
        "device": "cuda:2",
        "reward_kwargs": {"fake": {"marker": True}},
    }


def test_reward_factory_rejects_an_all_zero_objective() -> None:
    """Checks observation-only components cannot replace the training objective."""
    with pytest.raises(ValueError, match="At least one reward component"):
        build_reward_from_cfg(
            OmegaConf.create({}),
            built={"reward": ({"pickscore": 0.0}, {})},
            device="cpu",
        )
