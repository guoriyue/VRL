from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

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
from vrl.config.builders import RewardRuntimeConfig, build_configs
from vrl.config.loading import load_config
from vrl.config.precision import RolePrecision
from vrl.families.registry import get_model_family_entry
from vrl.ray.resources import resolve_distributed_resources
from vrl.rollouts.collector.config import build_rollout_config_from_cfg
from vrl.scripts.common.factory import (
    build_algorithm_and_evaluator_from_cfg,
    build_reward,
    validate_reward_memory_parking,
)
from vrl.scripts.common.online import OnlineRunConfig


def _built_reward(
    weights: dict[str, float],
    kwargs: dict[str, dict],
) -> SimpleNamespace:
    return SimpleNamespace(
        reward=RewardRuntimeConfig(weights=weights, kwargs=kwargs),
    )


def test_diffusion_grpo_evaluator_uses_resolved_rollout_sde_config() -> None:
    """Checks diffusion GRPO evaluator uses resolved rollout SDE config."""
    cfg = load_config(
        "experiment/wan_2_1/online_grpo_ocr",
        overrides=[
            "rollout.noise_level=0.37",
            "rollout.sde.type=cps",
        ],
    )
    collector_config = build_rollout_config_from_cfg(cfg)

    pair = build_algorithm_and_evaluator_from_cfg(
        cfg,
        family_entry=get_model_family_entry("wan_2_1"),
        built=build_configs(cfg),
        collector_config=collector_config,
        scheduler=object(),
    )

    assert pair.evaluator.noise_level == 0.37
    assert pair.evaluator.sde_type == "cps"
    assert collector_config.request_sampling["denoise_mode"] == "native"


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
        "experiment/sd3_5/online_grpo_ocr",
        overrides=[f"/recipe/online={recipe}"],
    )

    pair = build_algorithm_and_evaluator_from_cfg(
        cfg,
        family_entry=get_model_family_entry("sd3_5"),
        built=build_configs(cfg),
        collector_config=build_rollout_config_from_cfg(cfg),
        scheduler=object(),
    )

    assert type(pair.algorithm) is expected_algorithm


def test_chunk_autoregressive_factory_builds_grouped_grpo_evaluator() -> None:
    cfg = load_config("experiment/sd3_5/online_grpo_ocr")

    pair = build_algorithm_and_evaluator_from_cfg(
        cfg,
        family_entry=get_model_family_entry("causvid"),
        built=build_configs(cfg),
        collector_config=build_rollout_config_from_cfg(cfg),
    )

    assert type(pair.algorithm) is GRPO
    assert type(pair.evaluator).__name__ == "ChunkAutoregressiveDenoiseLogProbEvaluator"


def test_generation_only_chunk_family_fails_before_algorithm_construction() -> None:
    cfg = load_config("experiment/sd3_5/online_grpo_ocr")

    with pytest.raises(RuntimeError, match=r"generation-only.*no trainable actions"):
        build_algorithm_and_evaluator_from_cfg(
            cfg,
            family_entry=get_model_family_entry("magi_1"),
            built=build_configs(cfg),
            collector_config=build_rollout_config_from_cfg(cfg),
        )


@pytest.mark.parametrize(
    ("recipe", "message"),
    [
        ("flow_matching_dance_grpo", "random denoise-timestep subset"),
        ("flow_matching_dppo", "reverse-SDE dt signals"),
        ("flow_matching_grpo_guard", "reverse-SDE dt signals"),
    ],
)
def test_chunk_autoregressive_factory_rejects_undefined_algorithm_semantics(
    recipe: str,
    message: str,
) -> None:
    cfg = load_config(
        "experiment/sd3_5/online_grpo_ocr",
        overrides=[f"/recipe/online={recipe}"],
    )

    with pytest.raises(ValueError, match=message):
        build_algorithm_and_evaluator_from_cfg(
            cfg,
            family_entry=get_model_family_entry("causvid"),
            built=build_configs(cfg),
            collector_config=build_rollout_config_from_cfg(cfg),
        )


def test_chunk_autoregressive_factory_rejects_non_fp32_transition_math() -> None:
    cfg = load_config(
        "experiment/sd3_5/online_grpo_ocr",
        overrides=["precision.diffusion_math.dtype=bf16"],
    )

    with pytest.raises(ValueError, match="exact fp32 Gaussian re-noise"):
        build_algorithm_and_evaluator_from_cfg(
            cfg,
            family_entry=get_model_family_entry("causvid"),
            built=build_configs(cfg),
            collector_config=build_rollout_config_from_cfg(cfg),
        )


def test_chunk_autoregressive_factory_rejects_full_sequence_sft_regularizer() -> None:
    cfg = load_config("experiment/sd3_5/online_grpo_ocr")
    built = build_configs(cfg)
    built.algorithm.sft_weight = 0.1

    with pytest.raises(ValueError, match=r"grouped causal-chunk replay.*sft_weight"):
        build_algorithm_and_evaluator_from_cfg(
            cfg,
            family_entry=get_model_family_entry("causvid"),
            built=built,
            collector_config=build_rollout_config_from_cfg(cfg),
        )


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
        "experiment/sd3_5/online_grpo_ocr",
        overrides=[f"/recipe/online={recipe}"],
    )
    built = build_configs(cfg)
    built = replace(built, algorithm=wrong_config)

    with pytest.raises(TypeError, match=expected_name):
        build_algorithm_and_evaluator_from_cfg(
            cfg,
            family_entry=get_model_family_entry("sd3_5"),
            built=built,
            collector_config=build_rollout_config_from_cfg(cfg),
            scheduler=object(),
        )


def test_wan_empty_lora_preserves_base_policy_initially() -> None:
    """Checks Wan empty LoRA preserves base policy initially."""
    cfg = load_config("experiment/wan_2_1/online_grpo_physics")

    build = get_model_family_entry("wan_2_1").resolve_model_build(
        cfg,
        torch.device("cpu"),
    )

    assert build.use_lora is True
    lora_config = build.lora
    assert lora_config is not None
    # Wan's apply_lora reads ``init_lora_weights`` with a True default, so an
    # empty training adapter still initially preserves base Wan output.
    assert lora_config.get("init_lora_weights", True) is True


def test_sana_aesthetic_keeps_cpu_observation_only_pickscore() -> None:
    """PickScore is logged on CPU but contributes zero optimization weight."""
    cfg = load_config("experiment/sana/online_grpo_aesthetic")
    cfg.distributed.resources.visible_devices = [0]
    built = build_configs(cfg)

    reward = build_reward(
        built=built,
        resources=resolve_distributed_resources(cfg),
        device="cuda:0",
    )

    assert [(name, weight) for name, weight, _ in reward.rewards] == [
        ("aesthetic", 1.0),
        ("pickscore", 0.0),
    ]
    pickscore = reward.rewards[1][2]
    assert pickscore.runtime._worker_config["device"] == "cpu"


def test_sana_family_defaults_to_native_fp16() -> None:
    """The public role precision matches the checkpoint's runtime invariant."""
    cfg = load_config("experiment/sana/online_grpo_aesthetic")
    built = build_configs(cfg)
    entry = get_model_family_entry("sana")
    build = entry.resolve_model_build(cfg, torch.device("cpu"))

    assert cfg.model.get("dtype") is None
    expected = RolePrecision(
        dtype="fp16",
        float32_precision="ieee",
        outer_autocast=False,
    )
    assert built.trainer.train_precision == built.trainer.rollout_precision
    assert built.root.rollout is not None
    assert (
        built.trainer.batch_plan.replay_samples_per_chunk == built.root.rollout.samples_per_chunk
    )
    assert build.parameter_dtype is torch.float16
    assert build.precision == expected
    assert (
        entry.resolve_model_build(cfg, torch.device("cpu"), for_rollout=False).precision
        == expected
    )
    assert build.rollout is not None
    assert build.rollout.prompt_encoder_dtype is torch.bfloat16


@pytest.mark.parametrize("role", ["training", "rollout"])
@pytest.mark.parametrize("dtype", ["bf16", "fp32"])
def test_sana_role_precision_follows_yaml(role: str, dtype: str) -> None:
    """The selected YAML role owns SANA's transformer execution policy."""
    cfg = load_config("experiment/sana/online_grpo_aesthetic")
    cfg.precision[role].dtype = dtype

    build = get_model_family_entry("sana").resolve_model_build(
        cfg,
        torch.device("cpu"),
        for_rollout=role == "rollout",
    )

    assert build.precision == RolePrecision(
        dtype=dtype,
        float32_precision="ieee",
        outer_autocast=False,
    )
    assert build.parameter_dtype is getattr(torch, {"bf16": "bfloat16", "fp32": "float32"}[dtype])


def test_sana_fullparam_pilot_disables_tf32_and_gates_backend_drift() -> None:
    """The strict first update must not consume clipping on backend mismatch."""
    cfg = load_config("experiment/sana/online_grpo_aesthetic_fullparam")
    built = build_configs(cfg)
    trainer = built.trainer
    run = OnlineRunConfig.from_root(built.root)

    assert cfg.model.use_lora is False
    assert cfg.precision.float32_precision == "ieee"
    assert cfg.model.lora is None
    assert cfg.algorithm.kl_coef == 0.0
    assert trainer.optim.optim_8bit is True
    assert trainer.ema.enable is False
    assert trainer.ppo_epochs == 1
    assert trainer.batch_plan.gradient_accumulation_steps == 1
    assert trainer.batch_plan.prompts_per_batch == 1
    assert trainer.batch_plan.n_samples_per_prompt == 8
    assert built.root.rollout is not None
    assert built.root.rollout.samples_per_chunk == 1
    assert trainer.batch_plan.replay_samples_per_chunk == 1
    assert run.total_epochs == 5
    assert run.save_freq == 1
    assert trainer.debug.first_step is True
    assert trainer.debug.max_abs_logprob_diff == pytest.approx(1.0e-6)
    assert trainer.precision_drift_guard.mode == "fail"
    assert trainer.precision_drift_guard.max_abs_log_ratio == pytest.approx(1.0e-6)
    assert trainer.precision_drift_guard.max_ratio_abs_dev == pytest.approx(1.0e-6)


def test_sana_fullparam_long_is_fresh_and_pins_reward_revisions() -> None:
    """The canonical curve starts from base with immutable scorer identities."""
    cfg = load_config("experiment/sana/online_grpo_aesthetic_fullparam_long")
    cfg.distributed.resources.visible_devices = [0]
    built = build_configs(cfg)
    trainer = built.trainer
    run = OnlineRunConfig.from_root(built.root)

    assert built.resume.checkpoint_path is None
    assert built.resume.strict is True
    assert run.total_epochs == 300
    assert run.save_freq == 5
    assert trainer.output_dir == "outputs/sana_aesthetic_fullparam_long"
    assert cfg.model.use_lora is False
    assert cfg.reward.kwargs.aesthetic.model_revision == (
        "32bd64288804d66eefd0ccbe215aa642df71cc41"
    )
    assert cfg.reward.kwargs.pickscore.processor_revision == (
        "1c2b8495b28150b8a4922ee1c8edee224c284c0c"
    )
    assert cfg.reward.kwargs.pickscore.model_revision == (
        "a4e4367c6dfa7288a00c550414478f865b875800"
    )
    reward = build_reward(
        built=built,
        resources=resolve_distributed_resources(cfg),
        device="cuda:0",
    )
    aesthetic_config = reward.rewards[0][2].runtime._worker_config
    pickscore_config = reward.rewards[1][2].runtime._worker_config
    assert aesthetic_config["model_revision"] == ("32bd64288804d66eefd0ccbe215aa642df71cc41")
    assert pickscore_config["device"] == "cpu"
    assert pickscore_config["processor_revision"] == ("1c2b8495b28150b8a4922ee1c8edee224c284c0c")
    assert pickscore_config["model_revision"] == ("a4e4367c6dfa7288a00c550414478f865b875800")


@pytest.mark.parametrize(
    "experiment",
    [
        "experiment/sana/online_grpo_aesthetic",
        "experiment/sana/online_grpo_pickscore_validation",
    ],
)
def test_sana_experiments_pin_the_validated_symmetric_chunk_shape(
    experiment: str,
) -> None:
    """The measured 8/8 shape is a parity and memory contract, not a tuning default."""
    built = build_configs(load_config(experiment))

    assert built.root.rollout is not None
    assert built.root.rollout.samples_per_chunk == 8
    assert built.trainer.batch_plan.replay_samples_per_chunk == 8


@pytest.mark.parametrize("configured_dtype", ["fp16", "bf16"])
def test_sana_rejects_redundant_or_conflicting_model_dtype(
    configured_dtype: str,
) -> None:
    """A family invariant must not also survive as a user-controlled knob."""
    cfg = load_config("experiment/sana/online_grpo_aesthetic")
    cfg.model.dtype = configured_dtype

    with pytest.raises(ValueError, match=r"model\.dtype is not configurable.*sana"):
        get_model_family_entry("sana").resolve_model_build(
            cfg,
            torch.device("cpu"),
        )


def test_ordinary_diffusion_family_rejects_duplicate_model_dtype() -> None:
    cfg = load_config("experiment/sd3_5/online_grpo_ocr")
    cfg.model.dtype = "fp16"

    with pytest.raises(ValueError, match=r"model\.dtype.*top-level precision"):
        get_model_family_entry("sd3_5").resolve_model_build(
            cfg,
            torch.device("cpu"),
        )


@pytest.mark.parametrize("dtype", ["bf16", "fp32"])
def test_sana_direct_tool_override_changes_storage_only(dtype: str) -> None:
    cfg = load_config("experiment/sana/online_grpo_aesthetic")

    build = get_model_family_entry("sana").resolve_model_build(
        cfg,
        torch.device("cpu"),
        parameter_dtype_override=dtype,
    )

    assert build.parameter_dtype is getattr(torch, {"bf16": "bfloat16", "fp32": "float32"}[dtype])
    assert build.precision == RolePrecision(
        dtype="fp16",
        float32_precision="ieee",
        outer_autocast=False,
    )


def test_token_objective_rejects_unused_math_precision_override() -> None:
    cfg = load_config("experiment/emu3/online_grpo_pickscore_validation")
    cfg.precision = {
        "float32_precision": "tf32",
        "training": {"dtype": "bf16", "outer_autocast": False},
        "rollout": {"dtype": "bf16", "outer_autocast": False},
        "diffusion_math": {"dtype": "bf16"},
    }
    built = build_configs(cfg)

    with pytest.raises(ValueError, match=r"precision\.diffusion_math\.dtype.*diffusion log-prob"):
        build_algorithm_and_evaluator_from_cfg(
            cfg,
            built=built,
            family_entry=get_model_family_entry("emu3"),
            collector_config=build_rollout_config_from_cfg(cfg),
        )


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
        memory_parking_required=None,
    ):
        del cls
        captured.update(
            score_dict=dict(score_dict),
            device=device,
            reward_kwargs=dict(reward_kwargs or {}),
            memory_parking_required=memory_parking_required,
        )
        return sentinel

    monkeypatch.setattr(MultiReward, "from_dict", classmethod(fake_from_dict))

    reward = build_reward(
        built=_built_reward({"fake": 1.0}, {"fake": {"marker": True}}),
        resources=None,
        device="cuda:2",
    )

    assert reward is sentinel
    assert captured == {
        "score_dict": {"fake": 1.0},
        "device": "cuda:2",
        "reward_kwargs": {"fake": {"marker": True}},
        "memory_parking_required": None,
    }


def test_reward_factory_rejects_an_all_zero_objective() -> None:
    """Checks observation-only components cannot replace the training objective."""
    with pytest.raises(ValueError, match="At least one reward component"):
        build_reward(
            built=_built_reward({"pickscore": 0.0}, {}),
            resources=None,
            device="cpu",
        )


def _shared_reward_cfg(component: str) -> object:
    return OmegaConf.create(
        {
            "distributed": {
                "resources": {
                    "visible_devices": [0],
                    "trainer": {"devices": [0]},
                    "rollout": {
                        "devices": [0],
                        "gpus_per_worker": 1,
                        "gpu_pool": "trainer",
                    },
                },
            },
            "reward": {"components": {component: 1.0}, "kwargs": {component: {}}},
        },
    )


def test_shared_reward_capability_fails_before_component_construction(monkeypatch) -> None:
    """An unsupported trainer-shared reward fails before its model constructor."""
    from vrl.rewards.functions.geneval import GenEvalReward

    constructed = False

    def fail_if_constructed(self, *args, **kwargs):
        del self, args, kwargs
        nonlocal constructed
        constructed = True
        raise AssertionError("component construction must not run")

    monkeypatch.setattr(GenEvalReward, "__init__", fail_if_constructed)
    cfg = _shared_reward_cfg("geneval")

    with pytest.raises(ValueError, match="geneval"):
        build_reward(
            built=_built_reward({"geneval": 1.0}, {"geneval": {}}),
            resources=resolve_distributed_resources(cfg),
            device="cuda:0",
        )

    assert constructed is False


def test_shared_reward_topology_automatically_enables_parking(monkeypatch) -> None:
    """The lifecycle flag, not a YAML sleep knob, drives runtime parking."""
    from vrl.rewards.functions.registry import MultiReward

    captured: dict[str, object] = {}
    sentinel = object()

    def fake_from_dict(
        cls,
        score_dict,
        device="cuda",
        reward_kwargs=None,
        memory_parking_required=None,
    ):
        del cls
        captured.update(
            score_dict=dict(score_dict),
            device=device,
            reward_kwargs=dict(reward_kwargs or {}),
            memory_parking_required=memory_parking_required,
        )
        return sentinel

    monkeypatch.setattr(MultiReward, "from_dict", classmethod(fake_from_dict))
    cfg = _shared_reward_cfg("aesthetic")

    reward = build_reward(
        built=_built_reward({"aesthetic": 1.0}, {"aesthetic": {}}),
        resources=resolve_distributed_resources(cfg),
        device="cuda:0",
    )

    assert reward is sentinel
    assert captured["memory_parking_required"] is True


def test_reward_preflight_rejects_yaml_lifecycle_override() -> None:
    """Resource topology is the only public reward lifecycle source."""

    cfg = _shared_reward_cfg("aesthetic")
    built = _built_reward(
        {"aesthetic": 1.0},
        {"aesthetic": {"sleep_offload": True}},
    )
    with pytest.raises(ValueError, match="sleep_offload is topology-derived"):
        validate_reward_memory_parking(
            resources=resolve_distributed_resources(cfg),
            built=built,
        )

    with pytest.raises(ValueError, match="sleep_offload is topology-derived"):
        build_reward(built=built, resources=None, device="cuda:0")


def test_factory_rejects_driver_device_outside_reward_resource_topology() -> None:
    """The factory cannot replace resolved CUDA ownership with a CPU runtime."""
    cfg = _shared_reward_cfg("aesthetic")

    with pytest.raises(ValueError, match="execution-device source of truth"):
        build_reward(
            built=_built_reward({"aesthetic": 1.0}, {"aesthetic": {}}),
            resources=resolve_distributed_resources(cfg),
            device="cpu",
        )

    cpu_cfg = OmegaConf.create(
        {
            "distributed": {
                "resources": {
                    "visible_devices": [0, 1],
                    "trainer": {"devices": [0]},
                    "rollout": {"devices": [1], "gpus_per_worker": 1},
                    "reward": {
                        "devices": [],
                        "num_gpus": 0,
                        "gpus_per_worker": 0,
                        "num_workers": 1,
                    },
                },
            },
            "reward": {"components": {"ocr": 1.0}, "kwargs": {"ocr": {}}},
        },
    )
    with pytest.raises(ValueError, match="execution-device source of truth"):
        build_reward(
            built=_built_reward({"ocr": 1.0}, {"ocr": {}}),
            resources=resolve_distributed_resources(cpu_cfg),
            device="cuda:0",
        )
