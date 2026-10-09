from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from vrl.algorithms.grpo.continuous import (
    GRPO,
    FlowDPPO,
    GRPOGuard,
)
from vrl.config.builders import RewardRuntimeConfig, build_configs
from vrl.config.loading import load_config
from vrl.config.precision import RolePrecision
from vrl.config.schema import parse_config
from vrl.models.families.registry import get_model_family_entry
from vrl.ray.resources import ResolvedDistributedResources
from vrl.rollouts.collector.config import RolloutCollectorConfig
from vrl.run import ResolvedReward
from vrl.scripts.common.factory import AlgorithmEvaluatorPair


def _built_reward(
    weights: dict[str, float],
    kwargs: dict[str, dict],
    inference: dict[str, dict] | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        reward=RewardRuntimeConfig.from_cfg(
            OmegaConf.create(
                {
                    "reward": {
                        "components": weights,
                        "kwargs": kwargs,
                        "inference": inference or {},
                    },
                },
            ),
        ),
    )


def test_diffusion_grpo_evaluator_uses_resolved_rollout_sde_config() -> None:
    """The evaluator built for a diffusion GRPO run reads noise level and SDE type from the
    resolved rollout config, the collector's denoise options default to native mode, and the
    advantage estimator carries the reward weights.
    """
    cfg = load_config(
        "experiment/wan_2_1/online_grpo_ocr",
        overrides=[
            "rollout.noise_level=0.37",
            "rollout.sde.type=cps",
        ],
    )
    collector_config = RolloutCollectorConfig.from_root(parse_config(cfg))
    built = build_configs(cfg)

    pair = AlgorithmEvaluatorPair.from_configs(
        built,
        scheduler=object(),
    )

    assert pair.evaluator.noise_level == 0.37
    assert pair.evaluator.sde_type == "cps"
    assert collector_config.denoise is not None
    assert collector_config.denoise.denoise_mode == "native"
    assert pair.algorithm.component_weights == built.reward.weights


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
    # A trust-region objective needs a second epoch over the full batch to be
    # a valid run; plain GRPO accepts the same schedule.
    cfg = load_config(
        "experiment/sd3_5/online_grpo_ocr",
        overrides=[
            f"/recipe/online={recipe}",
            "actor.ppo_epochs=2",
            "actor.prompts_per_collection=0",
        ],
    )

    pair = AlgorithmEvaluatorPair.from_configs(
        build_configs(cfg),
        scheduler=object(),
    )

    assert type(pair.algorithm) is expected_algorithm


def test_nft_factory_passes_reward_weights_to_component_advantage_protocol() -> None:
    """A real recipe build uses normalized components rather than the raw total."""
    cfg = load_config(
        "experiment/sd3_5/online_grpo_ocr",
        overrides=[
            "/recipe/online=diffusion_nft",
            "trainer.entrypoint=vrl.scripts.train:train_online",
            "algorithm.advantage_combine=normalized_sum",
            "reward.components.locality_keep=0.3",
        ],
    )
    pair = AlgorithmEvaluatorPair.from_configs(
        build_configs(cfg),
    )
    components = {
        "ocr": torch.tensor([0.0, 1.0, 2.0]),
        "locality_keep": torch.tensor([0.01, 0.03, 0.02]),
    }
    result = pair.algorithm.compute_advantages_from_components(
        components["ocr"] + 0.3 * components["locality_keep"],
        components,
        torch.zeros(3, dtype=torch.long),
    )
    assert pair.evaluator is None
    torch.testing.assert_close(
        result, torch.tensor([-1.5921683, 0.3674235, 1.2247449]), atol=1e-5, rtol=0
    )


def _chunk_config(model_preset: str, *overrides: str):
    """The OCR GRPO recipe on a chunk-autoregressive family.

    The family's model preset overlays the sd3_5 recipe; the sd3_5-only
    ``model.executor`` section has no counterpart on these families.
    """

    cfg = load_config(
        "experiment/sd3_5/online_grpo_ocr", overrides=[f"+model/{model_preset}", *overrides]
    )
    del cfg.model["executor"]
    return cfg


def test_chunk_autoregressive_factory_builds_grouped_grpo_evaluator() -> None:
    cfg = _chunk_config("causvid=wan_1_3b_ar")

    pair = AlgorithmEvaluatorPair.from_configs(
        build_configs(cfg),
    )

    assert type(pair.algorithm) is GRPO
    assert type(pair.evaluator).__name__ == "ChunkAutoregressiveDenoiseLogProbEvaluator"


def test_generation_only_chunk_family_fails_before_algorithm_construction() -> None:
    cfg = _chunk_config("magi_1=4_5b_base")
    # The image recipe's CFG knob has no meaning for Magi's sampler.
    del cfg.sampling["guidance_scale"]

    with pytest.raises(RuntimeError, match=r"generation-only.*no trainable actions"):
        AlgorithmEvaluatorPair.from_configs(
            build_configs(cfg),
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
    # A trust-region objective needs a second epoch over the full batch to be
    # a valid run; plain GRPO accepts the same schedule.
    cfg = _chunk_config(
        "causvid=wan_1_3b_ar",
        f"/recipe/online={recipe}",
        "actor.ppo_epochs=2",
        "actor.prompts_per_collection=0",
    )

    with pytest.raises(ValueError, match=message):
        AlgorithmEvaluatorPair.from_configs(
            build_configs(cfg),
        )


def test_chunk_autoregressive_factory_rejects_non_fp32_transition_math() -> None:
    cfg = _chunk_config("causvid=wan_1_3b_ar", "precision.denoise_math.dtype=bf16")

    with pytest.raises(ValueError, match="exact fp32 Gaussian re-noise"):
        AlgorithmEvaluatorPair.from_configs(
            build_configs(cfg),
        )


def test_chunk_autoregressive_factory_rejects_full_sequence_sft_regularizer() -> None:
    cfg = _chunk_config("causvid=wan_1_3b_ar")
    built = build_configs(cfg)
    built.algorithm.sft_weight = 0.1

    with pytest.raises(ValueError, match=r"grouped causal-chunk replay.*sft_weight"):
        AlgorithmEvaluatorPair.from_configs(
            built,
        )


def test_sana_aesthetic_keeps_cpu_observation_only_pickscore() -> None:
    """PickScore is logged on CPU but contributes zero optimization weight."""
    cfg = load_config("experiment/sana/online_grpo_aesthetic")
    cfg.distributed.resources.visible_devices = [0]
    built = build_configs(cfg)
    from vrl.rewards.ray import RayRewardPlacement

    reward = ResolvedReward.from_plan(
        built.reward,
        ResolvedDistributedResources.from_root(parse_config(cfg)),
        trainer_device="cuda:0",
    ).build_function(ray_placement=RayRewardPlacement(shared_gpu_id=0, node_id="driver"))

    assert [(name, weight) for name, weight, _ in reward.rewards] == [
        ("aesthetic", 1.0),
        ("pickscore", 0.0),
    ]
    pickscore = reward.rewards[1][2]
    # Online rewards run as Ray actors; the resolved device travels in
    # the actor's worker_config.
    assert pickscore.scorer._launch.device == "cpu"


def test_sana_family_defaults_to_native_fp16() -> None:
    """The public role precision matches the checkpoint's runtime invariant."""
    cfg = load_config("experiment/sana/online_grpo_aesthetic")
    built = build_configs(cfg)
    entry = get_model_family_entry("sana")
    build = entry.resolve_model_build(
        built.root,
        torch.device("cpu"),
        precision=built.precision,
    )

    assert cfg.model.get("dtype") is None
    expected = RolePrecision(
        dtype="fp16",
        float32_precision="ieee",
        outer_autocast=False,
    )
    assert built.precision.stages_match
    assert built.root.rollout is not None
    assert (
        built.trainer.batch_plan.training_microbatch_size
        == built.root.rollout.samples_per_generation_batch
    )
    assert build.parameter_dtype is torch.float16
    assert build.precision == expected
    assert (
        entry.resolve_model_build(
            built.root,
            torch.device("cpu"),
            precision=built.precision,
            for_rollout=False,
        ).precision
        == expected
    )
    assert build.rollout is not None
    assert build.rollout.prompt_encoder_dtype is torch.bfloat16


@pytest.mark.parametrize(("role", "dtype"), [("training", "bf16"), ("rollout", "fp32")])
def test_sana_role_precision_follows_yaml(role: str, dtype: str) -> None:
    """The selected YAML role owns SANA's transformer execution policy."""
    cfg = load_config("experiment/sana/online_grpo_aesthetic")
    cfg.precision[role].dtype = dtype
    built = build_configs(cfg)

    build = get_model_family_entry("sana").resolve_model_build(
        built.root,
        torch.device("cpu"),
        precision=built.precision,
        for_rollout=role == "rollout",
    )

    assert build.precision == RolePrecision(
        dtype=dtype,
        float32_precision="ieee",
        outer_autocast=False,
    )
    assert build.parameter_dtype is getattr(torch, {"bf16": "bfloat16", "fp32": "float32"}[dtype])


def test_sana_fullparam_long_is_fresh_and_pins_reward_revisions() -> None:
    """The canonical curve starts from base with immutable scorer identities."""
    cfg = load_config("experiment/sana/online_grpo_aesthetic_fullparam_long")
    cfg.distributed.resources.visible_devices = [0]
    built = build_configs(cfg)
    from vrl.rewards.ray import RayRewardPlacement

    assert built.resume.checkpoint_path is None
    assert cfg.model.use_lora is False
    reward = ResolvedReward.from_plan(
        built.reward,
        ResolvedDistributedResources.from_root(parse_config(cfg)),
        trainer_device="cuda:0",
    ).build_function(ray_placement=RayRewardPlacement(shared_gpu_id=0, node_id="driver"))
    aesthetic_config = reward.rewards[0][2].scorer._launch.component_config
    pickscore_config = reward.rewards[1][2].scorer._launch.component_config
    assert aesthetic_config["model_revision"] == cfg.reward.kwargs.aesthetic.model_revision
    assert pickscore_config["device"] == "cpu"
    assert pickscore_config["processor_revision"] == cfg.reward.kwargs.pickscore.processor_revision
    assert pickscore_config["model_revision"] == cfg.reward.kwargs.pickscore.model_revision


def test_sana_rejects_redundant_or_conflicting_model_dtype() -> None:
    """A family invariant must not also survive as a user-controlled knob."""
    cfg = load_config("experiment/sana/online_grpo_aesthetic")
    cfg.model.dtype = "fp16"

    with pytest.raises(ValueError, match=r"unknown model\.dtype"):
        parse_config(cfg)


def test_sana_direct_tool_override_changes_storage_only() -> None:
    cfg = load_config("experiment/sana/online_grpo_aesthetic")
    built = build_configs(cfg)

    build = get_model_family_entry("sana").resolve_model_build(
        built.root,
        torch.device("cpu"),
        precision=built.precision,
        parameter_dtype_override="fp32",
    )

    assert build.parameter_dtype is torch.float32
    assert build.precision == RolePrecision(
        dtype="fp16",
        float32_precision="ieee",
        outer_autocast=False,
    )


def test_reward_factory_rejects_an_all_zero_objective() -> None:
    """Checks observation-only components cannot replace the training objective."""
    with pytest.raises(ValueError, match="At least one reward component"):
        ResolvedReward(
            config=_built_reward({"pickscore": 0.0}, {}).reward,
            device="cpu",
            memory_parking_required=False,
        ).build_function()


def _shared_reward_cfg(component: str) -> object:
    return OmegaConf.create(
        {
            "distributed": {
                "resources": {
                    "visible_devices": [0],
                    "trainer": {"devices": [0]},
                    "rollout": {"devices": [0]},
                },
            },
            "reward": {"components": {component: 1.0}, "kwargs": {component: {}}},
        },
    )


@pytest.mark.parametrize(
    "component_kwargs",
    [{"sleep_offload": True}, {"worker_config": {"sleep_offload": False}}],
)
def test_reward_config_rejects_yaml_lifecycle_override(component_kwargs) -> None:
    """Resource topology is the only public reward lifecycle source."""

    cfg = OmegaConf.create(
        {
            "reward": {
                "components": {"aesthetic": 1.0},
                "kwargs": {"aesthetic": component_kwargs},
            },
        },
    )

    with pytest.raises(ValueError, match="sleep_offload is topology-derived"):
        RewardRuntimeConfig.from_cfg(cfg)


def test_reward_inputs_derive_device_from_resource_topology() -> None:
    """The resource plan, not a caller device, is the execution-device source."""
    cfg = _shared_reward_cfg("aesthetic")
    shared = ResolvedReward.from_plan(
        _built_reward({"aesthetic": 1.0}, {"aesthetic": {}}).reward,
        ResolvedDistributedResources.from_root(parse_config(cfg)),
    )
    assert shared.device == "cuda:0"
    assert shared.memory_parking_required is True

    # A torchrun rank scores on logical cuda:0 while Ray keeps the physical ID.
    rank_local = _shared_reward_cfg("aesthetic")
    rank_local.distributed.resources.visible_devices = [2]
    rank_local.distributed.resources.trainer.devices = [2]
    rank_local.distributed.resources.rollout.devices = [2]
    rank_local_reward = ResolvedReward.from_plan(
        _built_reward({"aesthetic": 1.0}, {"aesthetic": {}}).reward,
        ResolvedDistributedResources.from_root(parse_config(rank_local)),
        trainer_device="cuda:0",
    )
    assert rank_local_reward.device == "cuda:0"
    assert rank_local_reward.memory_parking_required is True

    # HTTP components own their deployment externally: the plan resolved from
    # the same config reserves them no local GPU, so nothing parks.
    http_inference = {
        "unified_reward_video": {
            "kind": "http",
            "endpoint": "http://127.0.0.1:8300",
            "expected_model": "unified-reward-robotics",
        },
    }
    http_cfg = _shared_reward_cfg("unified_reward_video")
    http_cfg.distributed.resources.visible_devices = [2]
    http_cfg.distributed.resources.trainer.devices = [2]
    http_cfg.distributed.resources.rollout.devices = [2]
    http_cfg.reward.kwargs = {}
    http_cfg.reward.inference = http_inference
    http_reward = ResolvedReward.from_plan(
        _built_reward({"unified_reward_video": 1.0}, {}, http_inference).reward,
        ResolvedDistributedResources.from_root(parse_config(http_cfg)),
        trainer_device="cuda:0",
    )
    # The HTTP client runs in this process and reserves no local GPU.
    assert http_reward.device == "cpu"
    assert http_reward.memory_parking_required is False

    cpu_cfg = OmegaConf.create(
        {
            "distributed": {
                "resources": {
                    "visible_devices": [0, 1],
                    "trainer": {"devices": [0]},
                    "rollout": {"devices": [1]},
                },
            },
            "reward": {"components": {"ocr": 1.0}, "kwargs": {"ocr": {}}},
        },
    )
    # A CPU-only reward class scores on CPU even when the trainer runs on CUDA.
    cpu_reward = ResolvedReward.from_plan(
        _built_reward({"ocr": 1.0}, {"ocr": {}}).reward,
        ResolvedDistributedResources.from_root(parse_config(cpu_cfg)),
        trainer_device="cuda:0",
    )
    assert cpu_reward.device == "cpu"


def test_multi_gpu_engine_gate_requires_family_capability() -> None:
    """gpus_per_engine > 1 fails loud for a family without an installer."""
    from vrl.models.families.registry import get_model_family_entry

    wan = get_model_family_entry("wan_2_1")
    with pytest.raises(ValueError, match=r"wan_2_1.*gpus_per_engine=2"):
        wan.validate_gpus_per_engine(2)

    # Single-GPU engines never consult the capability; a capable family passes.
    wan.validate_gpus_per_engine(1)
    get_model_family_entry("sd3_5").validate_gpus_per_engine(2)


def test_ray_rewards_do_not_materialize_transport_files_in_the_run_output(tmp_path) -> None:
    """A run-owned scorer transfers media without making an artifact directory."""
    from vrl.rewards.artifacts import InMemoryRewardArtifactStore
    from vrl.rewards.ray import RayRewardScorer

    cfg = load_config(
        "experiment/sd3_5/online_grpo_ocr",
        overrides=[
            f"trainer.output_dir={tmp_path}/run",
            "reward.inference.ocr.kind=ray",
        ],
    )
    cfg.distributed.resources.visible_devices = [0]
    built = build_configs(cfg)
    reward = ResolvedReward.from_plan(
        built.reward,
        ResolvedDistributedResources.from_root(parse_config(cfg)),
        trainer_device="cuda:0",
    ).build_function()
    component = reward.rewards[0][2]
    assert isinstance(component.scorer, RayRewardScorer)
    assert isinstance(component.artifact_store, InMemoryRewardArtifactStore)
    assert not (tmp_path / "run" / "reward_artifacts").exists()


def test_online_reward_transport_is_rejected_before_resource_resolution(monkeypatch):
    from vrl.run import resolve_online_run

    cfg = load_config(
        "experiment/sana/online_grpo_pickscore_pickapic_sfw",
        overrides=["reward.inference.pickscore.kind=in_process"],
    )

    def unexpected_resources(*args, **kwargs):
        raise AssertionError("invalid reward deployment reached resource resolution")

    monkeypatch.setattr(ResolvedDistributedResources, "from_root", unexpected_resources)
    with pytest.raises(ValueError, match="in_process is not admitted for online training"):
        resolve_online_run(cfg)
    # Offline diagnostics can still resolve an in-process, zero-weight observer.
    offline = _built_reward(
        {"image_sharpness": 0}, {}, {"image_sharpness": {"kind": "in_process"}}
    )
    with pytest.raises(ValueError, match="weight > 0"):
        offline.reward.require_online_training()
