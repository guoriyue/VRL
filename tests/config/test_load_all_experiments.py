"""Per SPRINT_config_yaml_unification.md Phase 6 + the patch SPRINT Phase 5:
every experiment YAML must load via ``vrl.config.loader`` and expose the keys
downstream drivers expect; ``build_algorithm_config`` must dispatch only through
``algorithm.kind``.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from omegaconf import OmegaConf

from vrl.algorithms.dpo import DiffusionDPOConfig
from vrl.algorithms.diffusion_nft import DiffusionNFTConfig
from vrl.algorithms.grpo.continuous import GRPOConfig
from vrl.algorithms.grpo.multisegment import MultiSegmentTokenGRPOConfig
from vrl.algorithms.grpo.token import TokenGRPOConfig
from vrl.config.loader import (
    _REWARD_REQUIRED_KWARGS,
    build_algorithm_config,
    build_configs,
    build_reward_config,
    load_config,
    optional_none,
    validate_reward_config,
    validate_training_config,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = REPO_ROOT / "configs" / "experiment"

_EXPECTED_ALGO_TYPE = {
    "grpo": GRPOConfig,
    "token_grpo": TokenGRPOConfig,
    "token_grpo_multisegment": MultiSegmentTokenGRPOConfig,
    "diffusion_dpo": DiffusionDPOConfig,
    "diffusion_nft": DiffusionNFTConfig,
}

_EXPECTED_TRAIN_TARGET = {
    "cosmos_predict2_2b_grpo": "vrl.scripts.cosmos.train:train_cosmos_predict2_grpo",
    "cosmos_predict2_5_2b_diffusionnft": (
        "vrl.scripts.cosmos.train:train_cosmos_predict25_diffusion_nft"
    ),
    "janus_pro_1b_ocr_grpo": "vrl.scripts.janus_pro.train:train_janus_pro_ocr_grpo",
    "janus_pro_1b_r1_ocr_grpo": (
        "vrl.scripts.janus_pro.train:train_janus_pro_r1_ocr_grpo"
    ),
    "janus_pro_1b_r1_codex_qa_grpo": (
        "vrl.scripts.janus_pro.train:train_janus_pro_r1_codex_qa_grpo"
    ),
    "nextstep_1_ocr_grpo": "vrl.scripts.nextstep_1.train:train_nextstep_1_ocr_grpo",
    "sd3_5_ocr_grpo": "vrl.scripts.sd3_5.train:train_sd3_5_grpo",
    "wan_2_1_1_3b_dpo": "vrl.scripts.wan_2_1.train_dpo:train_wan_2_1_dpo",
    "wan_2_1_1_3b_ocr_grpo": "vrl.scripts.wan_2_1.train:train_wan_2_1_grpo",
}


def _experiment_names() -> list[str]:
    return sorted(p.stem for p in EXPERIMENT_DIR.glob("*.yaml"))


@pytest.mark.parametrize("name", _experiment_names())
def test_experiment_yaml_loads(name: str) -> None:
    cfg = load_config(f"experiment/{name}")
    # Every experiment must compose at least these layers via `defaults:`.
    assert "model" in cfg, f"{name} missing model.*"
    assert "trainer" in cfg, f"{name} missing trainer.*"
    assert "algorithm" in cfg, f"{name} missing algorithm.*"
    assert "data" in cfg, f"{name} missing data.* source"

    # model must have a path (required by every family driver).
    assert "path" in cfg.model, f"{name} missing model.path"
    assert "entrypoint" in cfg.trainer, f"{name} missing trainer.entrypoint"
    assert "output_dir" in cfg.trainer, f"{name} missing trainer.output_dir"
    assert "kind" in cfg.algorithm, f"{name} missing algorithm.kind"
    assert "adv_estimator" not in cfg.algorithm, f"{name} still uses algorithm.adv_estimator"


@pytest.mark.parametrize("name", _experiment_names())
def test_build_algorithm_config_dispatch(name: str) -> None:
    cfg = load_config(f"experiment/{name}")
    algo_cfg = build_algorithm_config(cfg)
    expected = _EXPECTED_ALGO_TYPE[str(cfg.algorithm.kind)]
    # Token-GRPO must be the exact subclass, not the parent.
    if expected is GRPOConfig:
        assert type(algo_cfg) is GRPOConfig, (
            f"{name}: expected exact GRPOConfig, got {type(algo_cfg).__name__}"
        )
    else:
        assert isinstance(algo_cfg, expected), (
            f"{name}: expected {expected.__name__}, got {type(algo_cfg).__name__}"
        )


def test_sd3_5_ocr_grpo_uses_continuous_grpo_config() -> None:
    cfg = load_config("experiment/sd3_5_ocr_grpo")
    algo_cfg = build_algorithm_config(cfg)

    assert type(algo_cfg) is GRPOConfig
    assert type(algo_cfg).__module__ == "vrl.algorithms.grpo.continuous"


@pytest.mark.parametrize("name", _experiment_names())
def test_unified_train_entrypoint_reads_yaml_entrypoint(name: str) -> None:
    from vrl.scripts.train import _import_callable, resolve_train_target

    cfg = load_config(f"experiment/{name}")
    target = resolve_train_target(cfg)
    assert target.import_path == _EXPECTED_TRAIN_TARGET[name]
    assert cfg.trainer.entrypoint == _EXPECTED_TRAIN_TARGET[name]
    assert callable(_import_callable(target.import_path))


def test_unified_train_entrypoint_requires_yaml_entrypoint() -> None:
    from vrl.scripts.train import resolve_train_target

    cfg = OmegaConf.create({"trainer": {}})
    with pytest.raises(ValueError, match=r"trainer\.entrypoint"):
        resolve_train_target(cfg)


def test_trainer_resume_from_cli_override_reaches_typed_config() -> None:
    cfg = load_config(
        "experiment/sd3_5_ocr_grpo",
        overrides=["trainer.resume_from=/tmp/checkpoint-10"],
    )
    built = build_configs(cfg)

    assert built["trainer"].resume_from == "/tmp/checkpoint-10"


def test_torch_profiler_cli_overrides_reach_typed_config() -> None:
    cfg = load_config(
        "experiment/sd3_5_ocr_grpo",
        overrides=[
            "trainer.torch_profiler.enabled=true",
            "trainer.torch_profiler.output_dir=/tmp/vrl-profiler",
            "trainer.torch_profiler.activities=[cpu]",
            "trainer.torch_profiler.skip_first=2",
            "trainer.torch_profiler.max_steps=3",
        ],
    )
    built = build_configs(cfg)

    profiler = built["trainer"].torch_profiler
    assert profiler.enabled is True
    assert profiler.output_dir == "/tmp/vrl-profiler"
    assert profiler.activities == ("cpu",)
    assert profiler.skip_first == 2
    assert profiler.max_steps == 3


def test_load_config_supports_defaults_override_for_distributed_preset() -> None:
    cfg = load_config(
        "experiment/sd3_5_ocr_grpo",
        overrides=["/base/distributed=ray_rollout"],
    )

    assert cfg.distributed.resources.allow_overlap is False
    assert cfg.distributed.resources.rollout.num_gpus == "auto"
    assert cfg.distributed.rollout.placement_strategy == "SPREAD"
    assert "base/distributed" not in cfg


def test_unified_train_entrypoint_rejects_empty_yaml_entrypoint() -> None:
    from vrl.scripts.train import resolve_train_target

    cfg = OmegaConf.create({"trainer": {"entrypoint": ""}})
    with pytest.raises(ValueError, match="non-empty"):
        resolve_train_target(cfg)


def test_adv_estimator_is_not_supported() -> None:
    cfg = OmegaConf.create({"algorithm": {"kind": "grpo", "adv_estimator": "dpo"}})
    with pytest.raises(ValueError, match="adv_estimator"):
        build_algorithm_config(cfg)


def test_kind_only_works_without_adv_estimator() -> None:
    cfg = OmegaConf.create({"algorithm": {"kind": "grpo"}})
    out = build_algorithm_config(cfg)
    assert type(out) is GRPOConfig


def test_adv_estimator_only_fails_fast() -> None:
    cfg = OmegaConf.create({"algorithm": {"adv_estimator": "token_grpo"}})
    with pytest.raises(ValueError, match="adv_estimator"):
        build_algorithm_config(cfg)


def test_unknown_kind_fails_fast() -> None:
    cfg = OmegaConf.create({"algorithm": {"kind": "qpo"}})
    with pytest.raises(ValueError, match=r"unknown algorithm\.kind"):
        build_algorithm_config(cfg)


# ---------------------------------------------------------------------------
# SPRINT patch 3 Phase 7: validation gates and grep audits.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", _experiment_names())
def test_validate_training_config_passes_for_all_active_experiments(
    name: str,
) -> None:
    """Every active experiment YAML must pass ``validate_training_config``.

    This is the top-level fail-fast contract: if any required path goes
    missing in the merged config, this gate flags the regression before
    runtime.
    """
    cfg = load_config(f"experiment/{name}")
    validate_training_config(cfg)


def test_aesthetic_kwargs_required() -> None:
    """Removing ``reward.kwargs.aesthetic.model_name`` from an aesthetic
    experiment must fail ``validate_reward_config`` with a clear message.
    """
    cfg = load_config("experiment/cosmos_predict2_2b_grpo")
    # Sanity: the aesthetic component is positive in this experiment.
    assert float(cfg.reward.components.get("aesthetic", 0.0)) > 0
    del cfg.reward.kwargs.aesthetic["model_name"]
    with pytest.raises(ValueError) as excinfo:
        validate_reward_config(cfg)
    msg = str(excinfo.value)
    assert "aesthetic" in msg and "model_name" in msg, msg


def test_pickscore_kwargs_required() -> None:
    """A positive PickScore component must declare model_name explicitly."""
    cfg = OmegaConf.create(
        {
            "reward": {
                "components": {"pickscore": 1.0},
                "kwargs": {"pickscore": {"processor_name": "processor"}},
            },
        },
    )
    with pytest.raises(ValueError) as excinfo:
        validate_reward_config(cfg)
    msg = str(excinfo.value)
    assert "pickscore" in msg and "model_name" in msg, msg


def test_codex_image_qa_kwargs_required() -> None:
    """A positive Codex image-QA component must declare its command explicitly."""
    cfg = OmegaConf.create(
        {
            "reward": {
                "components": {"codex_image_qa": 1.0},
                "kwargs": {"codex_image_qa": {"timeout_s": 30.0}},
            },
        },
    )
    with pytest.raises(ValueError) as excinfo:
        validate_reward_config(cfg)
    msg = str(excinfo.value)
    assert "codex_image_qa" in msg and "command" in msg, msg


@pytest.mark.parametrize("name", _experiment_names())
def test_build_reward_config_returns_explicit_kwargs_for_each_positive_component(
    name: str,
) -> None:
    """For every model-backed reward with weight > 0, ``build_reward_config``
    must surface the required kwargs subkeys explicitly.
    """
    cfg = load_config(f"experiment/{name}")
    if "reward" not in cfg:
        return
    weights, kwargs = build_reward_config(cfg)
    for component, required_subkeys in _REWARD_REQUIRED_KWARGS.items():
        if component not in weights:
            continue
        component_kwargs = kwargs.get(component, {})
        assert isinstance(component_kwargs, dict), (
            f"{name}: kwargs[{component!r}] must be a mapping, got "
            f"{type(component_kwargs).__name__}"
        )
        for subkey in required_subkeys:
            assert subkey in component_kwargs, (
                f"{name}: missing reward.kwargs.{component}.{subkey}"
            )


def test_validate_training_config_fails_when_trainer_output_dir_missing() -> None:
    cfg = load_config("experiment/wan_2_1_1_3b_ocr_grpo")
    del cfg.trainer["output_dir"]
    with pytest.raises(ValueError, match=r"trainer\.output_dir"):
        validate_training_config(cfg)


def test_validate_training_config_fails_when_actor_optim_lr_missing() -> None:
    cfg = load_config("experiment/wan_2_1_1_3b_ocr_grpo")
    del cfg.actor.optim["lr"]
    with pytest.raises(ValueError, match=r"actor\.optim\.lr"):
        validate_training_config(cfg)


def test_validate_training_config_fails_when_nextstep_noise_level_missing() -> None:
    """NextStep-1's continuous-token AR pipeline requires ``rollout.noise_level``."""
    cfg = load_config("experiment/nextstep_1_ocr_grpo")
    del cfg.rollout["noise_level"]
    with pytest.raises(ValueError, match=r"rollout\.noise_level"):
        validate_training_config(cfg)


def test_dpo_allows_explicit_null_max_train_samples() -> None:
    """DPO's ``data.max_train_samples`` legitimately opts in to ``null``.

    ``optional_none`` returns ``None`` only when YAML explicitly sets the
    key to ``null``; ``validate_training_config`` then accepts it.
    """
    cfg = load_config("experiment/wan_2_1_1_3b_dpo")
    cfg.data.max_train_samples = None
    assert optional_none(cfg, "data.max_train_samples") is None
    validate_training_config(cfg)


def test_dpo_fails_when_max_train_steps_missing() -> None:
    cfg = load_config("experiment/wan_2_1_1_3b_dpo")
    del cfg.trainer["max_train_steps"]
    with pytest.raises(ValueError, match=r"trainer\.max_train_steps"):
        validate_training_config(cfg)
