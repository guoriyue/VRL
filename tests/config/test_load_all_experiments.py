"""Core config-loader safety tests.

This file intentionally keeps broad experiment coverage in loop-style tests
instead of parametrizing the same assertion into dozens of collected tests.
"""

from __future__ import annotations

import contextlib
import re
import subprocess
import sys
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from vrl.config.builders import (
    BuiltConfigs,
    RewardRuntimeConfig,
)
from vrl.config.loading import (
    bundled_config_resource,
    list_bundled_configs,
    load_config,
)
from vrl.config.schema import RewardConfig, parse_config
from vrl.config.validation import require_training_config
from vrl.ray.resources import ResourcePlan
from vrl.trainers.data import prompts


def _validate_reward_parking(reward, resources) -> None:
    """The online recipe's pre-build check: a reward the plan parks must be able to."""

    if resources.reward_parking_mechanism is not None:
        reward.validate_memory_parking(resources.reward_torch_device())


def _is_offline(kind: str) -> bool:
    from vrl.algorithms.base import OfflineAlgorithm
    from vrl.config.algorithm import algorithm_class

    return issubclass(algorithm_class(kind), OfflineAlgorithm)


def _experiment_names() -> list[str]:
    return [
        name.removeprefix("experiment/").removesuffix(".yaml")
        for name in list_bundled_configs("experiment")
    ]


def _load_bundled_raw(name: str):
    resource = bundled_config_resource(name)
    with resource.open("r", encoding="utf-8") as stream:
        return OmegaConf.load(stream)


def _load_experiment_for_static_validation(name: str):
    """Complete runtime templates with inert test choices."""

    return load_config(f"experiment/{name}")


def test_load_config_enforces_mandatory_marker(tmp_path: Path) -> None:
    """Keys declared '???' must be set by the experiment or a dotlist override."""
    config = tmp_path / "exp.yaml"
    config.write_text("trainer:\n  entrypoint: ???\n  seed: 0\n")

    with pytest.raises(ValueError, match=r"trainer\.entrypoint"):
        load_config(config)

    cfg = load_config(config, overrides=["trainer.entrypoint=pkg.mod:fn"])
    assert cfg.trainer.entrypoint == "pkg.mod:fn"


def test_trainer_config_from_cfg_reports_all_missing_required_keys() -> None:
    """Missing required keys are reported together, each with its full YAML path."""
    cfg = OmegaConf.create(
        {
            "actor": {},
            "trainer": {},
            "rollout": {},
            "precision": {
                "float32_precision": "tf32",
                "training": {"dtype": "bf16"},
                "rollout": {"dtype": "bf16"},
            },
        },
    )

    from vrl.trainers.online.config import TrainerConfig

    with pytest.raises(ValueError) as exc:
        TrainerConfig.from_root(parse_config(cfg))

    message = str(exc.value)
    for path in (
        "actor.optim.lr",
        "actor.drop_zero_advantage",
        "actor.timestep_fraction",
        "trainer.output_dir",
        "rollout.prompts_per_batch",
        "rollout.n_samples_per_prompt",
    ):
        assert path in message


def test_config_groups_are_not_flattened() -> None:
    """Bundled config groups keep their nested layout: no experiment / model / sampling entry sits
    flat at the group root, and the retired ``task`` and ``profiling`` groups are empty.
    """
    flattened = [
        name
        for group in ("experiment", "model", "sampling")
        for name in list_bundled_configs(group)
        if len(Path(name).parts) == 2
    ]
    task_configs = list_bundled_configs("task")

    assert flattened == []
    assert task_configs == ()
    assert list_bundled_configs("profiling") == ()


def test_geometry_sampling_presets_do_not_own_text_encoder_lengths() -> None:
    geometry_presets = [
        name
        for name in list_bundled_configs("sampling")
        if Path(name).parts[1] in {"image", "video"}
    ]

    assert geometry_presets
    for name in geometry_presets:
        sampling = _load_bundled_raw(name).get("sampling", {})
        assert "max_sequence_length" not in sampling, name


def test_raw_yaml_has_no_user_specific_absolute_paths() -> None:
    """Committed configs must not depend on one contributor's home directory."""
    user_home = re.compile(
        r"(?:/home/[^/\s\"'{}]+/|/Users/[^/\s\"'{}]+/|/root/|"
        r"[A-Za-z]:[\\/]Users[\\/][^\\/\s\"'{}]+[\\/])",
    )
    offenders = []
    for logical_name in list_bundled_configs():
        resource = bundled_config_resource(logical_name)
        with resource.open("r", encoding="utf-8") as stream:
            lines = stream.read().splitlines()
        for line_number, line in enumerate(lines, start=1):
            if user_home.search(line):
                offenders.append(
                    f"vrl/config/presets/{logical_name}:{line_number}: {line.strip()}",
                )

    assert offenders == []


def test_experiments_are_grouped_by_model_family() -> None:
    """Every experiment lives at ``<family>/<recipe>``.

    Generation regime and policy-step math live in registry semantics, so
    the config path must not reintroduce the retired AR/diffusion taxonomy.
    """
    names = _experiment_names()
    groups = {Path(name).parts[0] for name in names}
    assert {"ar", "diffusion"}.isdisjoint(groups)
    assert all(len(Path(name).parts) == 2 for name in names)


def _reward_group_kwargs_keys(group_name: str) -> dict[str, set[str]]:
    """Component-name -> leaf-key set provided by a `/reward/<group>` default.

    This is the source of truth the inline-override rule is derived from, so the
    test never hand-maintains a parallel allowlist that rots when a reward group
    gains a knob.
    """
    raw = _load_bundled_raw(f"reward/{group_name}")
    kwargs = raw.get("reward", {}).get("kwargs", {}) or {}
    return {comp: set((sub or {}).keys()) for comp, sub in kwargs.items()}


def test_experiments_use_dataset_groups_and_only_override_reward_weights() -> None:
    """Experiments compose dataset groups and only fine-tune reward leaves inline.

    Two structural rules, both derived from the source of truth instead of a
    hand-kept path allowlist:

    * no experiment inlines a `data:` block (datasets come from `/dataset/...`);
    * an experiment's inline `reward.kwargs` may only override leaf scalars its
      reward group default already declared — it may NOT introduce a new
      component or a key the group never provided.
    """
    inline_data = []
    inline_reward_violations = []
    for rel in list_bundled_configs("experiment"):
        raw = _load_bundled_raw(rel)
        if "data" in raw:
            inline_data.append(rel)

        reward = raw.get("reward", None)
        if reward is None or "kwargs" not in reward:
            continue

        # Keys (per component) the imported reward groups already provide.
        provided: dict[str, set[str]] = {}
        for default in raw.get("defaults", []) or []:
            if isinstance(default, str) and default.startswith("/reward/"):
                group = default.split("/reward/", 1)[1]
                for comp, keys in _reward_group_kwargs_keys(group).items():
                    provided.setdefault(comp, set()).update(keys)

        for comp, sub in (reward.get("kwargs", {}) or {}).items():
            if comp not in provided:
                inline_reward_violations.append(f"{rel}: new component {comp!r}")
                continue
            extra = set((sub or {}).keys()) - provided[comp]
            if extra:
                inline_reward_violations.append(
                    f"{rel}: {comp} declares non-default keys {sorted(extra)}",
                )

    assert inline_data == []
    assert inline_reward_violations == []


def test_reward_configs_are_single_reward_building_blocks() -> None:
    """Every bundled ``reward/`` preset defines exactly one component, so presets compose by
    addition.
    """
    offenders = []
    for name in list_bundled_configs("reward"):
        raw = _load_bundled_raw(name)
        components = raw.get("reward", {}).get("components", {})
        if len(components) != 1:
            offenders.append(name)

    assert offenders == []


def test_all_experiments_load_and_validate() -> None:
    """Every bundled experiment loads, carries the required top-level sections and keys, has
    migrated off ``adv_estimator``, and its raw ``model`` block survives typed parsing key-for-
    key.
    """
    for name in _experiment_names():
        cfg = _load_experiment_for_static_validation(name)
        assert "model" in cfg, f"{name} missing model.*"
        assert "trainer" in cfg, f"{name} missing trainer.*"
        assert "algorithm" in cfg, f"{name} missing algorithm.*"
        assert "data" in cfg, f"{name} missing data.* source"
        if not _is_offline(str(cfg.algorithm.kind)):
            assert "reward" in cfg, f"{name} missing reward.* source"
        assert "path" in cfg.model, f"{name} missing model.path"
        assert "entrypoint" in cfg.trainer, f"{name} missing trainer.entrypoint"
        assert "output_dir" in cfg.trainer, f"{name} missing trainer.output_dir"
        assert "kind" in cfg.algorithm, f"{name} missing algorithm.kind"
        assert "adv_estimator" not in cfg.algorithm, f"{name} still uses adv_estimator"
        root, _ = require_training_config(cfg)
        raw_model = OmegaConf.to_container(cfg.model, resolve=True)
        typed_model = root.model
        assert isinstance(raw_model, dict)
        assert typed_model is not None
        typed_payload = typed_model.model_dump(exclude_unset=True)
        assert {key: typed_payload[key] for key in raw_model} == raw_model


class _ManifestRead(Exception):
    """The online prompt loader accepted the data section and went to read files."""


def _stop_at_manifest_read(*_args, **_kwargs):
    raise _ManifestRead


def test_all_online_experiments_pass_static_launch_preflight(monkeypatch) -> None:
    """Every active online recipe has a valid schedule and reward topology, a
    data section the online prompt loader accepts, and ships no rollout
    precision split by accident.

    The prompt loader runs for real up to its first file read (several
    manifests live outside the repository), so its loader dispatch is the one
    judging the data section.

    Unpinned roles share the trainer GPU, so a recipe needs exactly the ordinals
    its pinned ``devices`` name (one synthetic device when nothing is pinned:
    ``make verify`` hides CUDA, while this static test must still exercise
    single-GPU parking). Cross-node recipes size their own budget from counts.
    """

    def pinned_devices(node) -> list[int]:
        devices = node.get("devices", "auto")
        return [] if devices == "auto" else [int(device) for device in devices]

    monkeypatch.setattr(prompts, "load_prompt_dataset_index", _stop_at_manifest_read)
    monkeypatch.setattr(prompts.ImageCaptionPromptDataset, "from_section", _stop_at_manifest_read)
    failures = []
    for name in _experiment_names():
        cfg = _load_experiment_for_static_validation(name)
        if _is_offline(str(cfg.algorithm.kind)):
            continue

        if OmegaConf.select(cfg, "distributed.resources") is None:
            OmegaConf.update(cfg, "distributed.resources", {}, merge=True)
        resources_cfg = cfg.distributed.resources
        if resources_cfg.get("visible_devices", "auto") == "auto" and not resources_cfg.get(
            "cross_node", False
        ):
            pinned = [
                device
                for role in ("trainer", "rollout", "reward")
                for device in pinned_devices(resources_cfg.get(role) or {})
            ]
            resources_cfg.visible_devices = list(range(max(pinned, default=0) + 1))

        try:
            built = BuiltConfigs.from_cfg(cfg)
            root = parse_config(cfg)
            resources = ResourcePlan.from_root(root)
            resources.validate_schedule_mode(built.trainer.rollout_orchestration.schedule_mode)
            _validate_reward_parking(built.reward, resources)
            with contextlib.suppress(_ManifestRead):
                prompts.load_prompt_examples_from_config(root.data)
        except Exception as error:  # report the full active surface in one failure
            failures.append(f"{name}: {type(error).__name__}: {error}")
            continue
        if built.precision.training != built.precision.rollout:
            failures.append(f"{name}: training and rollout precision differ")
        if built.precision.denoise_math != "fp32":
            failures.append(f"{name}: denoise_math is {built.precision.denoise_math!r}")

    assert failures == []


def test_rollout_orchestration_group_override_uses_rollout_namespace() -> None:
    """``/base/rollout/orchestration=continuous`` lands under ``trainer.rollout_orchestration``
    and types into a continuous schedule with a positive staleness bound.
    """
    cfg = load_config(
        "experiment/sd3_5/online_grpo_ocr",
        overrides=["/base/rollout/orchestration=continuous"],
    )

    orchestration = cfg.trainer.rollout_orchestration
    assert orchestration.schedule_mode == "continuous"
    from vrl.trainers.core.types import RolloutOrchestrationConfig

    typed = RolloutOrchestrationConfig(
        **OmegaConf.to_container(orchestration, resolve=True),
    )
    assert typed.schedule_mode == "continuous"
    assert typed.continuous.max_stale_policy_versions >= 1


# Every recipe below also passes test_all_online_experiments_pass_static_launch_preflight
# (schedule mode and reward parking); these pin the topology each one resolves to.


def test_sd35_continuous_4gpu_acceptance_resolves_disjoint_resident_topology() -> None:
    """The reusable hardware gate must preserve the topology it validates."""

    cfg = load_config("experiment/sd3_5/online_grpo_ocr_continuous_4gpu_acceptance")
    resources = ResourcePlan.from_root(parse_config(cfg))

    assert resources.trainer_devices == (0,)
    assert resources.rollout_devices == (1, 2, 3)
    assert resources.reward_devices == ()
    assert resources.rollout_num_engines == 3
    assert resources.lifecycle.rollout_mode == "resident"
    assert not any(
        (
            resources.lifecycle.park_rollout_for_train,
            resources.lifecycle.park_rollout_for_reward,
            resources.lifecycle.park_trainer_for_reward,
            resources.lifecycle.offload_reward,
        ),
    )


def test_wan_robotics_continuous_resolves_balanced_four_l4_topology() -> None:
    """The robotics run must keep trainer, rollout, and reward GPUs disjoint."""

    cfg = load_config("experiment/wan_2_1/online_grpo_robotics_physics_4x_l4_continuous")
    resources = ResourcePlan.from_root(parse_config(cfg))

    assert resources.trainer_devices == (0,)
    assert resources.rollout_devices == (1, 2)
    assert resources.reward_devices == ()
    assert resources.rollout_num_engines == 2
    assert resources.lifecycle.rollout_mode == "resident"


@pytest.mark.parametrize(
    "name",
    [
        "cosmos_predict2/online_grpo_droid_overfit_validation_fsdp_4x_l4",
        "cosmos_predict2/online_grpo_droid_lora_480p_curve_fsdp_4x_l4",
        "wan_2_1/online_grpo_droid_fullparam_fsdp_3x_l4",
        "wan_2_1/online_grpo_droid_fullparam_fsdp_4x_l4",
        "wan_2_1/online_grpo_i2v_fsdp_2x_l4",
    ],
)
def test_rank_local_fsdp_recipes_colocate_one_rollout_per_rank(cuda_devices, name: str) -> None:
    """Each rank sees one GPU: its rollout engine shares it with the trainer shard
    and releases it before train."""

    cuda_devices(1)
    resources = ResourcePlan.from_root(parse_config(load_config(f"experiment/{name}")))

    assert resources.trainer_devices == resources.rollout_devices == (0,)
    assert resources.rollout_num_engines == 1
    assert resources.reward_devices == ()
    assert resources.lifecycle.rollout_mode == "on_demand"
    assert resources.lifecycle.park_rollout_for_train is True


def test_cosmos_predict2_overfit_fsdp_4x_l4_inherits_parent_geometry() -> None:
    """Per-rank geometry remains inherited from the parent; only the topology
    leaves may differ. Compare against the source of truth, not literals."""

    cfg = load_config("experiment/cosmos_predict2/online_grpo_droid_overfit_validation_fsdp_4x_l4")
    parent = load_config("experiment/cosmos_predict2/online_grpo_droid_overfit_validation")

    for path in (
        "actor.ppo_epochs",
        "actor.timestep_fraction",
        "rollout.n_samples_per_prompt",
        "rollout.prompts_per_batch",
        "trainer.total_epochs",
    ):
        assert OmegaConf.select(cfg, path) == OmegaConf.select(parent, path)


def test_cosmos_predict2_full_curve_fsdp_4x_l4_preserves_training_semantics() -> None:
    """The durable full-DROID run changes topology without changing its curve."""

    cfg = load_config("experiment/cosmos_predict2/online_grpo_droid_lora_480p_curve_fsdp_4x_l4")
    parent = load_config("experiment/cosmos_predict2/online_grpo_droid_lora_480p_curve")

    # The parent is the single source of truth for learning, dataset, and reward
    # semantics. Only the validated hardware-specific leaves may differ here.
    assert OmegaConf.to_container(cfg.algorithm, resolve=True) == OmegaConf.to_container(
        parent.algorithm,
        resolve=True,
    )
    assert OmegaConf.to_container(cfg.actor, resolve=True) == OmegaConf.to_container(
        parent.actor,
        resolve=True,
    )
    assert OmegaConf.to_container(cfg.model, resolve=True) == OmegaConf.to_container(
        parent.model,
        resolve=True,
    )
    assert OmegaConf.to_container(cfg.data, resolve=True) == OmegaConf.to_container(
        parent.data,
        resolve=True,
    )
    assert OmegaConf.to_container(cfg.reward, resolve=True) == OmegaConf.to_container(
        parent.reward,
        resolve=True,
    )
    sampling = OmegaConf.to_container(cfg.sampling, resolve=True)
    parent_sampling = OmegaConf.to_container(parent.sampling, resolve=True)
    assert isinstance(sampling, dict) and isinstance(parent_sampling, dict)
    sampling.pop("guidance_scale")
    parent_sampling.pop("guidance_scale")
    assert sampling == parent_sampling

    rollout = OmegaConf.to_container(cfg.rollout, resolve=True)
    parent_rollout = OmegaConf.to_container(parent.rollout, resolve=True)
    assert isinstance(rollout, dict) and isinstance(parent_rollout, dict)
    rollout.pop("samples_per_generation_batch")
    parent_rollout.pop("samples_per_generation_batch")
    assert rollout == parent_rollout

    trainer = OmegaConf.to_container(cfg.trainer, resolve=True)
    parent_trainer = OmegaConf.to_container(parent.trainer, resolve=True)
    assert isinstance(trainer, dict) and isinstance(parent_trainer, dict)
    for key in ("output_dir", "save_freq"):
        trainer.pop(key)
        parent_trainer.pop(key)
    assert trainer == parent_trainer


def test_wan_droid_fullparam_fsdp_4x_l4_uses_symmetric_reward_handoffs(cuda_devices) -> None:
    """All four ranks time-share local rollout and the complete robotics reward."""

    cuda_devices(1)
    cfg = load_config("experiment/wan_2_1/online_grpo_droid_fullparam_fsdp_4x_l4")
    resources = ResourcePlan.from_root(parse_config(cfg))

    assert resources.reward_torch_device(trainer_device="cuda:0") == "cuda:0"
    handoff = resources.lifecycle
    assert handoff.park_rollout_for_reward is True
    assert handoff.park_trainer_for_reward is True
    assert handoff.offload_reward is True


def test_masked_physical_ordinal_comes_from_the_config_knob_not_the_auto_path() -> None:
    """A non-zero rank-local ordinal is unreachable through the auto path.

    ``_auto_visible_cuda_devices`` returns ``tuple(range(n))``, so ``(1,)`` can
    never come out of it at any device count. The launcher writes the selected
    physical ordinal into ``visible_devices`` before resource resolution
    (tests/scripts/test_online_entrypoint.py), and that is the path this pins —
    the real ``_parse_devices`` / dedupe branches, on this recipe.
    """

    cfg = load_config("experiment/wan_2_1/online_grpo_droid_fullparam_fsdp_4x_l4")
    OmegaConf.update(cfg, "distributed.resources.visible_devices", [1], force_add=True)
    require_training_config(cfg)

    resources = ResourcePlan.from_root(parse_config(cfg))

    assert resources.visible_devices == (1,)
    assert resources.trainer_devices == resources.rollout_devices == (1,)
    assert resources.colocated is True


def test_unified_train_entrypoint_reads_yaml_entrypoint() -> None:
    """The YAML ``trainer.entrypoint`` is a dotted path that imports to a callable."""
    from vrl.utils.config import import_from_path

    cfg = load_config("experiment/sd3_5/online_grpo_ocr")

    assert callable(import_from_path(parse_config(cfg).trainer.entrypoint))


def test_cli_overrides_reach_typed_trainer_config() -> None:
    """CLI overrides for resume, torch profiler, drop-zero-advantage and batch width all reach the
    typed trainer / resume / rollout configs.
    """
    cfg = load_config(
        "experiment/sd3_5/online_grpo_ocr",
        overrides=[
            "trainer.resume_from=/tmp/checkpoint-10",
            "trainer.torch_profiler.enabled=true",
            "trainer.torch_profiler.activities=[cpu]",
            "actor.drop_zero_advantage=false",
            "rollout.samples_per_generation_batch=2",
        ],
    )
    built = BuiltConfigs.from_cfg(cfg)
    trainer = built.trainer

    assert built.resume_from == "/tmp/checkpoint-10"
    assert trainer.torch_profiler.enabled is True
    assert trainer.torch_profiler.activities == ("cpu",)
    assert trainer.drop_zero_advantage is False
    assert built.root.rollout is not None
    assert built.root.rollout.samples_per_generation_batch == 2


@pytest.mark.parametrize("value", ["0", "largest"])
def test_generation_chunk_rejects_non_positive_or_non_integer_values(value: str) -> None:
    cfg = load_config(
        "experiment/sd3_5/online_grpo_ocr",
        overrides=[f"rollout.samples_per_generation_batch={value}"],
    )

    with pytest.raises(ValueError, match=r"rollout\.samples_per_generation_batch"):
        BuiltConfigs.from_cfg(cfg)


def test_negative_reward_component_weights_are_rejected() -> None:
    cfg = load_config(
        "experiment/sd3_5/online_grpo_pickscore",
        overrides=[
            "+reward=nsfw_safety",
            "+dataset=anime_safety_stress",
            "trainer.output_dir=/test-only/anime-safety",
        ],
    )
    cfg.reward.components.nsfw_safety = -0.5

    with pytest.raises(ValueError, match=r"reward\.components\.nsfw_safety must be >= 0"):
        RewardConfig.from_cfg(cfg)


def test_public_reward_builder_validates_its_input() -> None:
    cfg = OmegaConf.create(
        {"reward": {"components": {"ocr": 1.0}, "kwargs": {"ocr": ["invalid"]}}},
    )

    with pytest.raises(ValueError, match=r"reward\.kwargs\.ocr must be a mapping"):
        RewardRuntimeConfig.from_section(RewardConfig.from_cfg(cfg))


def test_required_training_fields_fail_fast() -> None:
    cfg = load_config("experiment/wan_2_1/online_grpo_ocr")
    cfg.trainer.output_dir = "???"
    with pytest.raises(ValueError, match=r"trainer\.output_dir"):
        require_training_config(cfg)


def test_dpo_allows_explicit_null_max_train_samples() -> None:
    """An explicit ``data.max_train_samples=null`` is a valid DPO setting, not a missing field."""
    cfg = load_config("experiment/wan_2_1/offline_dpo_pickapic")
    cfg.data.max_train_samples = None

    assert parse_config(cfg).data.max_train_samples is None
    require_training_config(cfg)


def test_config_parsing_stays_torch_free() -> None:
    """Resolving any recipe must not load torch.

    Three package facades defer their torch-backed submodules to keep this true
    (vrl.trajectory, vrl.trainers.data, vrl.algorithms.grpo). Without this test
    the next eager re-export in any of them silently puts torch back on the
    config path, where nothing else would notice. Subprocess because the test
    session has already imported torch.
    """

    probe = (
        "import sys; "
        "from vrl.config.loading import load_config; "
        "from vrl.config.schema import parse_config; "
        "parse_config(load_config('experiment/sana/online_grpo_aesthetic')); "
        "raise SystemExit(1 if 'torch' in sys.modules else 0)"
    )
    assert subprocess.run([sys.executable, "-c", probe], check=False).returncode == 0
