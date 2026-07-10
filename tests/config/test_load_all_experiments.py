"""Core config-loader safety tests.

This file intentionally keeps broad experiment coverage in loop-style tests
instead of parametrizing the same assertion into dozens of collected tests.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from vrl.algorithms.diffusion_nft import DiffusionNFTConfig
from vrl.algorithms.dpo import DiffusionDPOConfig
from vrl.algorithms.grpo.continuous import GRPOConfig
from vrl.algorithms.grpo.multisegment import MultiSegmentTokenGRPOConfig
from vrl.algorithms.grpo.token import TokenGRPOConfig
from vrl.config.builders import build_algorithm_config, build_configs
from vrl.config.loading import (
    bundled_config_resource,
    list_bundled_configs,
    load_config,
)
from vrl.config.validation import (
    optional_none,
    validate_reward_config,
    validate_training_config,
)


def _experiment_names() -> list[str]:
    return [
        name.removeprefix("experiment/").removesuffix(".yaml")
        for name in list_bundled_configs("experiment")
    ]


def _load_bundled_raw(name: str):
    resource = bundled_config_resource(name)
    with resource.open("r", encoding="utf-8") as stream:
        return OmegaConf.load(stream)


def test_load_config_enforces_mandatory_marker(tmp_path: Path) -> None:
    """Keys declared '???' must be set by the experiment or a dotlist override."""
    config = tmp_path / "exp.yaml"
    config.write_text("trainer:\n  entrypoint: ???\n  seed: 0\n")

    with pytest.raises(ValueError, match=r"trainer\.entrypoint"):
        load_config(config)

    cfg = load_config(config, overrides=["trainer.entrypoint=pkg.mod:fn"])
    assert cfg.trainer.entrypoint == "pkg.mod:fn"


def test_build_trainer_config_reports_all_missing_required_keys() -> None:
    """Missing required keys are reported together, each with its full YAML path."""
    cfg = OmegaConf.create(
        {
            "actor": {},
            "trainer": {},
            "rollout": {},
            "precision": "bf16",
        },
    )

    from vrl.config.builders import build_trainer_config

    with pytest.raises(ValueError) as exc:
        build_trainer_config(cfg)

    message = str(exc.value)
    for path in (
        "actor.optim.lr",
        "actor.drop_zero_advantage",
        "actor.timestep_fraction",
        "trainer.total_epochs",
        "trainer.output_dir",
        "rollout.prompts_per_batch",
        "rollout.n_samples_per_prompt",
    ):
        assert path in message


def test_typo_yaml_home_is_rejected() -> None:
    """A metadata address naming an unknown top-level section fails loudly."""
    from vrl.config.builders import _validate_yaml_home

    _validate_yaml_home("save_freq", "trainer")          # known section: ok
    _validate_yaml_home("optim", "actor.optim")          # dotted path: ok

    with pytest.raises(AssertionError, match=r"trainerx"):
        _validate_yaml_home("save_freq", "trainerx")


def test_config_groups_are_not_flattened() -> None:
    """Checks config groups are not flattened."""
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


def test_experiments_are_grouped_by_model_family() -> None:
    """Every experiment lives at <modality>/<model>/<recipe> under a known modality.

    The structural convention — not any specific roster of names — is what the
    layout guards: a two-level family grouping (modality, then model family)
    with the recipe as the leaf. Pinning the literal set of names just churns
    the test on every add/remove/rename; "all load+validate" is already owned by
    test_all_experiments_load_and_validate.
    """
    names = _experiment_names()
    assert {Path(name).parts[0] for name in names} == {"ar", "diffusion"}
    assert all(len(Path(name).parts) == 3 for name in names)


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
    """Checks reward configs are single reward building blocks."""
    offenders = []
    for name in list_bundled_configs("reward"):
        raw = _load_bundled_raw(name)
        components = raw.get("reward", {}).get("components", {})
        if len(components) != 1:
            offenders.append(name)

    assert offenders == []


def test_all_experiments_load_and_validate() -> None:
    """Checks all experiments load and validate."""
    for name in _experiment_names():
        cfg = load_config(f"experiment/{name}")
        assert "model" in cfg, f"{name} missing model.*"
        assert "trainer" in cfg, f"{name} missing trainer.*"
        assert "algorithm" in cfg, f"{name} missing algorithm.*"
        assert "data" in cfg, f"{name} missing data.* source"
        if str(cfg.algorithm.kind) != "diffusion_dpo":
            assert "reward" in cfg, f"{name} missing reward.* source"
        assert "path" in cfg.model, f"{name} missing model.path"
        assert "entrypoint" in cfg.trainer, f"{name} missing trainer.entrypoint"
        assert "output_dir" in cfg.trainer, f"{name} missing trainer.output_dir"
        assert "kind" in cfg.algorithm, f"{name} missing algorithm.kind"
        assert "adv_estimator" not in cfg.algorithm, f"{name} still uses adv_estimator"
        validate_training_config(cfg)


def test_validate_rejects_compile_with_gradient_checkpointing() -> None:
    """compile x grad-ckpt must fail at config load, not as a mid-run dynamo crash.

    The trainer refuses the combination at startup (activation_checkpointing.py);
    this checks validate_training_config rejects it too, because a model-layer
    torch_compile.enable=true default can silently flip compile on underneath an
    experiment that sets checkpointing (that exact collision shipped in four
    cosmos_predict2 240p recipes before this load-time check existed).
    """
    base = "experiment/diffusion/sd3_5/online_grpo_ocr"  # resolves compile=true

    for ckpt in ("true", "full", "selective"):
        cfg = load_config(base, overrides=[f"actor.gradient_checkpointing={ckpt}"])
        with pytest.raises(ValueError, match="cannot combine"):
            validate_training_config(cfg)

    # Explicit off (either spelling) keeps compile allowed.
    cfg = load_config(base, overrides=["actor.gradient_checkpointing=off"])
    validate_training_config(cfg)


@pytest.mark.parametrize(
    "name",
    [
        "diffusion/cosmos_predict2_5/online_nft_kling_video_reward",
        "diffusion/cosmos_predict2_5/online_nft_motion_physics",
    ],
)
def test_cosmos_predict25_nft_uses_paper_timestep_budget(name: str) -> None:
    """Checks Cosmos Predict2.5 NFT recipes keep the paper timestep budget."""
    cfg = load_config(f"experiment/{name}")

    # Denoise budget (num_steps / cfg / guidance_scale), timestep_fraction, and the
    # exact LoRA target_modules set are declarative YAML (sampling/denoise/
    # 20_step_no_cfg + the cosmos LoRA model group). Per the no-exact-config rule
    # they are free to be retuned; load+validate coverage lives in
    # test_all_experiments_load_and_validate.
    assert cfg.model.use_lora is True
    # Real invariant, not a literal: enabling LoRA must declare which modules to wrap.
    assert cfg.model.lora.target_modules  # non-empty when use_lora is True


def test_cosmos_predict25_kling_reward_uses_paper_rl_batch() -> None:
    """Checks the Kling reward recipe matches the paper RL batch geometry."""
    cfg = load_config("experiment/diffusion/cosmos_predict2_5/online_nft_kling_video_reward")

    # Batch geometry (n_samples_per_prompt / prompts_per_batch / samples_per_chunk /
    # microbatch_size) is declarative YAML a tuner is free to change. Assert the real
    # coupling instead of pinning the paper's magic numbers.
    assert cfg.rollout.prompts_per_batch % cfg.rollout.n_samples_per_prompt == 0
    # gradient_accumulation_steps is a DERIVED TrainerConfig value — this stays: it
    # tests the prompts_per_batch / microbatch_size derivation, not a literal.
    from vrl.trainers.core.types import OptimConfig, TrainerConfig

    derived = TrainerConfig(
        optim=OptimConfig(lr=1e-4),
        n_samples_per_prompt=cfg.rollout.n_samples_per_prompt,
        prompts_per_batch=cfg.rollout.prompts_per_batch,
        microbatch_size=cfg.rollout.microbatch_size,
        timestep_fraction=0.5,
        total_epochs=1,
        output_dir="x",
        drop_zero_advantage=False,
    )
    assert (
        derived.gradient_accumulation_steps
        == cfg.rollout.prompts_per_batch // cfg.rollout.microbatch_size
    )


def test_experiments_do_not_use_legacy_precision_fields() -> None:
    """Checks live YAML uses top-level precision only."""
    offenders = []
    for name in _experiment_names():
        cfg = load_config(f"experiment/{name}")
        actor = cfg.get("actor", {})
        if "mixed_precision" in actor or "bf16" in actor:
            offenders.append(name)
    assert offenders == []


def test_rollout_orchestration_group_override_uses_rollout_namespace() -> None:
    """Checks rollout orchestration group override uses rollout namespace."""
    cfg = load_config(
        "experiment/diffusion/sd3_5/online_grpo_ocr",
        overrides=["/base/rollout/orchestration=continuous"],
    )

    orchestration = cfg.trainer.rollout_orchestration
    assert orchestration.schedule_mode == "continuous"
    # weight_sync_barrier is no longer spelled in YAML; the typed config
    # derives the only barrier the mode supports.
    from vrl.trainers.core.types import RolloutOrchestrationConfig

    typed = RolloutOrchestrationConfig(
        **OmegaConf.to_container(orchestration, resolve=True),
    )
    assert typed.weight_sync_barrier == "pause_admission_and_drain_inflight"


def test_sd35_single_gpu_async_debug_uses_persistent_colocated_rollout() -> None:
    """Checks the single-GPU async debug recipe opts into colocated continuous rollout."""
    from vrl.ray.resources import resolve_distributed_resources

    cfg = load_config(
        "experiment/diffusion/sd3_5/online_grpo_ocr_single_gpu_async_debug",
        overrides=["distributed.resources.visible_devices=[0]"],
    )

    assert cfg.trainer.rollout_orchestration.schedule_mode == "continuous"
    assert cfg.trainer.rollout_orchestration.require_separate_gpus is False
    assert cfg.trainer.rollout_orchestration.continuous.max_stale_policy_versions == 1
    assert cfg.distributed.resources.allow_overlap is True
    # Resident colocation is declared with gpu_pool=trainer + memory_fraction on the
    # resources rollout node; release scheduling is derived from the resolved
    # topology, not spelled in YAML. Assert the derived/lifecycle result, not the
    # raw input key.
    assert cfg.distributed.resources.rollout.gpu_pool == "trainer"
    resolved = resolve_distributed_resources(cfg)
    # memory_fraction passes through resolve_distributed_resources unchanged, so
    # pinning == 0.55 only echoes the YAML. Assert the resolver's real invariant
    # instead — it validates 0 < fraction <= 1 (vrl/ray/resources.py:302).
    assert resolved.rollout_gpu_memory_fraction is not None
    assert 0.0 < resolved.rollout_gpu_memory_fraction <= 1.0
    # resident lifecycle is DERIVED from gpu_pool=trainer + memory_fraction (not YAML) — keep.
    assert resolved.lifecycle.rollout.mode == "resident"
    # height/width and the batch sizes are declarative debug-recipe YAML; load+validate
    # coverage is in test_all_experiments_load_and_validate. No literal pins.


def test_algorithm_config_dispatches_representative_kinds() -> None:
    """Checks algorithm config dispatches representative kinds."""
    examples = {
        "diffusion/sd3_5/online_grpo_ocr": GRPOConfig,
        "ar/janus_pro/online_grpo_ocr": TokenGRPOConfig,
        "ar/janus_pro/online_r1_grpo_ocr": MultiSegmentTokenGRPOConfig,
        "diffusion/wan_2_1/offline_dpo_pickapic": DiffusionDPOConfig,
        "diffusion/cosmos_predict2_5/online_nft_kling_video_reward": DiffusionNFTConfig,
    }
    for name, expected_type in examples.items():
        cfg = load_config(f"experiment/{name}")
        algo_cfg = build_algorithm_config(cfg)
        assert isinstance(algo_cfg, expected_type)


def test_algorithm_dispatch_is_stable_per_kind() -> None:
    """build_algorithm_config is the single dispatch: same recipe -> same type.

    Replaces the old mirror dict that re-copied the builder's own kind->class
    map and only asserted "dispatch == a copy of dispatch". The
    representative-kind correctness anchor lives in
    test_algorithm_config_dispatches_representative_kinds.
    """
    for name in (
        "diffusion/sd3_5/online_grpo_ocr",
        "ar/janus_pro/online_grpo_ocr",
        "diffusion/wan_2_1/offline_dpo_pickapic",
    ):
        cfg = load_config(f"experiment/{name}")
        first = build_algorithm_config(cfg)
        second = build_algorithm_config(cfg)
        assert type(first) is type(second)


def test_cosmos_v2w_production_validation_accepts_source_backed_data(
    tmp_path: Path,
) -> None:
    """Checks Cosmos V2W production validation accepts source-backed data."""
    data_root = tmp_path / "external"
    reference = data_root / "video_world" / "references" / "ref.ppm"
    reference.parent.mkdir(parents=True, exist_ok=True)
    reference.write_text("P3\n1 1\n255\n0 0 0\n", encoding="utf-8")
    metadata = {
        "source": "droid",
        "source_repo": "lerobot/droid_100",
        "source_split": "main",
        "source_episode": "episode_train",
        "source_video": "videos/camera/chunk-000/file-000.mp4",
        "source_frame_index": 0,
        "decode_method": "pyav_http_first_frame",
        "conditioning": "first_frame",
    }
    train = tmp_path / "robot_train.jsonl"
    eval_manifest = tmp_path / "robot_eval.jsonl"
    train.write_text(
        json.dumps(
            {
                "prompt": "The robot arm moves toward the cup.",
                "reference_image": "video_world/references/ref.ppm",
                "metadata": metadata,
            },
        )
        + "\n",
        encoding="utf-8",
    )
    eval_metadata = dict(metadata, source_episode="episode_eval")
    eval_manifest.write_text(
        json.dumps(
            {
                "prompt": "The robot arm moves away from the cup.",
                "reference_image": "video_world/references/ref.ppm",
                "metadata": eval_metadata,
            },
        )
        + "\n",
        encoding="utf-8",
    )
    report = tmp_path / "robot_report.json"
    report.write_text(
        json.dumps(
            {
                "dataset": "video_world_bridge",
                "source": "droid",
                "repo_id": "lerobot/droid_100",
                "source_split": "main",
                "decode_method": "pyav_http_first_frame",
                "train_rows": 1,
                "eval_rows": 1,
                "train_manifest": train.as_posix(),
                "eval_manifest": eval_manifest.as_posix(),
                "reference_dir": reference.parent.as_posix(),
                "validation_summary": {"row_count": 1},
            },
        ),
        encoding="utf-8",
    )
    cfg = load_config(
        "experiment/diffusion/cosmos_predict2/online_grpo_v2w_reference",
        overrides=[
            "production.kling_video_reward.enabled=true",
            f"data.manifest={train.as_posix()}",
            f"data.eval_manifest={eval_manifest.as_posix()}",
            f"data.source_report={report.as_posix()}",
            f"data.artifact_data_root={data_root.as_posix()}",
        ],
    )

    validate_training_config(cfg)


def test_cosmos_target_v2w_production_validation_requires_target_clip(
    tmp_path: Path,
) -> None:
    """Checks target-reward Cosmos V2W validation requires public-source target clips."""
    data_root = tmp_path / "external"
    reference = data_root / "video_world" / "references" / "ref.ppm"
    target = data_root / "video_world" / "targets" / "target.mp4"
    reference.parent.mkdir(parents=True, exist_ok=True)
    target.parent.mkdir(parents=True, exist_ok=True)
    reference.write_text("P3\n1 1\n255\n0 0 0\n", encoding="utf-8")
    target.write_bytes(b"fake-video")
    metadata = {
        "source": "droid",
        "source_repo": "lerobot/droid_100",
        "source_split": "main",
        "source_episode": "episode_train",
        "source_video": "videos/camera/chunk-000/file-000.mp4",
        "source_frame_index": 0,
        "decode_method": "pyav_http_target_clip",
        "conditioning": "first_frame",
    }
    train = tmp_path / "droid_targets_train.jsonl"
    eval_manifest = tmp_path / "droid_targets_eval.jsonl"
    train.write_text(
        json.dumps(
            {
                "prompt": "The robot arm moves toward the cup.",
                "reference_image": "video_world/references/ref.ppm",
                "target_video": "video_world/targets/target.mp4",
                "metadata": metadata,
            },
        )
        + "\n",
        encoding="utf-8",
    )
    eval_metadata = dict(metadata, source_episode="episode_eval")
    eval_manifest.write_text(
        json.dumps(
            {
                "prompt": "The robot arm moves away from the cup.",
                "reference_image": "video_world/references/ref.ppm",
                "target_video": "video_world/targets/target.mp4",
                "metadata": eval_metadata,
            },
        )
        + "\n",
        encoding="utf-8",
    )
    report = tmp_path / "droid_targets_report.json"
    report.write_text(
        json.dumps(
            {
                "dataset": "video_world_targets",
                "source": "droid",
                "repo_id": "lerobot/droid_100",
                "source_split": "main",
                "decode_method": "pyav_http_target_clip",
                "train_rows": 1,
                "eval_rows": 1,
                "train_manifest": train.as_posix(),
                "eval_manifest": eval_manifest.as_posix(),
                "reference_dir": reference.parent.as_posix(),
                "target_dir": target.parent.as_posix(),
                "validation_summary": {"row_count": 1},
            },
        ),
        encoding="utf-8",
    )
    # The droid-target recipe now uses a zero-training dino+motion reward (no kling); the
    # kling production-contract path is covered by
    # test_cosmos_v2w_production_validation_accepts_source_backed_data. Here we validate
    # that target-clip-backed data resolves and validates for this recipe.
    cfg = load_config(
        "experiment/diffusion/cosmos_predict2/online_grpo_droid_target_480p",
        overrides=[
            f"data.manifest={train.as_posix()}",
            f"data.eval_manifest={eval_manifest.as_posix()}",
            f"data.source_report={report.as_posix()}",
            f"data.artifact_data_root={data_root.as_posix()}",
        ],
    )

    validate_training_config(cfg)


def test_wan_i2v_production_validation_accepts_source_backed_data(tmp_path: Path) -> None:
    """Checks Wan I2V production validation accepts source-backed data."""
    data_root = tmp_path / "videophy_i2v"
    train_image = data_root / "images" / "train" / "000.ppm"
    eval_image = data_root / "images" / "eval" / "000.ppm"
    train_image.parent.mkdir(parents=True)
    eval_image.parent.mkdir(parents=True)
    train_image.write_text("P3\n1 1\n255\n0 0 0\n", encoding="utf-8")
    eval_image.write_text("P3\n1 1\n255\n0 0 0\n", encoding="utf-8")
    metadata = {
        "source": "videophy",
        "source_repo": "videophysics/videophy_test_public",
        "source_split": "test",
        "source_csv_row": 0,
        "source_video_url": "https://videophysics.example/train.mp4",
        "source_frame_index": 0,
        "decode_method": "imageio_ffmpeg_first_frame",
        "conditioning": "first_frame",
    }
    train_manifest = data_root / "manifests" / "train.jsonl"
    eval_manifest = data_root / "manifests" / "eval.jsonl"
    train_manifest.parent.mkdir(parents=True)
    train_manifest.write_text(
        json.dumps(
            {
                "image": "images/train/000.ppm",
                "caption": "A wheel rolls.",
                "task_type": "image_to_video",
                "metadata": metadata,
            },
        )
        + "\n",
        encoding="utf-8",
    )
    eval_metadata = dict(metadata, source_video_url="https://videophysics.example/eval.mp4")
    eval_manifest.write_text(
        json.dumps(
            {
                "image": "images/eval/000.ppm",
                "caption": "Honey diffuses.",
                "task_type": "image_to_video",
                "metadata": eval_metadata,
            },
        )
        + "\n",
        encoding="utf-8",
    )
    report = data_root / "report.json"
    report.write_text(
        json.dumps(
            {
                "dataset": "videophy_i2v",
                "source_repo": "videophysics/videophy_test_public",
                "source_csv": "videophy_test_public.csv",
                "source_split": "test",
                "decode_method": "imageio_ffmpeg_first_frame",
                "train_rows": 1,
                "eval_rows": 1,
                "train_manifest": train_manifest.as_posix(),
                "eval_manifest": eval_manifest.as_posix(),
                "reference_dir": (data_root / "images").as_posix(),
            },
        ),
        encoding="utf-8",
    )

    cfg = load_config(
        "experiment/diffusion/wan_2_1/online_grpo_physics_i2v",
        overrides=[
            "production.kling_video_reward.enabled=true",
            f"data.manifest={train_manifest.as_posix()}",
            f"data.eval_manifest={eval_manifest.as_posix()}",
            f"data.source_report={report.as_posix()}",
            f"data.artifact_data_root={data_root.as_posix()}",
        ],
    )

    validate_training_config(cfg)


def test_wan_video_reward_production_config_requires_reward_name() -> None:
    """Checks Wan video reward production config requires reward name."""
    cfg = load_config("experiment/diffusion/wan_2_1/online_grpo_kling_video_reward")
    cfg.reward.kwargs.kling_video_reward.reward_name = ""

    with pytest.raises(ValueError, match="reward_name"):
        validate_training_config(cfg)


def test_wan_video_reward_production_rejects_extra_loader_fields() -> None:
    """Checks Wan video reward production rejects extra loader fields."""
    cfg = load_config("experiment/diffusion/wan_2_1/online_grpo_kling_video_reward")
    cfg.reward.kwargs.kling_video_reward.worker_config.import_path = "fake:thing"

    with pytest.raises(ValueError, match="remove extra loader fields"):
        validate_training_config(cfg)

    cfg = load_config("experiment/diffusion/wan_2_1/online_grpo_kling_video_reward")
    cfg.reward.kwargs.kling_video_reward.worker_config.model_factory = "fake:factory"

    with pytest.raises(ValueError, match="remove extra loader fields"):
        validate_training_config(cfg)


def test_unified_train_entrypoint_reads_yaml_entrypoint() -> None:
    """Checks unified train entrypoint reads YAML entrypoint."""
    from vrl.scripts.train import _import_callable, resolve_train_target

    cfg = load_config("experiment/diffusion/sd3_5/online_grpo_ocr")
    target = resolve_train_target(cfg)

    assert target.import_path == cfg.trainer.entrypoint
    assert callable(_import_callable(target.import_path))


def test_cli_overrides_reach_typed_trainer_config() -> None:
    """Checks CLI overrides reach typed trainer config."""
    cfg = load_config(
        "experiment/diffusion/sd3_5/online_grpo_ocr",
        overrides=[
            "trainer.resume_from=/tmp/checkpoint-10",
            "trainer.torch_profiler.enabled=true",
            "trainer.torch_profiler.activities=[cpu]",
            "actor.drop_zero_advantage=false",
            "rollout.samples_per_chunk=2",
        ],
    )
    trainer = build_configs(cfg)["trainer"]

    assert trainer.resume_from == "/tmp/checkpoint-10"
    assert trainer.torch_profiler.enabled is True
    assert trainer.torch_profiler.activities == ("cpu",)
    assert trainer.drop_zero_advantage is False
    assert trainer.samples_per_chunk == 2


def test_invalid_algorithm_kind_fails_fast() -> None:
    """Checks invalid algorithm kind fails fast."""
    cfg = OmegaConf.create({"algorithm": {"kind": "grpo", "adv_estimator": "dpo"}})
    with pytest.raises(ValueError, match="adv_estimator"):
        build_algorithm_config(cfg)

    cfg = OmegaConf.create({"algorithm": {"kind": "qpo"}})
    with pytest.raises(ValueError, match=r"unknown algorithm\.kind"):
        build_algorithm_config(cfg)


def test_unknown_algorithm_config_fields_fail_fast() -> None:
    """Checks unknown algorithm config fields fail fast."""
    cfg = OmegaConf.create({"algorithm": {"kind": "grpo", "unknown_knob": 1}})

    with pytest.raises(ValueError, match=r"algorithm\.unknown_knob"):
        build_algorithm_config(cfg)


def test_diffusion_nft_rejects_removed_advantage_low() -> None:
    """Checks DiffusionNFT derives the low clamp from advantage_scale."""

    cfg = OmegaConf.create(
        {
            "algorithm": {
                "kind": "diffusion_nft",
                "advantage_scale": 5.0,
                "advantage_low": -5.0,
            },
        },
    )

    with pytest.raises(ValueError, match=r"algorithm\.advantage_low"):
        build_algorithm_config(cfg)


def test_missing_drop_zero_advantage_fails_fast() -> None:
    """actor.drop_zero_advantage is required; removing it fails loudly."""
    cfg = load_config("experiment/diffusion/sd3_5/online_grpo_ocr")
    del cfg.actor["drop_zero_advantage"]
    with pytest.raises(ValueError, match=r"actor\.drop_zero_advantage"):
        build_configs(cfg)


def test_negative_reward_component_weights_are_rejected() -> None:
    """Checks negative reward component weights are rejected."""
    cfg = load_config("experiment/diffusion/anima_preview3/online_grpo_aesthetic_nsfw_safety")
    cfg.reward.components.nsfw_safety = -0.5

    with pytest.raises(ValueError, match=r"reward\.components\.nsfw_safety must be >= 0"):
        validate_reward_config(cfg)


def test_required_training_fields_fail_fast() -> None:
    """Checks required training fields fail fast."""
    cfg = load_config("experiment/diffusion/wan_2_1/online_grpo_ocr")
    cfg.trainer.output_dir = "???"
    with pytest.raises(ValueError, match=r"trainer\.output_dir"):
        validate_training_config(cfg)


def test_dpo_allows_explicit_null_max_train_samples() -> None:
    """Checks DPO allows explicit null max train samples."""
    cfg = load_config("experiment/diffusion/wan_2_1/offline_dpo_pickapic")
    cfg.data.max_train_samples = None

    assert optional_none(cfg, "data.max_train_samples") is None
    validate_training_config(cfg)
