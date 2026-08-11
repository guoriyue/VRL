"""Pure contract tests for the generation-only MAGI-1 subprocess adapter."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Any

import pytest
import torch
from omegaconf import OmegaConf

from vrl.config.loading import load_config
from vrl.config.precision import RolePrecision, resolve_precision_policy
from vrl.generation.execution.sample_batches import GenerationSampleBatch
from vrl.generation.types import GenerationRequest
from vrl.models.families.magi_1.model import (
    MAGI_1_SUPPORTED_SOURCE_REVISION,
    Magi1SubprocessConfig,
    Magi1SubprocessModel,
    _source_head_revision,
    build_magi_command,
    magi_subprocess_environment,
    normalize_magi_1_model_build,
    prepare_magi_runtime_config,
)
from vrl.models.families.magi_1.runtime import (
    Magi1BatchExecutor,
    build_magi_1_runtime_bundle,
)
from vrl.models.interfaces.runtime import ModelBuild, RolloutBuildOptions


def _base_config() -> dict[str, Any]:
    return {
        "model_config": {"params_dtype": "torch.bfloat16"},
        "runtime_config": {
            "seed": 1234,
            "num_frames": 96,
            "video_size_h": 720,
            "video_size_w": 720,
            "num_steps": 64,
            "fps": 24,
            "chunk_width": 6,
            "temporal_downsample_factor": 4,
            "window_size": 4,
            "noise2clean_kvrange": [5, 4, 3, 2],
            "load": "checkpoint",
            "t5_pretrained": "t5",
            "vae_pretrained": "vae",
        },
        "engine_config": {"pp_size": 1, "cp_size": 1},
    }


def _installation(tmp_path: Path) -> tuple[Magi1SubprocessConfig, Path]:
    source = tmp_path / "MAGI-1"
    entry = source / "inference" / "pipeline" / "entry.py"
    entry.parent.mkdir(parents=True)
    entry.write_text("# official entry placeholder\n", encoding="utf-8")
    for name in ("checkpoint", "t5", "vae"):
        (source / name).mkdir()
    config_path = source / "example" / "4.5B" / "4.5B_base_config.json"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(json.dumps(_base_config()), encoding="utf-8")
    return (
        Magi1SubprocessConfig(
            source_path=source,
            source_revision=MAGI_1_SUPPORTED_SOURCE_REVISION,
            config_path=config_path,
            python_executable=sys.executable,
            checkpoint_path=Path("checkpoint"),
            t5_pretrained_path=Path("t5"),
            vae_pretrained_path=Path("vae"),
        ),
        config_path,
    )


def _build(
    config: Magi1SubprocessConfig,
    *,
    rollout: bool,
) -> ModelBuild:
    return ModelBuild(
        model_name_or_path="",
        revision=None,
        device="cpu",
        parameter_dtype=torch.bfloat16,
        family="magi_1",
        precision=RolePrecision(
            dtype="bf16",
            float32_precision="ieee",
            outer_autocast=False,
        ),
        model_config={
            "source_path": str(config.source_path),
            "source_revision": config.source_revision,
            "config_path": str(config.config_path),
            "python_executable": config.python_executable,
            "timeout_seconds": config.timeout_seconds,
            "checkpoint_path": str(config.checkpoint_path),
            "t5_pretrained_path": str(config.t5_pretrained_path),
            "vae_pretrained_path": str(config.vae_pretrained_path),
        },
        sampling_config={},
        rollout=(RolloutBuildOptions(prompt_encoder_dtype=torch.float32) if rollout else None),
    )


def test_model_preset_owns_the_official_rollout_precision_contract() -> None:
    actor = load_config("base/actor", overrides=["actor.optim.lr=0.0"])
    model = load_config("model/magi_1/4_5b_base")
    precision = resolve_precision_policy(OmegaConf.merge(actor, model))

    assert precision.rollout.dtype == "bf16"
    assert precision.rollout.float32_precision == "ieee"
    assert precision.rollout.outer_autocast is False
    assert precision.prompt_encoder_dtype == "fp32"


def test_prepare_config_applies_paths_sampling_and_per_sample_seed(
    tmp_path: Path,
) -> None:
    config, _ = _installation(tmp_path)
    base = _base_config()

    prepared = prepare_magi_runtime_config(
        base,
        config=config,
        sampling={
            "seed": 7,
            "num_frames": 48,
            "height": 384,
            "width": 640,
            "num_steps": 12,
            "fps": 16,
        },
        sample_index=2,
    )

    runtime = prepared["runtime_config"]
    assert runtime["seed"] == 9
    assert runtime["num_frames"] == 48
    assert runtime["video_size_h"] == 384
    assert runtime["video_size_w"] == 640
    assert runtime["num_steps"] == 12
    assert runtime["fps"] == 16
    assert runtime["load"] == str(config.checkpoint_path)
    assert runtime["t5_pretrained"] == str(config.t5_pretrained_path)
    assert runtime["vae_pretrained"] == str(config.vae_pretrained_path)
    assert base["runtime_config"]["seed"] == 1234


def test_prepare_config_rejects_multi_process_official_config(tmp_path: Path) -> None:
    config, _ = _installation(tmp_path)
    base = _base_config()
    base["engine_config"]["pp_size"] = 2

    with pytest.raises(ValueError, match="single-process"):
        prepare_magi_runtime_config(
            base,
            config=config,
            sampling={},
            sample_index=0,
        )


@pytest.mark.parametrize(
    ("sampling", "message"),
    [
        ({"num_frames": 25}, "multiple.*24"),
        ({"height": 721}, "height.*multiple of 16"),
        ({"width": 719}, "width.*multiple of 16"),
        ({"num_steps": 63}, "divisible.*window_size=4"),
        ({"fps": 1}, "fps must be >= 2"),
        ({"guidance_scale": 7.5}, "does not map sampling.guidance_scale"),
        ({"negative_prompt": "blurry"}, "does not expose negative_prompt"),
    ],
)
def test_prepare_config_rejects_silently_rounded_or_ignored_sampling(
    tmp_path: Path,
    sampling: dict[str, Any],
    message: str,
) -> None:
    config, _ = _installation(tmp_path)

    with pytest.raises(ValueError, match=message):
        prepare_magi_runtime_config(
            _base_config(),
            config=config,
            sampling=sampling,
            sample_index=0,
        )


def test_command_targets_official_entry_and_i2v_flag(tmp_path: Path) -> None:
    config, _ = _installation(tmp_path)
    prepared_path = tmp_path / "prepared.json"
    output_path = tmp_path / "output.mp4"

    command = build_magi_command(
        config=config,
        prepared_config_path=prepared_path,
        mode="i2v",
        prompt="a red kite",
        output_path=output_path,
        conditioning_path="reference.png",
    )

    assert command[:2] == [config.python_executable, str(config.entry_path)]
    assert command[command.index("--config_file") + 1] == str(prepared_path)
    assert command[command.index("--mode") + 1] == "i2v"
    assert command[command.index("--prompt") + 1] == "a red kite"
    assert command[command.index("--image_path") + 1] == "reference.png"
    assert command[command.index("--output_path") + 1] == str(output_path)


def test_relative_python_path_is_absolute_before_subprocess_changes_cwd(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "third_party" / "MAGI-1"
    source.mkdir(parents=True)
    executable = source / ".venv" / "bin" / "python"
    executable.parent.mkdir(parents=True)
    executable.write_text("#!/bin/sh\n", encoding="utf-8")
    executable.chmod(0o755)
    config_path = source / "example" / "4.5B" / "4.5B_base_config.json"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(json.dumps(_base_config()), encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    config = Magi1SubprocessConfig(
        source_path=Path("third_party/MAGI-1"),
        source_revision=MAGI_1_SUPPORTED_SOURCE_REVISION,
        config_path=Path("example/4.5B/4.5B_base_config.json"),
        python_executable="third_party/MAGI-1/.venv/bin/python",
    )

    assert config.python_executable == str(executable)
    assert Path(config.python_executable).is_absolute()


def test_config_path_is_pinned_to_audited_4_5b_base_variant(tmp_path: Path) -> None:
    source = tmp_path / "MAGI-1"
    unsupported = source / "example" / "4.5B" / "4.5B_distill_config.json"
    unsupported.parent.mkdir(parents=True)
    unsupported.write_text(json.dumps(_base_config()), encoding="utf-8")

    with pytest.raises(ValueError, match=r"only the audited 4.5B base config"):
        Magi1SubprocessConfig(
            source_path=source,
            source_revision=MAGI_1_SUPPORTED_SOURCE_REVISION,
            config_path=unsupported,
        )


def test_environment_forces_isolated_single_process_rank(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, _ = _installation(tmp_path)
    monkeypatch.setenv("WORLD_SIZE", "8")
    monkeypatch.setenv("RANK", "4")
    monkeypatch.setenv("PYTHONPATH", "/existing")
    monkeypatch.setenv("SPECIAL_TOKEN_PATH", "/unverified/tokens.npz")
    monkeypatch.setenv("PAD_HQ", "0")
    monkeypatch.setenv("PAD_STATIC", "1")
    monkeypatch.setenv("prev_chunks_scale", "9.0")

    env = magi_subprocess_environment(config.source_path)

    assert env["WORLD_SIZE"] == "1"
    assert env["RANK"] == "0"
    assert env["LOCAL_RANK"] == "0"
    assert env["PYTHONPATH"] == str(config.source_path)
    assert env["SPECIAL_TOKEN_PATH"] == str(
        config.source_path / "example" / "assets" / "special_tokens.npz",
    )
    assert env["PAD_HQ"] == "1"
    assert env["PAD_STATIC"] == "0"
    assert env["prev_chunks_scale"] == "0.7"


def test_runtime_bundle_has_no_trainable_state_and_replay_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vrl.models.families.magi_1 import model as magi_model

    config, _ = _installation(tmp_path)
    monkeypatch.setattr(
        magi_model,
        "_source_head_revision",
        lambda source_path: MAGI_1_SUPPORTED_SOURCE_REVISION,
    )
    bundle = build_magi_1_runtime_bundle(_build(config, rollout=True))

    assert isinstance(bundle.model, Magi1SubprocessModel)
    assert bundle.trainable_modules == {}
    assert bundle.scheduler is None
    bundle.model.load_trainable_state({})
    with pytest.raises(RuntimeError, match="replay/training is unavailable"):
        bundle.model.load_trainable_state({"weight": torch.ones(1)})
    with pytest.raises(RuntimeError, match="replay/training is unavailable"):
        bundle.model.replay_forward(object())
    with bundle.model.disable_adapter():
        pass


def test_disabled_compile_is_accepted_but_enabled_compile_fails_early(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vrl.models.families.magi_1 import model as magi_model

    config, _ = _installation(tmp_path)
    monkeypatch.setattr(
        magi_model,
        "_source_head_revision",
        lambda source_path: MAGI_1_SUPPORTED_SOURCE_REVISION,
    )
    disabled = _build(config, rollout=True)
    assert disabled.model_config is not None
    disabled.model_config["torch_compile"] = {"enable": False, "mode": "default"}

    assert build_magi_1_runtime_bundle(disabled).model is not None

    enabled = _build(config, rollout=True)
    assert enabled.model_config is not None
    enabled.model_config["torch_compile"] = {"enable": True, "mode": "default"}
    with pytest.raises(RuntimeError, match="isolated subprocess"):
        build_magi_1_runtime_bundle(enabled)


def test_runtime_rejects_precision_knobs_the_subprocess_cannot_honor(
    tmp_path: Path,
) -> None:
    config, _ = _installation(tmp_path)

    wrong_dit = _build(config, rollout=True)
    wrong_dit.parameter_dtype = torch.float32
    with pytest.raises(ValueError, match=r"params_dtype=torch.bfloat16"):
        build_magi_1_runtime_bundle(wrong_dit)

    wrong_t5 = _build(config, rollout=True)
    wrong_t5.rollout = RolloutBuildOptions(prompt_encoder_dtype=torch.bfloat16)
    with pytest.raises(ValueError, match=r"T5 runtime fixes.*float32"):
        build_magi_1_runtime_bundle(wrong_t5)

    wrong_autocast = _build(config, rollout=True)
    wrong_autocast.precision = RolePrecision(
        dtype="bf16",
        float32_precision="ieee",
        outer_autocast=True,
    )
    with pytest.raises(ValueError, match="outside VRL autocast"):
        build_magi_1_runtime_bundle(wrong_autocast)

    wrong_tf32 = _build(config, rollout=True)
    wrong_tf32.precision = RolePrecision(
        dtype="bf16",
        float32_precision="tf32",
        outer_autocast=False,
    )
    with pytest.raises(ValueError, match="must be ieee"):
        build_magi_1_runtime_bundle(wrong_tf32)


def test_driver_normalizes_external_python_before_ray_serialization(
    tmp_path: Path,
) -> None:
    config, _ = _installation(tmp_path)
    build = _build(config, rollout=True)
    assert build.model_config is not None
    build.model_config["python_executable"] = "third_party/MAGI-1/.venv/bin/python"

    normalized = normalize_magi_1_model_build(build)

    executable = Path(str(normalized.model_config["python_executable"]))
    assert executable.is_absolute()
    assert executable == (
        Path(__file__).resolve().parents[4] / "third_party" / "MAGI-1" / ".venv" / "bin" / "python"
    )


def test_executor_calls_generation_model_one_sample_at_a_time(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vrl.models.families.magi_1 import model as magi_model

    config, _ = _installation(tmp_path)
    monkeypatch.setattr(
        magi_model,
        "_source_head_revision",
        lambda source_path: MAGI_1_SUPPORTED_SOURCE_REVISION,
    )
    model = Magi1SubprocessModel(config)
    sample_indices: list[int] = []

    def fake_generate_one(**kwargs: Any) -> tuple[torch.Tensor, int]:
        sample_indices.append(int(kwargs["sample_index"]))
        return torch.zeros(1, 3, 48, 2, 2), 2

    monkeypatch.setattr(model, "_generate_one", fake_generate_one)
    executor = Magi1BatchExecutor(model)
    request = GenerationRequest(
        request_id="request",
        family="magi_1",
        task="t2v",
        inputs=["a river"],
        samples_per_prompt=2,
    )

    result = executor.forward_batch(
        request,
        GenerationSampleBatch(
            prompt_index=0,
            sample_start=0,
            sample_count=2,
        ),
    )

    assert sample_indices == [0, 1]
    assert result.output.shape == (2, 3, 48, 2, 2)
    assert result.temporal_chunk_count == 2
    assert not result.has_trainable_trajectory


def test_executor_uses_prompt_major_flat_seed_indices(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vrl.models.families.magi_1 import model as magi_model

    config, _ = _installation(tmp_path)
    monkeypatch.setattr(
        magi_model,
        "_source_head_revision",
        lambda source_path: MAGI_1_SUPPORTED_SOURCE_REVISION,
    )
    model = Magi1SubprocessModel(config)
    sample_indices: list[int] = []

    def fake_generate_one(**kwargs: Any) -> tuple[torch.Tensor, int]:
        sample_indices.append(int(kwargs["sample_index"]))
        return torch.zeros(1, 3, 48, 2, 2), 2

    monkeypatch.setattr(model, "_generate_one", fake_generate_one)
    executor = Magi1BatchExecutor(model)
    request = GenerationRequest(
        request_id="request",
        family="magi_1",
        task="t2v",
        inputs=["a river", "a forest"],
        samples_per_prompt=2,
        sampling={"seed": 100},
    )

    for prompt_index in range(len(request.inputs)):
        executor.forward_batch(
            request,
            GenerationSampleBatch(
                prompt_index=prompt_index,
                sample_start=0,
                sample_count=2,
            ),
        )

    assert sample_indices == [0, 1, 2, 3]


def test_source_revision_mismatch_fails_before_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vrl.models.families.magi_1 import model as magi_model

    config, _ = _installation(tmp_path)
    monkeypatch.setattr(
        magi_model,
        "_source_head_revision",
        lambda source_path: "f" * 40,
    )

    with pytest.raises(RuntimeError, match="source checkout revision mismatch"):
        Magi1SubprocessModel(config)


def test_packaged_source_without_git_uses_audited_runtime_digest(
    tmp_path: Path,
) -> None:
    source_root = Path(__file__).resolve().parents[4] / "third_party" / "MAGI-1"
    packaged = tmp_path / "MAGI-1"
    shutil.copytree(
        source_root / "inference",
        packaged / "inference",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    (packaged / "example" / "4.5B").mkdir(parents=True)
    (packaged / "example" / "assets").mkdir(parents=True)
    shutil.copy2(
        source_root / "example" / "4.5B" / "4.5B_base_config.json",
        packaged / "example" / "4.5B" / "4.5B_base_config.json",
    )
    shutil.copy2(
        source_root / "example" / "assets" / "special_tokens.npz",
        packaged / "example" / "assets" / "special_tokens.npz",
    )

    assert _source_head_revision(packaged) == MAGI_1_SUPPORTED_SOURCE_REVISION

    entry = packaged / "inference" / "pipeline" / "entry.py"
    entry.write_text(
        entry.read_text(encoding="utf-8") + "\n# tampered\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="packaged runtime source digest mismatch"):
        _source_head_revision(packaged)


def test_model_revision_is_forwarded_to_weight_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vrl.models.families.magi_1 import model as magi_model

    config, _ = _installation(tmp_path)
    monkeypatch.setattr(
        magi_model,
        "_source_head_revision",
        lambda source_path: MAGI_1_SUPPORTED_SOURCE_REVISION,
    )
    snapshot = tmp_path / "snapshot"
    for relative in (
        "ckpt/magi/4.5B_base",
        "ckpt/t5",
        "ckpt/vae",
    ):
        (snapshot / relative).mkdir(parents=True)
    calls: list[dict[str, Any]] = []

    def fake_snapshot_download(**kwargs: Any) -> str:
        calls.append(kwargs)
        return str(snapshot)

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_snapshot_download)
    build = _build(config, rollout=True)
    build.model_name_or_path = "sand-ai/MAGI-1"
    assert build.model_config is not None
    for key in (
        "checkpoint_path",
        "t5_pretrained_path",
        "vae_pretrained_path",
    ):
        build.model_config.pop(key)
    build.revision = "immutable-weight-sha"

    resolved = Magi1SubprocessConfig.from_build(build)

    assert calls == [
        {
            "repo_id": "sand-ai/MAGI-1",
            "revision": "immutable-weight-sha",
            "allow_patterns": [
                "ckpt/magi/4.5B_base/**",
                "ckpt/t5/**",
                "ckpt/vae/**",
            ],
        }
    ]
    assert resolved.checkpoint_path == snapshot / "ckpt" / "magi" / "4.5B_base"
    assert resolved.t5_pretrained_path == snapshot / "ckpt" / "t5"
    assert resolved.vae_pretrained_path == snapshot / "ckpt" / "vae"


def test_bad_source_preflight_prevents_weight_download(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vrl.models.families.magi_1 import model as magi_model

    config, _ = _installation(tmp_path)
    build = _build(config, rollout=True)
    build.model_name_or_path = "sand-ai/MAGI-1"
    assert build.model_config is not None
    for key in (
        "checkpoint_path",
        "t5_pretrained_path",
        "vae_pretrained_path",
    ):
        build.model_config.pop(key)
    build.revision = "immutable-weight-sha"
    monkeypatch.setattr(
        magi_model,
        "_source_head_revision",
        lambda source_path: "f" * 40,
    )

    def unexpected_download(**kwargs: Any) -> str:
        raise AssertionError(f"weight download must not start: {kwargs}")

    monkeypatch.setattr("huggingface_hub.snapshot_download", unexpected_download)

    with pytest.raises(RuntimeError, match="source checkout revision mismatch"):
        Magi1SubprocessConfig.from_build(build)


def test_invalid_build_sampling_prevents_weight_download(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vrl.models.families.magi_1 import model as magi_model

    config, _ = _installation(tmp_path)
    build = _build(config, rollout=True)
    build.sampling_config = {"num_frames": 25}
    monkeypatch.setattr(
        magi_model,
        "_preflight_local_installation",
        lambda preflight: _base_config(),
    )

    def unexpected_weight_resolution(build: ModelBuild) -> tuple[Path, Path, Path]:
        raise AssertionError(f"weight resolution must not start: {build}")

    monkeypatch.setattr(
        magi_model,
        "_resolve_weight_components",
        unexpected_weight_resolution,
    )

    with pytest.raises(ValueError, match=r"multiple.*24"):
        Magi1SubprocessConfig.from_build(build)
