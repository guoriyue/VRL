"""Opt-in RL smoke tests against locally cached real model checkpoints."""

from __future__ import annotations

import asyncio
import gc
import inspect
import math
import os
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import torch

from vrl.config.builders import build_configs
from vrl.config.loading import load_config
from vrl.generation import GenerationIdFactory, GenerationOutput, GenerationRequest
from vrl.generation.execution.planner import build_engine_plan
from vrl.ray.dependencies import import_from_path
from vrl.rollouts.families import RolloutFamilyEntry, get_rollout_family_entry
from vrl.scripts.common.factory import (
    build_algorithm_and_evaluator_from_cfg,
    build_collector_from_cfg,
    build_rollout_config_from_cfg,
)
from vrl.trainers.online import OnlineTrainer
from vrl.trainers.online.diagnostics import trainable_state_digest
from vrl.trainers.precision import torch_dtype_for_trainer_precision

RUN_REAL_ENV = "WM_RUN_REAL_MODEL_TESTS"
CASE_FILTER_ENV = "WM_REAL_MODEL_RL_CASES"


@dataclass(frozen=True, slots=True)
class CheckpointField:
    cfg_path: str
    repo_id: str
    required_files: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class RealCheckpointCase:
    case_id: str
    config: str
    family: str
    prompt: str
    checkpoints: tuple[CheckpointField, ...]
    overrides: tuple[str, ...]
    min_cuda_memory_gib: float
    reference_image_cfg_path: str | None = None


CASES: tuple[RealCheckpointCase, ...] = (
    RealCheckpointCase(
        case_id="wan_2_1",
        config="experiment/online/ocr/video_diffusion_grpo",
        family="wan_2_1",
        prompt="A clear white sign that says RL",
        checkpoints=(
            CheckpointField(
                cfg_path="model.path",
                repo_id="Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
                required_files=("model_index.json",),
            ),
        ),
        overrides=(
            "model.family=wan_2_1",
            "model.torch_compile.enable=false",
            "algorithm.init_kl_coef=0.0",
            "algorithm.kl_reward=0.0",
            "algorithm.per_prompt_stat_tracking=false",
            "rollout.n=2",
            "rollout.rollout_batch_size=1",
            "rollout.sample_batch_size=1",
            "rollout.noise_level=0.7",
            "rollout.sde.window_size=0",
            "rollout.sde.window_range=[0,1]",
            "sampling.num_steps=1",
            "sampling.guidance_scale=1.0",
            "sampling.cfg=false",
            "sampling.height=128",
            "sampling.width=128",
            "sampling.num_frames=1",
            "sampling.max_sequence_length=64",
        ),
        min_cuda_memory_gib=24.0,
    ),
    RealCheckpointCase(
        case_id="sd3_5",
        config="experiment/online/ocr/image_flow_grpo",
        family="sd3_5",
        prompt="A square poster that says RL",
        checkpoints=(
            CheckpointField(
                cfg_path="model.path",
                repo_id="stabilityai/stable-diffusion-3.5-medium",
                required_files=("model_index.json",),
            ),
        ),
        overrides=(
            "model.torch_compile.enable=false",
            "actor.mixed_precision=bf16",
            "actor.bf16=true",
            "actor.gradient_accumulation_steps=0",
            "algorithm.init_kl_coef=0.0",
            "algorithm.kl_reward=0.0",
            "algorithm.per_prompt_stat_tracking=false",
            "rollout.n=2",
            "rollout.rollout_batch_size=1",
            "rollout.sample_batch_size=1",
            "rollout.noise_level=0.7",
            "rollout.sde.window_size=0",
            "rollout.sde.window_range=[0,1]",
            "sampling.num_steps=1",
            "sampling.guidance_scale=1.0",
            "sampling.cfg=false",
            "sampling.height=128",
            "sampling.width=128",
            "sampling.max_sequence_length=64",
        ),
        min_cuda_memory_gib=24.0,
    ),
    RealCheckpointCase(
        case_id="janus_pro",
        config="experiment/online/ocr/ar_discrete_token_grpo",
        family="janus_pro",
        prompt="Text RL on a small label",
        checkpoints=(
            CheckpointField(
                cfg_path="model.path",
                repo_id="deepseek-ai/Janus-Pro-1B",
                required_files=(
                    "config.json",
                    "preprocessor_config.json",
                    "processor_config.json",
                    "tokenizer.json",
                ),
            ),
        ),
        overrides=(
            "algorithm.init_kl_coef=0.0",
            "algorithm.kl_reward=0.0",
            "algorithm.per_prompt_stat_tracking=false",
            "rollout.n_samples_per_prompt=2",
            "rollout.rollout_batch_size=1",
            "rollout.max_text_length=64",
            "sampling.image_token_num=4",
            "sampling.image_size=64",
            "sampling.cfg_weight=1.0",
            "sampling.temperature=1.0",
            "sampling.use_vllm_paged_attention=false",
            "sampling.use_ar_scheduler=false",
            "sampling.ar_scheduler_batch_size=null",
        ),
        min_cuda_memory_gib=16.0,
    ),
    RealCheckpointCase(
        case_id="cosmos_predict2",
        config="experiment/online/aesthetic/video_diffusion_grpo",
        family="cosmos-predict2",
        prompt="A quiet street with a clear RL sign",
        checkpoints=(
            CheckpointField(
                cfg_path="model.path",
                repo_id="nvidia/Cosmos-Predict2-2B-Video2World",
                required_files=(
                    "model_index.json",
                    "transformer/config.json",
                    "transformer/diffusion_pytorch_model.safetensors",
                    "text_encoder/config.json",
                    "text_encoder/model.safetensors.index.json",
                    "vae/config.json",
                    "vae/diffusion_pytorch_model.safetensors",
                    "scheduler/scheduler_config.json",
                    "tokenizer/tokenizer.json",
                ),
            ),
        ),
        overrides=(
            "model.torch_compile.enable=false",
            "algorithm.init_kl_coef=0.0",
            "algorithm.kl_reward=0.0",
            "algorithm.per_prompt_stat_tracking=false",
            "rollout.n=2",
            "rollout.rollout_batch_size=1",
            "rollout.sample_batch_size=1",
            "rollout.noise_level=0.7",
            "rollout.sde.type=cps",
            "rollout.sde.window_size=0",
            "rollout.sde.window_range=[0,1]",
            "sampling.num_steps=2",
            "sampling.guidance_scale=1.0",
            "sampling.cfg=false",
            "sampling.height=128",
            "sampling.width=128",
            "sampling.num_frames=5",
            "sampling.fps=16",
            "sampling.max_sequence_length=64",
        ),
        min_cuda_memory_gib=28.0,
        reference_image_cfg_path="model.reference_image",
    ),
    RealCheckpointCase(
        case_id="cosmos_predict2_5",
        config="experiment/online/ocr/video_diffusion_nft",
        family="cosmos-predict2.5",
        prompt="A clear white sign that says RL",
        checkpoints=(
            CheckpointField(
                cfg_path="model.path",
                repo_id="nvidia/Cosmos-Predict2.5-2B",
                required_files=(
                    "model_index.json",
                    "transformer/config.json",
                    "transformer/diffusion_pytorch_model.safetensors",
                    "vae/config.json",
                    "vae/diffusion_pytorch_model.safetensors",
                    "scheduler/scheduler_config.json",
                ),
            ),
        ),
        overrides=(
            "model.torch_compile.enable=false",
            "model.skip_text_encoder=true",
            "algorithm.kl_reward=0.0",
            "algorithm.per_prompt_stat_tracking=false",
            "rollout.n=2",
            "rollout.rollout_batch_size=1",
            "rollout.sample_batch_size=1",
            "rollout.noise_level=0.7",
            "rollout.sde.type=cps",
            "rollout.sde.window_size=0",
            "rollout.sde.window_range=[0,1]",
            "sampling.num_steps=2",
            "sampling.guidance_scale=1.0",
            "sampling.cfg=false",
            "sampling.height=128",
            "sampling.width=128",
            "sampling.num_frames=1",
            "sampling.fps=16",
            "sampling.max_sequence_length=64",
        ),
        min_cuda_memory_gib=28.0,
    ),
    RealCheckpointCase(
        case_id="nextstep_1",
        config="experiment/online/ocr/ar_continuous_token_grpo",
        family="nextstep_1",
        prompt="Text RL on a small label",
        checkpoints=(
            CheckpointField(
                cfg_path="model.path",
                repo_id="stepfun-ai/NextStep-1.1",
                required_files=("config.json",),
            ),
            CheckpointField(
                cfg_path="model.vae_path",
                repo_id="stepfun-ai/NextStep-1-f8ch16-Tokenizer",
                required_files=("config.json",),
            ),
        ),
        overrides=(
            "algorithm.init_kl_coef=0.0",
            "algorithm.kl_reward=0.0",
            "algorithm.per_prompt_stat_tracking=false",
            "rollout.n_samples_per_prompt=2",
            "rollout.rollout_batch_size=1",
            "rollout.max_text_length=64",
            "rollout.noise_level=1.0",
            "sampling.image_token_num=4",
            "sampling.image_size=64",
            "sampling.num_flow_steps=1",
            "sampling.noise_level=1.0",
            "sampling.cfg_scale=1.0",
            "sampling.use_vllm_paged_attention=false",
            "sampling.use_ar_scheduler=false",
            "sampling.ar_scheduler_batch_size=null",
        ),
        min_cuda_memory_gib=64.0,
    ),
)


class _IndexReward:
    async def score_batch(self, rollouts: list[Any]) -> list[float]:
        return [float(i) for i, _ in enumerate(rollouts)]


class _InProcessGenerationRuntime:
    """Generation runtime that executes the real family executor in-process."""

    def __init__(self, executor: Any) -> None:
        self.executor = executor
        self.current_policy_version = 0

    async def generate(self, request: GenerationRequest) -> GenerationOutput:
        rows = GenerationIdFactory().build_sample_rows(request)
        with torch.no_grad():
            plan_fn = getattr(self.executor, "plan", None)
            if callable(plan_fn):
                plan = plan_fn(request, rows)
            else:
                plan = build_engine_plan(
                    request,
                    rows,
                    capability=self.executor.capability(),
                )
            return self.executor.forward_plan(request, rows, plan)

    async def release_memory(self) -> None:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


@pytest.mark.e2e
@pytest.mark.parametrize("case", CASES, ids=lambda case: case.case_id)
def test_real_checkpoint_online_rl_updates_trainable_weights(
    case: RealCheckpointCase,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _skip_unless_case_enabled(case)
    _skip_unless_cuda_has_memory(case.min_cuda_memory_gib)

    checkpoint_overrides = [
        f"{field.cfg_path}={_resolve_checkpoint_path(case, field)}"
        for field in case.checkpoints
    ]
    case_overrides = list(case.overrides)
    if case.reference_image_cfg_path is not None:
        reference_image = _write_reference_image(tmp_path)
        case_overrides.append(
            f"{case.reference_image_cfg_path}={reference_image.as_posix()}",
        )

    cfg = load_config(
        case.config,
        overrides=[
            *checkpoint_overrides,
            *case_overrides,
            *_common_training_overrides(tmp_path),
        ],
    )
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")

    entry = get_rollout_family_entry(case.family)
    bundle: Any | None = None
    collector: Any | None = None
    trainer: OnlineTrainer | None = None
    try:
        device = torch.device("cuda")
        built = build_configs(cfg)
        trainer_config = built["trainer"]
        dtype = torch_dtype_for_trainer_precision(trainer_config, torch)
        bundle = _build_runtime_bundle(entry, cfg, device, dtype)
        executor = _build_executor(entry, bundle.model, cfg)
        collector_config = build_rollout_config_from_cfg(cfg, entry)
        collector = build_collector_from_cfg(
            cfg,
            model=bundle.model,
            reward_fn=_IndexReward(),
            family=entry,
            collector_config=collector_config,
            runtime=_InProcessGenerationRuntime(executor),
        )
        pair = build_algorithm_and_evaluator_from_cfg(
            cfg,
            family=entry,
            built=built,
            collector_config=collector_config,
            scheduler=bundle.scheduler,
        )
        trainer = OnlineTrainer(
            algorithm=pair.algorithm,
            collector=collector,
            evaluator=pair.evaluator,
            model=bundle.model,
            ref_model=None,
            weight_syncer=None,
            sync_state_getter=None,
            config=trainer_config,
            device=device,
            stat_tracker=None,
        )

        before = trainable_state_digest(bundle.model)
        metrics = asyncio.run(trainer.step([case.prompt]))
        after = trainable_state_digest(bundle.model)

        assert trainer.state.step == 1
        assert trainer.state.global_step >= 1
        assert before["tensor_count"] > 0
        assert before["sha256"] != after["sha256"]
        assert metrics.reward_std > 0.0
        assert metrics.adv_zero_rate < 1.0
        _assert_finite("loss", metrics.loss)
        _assert_finite_positive("grad_norm", metrics.grad_norm)
    finally:
        if collector is not None:
            asyncio.run(collector.shutdown())
        del trainer, collector, bundle
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _common_training_overrides(tmp_path: Path) -> tuple[str, ...]:
    return (
        "trainer.total_epochs=1",
        "trainer.save_freq=0",
        "trainer.log_freq=1",
        f"trainer.output_dir={tmp_path.as_posix()}",
        "trainer.debug.first_step=false",
        "trainer.debug.grad_split=false",
        "trainer.profile=false",
        "trainer.torch_profiler.enabled=false",
        "trainer.rollout_orchestration.require_separate_gpus=false",
        "actor.ppo_epochs=1",
        "actor.gradient_checkpointing=false",
        "actor.timestep_fraction=1.0",
        "actor.ema.enable=false",
        "actor.optim.use_8bit_adam=false",
    )


def _build_runtime_bundle(
    entry: RolloutFamilyEntry,
    cfg: Any,
    device: torch.device,
    dtype: torch.dtype,
) -> Any:
    extractor = import_from_path(entry.runtime_spec_extractor)
    builder = import_from_path(entry.runtime_builder)
    spec = extractor(cfg, device, dtype)
    if entry.family == "janus_pro_r1":
        spec.task_variant = "ar_t2i_r1"
    return builder(spec)


def _build_executor(
    entry: RolloutFamilyEntry,
    model: Any,
    cfg: Any,
) -> Any:
    executor_cls = import_from_path(entry.executor_cls)
    kwargs: dict[str, Any] = {}
    signature = inspect.signature(executor_cls)
    if "sample_batch_size" in signature.parameters:
        kwargs["sample_batch_size"] = int(cfg.rollout.sample_batch_size)
    if "reference_image" in signature.parameters:
        kwargs["reference_image"] = getattr(cfg.model, "reference_image", None)
    return executor_cls(model, **kwargs)


def _skip_unless_case_enabled(case: RealCheckpointCase) -> None:
    if os.environ.get(RUN_REAL_ENV) != "1":
        pytest.skip(f"set {RUN_REAL_ENV}=1 to run real checkpoint RL e2e tests")
    requested = _requested_case_ids()
    if requested != {"cached"} and case.case_id not in requested and "all" not in requested:
        pytest.skip(f"{case.case_id} not selected by {CASE_FILTER_ENV}")


def _requested_case_ids() -> set[str]:
    raw = os.environ.get(CASE_FILTER_ENV, "cached")
    return {item.strip() for item in raw.split(",") if item.strip()}


def _skip_unless_cuda_has_memory(min_gib: float) -> None:
    if not torch.cuda.is_available():
        pytest.skip("real checkpoint RL e2e tests require CUDA")
    total_gib = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    if total_gib < min_gib:
        pytest.skip(
            f"real checkpoint requires at least {min_gib:.1f} GiB CUDA memory; "
            f"device has {total_gib:.1f} GiB",
        )


def _resolve_checkpoint_path(case: RealCheckpointCase, field: CheckpointField) -> Path:
    env_name = _checkpoint_env_name(case, field)
    override = os.environ.get(env_name)
    if override:
        path = Path(override).expanduser()
        if not path.exists():
            pytest.skip(f"{env_name} points to a missing checkpoint: {path}")
        missing = _missing_required_files(path, field.required_files)
        if missing:
            pytest.skip(
                f"{env_name} points to an incomplete checkpoint: {path}; "
                f"missing={missing}",
            )
        return path

    snapshot = _latest_hf_snapshot(field.repo_id, required_files=field.required_files)
    if snapshot is None:
        selected = _requested_case_ids()
        skip_prefix = (
            "cached checkpoint is missing"
            if selected == {"cached"}
            else f"{case.case_id} selected but cached checkpoint is missing"
        )
        pytest.skip(
            f"{skip_prefix}: {field.repo_id}. "
            f"Set {env_name} to a local checkpoint path to override.",
        )
    return snapshot


def _checkpoint_env_name(case: RealCheckpointCase, field: CheckpointField) -> str:
    return (
        "WM_REAL_CHECKPOINT_"
        f"{_env_token(case.case_id)}_"
        f"{_env_token(field.cfg_path)}"
    )


def _latest_hf_snapshot(
    repo_id: str,
    *,
    required_files: tuple[str, ...],
) -> Path | None:
    hub_root = _hf_hub_root()
    repo_cache = hub_root / ("models--" + repo_id.replace("/", "--"))
    snapshots_dir = repo_cache / "snapshots"
    if not snapshots_dir.is_dir():
        return None
    snapshots = [
        path for path in snapshots_dir.iterdir()
        if path.is_dir()
        and _snapshot_has_files(path)
        and not _missing_required_files(path, required_files)
    ]
    if not snapshots:
        return None
    return max(snapshots, key=lambda path: path.stat().st_mtime)


def _hf_hub_root() -> Path:
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return Path(hf_home).expanduser() / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def _snapshot_has_files(path: Path) -> bool:
    return any(child.is_file() or child.is_symlink() for child in _iter_snapshot_children(path))


def _missing_required_files(path: Path, relative_files: tuple[str, ...]) -> list[str]:
    return [name for name in relative_files if not (path / name).exists()]


def _write_reference_image(tmp_path: Path) -> Path:
    from PIL import Image

    path = tmp_path / "reference.png"
    Image.new("RGB", (128, 128), color=(112, 118, 126)).save(path)
    return path


def _iter_snapshot_children(path: Path) -> Iterable[Path]:
    try:
        yield from path.iterdir()
    except OSError:
        return


def _env_token(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").upper()


def _assert_finite_positive(name: str, value: float) -> None:
    _assert_finite(name, value)
    assert float(value) > 0.0, f"{name} must be positive, got {value}"


def _assert_finite(name: str, value: float) -> None:
    assert math.isfinite(float(value)), f"{name} must be finite, got {value}"
