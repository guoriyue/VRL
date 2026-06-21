"""Opt-in RL integration tests against locally cached real model checkpoints."""

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

from tests import ci_envs
from vrl.config.builders import build_configs
from vrl.config.loading import load_config
from vrl.generation import GenerationOutput, GenerationRequest, build_sample_rows
from vrl.generation.execution.planner import build_engine_plan
from vrl.ray.dependencies import import_from_path
from vrl.rollouts.families import RolloutFamilyEntry, get_rollout_family_entry
from vrl.scripts.common.factory import (
    build_algorithm_and_evaluator_from_cfg,
    build_collector_from_cfg,
    build_reward_from_cfg,
    build_rollout_config_from_cfg,
)
from vrl.trainers.online import OnlineTrainer
from vrl.trainers.precision import torch_dtype_for_trainer_precision
from vrl.utils.model_diagnostics import trainable_state_digest

RUN_REAL_ENV = "WM_RUN_REAL_MODEL_TESTS"
CASE_FILTER_ENV = "WM_REAL_MODEL_RL_CASES"


@dataclass(frozen=True, slots=True)
class CheckpointField:
    cfg_path: str
    repo_id: str
    required_files: tuple[str, ...] = ()
    allow_file: bool = False


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
    use_config_reward: bool = False
    replay_runtime_builder: str | None = None
    replay_runtime_spec_extractor: str | None = None
    synthetic_replay_rollout: bool = False
    # Sample->replay log-prob parity bound (GRPO ratio==1 invariant). With
    # ppo_epochs=1 the optimizer steps after the whole timestep loop, so every
    # replayed log-prob is computed against unchanged weights even at lr>0.
    # NFT cases report 0.0 (no evaluator log-probs) and pass trivially.
    logprob_parity_tol: float = 5e-3


CASES: tuple[RealCheckpointCase, ...] = (
    RealCheckpointCase(
        case_id="wan_2_1",
        config="experiment/diffusion/wan_2_1/online_grpo_ocr",
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
            "algorithm.kl_coef=0.0",
            "algorithm.kl_reward_coef=0.0",
            "actor.drop_zero_advantage=false",
            "rollout.n_samples_per_prompt=2",
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
        # wan's bf16 replay carries ~2.6e-3 mean recompute noise (cross-model
        # smoke 2026-06-09) — a known, separately-tracked warn, not the EDM bug.
        logprob_parity_tol=1e-2,
    ),
    RealCheckpointCase(
        case_id="sd3_5",
        config="experiment/diffusion/sd3_5/online_grpo_ocr",
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
            "precision=bf16",
            "actor.gradient_accumulation_steps=0",
            "algorithm.kl_coef=0.0",
            "algorithm.kl_reward_coef=0.0",
            "actor.drop_zero_advantage=false",
            "rollout.n_samples_per_prompt=2",
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
    # New diffusion-RL algorithms (SPRINT_diffusion_algorithm_parity), each run
    # on SD3.5 by swapping only the /recipe/online group. flow_dppo / grpo_guard
    # exercise the §3 old_prev_sample_mean store->replay path on the real
    # executor; on the first step rollout==replay so the trust-region KL is ~0
    # (no masking / ratio_mean_bias), which keeps a live gradient. Their losses
    # report no sample->replay logprob diff, so parity is checked permissively.
    *(
        RealCheckpointCase(
            case_id=f"sd3_5_{_algo}",
            config="experiment/diffusion/sd3_5/online_grpo_ocr",
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
                f"/recipe/online=flow_matching_{_recipe}",
                "model.torch_compile.enable=false",
                "precision=bf16",
                "actor.gradient_accumulation_steps=0",
                # No ref model in this harness; dance_grpo uses GRPO's loss, so a
                # leaked experiment kl_coef>0 would demand one. (flow_dppo /
                # grpo_guard ignore kl_coef in their own loss.)
                "algorithm.kl_coef=0.0",
                "algorithm.kl_reward_coef=0.0",
                "actor.drop_zero_advantage=false",
                "rollout.n_samples_per_prompt=2",
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
            logprob_parity_tol=1.0,  # DPPO/Guard/Dance don't report a logprob diff
        )
        for _algo, _recipe in (
            ("dance_grpo", "dance_grpo"),
            ("flow_dppo", "dppo"),
            ("grpo_guard", "grpo_guard"),
        )
    ),
    RealCheckpointCase(
        case_id="janus_pro",
        config="experiment/ar/janus_pro/online_grpo_ocr",
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
            "algorithm.kl_coef=0.0",
            "algorithm.kl_reward_coef=0.0",
            "actor.drop_zero_advantage=false",
            "rollout.n_samples_per_prompt=2",
            "rollout.rollout_batch_size=1",
            "rollout.max_text_length=64",
            "sampling.image_token_num=4",
            "sampling.image_size=64",
            "sampling.guidance_scale=1.0",
            "sampling.temperature=1.0",
            "sampling.attention_backend=torch_native",
            "sampling.use_ar_scheduler=false",
            "sampling.ar_scheduler_batch_size=null",
        ),
        min_cuda_memory_gib=16.0,
    ),
    RealCheckpointCase(
        case_id="cosmos_predict2",
        config="experiment/diffusion/cosmos_predict2/online_grpo_kling_video_reward",
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
            "algorithm.kl_coef=0.0",
            "algorithm.kl_reward_coef=0.0",
            "actor.drop_zero_advantage=false",
            "rollout.n_samples_per_prompt=2",
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
        use_config_reward=True,
    ),
    RealCheckpointCase(
        case_id="cosmos_predict2_5",
        config="experiment/diffusion/cosmos_predict2_5/online_nft_kling_video_reward",
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
            "algorithm.kl_reward_coef=0.0",
            "actor.drop_zero_advantage=false",
            "rollout.n_samples_per_prompt=2",
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
        use_config_reward=True,
    ),
    RealCheckpointCase(
        case_id="cosmos_anima",
        config="experiment/diffusion/anima_preview3/online_grpo_aesthetic",
        family="cosmos-predict2-anima",
        prompt="anime portrait of a small white sign that says RL",
        checkpoints=(
            CheckpointField(
                cfg_path="model.path",
                repo_id="circlestone-labs/Anima",
                required_files=(
                    "split_files/diffusion_models/anima-preview3-base.safetensors",
                ),
                allow_file=True,
            ),
        ),
        overrides=(
            "model.torch_compile.enable=false",
            "actor.gradient_accumulation_steps=0",
            "algorithm.kl_coef=0.0",
            "algorithm.kl_reward_coef=0.0",
            "actor.drop_zero_advantage=false",
            "rollout.n_samples_per_prompt=2",
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
        min_cuda_memory_gib=28.0,
        replay_runtime_builder=(
            "vrl.models.diffusion.cosmos.anima.runtime:"
            "build_anima_replay_runtime_bundle"
        ),
        replay_runtime_spec_extractor=(
            "vrl.models.diffusion.cosmos.anima.runtime:"
            "extract_anima_replay_runtime_spec"
        ),
        synthetic_replay_rollout=True,
    ),
    RealCheckpointCase(
        case_id="cosmos_anima_safe",
        config="experiment/diffusion/anima_preview3/online_grpo_aesthetic_nsfw_safety",
        family="cosmos-predict2-anima",
        prompt="anime portrait of a small white sign that says RL",
        checkpoints=(
            CheckpointField(
                cfg_path="model.path",
                repo_id="circlestone-labs/Anima",
                required_files=(
                    "split_files/diffusion_models/anima-preview3-base.safetensors",
                ),
                allow_file=True,
            ),
        ),
        overrides=(
            "model.torch_compile.enable=false",
            "actor.gradient_accumulation_steps=0",
            "algorithm.kl_coef=0.0",
            "algorithm.kl_reward_coef=0.0",
            "actor.drop_zero_advantage=false",
            "rollout.n_samples_per_prompt=2",
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
        min_cuda_memory_gib=28.0,
        replay_runtime_builder=(
            "vrl.models.diffusion.cosmos.anima.runtime:"
            "build_anima_replay_runtime_bundle"
        ),
        replay_runtime_spec_extractor=(
            "vrl.models.diffusion.cosmos.anima.runtime:"
            "extract_anima_replay_runtime_spec"
        ),
        synthetic_replay_rollout=True,
    ),
    RealCheckpointCase(
        case_id="nextstep_1",
        config="experiment/ar/nextstep_1/online_grpo_ocr",
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
            "algorithm.kl_coef=0.0",
            "algorithm.kl_reward_coef=0.0",
            "actor.drop_zero_advantage=false",
            "rollout.n_samples_per_prompt=2",
            "rollout.rollout_batch_size=1",
            "rollout.max_text_length=64",
            "rollout.noise_level=1.0",
            "sampling.image_token_num=4",
            "sampling.image_size=64",
            "sampling.num_steps=1",
            "sampling.noise_level=1.0",
            "sampling.guidance_scale=1.0",
            "sampling.attention_backend=torch_native",
            "sampling.use_ar_scheduler=false",
            "sampling.ar_scheduler_batch_size=null",
        ),
        min_cuda_memory_gib=64.0,
    ),
)


class _IndexReward:
    async def score_batch(self, rollouts: list[Any]) -> list[float]:
        return [float(i) for i, _ in enumerate(rollouts)]


class _DirectExecutorGenerationRuntime:
    """Test-only runtime that drives the real family executor without Ray actors.

    Real-checkpoint e2e tests use this to validate replay/trainer integration
    without making Ray scheduling part of the assertion surface.
    """

    def __init__(self, executor: Any) -> None:
        self.executor = executor
        self.current_policy_version = 0

    async def generate(self, request: GenerationRequest) -> GenerationOutput:
        rows = build_sample_rows(request)
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

    async def release(self) -> None:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def is_colocated(self) -> bool:
        return False


class _SyntheticDiffusionReplayCollector:
    """Collector that exercises replay training without full generation assets."""

    def __init__(
        self,
        *,
        model: Any,
        case: RealCheckpointCase,
        cfg: Any,
        device: torch.device,
    ) -> None:
        self.model = model
        self.case = case
        self.cfg = cfg
        self.device = device
        self.runtime = _StaticPolicyRuntime()

    async def collect_unscored(self, prompts: list[str], **kwargs: Any) -> Any:
        return _synthetic_diffusion_replay_batch(
            model=self.model,
            case=self.case,
            cfg=self.cfg,
            prompts=prompts,
            group_size=int(kwargs["group_size"]),
            policy_version=kwargs.get("policy_version"),
            device=self.device,
        )

    async def score_rollouts(self, pendings: Any) -> Any:
        return list(pendings)

    async def release_runtime_memory(self) -> None:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    async def shutdown(self) -> None:
        return None


class _StaticPolicyRuntime:
    current_policy_version = 0


@pytest.mark.e2e
@pytest.mark.parametrize("case", CASES, ids=lambda case: case.case_id)
def test_real_checkpoint_online_rl_updates_trainable_weights(
    case: RealCheckpointCase,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Checks real checkpoint online RL updates trainable weights."""
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
    if case.use_config_reward:
        case_overrides.extend(_ray_reward_overrides(tmp_path))

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
    reward_fn: Any | None = None
    try:
        device = torch.device("cuda")
        built = build_configs(cfg)
        trainer_config = built["trainer"]
        dtype = torch_dtype_for_trainer_precision(trainer_config, torch)
        bundle = _build_runtime_bundle(case, entry, cfg, device, dtype)
        collector_config = build_rollout_config_from_cfg(cfg, entry)
        if case.synthetic_replay_rollout:
            collector = _SyntheticDiffusionReplayCollector(
                model=bundle.model,
                case=case,
                cfg=cfg,
                device=device,
            )
            reward_fn = None
        else:
            executor = _build_executor(entry, bundle.model, cfg)
            reward_fn = (
                build_reward_from_cfg(cfg, built=built, device=str(device))
                if case.use_config_reward else _IndexReward()
            )
            collector = build_collector_from_cfg(
                cfg,
                model=bundle.model,
                reward_fn=reward_fn,
                family=entry,
                collector_config=collector_config,
                runtime=_DirectExecutorGenerationRuntime(executor),
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
        if not case.synthetic_replay_rollout:
            # Sample->replay log-prob parity (GRPO ratio==1). Regression guard
            # for family-specific replay wiring: predict2 sat at mean 13.9
            # before the EDM sigma-domain fix while every metric here still
            # looked alive. Synthetic-replay cases fabricate old log-probs and
            # are excluded.
            assert metrics.logprob_abs_diff_mean < case.logprob_parity_tol, (
                f"{case.case_id}: replay log-prob diverged from collection "
                f"(mean {metrics.logprob_abs_diff_mean:.6f} >= "
                f"{case.logprob_parity_tol}) — broken sample/replay parity"
            )
        if case.use_config_reward:
            _assert_ray_reward_artifacts(tmp_path, reward_fn)
    finally:
        if reward_fn is not None:
            asyncio.run(_shutdown_if_present(reward_fn))
        if collector is not None:
            asyncio.run(collector.shutdown())
        del trainer, collector, reward_fn, bundle
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _common_training_overrides(tmp_path: Path) -> tuple[str, ...]:
    return (
        "trainer.total_epochs=1",
        "trainer.save_freq=0",
        f"trainer.output_dir={tmp_path.as_posix()}",
        "trainer.debug.first_step=false",
        "trainer.profile=false",
        "trainer.torch_profiler.enabled=false",
        "trainer.rollout_orchestration.require_separate_gpus=false",
        "actor.ppo_epochs=1",
        "actor.gradient_checkpointing=false",
        "actor.timestep_fraction=1.0",
        "actor.ema.enable=false",
    )


def build_tensor_mean_model(worker_config):
    """Test RewardModel factory: score = mean of the artifact tensor."""

    def _model(*, artifact, request):
        tensor = torch.load(Path(artifact.path), map_location="cpu").float()
        key = request.score_key.split("+", 1)[0].strip()
        return {key: float(tensor.mean().item())}

    return _model


def _ray_reward_overrides(tmp_path: Path) -> tuple[str, ...]:
    artifact_dir = tmp_path / "reward_artifacts"
    debug_dir = tmp_path / "reward_debug"
    model_factory = "tests.e2e.test_real_checkpoint_rl:build_tensor_mean_model"
    return (
        f"reward.kwargs.kling_video_reward.artifact_dir={artifact_dir.as_posix()}",
        f"reward.kwargs.kling_video_reward.debug_dir={debug_dir.as_posix()}",
        "reward.kwargs.kling_video_reward.release_after_score=true",
        "reward.kwargs.kling_video_reward.num_workers=1",
        "reward.kwargs.kling_video_reward.cpus_per_worker=0.5",
        "reward.kwargs.kling_video_reward.gpus_per_worker=0.0",
        f"reward.kwargs.kling_video_reward.worker_config.model_factory={model_factory}",
        "reward.kwargs.kling_video_reward.worker_config.reward_model_version=e2e-tensor-mean",
    )


def _assert_ray_reward_artifacts(tmp_path: Path, reward_fn: Any) -> None:
    components = getattr(reward_fn, "last_components", {})
    assert "kling_video_reward" in components
    assert len(components["kling_video_reward"]) >= 2
    assert (tmp_path / "reward_artifacts" / "manifest.jsonl").exists()
    assert (tmp_path / "reward_debug" / "kling_video_reward_results.jsonl").exists()


async def _shutdown_if_present(value: Any) -> None:
    shutdown = getattr(value, "shutdown", None)
    if shutdown is not None:
        result = shutdown()
        if inspect.isawaitable(result):
            await result


def _build_runtime_bundle(
    case: RealCheckpointCase,
    entry: RolloutFamilyEntry,
    cfg: Any,
    device: torch.device,
    dtype: torch.dtype,
) -> Any:
    extractor_path = case.replay_runtime_spec_extractor or entry.runtime_spec_extractor
    builder_path = case.replay_runtime_builder or entry.runtime_builder
    extractor = import_from_path(extractor_path)
    builder = import_from_path(builder_path)
    spec = extractor(cfg, device, dtype)
    if entry.family == "janus_pro_r1":
        spec.ar_task = "ar_t2i_r1"
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


def _synthetic_diffusion_replay_batch(
    *,
    model: Any,
    case: RealCheckpointCase,
    cfg: Any,
    prompts: list[str],
    group_size: int,
    policy_version: int | None,
    device: torch.device,
) -> Any:
    from vrl.math.diffusion.flow_matching import sde_step_with_logprob
    from vrl.rollouts.batch import RolloutBatch
    from vrl.trajectory import build_diffusion_trajectory, build_training_view

    num_steps = max(1, int(cfg.sampling.num_steps))
    height = int(cfg.sampling.height)
    width = int(cfg.sampling.width)
    latent_scale = 8
    transformer_cfg = getattr(getattr(model, "transformer", None), "config", None)
    in_channels = int(getattr(transformer_cfg, "in_channels", 16))
    text_embed_dim = int(getattr(transformer_cfg, "text_embed_dim", 1024))
    dtype = _model_tensor_dtype(model)

    if hasattr(model, "set_num_steps"):
        model.set_num_steps(num_steps)
    else:
        model.scheduler.set_timesteps(num_steps, device=device)

    request = GenerationRequest(
        request_id=f"{case.case_id}:synthetic-replay",
        family=case.family,
        task="text_to_image",
        prompts=list(prompts),
        samples_per_prompt=group_size,
        sampling={
            "height": height,
            "width": width,
            "num_frames": 1,
            "num_steps": num_steps,
            "guidance_scale": float(cfg.sampling.guidance_scale),
            "cfg": bool(cfg.sampling.cfg),
        },
        return_artifacts={"trajectory"},
        policy_version=None if policy_version is None else int(policy_version),
    )
    sample_rows = build_sample_rows(request)
    batch_size = len(sample_rows)
    generator = torch.Generator(device=device)
    generator.manual_seed(1729)

    observations = torch.randn(
        (
            batch_size,
            num_steps,
            in_channels,
            1,
            max(1, height // latent_scale),
            max(1, width // latent_scale),
        ),
        generator=generator,
        device=device,
        dtype=torch.float32,
    )
    prompt_embeds = torch.randn(
        (batch_size, 512, text_embed_dim),
        generator=generator,
        device=device,
        dtype=dtype,
    )
    padding_mask = torch.zeros(
        (batch_size, 1, height, width),
        device=device,
        dtype=dtype,
    )
    replay_tensors: dict[str, Any] = {
        "prompt_embeds": prompt_embeds,
        "negative_prompt_embeds": None,
        "padding_mask": padding_mask,
    }
    context = {
        "guidance_scale": float(cfg.sampling.guidance_scale),
        "cfg": bool(cfg.sampling.cfg),
        "model_family": case.family,
    }
    timesteps = model.scheduler.timesteps[:num_steps].to(device)
    timesteps = timesteps.unsqueeze(0).expand(batch_size, -1).clone()
    noise_level = float(cfg.rollout.noise_level)
    sde_type = str(getattr(getattr(cfg.rollout, "sde", None), "type", "flow_grpo"))

    action_steps: list[torch.Tensor] = []
    old_log_prob_steps: list[torch.Tensor] = []
    with torch.no_grad():
        for step_idx in range(num_steps):
            state = model.restore_eval_state(
                replay_tensors,
                context,
                observations[:, step_idx],
                step_idx,
            )
            noise_pred = model.forward_step(state, step_idx)["noise_pred"]
            sde = sde_step_with_logprob(
                model.scheduler,
                noise_pred,
                timesteps[:, step_idx],
                observations[:, step_idx],
                generator=generator,
                noise_level=noise_level,
                sde_type=sde_type,
            )
            action_steps.append(sde.prev_sample.detach())
            old_log_prob_steps.append(sde.log_prob.detach())

    actions = torch.stack(action_steps, dim=1)
    old_log_prob = torch.stack(old_log_prob_steps, dim=1)
    kl = torch.zeros_like(old_log_prob)
    trajectory = build_diffusion_trajectory(
        request=request,
        sample_rows=sample_rows,
        observations=observations.detach(),
        actions=actions,
        old_log_prob=old_log_prob,
        timesteps=timesteps,
        kl=kl,
        replay_tensors=replay_tensors,
        context=context,
    )
    rewards = torch.arange(batch_size, device=device, dtype=torch.float32)
    return RolloutBatch(
        observations=observations.detach(),
        actions=actions,
        rewards=rewards,
        dones=torch.ones(batch_size, device=device, dtype=torch.bool),
        group_ids=trajectory.group_ids,
        extras={},
        context=dict(trajectory.context),
        videos=None,
        prompts=[row.prompt for row in sample_rows],
        trajectory=trajectory,
        training_view=build_training_view(trajectory, primary_segment="denoise"),
    )


def _model_tensor_dtype(model: Any) -> torch.dtype:
    for parameter in model.parameters():
        return parameter.dtype
    return torch.bfloat16


def _skip_unless_case_enabled(case: RealCheckpointCase) -> None:
    if not ci_envs.WM_RUN_REAL_MODEL_TESTS:
        pytest.skip(f"set {RUN_REAL_ENV}=1 to run real checkpoint RL e2e tests")
    requested = _requested_case_ids()
    if requested != {"cached"} and case.case_id not in requested and "all" not in requested:
        pytest.skip(f"{case.case_id} not selected by {CASE_FILTER_ENV}")


def _requested_case_ids() -> set[str]:
    raw = ci_envs.WM_REAL_MODEL_RL_CASES
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
        if path.is_file() and field.allow_file:
            return path
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
            f"Set {env_name} to a filesystem checkpoint path to override.",
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
