"""Native Cosmos/CPS/GRPO through a complete CP OnlineTrainer.step."""

import asyncio
import copy
import json
import os
import random
import time
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from diffusers import CosmosTransformer3DModel, UniPCMultistepScheduler

from tests.trainers._strategy_policies import free_port
from tests.trainers.online._collector_control import CollectorControlFake
from vrl.algorithms.grpo.continuous import GRPO, GRPOConfig
from vrl.config.precision import RolePrecision
from vrl.generation import GenerationRequest, GenerationSampleRow
from vrl.math.denoise.flow_matching import sde_step_with_logprob
from vrl.models.families.cosmos.predict2_5.model import (
    CosmosPredict25Model,
    CosmosPredict25ReplayModel,
    CosmosPredict25SamplingState,
)
from vrl.models.interfaces.runtime import ModelBuild, RuntimeBundle
from vrl.models.precision import apply_float32_precision, fixed_row_linear_compute
from vrl.rollouts.batch import RolloutBatch
from vrl.rollouts.evaluators.denoise.sde_logprob import DenoiseSDELogProbEvaluator
from vrl.rollouts.orchestration.strict_on_policy import ContextParallelStrictRolloutSchedule
from vrl.trainers.checkpointing import (
    TrainingCheckpoint,
    _require_equal_tensor_tree,
    restore_rng_state,
    restore_training_checkpoint,
    save_training_checkpoint,
)
from vrl.trainers.core.types import EMAConfig, OptimConfig
from vrl.trainers.distributed import DistributedTrainingContext, init_training_process_group
from vrl.trainers.online.config import OnlineBatchPlan, TrainerConfig
from vrl.trainers.online.trainer import OnlineTrainer
from vrl.trainers.strategy import ContextParallelStrategy
from vrl.trajectory.builders import build_diffusion_trajectory


class _Collector(CollectorControlFake):
    def __init__(self, model, *, released=False):
        self.model = model
        self.released = released
        self.calls = 0
        self.versions = []
        self.generation_runtime = SimpleNamespace(
            current_policy_version=0, requires_driver_model_offload=False
        )

    async def evaluate_rollout(self, batches):
        return list(batches)

    async def generate_rollout(self, request):
        prompts = request.inputs
        kwargs = request.options
        self.calls += 1
        self.versions.append(kwargs.get("policy_version"))
        count = kwargs["group_size"]
        model = self.model
        device = model.device
        channels, frames, height, width = (16, 9, 60, 104) if self.released else (4, 3, 4, 4)
        tokens, text_dim = (512, 100352) if self.released else (3, 32)
        dtype = torch.bfloat16 if self.released else torch.float32
        latents = torch.randn(count, channels, frames, height, width, device=device, dtype=dtype)
        embeds = torch.randn(count, tokens, text_dim, device=device, dtype=dtype)
        state = CosmosPredict25SamplingState(
            latents=latents,
            timesteps=model.scheduler.timesteps,
            scheduler=model.scheduler,
            prompt_embeds=embeds,
            negative_prompt_embeds=torch.randn_like(embeds) if self.released else None,
            guidance_scale=5.0 if self.released else 1.0,
            do_cfg=self.released,
            cond_latent=torch.zeros_like(latents),
            cond_mask=torch.zeros(count, 1, frames, height, width, device=device, dtype=dtype),
            cond_indicator=torch.zeros(count, 1, frames, 1, 1, device=device, dtype=dtype),
            padding_mask=torch.zeros(1, 1, height, width, device=device, dtype=dtype),
            height=height * 8,
            width=width * 8,
            num_frames=(frames - 1) * 4 + 1,
            fps=16,
        )
        context = model.export_batch_context(state)
        replay = model.export_replay_tensors(state)
        observations, actions, logprobs, times = [], [], [], []
        branches = [
            module
            for name, module in model.transformer.named_modules()
            if isinstance(module, torch.nn.Linear) and (".lora_A." in name or ".lora_B." in name)
        ]
        with torch.no_grad(), fixed_row_linear_compute(model.transformer, fp32_modules=branches):
            for index in range(2):
                if self.released:
                    print(f"rollout collection={self.calls} transition={index} start", flush=True)
                prediction = model.forward_step(state, index)["noise_pred"]
                time = model.scheduler.timesteps[index].expand(count)
                draw = sde_step_with_logprob(
                    model.scheduler,
                    prediction,
                    time,
                    state.latents,
                    noise_level=0.7,
                    sde_type="cps",
                )
                observations.append(state.latents)
                actions.append(draw.prev_sample)
                logprobs.append(draw.log_prob)
                times.append(time)
                state.latents = draw.prev_sample
        request = GenerationRequest(
            request_id=f"composition-{self.calls}",
            family="cosmos-predict2.5",
            task="t2v",
            inputs=list(prompts),
            samples_per_prompt=count,
        )
        rows = [
            GenerationSampleRow(
                prompt_index=0,
                sample_index=i,
                prompt=str(prompts[0]),
                sample_id=f"{request.request_id}-{i}",
            )
            for i in range(count)
        ]
        old = torch.stack(logprobs, dim=1)
        self.last_actions = torch.stack(actions, dim=1).detach().cpu()
        trajectory = build_diffusion_trajectory(
            request=request,
            sample_rows=rows,
            observations=torch.stack(observations, dim=1),
            actions=torch.stack(actions, dim=1),
            old_log_prob=old,
            timesteps=torch.stack(times, dim=1),
            kl=torch.zeros_like(old),
            replay_tensors=replay,
            context=context,
        )
        return RolloutBatch(
            rewards=torch.arange(count, dtype=torch.float32),
            group_ids=torch.zeros(count, dtype=torch.long),
            trajectory=trajectory,
            context=context,
        )


class _Syncer:
    def __init__(self, collector):
        self.collector = collector
        self.calls = 0

    @property
    def current_policy_version(self):
        return self.collector.generation_runtime.current_policy_version

    async def push(self, state):
        parameters = dict(self.collector.model.named_parameters())
        with torch.no_grad():
            for name, value in state.items():
                parameters[name].copy_(value)
        self.calls += 1
        self.collector.generation_runtime.current_policy_version += 1


def _worker(rank, rendezvous, root, cuda=False, phase=None, released_model=None):
    torch.set_num_threads(1)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)
    apply_float32_precision("ieee")
    if cuda:
        torch.cuda.set_device(rank)
    device = torch.device("cuda", rank) if cuda else torch.device("cpu")
    rollout_device = torch.device("cuda", 2) if cuda and rank == 0 else torch.device("cpu")
    context = DistributedTrainingContext("context_parallel", rank, 2, device)
    if cuda:
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(rendezvous)
        init_training_process_group(context, backend="nccl")
    else:
        dist.init_process_group(
            "gloo",
            init_method=rendezvous,
            rank=rank,
            world_size=2,
            timeout=timedelta(seconds=120),
        )
    strategy = ContextParallelStrategy(context, cp_size=2)
    trainer = None
    try:
        torch.manual_seed(31)
        if released_model:
            assert cuda
            print(f"rank={rank} phase={phase} load={released_model}", flush=True)
        transformer = (
            CosmosTransformer3DModel.from_pretrained(
                released_model,
                subfolder="transformer",
                torch_dtype=torch.bfloat16,
                local_files_only=True,
            )
            if released_model
            else CosmosTransformer3DModel(
                in_channels=5,
                out_channels=4,
                num_attention_heads=2,
                attention_head_dim=16,
                num_layers=1,
                mlp_ratio=2,
                text_embed_dim=32,
                adaln_lora_dim=8,
                max_size=(4, 16, 16),
                patch_size=(1, 2, 2),
                concat_padding_mask=True,
                extra_pos_embed_type=None,
            )
        )
        scheduler = (
            UniPCMultistepScheduler.from_pretrained(
                released_model,
                subfolder="scheduler",
                local_files_only=True,
            )
            if released_model
            else UniPCMultistepScheduler(
                prediction_type="flow_prediction",
                use_flow_sigmas=True,
                use_karras_sigmas=True,
                sigma_max=200,
                sigma_min=0.01,
            )
        )
        scheduler.set_timesteps(20 if released_model else 4, device=device)
        precision = RolePrecision(
            dtype="bf16" if released_model else "fp32",
            float32_precision="ieee",
            outer_autocast=bool(released_model),
        )
        model = CosmosPredict25ReplayModel(
            transformer=transformer.to(device), scheduler=scheduler, device=device
        )
        model.precision = precision
        model.apply_lora(
            ModelBuild(
                model_name_or_path=released_model or "tiny-cosmos",
                revision=Path(released_model).name if released_model else "test",
                family="cosmos-predict2.5",
                device=device,
                parameter_dtype=torch.bfloat16 if released_model else torch.float32,
                precision=precision,
                model_config={
                    "use_lora": True,
                    "lora": {
                        "rank": 32 if released_model else 2,
                        "alpha": 64 if released_model else 4,
                        "target_modules": ["to_q", "to_k", "to_v", "to_out.0"]
                        + (["ff.net.0.proj", "ff.net.2"] if released_model else []),
                    },
                },
            )
        )
        rollout_scheduler = UniPCMultistepScheduler.from_config(scheduler.config)
        rollout_scheduler.set_timesteps(20 if released_model else 4, device=rollout_device)
        rollout = CosmosPredict25Model(
            pipeline=SimpleNamespace(
                transformer=copy.deepcopy(model.transformer).to(rollout_device),
                scheduler=rollout_scheduler,
                device=rollout_device,
            ),
            device=rollout_device,
        )
        rollout.precision = precision
        if released_model:
            model.transformer.enable_gradient_checkpointing()
            model.transformer.train()
        collector = _Collector(rollout, released=bool(released_model))
        syncer = _Syncer(collector)
        before = {
            name: p.detach().clone() for name, p in model.named_parameters() if p.requires_grad
        }
        trainer = OnlineTrainer(
            algorithm=GRPO(GRPOConfig(kl_coef=0)),
            collector=collector,
            evaluator=DenoiseSDELogProbEvaluator(scheduler, noise_level=0.7, sde_type="cps"),
            model=model,
            strategy=strategy,
            device=device,
            weight_syncer=syncer,
            sync_state_getter=lambda: {
                name: p.detach().cpu().clone()
                for name, p in model.named_parameters()
                if p.requires_grad
            },
            config=TrainerConfig(
                batch_plan=OnlineBatchPlan(prompts_per_batch=1, n_samples_per_prompt=2),
                timestep_fraction=1.0,
                drop_zero_advantage=False,
                optim=OptimConfig(lr=1e-4),
                ema=EMAConfig(enable=phase is not None, decay=0.9, update_interval=1),
                train_precision="bf16" if released_model else "no",
                output_dir=str(Path(root) / f"rank-{rank}"),
            ),
        )
        trainer.rollout_schedule = ContextParallelStrictRolloutSchedule(
            owner=trainer.rollout_schedule if rank == 0 else None,
            groups=strategy.groups,
            spool_dir=root,
        )
        bundle = RuntimeBundle(
            model=model,
            trainable_modules={"transformer": model.transformer},
            scheduler=scheduler,
            raw_handle=None,
            precision=precision,
        )
        identity = (
            {"schema": "released-cosmos-cp-composition/v1", "checkpoint": released_model}
            if released_model
            else {"schema": "tiny-cosmos-cp-composition/v1"}
        )
        checkpoint_dir = Path(root) / "checkpoint-1"
        if phase == "resume":
            checkpoint = TrainingCheckpoint.load(checkpoint_dir)
            restore_training_checkpoint(
                checkpoint,
                trainer=trainer,
                bundle=bundle,
                family="cosmos-predict2.5",
                expected_model_identity=identity,
                strict=True,
            )
            restore_rng_state(checkpoint.rng_state, rank=rank, world_size=2, strict=True)
            assert trainer.state.step == 1
            assert trainer._ema.has_updates
        updates = 1 if phase == "resume" else 2
        measurements = []
        if released_model:
            torch.cuda.reset_peak_memory_stats(device)
            if rank == 0:
                torch.cuda.reset_peak_memory_stats(rollout_device)
        for index in range(updates):
            started = time.perf_counter()
            if released_model:
                print(
                    f"rank={rank} phase={phase} update={trainer.state.step + 1} start", flush=True
                )
            metrics = asyncio.run(trainer.step(["controlled text"]))
            assert metrics.grad_norm > 0
            assert metrics.initial_replay.finite
            assert metrics.initial_replay.logprob_abs_diff_max <= 1e-3
            if released_model:
                torch.cuda.synchronize(device)
                measurement = {
                    "step": trainer.state.step,
                    "seconds": time.perf_counter() - started,
                    "grad_norm": metrics.grad_norm,
                    "replay_max_abs": metrics.initial_replay.logprob_abs_diff_max,
                    "training_peak_allocated": torch.cuda.max_memory_allocated(device),
                    "rollout_peak_allocated": torch.cuda.max_memory_allocated(rollout_device)
                    if rank == 0
                    else None,
                }
                measurements.append(measurement)
                print(json.dumps({"rank": rank, "phase": phase, **measurement}), flush=True)
            if phase == "control" and index == 0:
                assert trainer._ema.has_updates
                save_training_checkpoint(
                    checkpoint_dir,
                    trainer=trainer,
                    bundle=bundle,
                    family="cosmos-predict2.5",
                    model_identity=identity,
                    progress={"next_step": 1},
                    strategy=strategy,
                )
                strategy.collectives.barrier()
        assert next(model.parameters()).device == device
        assert next(rollout.parameters()).device == rollout_device
        assert trainer.state.step == 2
        assert trainer.state.global_step >= 2
        assert any(
            not torch.equal(before[name], p)
            for name, p in model.named_parameters()
            if name in before
        )
        assert collector.calls == (updates if rank == 0 else 0)
        assert collector.versions == (list(range(1, updates + 1)) if rank == 0 else [])
        assert syncer.calls == (updates + 1 if rank == 0 else 0)
        for parameter in model.parameters():
            reference = parameter.detach().clone()
            dist.broadcast(reference, src=0)
            torch.testing.assert_close(parameter, reference, rtol=0, atol=0)
        if phase is not None:
            assert trainer._ema.num_updates == trainer.state.global_step
            trainable = [p for p in model.parameters() if p.requires_grad]
            assert any(
                not torch.equal(shadow, parameter)
                for shadow, parameter in zip(trainer._ema.ema_parameters, trainable, strict=True)
            )
            outcome = {
                "model": model.state_dict(),
                "trainer": trainer.state_dict(),
                "actions": collector.last_actions if rank == 0 else None,
                "next_rng": {
                    "torch": torch.rand(8),
                    "python": random.random(),
                    "numpy": torch.from_numpy(np.random.rand(8)),
                    "training": torch.rand(8, device=device),
                    "rollout": torch.rand(8, device=rollout_device) if rank == 0 else None,
                },
            }
            reference_path = Path(root) / f"control-rank-{rank}.pt"
            if phase == "control":
                torch.save(outcome, reference_path)
            else:
                reference = torch.load(reference_path, map_location="cpu", weights_only=False)
                _require_equal_tensor_tree(reference, outcome, label="fresh-process resume")
        if released_model:
            (Path(root) / f"{phase}-rank-{rank}.json").write_text(
                json.dumps(
                    {
                        "phase": phase,
                        "rank": rank,
                        "model": released_model,
                        "measurements": measurements,
                        "assertions_passed": True,
                    },
                    indent=2,
                )
                + "\n"
            )
    finally:
        if trainer is not None:
            asyncio.run(trainer.rollout_schedule.shutdown())
        strategy.shutdown()


def test_native_cosmos_cp_online_step(tmp_path):
    mp.spawn(_worker, args=((tmp_path / "gloo").as_uri(), str(tmp_path)), nprocs=2, join=True)


@pytest.mark.distributed
@pytest.mark.gpu
def test_native_cosmos_cp_online_step_disjoint_cuda(tmp_path):
    if torch.cuda.device_count() < 3:
        pytest.skip("requires two training GPUs and one disjoint rollout GPU")
    mp.spawn(_worker, args=(free_port(), str(tmp_path), True), nprocs=2, join=True)


def _run_resume_comparison(tmp_path, *, cuda):
    for phase in ("control", "resume"):
        rendezvous = free_port() if cuda else (tmp_path / phase).as_uri()
        mp.spawn(
            _worker,
            args=(rendezvous, str(tmp_path), cuda, phase),
            nprocs=2,
            join=True,
        )


def test_native_cosmos_cp_checkpoint_resume_ema(tmp_path):
    _run_resume_comparison(tmp_path, cuda=False)


@pytest.mark.distributed
@pytest.mark.gpu
def test_native_cosmos_cp_checkpoint_resume_ema_disjoint_cuda(tmp_path):
    if torch.cuda.device_count() < 3:
        pytest.skip("requires two training GPUs and one disjoint rollout GPU")
    _run_resume_comparison(tmp_path, cuda=True)
