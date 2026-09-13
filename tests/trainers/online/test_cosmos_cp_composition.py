"""Native Cosmos/CPS/GRPO through a complete CP OnlineTrainer.step."""

import asyncio
import copy
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from diffusers import CosmosTransformer3DModel, UniPCMultistepScheduler

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
from vrl.models.interfaces.runtime import ModelBuild
from vrl.models.precision import fixed_row_linear_compute
from vrl.rollouts.batch import RolloutBatch
from vrl.rollouts.evaluators.denoise.sde_logprob import DiffusionSDELogProbEvaluator
from vrl.rollouts.orchestration.strict_on_policy import ContextParallelStrictRolloutSchedule
from vrl.trainers.core.types import EMAConfig, OptimConfig
from vrl.trainers.distributed import DistributedTrainingContext
from vrl.trainers.online import OnlineTrainer
from vrl.trainers.online.config import OnlineBatchPlan, TrainerConfig
from vrl.trainers.strategy import ContextParallelStrategy
from vrl.trajectory import build_diffusion_trajectory


class _Collector(CollectorControlFake):
    def __init__(self, model):
        self.model = model
        self.calls = 0
        self.versions = []
        self.generation_runtime = SimpleNamespace(
            current_policy_version=0, requires_driver_model_offload=False
        )

    async def score_rollouts(self, batches):
        return list(batches)

    async def collect_unscored(self, prompts, **kwargs):
        self.calls += 1
        self.versions.append(kwargs.get("policy_version"))
        count = kwargs["group_size"]
        model = self.model
        latents = torch.randn(count, 4, 3, 4, 4)
        state = CosmosPredict25SamplingState(
            latents=latents,
            timesteps=model.scheduler.timesteps,
            scheduler=model.scheduler,
            prompt_embeds=torch.randn(count, 3, 32),
            negative_prompt_embeds=None,
            guidance_scale=1.0,
            do_cfg=False,
            cond_latent=torch.zeros_like(latents),
            cond_mask=torch.zeros(count, 1, 3, 4, 4),
            cond_indicator=torch.zeros(count, 1, 3, 1, 1),
            padding_mask=torch.zeros(1, 1, 4, 4),
            height=32,
            width=32,
            num_frames=9,
            fps=16,
        )
        context = model.export_batch_context(state)
        replay = model.export_replay_tensors(state)
        observations, actions, logprobs, times = [], [], [], []
        with torch.no_grad(), fixed_row_linear_compute(model.transformer):
            for index in range(2):
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


def _worker(rank, rendezvous, root):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", init_method=rendezvous, rank=rank, world_size=2, timeout=timedelta(seconds=120)
    )
    strategy = ContextParallelStrategy(
        DistributedTrainingContext("context_parallel", rank, 2, torch.device("cpu")), cp_size=2
    )
    trainer = None
    try:
        torch.manual_seed(31)
        transformer = CosmosTransformer3DModel(
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
        scheduler = UniPCMultistepScheduler(
            prediction_type="flow_prediction",
            use_flow_sigmas=True,
            use_karras_sigmas=True,
            sigma_max=200,
            sigma_min=0.01,
        )
        scheduler.set_timesteps(4)
        precision = RolePrecision(dtype="fp32", float32_precision="ieee", outer_autocast=False)
        model = CosmosPredict25ReplayModel(
            transformer=transformer, scheduler=scheduler, device=torch.device("cpu")
        )
        model.precision = precision
        model.apply_lora(
            ModelBuild(
                model_name_or_path="tiny-cosmos",
                revision="test",
                family="cosmos-predict2.5",
                device=torch.device("cpu"),
                parameter_dtype=torch.float32,
                precision=precision,
                model_config={
                    "use_lora": True,
                    "lora": {
                        "rank": 2,
                        "alpha": 4,
                        "target_modules": ["to_q", "to_k", "to_v", "to_out.0"],
                    },
                },
            )
        )
        rollout = CosmosPredict25Model(
            pipeline=SimpleNamespace(
                transformer=copy.deepcopy(model.transformer),
                scheduler=copy.deepcopy(scheduler),
                device=torch.device("cpu"),
            ),
            device=torch.device("cpu"),
        )
        rollout.precision = precision
        collector = _Collector(rollout)
        syncer = _Syncer(collector)
        before = {
            name: p.detach().clone() for name, p in model.named_parameters() if p.requires_grad
        }
        trainer = OnlineTrainer(
            algorithm=GRPO(GRPOConfig(kl_coef=0)),
            collector=collector,
            evaluator=DiffusionSDELogProbEvaluator(scheduler, noise_level=0.7, sde_type="cps"),
            model=model,
            strategy=strategy,
            device="cpu",
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
                ema=EMAConfig(),
                train_precision="no",
                output_dir=str(Path(root) / f"rank-{rank}"),
            ),
        )
        trainer.rollout_schedule = ContextParallelStrictRolloutSchedule(
            owner=trainer.rollout_schedule if rank == 0 else None,
            groups=strategy.groups,
            spool_dir=root,
        )
        for _ in range(2):
            metrics = asyncio.run(trainer.step(["controlled text"]))
            assert metrics.grad_norm > 0
            assert metrics.initial_replay.finite
            assert metrics.initial_replay.logprob_abs_diff_max <= 1e-3
        assert trainer.state.step == 2
        assert trainer.state.global_step >= 2
        assert any(
            not torch.equal(before[name], p)
            for name, p in model.named_parameters()
            if name in before
        )
        assert collector.calls == (2 if rank == 0 else 0)
        assert collector.versions == ([1, 2] if rank == 0 else [])
        assert syncer.calls == (3 if rank == 0 else 0)
        for parameter in model.parameters():
            reference = parameter.detach().clone()
            dist.broadcast(reference, src=0)
            torch.testing.assert_close(parameter, reference, rtol=0, atol=0)
    finally:
        if trainer is not None:
            asyncio.run(trainer.rollout_schedule.shutdown())
        strategy.shutdown()


def test_native_cosmos_cp_online_step(tmp_path):
    mp.spawn(_worker, args=((tmp_path / "gloo").as_uri(), str(tmp_path)), nprocs=2, join=True)
