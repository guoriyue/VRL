"""Tiny VDN native executor/evaluator compose with OnlineTrainer and cold resume.

The collector control and rewards are controlled doubles; model computation,
sampling, GRPO, optimizer, weight delivery and checkpoint APIs are real.
"""

import asyncio
import os
from types import SimpleNamespace

import pytest
import torch

from tests.models.steps.denoise.fixtures import build_tiny_vdn_h3_model, stamp_model_precision
from tests.trainers.online._collector_control import CollectorControlFake
from vrl.algorithms.grpo.continuous import GRPO, GRPOConfig
from vrl.generation import GenerationRequest, GenerationSampleRow
from vrl.generation.execution.sample_batches import GenerationSampleBatch
from vrl.models.families.vdn_h3.model import VDNH3Model, VDNH3ReplayModel
from vrl.models.families.vdn_h3.runtime import VDNH3BatchExecutor
from vrl.models.interfaces.runtime import RuntimeBundle
from vrl.rollouts.batch import RolloutBatch
from vrl.rollouts.evaluators.denoise.sde_logprob import DenoiseSDELogProbEvaluator
from vrl.trainers.checkpointing import (
    TrainingCheckpoint,
    restore_rng_state,
    restore_training_checkpoint,
    save_training_checkpoint,
)
from vrl.trainers.core.types import EMAConfig, OptimConfig
from vrl.trainers.distributed import DistributedTrainingContext
from vrl.trainers.online.config import OnlineBatchPlan, TrainerConfig
from vrl.trainers.online.trainer import OnlineTrainer
from vrl.trainers.strategy import SingleProcessStrategy
from vrl.trajectory.builders import build_diffusion_trajectory

pytest.importorskip("src.models.hybrid_attention")


class _Collector(CollectorControlFake):
    def __init__(self, model):
        self.model = model
        self.executor = VDNH3BatchExecutor(model)
        self.generation_runtime = SimpleNamespace(
            current_policy_version=0, requires_driver_model_offload=False
        )
        self.versions = []

    async def evaluate_rollout(self, batches):
        return list(batches)

    async def generate_rollout(self, prepared, **kwargs):
        assert prepared.options["group_size"] == 2
        self.versions.append(prepared.options["policy_version"])
        request = GenerationRequest(
            request_id="vdn-online-test",
            family="vdn_h3",
            task="t2v",
            inputs=list(prepared.inputs),
            samples_per_prompt=2,
            sampling={
                "num_steps": 3,
                "height": 16,
                "width": 16,
                "num_frames": 8,
                "fps": 24,
                "guidance_scale": 1.0,
                "max_sequence_length": 8,
                "seed": 17,
            },
        )
        with torch.no_grad():
            results = [
                self.executor.forward_batch(
                    request,
                    GenerationSampleBatch(prompt_index=0, sample_start=index, sample_count=1),
                )
                for index in range(2)
            ]
        self.last_actions = torch.cat([result.actions for result in results]).clone()
        self.params = self.executor.parse_sampling_params(request)
        trajectory = build_diffusion_trajectory(
            request=request,
            sample_rows=[
                GenerationSampleRow(
                    prompt_index=0,
                    sample_index=index,
                    prompt=str(prepared.inputs[0]),
                    sample_id=f"vdn-{index}",
                )
                for index in range(2)
            ],
            observations=torch.cat([result.observations for result in results]),
            actions=self.last_actions,
            old_log_prob=torch.cat([result.log_probs for result in results]),
            timesteps=torch.cat([result.timesteps for result in results]),
            kl=torch.cat([result.kl for result in results]),
            replay_tensors={
                key: torch.cat([result.replay_tensors[key] for result in results])
                for key in results[0].replay_tensors
            },
            context=results[0].context,
        )
        return RolloutBatch(
            rewards=torch.tensor([0.0, 1.0]),
            group_ids=torch.zeros(2, dtype=torch.long),
            trajectory=trajectory,
            context=results[0].context,
        )


class _Syncer:
    def __init__(self, collector):
        self.collector = collector

    @property
    def current_policy_version(self):
        return self.collector.generation_runtime.current_policy_version

    async def push(self, state):
        parameters = dict(self.collector.model.named_parameters())
        with torch.no_grad():
            for name, value in state.items():
                parameters[name].copy_(value)
        self.collector.generation_runtime.current_policy_version += 1


def _build(root, *, device):
    rollout_components = build_tiny_vdn_h3_model(seed=0).pipeline
    for name in ("transformer", "text_encoder", "vae", "audio_vae"):
        getattr(rollout_components, name).to(device)
    rollout = VDNH3Model(pipeline=rollout_components, device=device)
    stamp_model_precision(rollout)
    components = build_tiny_vdn_h3_model(seed=1).pipeline
    model = VDNH3ReplayModel(
        transformer=components.transformer.to(device),
        scheduler=components.scheduler,
        audio_scheduler=components.audio_scheduler,
        device=device,
    )
    stamp_model_precision(model)
    model.transformer.load_state_dict(rollout.transformer.state_dict(), strict=True)
    model.transformer.requires_grad_(False)
    model.transformer.transformer_blocks[0].attn.to_out_linear.requires_grad_(True)
    model.set_num_steps(3)
    collector = _Collector(rollout)
    trainer = OnlineTrainer(
        algorithm=GRPO(GRPOConfig(kl_coef=0.0)),
        collector=collector,
        evaluator=DenoiseSDELogProbEvaluator(model.scheduler),
        model=model,
        strategy=SingleProcessStrategy(
            DistributedTrainingContext(
                strategy="single_process",
                rank=0,
                world_size=1,
                device=device,
            )
        ),
        device=device,
        weight_syncer=_Syncer(collector),
        sync_state_getter=lambda: {
            name: p.detach().cpu().clone()
            for name, p in model.named_parameters()
            if p.requires_grad
        },
        config=TrainerConfig(
            batch_plan=OnlineBatchPlan(
                prompts_per_batch=1, n_samples_per_prompt=2, training_microbatch_size=1
            ),
            timestep_fraction=1.0,
            drop_zero_advantage=False,
            optim=OptimConfig(lr=1e-4, weight_decay=0.0),
            ema=EMAConfig(enable=True, decay=0.9, update_interval=1),
            train_precision="no",
            output_dir=str(root),
        ),
    )
    bundle = RuntimeBundle(
        model=model,
        trainable_modules={"transformer": model.transformer},
        scheduler=model.scheduler,
        raw_handle=None,
        precision=model.precision,
    )
    return trainer, collector, bundle


def _verify_update_and_resume(tmp_path, *, device):
    trainer, collector, bundle = _build(tmp_path / "control", device=device)
    before = {
        name: p.detach().clone() for name, p in trainer.model.named_parameters() if p.requires_grad
    }
    first = asyncio.run(trainer.step(["a wooden block"]))
    assert first.grad_norm > 0 and first.initial_replay.finite
    assert first.initial_replay.logprob_abs_diff_max < 1e-6
    assert any(
        not torch.equal(before[name], p)
        for name, p in trainer.model.named_parameters()
        if name in before
    )
    path = tmp_path / "checkpoint-1"
    identity = {"schema": "tiny-vdn-grpo-composition/v1"}
    save_training_checkpoint(
        path,
        trainer=trainer,
        bundle=bundle,
        family="vdn_h3",
        model_identity=identity,
        progress={"next_step": 1},
        strategy=trainer._strategy,
    )
    second = asyncio.run(trainer.step(["a wooden block"]))
    assert second.grad_norm > 0 and second.initial_replay.logprob_abs_diff_max < 1e-6
    assert collector.versions == [1, 2]
    restored, resumed_collector, resumed_bundle = _build(tmp_path / "resumed", device=device)
    checkpoint = TrainingCheckpoint.load(path)
    restore_training_checkpoint(
        checkpoint,
        trainer=restored,
        bundle=resumed_bundle,
        family="vdn_h3",
        expected_model_identity=identity,
        strict=True,
    )
    restore_rng_state(checkpoint.rng_state, rank=0, world_size=1, strict=True)
    resumed = asyncio.run(restored.step(["a wooden block"]))
    assert resumed.grad_norm == second.grad_norm
    assert resumed.initial_replay.logprob_abs_diff_max < 1e-6
    assert trainer.state.step == restored.state.step == 2
    torch.testing.assert_close(
        collector.last_actions, resumed_collector.last_actions, rtol=0, atol=0
    )
    torch.testing.assert_close(
        trainer.model.state_dict(), restored.model.state_dict(), rtol=0, atol=0
    )
    left, right = trainer.state_dict(), restored.state_dict()
    torch.testing.assert_close(
        left["optimizer"]["state"], right["optimizer"]["state"], rtol=0, atol=0
    )
    torch.testing.assert_close(left["ema"], right["ema"], rtol=0, atol=0)


def test_online_grpo_sync_accumulation_and_checkpoint_resume(tmp_path):
    _verify_update_and_resume(tmp_path, device=torch.device("cpu"))


@pytest.mark.gpu
@pytest.mark.skipif(os.environ.get("VRL_VDN_GRPO_CUDA") != "1", reason="Requires one reserved GPU")
def test_cuda_online_grpo_sync_accumulation_and_checkpoint_resume(tmp_path):
    assert torch.cuda.is_available()
    deterministic = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        _verify_update_and_resume(tmp_path, device=torch.device("cuda:0"))
    finally:
        torch.use_deterministic_algorithms(deterministic)
