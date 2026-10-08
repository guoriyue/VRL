"""VDN-H3 online GRPO composes end to end on the real stack, including cold resume.

The VDN experiment preset is resolved onto a tiny MiniMax-H3 modular snapshot
(saved through diffusers) and a tiny VDN artifact (saved through upstream's own
checkpoint writer). From there everything is production code: the family
loader grafts the hybrid attention onto both the rollout policy and the replay
model, ``InProcessGenerationRuntime`` runs the rollout worker body, the real
collector scores through a real reward runtime, the trainer is wired the way
the online recipe wires it, weights reach the rollout through the real syncer,
and the checkpoint writer/restorer move the state into a freshly built stack.
"""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import torch

from tests.models.steps.denoise.fixtures import (
    write_tiny_minimax_h3_snapshot,
    write_tiny_vdn_h3_checkpoint,
)
from tests.rollouts.collector._helpers import IndexReward, Trace

pytest.importorskip("src.models.hybrid_attention")

_PROMPT = "a wooden block"

# The VDN experiment's composition with a CPU, model-free reward in place of
# the production video reward service.
_EXPERIMENT = """\
defaults:
  - /recipe/online/denoise_grpo
  - /model/vdn_h3/8nfe
  - /sampling/video/h3_768p_124f
  - /sampling/denoise/8_step_no_cfg
  - /dataset/videophy
  - /reward/image_sharpness
"""


@dataclass
class _Stack:
    trainer: Any
    runtime: Any
    bundle: Any
    identity: dict[str, Any]
    trace: Trace


def _stack(
    monkeypatch: pytest.MonkeyPatch,
    root: Path,
    *,
    snapshot: Path,
    artifact: Path,
    overrides: tuple[str, ...] = (),
) -> _Stack:
    """Resolve the tiny VDN run and build its stack the way the online recipe does."""

    from tests.generation._in_process_runtime import InProcessGenerationRuntime
    from vrl import run
    from vrl.config.loading import load_config
    from vrl.rewards.runtime import RewardFunctionRuntime
    from vrl.rollouts.collector import RolloutCollector
    from vrl.scripts.common.factory import AlgorithmEvaluatorPair
    from vrl.trainers.distributed import DistributedTrainingContext
    from vrl.trainers.online.trainer import OnlineTrainer
    from vrl.trainers.strategy import build_strategy
    from vrl.trainers.weight_sync import RayRuntimeWeightSyncer

    root.mkdir(parents=True, exist_ok=True)
    manifest = root / "prompts.jsonl"
    manifest.write_text(json.dumps({"prompt": _PROMPT}) + "\n", encoding="utf-8")
    experiment = root / "vdn_h3_online.yaml"
    experiment.write_text(_EXPERIMENT, encoding="utf-8")
    cfg = load_config(
        str(experiment),
        overrides=[
            f"model.path={snapshot}",
            "model.revision=null",
            f"model.vdn_checkpoint={artifact}",
            # Upstream's eager reference: the window softmax that runs on CPU.
            "model.softmax_backend=ref",
            "model.lora.rank=2",
            "model.lora.alpha=2",
            f"data.manifest={manifest}",
            f"trainer.output_dir={root / 'run'}",
            "trainer.total_epochs=1",
            "precision.training.dtype=fp32",
            "precision.rollout.dtype=fp32",
            "sampling.width=16",
            "sampling.height=16",
            "sampling.num_frames=8",
            "sampling.num_steps=3",
            "sampling.max_sequence_length=8",
            # One fixed request seed: every collect draws the same rollout noise.
            "sampling.seed=17",
            "rollout.sde.window_range=[0,3]",
            "rollout.n_samples_per_prompt=2",
            "rollout.prompts_per_batch=1",
            "rollout.samples_per_generation_batch=1",
            "actor.training_microbatch_size=1",
            "actor.timestep_fraction=1.0",
            "actor.drop_zero_advantage=false",
            "actor.optim.lr=1e-4",
            "actor.optim.weight_decay=0.0",
            "actor.ema.enable=true",
            "actor.ema.decay=0.9",
            "actor.ema.update_interval=1",
            "algorithm.kl_coef=0.0",
            "distributed.resources.rollout.num_gpus=0",
            *overrides,
        ],
    )
    resolved = run.resolve_online_run(cfg)
    built = resolved.built
    replay = run.resolve_model(
        resolved.family,
        built.root,
        resolved.device,
        precision=built.precision,
        for_rollout=False,
    )
    bundle = replay.materialize(context="vdn composition test")
    runtime = InProcessGenerationRuntime(resolved.ray_launch_inputs(replay))
    trace = Trace(monkeypatch)
    trace.watch(runtime, "generate", "generate")
    trace.watch(runtime, "update_weights", "update_weights")
    collector = RolloutCollector.from_family(
        resolved.family,
        # Scores [0, 1] for the two samples of the one prompt group.
        reward_runtime=RewardFunctionRuntime(IndexReward()),
        config=resolved.collector,
        lifecycle=resolved.resources.lifecycle,
    )
    collector.set_generation_runtime(runtime)
    # The recipe's strategy: built from the run's own training context, so it
    # places the trainable roots on the trainer device.
    strategy = build_strategy(
        built.root, DistributedTrainingContext.from_root(built.root, device=resolved.device)
    )
    pair = AlgorithmEvaluatorPair.from_configs(
        built,
        scheduler=getattr(bundle, "scheduler", None),
    )
    trainer = OnlineTrainer(
        algorithm=pair.algorithm,
        collector=collector,
        evaluator=pair.evaluator,
        model=bundle.model,
        weight_syncer=RayRuntimeWeightSyncer(runtime),
        sync_state_getter=lambda: strategy.export_rollout_state(bundle),
        config=built.trainer,
        strategy=strategy,
    )
    return _Stack(
        trainer=trainer, runtime=runtime, bundle=bundle, identity=replay.identity, trace=trace
    )


def _trainable(stack: _Stack) -> dict[str, torch.Tensor]:
    transformer = stack.bundle.model.trainable_modules["transformer"]
    return {
        name: parameter.detach().clone()
        for name, parameter in transformer.named_parameters()
        if parameter.requires_grad
    }


def _verify_update_and_resume(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *, overrides: tuple[str, ...] = ()
) -> None:
    from vrl.trainers.checkpointing import (
        TrainingCheckpoint,
        restore_rng_state,
        save_training_checkpoint,
    )

    snapshot = write_tiny_minimax_h3_snapshot(tmp_path / "minimax-h3")
    artifact = write_tiny_vdn_h3_checkpoint(tmp_path / "vdn-h3.pt")
    control = _stack(
        monkeypatch,
        tmp_path / "control",
        snapshot=snapshot,
        artifact=artifact,
        overrides=overrides,
    )
    before = _trainable(control)
    assert before, "the VDN LoRA adapters are the trainable parameters"

    first = asyncio.run(control.trainer.step([_PROMPT]))

    assert first.grad_norm > 0 and first.initial_replay.finite
    # The rollout policy and the replay model are the same grafted weights.
    assert first.initial_replay.logprob_abs_diff_max < 1e-6
    after = _trainable(control)
    assert any(not torch.equal(before[name], after[name]) for name in before)

    path = tmp_path / "checkpoint-1"
    save_training_checkpoint(
        path,
        trainer=control.trainer,
        bundle=control.bundle,
        family="vdn_h3",
        model_identity=control.identity,
        progress={"next_step": 1},
        strategy=control.trainer._strategy,
    )
    second = asyncio.run(control.trainer.step([_PROMPT]))
    assert second.grad_norm > 0 and second.initial_replay.logprob_abs_diff_max < 1e-6
    # Each collect ran under the version the syncer published just before it.
    assert [request.policy_version for request in control.trace.requests] == [1, 2]

    resumed = _stack(
        monkeypatch,
        tmp_path / "resumed",
        snapshot=snapshot,
        artifact=artifact,
        overrides=overrides,
    )
    checkpoint = TrainingCheckpoint.load(path)
    checkpoint.restore_training(
        trainer=resumed.trainer,
        bundle=resumed.bundle,
        family="vdn_h3",
        expected_model_identity=resumed.identity,
    )
    restore_rng_state(checkpoint.rng_state, rank=0, world_size=1)
    replayed = asyncio.run(resumed.trainer.step([_PROMPT]))

    # The fresh stack pushes the restored weights before its first collect, so
    # it reproduces the control's second update exactly.
    assert replayed.grad_norm == second.grad_norm
    assert replayed.initial_replay.logprob_abs_diff_max < 1e-6
    assert control.trainer.state.step == resumed.trainer.state.step == 2
    control_actions = control.trace.results["generate"][-1].trajectory
    resumed_actions = resumed.trace.results["generate"][-1].trajectory
    torch.testing.assert_close(
        control_actions.segments["denoise"].tensors["actions"].value,
        resumed_actions.segments["denoise"].tensors["actions"].value,
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        control.trainer.model.state_dict(), resumed.trainer.model.state_dict(), rtol=0, atol=0
    )
    left, right = control.trainer.state_dict(), resumed.trainer.state_dict()
    torch.testing.assert_close(
        left["optimizer"]["state"], right["optimizer"]["state"], rtol=0, atol=0
    )
    torch.testing.assert_close(left["ema"], right["ema"], rtol=0, atol=0)


def test_online_grpo_sync_accumulation_and_checkpoint_resume(monkeypatch, tmp_path):
    _verify_update_and_resume(monkeypatch, tmp_path)


@pytest.mark.gpu
@pytest.mark.skipif(os.environ.get("VRL_VDN_GRPO_CUDA") != "1", reason="Requires one reserved GPU")
def test_cuda_online_grpo_sync_accumulation_and_checkpoint_resume(monkeypatch, tmp_path):
    assert torch.cuda.is_available()
    deterministic = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        # The same real stack with the trainer and the rollout worker on card 0.
        _verify_update_and_resume(
            monkeypatch, tmp_path, overrides=("distributed.resources.rollout.devices=[0]",)
        )
    finally:
        torch.use_deterministic_algorithms(deterministic)
