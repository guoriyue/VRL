"""A real tiny VDN rollout drives reward-signed GRPO gradients and an update.

Random tiny weights and controlled rewards isolate framework correctness; this
is not released-model quality or the distributed online trainer acceptance.
"""

import pytest
import torch

from tests.models.steps.denoise.fixtures import build_tiny_vdn_h3_model, stamp_model_precision
from vrl.algorithms.grpo.continuous import GRPO, GRPOConfig
from vrl.algorithms.trajectory import AlgorithmInput
from vrl.generation.execution.sample_batches import GenerationSampleBatch
from vrl.generation.types import GenerationRequest
from vrl.math.denoise.flow_matching import sde_step_with_logprob
from vrl.models.families.vdn_h3.model import VDNH3ReplayModel
from vrl.models.families.vdn_h3.runtime import VDNH3BatchExecutor
from vrl.rollouts.evaluators.types import SegmentSignal, TrajectorySignalBatch

pytest.importorskip("src.models.hybrid_attention")


def test_native_rollout_replay_grpo_reward_direction_and_optimizer_update():
    rollout = build_tiny_vdn_h3_model(seed=0)
    components = build_tiny_vdn_h3_model(seed=1).pipeline
    replay = VDNH3ReplayModel(
        transformer=components.transformer,
        scheduler=components.scheduler,
        audio_scheduler=components.audio_scheduler,
        device=torch.device("cpu"),
    )
    stamp_model_precision(replay)
    replay.transformer.load_state_dict(rollout.transformer.state_dict(), strict=True)
    replay.transformer.requires_grad_(False)
    projection = replay.transformer.transformer_blocks[0].attn.to_out_linear
    projection.requires_grad_(True)
    executor = VDNH3BatchExecutor(rollout)
    request = GenerationRequest(
        request_id="vdn-grpo-correctness",
        family="vdn_h3",
        task="t2v",
        inputs=["a wooden block"],
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
            executor.forward_batch(
                request, GenerationSampleBatch(prompt_index=0, sample_start=index, sample_count=1)
            )
            for index in range(2)
        ]
    assert not torch.equal(results[0].actions, results[1].actions)
    params = executor.parse_sampling_params(request)

    def rescore(*, step_index=0):
        values = []
        for result in results:
            state = replay.restore_eval_state(
                result.replay_tensors,
                result.context,
                result.observations[:, step_index],
                step_index,
            )
            prediction = replay.forward_step(state, step_index)["noise_pred"]
            values.append(
                sde_step_with_logprob(
                    state.scheduler,
                    prediction,
                    result.timesteps[:, step_index],
                    result.observations[:, step_index],
                    prev_sample=result.actions[:, step_index],
                    noise_level=params.sde.noise_level,
                    sde_type=params.sde.sde_type,
                    step_index=step_index,
                ).log_prob
            )
        return torch.cat(values)

    with torch.no_grad():
        for step_index in reversed(range(results[0].log_probs.shape[1])):
            expected = torch.cat([result.log_probs[:, step_index] for result in results])
            torch.testing.assert_close(rescore(step_index=step_index), expected, rtol=0, atol=1e-6)
    old = torch.cat([result.log_probs[:, 0] for result in results])
    current = rescore()
    torch.testing.assert_close(current, old, rtol=0, atol=1e-6)
    group_ids = torch.zeros(2, dtype=torch.long)
    algorithm = GRPO(GRPOConfig(kl_coef=0.0))
    advantages = algorithm.compute_advantages_from_tensors(torch.tensor([0.0, 1.0]), group_ids)
    assert advantages[0] < 0 < advantages[1]

    def loss_for(log_prob, advantage):
        signals = TrajectorySignalBatch(
            segments={
                "denoise": SegmentSignal(
                    name="denoise",
                    distribution="flow_matching",
                    log_prob=log_prob,
                    old_log_prob=old,
                    mask=torch.ones_like(old),
                )
            },
            group_ids=group_ids,
            primary_segment="denoise",
        )
        return algorithm.compute_loss(AlgorithmInput(signals=signals, advantages=advantage))[0]

    loss = loss_for(current, advantages)
    derivative = torch.autograd.grad(loss, current, retain_graph=True)[0]
    torch.testing.assert_close(derivative, -advantages / 2, rtol=1e-5, atol=1e-7)
    gradient = torch.autograd.grad(loss, projection.weight, retain_graph=True)[0]
    reversed_gradient = torch.autograd.grad(
        loss_for(current, -advantages), projection.weight, retain_graph=True
    )[0]
    torch.testing.assert_close(reversed_gradient, -gradient, rtol=1e-5, atol=1e-7)
    zero = algorithm.compute_advantages_from_tensors(torch.ones(2), group_ids)
    zero_gradient = torch.autograd.grad(
        loss_for(current, zero), projection.weight, retain_graph=True
    )[0]
    assert torch.count_nonzero(zero_gradient) == 0
    assert torch.isfinite(gradient).all() and torch.count_nonzero(gradient) > 0
    before = projection.weight.detach().clone()
    optimizer = torch.optim.AdamW(projection.parameters(), lr=1e-4, weight_decay=0)
    loss.backward()
    optimizer.step()
    assert torch.isfinite(projection.weight).all()
    assert not torch.equal(before, projection.weight)
    with torch.no_grad():
        assert loss_for(rescore(), advantages) < loss.detach()
