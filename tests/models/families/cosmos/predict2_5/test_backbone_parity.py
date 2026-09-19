"""Real-inference test for the Cosmos Predict2.5 CFG backbone wrapper.

No fake transformer: a real (tiny, cache-free) ``CosmosTransformer3DModel`` runs
through ``forward_step``. Predict2.5 issues TWO separate forwards and blends the
ground-truth velocity over conditioned regions via ``cond_mask``; the wrapper's
CFG combination is verified against the model's OWN real branch outputs.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch

from tests.models.steps.denoise.fixtures import (
    TINY_COSMOS_LATENT_SHAPE,
    TINY_COSMOS_TEXT_DIM,
    build_tiny_cosmos_transformer,
    lora_test_build,
    record_forward_calls,
    stamp_model_precision,
)
from vrl.generation.types import GenerationRequest, GenerationSampleRow
from vrl.models.families.cosmos.predict2_5.model import (
    CosmosPredict25Model,
    CosmosPredict25ReplayModel,
    CosmosPredict25SamplingState,
)
from vrl.rollouts.batch import RolloutBatch
from vrl.trajectory.builders import build_diffusion_trajectory

_GUIDANCE = 2.0


def _state() -> CosmosPredict25SamplingState:
    b = TINY_COSMOS_LATENT_SHAPE[0]
    # Partial conditioning mask: top rows are clamped to the ground-truth
    # velocity, the rest carry the (prompt-dependent) transformer output, so the
    # cond/uncond branches genuinely differ.
    cond_mask = torch.zeros(b, 1, 1, 4, 4)
    cond_mask[..., :2, :] = 1.0
    return CosmosPredict25SamplingState(
        latents=torch.randn(TINY_COSMOS_LATENT_SHAPE),
        timesteps=torch.tensor([0.0]),
        scheduler=SimpleNamespace(sigmas=torch.tensor([0.75])),
        prompt_embeds=torch.randn(b, 3, TINY_COSMOS_TEXT_DIM),
        negative_prompt_embeds=torch.randn(b, 3, TINY_COSMOS_TEXT_DIM),
        guidance_scale=_GUIDANCE,
        do_cfg=True,
        cond_latent=torch.randn(TINY_COSMOS_LATENT_SHAPE),
        cond_mask=cond_mask,
        cond_indicator=torch.zeros(b, 1, 1, 1, 1),
        padding_mask=torch.zeros(1, 1, 4, 4),
        height=4,
        width=4,
        num_frames=1,
        fps=16,
    )


def test_cosmos_predict25_forward_step_runs_real_unbatched_cfg() -> None:
    """Two separate real forwards; predict2.5 keeps predict2's ``cond + g*(cond - uncond)``
    combine but emits it as-is (no EDM sigma conversion), and the prompt still shows through
    the cond_mask blend.
    """
    transformer = build_tiny_cosmos_transformer()
    calls = record_forward_calls(transformer)
    model = CosmosPredict25Model(
        pipeline=SimpleNamespace(transformer=transformer, device=torch.device("cpu")),
        device=torch.device("cpu"),
    )
    stamp_model_precision(model)
    state = _state()

    out = model.forward_step(state, 0)

    assert len(calls) == 2
    # The step's sigma reaches the transformer as the (unconditioned) timestep.
    for call in calls:
        torch.testing.assert_close(call["timestep"], torch.full_like(call["timestep"], 0.75))
    uncond, cond = out["noise_pred_uncond"], out["noise_pred_cond"]
    torch.testing.assert_close(out["noise_pred"], cond + _GUIDANCE * (cond - uncond))
    assert out["noise_pred"].shape == TINY_COSMOS_LATENT_SHAPE
    # Prompt response survives the cond_mask blend in the unconditioned region.
    assert not torch.allclose(cond, uncond)


def test_predict25_full_finetune_supports_real_forward_backward() -> None:
    transformer = build_tiny_cosmos_transformer()
    transformer.requires_grad_(False)
    model = CosmosPredict25ReplayModel(transformer=transformer, scheduler=object(), device="cpu")
    build = lora_test_build({}, family="cosmos-predict2.5")
    build.model_config = {"use_lora": False}
    model.apply_full_finetune(build)
    stamp_model_precision(model)

    output = model.forward_step(_state(), 0)["noise_pred"]
    output.square().mean().backward()

    assert all(parameter.requires_grad for parameter in transformer.parameters())
    gradients = [
        parameter.grad for parameter in transformer.parameters() if parameter.grad is not None
    ]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert any(gradient.abs().sum() > 0 for gradient in gradients)


def test_replay_forward_on_caller_latents_feeds_the_step_sigma() -> None:
    """The forward-process objectives evaluate a re-noised clean latent through
    ``replay_forward_with_latents``; on Predict2.5 that must reach the transformer
    at the step's flow sigma (``scheduler.sigmas[idx]`` in [0, 1]) exactly as the
    rollout and SDE replay do, not at the raw ``[0, 1000]`` grid value the
    trajectory stores as ``timesteps``."""
    from diffusers import UniPCMultistepScheduler

    b = TINY_COSMOS_LATENT_SHAPE[0]
    scheduler = UniPCMultistepScheduler(
        use_flow_sigmas=True, prediction_type="flow_prediction", num_train_timesteps=1000
    )
    scheduler.set_timesteps(4)
    step_idx = 2
    transformer = build_tiny_cosmos_transformer()
    calls = record_forward_calls(transformer)
    model = CosmosPredict25ReplayModel(transformer=transformer, scheduler=scheduler, device="cpu")
    stamp_model_precision(model)

    steps = len(scheduler.timesteps)
    context = {
        "guidance_scale": _GUIDANCE,
        "cfg": True,
        "height": 4,
        "width": 4,
        "num_frames": 1,
        "fps": 16,
    }
    trajectory = build_diffusion_trajectory(
        request=GenerationRequest(
            request_id="predict25-forward-process",
            family="cosmos-predict2.5",
            task="t2w",
            inputs=["a test prompt"],
            samples_per_prompt=b,
        ),
        sample_rows=[
            GenerationSampleRow(
                prompt_index=0, sample_index=i, prompt="a test prompt", sample_id=f"s{i}"
            )
            for i in range(b)
        ],
        observations=torch.zeros(b, steps, *TINY_COSMOS_LATENT_SHAPE[1:]),
        actions=torch.zeros(b, steps, *TINY_COSMOS_LATENT_SHAPE[1:]),
        old_log_prob=torch.zeros(b, steps),
        timesteps=scheduler.timesteps.to(torch.float32).repeat(b, 1),
        replay_tensors={
            "prompt_embeds": torch.randn(b, 3, TINY_COSMOS_TEXT_DIM),
            "negative_prompt_embeds": torch.randn(b, 3, TINY_COSMOS_TEXT_DIM),
            "latents_clean": torch.zeros(TINY_COSMOS_LATENT_SHAPE),
            "cond_mask": torch.zeros(b, 1, 1, 4, 4),
            "cond_indicator": torch.zeros(b, 1, 1, 1, 1),
            "padding_mask": torch.zeros(b, 1, 4, 4),
        },
        context=context,
    )
    batch = RolloutBatch(
        rewards=torch.zeros(b),
        group_ids=torch.zeros(b, dtype=torch.long),
        context=context,
        trajectory=trajectory,
    )
    xt = torch.randn(TINY_COSMOS_LATENT_SHAPE)

    with torch.no_grad():
        out = model.replay_forward_with_latents(
            batch, step_idx, xt, classifier_free_guidance=False
        )["noise_pred"]

    # CFG forced off: one conditional forward, at the step's sigma, on the caller's latent.
    assert len(calls) == 1
    sigma = float(scheduler.sigmas[step_idx])
    assert 0.0 < sigma < 1.0
    torch.testing.assert_close(calls[0]["timestep"], torch.full_like(calls[0]["timestep"], sigma))
    assert out.shape == TINY_COSMOS_LATENT_SHAPE
