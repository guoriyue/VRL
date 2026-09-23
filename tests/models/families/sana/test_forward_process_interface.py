"""SANA on the forward-process objective surface (DiffusionNFT / V-GRPO).

The objectives re-noise the clean latent and hand it to the shared
``replay_forward_with_latents`` with classifier-free guidance forced off. SANA
must therefore export ``latents_clean`` and rebuild a conditional-only state
from a trajectory that recorded CFG; the real (tiny) ``SanaTransformer2DModel``
answers exactly as the conditional ``forward_step`` branch does.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch

from tests.models.steps.denoise.fixtures import (
    TINY_SANA_CAPTION_DIM,
    TINY_SANA_LATENT_SHAPE,
    build_tiny_transformer,
    stamp_model_precision,
)
from vrl.generation.types import GenerationRequest, GenerationSampleRow
from vrl.models.families.sana.model import SanaModel, SanaSamplingState
from vrl.rollouts.batch import RolloutBatch
from vrl.trajectory.builders import build_diffusion_trajectory

_TEXT_LEN = 3
_BATCH = TINY_SANA_LATENT_SHAPE[0]
_GUIDANCE = 4.5


def _model() -> SanaModel:
    model = SanaModel(
        pipeline=SimpleNamespace(transformer=build_tiny_transformer("sana"), device="cpu"),
        device=torch.device("cpu"),
    )
    stamp_model_precision(model)
    return model


def _state(
    *, latents: torch.Tensor, do_cfg: bool, **conditioning: torch.Tensor
) -> SanaSamplingState:
    return SanaSamplingState(
        latents=latents,
        timesteps=torch.tensor([[5.0] * _BATCH]),
        scheduler=None,
        prompt_embeds=conditioning["prompt_embeds"],
        prompt_attention_mask=conditioning["prompt_attention_mask"],
        negative_prompt_embeds=conditioning["negative_prompt_embeds"] if do_cfg else None,
        negative_prompt_attention_mask=(
            conditioning["negative_prompt_attention_mask"] if do_cfg else None
        ),
        guidance_scale=_GUIDANCE if do_cfg else 1.0,
        do_cfg=do_cfg,
    )


def test_replay_tensors_carry_the_final_latent_for_the_forward_process_objectives() -> None:
    latents = torch.randn(TINY_SANA_LATENT_SHAPE, requires_grad=True)
    state = _state(
        latents=latents,
        do_cfg=False,
        prompt_embeds=torch.randn(_BATCH, _TEXT_LEN, TINY_SANA_CAPTION_DIM),
        prompt_attention_mask=torch.ones(_BATCH, _TEXT_LEN, dtype=torch.long),
    )

    exported = _model().export_replay_tensors(state)

    assert torch.equal(exported["latents_clean"], latents.detach())
    assert exported["latents_clean"].requires_grad is False


def test_replay_forward_on_caller_latents_can_force_the_conditional_branch() -> None:
    """``classifier_free_guidance=False`` gives exactly the conditional ``forward_step``
    output on the caller's latent even though the trajectory recorded CFG; left at
    the default the rollout's CFG combine is reproduced instead."""
    torch.manual_seed(0)
    model = _model()
    conditioning = {
        "prompt_embeds": torch.randn(_BATCH, _TEXT_LEN, TINY_SANA_CAPTION_DIM),
        "prompt_attention_mask": torch.ones(_BATCH, _TEXT_LEN, dtype=torch.long),
        "negative_prompt_embeds": torch.randn(_BATCH, _TEXT_LEN, TINY_SANA_CAPTION_DIM),
        "negative_prompt_attention_mask": torch.ones(_BATCH, _TEXT_LEN, dtype=torch.long),
    }
    xt = torch.randn(TINY_SANA_LATENT_SHAPE)
    recorded = _state(latents=torch.zeros(TINY_SANA_LATENT_SHAPE), do_cfg=True, **conditioning)
    context = {**model.export_batch_context(recorded), "height": 64, "width": 64}
    trajectory = build_diffusion_trajectory(
        request=GenerationRequest(
            request_id="sana-forward-process",
            family="sana",
            task="t2i",
            inputs=["a test prompt"],
            samples_per_prompt=_BATCH,
        ),
        sample_rows=[
            GenerationSampleRow(
                prompt_index=0, sample_index=i, prompt="a test prompt", sample_id=f"s{i}"
            )
            for i in range(_BATCH)
        ],
        observations=torch.zeros(_BATCH, 1, *TINY_SANA_LATENT_SHAPE[1:]),
        actions=torch.zeros(_BATCH, 1, *TINY_SANA_LATENT_SHAPE[1:]),
        old_log_prob=torch.zeros(_BATCH, 1),
        timesteps=torch.full((_BATCH, 1), 5.0),
        replay_tensors=model.export_replay_tensors(recorded),
        context=context,
    )
    batch = RolloutBatch(
        rewards=torch.zeros(_BATCH),
        group_ids=torch.zeros(_BATCH, dtype=torch.long),
        context=context,
        trajectory=trajectory,
    )

    with torch.no_grad():
        conditional = model.replay_forward_with_latents(
            batch, 0, xt, classifier_free_guidance=False
        )["noise_pred"]
        guided = model.replay_forward_with_latents(batch, 0, xt)["noise_pred"]
        via_step_cond = model.forward_step(_state(latents=xt, do_cfg=False, **conditioning), 0)
        via_step_cfg = model.forward_step(_state(latents=xt, do_cfg=True, **conditioning), 0)

    torch.testing.assert_close(conditional, via_step_cond["noise_pred"])
    torch.testing.assert_close(guided, via_step_cfg["noise_pred"])
    assert not torch.allclose(conditional, guided)
