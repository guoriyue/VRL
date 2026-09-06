"""Real-inference test for the Cosmos Anima CFG backbone wrapper.

No fake transformer: a real (tiny, cache-free) ``CosmosTransformer3DModel`` runs
through ``AnimaModel.forward_step``. Anima follows ComfyUI's ModelSamplingDiscreteFlow
CONST path, so it differs from predict2/2.5 in three ways this test pins against the
model's OWN real branch outputs (copying the predict2.5 exemplar verbatim would be a
bug here):

1. CFG formula is ``combined = uncond + guidance * (cond - uncond)`` (model.py:326),
   NOT predict2's ``cond + guidance * (cond - uncond)``.
2. The transformer receives the RAW sigma as ``timestep`` (model.py:301-302); there
   is no EDM->timestep conversion, so the recorded ``timestep`` equals the sigma.
3. Anima feeds latents to the transformer DIRECTLY — no runner-side channel
   expansion — so the latents carry ``transformer.config.in_channels`` channels and
   ``in_channels == out_channels`` (see ``build_tiny_anima_transformer``).
"""

from __future__ import annotations

from types import SimpleNamespace

import torch

from tests.models.steps.denoise.fixtures import (
    TINY_COSMOS_LATENT_SHAPE,
    TINY_COSMOS_TEXT_DIM,
    build_tiny_anima_transformer,
    record_forward_calls,
    stamp_model_precision,
)
from vrl.models.families.cosmos.anima.model import (
    AnimaModel,
    AnimaSamplingState,
)

_SIGMA = 0.7
_GUIDANCE = 2.0


def _state(*, do_cfg: bool = True) -> AnimaSamplingState:
    b = TINY_COSMOS_LATENT_SHAPE[0]
    _, _, _, h, w = TINY_COSMOS_LATENT_SHAPE
    return AnimaSamplingState(
        # in_channels == out_channels for Anima, so the latents carry the bare
        # latent channels and the transformer adds the padding-mask channel inside.
        latents=torch.randn(TINY_COSMOS_LATENT_SHAPE),
        timesteps=torch.tensor([700.0]),
        scheduler=SimpleNamespace(sigmas=torch.tensor([_SIGMA])),
        prompt_embeds=torch.randn(b, 3, TINY_COSMOS_TEXT_DIM),
        negative_prompt_embeds=torch.randn(b, 3, TINY_COSMOS_TEXT_DIM) if do_cfg else None,
        guidance_scale=_GUIDANCE,
        do_cfg=do_cfg,
        padding_mask=torch.zeros(1, 1, h, w),
    )


def _model(transformer: torch.nn.Module) -> AnimaModel:
    model = AnimaModel(
        transformer=transformer,
        text_encoder=None,
        llm_adapter=torch.nn.Identity(),
        vae=None,
        scheduler=None,
        image_processor=None,
        qwen_tokenizer=None,
        t5_tokenizer=None,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    stamp_model_precision(model)
    return model


def test_anima_forward_step_runs_real_unbatched_cfg() -> None:
    """Two separate forwards on the real backbone, combined the CONST-path way."""
    transformer = build_tiny_anima_transformer()
    calls = record_forward_calls(transformer)
    model = _model(transformer)
    state = _state()

    out = model.forward_step(state, 0)

    # Anima runs cond and uncond as two separate forwards (not batched).
    assert len(calls) == 2
    uncond, cond = out["noise_pred_uncond"], out["noise_pred_cond"]
    # Anima's CFG combine differs from predict2: uncond + g*(cond - uncond).
    torch.testing.assert_close(out["noise_pred"], uncond + _GUIDANCE * (cond - uncond))
    assert out["noise_pred"].shape == TINY_COSMOS_LATENT_SHAPE
    # The raw sigma is passed straight through as the timestep (no EDM conversion).
    torch.testing.assert_close(
        calls[0]["timestep"],
        torch.full((TINY_COSMOS_LATENT_SHAPE[0],), _SIGMA),
    )
    # Anima feeds the latents to the transformer directly (no channel expansion).
    torch.testing.assert_close(calls[0]["hidden_states"], state.latents)
    # The real transformer responds to the prompt (cond vs uncond differ).
    assert not torch.allclose(cond, uncond)


def test_cond_branch_runs_first_on_the_positive_embeds() -> None:
    """Branch order is load-bearing: the CFG combine treats the first forward as cond.

    Swapping the two forwards keeps every shape and dtype and only shifts the
    guided result, which the parity test above cannot see. Pinning WHICH embeds
    each call received (by identity, not by value) is the only way to catch it.
    """
    transformer = build_tiny_anima_transformer()
    calls = record_forward_calls(transformer)
    state = _state()

    out = _model(transformer).forward_step(state, 0)

    assert calls[0]["encoder_hidden_states"] is state.prompt_embeds
    assert calls[1]["encoder_hidden_states"] is state.negative_prompt_embeds
    # Clean-data training reads the raw conditional velocity, never the guided one.
    assert AnimaModel.diffusion_pretraining_prediction(None, out) is out["noise_pred_cond"]


def test_cfg_off_runs_one_forward_and_reports_a_zero_uncond() -> None:
    """CFG-off must not fabricate a second forward; ``noise_pred`` is the raw cond."""
    transformer = build_tiny_anima_transformer()
    calls = record_forward_calls(transformer)

    out = _model(transformer).forward_step(_state(do_cfg=False), 0)

    assert len(calls) == 1
    torch.testing.assert_close(out["noise_pred"], out["noise_pred_cond"])
    assert torch.count_nonzero(out["noise_pred_uncond"]) == 0
