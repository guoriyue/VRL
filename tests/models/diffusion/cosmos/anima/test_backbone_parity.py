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

from tests.models.diffusion.fixtures import (
    TINY_COSMOS_LATENT_SHAPE,
    TINY_COSMOS_TEXT_DIM,
    build_tiny_anima_transformer,
    record_forward_calls,
    stamp_test_contract,
)
from vrl.models.diffusion.cosmos.anima.model import (
    AnimaModel,
    AnimaSamplingState,
)

_SIGMA = 0.7
_GUIDANCE = 2.0


def _state() -> AnimaSamplingState:
    b = TINY_COSMOS_LATENT_SHAPE[0]
    _, _, _, h, w = TINY_COSMOS_LATENT_SHAPE
    return AnimaSamplingState(
        # in_channels == out_channels for Anima, so the latents carry the bare
        # latent channels and the transformer adds the padding-mask channel inside.
        latents=torch.randn(TINY_COSMOS_LATENT_SHAPE),
        timesteps=torch.tensor([700.0]),
        scheduler=SimpleNamespace(sigmas=torch.tensor([_SIGMA])),
        prompt_embeds=torch.randn(b, 3, TINY_COSMOS_TEXT_DIM),
        negative_prompt_embeds=torch.randn(b, 3, TINY_COSMOS_TEXT_DIM),
        guidance_scale=_GUIDANCE,
        do_cfg=True,
        padding_mask=torch.zeros(1, 1, h, w),
    )


def test_anima_forward_step_runs_real_unbatched_cfg() -> None:
    """Checks Anima forward step runs real unbatched CFG on a real backbone."""
    transformer = build_tiny_anima_transformer()
    calls = record_forward_calls(transformer)
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
    stamp_test_contract(model)
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
