"""Interface tests for the FLUX DiffusionNFT port (Phase A).

DiffusionNFT was originally implemented only for cosmos/predict2.5. These tests
verify the three things the NFT loss + ``after_optimizer_step`` dispatch to on a
model (``diffusion_nft_prepare_transformer_input``, a frozen ``previous`` LoRA
adapter, and ``sync_previous_policy_adapter``) now work for FLUX — without
loading the real ~12B FLUX transformer.

The forward-kwargs test uses a dummy stand-in transformer because the only thing
``diffusion_nft_prepare_transformer_input`` reads off the transformer is its
dtype + ``config.guidance_embeds``; everything else is pure shape arithmetic
(the packed rotary-position grid). The adapter test uses a genuine tiny PEFT
transformer so ``attach_previous_policy_adapter`` exercises the real
``add_adapter`` / freeze / copy path, exactly as the runtime builder does.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from tests.models.steps.denoise.fixtures import (
    _TINY_WAN_LORA_TARGETS,
    build_tiny_wan_transformer,
)
from vrl.models.families.flux.model import FluxModel, FluxReplayModel


class _DummyFluxTransformer(nn.Module):
    """Minimal stand-in: only ``.dtype`` + ``.config.guidance_embeds`` are read."""

    def __init__(self, *, guidance_embeds: bool = True) -> None:
        super().__init__()
        self.config = SimpleNamespace(guidance_embeds=guidance_embeds)
        self._p = nn.Parameter(torch.zeros(1, dtype=torch.bfloat16))

    @property
    def dtype(self) -> torch.dtype:
        return torch.bfloat16


def _replay_model(*, guidance_embeds: bool = True) -> FluxReplayModel:
    return FluxReplayModel(
        transformer=_DummyFluxTransformer(guidance_embeds=guidance_embeds),
        scheduler=None,
        device="cpu",
    )


def _packed_inputs(*, batch: int = 2, grid: int = 16, channels: int = 64):
    """A 256x256 FLUX latent packs to a grid x grid token sequence."""

    seq_len = grid * grid
    latents = torch.randn(batch, seq_len, channels)
    prompt_embeds = torch.randn(batch, 7, 4096)
    pooled = torch.randn(batch, 768)
    timestep = torch.full((batch,), 500.0)  # [0, 1000] flow-match grid -> t=0.5
    return latents, prompt_embeds, pooled, timestep


def test_flux_models_expose_nft_interface() -> None:
    """Both the rollout and replay FLUX models expose the NFT model contract."""

    for cls in (FluxModel, FluxReplayModel):
        assert hasattr(cls, "diffusion_nft_prepare_transformer_input")
        assert hasattr(cls, "sync_previous_policy_adapter")
        assert hasattr(cls, "attach_previous_policy_adapter")


def test_prepare_transformer_input_kwargs_and_grid() -> None:
    """The forward kwargs match FluxTransformer2DModel.forward and the rotary grid."""

    model = _replay_model()
    latents, prompt_embeds, pooled, timestep = _packed_inputs(grid=16)

    out = model.diffusion_nft_prepare_transformer_input(
        latents=latents,
        prompt_embeds=prompt_embeds,
        prompt_attention_mask=None,
        pooled_prompt_embeds=pooled,
        timestep=timestep,
        num_frames=1,
        height=256,
        width=256,
    )

    assert set(out) == {
        "hidden_states",
        "timestep",
        "encoder_hidden_states",
        "pooled_projections",
        "img_ids",
        "txt_ids",
        "guidance",
        "return_dict",
    }
    assert out["return_dict"] is False
    # packed 16x16 grid -> 256 position ids; txt_ids one row per text token.
    assert tuple(out["img_ids"].shape) == (256, 3)
    assert tuple(out["txt_ids"].shape) == (prompt_embeds.shape[1], 3)
    assert out["img_ids"].dtype == torch.float32 and out["txt_ids"].dtype == torch.float32
    # raw 500 on the [0, 1000] grid is fed to the transformer as t/1000 = 0.5,
    # matching forward_step's /1000 convention.
    assert torch.allclose(out["timestep"].float(), torch.full((2,), 0.5))
    # FLUX.1-dev embeds a guidance scalar; absent an explicit scale, default 3.5.
    assert out["guidance"] is not None
    assert tuple(out["guidance"].shape) == (2,)
    assert float(out["guidance"][0]) == pytest.approx(3.5)


def test_prepare_transformer_input_threads_rollout_guidance() -> None:
    """The rollout guidance scalar (from batch.context) overrides the default."""

    model = _replay_model()
    latents, prompt_embeds, pooled, timestep = _packed_inputs()
    out = model.diffusion_nft_prepare_transformer_input(
        latents=latents,
        prompt_embeds=prompt_embeds,
        prompt_attention_mask=None,
        pooled_prompt_embeds=pooled,
        timestep=timestep,
        num_frames=1,
        height=256,
        width=256,
        guidance_scale=4.5,
    )
    assert float(out["guidance"][0]) == pytest.approx(4.5)


def test_prepare_transformer_input_requires_pooled_embeds() -> None:
    """FLUX needs the CLIP pooled vector; a missing one fails loud, not silently."""

    model = _replay_model()
    latents, prompt_embeds, _pooled, timestep = _packed_inputs()
    with pytest.raises(RuntimeError, match="pooled_prompt_embeds"):
        model.diffusion_nft_prepare_transformer_input(
            latents=latents,
            prompt_embeds=prompt_embeds,
            prompt_attention_mask=None,
            pooled_prompt_embeds=None,
            timestep=timestep,
            num_frames=1,
            height=256,
            width=256,
        )


def test_prepare_transformer_input_rejects_inconsistent_grid() -> None:
    """A packed sequence length that cannot form the height:width grid fails loud."""

    model = _replay_model()
    # 255 is not a square count -> no integer 1:1 grid for a 256x256 image.
    latents = torch.randn(2, 255, 64)
    prompt_embeds = torch.randn(2, 7, 4096)
    pooled = torch.randn(2, 768)
    with pytest.raises(RuntimeError, match="packed latent grid"):
        model.diffusion_nft_prepare_transformer_input(
            latents=latents,
            prompt_embeds=prompt_embeds,
            prompt_attention_mask=None,
            pooled_prompt_embeds=pooled,
            timestep=torch.full((2,), 500.0),
            num_frames=1,
            height=256,
            width=256,
        )


def _peft_default_only_model() -> FluxReplayModel:
    """A FLUX replay model carrying a real PEFT ``default`` adapter (no previous yet).

    Built via ``get_peft_model`` exactly as the production LoRA path does, so the
    transformer is a PeftModel whose ``add_adapter(name, config)`` signature the
    ``previous`` attach relies on.
    """

    from peft import LoraConfig, get_peft_model

    base = build_tiny_wan_transformer()
    base.requires_grad_(False)
    peft_t = get_peft_model(
        base,
        LoraConfig(
            r=4,
            lora_alpha=8,
            init_lora_weights="gaussian",
            target_modules=_TINY_WAN_LORA_TARGETS,
        ),
    )
    return FluxReplayModel(transformer=peft_t, scheduler=None, device="cpu")


def _build() -> SimpleNamespace:
    return SimpleNamespace(
        lora={"rank": 4, "alpha": 8, "target_modules": _TINY_WAN_LORA_TARGETS},
    )


def test_attach_previous_policy_adapter_builds_frozen_mirror() -> None:
    """attach builds a frozen ``previous`` adapter seeded from ``default``."""

    model = _peft_default_only_model()
    assert "previous" not in model.transformer.peft_config

    model.attach_previous_policy_adapter(_build())

    assert "previous" in model.transformer.peft_config
    prev = {n: p for n, p in model.transformer.named_parameters() if ".previous." in n}
    assert prev, "no previous-adapter params were created"
    # forward-only mirror: every previous param is frozen (DDP reducer ignores it).
    assert all(not p.requires_grad for p in prev.values())
    # at least one default param stays trainable (the optimized adapter).
    trainable = [
        n for n, p in model.transformer.named_parameters() if ".default." in n and p.requires_grad
    ]
    assert trainable
    # seeded == default: a default lora_A and its previous twin match after attach.
    a_name = next(n for n in prev if "lora_A" in n)
    d_name = a_name.replace(".previous.", ".default.")
    named = dict(model.transformer.named_parameters())
    assert torch.allclose(named[a_name], named[d_name])


def test_sync_previous_policy_adapter_refreshes_from_default() -> None:
    """sync(decay=0) re-copies the trained ``default`` weights into ``previous``."""

    model = _peft_default_only_model()
    model.attach_previous_policy_adapter(_build())
    named = dict(model.transformer.named_parameters())
    a_name = next(n for n in named if ".previous." in n and "lora_A" in n)
    d_name = a_name.replace(".previous.", ".default.")

    # Simulate an optimizer step moving the trainable default away from previous.
    with torch.no_grad():
        named[d_name].add_(1.0)
    assert not torch.allclose(named[a_name], named[d_name])

    model.sync_previous_policy_adapter(decay=0.0)
    assert torch.allclose(named[a_name], named[d_name])
