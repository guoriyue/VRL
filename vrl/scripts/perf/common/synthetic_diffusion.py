"""Synthetic production-shape diffusion transformers for perf probes."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
import torch.nn as nn


def build_synthetic_inputs(
    family: str,
    *,
    batch: int,
    device: torch.device,
    dtype: torch.dtype,
    layers: int | None = None,
    concat_padding_mask: bool = True,
) -> tuple[nn.Module, dict[str, Any]]:
    """Config-init transformer + forward kwargs for a production diffusion family.

    Random weights are intentional: kernel time, compile guards, and GEMM shapes are
    driven by model structure and tensor shapes, not checkpoint values.
    """

    torch.manual_seed(0)

    if family == "sd3_5":
        from diffusers import SD3Transformer2DModel

        hidden = 1536
        model = SD3Transformer2DModel(
            sample_size=64,
            patch_size=2,
            in_channels=16,
            out_channels=16,
            num_layers=layers or 24,
            attention_head_dim=64,
            num_attention_heads=24,
            joint_attention_dim=4096,
            caption_projection_dim=hidden,
            pooled_projection_dim=2048,
            pos_embed_max_size=192,
        ).to(device=device, dtype=dtype).eval()
        h = w = 64
        seq = 333
        kwargs: dict[str, Any] = dict(
            hidden_states=torch.randn(batch, 16, h, w, device=device, dtype=dtype),
            encoder_hidden_states=torch.randn(batch, seq, 4096, device=device, dtype=dtype),
            pooled_projections=torch.randn(batch, 2048, device=device, dtype=dtype),
            timestep=torch.randint(0, 1000, (batch,), device=device).to(dtype),
            return_dict=False,
        )
        return model, kwargs

    if family in ("cosmos-predict2", "cosmos-predict2.5"):
        from diffusers import CosmosTransformer3DModel

        in_ch = 16 + (1 if concat_padding_mask else 0)
        model = CosmosTransformer3DModel(
            in_channels=in_ch,
            out_channels=16,
            num_attention_heads=16,
            attention_head_dim=128,
            num_layers=layers or 28,
            mlp_ratio=4.0,
            text_embed_dim=1024,
            adaln_lora_dim=256,
            max_size=(128, 240, 240),
            patch_size=(1, 2, 2),
            concat_padding_mask=concat_padding_mask,
        ).to(device=device, dtype=dtype).eval()
        t, h, w = 1, 44, 80
        seq = 512
        kwargs = dict(
            hidden_states=torch.randn(batch, in_ch, t, h, w, device=device, dtype=dtype),
            timestep=torch.rand(batch, device=device, dtype=dtype),
            encoder_hidden_states=torch.randn(batch, seq, 1024, device=device, dtype=dtype),
            padding_mask=(
                torch.zeros(1, 1, h, w, device=device, dtype=dtype)
                if concat_padding_mask
                else None
            ),
            return_dict=False,
        )
        return model, kwargs

    if family == "wan_2_1":
        from diffusers import WanTransformer3DModel

        model = WanTransformer3DModel(
            patch_size=(1, 2, 2),
            num_attention_heads=40,
            attention_head_dim=128,
            in_channels=16,
            out_channels=16,
            text_dim=4096,
            freq_dim=256,
            ffn_dim=13824,
            num_layers=layers or 40,
            rope_max_seq_len=1024,
        ).to(device=device, dtype=dtype).eval()
        t, h, w = 3, 30, 52
        seq = 512
        kwargs = dict(
            hidden_states=torch.randn(batch, 16, t, h, w, device=device, dtype=dtype),
            timestep=torch.rand(batch, device=device, dtype=dtype),
            encoder_hidden_states=torch.randn(batch, seq, 4096, device=device, dtype=dtype),
            return_dict=False,
        )
        return model, kwargs

    raise ValueError(f"unknown family {family!r}")


def build_synthetic_forward(
    family: str,
    *,
    batch: int,
    device: torch.device,
    dtype: torch.dtype,
    layers: int | None = None,
    concat_padding_mask: bool = True,
) -> tuple[nn.Module, Callable[[], Any]]:
    """Config-init transformer plus a no-grad forward thunk."""

    model, kwargs = build_synthetic_inputs(
        family,
        batch=batch,
        device=device,
        dtype=dtype,
        layers=layers,
        concat_padding_mask=concat_padding_mask,
    )

    def forward_fn() -> Any:
        with torch.no_grad():
            return model(**kwargs)

    return model, forward_fn
