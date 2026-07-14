"""NVFP4's conservative MLP targeting against real diffusers module paths."""

from __future__ import annotations

import pytest
from torch import nn

from tests.models.diffusion.fixtures import (
    build_tiny_flux_transformer,
    build_tiny_lumina2_transformer,
    build_tiny_qwen_image_transformer,
    build_tiny_sana_transformer,
    build_tiny_sd3_transformer,
    build_tiny_wan_transformer,
)
from vrl.nn.quantization.targeting import LinearTargetProfile, matches_linear_target


@pytest.mark.parametrize(
    "builder",
    [
        build_tiny_wan_transformer,
        build_tiny_sd3_transformer,
        build_tiny_flux_transformer,
        build_tiny_qwen_image_transformer,
        build_tiny_lumina2_transformer,
    ],
)
def test_nvfp4_selects_real_mlp_paths_but_not_attention(builder) -> None:
    transformer = builder()
    linear_paths = [
        path for path, module in transformer.named_modules() if isinstance(module, nn.Linear)
    ]

    selected = [
        path for path in linear_paths if matches_linear_target(path, LinearTargetProfile.MLP_ONLY)
    ]

    assert selected
    assert not any("attn" in path for path in selected)
    assert not any(path.endswith(("to_q", "to_k", "to_v", "to_out.0")) for path in selected)


def test_nvfp4_does_not_misclassify_sana_attention_when_mlp_uses_convolutions() -> None:
    transformer = build_tiny_sana_transformer()
    linear_paths = [
        path for path, module in transformer.named_modules() if isinstance(module, nn.Linear)
    ]

    assert not any(
        matches_linear_target(path, LinearTargetProfile.MLP_ONLY) for path in linear_paths
    )
