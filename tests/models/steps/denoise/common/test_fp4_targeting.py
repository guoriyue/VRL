"""NVFP4's conservative MLP targeting against real diffusers module paths."""

from __future__ import annotations

import pytest
from torch import nn

from tests.models.steps.denoise.fixtures import (
    build_tiny_transformer,
)
from vrl.nn.quantization.targeting import LinearTargetProfile


@pytest.mark.parametrize("name", ["wan", "sd3", "flux", "qwen_image", "lumina2"])
def test_nvfp4_selects_real_mlp_paths_but_not_attention(name) -> None:
    transformer = build_tiny_transformer(name)
    linear_paths = [
        path for path, module in transformer.named_modules() if isinstance(module, nn.Linear)
    ]

    selected = [path for path in linear_paths if LinearTargetProfile.MLP_ONLY.matches(path)]

    assert selected
    assert not any("attn" in path for path in selected)
    assert not any(path.endswith(("to_q", "to_k", "to_v", "to_out.0")) for path in selected)


def test_nvfp4_does_not_misclassify_sana_attention_when_mlp_uses_convolutions() -> None:
    transformer = build_tiny_transformer("sana")
    linear_paths = [
        path for path, module in transformer.named_modules() if isinstance(module, nn.Linear)
    ]

    assert not any(LinearTargetProfile.MLP_ONLY.matches(path) for path in linear_paths)
