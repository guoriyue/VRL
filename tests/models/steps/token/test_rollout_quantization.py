"""AR rollout quantization: policy trunks swap, heads/embeddings stay.

Uses a tiny REAL Emu3 model (config-init, no weights) so the exclusion rules
are checked against genuine HF module paths, not a fake that restates them.
"""

from __future__ import annotations

import pytest
from torch import nn

from vrl.config.precision import QuantizationPolicy, RolePrecision
from vrl.nn.quantization import Fp4Linear, Fp8Linear, QuantizedLinear


def _build_emu3_bundle(
    monkeypatch: pytest.MonkeyPatch,
    build,
    model,
    *,
    replay: bool = False,
):
    """Exercise the production registry build with a test-owned tiny model."""

    from vrl.models.families.registry import get_model_family_entry
    from vrl.utils import config as config_utils

    entry = get_model_family_entry("emu3")
    recipe = entry.family_build
    target_model_cls = recipe.replay_cls if replay else recipe.model_cls

    def import_test_object(path: str):
        if path == recipe.config_builder:
            return lambda _build: {}
        if path == recipe.config_cls:
            return lambda **_kwargs: object()
        if path == target_model_cls:
            return lambda _config: model
        raise AssertionError(f"unexpected import path: {path}")

    monkeypatch.setattr(config_utils, "import_from_path", import_test_object)
    return entry.build_replay(build) if replay else entry.build_rollout(build)


def _tiny_emu3_model(*, use_lora: bool = False):
    """Tiny real Emu3 with a trunk WIDE enough to pass the fp8 size gate.

    The swap only quantizes linears with in/out features >= 1024 (small GEMMs
    are cheap and quantizing them only adds drift), so the stock ~32-dim tiny
    fixture swaps nothing; widen the trunk while keeping it one layer.
    """
    import torch
    from transformers.models.emu3.modeling_emu3 import Emu3ForConditionalGeneration

    from tests.models.families.emu3.fixtures import _stub_processor, tiny_hf_emu3_config
    from vrl.models.families.emu3.config import Emu3Config
    from vrl.models.families.emu3.model import Emu3Model

    hf_config = tiny_hf_emu3_config()
    hf_config.text_config.hidden_size = 1024
    hf_config.text_config.intermediate_size = 1024
    hf_config.text_config.num_hidden_layers = 1
    hf_config.text_config.num_attention_heads = 8
    hf_config.text_config.num_key_value_heads = 8
    hf_config.text_config.head_dim = 128
    torch.manual_seed(0)
    hf_model = Emu3ForConditionalGeneration(hf_config)
    config = Emu3Config(
        model_path="tiny-emu3",
        dtype="float32",
        device="cpu",
        use_lora=use_lora,
    )
    return Emu3Model(config, emu3=hf_model, processor=_stub_processor())


def test_ar_quantize_rollout_fp8_swaps_trunk_not_heads() -> None:
    """Attention/MLP linears swap; lm_head and embeddings stay high precision."""
    torch = pytest.importorskip("torch")
    model = _tiny_emu3_model()

    swapped = model.quantize_rollout_fp8()

    assert swapped, "no linears swapped — min_features/exclude gate everything"
    assert all("head" not in path for path in swapped)
    assert all("embed" not in path for path in swapped)
    quantized = {
        name
        for name, module in model.language_model.named_modules()
        if isinstance(module, QuantizedLinear)
    }
    assert quantized, "swap reported paths but no QuantizedLinear present"
    # The vocabulary head must remain a plain Linear.
    head = model.language_model.get_output_embeddings()
    assert head is None or not isinstance(head, QuantizedLinear)
    del torch


def test_ar_quantize_rollout_nvfp4_targets_mlp_not_attention_or_heads() -> None:
    model = _tiny_emu3_model()

    swapped = model.quantize_rollout_nvfp4()

    assert swapped
    assert all("mlp" in path for path in swapped)
    assert not any("self_attn" in path for path in swapped)
    assert all("head" not in path and "embed" not in path for path in swapped)


@pytest.mark.parametrize(
    ("format_name", "linear_type"),
    [("fp8", Fp8Linear), ("nvfp4", Fp4Linear)],
)
def test_ar_worker_guard_requires_the_requested_format(
    format_name: str,
    linear_type: type[QuantizedLinear],
) -> None:
    """AR models expose language_model, not diffusion's transformer attribute."""
    from vrl.models.interfaces.runtime import (
        ModelBuild,
        RolloutBuildOptions,
    )
    from vrl.models.loader import assert_rollout_quantization_applied

    class _ArPolicy(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.language_model = nn.Sequential(
                linear_type(nn.Linear(64, 64, bias=False)),
            )

    build = ModelBuild(
        model_name_or_path="fake/repo",
        revision=None,
        device="cpu",
        parameter_dtype="bf16",
        family="emu3",
        precision=RolePrecision(
            "bf16",
            "tf32",
            QuantizationPolicy(format=format_name),
            outer_autocast=False,
        ),
        rollout=RolloutBuildOptions(
            prompt_encoder_dtype="bf16",
        ),
    )
    assert_rollout_quantization_applied(_ArPolicy(), build)


def test_ar_worker_guard_rejects_a_different_quantization_format() -> None:
    from vrl.models.interfaces.runtime import (
        ModelBuild,
        RolloutBuildOptions,
    )
    from vrl.models.loader import assert_rollout_quantization_applied

    model = nn.Module()
    model.language_model = nn.Sequential(Fp8Linear(nn.Linear(64, 64, bias=False)))
    build = ModelBuild(
        model_name_or_path="fake/repo",
        revision=None,
        device="cpu",
        parameter_dtype="bf16",
        family="emu3",
        precision=RolePrecision(
            "bf16",
            "tf32",
            QuantizationPolicy(format="nvfp4"),
            outer_autocast=False,
        ),
        rollout=RolloutBuildOptions(
            prompt_encoder_dtype="bf16",
        ),
    )

    with pytest.raises(RuntimeError, match="0 nvfp4 quantized linear"):
        assert_rollout_quantization_applied(model, build)


def test_ar_builder_rejects_unsupported_nvfp4_before_quantization_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vrl.models.interfaces.runtime import (
        ModelBuild,
        RolloutBuildOptions,
    )

    class _ArPolicy(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.language_model = nn.Linear(64, 64, bias=False)
            self.quantize_calls = 0

        def quantize_rollout_nvfp4(self) -> list[str]:
            self.quantize_calls += 1
            return ["language_model"]

    monkeypatch.setattr("vrl.nn.quantization.nvfp4_available", lambda _device: False)
    model = _ArPolicy()
    build = ModelBuild(
        model_name_or_path="fake/repo",
        revision=None,
        device="cpu",
        parameter_dtype="bf16",
        family="emu3",
        precision=RolePrecision(
            "bf16",
            "tf32",
            QuantizationPolicy(format="nvfp4"),
            outer_autocast=False,
        ),
        rollout=RolloutBuildOptions(
            prompt_encoder_dtype="bf16",
        ),
    )

    with pytest.raises(RuntimeError, match="NVFP4-capable CUDA target"):
        _build_emu3_bundle(monkeypatch, build, model)

    assert model.quantize_calls == 0


@pytest.mark.parametrize("format_name", ["fp8", "nvfp4"])
def test_ar_builder_applies_rollout_quantization_and_replay_does_not(
    format_name: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The shared AR builder quantizes rollout bundles only."""
    from vrl.models.interfaces.runtime import (
        ModelBuild,
        RolloutBuildOptions,
    )
    from vrl.models.loader import assert_rollout_quantization_applied

    if format_name == "nvfp4":
        monkeypatch.setattr("vrl.nn.quantization.nvfp4_available", lambda _device: True)

    def _build(
        quantization_format: str | None,
        *,
        for_rollout: bool = True,
        use_lora: bool = False,
    ) -> ModelBuild:
        return ModelBuild(
            model_name_or_path="fake/repo",
            revision=None,
            device="cpu",
            parameter_dtype="float32",
            family="emu3",
            precision=RolePrecision(
                "fp32",
                "tf32",
                None
                if quantization_format is None
                else QuantizationPolicy(format=quantization_format),
                outer_autocast=False,
            ),
            model_config={"path": "fake/repo", "use_lora": use_lora},
            rollout=(
                RolloutBuildOptions(
                    prompt_encoder_dtype="float16",
                )
                if for_rollout
                else None
            ),
        )

    rollout_model = _tiny_emu3_model()
    rollout_build = _build(format_name)
    _build_emu3_bundle(monkeypatch, rollout_build, rollout_model)
    assert any(isinstance(m, QuantizedLinear) for m in rollout_model.language_model.modules()), (
        "rollout bundle did not quantize"
    )
    # The common worker backstop must inspect the AR policy core, not assume the
    # diffusion-only ``model.transformer`` shape.
    assert_rollout_quantization_applied(rollout_model, rollout_build)

    with pytest.raises(
        RuntimeError,
        match=rf"rollout policy core has 0 {format_name} quantized",
    ):
        assert_rollout_quantization_applied(_tiny_emu3_model(), _build(format_name))

    replay_model = _tiny_emu3_model(use_lora=True)
    _build_emu3_bundle(
        monkeypatch,
        _build(None, for_rollout=False, use_lora=True),
        replay_model,
        replay=True,
    )
    assert not any(
        isinstance(m, QuantizedLinear) for m in replay_model.language_model.modules()
    ), "replay bundle must keep its base-precision parameters unquantized"

    with pytest.raises(ValueError, match="replay runtime construction"):
        _build_emu3_bundle(
            monkeypatch,
            _build(format_name),
            _tiny_emu3_model(),
            replay=True,
        )

    plain_model = _tiny_emu3_model()
    _build_emu3_bundle(monkeypatch, _build(None), plain_model)
    assert not any(isinstance(m, QuantizedLinear) for m in plain_model.language_model.modules())
