"""AR rollout fp8 quantization: trunk swaps, heads/embeddings stay.

Uses a tiny REAL Emu3 model (config-init, no weights) so the exclusion rules
are checked against genuine HF module paths, not a fake that restates them.
"""

from __future__ import annotations

import pytest

from vrl.nn.quantization import QuantizedLinear


def _tiny_emu3_model():
    """Tiny real Emu3 with a trunk WIDE enough to pass the fp8 size gate.

    The swap only quantizes linears with in/out features >= 1024 (small GEMMs
    are cheap and quantizing them only adds drift), so the stock ~32-dim tiny
    fixture swaps nothing; widen the trunk while keeping it one layer.
    """
    import torch
    from transformers.models.emu3.modeling_emu3 import Emu3ForConditionalGeneration

    from tests.models.ar.emu3.fixtures import _stub_processor, tiny_hf_emu3_config
    from vrl.models.ar.emu3.model import Emu3Config, Emu3Model

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
        model_path="tiny-emu3", dtype="float32", device="cpu", use_lora=False,
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


def test_ar_builder_applies_rollout_quantization_and_replay_does_not() -> None:
    """The shared AR builder quantizes rollout bundles only."""
    from vrl.models.ar.build import build_ar_runtime_bundle
    from vrl.models.ar.capabilities import ar_discrete_family_capability
    from vrl.models.interfaces.runtime import RuntimeBuildSpec

    capability = ar_discrete_family_capability("emu3", "ar_t2i")

    def _spec(quant: str | None) -> RuntimeBuildSpec:
        return RuntimeBuildSpec(
            model_name_or_path="fake/repo",
            device="cpu",
            dtype="float32",
            ar_task="ar_t2i",
            model_config={"path": "fake/repo", "use_lora": False},
            rollout_quantization=quant,
        )

    rollout_model = _tiny_emu3_model()
    build_ar_runtime_bundle(_spec("fp8"), model=rollout_model, capability=capability)
    assert any(
        isinstance(m, QuantizedLinear) for m in rollout_model.language_model.modules()
    ), "rollout bundle did not quantize"

    replay_model = _tiny_emu3_model()
    build_ar_runtime_bundle(
        _spec("fp8"), model=replay_model, capability=capability, replay=True,
    )
    assert not any(
        isinstance(m, QuantizedLinear) for m in replay_model.language_model.modules()
    ), "replay bundle must keep the bf16 master unquantized"

    plain_model = _tiny_emu3_model()
    build_ar_runtime_bundle(_spec(None), model=plain_model, capability=capability)
    assert not any(
        isinstance(m, QuantizedLinear) for m in plain_model.language_model.modules()
    )
