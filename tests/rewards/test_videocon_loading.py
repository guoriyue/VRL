"""Compatibility coverage for the optional legacy VideoCon model loader."""

import json
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch
from transformers import LlamaTokenizer, PretrainedConfig

from vrl.rewards.models import videocon_physics


@pytest.mark.parametrize("fp32_modules", [["wo"], ["actual_module"], None])
def test_composite_config_logging_does_not_construct_vendor_defaults(
    tmp_path, monkeypatch, fp32_modules
):
    class LegacyConfig(PretrainedConfig):
        is_composition = True

        def __init__(self, text_config=None, **kwargs):
            if text_config is None:
                raise ImportError("legacy default imports a missing sibling package")
            super().__init__(**kwargs)
            self.text_config = text_config
            self.vendor_initialized = True

    captured = {}

    class Model:
        config_class = LegacyConfig

        @staticmethod
        def from_pretrained(path, *, config, torch_dtype):
            captured.update(path=path, config=config, dtype=torch_dtype)
            assert config.vendor_initialized
            # This is where modern Transformers serializes the composite config.
            assert json.loads(config.to_json_string())["text_config"] == {"model_type": "llama"}
            return torch.nn.Linear(1, 1).to(torch_dtype)

    (tmp_path / "config.json").write_text(json.dumps({"text_config": {"model_type": "llama"}}))
    monkeypatch.setattr(videocon_physics, "resolve_model_root", lambda *a, **kw: tmp_path)
    tokenizer = SimpleNamespace(encode=lambda text, **kw: [3869 if text == "Yes" else 1939])
    monkeypatch.setattr(LlamaTokenizer, "from_pretrained", lambda *a, **kw: tokenizer)
    modeling = ModuleType("mplug_owl_video.modeling_mplug_owl")
    modeling.MplugOwlForConditionalGeneration = Model
    vendor_base = type("VendorBase", (), {"_keep_in_fp32_modules": fp32_modules})
    modeling.MplugOwlPreTrainedModel = vendor_base
    processing = ModuleType("mplug_owl_video.processing_mplug_owl")
    processing.MplugOwlImageProcessor = SimpleNamespace(from_pretrained=lambda *a: object())

    class Processor:
        def __init__(self, image_processor, tokenizer):
            assert self.get_attributes() == []
            self.image_processor = image_processor
            self.tokenizer = tokenizer

    processing.MplugOwlProcessor = Processor
    monkeypatch.setitem(sys.modules, "mplug_owl_video.modeling_mplug_owl", modeling)
    monkeypatch.setitem(sys.modules, "mplug_owl_video.processing_mplug_owl", processing)

    reward = videocon_physics.VideoConPhysicsModel({"device": "cpu", "dtype": "bfloat16"})

    assert captured["path"] == str(tmp_path)
    assert captured["dtype"] == torch.bfloat16
    assert reward.token_id_yes == 3869
    assert reward.token_id_no == 1939
    assert next(reward.model.parameters()).device.type == "cpu"
    assert not LegacyConfig.has_no_defaults_at_init
    assert vendor_base._keep_in_fp32_modules == ([] if fp32_modules == ["wo"] else fp32_modules)
