from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch import nn

from tests.models.steps.denoise.fixtures import lora_test_build
from vrl.models.families.wan_2_1.model import WanT2VDiffusersModel

pytest.importorskip("peft")


class _TinyTransformer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(2, 2, bias=False)


def _model() -> WanT2VDiffusersModel:
    return WanT2VDiffusersModel(
        pipeline=SimpleNamespace(
            transformer=_TinyTransformer(),
            config={},
            device="cpu",
            scheduler=None,
        ),
        device="cpu",
    )


def _lora_values(dropout: float | None) -> dict[str, Any]:
    values: dict[str, Any] = {
        "rank": 2,
        "alpha": 4,
        "target_modules": ["proj"],
    }
    if dropout is not None:
        values["dropout"] = dropout
    return values


@pytest.mark.parametrize(
    ("configured_dropout", "expected_dropout"),
    ((0.4, 0.4), (None, 0.0)),
)
def test_wan_fresh_adapter_preserves_base_output_and_effective_dropout(
    configured_dropout: float | None,
    expected_dropout: float,
) -> None:
    model = _model()
    inputs = torch.ones(1, 2)
    before = model.transformer.proj(inputs).detach()

    model.apply_lora(
        lora_test_build(
            _lora_values(configured_dropout),
            family="wan_2_1",
            lora_path=None,
            model_config={"boundary_ratio": None, "trainable_transformers": ["transformer"]},
        ),
    )

    assert model.transformer.peft_config["default"].lora_dropout == expected_dropout
    torch.testing.assert_close(model.transformer.proj(inputs), before, rtol=0, atol=0)


def test_wan_warm_start_validates_effective_topology(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    class _Wrapped(nn.Module):
        def set_adapter(self, name: str) -> None:
            captured["active_adapter"] = name

    def fake_load(
        base: Any,
        adapter_path: str,
        **kwargs: Any,
    ) -> Any:
        captured.update(base=base, adapter_path=adapter_path, **kwargs)
        return _Wrapped()

    monkeypatch.setattr(
        "vrl.models.peft_adapter.load_trainable_lora_adapter",
        fake_load,
    )
    model = _model()
    base = model.transformer

    model.apply_lora(
        lora_test_build(
            _lora_values(0.4),
            family="wan_2_1",
            lora_path="/adapter",
            model_config={"boundary_ratio": None, "trainable_transformers": ["transformer"]},
        ),
    )

    assert captured == {
        "base": base,
        "adapter_path": "/adapter",
        # Warm start must construct adapters in the transformer's resolved
        # dtype (no peft fp32 upcast) so it matches the fresh-create branch.
        "autocast_adapter_dtype": False,
        "expected_rank": 2,
        "expected_alpha": 4,
        "expected_dropout": 0.4,
        "expected_target_modules": ["proj"],
        "adapter_name": "default",
        "active_adapter": "default",
    }


def test_wan_warm_start_validation_failure_keeps_raw_transformer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def reject(*_args: Any, **_kwargs: Any) -> Any:
        raise ValueError("topology mismatch")

    monkeypatch.setattr(
        "vrl.models.peft_adapter.load_trainable_lora_adapter",
        reject,
    )
    model = _model()
    base = model.transformer

    with pytest.raises(ValueError, match="topology mismatch"):
        model.apply_lora(
            lora_test_build(
                _lora_values(0.4),
                family="wan_2_1",
                lora_path="/adapter",
                model_config={"boundary_ratio": None, "trainable_transformers": ["transformer"]},
            ),
        )

    assert model.transformer is base


@pytest.mark.parametrize("adapter_dtype", [None, "float32"])
@pytest.mark.parametrize("warm_start", [False, True])
def test_wan_adapter_storage_preserves_frozen_base(
    adapter_dtype: str | None,
    warm_start: bool,
    tmp_path: Path,
) -> None:
    model = _model()
    model.transformer.to(dtype=torch.bfloat16)
    base = model.transformer.proj.weight
    before = base.detach().clone()
    pointer = base.data_ptr()
    build = lora_test_build(
        {**_lora_values(0.0), "parameter_dtype": adapter_dtype},
        family="wan_2_1",
        lora_path=None,
    )
    if warm_start:
        source = _model()
        source.apply_lora(build)
        source.transformer.save_pretrained(tmp_path)
        build.model_config["lora"]["path"] = str(tmp_path)
    model.apply_lora(build)
    expected = torch.float32 if adapter_dtype else torch.bfloat16
    trainable = [p for p in model.transformer.parameters() if p.requires_grad]
    assert trainable and all(p.dtype == expected for p in trainable)
    assert not base.requires_grad
    assert base.dtype == torch.bfloat16 and base.data_ptr() == pointer
    assert torch.equal(base, before)
