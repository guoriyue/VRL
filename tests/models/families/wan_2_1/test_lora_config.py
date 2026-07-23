from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from torch import nn

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
def test_wan_fresh_adapter_preserves_effective_dropout(
    configured_dropout: float | None,
    expected_dropout: float,
) -> None:
    model = _model()

    model.apply_lora(
        SimpleNamespace(
            lora_path=None,
            lora=_lora_values(configured_dropout),
            defer_trainable_device_move=False,
        ),
    )

    assert model.transformer.peft_config["default"].lora_dropout == expected_dropout


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
        "vrl.models.families.wan_2_1.model.load_trainable_lora_adapter",
        fake_load,
    )
    model = _model()
    base = model.transformer

    model.apply_lora(
        SimpleNamespace(
            lora_path="/adapter",
            lora=_lora_values(0.4),
            defer_trainable_device_move=False,
        ),
    )

    assert captured == {
        "base": base,
        "adapter_path": "/adapter",
        "expected_rank": 2,
        "expected_alpha": 4,
        "expected_dropout": 0.4,
        "expected_target_modules": ["proj"],
        "active_adapter": "default",
    }


def test_wan_warm_start_validation_failure_keeps_raw_transformer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def reject(*_args: Any, **_kwargs: Any) -> Any:
        raise ValueError("topology mismatch")

    monkeypatch.setattr(
        "vrl.models.families.wan_2_1.model.load_trainable_lora_adapter",
        reject,
    )
    model = _model()
    base = model.transformer

    with pytest.raises(ValueError, match="topology mismatch"):
        model.apply_lora(
            SimpleNamespace(
                lora_path="/adapter",
                lora=_lora_values(0.4),
                defer_trainable_device_move=False,
            ),
        )

    assert model.transformer is base
