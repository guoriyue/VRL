from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from torch import nn

from vrl.models.families.cosmos.predict2_5.model import (
    CosmosPredict25ReplayModel,
)


class _Base(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(2, 2, bias=False)


def test_predict25_warm_start_validates_effective_topology(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    class _Wrapped(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.peft_config = {"previous": object()}

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
        "vrl.models.families.cosmos.predict2_5.model.load_trainable_lora_adapter",
        fake_load,
    )
    monkeypatch.setattr(
        "vrl.models.families.cosmos.predict2_5.model._copy_adapter_weights",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "vrl.models.families.cosmos.predict2_5.model._freeze_checkpoint_owned_adapter_params",
        lambda *_args, **_kwargs: None,
    )
    model = CosmosPredict25ReplayModel(
        transformer=_Base(),
        scheduler=object(),
        device="cpu",
    )
    base = model.transformer

    model.apply_lora(
        SimpleNamespace(
            lora_path="/adapter",
            lora={
                "rank": 2,
                "alpha": 4,
                "dropout": 0.3,
                "target_modules": ["proj"],
            },
            defer_trainable_device_move=False,
        ),
    )

    assert captured == {
        "base": base,
        "adapter_path": "/adapter",
        "expected_rank": 2,
        "expected_alpha": 4,
        "expected_dropout": 0.3,
        "expected_target_modules": ["proj"],
        "adapter_name": "default",
        "active_adapter": "default",
    }


def test_predict25_warm_start_validation_failure_keeps_raw_transformer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def reject(*_args: Any, **_kwargs: Any) -> Any:
        raise ValueError("topology mismatch")

    monkeypatch.setattr(
        "vrl.models.families.cosmos.predict2_5.model.load_trainable_lora_adapter",
        reject,
    )
    model = CosmosPredict25ReplayModel(
        transformer=_Base(),
        scheduler=object(),
        device="cpu",
    )
    base = model.transformer

    with pytest.raises(ValueError, match="topology mismatch"):
        model.apply_lora(
            SimpleNamespace(
                lora_path="/adapter",
                lora={
                    "rank": 2,
                    "alpha": 4,
                    "dropout": 0.3,
                    "target_modules": ["proj"],
                },
                defer_trainable_device_move=False,
            ),
        )

    assert model.transformer is base
