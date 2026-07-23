from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch import nn

from vrl.models.steps.denoise.common.lora import (
    LoraModelMixin,
    build_lora_config,
)

pytest.importorskip("peft")


class _TinyTransformer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(2, 2, bias=False)


class _Policy(LoraModelMixin):
    def __init__(self) -> None:
        self.transformer = _TinyTransformer()
        self.device = "cpu"

    def _set_transformer(self, transformer: Any) -> None:
        self.transformer = transformer


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
    ((0.35, 0.35), (None, 0.0)),
)
def test_shared_fresh_adapter_preserves_effective_dropout(
    configured_dropout: float | None,
    expected_dropout: float,
) -> None:
    policy = _Policy()
    policy.apply_lora(
        SimpleNamespace(
            parameter_dtype=torch.float32,
            rollout=None,
            defer_trainable_device_move=False,
            lora_path=None,
            lora=_lora_values(configured_dropout),
        ),
    )

    assert policy.transformer.peft_config["default"].lora_dropout == expected_dropout


@pytest.mark.parametrize(
    ("configured_dropout", "expected_dropout"),
    ((0.35, 0.35), (None, 0.0)),
)
def test_previous_adapter_config_preserves_effective_dropout(
    configured_dropout: float | None,
    expected_dropout: float,
) -> None:
    config = build_lora_config(_lora_values(configured_dropout))

    assert config.lora_dropout == expected_dropout


def test_shared_warm_start_validates_effective_topology(
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
        "vrl.models.steps.denoise.common.lora.load_trainable_lora_adapter",
        fake_load,
    )
    policy = _Policy()
    base = policy.transformer

    policy.apply_lora(
        SimpleNamespace(
            parameter_dtype=torch.float32,
            rollout=None,
            defer_trainable_device_move=False,
            lora_path="/adapter",
            lora=_lora_values(0.35),
        ),
    )

    assert captured == {
        "base": base,
        "adapter_path": "/adapter",
        "expected_rank": 2,
        "expected_alpha": 4,
        "expected_dropout": 0.35,
        "expected_target_modules": ["proj"],
        "active_adapter": "default",
    }


def test_shared_warm_start_validation_failure_keeps_raw_transformer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def reject(*_args: Any, **_kwargs: Any) -> Any:
        raise ValueError("topology mismatch")

    monkeypatch.setattr(
        "vrl.models.steps.denoise.common.lora.load_trainable_lora_adapter",
        reject,
    )
    policy = _Policy()
    base = policy.transformer

    with pytest.raises(ValueError, match="topology mismatch"):
        policy.apply_lora(
            SimpleNamespace(
                parameter_dtype=torch.float32,
                rollout=None,
                defer_trainable_device_move=False,
                lora_path="/adapter",
                lora=_lora_values(0.35),
            ),
        )

    assert policy.transformer is base
