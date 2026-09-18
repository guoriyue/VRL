from __future__ import annotations

from typing import Any

import pytest
import torch
from torch import nn

from tests.models.steps.denoise.fixtures import lora_test_build
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
    model = CosmosPredict25ReplayModel(
        transformer=_Base(),
        scheduler=object(),
        device="cpu",
    )
    base = model.transformer

    model.apply_lora(
        lora_test_build(
            family="cosmos_predict2_5",
            lora_path="/adapter",
            lora={
                "rank": 2,
                "alpha": 4,
                "dropout": 0.3,
                "target_modules": ["proj"],
            },
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
        "autocast_adapter_dtype": True,
        "active_adapter": "default",
    }


def test_predict25_warm_start_validation_failure_keeps_raw_transformer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def reject(*_args: Any, **_kwargs: Any) -> Any:
        raise ValueError("topology mismatch")

    monkeypatch.setattr(
        "vrl.models.peft_adapter.load_trainable_lora_adapter",
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
            lora_test_build(
                family="cosmos_predict2_5",
                lora_path="/adapter",
                lora={
                    "rank": 2,
                    "alpha": 4,
                    "dropout": 0.3,
                    "target_modules": ["proj"],
                },
            ),
        )

    assert model.transformer is base


@pytest.mark.parametrize("previous_policy_adapter", [False, True])
def test_predict25_build_only_attaches_requested_previous(previous_policy_adapter: bool) -> None:
    model = CosmosPredict25ReplayModel(transformer=_Base(), scheduler=object(), device="cpu")
    model.apply_lora(
        lora_test_build(
            family="cosmos_predict2_5",
            lora={"rank": 2, "alpha": 4, "target_modules": ["proj"]},
            previous_policy_adapter=previous_policy_adapter,
        ),
    )
    assert ("previous" in model.transformer.peft_config) is previous_policy_adapter
    if previous_policy_adapter:
        parameters = dict(model.transformer.named_parameters())
        for name, parameter in parameters.items():
            if ".previous." in name:
                assert not parameter.requires_grad
                torch.testing.assert_close(
                    parameter, parameters[name.replace(".previous.", ".default.")]
                )
