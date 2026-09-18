from __future__ import annotations

from typing import Any

import pytest
import torch
from torch import nn

from tests.models.steps.denoise.fixtures import lora_test_build
from vrl.models.steps.denoise import DiffusionModelBase

pytest.importorskip("peft")


class _TinyTransformer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(2, 2, bias=False)


class _Policy(DiffusionModelBase):
    """Smallest real base subclass: the attach path is the base's own."""

    def __init__(self) -> None:
        super().__init__()
        self.transformer = _TinyTransformer()
        self.device = "cpu"

    def encode_prompt(self, prompt, negative_prompt=None, **kwargs):  # pragma: no cover
        raise NotImplementedError

    def prepare_sampling(self, request, encoded, **kwargs):  # pragma: no cover
        raise NotImplementedError

    def forward_step(self, state, step_idx):  # pragma: no cover
        raise NotImplementedError

    def decode_latents(self, latents):  # pragma: no cover
        raise NotImplementedError


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
        lora_test_build(
            _lora_values(configured_dropout),
            family="sd3_5",
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
    policy = _Policy()
    policy.apply_lora(lora_test_build(_lora_values(configured_dropout), family="sd3_5"))
    policy.attach_previous_policy_adapter()

    config = policy.transformer.peft_config
    assert config["previous"] is not config["default"]
    assert config["previous"].lora_dropout == expected_dropout
    assert config["previous"].r == config["default"].r
    assert config["previous"].target_modules == config["default"].target_modules


@pytest.mark.parametrize(
    "autocast,parameter_dtype,expected_dtype",
    [
        (True, None, torch.float32),
        (False, None, torch.bfloat16),
        (False, "float32", torch.float32),
    ],
)
def test_previous_adapter_matches_trainable_storage(
    autocast, parameter_dtype, expected_dtype
) -> None:
    policy = _Policy()
    policy.transformer.to(dtype=torch.bfloat16)
    policy.apply_lora(
        lora_test_build(
            {
                **_lora_values(None),
                "autocast_adapter_dtype": autocast,
                "parameter_dtype": parameter_dtype,
                "previous_adapter": True,
            },
            family="flux",
        ),
    )
    parameters = dict(policy.transformer.named_parameters())
    for name, parameter in parameters.items():
        if ".default." in name:
            previous = parameters[name.replace(".default.", ".previous.")]
            assert parameter.dtype == previous.dtype == expected_dtype
            torch.testing.assert_close(previous, parameter, rtol=0, atol=0)
            assert not previous.requires_grad


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
        "vrl.models.peft_adapter.load_trainable_lora_adapter",
        fake_load,
    )
    policy = _Policy()
    base = policy.transformer

    policy.apply_lora(
        lora_test_build(
            _lora_values(0.35),
            family="sd3_5",
            lora_path="/adapter",
        ),
    )

    assert captured == {
        "base": base,
        "adapter_path": "/adapter",
        "expected_rank": 2,
        "expected_alpha": 4,
        "expected_dropout": 0.35,
        "expected_target_modules": ["proj"],
        "adapter_name": "default",
        "autocast_adapter_dtype": True,
        "active_adapter": "default",
    }


def test_shared_warm_start_validation_failure_keeps_raw_transformer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def reject(*_args: Any, **_kwargs: Any) -> Any:
        raise ValueError("topology mismatch")

    monkeypatch.setattr(
        "vrl.models.peft_adapter.load_trainable_lora_adapter",
        reject,
    )
    policy = _Policy()
    base = policy.transformer

    with pytest.raises(ValueError, match="topology mismatch"):
        policy.apply_lora(
            lora_test_build(
                _lora_values(0.35),
                family="sd3_5",
                lora_path="/adapter",
            ),
        )

    assert policy.transformer is base
