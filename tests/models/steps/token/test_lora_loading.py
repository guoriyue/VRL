from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from vrl.models.steps.token.lora import install_token_lora_adapter


def test_token_warm_start_validates_resolved_family_topology(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = object()
    sentinel = object()
    captured: dict[str, Any] = {}

    def fake_load(
        model: Any,
        adapter_path: str,
        **kwargs: Any,
    ) -> Any:
        captured.update(model=model, adapter_path=adapter_path, **kwargs)
        return sentinel

    monkeypatch.setattr(
        "vrl.models.steps.token.lora.load_trainable_lora_adapter",
        fake_load,
    )

    loaded = install_token_lora_adapter(
        base,
        SimpleNamespace(
            lora_path="/adapter",
            lora_rank=8,
            lora_alpha=16,
            lora_dropout=0.2,
            lora_target_modules=("v_proj", "q_proj"),
        ),
    )

    assert loaded is sentinel
    assert captured == {
        "model": base,
        "adapter_path": "/adapter",
        "expected_rank": 8,
        "expected_alpha": 16,
        "expected_dropout": 0.2,
        "expected_target_modules": ("v_proj", "q_proj"),
        "expected_task_type": None,
    }


def test_token_warm_start_validation_failure_is_not_silently_replaced(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def reject(*_args: Any, **_kwargs: Any) -> Any:
        raise ValueError("topology mismatch")

    monkeypatch.setattr(
        "vrl.models.steps.token.lora.load_trainable_lora_adapter",
        reject,
    )

    with pytest.raises(
        RuntimeError,
        match="failed to load trainable token LoRA adapter",
    ) as error:
        install_token_lora_adapter(
            object(),
            SimpleNamespace(
                lora_path="/adapter",
                lora_rank=8,
                lora_alpha=16,
                lora_dropout=0.2,
                lora_target_modules=("v_proj", "q_proj"),
            ),
        )

    assert isinstance(error.value.__cause__, ValueError)
    assert str(error.value.__cause__) == "topology mismatch"
