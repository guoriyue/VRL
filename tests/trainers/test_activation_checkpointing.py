from __future__ import annotations

import logging

import pytest
from omegaconf import OmegaConf

from vrl.trainers.activation_checkpointing import (
    enable_transformer_gradient_checkpointing,
)


def _config(mode: str | bool):
    return OmegaConf.create(
        {
            "actor": {"gradient_checkpointing": mode},
            "model": {"torch_compile": {"enable": False}},
        },
    )


@pytest.mark.parametrize("mode", ["off", False])
def test_gradient_checkpointing_off_skips_trainable_modules(mode: str | bool) -> None:
    class _Transformer:
        def enable_gradient_checkpointing(self) -> None:
            raise AssertionError("off must not enable gradient checkpointing")

    bundle = type("_Bundle", (), {"trainable_modules": {"transformer": _Transformer()}})()

    enable_transformer_gradient_checkpointing(bundle, _config(mode))


@pytest.mark.parametrize("mode", ["full", True])
def test_full_gradient_checkpointing_uses_the_native_method(mode: str | bool) -> None:
    calls: list[dict[str, object]] = []

    class _Transformer:
        def enable_gradient_checkpointing(self, **kwargs: object) -> None:
            calls.append(kwargs)

    bundle = type("_Bundle", (), {"trainable_modules": {"transformer": _Transformer()}})()

    enable_transformer_gradient_checkpointing(bundle, _config(mode))

    assert calls == [{}]


def test_selective_gradient_checkpointing_passes_a_custom_function() -> None:
    calls: list[dict[str, object]] = []

    class _Transformer:
        def enable_gradient_checkpointing(self, **kwargs: object) -> None:
            calls.append(kwargs)

    bundle = type("_Bundle", (), {"trainable_modules": {"transformer": _Transformer()}})()

    enable_transformer_gradient_checkpointing(bundle, _config("selective"))

    assert len(calls) == 1
    assert callable(calls[0]["gradient_checkpointing_func"])


def test_selective_gradient_checkpointing_reports_legacy_full_fallback(
    caplog: pytest.LogCaptureFixture,
) -> None:
    calls = 0

    class _LegacyTransformer:
        def enable_gradient_checkpointing(self) -> None:
            nonlocal calls
            calls += 1

    bundle = type(
        "_Bundle",
        (),
        {"trainable_modules": {"transformer": _LegacyTransformer()}},
    )()

    with caplog.at_level(logging.WARNING):
        enable_transformer_gradient_checkpointing(bundle, _config("selective"))

    assert calls == 1
    assert "falling back to full checkpointing" in caplog.text
