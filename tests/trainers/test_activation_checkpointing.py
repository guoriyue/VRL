from __future__ import annotations

import copy
import logging
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from vrl.config.schema import parse_config
from vrl.trainers.activation_checkpointing import (
    cpu_checkpoint_func,
    enable_transformer_gradient_checkpointing,
)


def _config(mode: str | bool):
    return OmegaConf.create(
        {
            "actor": {"gradient_checkpointing": mode},
            "model": {"family": "sd3_5", "torch_compile": {"enable": False}},
        },
    )


def test_absent_gradient_checkpointing_defaults_to_off() -> None:
    class _Transformer:
        def enable_gradient_checkpointing(self) -> None:
            raise AssertionError("an absent public key must not enable checkpointing")

    cfg = OmegaConf.create(
        {
            "actor": {},
            "model": {"family": "sd3_5", "torch_compile": {"enable": False}},
        },
    )
    bundle = type("_Bundle", (), {"trainable_modules": {"transformer": _Transformer()}})()

    enable_transformer_gradient_checkpointing(bundle, parse_config(cfg))


def test_cpu_checkpoint_installs_explicit_function_and_rejects_fallback():
    calls = []
    module = SimpleNamespace(enable_gradient_checkpointing=lambda **kwargs: calls.append(kwargs))
    enable_transformer_gradient_checkpointing(
        SimpleNamespace(trainable_modules={"transformer": module}),
        parse_config(_config("full_cpu")),
    )
    assert calls == [{"gradient_checkpointing_func": cpu_checkpoint_func}]
    module = SimpleNamespace(enable_gradient_checkpointing=lambda: None)
    with pytest.raises(ValueError, match="cannot install full_cpu"):
        enable_transformer_gradient_checkpointing(
            SimpleNamespace(trainable_modules={"transformer": module}),
            parse_config(_config("full_cpu")),
        )


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_cpu_checkpoint_preserves_output_input_and_parameter_gradients(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    torch.manual_seed(19)
    direct = (
        torch.nn.Sequential(
            torch.nn.Linear(8, 16), torch.nn.Dropout(0.2), torch.nn.SiLU(), torch.nn.Linear(16, 4)
        )
        .double()
        .to(device)
    )
    checkpointed = copy.deepcopy(direct)
    x = torch.randn(3, 8, dtype=torch.float64, device=device, requires_grad=True)
    y = x.detach().clone().requires_grad_(True)
    torch.manual_seed(27)
    expected = direct(x)
    expected.square().sum().backward()
    torch.manual_seed(27)
    actual = cpu_checkpoint_func(checkpointed, y)
    actual.square().sum().backward()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(y.grad, x.grad, rtol=0, atol=0)
    for left, right in zip(direct.parameters(), checkpointed.parameters(), strict=True):
        torch.testing.assert_close(left.grad, right.grad, rtol=0, atol=0)


@pytest.mark.parametrize("mode", ["off", False])
def test_gradient_checkpointing_off_skips_trainable_modules(mode: str | bool) -> None:
    class _Transformer:
        def enable_gradient_checkpointing(self) -> None:
            raise AssertionError("off must not enable gradient checkpointing")

    bundle = type("_Bundle", (), {"trainable_modules": {"transformer": _Transformer()}})()

    enable_transformer_gradient_checkpointing(bundle, parse_config(_config(mode)))


@pytest.mark.parametrize("mode", ["full", True])
def test_full_gradient_checkpointing_uses_the_native_method(mode: str | bool) -> None:
    calls: list[dict[str, object]] = []

    class _Transformer:
        def enable_gradient_checkpointing(self, **kwargs: object) -> None:
            calls.append(kwargs)

    bundle = type("_Bundle", (), {"trainable_modules": {"transformer": _Transformer()}})()

    enable_transformer_gradient_checkpointing(bundle, parse_config(_config(mode)))

    assert calls == [{}]


def test_selective_gradient_checkpointing_passes_a_custom_function() -> None:
    calls: list[dict[str, object]] = []

    class _Transformer:
        def enable_gradient_checkpointing(self, **kwargs: object) -> None:
            calls.append(kwargs)

    bundle = type("_Bundle", (), {"trainable_modules": {"transformer": _Transformer()}})()

    enable_transformer_gradient_checkpointing(bundle, parse_config(_config("selective")))

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
        enable_transformer_gradient_checkpointing(bundle, parse_config(_config("selective")))

    assert calls == 1
    assert "falling back to full checkpointing" in caplog.text


def test_checkpointing_rejects_replay_compile_but_accepts_rollout_scope() -> None:
    """The collision is with compiling the REPLAY policy; a rollout-scoped
    compile leaves the checkpointed trainer eager and must pass."""

    def _cfg(compile_block: dict):
        # The check reads the compile matrix off a parsed root, as the runtime
        # apply path hands it one.
        return parse_config(
            OmegaConf.create(
                {
                    "actor": {"gradient_checkpointing": "full"},
                    "model": {"family": "sd3_5", "torch_compile": compile_block},
                },
            )
        )

    with pytest.raises(ValueError, match="gradient_checkpointing"):
        enable_transformer_gradient_checkpointing(None, _cfg({"enable": True}))

    module = SimpleNamespace(enable_gradient_checkpointing=lambda: None)
    bundle = SimpleNamespace(trainable_modules={"transformer": module})
    enable_transformer_gradient_checkpointing(
        bundle,
        _cfg({"enable": True, "scope": "rollout"}),
    )


def test_selective_checkpointing_preserves_errors_inside_supported_method() -> None:
    calls: list[object] = []

    class _Transformer:
        def enable_gradient_checkpointing(self, gradient_checkpointing_func=None) -> None:
            calls.append(gradient_checkpointing_func)
            if gradient_checkpointing_func is not None:
                raise TypeError("checkpoint setup failed inside the model")

    bundle = SimpleNamespace(trainable_modules={"transformer": _Transformer()})
    with pytest.raises(TypeError, match="checkpoint setup failed inside the model"):
        enable_transformer_gradient_checkpointing(bundle, parse_config(_config("selective")))

    assert len(calls) == 1
