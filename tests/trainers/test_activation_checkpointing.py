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
    selective_checkpoint_func,
)


def _config(mode: str | bool):
    return OmegaConf.create(
        {
            "actor": {"gradient_checkpointing": mode},
            "model": {"family": "sd3_5", "torch_compile": {"enable": False}},
        },
    )


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


def test_cpu_checkpoint_preserves_output_input_and_parameter_gradients():
    torch.manual_seed(19)
    direct = torch.nn.Sequential(
        torch.nn.Linear(8, 16), torch.nn.Dropout(0.2), torch.nn.SiLU(), torch.nn.Linear(16, 4)
    ).double()
    checkpointed = copy.deepcopy(direct)
    x = torch.randn(3, 8, dtype=torch.float64, requires_grad=True)
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


def test_root_compile_traces_selective_checkpointing_to_eager_gradients() -> None:
    # A root-compiled transformer forward loops over its blocks through the
    # checkpointing function, so dynamo traces selective_checkpoint_func itself;
    # a context_fn closure created there was an InternalTorchDynamoError.
    class _Blocks(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.blocks = torch.nn.ModuleList(
                torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.SiLU()) for _ in range(2)
            )
            self.checkpointed = False

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            for block in self.blocks:
                x = selective_checkpoint_func(block, x) if self.checkpointed else block(x)
            return x

    torch.manual_seed(11)
    eager = _Blocks()
    compiled_root = copy.deepcopy(eager)
    compiled_root.checkpointed = True
    x = torch.randn(4, 8)

    eager(x).square().sum().backward()
    torch.compile(compiled_root, fullgraph=True)(x).square().sum().backward()

    for left, right in zip(eager.parameters(), compiled_root.parameters(), strict=True):
        torch.testing.assert_close(right.grad, left.grad, rtol=1e-5, atol=1e-6)
