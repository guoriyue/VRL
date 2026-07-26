"""Tests for shared AR attention backend selection."""

from __future__ import annotations

from typing import Any

from vrl.nn.modules import ar_attention_backends as backends


class _StubBackend:
    def __init__(
        self,
        model: Any,
        family: str,
        block_size: Any = None,
        cache_dtype: Any = None,
    ) -> None:
        self.model = model
        self.family = family
        self.block_size = block_size
        self.cache_dtype = cache_dtype


def test_resolve_vllm_forwards_family_and_backend_kwargs(monkeypatch) -> None:
    """Checks vLLM selection forwards family and vLLM-specific kwargs."""

    def build_backend(
        model: Any,
        *,
        family: str,
        block_size: int = 16,
        cache_dtype: str = "auto",
    ) -> _StubBackend:
        return _StubBackend(model, family, block_size, cache_dtype)

    monkeypatch.setattr(backends, "build_vllm_attention_backend", build_backend)

    resolved = backends.resolve_attention_backend(
        "janus_pro",
        "vllm_paged",
        "model",
        block_size=8,
        cache_dtype="fp8",
    )

    assert (resolved.model, resolved.family, resolved.block_size, resolved.cache_dtype) == (
        "model",
        "janus_pro",
        8,
        "fp8",
    )


def test_resolve_torch_native_does_not_forward_vllm_only_kwargs(monkeypatch) -> None:
    """Checks torch_native selection keeps vLLM-only configuration out of its builder."""

    def build_backend(model: Any, *, family: str) -> _StubBackend:
        return _StubBackend(model, family)

    monkeypatch.setattr(backends, "build_torch_native_backend", build_backend)

    resolved = backends.resolve_attention_backend(
        "nextstep_1",
        "torch_native",
        "model",
        block_size=8,
        cache_dtype="fp8",
    )

    assert (resolved.model, resolved.family, resolved.block_size, resolved.cache_dtype) == (
        "model",
        "nextstep_1",
        None,
        None,
    )


def test_unknown_backend_error_lists_available_names() -> None:
    """Checks unknown backend errors point to the supported backend names."""

    try:
        backends.resolve_attention_backend("janus_pro", "flashinfer", "model")
    except ValueError as exc:
        assert "unknown attention backend" in str(exc)
        assert "vllm_paged" in str(exc)
        assert "torch_native" in str(exc)
    else:
        raise AssertionError("expected unknown backend to raise")


def test_attention_backend_name_reads_explicit_name_with_paged_default() -> None:
    """Checks attention backend name reads explicit name with paged default."""

    assert backends.attention_backend_name({"attention_backend": "torch_native"}) == "torch_native"
    assert backends.attention_backend_name({"attention_backend": "vllm_paged"}) == "vllm_paged"
    assert backends.attention_backend_name({}) == "vllm_paged"
