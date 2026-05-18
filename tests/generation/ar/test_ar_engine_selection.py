"""Tests for explicit AR engine selection."""

from __future__ import annotations

import pytest

from vrl.generation.ar.executor import ARPipelineExecutorBase
from vrl.generation.types import GenerationRequest


class _Executor(ARPipelineExecutorBase):
    family = "janus_pro"
    task = "ar_t2i"


def test_native_ar_engine_is_default() -> None:
    assert _Executor().require_native_ar_engine(_request()) == "native"


def test_unknown_ar_engine_is_rejected() -> None:
    request = _request(ar_engine="hf")

    with pytest.raises(ValueError, match="ar_engine"):
        _Executor().require_native_ar_engine(request)


def test_vllm_full_engine_selector_is_rejected_without_import_gate() -> None:
    with pytest.raises(ValueError, match="not a supported full-engine"):
        _Executor().require_native_ar_engine(_request(ar_engine="vllm"))


def _request(ar_engine: str | None = None) -> GenerationRequest:
    sampling = {}
    if ar_engine is not None:
        sampling["ar_engine"] = ar_engine
    return GenerationRequest(
        request_id="req",
        family="janus_pro",
        task="ar_t2i",
        prompts=["prompt"],
        samples_per_prompt=1,
        sampling=sampling,
    )
