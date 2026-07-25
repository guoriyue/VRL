"""Tests for explicit AR engine selection."""

from __future__ import annotations

from dataclasses import fields

import pytest

from vrl.generation.bindings.token_autoregressive.executor import (
    ARChunkExecutorBase,
    ARChunkInputs,
)
from vrl.generation.types import GenerationRequest


class _Executor(ARChunkExecutorBase):
    family = "janus_pro"
    task = "ar_t2i"


def test_chunk_inputs_do_not_duplicate_loop_shape() -> None:
    names = {field.name for field in fields(ARChunkInputs)}

    assert "max_new_tokens" not in names
    assert "decode_dtype" not in names


@pytest.mark.parametrize("ar_engine", [None, "native"])
def test_native_ar_engine_is_accepted(ar_engine: str | None) -> None:
    # Both an unset selector and an explicit "native" must resolve to native parity.
    """Checks native AR engine is accepted."""
    assert _Executor(object()).require_native_ar_engine(_request(ar_engine=ar_engine)) == "native"


def test_unknown_ar_engine_is_rejected_via_native_requirement_branch() -> None:
    # An arbitrary unknown selector must hit the generic "must be 'native'" branch,
    # not the vllm-specific one — a loose "ar_engine" match would pass for either.
    """Checks unknown AR engine is rejected via native requirement branch."""
    with pytest.raises(ValueError, match="must be 'native'"):
        _Executor(object()).require_native_ar_engine(_request(ar_engine="hf"))


def test_vllm_full_engine_selector_is_rejected_with_actionable_message() -> None:
    """Checks vLLM full engine selector is rejected with actionable message."""
    with pytest.raises(ValueError, match="not a supported full-engine") as excinfo:
        _Executor(object()).require_native_ar_engine(_request(ar_engine="vllm"))

    # The message must name the offending selector so the caller knows what to drop.
    assert "vllm" in str(excinfo.value)


def _request(ar_engine: str | None = None) -> GenerationRequest:
    sampling = {}
    if ar_engine is not None:
        sampling["ar_engine"] = ar_engine
    return GenerationRequest(
        request_id="req",
        family="janus_pro",
        task="ar_t2i",
        inputs=["prompt"],
        samples_per_prompt=1,
        sampling=sampling,
    )
