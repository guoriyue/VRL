"""Contract + equivalence tests for the torch_native AR attention backend.

The skeleton (:class:`TorchNativeDecoderAttentionBackend`) is shared by every
``torch_native`` family: it drives one HF trunk call and splits the returned KV
into one cache per row for ``sequence_states``, re-concatenating the selected
rows the runner hands back. These tests pin two properties that the
bit-identical refactor relies on:

1. The skeleton selects / concatenates / re-splits rows in order, so a runner that
   indexes ``sequence_states`` by ``row_indices`` sees exactly the rows it asked for.
2. A batch of rows stepped as one concatenated forward is identical to stepping
   each row separately — the batch-independence that makes folding the old
   per-branch decode calls into one step bit-identical.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch

from vrl.nn.layers.attention.paged import (
    ARAttentionConfig,
    ARAttentionPrefillInput,
    ARAttentionStepInput,
)
from vrl.nn.modules.ar_attention_backends import build_torch_native_backend
from vrl.nn.modules.torch_attention import TorchNativeDecoderAttentionBackend


def _config() -> ARAttentionConfig:
    return ARAttentionConfig(family="stub")


class _RowIdentityTrunk:
    """Trunk whose KV is a per-row identity tensor, so row order is observable."""

    def __init__(self) -> None:
        self.stepped_past: Any = None

    def __call__(
        self,
        *,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        past_key_values: Any,
        use_cache: bool,
        output_hidden_states: bool,
    ) -> SimpleNamespace:
        del attention_mask, use_cache, output_hidden_states
        if past_key_values is None:
            past = torch.arange(inputs_embeds.shape[0], dtype=torch.float32).reshape(-1, 1)
        else:
            self.stepped_past = past_key_values.clone()
            past = past_key_values + 100.0
        return SimpleNamespace(hidden_states=(inputs_embeds,), past_key_values=past)


def test_skeleton_selects_and_reconcats_rows_in_order() -> None:
    """prefill splits KV into per-row states; step concats the rows it is handed,
    in the given order, and re-splits the result one cache per row."""

    trunk = _RowIdentityTrunk()
    backend = TorchNativeDecoderAttentionBackend(trunk=trunk, config=_config())

    pre = backend.prefill(
        ARAttentionPrefillInput(
            inputs_embeds=torch.randn(3, 2, 4),
            attention_mask=torch.ones(3, 2),
            branch="cond",
        )
    )
    assert len(pre.sequence_states) == 3
    assert pre.last_hidden.shape == (3, 4)

    # Hand back rows [2, 0] (reordered subset) — the backend must rebuild KV in that order.
    out = backend.step(
        ARAttentionStepInput(
            input_embeds=torch.randn(2, 1, 4),
            attention_mask=torch.ones(2, 1),
            sequence_states=(pre.sequence_states[2], pre.sequence_states[0]),
        )
    )
    assert torch.equal(trunk.stepped_past, torch.tensor([[2.0], [0.0]]))
    assert len(out.sequence_states) == 2
    assert torch.equal(out.sequence_states[0], torch.tensor([[102.0]]))
    assert torch.equal(out.sequence_states[1], torch.tensor([[100.0]]))


class _RowIndependentTrunk:
    """Trunk whose KV is the running per-row sum — no cross-batch interaction."""

    def __call__(
        self,
        *,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        past_key_values: Any,
        use_cache: bool,
        output_hidden_states: bool,
    ) -> SimpleNamespace:
        del attention_mask, use_cache, output_hidden_states
        summed = inputs_embeds.sum(dim=1)
        past = summed if past_key_values is None else past_key_values + summed
        return SimpleNamespace(hidden_states=(past.unsqueeze(1),), past_key_values=past)


def test_batched_step_equals_per_row_steps() -> None:
    """Stepping [row0; row1] as one batch == stepping each row alone — the
    batch-independence that makes the concatenated cond/uncond step bit-identical
    to the old per-branch decode calls."""

    torch.manual_seed(0)
    backend = TorchNativeDecoderAttentionBackend(
        trunk=_RowIndependentTrunk(),
        config=_config(),
    )

    pre = backend.prefill(
        ARAttentionPrefillInput(
            inputs_embeds=torch.randn(2, 3, 4),
            attention_mask=torch.ones(2, 3),
            branch="cond",
        )
    )
    step_embed = torch.randn(2, 1, 4)

    batched = backend.step(
        ARAttentionStepInput(
            input_embeds=step_embed,
            attention_mask=torch.ones(2, 1),
            sequence_states=tuple(pre.sequence_states),
        )
    )

    separate = [
        backend.step(
            ARAttentionStepInput(
                input_embeds=step_embed[row : row + 1],
                attention_mask=torch.ones(1, 1),
                sequence_states=(pre.sequence_states[row],),
            )
        )
        for row in range(2)
    ]

    torch.testing.assert_close(
        batched.last_hidden,
        torch.cat([out.last_hidden for out in separate], dim=0),
    )
    for row in range(2):
        torch.testing.assert_close(
            batched.sequence_states[row],
            separate[row].sequence_states[0],
        )


class _CachelessTrunk:
    """Trunk that answers without a KV cache (``use_cache`` silently ignored)."""

    def __call__(
        self,
        *,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        past_key_values: Any,
        use_cache: bool,
        output_hidden_states: bool,
    ) -> SimpleNamespace:
        del attention_mask, past_key_values, use_cache, output_hidden_states
        return SimpleNamespace(hidden_states=(inputs_embeds,), past_key_values=None)


def test_missing_past_key_values_fails_with_the_backend_label() -> None:
    """A cache-less trunk must fail loud, naming the backend that asked for it."""

    backend = TorchNativeDecoderAttentionBackend(trunk=_CachelessTrunk(), config=_config())

    with pytest.raises(
        RuntimeError, match="stub_torch_native_attention trunk forward returned no"
    ):
        backend.prefill(
            ARAttentionPrefillInput(
                inputs_embeds=torch.randn(2, 3, 4),
                attention_mask=torch.ones(2, 3),
                branch="cond",
            )
        )


class _StubHFTrunk:
    """Minimal HF-style trunk that exposes hidden_states and raw past_key_values."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def __call__(
        self,
        *,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        use_cache: bool,
        output_hidden_states: bool,
        past_key_values: Any = None,
    ) -> SimpleNamespace:
        self.calls.append(
            {
                "attention_mask": attention_mask,
                "use_cache": use_cache,
                "output_hidden_states": output_hidden_states,
                "past_key_values": past_key_values,
            }
        )
        hidden = inputs_embeds + 1.0
        if past_key_values is None:
            past = torch.arange(inputs_embeds.shape[0], dtype=torch.float32).reshape(-1, 1)
        else:
            past = past_key_values + 10.0
        return SimpleNamespace(hidden_states=(hidden,), past_key_values=past)


class _StubHFModel:
    config = SimpleNamespace(model_path="stub-model")
    dtype = torch.float32
    device = torch.device("cpu")

    def __init__(self) -> None:
        self.trunk = _StubHFTrunk()

    def _lm_trunk(self) -> _StubHFTrunk:
        return self.trunk


def test_shared_torch_native_builder_uses_hf_cache_forward() -> None:
    """Checks shared torch_native builder uses _lm_trunk without model-specific hooks."""

    torch.manual_seed(0)
    model = _StubHFModel()
    backend = build_torch_native_backend(model, family="hooked_family")
    pre = backend.prefill(
        ARAttentionPrefillInput(
            inputs_embeds=torch.randn(2, 3, 4),
            attention_mask=torch.ones(2, 3),
            branch="cond",
        )
    )

    assert backend.config.family == "hooked_family"
    assert backend.backend_label == "hooked_family_torch_native_attention"
    assert pre.last_hidden.shape == (2, 4)
    assert model.trunk.calls[0]["output_hidden_states"] is True
    # Prefill and step share one forward, so prefill passes the HF default explicitly.
    assert model.trunk.calls[0]["past_key_values"] is None

    out = backend.step(
        ARAttentionStepInput(
            input_embeds=torch.randn(2, 1, 4),
            attention_mask=torch.ones(2, 4),
            sequence_states=tuple(pre.sequence_states),
        )
    )

    assert out.last_hidden.shape == (2, 4)
    assert model.trunk.calls[1]["past_key_values"] is not None
