"""Contract + equivalence tests for the torch_native AR attention backend.

The skeleton (:class:`TorchNativeDecoderAttentionBackend`) is shared by every
``torch_native`` family: it splits the family KV into one cache per row for
``sequence_states`` and re-concatenates the selected rows the runner hands back.
These tests pin two properties that the bit-identical refactor relies on:

1. The skeleton selects / concatenates / re-splits rows in order, so a runner that
   indexes ``sequence_states`` by ``row_indices`` sees exactly the rows it asked for.
2. A NextStep-style ``{past_key_values, last_hidden}`` KV stepped as one concatenated
   batch is identical to stepping each row separately — the batch-independence that
   makes folding the old per-branch ``_step_llm`` calls into one step bit-identical.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import torch

from vrl.nn.layers.attention.paged import (
    ARAttentionConfig,
    ARAttentionPrefillInput,
    ARAttentionStepInput,
)
from vrl.nn.modules.ar_attention_backends import build_torch_native_backend
from vrl.nn.modules.torch_attention import TorchNativeDecoderAttentionBackend


def _config() -> ARAttentionConfig:
    return ARAttentionConfig(
        family="stub",
        extra={"backend_label": "stub_torch_native"},
    )


def test_skeleton_selects_and_reconcats_rows_in_order() -> None:
    """prefill splits KV into per-row states; step concats the rows it is handed,
    in the given order, and re-splits the result one cache per row."""

    seen: dict[str, Any] = {}

    def prefill_fn(embeds: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, Any]:
        del mask
        batch = embeds.shape[0]
        # Per-row identity KV so we can assert ordering survives split/concat.
        kv = torch.arange(batch, dtype=torch.float32).reshape(batch, 1)
        return embeds[:, -1, :], kv

    def step_fn(embeds: torch.Tensor, mask: torch.Tensor, kv: Any) -> tuple[torch.Tensor, Any]:
        del mask
        seen["kv"] = kv.clone()
        return embeds[:, 0, :], kv + 100.0

    backend = TorchNativeDecoderAttentionBackend(
        config=_config(), prefill_fn=prefill_fn, step_fn=step_fn
    )

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
    assert torch.equal(seen["kv"], torch.tensor([[2.0], [0.0]]))
    assert len(out.sequence_states) == 2
    assert torch.equal(out.sequence_states[0], torch.tensor([[102.0]]))
    assert torch.equal(out.sequence_states[1], torch.tensor([[100.0]]))


class _StubKVModel:
    """Minimal NextStep-style model: dict KV with a per-row ``_step_llm``."""

    def _init_kv(self, embeds: torch.Tensor, mask: torch.Tensor | None) -> dict[str, Any]:
        del mask
        return {"past_key_values": embeds.sum(dim=1), "last_hidden": embeds[:, -1]}

    @staticmethod
    def _last_hidden(kv: dict[str, Any]) -> torch.Tensor:
        return kv["last_hidden"]

    def _step_llm(self, kv: dict[str, Any], new_embed: torch.Tensor) -> tuple[dict[str, Any], torch.Tensor]:
        past = kv["past_key_values"] + new_embed  # per-row, no cross-batch interaction
        return {"past_key_values": past, "last_hidden": past}, past


def _build_stub_backend(model: _StubKVModel) -> TorchNativeDecoderAttentionBackend:
    def prefill_fn(embeds: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, Any]:
        kv = model._init_kv(embeds, mask)
        return model._last_hidden(kv), kv

    def step_fn(embeds: torch.Tensor, mask: torch.Tensor, kv: Any) -> tuple[torch.Tensor, Any]:
        del mask
        kv_next, last_hidden = model._step_llm(kv, embeds[:, 0, :])
        return last_hidden, kv_next

    return TorchNativeDecoderAttentionBackend(
        config=_config(), prefill_fn=prefill_fn, step_fn=step_fn
    )


def test_dict_kv_batched_step_equals_per_row_steps() -> None:
    """Stepping [row0; row1] as one batch == stepping each row alone — the
    batch-independence that makes the concatenated cond/uncond step bit-identical
    to the old per-branch ``_step_llm`` calls."""

    torch.manual_seed(0)
    model = _StubKVModel()
    backend = _build_stub_backend(model)

    embeds = torch.randn(2, 3, 4)
    pre = backend.prefill(
        ARAttentionPrefillInput(
            inputs_embeds=embeds, attention_mask=torch.ones(2, 3), branch="cond"
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
            batched.sequence_states[row]["past_key_values"],
            separate[row].sequence_states[0]["past_key_values"],
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
    assert pre.last_hidden.shape == (2, 4)
    assert model.trunk.calls[0]["output_hidden_states"] is True

    out = backend.step(
        ARAttentionStepInput(
            input_embeds=torch.randn(2, 1, 4),
            attention_mask=torch.ones(2, 4),
            sequence_states=tuple(pre.sequence_states),
        )
    )

    assert out.last_hidden.shape == (2, 4)
    assert model.trunk.calls[1]["past_key_values"] is not None
