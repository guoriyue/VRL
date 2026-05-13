"""Core Janus-Pro replay ownership tests."""

from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn

from vrl.models.families.janus_pro.policy import (
    JANUS_IMAGE_VOCAB_SIZE,
    JanusProConfig,
    JanusProPolicy,
)
from vrl.models.interfaces.ar_policy import AutoregressivePolicy
from vrl.rollouts.collector import (
    JanusProCollectorConfig,
    build_rollout_collector,
)

HIDDEN = 32
TEXT_VOCAB = 64


# ---------------------------------------------------------------------------
# Stubs — mirror tests/models/test_janus_wrapper.py + tests/rollouts/...
# ---------------------------------------------------------------------------


class _StubLM(nn.Module):
    """Identity trunk: last_hidden_state == inputs_embeds."""

    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Embedding(TEXT_VOCAB, HIDDEN)

    @property
    def model(self) -> _StubLM:
        # Property — not attribute — so ``train()`` does not infinite-recurse.
        return self

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed

    def forward(
        self,
        inputs_embeds: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        use_cache: bool = False,
        past_key_values: object = None,
        output_hidden_states: bool = False,
    ) -> SimpleNamespace:
        return SimpleNamespace(
            last_hidden_state=inputs_embeds,
            past_key_values=past_key_values,
        )


class _StubVQ(nn.Module):
    def decode_code(self, ids: torch.Tensor, shape: list[int]) -> torch.Tensor:
        B, _, h, w = shape
        return torch.zeros(B, 3, h * 16, w * 16)


class _StubMMGPT(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.language_model = _StubLM()
        self.gen_vision_model = _StubVQ()
        self.gen_head = nn.Linear(HIDDEN, JANUS_IMAGE_VOCAB_SIZE)
        self.gen_aligner = nn.Identity()
        self.gen_embed = nn.Embedding(JANUS_IMAGE_VOCAB_SIZE, HIDDEN)

    def prepare_gen_img_embeds(self, ids: torch.Tensor) -> torch.Tensor:
        return self.gen_embed(ids)


def _build_stub_model(*, unfreeze_gen_head: bool = False) -> JanusProPolicy:
    cfg = JanusProConfig(use_lora=False)
    model = JanusProPolicy(config=cfg, mmgpt=_StubMMGPT(), processor=object())
    if unfreeze_gen_head:
        for p in model.mmgpt.gen_head.parameters():
            p.requires_grad_(True)
    return model


def test_janus_collector_has_no_forward_step() -> None:
    """Sprint Task 7: the deprecated ``forward_step`` shim is gone.

    Train-time replay ownership lives on ``model.replay_forward`` and the
    evaluator calls the model directly. Collectors expose only ``collect()``.
    """
    collector = build_rollout_collector(
        "janus_pro",
        model=_build_stub_model(),
        reward_fn=None,
        config=JanusProCollectorConfig(image_token_num=4, image_size=64),
    )
    assert not hasattr(collector, "forward_step")


def test_janus_policy_inherits_ar_protocol() -> None:
    model = _build_stub_model()
    assert AutoregressivePolicy in type(model).__mro__
    assert isinstance(model, AutoregressivePolicy)
