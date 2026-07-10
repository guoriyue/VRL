"""Contract tests for the two janus_pro vs upstream-Janus divergences that are
correctness-critical (SPRINT_janus_pro_upstream_reconcile Phase 1). Both are places
where naively "aligning to upstream" would silently break us:

* CFG log-prob is scored from ``cond_logits`` while sampling is from the ``guided``
  distribution — the GRPO old_log_prob invariant. Reverting the source to ``guided``
  flips lp and these tests go red.
* VQ latent channels are resolved from the LIVE quantizer, NOT hardcoded to ``8`` like
  upstream's ``generation_inference.py`` (wrong on any variant whose codebook != 8).

CPU-only, hand-built stubs (see tests/models/ar/fixtures.py) — no checkpoint, no GPU.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from tests.models.ar.fixtures import StubVQ, build_stub_janus_model
from vrl.models.ar.janus_pro.runner import JanusProARModelRunner, JanusProARState


class _FixedLogitsHead(nn.Module):
    """``gen_head`` returning caller-fixed logits, so the test owns the exact
    cond/uncond split ``_sample_cfg_image_token`` consumes (the hidden is ignored)."""

    def __init__(self, logits: torch.Tensor) -> None:
        super().__init__()
        self._logits = logits

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self._logits


def _runner_with_fixed_logits(logits: torch.Tensor) -> JanusProARModelRunner:
    model = build_stub_janus_model(
        language_model=nn.Identity(),
        hidden_size=8,
        image_vocab_size=int(logits.shape[-1]),
        gen_head=_FixedLogitsHead(logits),
    )
    # attention_backend is unused by _sample_cfg_image_token (it only reads
    # mmgpt.gen_head + the state's guidance_scale/temperature).
    return JanusProARModelRunner(model, attention_backend=None)


def _state(*, guidance_scale: float, temperature: float) -> JanusProARState:
    return JanusProARState(
        token_ids=torch.empty(1, 1, dtype=torch.long),
        logprobs=torch.empty(1, 1),
        guidance_scale=guidance_scale,
        temperature=temperature,
        image_token_num=1,
    )


def test_cfg_logprob_is_scored_from_cond_not_guided() -> None:
    # cond and guided are deliberately different distributions at every token, so
    # switching the log-prob source cond -> guided changes lp and fails here.
    cond = torch.tensor([3.0, 2.0, 1.0, 0.0])
    uncond = torch.tensor([0.0, 1.0, 2.0, 3.0])
    # [2B=2, L=1, vocab=4]: row 0 = cond, row 1 = uncond (runner does chunk(2, dim=0)).
    logits = torch.stack([cond, uncond]).unsqueeze(1)
    runner = _runner_with_fixed_logits(logits)
    state = _state(guidance_scale=2.0, temperature=1.0)
    hidden = torch.zeros(2, 1, 8)

    torch.manual_seed(0)
    sampled, lp = runner._sample_cfg_image_token(state, hidden)

    guided = uncond + state.guidance_scale * (cond - uncond)
    cond_lp = F.log_softmax(cond / state.temperature, dim=-1)[sampled]
    guided_lp = F.log_softmax(guided / state.temperature, dim=-1)[sampled]

    # The invariant: lp is the CONDITIONAL log-prob of the sampled token...
    assert torch.allclose(lp, cond_lp, atol=1e-6)
    # ...and NOT the guided/behavior log-prob (the two genuinely differ here, so an
    # "align to upstream -> score guided" edit is caught).
    assert not torch.allclose(lp, guided_lp, atol=1e-4)


def test_vq_latent_channels_resolved_dynamically_not_hardcoded_8() -> None:
    # Upstream hardcodes shape[1]=8; a variant whose codebook width != 8 must still
    # resolve correctly. 11 is chosen so a reverted `return 8` fails this assertion.
    model = build_stub_janus_model(
        language_model=nn.Identity(),
        hidden_size=8,
        image_vocab_size=16,
        gen_vision_model=StubVQ(vocab_size=16, latent_channels=11),
    )
    assert model._resolve_vq_latent_channels() == 11


def test_vq_latent_channels_override_takes_precedence() -> None:
    model = build_stub_janus_model(
        language_model=nn.Identity(),
        hidden_size=8,
        image_vocab_size=16,
        gen_vision_model=StubVQ(vocab_size=16, latent_channels=11),
        vq_latent_channels=5,
    )
    assert model._resolve_vq_latent_channels() == 5


def test_vq_latent_channels_raises_without_quantizer_or_override() -> None:
    # No override + no live quantizer embedding -> fail loudly instead of guessing 8.
    model = build_stub_janus_model(
        language_model=nn.Identity(),
        hidden_size=8,
        image_vocab_size=16,
        gen_vision_model=StubVQ(),
    )
    with pytest.raises(RuntimeError, match="Could not resolve"):
        model._resolve_vq_latent_channels()
