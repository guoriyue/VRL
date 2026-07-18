"""Emu3 CFG sample-vs-score contract tests (mirrors the janus_pro contract).

The GRPO old_log_prob invariant: tokens are SAMPLED from the CFG-guided
masked distribution but SCORED under the masked, renormalized conditional
logits. Flipping the score source to ``guided`` (an "align to upstream" edit
— upstream computes no log-prob at all) fails these tests. The Emu3 addition
on top of janus: the structural mask must silence the structural columns at
free positions and pin forced positions to lp == 0.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from tests.models.families.emu3.fixtures import TINY_EOL_IDX, TINY_GEN_VOCAB, TINY_IMAGE_VOCAB
from vrl.models.families.emu3.runner import Emu3ARState, Emu3TokenRunner


class _FixedLogitsModel:
    """Model stub returning caller-fixed generation-vocab logits, so the test
    owns the exact cond/uncond split ``_sample_cfg_image_token`` consumes."""

    def __init__(self, logits: torch.Tensor) -> None:
        self._logits = logits
        self.image_vocab_size = TINY_IMAGE_VOCAB

    def image_gen_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        return self._logits


def _state(*, forced: torch.Tensor, guidance_scale: float = 2.0) -> Emu3ARState:
    return Emu3ARState(
        token_ids=torch.empty(1, forced.shape[0], dtype=torch.long),
        logprobs=torch.empty(1, forced.shape[0]),
        guidance_scale=guidance_scale,
        temperature=1.0,
        total_token_num=int(forced.shape[0]),
        forced_gen_index=forced,
    )


def _fixed_runner(cond: torch.Tensor, uncond: torch.Tensor) -> Emu3TokenRunner:
    # [2B=2, L=1, V_gen]: row 0 = cond, row 1 = uncond (runner chunks dim 0).
    logits = torch.stack([cond, uncond]).unsqueeze(1)
    # attention_backend is unused by _sample_cfg_image_token.
    return Emu3TokenRunner(_FixedLogitsModel(logits), attention_backend=None)


def test_free_position_scores_cond_not_guided_and_masks_structural() -> None:
    # Structural columns get the LARGEST raw logits: without the mask they
    # would dominate sampling; with it they must never be sampled.
    cond = torch.tensor([3.0, 2.0, 1.0, 0.0, -1.0, -2.0, -3.0, -4.0, 9.0, 9.0, 9.0, 9.0])
    uncond = torch.tensor([-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 9.0, 9.0, 9.0, 9.0])
    assert cond.shape == (TINY_GEN_VOCAB,)
    runner = _fixed_runner(cond, uncond)
    state = _state(forced=torch.tensor([-1]))
    hidden = torch.zeros(2, 1, 4)

    neg_inf = float("-inf")
    masked = torch.full((TINY_GEN_VOCAB,), neg_inf)
    masked[:TINY_IMAGE_VOCAB] = cond[:TINY_IMAGE_VOCAB]
    guided = uncond + state.guidance_scale * (cond - uncond)
    guided_masked = torch.full((TINY_GEN_VOCAB,), neg_inf)
    guided_masked[:TINY_IMAGE_VOCAB] = guided[:TINY_IMAGE_VOCAB]

    for seed in range(20):
        torch.manual_seed(seed)
        sampled, lp = runner._sample_cfg_image_token(state, hidden, position=0)
        # Structural columns are illegal at free positions.
        assert int(sampled) < TINY_IMAGE_VOCAB
        cond_lp = F.log_softmax(masked, dim=-1)[sampled]
        guided_lp = F.log_softmax(guided_masked, dim=-1)[sampled]
        # The invariant: lp is the masked CONDITIONAL log-prob...
        assert torch.allclose(lp, cond_lp, atol=1e-6)
        # ...and NOT the guided/behavior log-prob (they genuinely differ).
        assert not torch.allclose(lp, guided_lp, atol=1e-4)


def test_forced_position_returns_forced_token_with_zero_logprob() -> None:
    cond = torch.arange(TINY_GEN_VOCAB, dtype=torch.float32)
    uncond = -cond
    runner = _fixed_runner(cond, uncond)
    state = _state(forced=torch.tensor([TINY_EOL_IDX]))
    hidden = torch.zeros(2, 1, 4)

    torch.manual_seed(0)
    sampled, lp = runner._sample_cfg_image_token(state, hidden, position=0)

    assert int(sampled) == TINY_EOL_IDX
    # Single legal token -> renormalized conditional log-prob is exactly 0,
    # matching what replay recomputes for the same masked position.
    assert torch.allclose(lp, torch.zeros(1), atol=1e-6)
