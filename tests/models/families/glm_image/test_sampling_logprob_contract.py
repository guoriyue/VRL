"""GLM-Image sample-vs-score contract tests.

GLM-Image has NO AR-side CFG (the reference generation is plain
``do_sample=True`` with temperature/top_p from generation_config), so the
contract reduces to: tokens are SAMPLED from the top_p-filtered
temperature-scaled distribution but SCORED under the plain (unfiltered)
temperature-scaled conditional over the restricted codebook — nucleus
filtering is only the behavior distribution, the conditional is the policy
GRPO optimizes. The vocab restriction is a codebook-prefix slice of the
16512-wide lm_head, so image_start/image_end can never be sampled mid-raster.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from tests.models.families.glm_image.fixtures import TINY_CODEBOOK, TINY_VISION_VOCAB
from vrl.models.families.glm_image.runner import GlmImageARState, GlmImageTokenRunner


class _FixedLogitsModel:
    """Model stub returning caller-fixed VISION-VOCAB logits so the test owns
    the exact distribution ``_sample_image_token`` consumes (including the
    illegal image_start/image_end columns that the slice must remove)."""

    def __init__(self, vision_logits: torch.Tensor) -> None:
        self._vision_logits = vision_logits
        self.image_vocab_size = TINY_CODEBOOK

    def image_gen_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        del hidden
        return self._vision_logits[..., : self.image_vocab_size]


def _state(*, temperature: float = 0.9, top_p: float = 0.75) -> GlmImageARState:
    return GlmImageARState(
        token_ids=torch.empty(1, 4, dtype=torch.long),
        logprobs=torch.empty(1, 4),
        temperature=temperature,
        top_p=top_p,
        base_position_schedule=torch.zeros(3, 5, dtype=torch.long),
        prompt_valid_lens=torch.tensor([3]),
    )


def test_sampled_tokens_stay_inside_codebook_despite_hot_special_columns() -> None:
    # The image_start/image_end/reserved columns get the LARGEST raw logits:
    # without the codebook slice they would dominate sampling.
    vision = torch.full((1, 1, TINY_VISION_VOCAB), -2.0)
    vision[..., :TINY_CODEBOOK] = torch.linspace(-1.0, 1.0, TINY_CODEBOOK)
    vision[..., TINY_CODEBOOK:] = 9.0
    runner = GlmImageTokenRunner(_FixedLogitsModel(vision))
    state = _state()

    for seed in range(20):
        torch.manual_seed(seed)
        sampled, lp = runner._sample_image_token(state, torch.zeros(1, 1, 4))
        assert 0 <= int(sampled) < TINY_CODEBOOK
        assert lp.item() < 0.0


def test_logprob_scored_under_unfiltered_conditional_not_topp_behavior() -> None:
    torch.manual_seed(0)
    vision = torch.randn(1, 1, TINY_VISION_VOCAB)
    runner = GlmImageTokenRunner(_FixedLogitsModel(vision))
    temperature, top_p = 0.9, 0.5
    state = _state(temperature=temperature, top_p=top_p)

    codebook = vision[0, 0, :TINY_CODEBOOK].float()
    plain_lp = F.log_softmax(codebook / temperature, dim=-1)
    # Behavior distribution: top_p filtering ON TOP of temperature scaling.
    scaled = codebook / temperature
    sorted_logits, sorted_idx = torch.sort(scaled, descending=True)
    cumprobs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    remove = cumprobs > top_p
    remove[1:] = remove[:-1].clone()
    remove[0] = False
    kept = torch.ones(TINY_CODEBOOK, dtype=torch.bool)
    kept[sorted_idx[remove]] = False
    behavior_lp = F.log_softmax(
        scaled.masked_fill(~kept, float("-inf")),
        dim=-1,
    )

    saw_divergent_score = False
    for seed in range(50):
        torch.manual_seed(seed)
        sampled, lp = runner._sample_image_token(state, torch.zeros(1, 1, 4))
        token = int(sampled)
        # Behavior: only nucleus tokens are ever sampled.
        assert kept[token], "sampled a token outside the top_p nucleus"
        # Policy: lp is the UNFILTERED conditional log-prob...
        assert torch.allclose(lp, plain_lp[token], atol=1e-6)
        # ...which genuinely differs from the renormalized behavior log-prob
        # whenever filtering removed mass.
        if not torch.allclose(lp, behavior_lp[token], atol=1e-4):
            saw_divergent_score = True
    assert saw_divergent_score


def test_top_p_one_disables_filtering() -> None:
    torch.manual_seed(1)
    vision = torch.randn(1, 1, TINY_VISION_VOCAB)
    runner = GlmImageTokenRunner(_FixedLogitsModel(vision))
    state = _state(temperature=1.0, top_p=1.0)

    plain_lp = F.log_softmax(vision[0, 0, :TINY_CODEBOOK].float(), dim=-1)
    seen: set[int] = set()
    for seed in range(200):
        torch.manual_seed(seed)
        sampled, lp = runner._sample_image_token(state, torch.zeros(1, 1, 4))
        seen.add(int(sampled))
        assert torch.allclose(lp, plain_lp[int(sampled)], atol=1e-6)
    # With no filtering, low-probability codebook tokens remain reachable.
    assert len(seen) > TINY_CODEBOOK // 2
