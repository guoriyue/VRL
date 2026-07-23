"""Contract tests for the two janus_pro vs upstream-Janus divergences that are
correctness-critical (SPRINT_janus_pro_upstream_reconcile Phase 1). Both are places
where naively "aligning to upstream" would silently break us:

* CFG log-prob is scored from ``cond_logits`` while sampling is from the ``guided``
  distribution — the GRPO old_log_prob invariant. Reverting the source to ``guided``
  flips lp and these tests go red.
* VQ latent channels are resolved from the LIVE quantizer, NOT hardcoded to ``8`` like
  upstream's ``generation_inference.py`` (wrong on any variant whose codebook != 8).

CPU-only, hand-built stubs (see tests/models/steps/token/fixtures.py) — no checkpoint, no GPU.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from tests.models.steps.token.fixtures import StubVQ, build_stub_janus_model
from vrl.models.families.janus_pro.runner import JanusProARModelRunner, JanusProARState


class _FixedLogitsHead(nn.Module):
    """``gen_head`` returning caller-fixed logits, so the test owns the exact
    cond/uncond split ``_sample_cfg_image_token`` consumes (the hidden is ignored)."""

    def __init__(self, logits: torch.Tensor) -> None:
        super().__init__()
        self._logits = logits

    def forward(self, _hidden: torch.Tensor) -> torch.Tensor:
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
        total_token_num=1,
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
    sampled, lp = runner._sample_cfg_image_token(state, hidden, position=0)

    guided = uncond + state.guidance_scale * (cond - uncond)
    cond_lp = F.log_softmax(cond / state.temperature, dim=-1)[sampled]
    guided_lp = F.log_softmax(guided / state.temperature, dim=-1)[sampled]

    # The invariant: lp is the CONDITIONAL log-prob of the sampled token...
    assert torch.allclose(lp, cond_lp, atol=1e-6)
    # ...and NOT the guided/behavior log-prob (the two genuinely differ here, so an
    # "align to upstream -> score guided" edit is caught).
    assert not torch.allclose(lp, guided_lp, atol=1e-4)


def test_paged_cfg_init_rejects_zero_temperature_before_prefill() -> None:
    """Greedy decoding must use an explicit policy mode, not a zero clamp."""
    logits = torch.zeros(2, 1, 4)
    runner = _runner_with_fixed_logits(logits)
    embeds = torch.zeros(1, 1, 8)
    mask = torch.ones(1, 1, dtype=torch.long)

    with pytest.raises(ValueError, match="temperature must be finite and > 0"):
        runner.init_token(
            embeds,
            embeds,
            mask,
            mask,
            temperature=0.0,
            image_token_num=1,
        )


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


def test_decode_image_tokens_enforces_requested_geometry() -> None:
    model = build_stub_janus_model(
        language_model=nn.Identity(),
        hidden_size=8,
        image_vocab_size=16,
        gen_vision_model=StubVQ(vocab_size=16, latent_channels=11),
    )
    token_ids = torch.zeros(2, 4, dtype=torch.long)

    decoded = model.decode_image_tokens(token_ids, image_size=32)

    assert decoded.shape == (2, 3, 32, 32)
    with pytest.raises(ValueError, match="requested 64, expected 32"):
        model.decode_image_tokens(token_ids, image_size=64)


def test_decode_image_tokens_rejects_non_square_token_grid() -> None:
    model = build_stub_janus_model(
        language_model=nn.Identity(),
        hidden_size=8,
        image_vocab_size=16,
        gen_vision_model=StubVQ(vocab_size=16, latent_channels=11),
    )

    with pytest.raises(ValueError, match="square image-token grid"):
        model.decode_image_tokens(torch.zeros(1, 3, dtype=torch.long), image_size=32)


def test_decode_geometry_guard_survives_optimized_python() -> None:
    repo_root = Path(__file__).resolve().parents[4]
    script = """
import torch
import torch.nn as nn
from tests.models.steps.token.fixtures import StubVQ, build_stub_janus_model

model = build_stub_janus_model(
    language_model=nn.Identity(),
    hidden_size=8,
    image_vocab_size=16,
    gen_vision_model=StubVQ(vocab_size=16, latent_channels=11),
)
try:
    model.decode_image_tokens(torch.zeros(1, 3, dtype=torch.long), image_size=32)
except ValueError:
    pass
else:
    raise SystemExit("optimized Python skipped the Janus geometry guard")
"""

    result = subprocess.run(
        [sys.executable, "-O", "-c", script],
        cwd=repo_root,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stderr or result.stdout


def test_janus_checkpoint_type_check_is_a_runtime_error(monkeypatch) -> None:
    from vrl.models.families.janus_pro.model import (
        JanusProConfig,
        _load_janus_from_pretrained,
    )

    class MultiModalityCausalLM:
        pass

    class Processor:
        @classmethod
        def from_pretrained(cls, *_args, **_kwargs):
            return object()

    models_module = ModuleType("janus.models")
    models_module.MultiModalityCausalLM = MultiModalityCausalLM
    models_module.VLChatProcessor = Processor
    janus_module = ModuleType("janus")
    janus_module.models = models_module
    monkeypatch.setitem(sys.modules, "janus", janus_module)
    monkeypatch.setitem(sys.modules, "janus.models", models_module)

    import transformers

    monkeypatch.setattr(
        transformers.AutoModelForCausalLM,
        "from_pretrained",
        lambda *_args, **_kwargs: SimpleNamespace(),
    )

    with pytest.raises(TypeError, match="is not MultiModalityCausalLM"):
        _load_janus_from_pretrained(JanusProConfig())
