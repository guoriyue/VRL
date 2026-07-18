"""Core Janus-Pro replay ownership tests."""

from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn

from tests.models.ar.fixtures import build_stub_janus_model
from vrl.families.registry import get_model_family_entry
from vrl.generation import GenerationRequest, GenerationSampleRow
from vrl.models.ar.janus_pro.model import (
    JANUS_IMAGE_VOCAB_SIZE,
    JanusProModel,
)
from vrl.models.interfaces import ReplayResult
from vrl.rollouts.batch import RolloutBatch
from vrl.rollouts.collector import build_rollout_collector
from vrl.rollouts.collector.config import RolloutCollectorConfig
from vrl.trajectory import build_ar_discrete_trajectory, build_training_view

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


def _build_stub_model(*, unfreeze_gen_head: bool = False) -> JanusProModel:
    return build_stub_janus_model(
        language_model=_StubLM(),
        hidden_size=HIDDEN,
        image_vocab_size=JANUS_IMAGE_VOCAB_SIZE,
        unfreeze_gen_head=unfreeze_gen_head,
    )


def _request() -> GenerationRequest:
    return GenerationRequest(
        request_id="req",
        family="janus_pro",
        task="ar_t2i",
        inputs=["draw text"],
        samples_per_prompt=2,
    )


def _sample_rows() -> list[GenerationSampleRow]:
    request = _request()
    return [
        GenerationSampleRow(
            prompt_index=0,
            sample_index=index,
            prompt=request.prompts[0],
            prompt_id="p0",
            group_id="g0",
            sample_id=f"s{index}",
            trajectory_id=f"t{index}",
            seed=None,
        )
        for index in range(2)
    ]


def _discrete_batch() -> RolloutBatch:
    token_ids = torch.tensor([[1, 2], [2, 3]])
    trajectory = build_ar_discrete_trajectory(
        request=_request(),
        sample_rows=_sample_rows(),
        token_ids=token_ids,
        token_log_probs=torch.zeros_like(token_ids, dtype=torch.float32),
        token_mask=torch.ones_like(token_ids, dtype=torch.float32),
        prompt_input_ids=torch.ones(2, 3, dtype=torch.long),
        prompt_attention_mask=torch.ones(2, 3, dtype=torch.long),
        uncond_input_ids=torch.zeros(2, 3, dtype=torch.long),
        uncond_attention_mask=torch.ones(2, 3, dtype=torch.long),
        context={"model_family": "janus_pro"},
    )
    return RolloutBatch(
        observations=torch.ones(2, 1, 3, dtype=torch.long),
        actions=token_ids,
        rewards=torch.zeros(2),
        dones=torch.ones(2, dtype=torch.bool),
        group_ids=torch.tensor([0, 0]),
        trajectory=trajectory,
        training_view=build_training_view(trajectory),
    )


def test_janus_collector_has_no_forward_step() -> None:
    """Collectors expose collect(); train-time replay lives on the model."""
    collector = build_rollout_collector(
        get_model_family_entry("janus_pro"),
        reward_fn=None,
        config=RolloutCollectorConfig(
            values={
                "n_samples_per_prompt": 1,
                "guidance_scale": 5.0,
                "temperature": 1.0,
                "image_token_num": 4,
                "image_size": 64,
                "max_text_length": 256,
            },
        ),
    )
    assert not hasattr(collector, "forward_step")


def test_janus_model_exposes_trainer_replay_methods() -> None:
    """Checks Janus model exposes trainer replay methods."""
    model = _build_stub_model()
    assert callable(model.replay_forward)
    assert callable(model.disable_adapter)
    assert callable(model.load_trainable_state)


def test_janus_model_replay_forward_returns_typed_replay_result() -> None:
    """Checks Janus model replay forward returns typed replay result."""
    model = _build_stub_model()
    batch = _discrete_batch()

    result = model.replay_forward(batch)

    assert isinstance(result, ReplayResult)
    segment = result.segments["image_tokens"]
    assert segment.segment == "image_tokens"
    assert set(segment.values) == {"logits", "image_token_ids"}
    assert segment.values["logits"].shape == (2, 2, JANUS_IMAGE_VOCAB_SIZE)
    assert torch.equal(segment.values["image_token_ids"], batch.actions)


def test_janus_disable_adapter_without_lora_is_noop() -> None:
    """Checks Janus disable adapter without LoRA is no-op."""
    model = _build_stub_model()

    with model.disable_adapter():
        pass
