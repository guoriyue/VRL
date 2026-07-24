"""Shared structural stubs for the Janus MMGPT surface used by AR tests.

Unlike :mod:`tests.models.steps.denoise.fixtures` — whose ``build_tiny_*`` helpers
construct *genuine* diffusers transformers straight from config — these are
hand-written stubs, not tiny-real models. AR has no config-init path that a
clean CI can build, for reasons that differ **per path**:

* NextStep — ``NextStep.from_pretrained`` / ``NextStepPipeline`` come from the
  ``nextstep_model`` / ``gen_pipeline`` packages, which are not importable
  anywhere in this repo (``ModuleNotFoundError``): an undeclared dependency with
  no in-repo config-init, so there is nothing to build a tiny-real model from.
* Janus replay path — ``MultiModalityCausalLM`` *does* have a config-init route
  (it is a standard HF ``PreTrainedModel``; ``JanusProReplayCore`` already
  constructs a real ``LlamaForCausalLM(config.language_config)`` trunk), but the
  ``janus`` package is editable-only on this machine and is **not declared in
  ``pyproject.toml``**, so a clean-install CI cannot import it — a real Llama
  trunk would break clean CI.
* Janus decode / paged path — the decode and paged-attention tests use
  behavior-injection recording doubles (e.g. ``hidden[:, -1, 0] = 10 + call_index``,
  scripted KV cache, ``calls == []`` to prove the paged backend took over). A real
  trunk would *erase* exactly the deterministic instrumentation those assertions
  pin, and the full ``MultiModalityCausalLM`` surface needs a vision tower that is
  not config-init-trivial.

So this module canonicalizes only the **structural scaffold** that the four
Janus tests were each re-declaring (and drifting on — ``test_r1_model`` omitted
``gen_aligner`` entirely): the ``_StubMMGPT`` container with the verified
attribute surface (``language_model`` / ``gen_head`` / ``gen_vision_model`` /
``gen_aligner`` / ``gen_embed`` / ``prepare_gen_img_embeds``) and the ``_StubVQ``
frozen-decoder boundary. The production contract is
``JanusProModel.__init__`` (``model.py:194``) asserting
``gen_head`` / ``gen_vision_model`` / ``language_model`` exist, and
``prepare_gen_img_embeds`` being ``gen_aligner(gen_embed(ids))``
(``model.py:1102``) — which is why ``gen_aligner`` must be present and the
embed must route through it.

Each test's ``language_model.forward`` behavior is load-bearing, deterministic
instrumentation that differs per file (scripted KV values, ``calls == []``,
identity trunk, ``.logits`` head), so the language model is **injected by the
caller**, never canonicalized here. ``record_forward_calls`` is reused from
the diffusion fixtures (it only registers a forward pre-hook; it is
family-agnostic).
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from tests.models.steps.denoise.fixtures import record_forward_calls
from vrl.models.families.janus_pro.config import JanusProConfig
from vrl.models.families.janus_pro.model import (
    JANUS_IMAGE_PATCH_SIZE,
    JanusProModel,
)

__all__ = [
    "RecordingHead",
    "StubVQ",
    "build_stub_janus_mmgpt",
    "build_stub_janus_model",
    "record_forward_calls",
]


class StubVQ(nn.Module):
    """Frozen VQ-decoder boundary: wrappers only read the decoded shape.

    ``decode_code`` returns zeros at the upsampled
    ``H*JANUS_IMAGE_PATCH_SIZE x W*JANUS_IMAGE_PATCH_SIZE`` geometry. When
    ``latent_channels`` is given, also carries ``quantize.embedding`` so
    ``_resolve_vq_latent_channels`` can read the per-token latent width (the R1
    path needs this; the decode/replay paths never touch it).
    """

    def __init__(
        self,
        *,
        vocab_size: int | None = None,
        latent_channels: int | None = None,
    ) -> None:
        super().__init__()
        if vocab_size is not None and latent_channels is not None:
            self.quantize = nn.Module()
            self.quantize.embedding = nn.Embedding(vocab_size, latent_channels)

    def decode_code(self, ids: torch.Tensor, shape: list[int]) -> torch.Tensor:
        del ids
        batch, _, height, width = shape
        return torch.zeros(
            batch,
            3,
            height * JANUS_IMAGE_PATCH_SIZE,
            width * JANUS_IMAGE_PATCH_SIZE,
        )


class RecordingHead(nn.Module):
    """``gen_head`` that records every hidden tensor it projects.

    Routes ``hidden[..., 0]`` into logit channel 0 (all other logits ``-20``), so
    decode tests can read back the scripted ``language_model`` hidden values from
    ``head.inputs``. Used by the decode/paged tests; the replay/R1 tests use a
    plain ``nn.Linear`` head instead.
    """

    def __init__(self, *, image_vocab_size: int) -> None:
        super().__init__()
        self._image_vocab_size = image_vocab_size
        self.inputs: list[torch.Tensor] = []

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        self.inputs.append(hidden.detach().clone())
        logits = torch.full(
            (*hidden.shape[:-1], self._image_vocab_size),
            -20.0,
            device=hidden.device,
        )
        logits[..., 0] = hidden[..., 0]
        return logits


class _ClampingEmbed(nn.Module):
    """``gen_embed`` that clamps/mods ids into range before embedding.

    The R1 generate path feeds real sampled image-token ids that can exceed the
    stub embedding's vocab, so ids are clamped and wrapped (matching the original
    ``test_r1_model`` stub); the decode/replay paths sample ids already in range
    and use a plain ``nn.Embedding`` instead.
    """

    def __init__(self, *, vocab_size: int, hidden_size: int) -> None:
        super().__init__()
        self._vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size, hidden_size)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return self.embed(ids.clamp_min(0) % self._vocab_size)


class _StubMMGPT(nn.Module):
    """Canonical container mirroring the Janus ``MultiModalityCausalLM`` surface.

    Holds the structural scaffold every Janus test shares; the caller injects the
    ``language_model`` (whose forward behavior is the per-test instrumentation) and
    chooses the ``gen_head`` (recording head vs plain linear). ``gen_aligner`` is an
    identity, so ``prepare_gen_img_embeds`` reproduces the production
    ``gen_aligner(gen_embed(ids))`` (``model.py:1102``) while staying behaviorally
    equal to a bare ``gen_embed(ids)``.
    """

    def __init__(
        self,
        *,
        language_model: nn.Module,
        gen_head: nn.Module,
        gen_vision_model: nn.Module,
        gen_embed: nn.Module,
    ) -> None:
        super().__init__()
        self.language_model = language_model
        self.gen_head = gen_head
        self.gen_vision_model = gen_vision_model
        self.gen_aligner = nn.Identity()
        self.gen_embed = gen_embed

    def prepare_gen_img_embeds(self, ids: torch.Tensor) -> torch.Tensor:
        return self.gen_aligner(self.gen_embed(ids))


def build_stub_janus_mmgpt(
    *,
    language_model: nn.Module,
    hidden_size: int,
    image_vocab_size: int,
    gen_head: nn.Module | None = None,
    gen_vision_model: nn.Module | None = None,
    embed_ids: str = "raw",
) -> _StubMMGPT:
    """Build the canonical ``_StubMMGPT`` container around an injected trunk.

    ``language_model`` is supplied by the caller — its forward behavior is the
    per-test instrumentation that must not be canonicalized. ``gen_head`` defaults
    to a plain ``nn.Linear(hidden_size, image_vocab_size)``; pass a
    :class:`RecordingHead` when the test reads back scripted hidden values.
    ``gen_vision_model`` defaults to a bare :class:`StubVQ`; pass one carrying
    ``quantize.embedding`` for paths that resolve latent channels.

    ``embed_ids`` controls how ``gen_embed`` is fed: ``"raw"`` passes ids straight
    through (decode/replay paths sample ids already in range), while ``"clamp"``
    clamps and mods ids into range (the R1 path feeds real sampled ids that can
    exceed the embedding range).
    """

    if gen_head is None:
        gen_head = nn.Linear(hidden_size, image_vocab_size)
    if gen_vision_model is None:
        gen_vision_model = StubVQ()

    if embed_ids == "clamp":
        gen_embed: nn.Module = _ClampingEmbed(
            vocab_size=image_vocab_size,
            hidden_size=hidden_size,
        )
    else:
        gen_embed = nn.Embedding(image_vocab_size, hidden_size)

    return _StubMMGPT(
        language_model=language_model,
        gen_head=gen_head,
        gen_vision_model=gen_vision_model,
        gen_embed=gen_embed,
    )


def build_stub_janus_model(
    *,
    language_model: nn.Module,
    hidden_size: int,
    image_vocab_size: int,
    gen_head: nn.Module | None = None,
    gen_vision_model: nn.Module | None = None,
    embed_ids: str = "raw",
    processor: Any = None,
    unfreeze_gen_head: bool = False,
    **config_kwargs: Any,
) -> JanusProModel:
    """Wrap :func:`build_stub_janus_mmgpt` into a real ``JanusProModel``.

    Replaces each test's local ``_model()`` / ``_build_stub_model()`` helper. Extra
    ``config_kwargs`` (e.g. ``image_token_num``, ``device``) flow
    straight into ``JanusProConfig``; ``use_lora`` defaults off so the stub trunk is
    not PEFT-wrapped.
    """

    config_kwargs.setdefault("use_lora", False)
    mmgpt = build_stub_janus_mmgpt(
        language_model=language_model,
        hidden_size=hidden_size,
        image_vocab_size=image_vocab_size,
        gen_head=gen_head,
        gen_vision_model=gen_vision_model,
        embed_ids=embed_ids,
    )
    model = JanusProModel(
        config=JanusProConfig(**config_kwargs),
        mmgpt=mmgpt,
        processor=processor if processor is not None else object(),
    )
    if unfreeze_gen_head:
        for p in model.mmgpt.gen_head.parameters():
            p.requires_grad_(True)
    return model
