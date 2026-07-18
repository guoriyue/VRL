"""Tiny-config fixtures for the LlamaGen AR tests.

Unlike the Janus fixtures (hand-written structural stubs, see
``tests/models/steps/token/fixtures.py`` for why), the LlamaGen GPT is *vendored*, so a
genuine tiny t2i Transformer can be built straight from ``ModelArgs`` on a
clean CI — the trunk under test is the real vendored module, not a stub. Only
the checkpoint-external pieces are stubbed: the frozen T5 encoder/tokenizer
(deterministic embedding table) and the VQ decoder boundary (wrappers only
read the decoded shape).
"""

from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn

from vrl.models.families.llamagen.model import LlamaGenConfig, LlamaGenModel
from vrl.models.families.llamagen.vendor.gpt import ModelArgs, Transformer

# Tiny t2i geometry: 4x4 token grid, 4-token caption prefix, 8-dim head.
TINY_DIM = 32
TINY_N_LAYER = 2
TINY_N_HEAD = 4
TINY_BLOCK_SIZE = 16  # 4x4 grid
TINY_CLS_TOKEN_NUM = 4
TINY_CAPTION_DIM = 12
TINY_VOCAB_SIZE = 24
TINY_DOWNSAMPLE = 16
TINY_T5_VOCAB = 32

__all__ = [
    "TINY_BLOCK_SIZE",
    "TINY_CAPTION_DIM",
    "TINY_CLS_TOKEN_NUM",
    "TINY_DOWNSAMPLE",
    "TINY_T5_VOCAB",
    "TINY_VOCAB_SIZE",
    "StubT5Encoder",
    "StubT5Tokenizer",
    "StubVQ",
    "build_tiny_gpt",
    "build_tiny_llamagen_model",
]


def build_tiny_gpt(**overrides) -> Transformer:
    """Genuine vendored t2i Transformer at test scale, regularizers zeroed.

    Zeroed dropout/drop-path mirrors ``model._load_llamagen_gpt``: the RL path
    never enables stochastic regularization, so train/eval forwards stay
    numerically identical.
    """
    args = dict(
        dim=TINY_DIM,
        n_layer=TINY_N_LAYER,
        n_head=TINY_N_HEAD,
        block_size=TINY_BLOCK_SIZE,
        cls_token_num=TINY_CLS_TOKEN_NUM,
        caption_dim=TINY_CAPTION_DIM,
        vocab_size=TINY_VOCAB_SIZE,
        model_type="t2i",
        token_dropout_p=0.0,
        attn_dropout_p=0.0,
        resid_dropout_p=0.0,
        ffn_dropout_p=0.0,
        drop_path_rate=0.0,
        class_dropout_prob=0.0,
    )
    args.update(overrides)
    torch.manual_seed(0)
    gpt = Transformer(ModelArgs(**args))
    # The vendored init zeroes the output head (a from-scratch pretraining
    # convention); tests need non-degenerate logits, so re-randomize it the
    # way a trained checkpoint would populate it.
    nn.init.normal_(gpt.output.weight, std=0.02)
    return gpt.eval()


class StubT5Encoder(nn.Module):
    """Deterministic stand-in for the frozen flan-t5-xl encoder.

    ``last_hidden_state = embedding(input_ids)`` — a fixed-seed table, so the
    same prompt ids always produce the same caption features (what replay
    relies on with the real frozen T5).
    """

    def __init__(
        self, *, vocab_size: int = TINY_T5_VOCAB, caption_dim: int = TINY_CAPTION_DIM
    ) -> None:
        super().__init__()
        torch.manual_seed(1)
        self.embed = nn.Embedding(vocab_size, caption_dim)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None):
        del attention_mask
        return SimpleNamespace(last_hidden_state=self.embed(input_ids))


class StubT5Tokenizer:
    """Char-hash tokenizer with T5-style right padding and variable lengths."""

    pad_token_id = 0

    def __call__(
        self,
        texts: list[str],
        *,
        max_length: int,
        padding: str = "max_length",
        truncation: bool = True,
        return_attention_mask: bool = True,
        add_special_tokens: bool = True,
        return_tensors: str = "pt",
    ) -> dict[str, torch.Tensor]:
        del padding, truncation, return_attention_mask, add_special_tokens, return_tensors
        ids = torch.zeros(len(texts), max_length, dtype=torch.long)
        mask = torch.zeros(len(texts), max_length, dtype=torch.long)
        for row, text in enumerate(texts):
            tokens = [1 + (ord(ch) % (TINY_T5_VOCAB - 1)) for ch in text][:max_length]
            if not tokens:
                tokens = [1]
            ids[row, : len(tokens)] = torch.tensor(tokens, dtype=torch.long)
            mask[row, : len(tokens)] = 1
        return {"input_ids": ids, "attention_mask": mask}


class StubVQ(nn.Module):
    """Frozen VQ-decoder boundary: wrappers only read the decoded shape."""

    def __init__(self, *, vocab_size: int = TINY_VOCAB_SIZE, codebook_dim: int = 8) -> None:
        super().__init__()
        self.quantize = nn.Module()
        self.quantize.embedding = nn.Embedding(vocab_size, codebook_dim)
        self.decode_shapes: list[list[int]] = []

    def decode_code(self, ids: torch.Tensor, shape: list[int]) -> torch.Tensor:
        del ids
        self.decode_shapes.append(list(shape))
        batch, _, height, width = shape
        return torch.zeros(batch, 3, height * TINY_DOWNSAMPLE, width * TINY_DOWNSAMPLE)


def build_tiny_llamagen_model(
    *,
    replay: bool = False,
    **config_kwargs,
) -> LlamaGenModel:
    """Wrap a genuine tiny vendored GPT into a real ``LlamaGenModel``."""
    from vrl.models.families.llamagen.model import LlamaGenReplayModel

    config_kwargs.setdefault("use_lora", False)
    config_kwargs.setdefault("device", "cpu")
    config_kwargs.setdefault("dtype", "float32")
    config_kwargs.setdefault("image_token_num", TINY_BLOCK_SIZE)
    config_kwargs.setdefault("cls_token_num", TINY_CLS_TOKEN_NUM)
    config_kwargs.setdefault("caption_dim", TINY_CAPTION_DIM)
    config_kwargs.setdefault("guidance_scale", 3.0)
    config_kwargs.setdefault("top_k", 0)
    config_kwargs.setdefault("top_p", 1.0)
    cls = LlamaGenReplayModel if replay else LlamaGenModel
    return cls(
        LlamaGenConfig(**config_kwargs),
        gpt=build_tiny_gpt(),
        vq_model=None if replay else StubVQ(),
        t5_encoder=StubT5Encoder(),
        t5_tokenizer=StubT5Tokenizer(),
    )
