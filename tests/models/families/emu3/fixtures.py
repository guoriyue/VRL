"""Tiny-real Emu3 fixtures built straight from transformers configs.

Unlike the Janus fixtures (hand-written stubs, because the ``janus`` package
is not a declared dependency), Emu3 ships inside ``transformers``, so these
helpers construct GENUINE ``Emu3ForConditionalGeneration`` / replay-core
modules from tiny configs — no checkpoint download, CPU-only, float32.

The tiny vocabulary layout (kept deliberately small so masks are readable):

  * BPE ids 40..47  — the 8 ``<|visual token NNNNNN|>`` image tokens
  * BPE id 48       — EOL ``<|extra_200|>``
  * BPE id 49       — EOF ``<|extra_201|>``
  * BPE id 50       — EOI ``<|image end|>``
  * BPE id 3        — EOS (from the text config)

so the generation vocab is 12 wide: image indices 0..7, then EOL=8, EOF=9,
EOI=10, EOS=11.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch

from vrl.models.families.emu3.config import Emu3Config
from vrl.models.families.emu3.model import (
    Emu3Model,
    Emu3ReplayCore,
    Emu3ReplayModel,
)

TINY_IMAGE_VOCAB = 8
TINY_GEN_VOCAB = TINY_IMAGE_VOCAB + 4  # + EOL/EOF/EOI/EOS
TINY_HIDDEN = 16
TINY_EOS_TOKEN_ID = 3
# Generation-vocab indices of the structural columns.
TINY_EOL_IDX = TINY_IMAGE_VOCAB
TINY_EOF_IDX = TINY_IMAGE_VOCAB + 1
TINY_EOI_IDX = TINY_IMAGE_VOCAB + 2
TINY_EOS_IDX = TINY_IMAGE_VOCAB + 3


def tiny_hf_emu3_config():
    """Build a tiny (~230k param) transformers Emu3Config."""

    from transformers.models.emu3.configuration_emu3 import (
        Emu3Config as HFEmu3Config,
    )
    from transformers.models.emu3.configuration_emu3 import (
        Emu3TextConfig,
        Emu3VQVAEConfig,
    )

    vocab_map = {f"<|visual token {i:06d}|>": 40 + i for i in range(TINY_IMAGE_VOCAB)}
    vocab_map["<|extra_200|>"] = 48
    vocab_map["<|extra_201|>"] = 49
    vocab_map["<|image end|>"] = 50
    vocab_map["<image>"] = 51
    text_config = Emu3TextConfig(
        vocab_size=64,
        hidden_size=TINY_HIDDEN,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=512,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=TINY_EOS_TOKEN_ID,
    )
    # base_channels/hidden_size must stay GroupNorm(32)-divisible.
    vq_config = Emu3VQVAEConfig(
        codebook_size=8,
        embed_dim=4,
        latent_channels=4,
        base_channels=32,
        channel_multiplier=[1, 1],
        num_res_blocks=1,
        attn_resolutions=[],
        hidden_size=32,
        num_attention_heads=1,
    )
    return HFEmu3Config(
        vq_config=vq_config,
        text_config=text_config,
        vocabulary_map=vocab_map,
    )


def _stub_processor() -> SimpleNamespace:
    """Minimal processor surface: only ``tokenizer`` id attrs are read on the
    non-encode paths these tests exercise (real prompt encoding needs the
    checkpoint tokenizer files, which tiny tests never download)."""

    return SimpleNamespace(
        tokenizer=SimpleNamespace(
            pad_token_id=0,
            eos_token_id=TINY_EOS_TOKEN_ID,
            padding_side="left",
        ),
    )


def build_tiny_emu3_model(*, use_lora: bool = False) -> Emu3Model:
    from transformers.models.emu3.modeling_emu3 import Emu3ForConditionalGeneration

    torch.manual_seed(0)
    hf_model = Emu3ForConditionalGeneration(tiny_hf_emu3_config())
    config = Emu3Config(
        model_path="tiny-emu3",
        dtype="float32",
        device="cpu",
        use_lora=use_lora,
    )
    return Emu3Model(config, emu3=hf_model, processor=_stub_processor())


def build_tiny_emu3_replay_model(*, use_lora: bool = False) -> Emu3ReplayModel:
    torch.manual_seed(0)
    core = Emu3ReplayCore(tiny_hf_emu3_config())
    config = Emu3Config(
        model_path="tiny-emu3",
        dtype="float32",
        device="cpu",
        use_lora=use_lora,
    )
    return Emu3ReplayModel(config, emu3=core)


__all__ = [
    "TINY_EOF_IDX",
    "TINY_EOI_IDX",
    "TINY_EOL_IDX",
    "TINY_EOS_IDX",
    "TINY_EOS_TOKEN_ID",
    "TINY_GEN_VOCAB",
    "TINY_HIDDEN",
    "TINY_IMAGE_VOCAB",
    "build_tiny_emu3_model",
    "build_tiny_emu3_replay_model",
    "tiny_hf_emu3_config",
]
