"""Tiny real (cache-free) Kling VideoReward models for CPU tests.

``build_tiny_kling_reward_model`` is a genuine ``KlingQwen2VLRewardModel`` built
straight from a ``Qwen2VLConfig`` (~32K parameters), so the pooling branches,
the checkpoint loader, and PEFT wrapping run their real code on CPU.
``build_tiny_qwen2vl_repo`` additionally writes the model and a genuine
``Qwen2VLProcessor`` to disk with transformers' own ``save_pretrained`` so
``_create_model_and_processor`` can take its real ``from_pretrained`` path
offline. Mirrors ``tests/models/steps/denoise/fixtures.py``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

TINY_HIDDEN_SIZE = 16

# The structural core of Qwen2-VL's chat template: one video placeholder inside
# vision markers (the processor expands ``<|video_pad|>`` per grid cell) and the
# text content, framed by im_start/im_end. Owned by the test repository, not a
# copy of the hub file.
_TINY_CHAT_TEMPLATE = (
    "{% for message in messages %}<|im_start|>{{ message['role'] }}\n"
    "{% for content in message['content'] %}"
    "{% if content['type'] == 'video' %}<|vision_start|><|video_pad|><|vision_end|>"
    "{% elif content['type'] == 'text' %}{{ content['text'] }}{% endif %}"
    "{% endfor %}<|im_end|>\n{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)


def build_tiny_kling_reward_model(
    *,
    output_dim: int = 4,
    reward_token: str = "last",
    special_token_ids: list[int] | None = None,
    pad_token_id: int | None = 0,
    vocab_size: int = 64,
    seed: int = 0,
) -> Any:
    """A tiny real ``KlingQwen2VLRewardModel`` on CPU, random-init from ``seed``.

    ``bos_token_id``/``eos_token_id`` are cleared at both config levels (the
    Qwen2-VL defaults sit far outside a 64-token vocabulary) and
    ``pad_token_id`` is assigned after construction because ``Qwen2VLConfig``
    does not accept it as a constructor argument.
    """

    from transformers import Qwen2VLConfig

    from vrl.rewards.models.kling_video_reward import KlingQwen2VLRewardModel

    config = Qwen2VLConfig(
        text_config={
            "vocab_size": vocab_size,
            "hidden_size": TINY_HIDDEN_SIZE,
            "intermediate_size": 32,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "max_position_embeddings": 64,
            "bos_token_id": None,
            "eos_token_id": None,
            "rope_parameters": {
                "rope_type": "default",
                "rope_theta": 10000.0,
                "mrope_section": [2, 1, 1],
            },
        },
        vision_config={
            "depth": 1,
            "hidden_size": TINY_HIDDEN_SIZE,
            "embed_dim": TINY_HIDDEN_SIZE,
            "num_heads": 2,
            "in_chans": 3,
            "spatial_patch_size": 14,
            "temporal_patch_size": 2,
            "out_hidden_size": TINY_HIDDEN_SIZE,
        },
        bos_token_id=None,
        eos_token_id=None,
    )
    config.pad_token_id = pad_token_id
    with torch.random.fork_rng(devices=[], device_type="cpu"):
        torch.manual_seed(seed)
        return KlingQwen2VLRewardModel(
            config,
            output_dim=output_dim,
            reward_token=reward_token,
            special_token_ids=special_token_ids,
        )


def head_logits(model: Any, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Per-token reward-head logits, the reference every pooling branch selects from."""

    with torch.no_grad():
        hidden = model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )[0]
        return model.rm_head(hidden)


def build_tiny_qwen2vl_processor(root: Path) -> Any:
    """Write a real ``Qwen2VLProcessor`` over a byte-level vocabulary under ``root``.

    The tokenizer is a byte-level vocabulary with zero merges plus the Qwen2-VL
    control tokens the chat template and video processor emit (registered as
    special tokens so the zero-merge BPE keeps them atomic).
    """

    from transformers import (
        Qwen2Tokenizer,
        Qwen2VLImageProcessor,
        Qwen2VLProcessor,
        Qwen2VLVideoProcessor,
    )
    from transformers.convert_slow_tokenizer import bytes_to_unicode

    root.mkdir(parents=True, exist_ok=True)
    alphabet = list(bytes_to_unicode().values())
    control = [
        "<|endoftext|>",
        "<|im_start|>",
        "<|im_end|>",
        "<|vision_start|>",
        "<|vision_end|>",
        "<|video_pad|>",
        "<|image_pad|>",
    ]
    vocab = {token: index for index, token in enumerate([*alphabet, *control])}
    (root / "vocab.json").write_text(json.dumps(vocab), encoding="utf-8")
    (root / "merges.txt").write_text("#version: 0.2\n", encoding="utf-8")
    tokenizer = Qwen2Tokenizer(str(root / "vocab.json"), str(root / "merges.txt"))
    # Register the control tokens as special tokens so the zero-merge BPE keeps
    # them atomic (a plain vocabulary entry would be split back into bytes).
    tokenizer.add_special_tokens({"additional_special_tokens": control[1:]})
    tokenizer.chat_template = _TINY_CHAT_TEMPLATE
    # ``video_processor=`` is required: Qwen2VLProcessor raises on a missing one.
    processor = Qwen2VLProcessor(
        image_processor=Qwen2VLImageProcessor(),
        tokenizer=tokenizer,
        video_processor=Qwen2VLVideoProcessor(),
    )
    processor.save_pretrained(root)
    return processor


def vision_token_ids(tokenizer: Any) -> dict[str, int]:
    """The config ids a Qwen2-VL forward splices vision features at, for ``tokenizer``.

    The Qwen2-VL defaults point at the 151k hub vocabulary, not a tiny one.
    """

    return {
        name: tokenizer.convert_tokens_to_ids(token)
        for name, token in (
            ("vision_start_token_id", "<|vision_start|>"),
            ("vision_end_token_id", "<|vision_end|>"),
            ("video_token_id", "<|video_pad|>"),
            ("image_token_id", "<|image_pad|>"),
        )
    }


def build_tiny_qwen2vl_repo(root: Path, *, output_dim: int = 1, seed: int = 0) -> Path:
    """Write a tiny Qwen2-VL reward model + real ``Qwen2VLProcessor`` under ``root``.

    Eight spare ids are left above the vocabulary so ``add_special_tokens`` +
    ``resize_token_embeddings`` have room for the three ``<|*_reward|>`` tokens.
    """

    tokenizer = build_tiny_qwen2vl_processor(root).tokenizer
    model = build_tiny_kling_reward_model(
        output_dim=output_dim,
        vocab_size=len(tokenizer) + 8,
        pad_token_id=tokenizer.pad_token_id,
        seed=seed,
    )
    for name, token_id in vision_token_ids(tokenizer).items():
        setattr(model.config, name, token_id)
    model.save_pretrained(root)
    return root


__all__ = [
    "TINY_HIDDEN_SIZE",
    "build_tiny_kling_reward_model",
    "build_tiny_qwen2vl_processor",
    "build_tiny_qwen2vl_repo",
    "head_logits",
    "vision_token_ids",
]
