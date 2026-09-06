"""Tiny real (cache-free) CLIP repositories for CPU reward-model tests.

``build_tiny_clip_repo`` writes a genuine ``CLIPModel`` plus a genuine
``CLIPProcessor`` to a directory with transformers' own ``save_pretrained``, so
the production loaders (``CLIPModel.from_pretrained`` / ``CLIPProcessor``)
read it through their real code paths — no download, no cached weights, no
hand-written fake that only echoes what it was told. Mirrors
``tests/models/steps/denoise/fixtures.py``.
"""

from __future__ import annotations

import json
from importlib import resources
from pathlib import Path

import torch


def shipped_aesthetic_projection_dim() -> int:
    """The CLIP projection width the shipped LAION aesthetic head was trained on.

    Read off the packaged asset rather than hard-coded: the head's first
    ``Linear`` input width is the only source of truth for this number.
    """

    asset = resources.files("vrl.rewards.assets").joinpath("sac+logos+ava1-l14-linearMSE.pth")
    state = torch.load(asset, map_location="cpu", weights_only=True)
    return int(state["layers.0.weight"].shape[1])


def build_tiny_clip_repo(
    root: Path,
    *,
    projection_dim: int,
    logit_scale_init_value: float,
    seed: int = 0,
) -> Path:
    """Write a tiny real CLIP model + processor repository under ``root``.

    The tokenizer vocabulary is the byte-level alphabet with zero merges, which
    tokenizes any prompt; the image processor resizes to the model's 8x8 grid.
    Weights are seeded inside a CPU-only ``fork_rng`` (a bare ``fork_rng()``
    would initialise a CUDA context in the CPU lane) and written to disk, so
    nothing random happens when a test later loads the repository.
    """

    from transformers import (
        CLIPConfig,
        CLIPImageProcessor,
        CLIPModel,
        CLIPProcessor,
        CLIPTokenizer,
    )
    from transformers.convert_slow_tokenizer import bytes_to_unicode

    root.mkdir(parents=True, exist_ok=True)
    alphabet = list(bytes_to_unicode().values())
    vocab = {
        token: index for index, token in enumerate([*alphabet, "<|startoftext|>", "<|endoftext|>"])
    }
    (root / "vocab.json").write_text(json.dumps(vocab), encoding="utf-8")
    (root / "merges.txt").write_text("#version: 0.2\n", encoding="utf-8")
    tokenizer = CLIPTokenizer(str(root / "vocab.json"), str(root / "merges.txt"))
    image_processor = CLIPImageProcessor(
        size={"shortest_edge": 8},
        crop_size={"height": 8, "width": 8},
    )
    CLIPProcessor(image_processor=image_processor, tokenizer=tokenizer).save_pretrained(root)

    config = CLIPConfig(
        text_config={
            "vocab_size": len(vocab),
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "max_position_embeddings": 16,
        },
        vision_config={
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "image_size": 8,
            "patch_size": 4,
        },
        projection_dim=projection_dim,
        logit_scale_init_value=logit_scale_init_value,
    )
    with torch.random.fork_rng(devices=[], device_type="cpu"):
        torch.manual_seed(seed)
        model = CLIPModel(config)
    model.save_pretrained(root)
    return root


def build_tiny_qwen_vl_judge_repo(root: Path, *, seed: int = 0) -> Path:
    """A tiny generative ``Qwen2VLForConditionalGeneration`` + real processor under ``root``.

    The judge base loads it through ``AutoProcessor`` / ``AutoModelForImageTextToText``
    exactly as it loads VideoScore2; generation runs in milliseconds on CPU.
    """

    from transformers import Qwen2VLForConditionalGeneration

    from tests.rewards.kling_video_reward.fixtures import (
        build_tiny_kling_reward_model,
        build_tiny_qwen2vl_processor,
        vision_token_ids,
    )

    tokenizer = build_tiny_qwen2vl_processor(root).tokenizer
    config = build_tiny_kling_reward_model(
        vocab_size=len(tokenizer),
        pad_token_id=tokenizer.pad_token_id,
    ).config
    for name, token_id in vision_token_ids(tokenizer).items():
        setattr(config, name, token_id)
    with torch.random.fork_rng(devices=[], device_type="cpu"):
        torch.manual_seed(seed)
        model = Qwen2VLForConditionalGeneration(config)
    model.save_pretrained(root)
    return root


__all__ = [
    "build_tiny_clip_repo",
    "build_tiny_qwen_vl_judge_repo",
    "shipped_aesthetic_projection_dim",
]
