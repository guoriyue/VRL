"""Encode a manifest's target videos into the ``data.sft_latents`` shard.

The GRPO diffusion-loss regularizer (``algorithm.sft_weight``, the
Cosmos-Predict2.5 paper's anti-reward-hacking term) trains against CLEAN
fine-tuning latents, but the trainer's replay model deliberately loads no
VAE — so the encode happens here, once, offline, with the same full bundle
the rollout workers build. The shard must be encoded with the SAME model and
sampling geometry as the training run, which is why this script takes the
experiment config rather than free-form arguments:

    python -m vrl.scripts.diffusion.encode_targets \\
        --experiment diffusion/cosmos_predict2/online_grpo_droid_target_480p \\
        --out data/droid/sft_latents_480p_93f.pt

Requires the family model to expose ``encode_video_to_latents`` (Cosmos
Predict2 does; add it to another family before pointing this script at it)
and every manifest row to carry a ``target_video`` artifact.
"""

from __future__ import annotations

import argparse
import logging
from typing import Any

logger = logging.getLogger(__name__)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment",
        required=True,
        help="experiment config name, e.g. diffusion/cosmos_predict2/online_grpo_...",
    )
    parser.add_argument("--out", required=True, help="shard output path (.pt)")
    parser.add_argument(
        "--device",
        default=None,
        help="cuda/cpu (default: cuda if available; cpu works, slowly)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="encode only the first N manifest rows (smoke runs)",
    )
    return parser


def _video_at_sampling_geometry(
    path: str,
    *,
    height: int,
    width: int,
    num_frames: int,
) -> Any:
    """Read a target video as ``[1, C, T, H, W]`` in [0,1] at the train shape."""

    import torch.nn.functional as F

    from vrl.utils.media import read_video_frames

    frames = read_video_frames(path, num_frames=num_frames)  # [T,H,W,3] in [0,1]
    video = frames.permute(0, 3, 1, 2)  # [T,3,H,W]
    if video.shape[-2:] != (height, width):
        video = F.interpolate(
            video,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )
    return video.permute(1, 0, 2, 3).unsqueeze(0)  # [1,3,T,H,W]


def main() -> None:
    args = _build_arg_parser().parse_args()
    logging.basicConfig(level=logging.INFO)

    import torch
    from omegaconf import OmegaConf

    from vrl.config.loading import load_config
    from vrl.rollouts.families.registry import (
        get_rollout_family_entry,
        normalize_rollout_family,
    )
    from vrl.trainers.data import load_prompt_examples_from_config
    from vrl.trainers.data.artifacts import (
        resolve_prompt_example_artifacts,
        save_sft_latents,
    )
    from vrl.utils.config import import_from_path

    cfg = load_config(f"experiment/{args.experiment}")
    family = normalize_rollout_family(str(cfg.model.family))
    entry = get_rollout_family_entry(family)

    device = torch.device(
        args.device
        if args.device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu"),
    )
    height = int(OmegaConf.select(cfg, "sampling.height"))
    width = int(OmegaConf.select(cfg, "sampling.width"))
    num_frames = int(OmegaConf.select(cfg, "sampling.num_frames", default=1) or 1)

    examples = load_prompt_examples_from_config(cfg)
    if args.limit is not None:
        examples = examples[: int(args.limit)]
    if not examples:
        raise ValueError("the training manifest resolved to zero examples")

    # The exact builder path the rollout workers import — same model, same
    # weights, same latent space as the run this shard will regularize.
    build = import_from_path(entry.runtime_builder)
    extract = import_from_path(entry.runtime_spec_extractor)
    weight_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    bundle = build(extract(cfg, device, weight_dtype))
    model = bundle.model
    encode = getattr(model, "encode_video_to_latents", None)
    if not callable(encode):
        raise NotImplementedError(
            f"{type(model).__name__} exposes no encode_video_to_latents; add "
            "the family's VAE-encode inverse (see CosmosPredict2Model) before "
            "encoding sft latents for it",
        )

    latents_by_prompt: dict[str, Any] = {}
    for index, example in enumerate(examples):
        resolved = resolve_prompt_example_artifacts(example)
        target = getattr(resolved, "target_video", None) or getattr(
            example, "target_video", None,
        )
        if not target:
            raise ValueError(
                f"manifest row {index} ({example.prompt!r}) has no target_video; "
                "the sft shard needs one clean video per prompt",
            )
        video = _video_at_sampling_geometry(
            str(target),
            height=height,
            width=width,
            num_frames=num_frames,
        )
        latents = encode(video.to(device))
        latents_by_prompt[str(example.prompt)] = latents.squeeze(0).detach().cpu()
        logger.info(
            "[%d/%d] encoded %s -> %s",
            index + 1,
            len(examples),
            target,
            tuple(latents_by_prompt[str(example.prompt)].shape),
        )

    save_sft_latents(
        args.out,
        family=family,
        model_path=str(OmegaConf.select(cfg, "model.path", default="")),
        latents_by_prompt=latents_by_prompt,
    )
    logger.info(
        "wrote %d prompt latents to %s (set data.sft_latents to this path and "
        "algorithm.sft_weight > 0 to enable the regularizer)",
        len(latents_by_prompt),
        args.out,
    )


if __name__ == "__main__":
    main()
