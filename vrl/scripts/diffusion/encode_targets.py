"""Encode a manifest's target videos into the ``data.sft_latents`` shard.

The GRPO diffusion-loss regularizer (``algorithm.sft_weight``, the
Cosmos-Predict2.5 paper's anti-reward-hacking term) trains against CLEAN
fine-tuning latents, but the trainer's replay model deliberately loads no
VAE — so the encode happens here, once, offline, with the same full bundle
the rollout workers build. The shard must be encoded with the SAME model and
sampling geometry as the training run, which is why this script takes the
experiment config rather than free-form arguments:

    python -m vrl.scripts.diffusion.encode_targets \\
        --experiment diffusion/cosmos_predict2_5/online_grpo_droid_sft_numerics_240p_33f \\
        --out data/external/video_world/sft_latents/cosmos_predict25_240p_33f.pt \\
        --preview-out outputs/cosmos_predict25_sft_target_roundtrip.mp4

Requires the family model to expose ``encode_video_to_latents`` and every
manifest row to carry a ``target_video`` artifact. The shard is keyed by that
stable artifact identity, not prompt text: real fine-tuning manifests may use
the same instruction for several distinct target videos.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
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
    parser.add_argument(
        "--preview-out",
        default=None,
        help="decode the first encoded target and write an mp4 round-trip preview",
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
    if int(frames.shape[0]) != num_frames:
        raise ValueError(
            f"target video {path} yielded {int(frames.shape[0])} frames, but the "
            f"training sampling geometry requires {num_frames}; rebuild the target "
            "clip at the training frame count instead of silently padding or "
            "interpolating supervision",
        )
    video = frames.permute(0, 3, 1, 2)  # [T,3,H,W]
    if video.shape[-2:] != (height, width):
        video = F.interpolate(
            video,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )
    return video.permute(1, 0, 2, 3).unsqueeze(0)  # [1,3,T,H,W]


def _resolve_target_videos(
    examples: list[Any],
    *,
    data_root: str | Path | None,
    allow_absolute: bool,
) -> list[tuple[str, str]]:
    """Resolve and validate stable target identities before loading a model."""

    from vrl.trainers.data.artifacts import resolve_prompt_example_artifacts

    targets: list[tuple[str, str]] = []
    seen_target_keys: set[str] = set()
    for index, example in enumerate(examples):
        target_key = str(getattr(example, "target_video", "") or "").strip()
        if not target_key:
            raise ValueError(
                f"manifest row {index} ({example.prompt!r}) has no target_video; "
                "the sft shard needs one clean video per training example",
            )
        if target_key in seen_target_keys:
            raise ValueError(
                f"manifest row {index} repeats target_video {target_key!r}; "
                "target_video is the sft shard identity and must be unique",
            )
        seen_target_keys.add(target_key)
        resolved = resolve_prompt_example_artifacts(
            example,
            data_root=data_root,
            allow_absolute=allow_absolute,
        )
        resolved_target = Path(resolved.target_video)
        if not resolved_target.is_file():
            raise FileNotFoundError(
                f"manifest row {index} target_video does not exist: {resolved_target}",
            )
        targets.append((target_key, str(resolved_target)))
    return targets


def main() -> None:
    args = _build_arg_parser().parse_args()
    logging.basicConfig(level=logging.INFO)

    import torch
    from omegaconf import OmegaConf

    from vrl.config.loading import load_config
    from vrl.rollouts.families.registry import (
        get_rollout_family_entry,
        resolve_entry_model_build,
    )
    from vrl.trainers.data import load_prompt_examples_from_config
    from vrl.trainers.data.sft_latents import save_sft_latents
    from vrl.utils.config import import_from_path

    cfg = load_config(f"experiment/{args.experiment}")
    entry = get_rollout_family_entry(str(cfg.model.family))
    family = entry.family

    device = torch.device(
        args.device
        if args.device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu"),
    )
    height = int(OmegaConf.select(cfg, "sampling.height"))
    width = int(OmegaConf.select(cfg, "sampling.width"))
    num_frames = int(OmegaConf.select(cfg, "sampling.num_frames", default=1) or 1)
    fps = float(OmegaConf.select(cfg, "sampling.fps", default=16) or 16)

    examples = load_prompt_examples_from_config(cfg.data)
    if args.limit is not None:
        examples = examples[: int(args.limit)]
    if not examples:
        raise ValueError("the training manifest resolved to zero examples")

    data_root = OmegaConf.select(cfg, "data.artifact_data_root", default=None)
    allow_absolute = bool(
        OmegaConf.select(cfg, "data.allow_absolute_artifact_paths", default=False),
    )
    targets = _resolve_target_videos(
        examples,
        data_root=data_root,
        allow_absolute=allow_absolute,
    )

    # The exact builder path the rollout workers import — same model, same
    # weights, same latent space as the run this shard will regularize.
    build = import_from_path(entry.runtime_builder)
    weight_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    bundle = build(
        resolve_entry_model_build(
            entry,
            cfg,
            device,
            parameter_dtype_override=weight_dtype,
        ),
    )
    model = bundle.model
    encode = getattr(model, "encode_video_to_latents", None)
    if not callable(encode):
        raise NotImplementedError(
            f"{type(model).__name__} exposes no encode_video_to_latents; add "
            "the family's VAE-encode inverse (see CosmosPredict2Model) before "
            "encoding sft latents for it",
        )

    latents_by_target: dict[str, Any] = {}
    for index, (target_key, target) in enumerate(targets):
        video = _video_at_sampling_geometry(
            str(target),
            height=height,
            width=width,
            num_frames=num_frames,
        )
        latents = encode(video.to(device))
        latents_by_target[target_key] = latents.squeeze(0).detach().cpu()
        if args.preview_out and index == 0:
            from vrl.utils.media import write_mp4

            decoded = model.decode_latents(latents)
            write_mp4(decoded, args.preview_out, fps=fps)
            logger.info("wrote first-target round-trip preview to %s", args.preview_out)
        logger.info(
            "[%d/%d] encoded %s -> %s",
            index + 1,
            len(examples),
            target,
            tuple(latents_by_target[target_key].shape),
        )

    save_sft_latents(
        args.out,
        family=family,
        model_path=str(OmegaConf.select(cfg, "model.path", default="")),
        model_revision=str(OmegaConf.select(cfg, "model.revision", default="") or ""),
        latents_by_target=latents_by_target,
    )
    logger.info(
        "wrote %d target latents to %s (set data.sft_latents to this path and "
        "algorithm.sft_weight > 0 to enable the regularizer)",
        len(latents_by_target),
        args.out,
    )


if __name__ == "__main__":
    main()
