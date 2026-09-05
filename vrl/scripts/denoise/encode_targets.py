"""Encode a manifest's clean target images or videos into ``data.sft_latents``.

The GRPO diffusion-loss regularizer (``algorithm.sft_weight``, the
Cosmos-Predict2.5 paper's anti-reward-hacking term) trains against CLEAN
fine-tuning latents, but the trainer's replay model deliberately loads no
VAE — so the encode happens here, once, offline, with the same full bundle
the rollout workers build. The shard must be encoded with the SAME model and
sampling geometry as the training run, which is why this script takes the
experiment config rather than free-form arguments:

    python -m vrl.scripts.denoise.encode_targets \\
        --experiment diffusion/cosmos_predict2_5/online_grpo_droid_sft_numerics_240p_33f \\
        --out data/external/video_world/sft_latents/cosmos_predict25_240p_33f.pt \\
        --preview-out outputs/cosmos_predict25_sft_target_roundtrip.mp4

Requires the family model to expose ``encode_video_to_latents`` and every
manifest row to carry exactly one ``target_image`` or ``target_video`` artifact.
The shard is keyed by that stable artifact identity, not prompt text: a
fine-tuning manifest may use the same instruction for several clean targets.
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
        help="decode the first target and write a PNG (image) or MP4 (video) preview",
    )
    parser.add_argument(
        "--storage-dtype",
        choices=("preserve", "bf16", "fp16", "fp32"),
        default="preserve",
        help="On-disk latent dtype; bf16 reduces replicated trainer host memory.",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help=(
            "Ordered preset overlays (+reward=ocr +dataset=...) and OmegaConf "
            "dotlist overrides (model.use_lora=false sampling.num_steps=40)."
        ),
    )
    return parser


def _target_at_sampling_geometry(
    path: str,
    *,
    media_type: str,
    height: int,
    width: int,
    num_frames: int,
) -> Any:
    """Read a clean target as ``[1, C, T, H, W]`` at the training shape."""

    import torch.nn.functional as F

    from vrl.utils.media import read_image_as_frames, read_video_frames

    if media_type == "target_image":
        if num_frames != 1:
            raise ValueError(
                f"target image {path} cannot supervise a {num_frames}-frame run; "
                "provide target_video instead of silently repeating one frame",
            )
        frames = read_image_as_frames(path)
    elif media_type == "target_video":
        frames = read_video_frames(path, num_frames=num_frames)
    else:  # pragma: no cover - resolve_clean_target owns this closed set
        raise ValueError(f"unsupported clean target field: {media_type}")
    if int(frames.shape[0]) != num_frames:
        raise ValueError(
            f"clean target {path} yielded {int(frames.shape[0])} frames, but the "
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


def _resolve_clean_targets(
    examples: list[Any],
    *,
    data_root: str | Path | None,
    allow_absolute: bool,
) -> list[tuple[str, str, str]]:
    """Resolve and validate stable target identities before loading a model."""

    from vrl.trainers.data.artifacts import resolve_prompt_example_artifacts
    from vrl.trainers.data.sft_latents import resolve_clean_target

    targets: list[tuple[str, str, str]] = []
    seen_target_keys: set[str] = set()
    for index, example in enumerate(examples):
        try:
            target = resolve_clean_target(example)
        except ValueError as exc:
            raise ValueError(f"manifest row {index} ({example.prompt!r}): {exc}") from exc
        if target.key in seen_target_keys:
            raise ValueError(
                f"manifest row {index} repeats clean target {target.key!r}; "
                "the target artifact is the sft shard identity and must be unique",
            )
        seen_target_keys.add(target.key)
        resolved = resolve_prompt_example_artifacts(
            example,
            data_root=data_root,
            allow_absolute=allow_absolute,
        )
        resolved_target = Path(str(getattr(resolved, target.field)))
        if not resolved_target.is_file():
            raise FileNotFoundError(
                f"manifest row {index} {target.field} does not exist: {resolved_target}",
            )
        targets.append((target.key, str(resolved_target), target.field))
    return targets


def main(argv: list[str] | None = None) -> None:
    args = _build_arg_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO)

    import torch

    from vrl.config.loading import load_config
    from vrl.config.precision import resolve_precision_policy
    from vrl.config.schema import parse_config
    from vrl.models.families.registry import (
        get_model_family_entry,
    )
    from vrl.trainers.data import load_prompt_examples_from_config
    from vrl.trainers.data.sft_latents import save_sft_latents

    cfg = load_config(f"experiment/{args.experiment}", overrides=args.overrides)
    root = parse_config(cfg)
    precision = resolve_precision_policy(root)
    if root.model is None:
        raise ValueError("target encoding requires model configuration")
    entry = get_model_family_entry(str(root.model.family))
    family = entry.family

    device = torch.device(
        args.device
        if args.device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu"),
    )
    sampling = root.sampling
    if sampling is None:
        raise ValueError("target encoding requires sampling configuration")
    height = int(sampling.height)
    width = int(sampling.width)
    num_frames = int(getattr(sampling, "num_frames", None) or 1)
    fps = float(getattr(sampling, "fps", None) or 16)

    examples = load_prompt_examples_from_config(root.data)
    if args.limit is not None:
        examples = examples[: int(args.limit)]
    if not examples:
        raise ValueError("the training manifest resolved to zero examples")

    data_root = root.data.artifact_data_root if root.data is not None else None
    allow_absolute = bool(root.data.allow_absolute_artifact_paths) if root.data else False
    targets = _resolve_clean_targets(
        examples,
        data_root=data_root,
        allow_absolute=allow_absolute,
    )

    # Use the same resolved family entry and build path as rollout workers.
    # CUDA follows the experiment's rollout precision. CPU explicitly promotes
    # parameter storage because several diffusion kernels do not support BF16
    # there; that compatibility override is owned by this offline tool.
    parameter_dtype_override = torch.float32 if device.type == "cpu" else None
    bundle = entry.build_rollout(
        entry.resolve_model_build(
            root,
            device,
            precision=precision,
            parameter_dtype_override=parameter_dtype_override,
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
    storage_dtype = {
        "preserve": None,
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[args.storage_dtype]
    for index, (target_key, target, media_type) in enumerate(targets):
        video = _target_at_sampling_geometry(
            str(target),
            media_type=media_type,
            height=height,
            width=width,
            num_frames=num_frames,
        )
        latents = encode(video.to(device))
        stored = latents.squeeze(0).detach()
        if storage_dtype is not None:
            stored = stored.to(dtype=storage_dtype)
        latents_by_target[target_key] = stored.cpu()
        if args.preview_out and index == 0:
            from vrl.utils.media import write_mp4, write_png

            decoded = model.decode_latents(latents)
            preview_path = Path(args.preview_out)
            if media_type == "target_image":
                if preview_path.suffix.lower() != ".png":
                    raise ValueError("an image target round-trip preview must end in .png")
                write_png(decoded[0], preview_path)
            else:
                if preview_path.suffix.lower() != ".mp4":
                    raise ValueError("a video target round-trip preview must end in .mp4")
                write_mp4(decoded, preview_path, fps=fps)
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
        model_path=str(root.model.path or ""),
        model_revision=str(root.model.revision or ""),
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
