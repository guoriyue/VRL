"""Encode a manifest's clean target images or videos into ``data.sft_latents``.

The GRPO diffusion-loss regularizer (``algorithm.sft_weight``, the
Cosmos-Predict2.5 paper's anti-reward-hacking term) trains against CLEAN
fine-tuning latents, but the trainer's replay model deliberately loads no
VAE — so the encode happens here, once, offline, with the same full bundle
the rollout workers build. The shard must be encoded with the SAME model and
sampling geometry as the training run, which is why this script takes the
training config rather than free-form arguments. The shard is written to the
config's ``data.sft_latents``, and the first target's VAE round trip is
written beside it as ``<shard stem>_roundtrip.png`` (image) or ``.mp4`` (video):

    python -m vrl.scripts.denoise.encode_targets \\
        --config experiment/cosmos_predict2_5/online_grpo_droid_sft_numerics_240p_33f

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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Bundled config name or YAML path.")
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
    else:  # pragma: no cover - CleanTargetRef.from_source owns this closed set
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
) -> list[tuple[str, str, str]]:
    """Resolve and validate stable target identities before loading a model."""
    from vrl.trainers.data.sft_latents import CleanTargetRef
    from vrl.utils.paths import RootedPaths

    roots = RootedPaths.data_root(data_root)
    targets: list[tuple[str, str, str]] = []
    seen_target_keys: set[str] = set()
    for index, example in enumerate(examples):
        try:
            target = CleanTargetRef.from_source(example)
        except ValueError as exc:
            raise ValueError(f"manifest row {index} ({example.prompt!r}): {exc}") from exc
        if target.key in seen_target_keys:
            raise ValueError(
                f"manifest row {index} repeats clean target {target.key!r}; "
                "the target artifact is the sft shard identity and must be unique",
            )
        seen_target_keys.add(target.key)
        resolved_target = roots.resolve(str(getattr(example, target.field)))
        if not resolved_target.is_file():
            raise FileNotFoundError(
                f"manifest row {index} {target.field} does not exist: {resolved_target}",
            )
        targets.append((target.key, str(resolved_target), target.field))
    return targets


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO)

    import torch

    from vrl.config.loading import load_config
    from vrl.config.precision import PrecisionPolicy
    from vrl.config.schema import parse_config
    from vrl.models.dtypes import resolve_torch_dtype
    from vrl.models.families.registry import (
        get_model_family_entry,
    )
    from vrl.trainers.data.prompts import load_prompt_examples_from_config
    from vrl.trainers.data.sft_latents import save_sft_latents
    from vrl.utils.paths import user_path

    root = parse_config(load_config(args.config, overrides=args.overrides))
    precision = PrecisionPolicy.from_section(root.precision)
    if root.model is None:
        raise ValueError("target encoding requires model configuration")
    entry = get_model_family_entry(str(root.model.family))
    family = entry.family

    if root.data is None or not root.data.sft_latents:
        raise ValueError("target encoding writes data.sft_latents; set it in the config")
    out = user_path(str(root.data.sft_latents))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sampling = root.sampling
    if sampling is None:
        raise ValueError("target encoding requires sampling configuration")
    height = int(sampling.height)
    width = int(sampling.width)
    num_frames = int(getattr(sampling, "num_frames", None) or 1)
    fps = float(getattr(sampling, "fps", None) or 16)

    examples = load_prompt_examples_from_config(root.data)
    if not examples:
        raise ValueError("the training manifest resolved to zero examples")

    targets = _resolve_clean_targets(examples, data_root=root.data.artifact_data_root)

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
        expected_identity=None,
    )
    model = bundle.model
    encode = getattr(model, "encode_video_to_latents", None)
    if not callable(encode):
        raise NotImplementedError(
            f"{type(model).__name__} exposes no encode_video_to_latents; add "
            "the family's VAE-encode inverse (see CosmosPredict2PipelineModel) before "
            "encoding sft latents for it",
        )

    latents_by_target: dict[str, Any] = {}
    storage_dtype = (
        None if args.storage_dtype == "preserve" else resolve_torch_dtype(args.storage_dtype)
    )
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
        if index == 0:
            from vrl.utils.media import write_mp4, write_png

            decoded = model.decode_latents(latents)
            if media_type == "target_image":
                preview_path = out.with_name(f"{out.stem}_roundtrip.png")
                write_png(decoded[0], preview_path)
            else:
                preview_path = out.with_name(f"{out.stem}_roundtrip.mp4")
                write_mp4(decoded, preview_path, fps=fps)
            logger.info("wrote first-target round-trip preview to %s", preview_path)
        logger.info(
            "[%d/%d] encoded %s -> %s",
            index + 1,
            len(examples),
            target,
            tuple(latents_by_target[target_key].shape),
        )

    save_sft_latents(
        out,
        family=family,
        model_path=str(root.model.path or ""),
        model_revision=str(root.model.revision or ""),
        latents_by_target=latents_by_target,
    )
    logger.info(
        "wrote %d target latents to %s (algorithm.sft_weight > 0 enables the regularizer)",
        len(latents_by_target),
        out,
    )


if __name__ == "__main__":
    main()
