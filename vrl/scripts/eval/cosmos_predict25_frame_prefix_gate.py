"""Live Cosmos Predict2.5 frame-prefix conditioning gate.

The earlier world-model steppability probe mixed already-answered static
inventory with one unrun GPU check.  This command keeps only that remaining
proof: load Cosmos Predict2.5 through VRL's production family builder, encode a
real video prefix, and verify that the upstream production-family pipeline can
construct non-empty, shape-compatible frame-prefix tensors.

This does not prove that VRL's wrapper or forward path consumes those tensors;
the wrapper still calls the pipeline with ``num_frames_in=0``.  It also does
not prove that an action-conditioned policy changes the generated continuation.
"""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from vrl import run
from vrl.config.loading import load_config
from vrl.config.precision import resolve_precision_policy
from vrl.config.schema import parse_config
from vrl.models.families.registry import get_model_family_entry
from vrl.scripts.eval._device import resolve_eval_device
from vrl.utils.artifacts import sha256_file
from vrl.utils.cuda_memory import release_cuda_memory
from vrl.utils.media import read_video_frames

REPORT_SCHEMA = "vrl.cosmos-predict25-frame-prefix-gate/v1"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="experiment/cosmos_predict2_5/online_nft_kling_video_reward",
        help="Cosmos Predict2.5 config used by the production family builder.",
    )
    parser.add_argument("--prefix-video", type=Path, required=True)
    parser.add_argument(
        "--prefix-frames",
        type=int,
        choices=(1, 5),
        default=5,
        help="One or two Cosmos latent conditioning frames (1 or 5 pixel frames).",
    )
    parser.add_argument("--num-frames", type=int, default=0)
    parser.add_argument("--height", type=int, default=0)
    parser.add_argument("--width", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/gates/cosmos_predict25_frame_prefix.json"),
    )
    parser.add_argument("overrides", nargs="*", help="OmegaConf dotlist overrides.")
    return parser


def _sampling_dimension(args: argparse.Namespace, cfg: Any, name: str) -> int:
    override = int(getattr(args, name))
    if override > 0:
        return override
    value = OmegaConf.select(cfg, f"sampling.{name}", default=0)
    if int(value or 0) <= 0:
        raise ValueError(f"sampling.{name} must be positive or supplied on the CLI")
    return int(value)


def _prepare_prefix_video(
    path: Path,
    *,
    prefix_frames: int,
    num_frames: int,
    height: int,
    width: int,
) -> torch.Tensor:
    frames = read_video_frames(path)
    if frames.shape[0] < prefix_frames:
        raise ValueError(
            f"prefix video has {frames.shape[0]} frames; need at least {prefix_frames}",
        )
    prefix = frames[-prefix_frames:].permute(0, 3, 1, 2)
    prefix = F.interpolate(prefix, size=(height, width), mode="bilinear", align_corners=False)
    prefix = prefix.permute(1, 0, 2, 3).unsqueeze(0)
    if prefix_frames < num_frames:
        prefix = torch.cat(
            [prefix, prefix[:, :, -1:].repeat(1, 1, num_frames - prefix_frames, 1, 1)],
            dim=2,
        )
    return prefix.mul(2.0).sub(1.0)


def run_gate(args: argparse.Namespace) -> dict[str, Any]:
    prefix_path = args.prefix_video.expanduser().resolve()
    if not prefix_path.is_file():
        raise FileNotFoundError(f"prefix video does not exist: {prefix_path}")

    cfg = load_config(args.config, overrides=args.overrides)
    root = parse_config(cfg)
    if root.model is None:
        raise ValueError("frame-prefix gate requires model configuration")
    entry = get_model_family_entry(str(root.model.family))
    if entry.family != "cosmos-predict2.5":
        raise ValueError(
            f"frame-prefix gate requires family 'cosmos-predict2.5'; got {entry.family!r}",
        )

    height = _sampling_dimension(args, cfg, "height")
    width = _sampling_dimension(args, cfg, "width")
    num_frames = _sampling_dimension(args, cfg, "num_frames")
    if num_frames < args.prefix_frames:
        raise ValueError("--num-frames must be >= --prefix-frames")
    video = _prepare_prefix_video(
        prefix_path,
        prefix_frames=int(args.prefix_frames),
        num_frames=num_frames,
        height=height,
        width=width,
    )

    device = resolve_eval_device(str(args.device))
    precision = resolve_precision_policy(root)
    resolved = run.resolve_model(entry, root, device, precision=precision, for_rollout=True)
    bundle = None
    try:
        bundle = resolved.materialize(context="Cosmos Predict2.5 frame-prefix gate")
        model = bundle.model.eval()
        pipe = model.pipeline
        video = video.to(device=device, dtype=pipe.vae.dtype)
        num_channels = int(pipe.transformer.config.in_channels) - 1
        latents, cond_latent, cond_mask, cond_indicator = pipe.prepare_latents(
            video=video,
            batch_size=1,
            num_channels_latents=num_channels,
            height=height,
            width=width,
            num_frames_in=int(args.prefix_frames),
            num_frames_out=num_frames,
            do_classifier_free_guidance=False,
            dtype=torch.float32,
            device=device,
            generator=torch.Generator(device="cpu").manual_seed(0),
            latents=None,
        )
        if cond_latent.shape != latents.shape:
            raise RuntimeError(
                "frame-prefix latent shape does not match denoise latent shape: "
                f"condition={tuple(cond_latent.shape)} denoise={tuple(latents.shape)}",
            )
        if cond_mask.shape[2:] != latents.shape[2:]:
            raise RuntimeError(
                "frame-prefix mask shape does not match denoise latent shape: "
                f"mask={tuple(cond_mask.shape)} denoise={tuple(latents.shape)}",
            )
        if not bool(cond_mask.any().item()) or not bool(cond_indicator.any().item()):
            raise RuntimeError("real frame prefix produced an empty conditioning mask")
        conditioned = cond_mask * cond_latent + (1 - cond_mask) * latents
        if torch.equal(conditioned, latents):
            raise RuntimeError("frame-prefix tensors do not alter the mixed latent tensor")

        return {
            "schema": REPORT_SCHEMA,
            "status": "passed",
            "config": str(args.config),
            "model_identity": resolved.identity,
            "prefix_video": {
                "path": str(prefix_path),
                "sha256": sha256_file(prefix_path),
                "conditioned_pixel_frames": int(args.prefix_frames),
            },
            "sampling": {"height": height, "width": width, "num_frames": num_frames},
            "tensors": {
                "latents": list(latents.shape),
                "condition": list(cond_latent.shape),
                "mask": list(cond_mask.shape),
                "indicator": list(cond_indicator.shape),
                "conditioned_values": int(torch.count_nonzero(cond_mask).item()),
            },
            "claim": (
                "the production-family pipeline constructs non-empty, "
                "shape-compatible frame-prefix tensors"
            ),
            "non_goal": (
                "VRL wrapper/forward integration, action fidelity, and visible "
                "continuation response remain unproven"
            ),
        }
    finally:
        del bundle
        release_cuda_memory()


def _write_report(path: Path, report: dict[str, Any]) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        report = run_gate(args)
    except Exception as exc:
        report = {
            "schema": REPORT_SCHEMA,
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
            "trace": traceback.format_exc(limit=8),
        }
        _write_report(args.out, report)
        print(json.dumps(report, indent=2, sort_keys=True))
        raise SystemExit(2) from exc
    _write_report(args.out, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
