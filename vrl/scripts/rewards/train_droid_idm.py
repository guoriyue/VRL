"""Train the vision-only DROID IDM that backs the action_following reward.

SPRINT_idm_action_following_reward Phase 2. Supervision comes straight from
the droid_targets manifests (``target_video`` mp4 + 15Hz frame-aligned
``metadata.target_actions``), so the IDM learns "frame pair -> that label
vector" in whatever convention the manifest recorded — the reward compares in
the same space, so the exact DROID action semantics never need pinning.

The checkpoint is self-describing (arch width, action normalization, input
size); ``vrl.rewards.models.action_following.load_idm_checkpoint`` is the one
reader. Rebuild data with ``python -m vrl.scripts.data.setup
video-world-targets`` if the manifests are missing.

Run:
    python -m vrl.scripts.rewards.train_droid_idm \
        --out outputs/idm/droid_idm.pt --epochs 200
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from vrl.rewards.models.action_following import FramePairIDM, frame_pairs_from_clip
from vrl.utils.media import read_video_frames

_DEFAULT_MANIFEST_DIR = Path("data/external/video_world/manifests")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--train-manifest", type=Path, default=_DEFAULT_MANIFEST_DIR / "droid_targets_train.jsonl"
    )
    parser.add_argument(
        "--eval-manifest", type=Path, default=_DEFAULT_MANIFEST_DIR / "droid_targets_eval.jsonl"
    )
    parser.add_argument("--out", type=Path, default=Path("outputs/idm/droid_idm.pt"))
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--width", type=int, default=64, help="conv channel width")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--no-augment",
        dest="augment",
        action="store_false",
        help="disable random crop + brightness augmentation (both action-safe; "
        "horizontal flip is NOT — it would mirror the x/yaw action axes)",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    return parser


def load_pair_dataset(
    manifest: Path,
    *,
    image_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """All (frame pair, action) examples of one manifest, fully in memory.

    46 clips x 32 pairs at 128px is ~1.5 GB fp32 — small enough that decoding
    once beats a streaming loader, and identical epochs make overfitting
    diagnostics readable.
    """

    pairs: list[torch.Tensor] = []
    actions: list[torch.Tensor] = []
    root = manifest.parent.parent.parent  # data root: manifests/ -> video_world/ -> external/
    for line in manifest.read_text().splitlines():
        row = json.loads(line)
        raw = row.get("metadata", {}).get("target_actions")
        if not raw:
            continue
        video_path = root / row["target_video"]
        frames = read_video_frames(video_path)
        clip_pairs = frame_pairs_from_clip(frames, image_size=image_size)
        labels = torch.as_tensor(raw, dtype=torch.float32)
        count = min(clip_pairs.shape[0], labels.shape[0])
        pairs.append(clip_pairs[:count])
        actions.append(labels[:count])
    if not pairs:
        raise SystemExit(f"no rows with target_actions in {manifest}")
    return torch.cat(pairs), torch.cat(actions)


def _augment_pairs(x: torch.Tensor) -> torch.Tensor:
    """Action-safe augmentation: shared random crop + brightness per pair.

    Both frames of a pair get the SAME crop/brightness so the motion between
    them is preserved; flips would mirror the action axes and are excluded.
    """

    batch, _, size, _ = x.shape
    pad = max(4, size // 16)
    padded = torch.nn.functional.pad(x, (pad, pad, pad, pad), mode="replicate")
    out = torch.empty_like(x)
    offsets = torch.randint(0, 2 * pad + 1, (batch, 2), device=x.device)
    for i in range(batch):
        top, left = int(offsets[i, 0]), int(offsets[i, 1])
        out[i] = padded[i, :, top : top + size, left : left + size]
    scale = 0.8 + 0.4 * torch.rand(batch, 1, 1, 1, device=x.device)
    return (out * scale).clamp(0.0, 1.0)


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    train_x, train_a = load_pair_dataset(args.train_manifest, image_size=args.image_size)
    eval_x, eval_a = load_pair_dataset(args.eval_manifest, image_size=args.image_size)
    action_dim = int(train_a.shape[1])
    print(
        f"train pairs {tuple(train_x.shape)}  eval pairs {tuple(eval_x.shape)}  action_dim {action_dim}"
    )

    # Normalization is part of the checkpoint contract. Floor each dim's std
    # at 10% of the median std: a near-constant dim (droid_100's gripper is
    # ~always open, std 0.018 vs ~0.2 for the pose dims) would otherwise put
    # a 10-50x error amplifier on the one dim carrying the least signal and
    # dominate both the loss and the reward's exp(-mse) score.
    action_mean = train_a.mean(dim=0)
    raw_std = train_a.std(dim=0)
    action_std = raw_std.clamp_min(max(0.1 * float(raw_std.median()), 1e-4))
    train_z = (train_a - action_mean) / action_std
    eval_z = (eval_a - action_mean) / action_std

    model = FramePairIDM(action_dim=action_dim, width=args.width).to(device)
    total = sum(p.numel() for p in model.parameters())
    print(f"FramePairIDM parameters: {total / 1e6:.2f}M")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    eval_x_dev, eval_z_dev = eval_x.to(device), eval_z.to(device)
    best_eval = float("inf")
    args.out.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(train_x.shape[0])
        losses = []
        for start in range(0, len(perm), args.batch_size):
            idx = perm[start : start + args.batch_size]
            x = train_x[idx].to(device, non_blocking=True)
            z = train_z[idx].to(device, non_blocking=True)
            if args.augment:
                x = _augment_pairs(x)
            loss = torch.nn.functional.mse_loss(model(x), z)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss))
        model.eval()
        with torch.no_grad():
            eval_mse = float(torch.nn.functional.mse_loss(model(eval_x_dev), eval_z_dev))
        if eval_mse < best_eval:
            best_eval = eval_mse
            torch.save(
                {
                    "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                    "action_mean": action_mean,
                    "action_std": action_std,
                    "action_dim": action_dim,
                    "width": args.width,
                    "image_size": args.image_size,
                    "train_pairs": int(train_x.shape[0]),
                    "eval_mse_z": eval_mse,
                },
                args.out,
            )
        if epoch % 10 == 0 or epoch == args.epochs - 1:
            print(
                f"epoch {epoch:4d}  train_mse_z {sum(losses) / len(losses):.4f}  "
                f"eval_mse_z {eval_mse:.4f}  (predict-zero baseline = 1.0)"
            )

    with torch.no_grad():
        pred = model(eval_x_dev).cpu()
    per_dim_var = eval_z.var(dim=0).clamp_min(1e-8)
    r2 = 1.0 - ((pred - eval_z).pow(2).mean(dim=0) / per_dim_var)
    print("held-out per-dim R^2 (z-space):", [round(float(v), 3) for v in r2])
    print(f"best eval_mse_z {best_eval:.4f} -> {args.out}")


if __name__ == "__main__":
    main()
