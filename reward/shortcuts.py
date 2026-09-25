"""Shortcut candidates for edit rewards: outputs that do not do the task but might score.

``reward.stress`` perturbs pixels to see whether a scorer is fooled by damage;
this module builds the opposite probe for image-editing rewards, the cheap
ways a policy can *avoid* the task: hand the source back unchanged, shift or
re-crop the whole frame, or return an unrelated picture. Each is written as a
``reward_stress`` transform beside its baseline (the real candidate), so the
existing ``Analysis.stress`` report pairs them, and the reward card asks one
question per shortcut: how often did it score at or above the genuine output.

Task-specific shortcuts (an object copied instead of moved, text garbled) are
outside this generic set; add them as further transforms with the same
metadata contract when a task needs them.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from pathlib import Path

from PIL import Image

from vrl.rewards.evaluation import MediaRow, load_media_manifest
from vrl.utils.artifacts import sha256_file
from vrl.utils.json_files import canonical_json_sha256, write_json

SHIFT_SHARE = 0.03  # of the shorter side; the object-move probe's whole-frame shift
CROP_SHARE = 0.04  # margin cropped away before resizing back


def _unchanged_source(row: MediaRow, candidate: Image.Image, _: Sequence[MediaRow]) -> Image.Image:
    source_path = row.assets.get("reference_image")
    if source_path is None:
        raise ValueError(
            f"shortcut 'unchanged_source' needs a reference_image asset on {row.sample_id!r}"
        )
    with Image.open(source_path) as source:
        return source.convert("RGB").resize(candidate.size, Image.Resampling.LANCZOS)


def _shift_frame(_: MediaRow, candidate: Image.Image, __: Sequence[MediaRow]) -> Image.Image:
    width, height = candidate.size
    step = max(1, round(min(width, height) * SHIFT_SHARE))
    shifted = Image.new("RGB", candidate.size)
    shifted.paste(candidate, (step, step))
    # Fill the uncovered band by replicating the edge, as a resampled frame would.
    shifted.paste(candidate.crop((0, 0, width, 1)).resize((width, step)), (0, 0))
    shifted.paste(candidate.crop((0, 0, 1, height)).resize((step, height)), (0, 0))
    return shifted


def _crop_zoom(_: MediaRow, candidate: Image.Image, __: Sequence[MediaRow]) -> Image.Image:
    width, height = candidate.size
    dx, dy = round(width * CROP_SHARE), round(height * CROP_SHARE)
    return candidate.crop((dx, dy, width - dx, height - dy)).resize(
        candidate.size, Image.Resampling.LANCZOS
    )


def _other_scene(row: MediaRow, candidate: Image.Image, rows: Sequence[MediaRow]) -> Image.Image:
    if len(rows) < 2:
        raise ValueError("shortcut 'other_scene' needs at least two manifest rows")
    index = next(i for i, other in enumerate(rows) if other.sample_id == row.sample_id)
    with Image.open(rows[(index + 1) % len(rows)].path) as other:
        return other.convert("RGB").resize(candidate.size, Image.Resampling.LANCZOS)


SHORTCUTS: dict[str, Callable[[MediaRow, Image.Image, Sequence[MediaRow]], Image.Image]] = {
    "unchanged_source": _unchanged_source,
    "shift_frame": _shift_frame,
    "crop_zoom": _crop_zoom,
    "other_scene": _other_scene,
}


def build_shortcut_manifest(
    manifest: Path,
    output_dir: Path,
    *,
    shortcuts: Sequence[str] | None = None,
    seed: int = 42,
) -> Path:
    """Write the baseline candidate plus one image per shortcut for every row."""

    names = list(SHORTCUTS) if shortcuts is None else list(shortcuts)
    unknown = sorted(set(names) - set(SHORTCUTS))
    if unknown or not names:
        raise ValueError(f"unknown shortcuts {unknown}; known: {sorted(SHORTCUTS)}")
    rows = load_media_manifest(manifest)
    if any("reward_stress" in row.metadata for row in rows):
        raise ValueError(
            "reward_stress metadata is reserved; nested stress inputs are unsupported"
        )
    output_dir.mkdir(parents=True, exist_ok=False)
    images = output_dir / "images"
    images.mkdir()
    records = []
    for row in rows:
        source_hash = sha256_file(row.path)
        with Image.open(row.path) as opened:
            if getattr(opened, "n_frames", 1) != 1:
                raise ValueError("shortcut inputs must be single images")
            candidate = opened.convert("RGB")
        variants = {"baseline": candidate}
        for name in names:
            variants[name] = SHORTCUTS[name](row, candidate, rows)
        baseline_id = f"{row.sample_id}:baseline"
        for transform, image in variants.items():
            sample_id = f"{row.sample_id}:{transform}"
            path = images / f"{canonical_json_sha256(sample_id, allow_nan=False)}.png"
            image.save(path)
            records.append(
                {
                    "sample_id": sample_id,
                    "prompt_id": row.prompt_id,
                    "prompt": row.prompt,
                    "path": str(path.resolve()),
                    "assets": {key: str(value) for key, value in row.assets.items()},
                    "metadata": {
                        **row.metadata,
                        "reward_stress": {
                            "source_sample_id": row.sample_id,
                            "source_sha256": source_hash,
                            "baseline_id": baseline_id,
                            "transform": transform,
                            "seed": seed,
                        },
                    },
                }
            )
    path = output_dir / "media.jsonl"
    path.write_text(
        "".join(json.dumps(r, sort_keys=True, allow_nan=False) + "\n" for r in records)
    )
    write_json(
        output_dir / "recipe.json",
        {
            "seed": seed,
            "shortcuts": names,
            "shift_share": SHIFT_SHARE,
            "crop_share": CROP_SHARE,
        },
    )
    return path


__all__ = ["SHORTCUTS", "build_shortcut_manifest"]
