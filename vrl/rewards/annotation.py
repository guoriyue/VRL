"""Offline blinded review packets from content-bound scoring snapshots.

Pair selection and calibration/holdout assignment are explicit operator inputs.
The packet copies original media, never derives preference labels from scores.
"""

from __future__ import annotations

import json
import random
import shutil
import tempfile
from importlib.resources import files
from pathlib import Path
from typing import Any

from vrl.rewards.calibration import PreferencePair, validate_preferences
from vrl.utils.artifacts import sha256_file
from vrl.utils.json_files import canonical_json_sha256, write_json


def export_preference_review(
    evaluation: dict[str, Any],
    pairs: list[dict[str, Any]],
    output: Path,
    *,
    seed: int,
) -> dict[str, Any]:
    """Publish a portable local HTML review, with randomized order and A/B sides.

    A separate audit manifest binds display order to original sample identities.
    This is presentation blinding, not access control against inspecting files.
    """
    if type(seed) is not int or not pairs:
        raise ValueError("review needs an integer seed and non-empty pair specifications")
    if any("preference" in pair for pair in pairs):
        raise ValueError("review pair specifications must not contain preference labels")
    # Reuse the calibration identity/leakage gate without emitting placeholder labels.
    validated = [PreferencePair(**pair, preference="unsure") for pair in pairs]
    validate_preferences(evaluation, validated)
    if output.exists():
        raise FileExistsError(output)
    rng = random.Random(seed)
    rng.shuffle(validated)
    records = evaluation["records"]
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    media = stage / "media"
    media.mkdir()
    copied: dict[tuple[str, str], dict[str, str]] = {}

    def copy_media(path: str, digest: str) -> dict[str, str]:
        source = Path(path)
        extension = source.suffix.lower()
        if extension in {".png", ".jpg", ".jpeg", ".webp", ".gif"}:
            kind = "image"
        elif extension in {".mp4", ".webm"}:
            kind = "video"
        else:
            raise ValueError(f"unsupported browser review media extension: {extension}")
        if sha256_file(source) != digest:
            raise ValueError(f"review media changed since scoring: {source}")
        key = (digest, extension)
        if key not in copied:
            relative = f"media/{len(copied):06d}{extension}"
            shutil.copyfile(source, stage / relative)
            if sha256_file(stage / relative) != digest or sha256_file(source) != digest:
                raise ValueError(f"review media changed while copying: {source}")
            copied[key] = {"path": relative, "kind": kind}
        return copied[key]

    try:
        display, mapping = [], {}
        for index, pair in enumerate(validated):
            left = records[pair.left]["input"]
            right = records[pair.right]["input"]
            if left["prompt"] != right["prompt"]:
                raise ValueError("review pairs must share exact prompt text")
            if left.get("assets", {}) != right.get("assets", {}) or left.get(
                "asset_sha256", {}
            ) != right.get("asset_sha256", {}):
                raise ValueError("review pairs must share auxiliary inputs")
            first, second = (copy_media(item["path"], item["sha256"]) for item in (left, right))
            reverse = bool(rng.getrandbits(1))
            if reverse:
                first, second = second, first
            review_key = f"pair-{index:06d}"
            reference = left.get("assets", {}).get("reference_image")
            display.append(
                {
                    "key": review_key,
                    "prompt": left["prompt"],
                    "dimension": pair.dimension,
                    "a": first,
                    "b": second,
                    "reference": copy_media(reference, left["asset_sha256"]["reference_image"])
                    if reference
                    else None,
                }
            )
            mapping[review_key] = {
                "pair": pair.model_dump(mode="json", exclude={"preference"}),
                "reversed": reverse,
                "left_sha256": left["sha256"],
                "right_sha256": right["sha256"],
                "asset_sha256": left.get("asset_sha256", {}),
            }
        manifest = {
            "schema": "vrl.preference-review.v1",
            "run_id": evaluation["run_id"],
            "seed": seed,
            "mapping": mapping,
            "display": display,
        }
        manifest = {"review_id": canonical_json_sha256(manifest, allow_nan=False), **manifest}
        write_json(stage / "audit.json", manifest)
        public = json.dumps(
            {"review_id": manifest["review_id"], "pairs": display}, ensure_ascii=True
        ).replace("<", "\\u003c")
        template = files("vrl.rewards.assets").joinpath("preference_review.html").read_text()
        (stage / "index.html").write_text(template.replace("__REVIEW_DATA__", public))
        stage.rename(output)
        return {"review_id": manifest["review_id"], "pairs": len(display), "output": str(output)}
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def import_preference_review(
    manifest: dict[str, Any], annotations: dict[str, Any]
) -> list[PreferencePair]:
    """Map explicitly answered A/B choices to original calibration sample IDs."""
    payload = {key: value for key, value in manifest.items() if key != "review_id"}
    if manifest.get("schema") != "vrl.preference-review.v1" or canonical_json_sha256(
        payload, allow_nan=False
    ) != manifest.get("review_id"):
        raise ValueError("review manifest schema or digest mismatch")
    if (
        set(annotations) != {"review_id", "answers"}
        or annotations["review_id"] != manifest["review_id"]
    ):
        raise ValueError("annotations belong to a different or invalid review")
    answers = annotations["answers"]
    if not isinstance(answers, dict) or not answers:
        raise ValueError("review has no explicit answers")
    pairs = []
    for key, answer in answers.items():
        if key not in manifest["mapping"] or answer not in ("a", "b", "tie", "unsure"):
            raise ValueError("unknown review pair or answer")
        entry = manifest["mapping"][key]
        preference = answer
        if answer in ("a", "b"):
            preference = "left" if (answer == "a") != entry["reversed"] else "right"
        pairs.append(PreferencePair(**entry["pair"], preference=preference))
    return pairs
