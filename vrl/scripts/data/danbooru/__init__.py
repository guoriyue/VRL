"""Compatibility facade for Danbooru-derived dataset builders.

Implementation ownership is split across metadata parsing, anatomy prompts,
safety prompts, image assets, and CLI composition. Existing callers can keep
importing this package exactly as they imported the former flat module.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from vrl.scripts.data.common import write_jsonl
from vrl.scripts.data.danbooru.anatomy import (
    PromptRow,
    anatomy_constraints,
    bucket_from_tags,
    build_anatomy_prompts,
    build_prompt_rows,
    is_anatomy_positive,
    prompt_anchor_texts,
    prompt_from_tags,
    split_prompt_rows,
    write_prompt_report,
)
from vrl.scripts.data.danbooru.assets import (
    build_positive_images,
    download_danbooru_images,
    hand_crop_rows,
    http_download,
    positive_image_rows,
    select_positive_targets,
)
from vrl.scripts.data.danbooru.cli import (
    build_default_manifests,
    manifest_setup_hints,
)
from vrl.scripts.data.danbooru.cli import (
    main as _main,
)
from vrl.scripts.data.danbooru.cli import (
    register as _register,
)

# Preserve named attributes from the flat module without widening its historical
# wildcard-import surface.
from vrl.scripts.data.danbooru.config import (  # noqa: F401
    ACTION_BUCKET_TAGS,
    ANATOMY_DIR,
    ANATOMY_EVAL_LIMIT,
    ANATOMY_EVAL_OUTPUT,
    ANATOMY_MIN_SCORE,
    ANATOMY_PREFERRED_MIN_SCORE,
    ANATOMY_PROMPT_STYLE,
    ANATOMY_REPORT_OUTPUT,
    ANATOMY_SEED,
    ANATOMY_TRAIN_LIMIT,
    ANATOMY_TRAIN_OUTPUT,
    ANIME_FETCH_IMAGES_COMMAND,
    ANIME_POSITIVES_COMMAND,
    ANIME_PROMPTS_COMMAND,
    ANIME_SAFETY_PROMPTS_COMMAND,
    ARM_TAGS,
    CLOTHING_TAGS,
    DANBOORU_METADATA_FILE,
    DANBOORU_REPO_ID,
    DEFAULT_BUCKET_WEIGHTS,
    DOMAIN,
    EXCLUDE_TAGS,
    FEET_TAGS,
    HAND_FOCUS_ALLOWED_EXCLUDE_TAGS,
    HAND_FOCUS_BUCKET_TAGS,
    HAND_TAGS,
    OUTPUT_DIR,
    POSE_TAGS,
    POSITIVE_IMAGE_LIMIT,
    POSITIVE_IMAGE_MIN_SCORE,
    POSITIVE_IMAGE_SOURCE,
    POSITIVE_IMAGES_OUTPUT,
    PROMPT_ANCHOR_TAGS,
    SAFETY_CANDIDATE_POOL_FACTOR,
    SAFETY_DIR,
    SAFETY_EVAL_LIMIT,
    SAFETY_EVAL_OUTPUT,
    SAFETY_EXCLUDED_TAGS,
    SAFETY_MIN_RISK_TAGS,
    SAFETY_MIN_SCORE,
    SAFETY_PROMPT_TAG_LIMIT,
    SAFETY_REPORT_OUTPUT,
    SAFETY_RISK_TAGS,
    SAFETY_SEED,
    SAFETY_TARGET_RATINGS,
    SAFETY_TRAIN_LIMIT,
    SAFETY_TRAIN_OUTPUT,
    SCENE_TAGS,
    SUBJECT_PROMPT_TAGS,
    SUBJECT_TAGS,
    TEMPLATE_ID,
)
from vrl.scripts.data.danbooru.metadata import (
    download_metadata_file,
    iter_metadata,
    normalize_tags,
    record_id,
    record_score,
)
from vrl.scripts.data.danbooru.safety import (  # noqa: F401
    SAFETY_RATING_ALIASES,
    build_danbooru_safety_prompt_rows,
    build_safety_prompts,
    split_safety_prompt_rows,
    write_safety_report,
)

# Compatibility injection seam used by setup CLI tests and offline callers.
_http_download = http_download


def register(subparsers: Any) -> None:
    """Register the historical Danbooru commands on a setup CLI parser."""

    _register(
        subparsers,
        fetch=lambda url, target: _http_download(url, target),
    )


def main(argv: Sequence[str] | None = None) -> None:
    """Run the historical flat-module CLI through the package composition."""

    _main(
        argv,
        fetch=lambda url, target: _http_download(url, target),
    )


__all__ = [
    "DEFAULT_BUCKET_WEIGHTS",
    "SAFETY_TARGET_RATINGS",
    "PromptRow",
    "anatomy_constraints",
    "bucket_from_tags",
    "build_anatomy_prompts",
    "build_danbooru_safety_prompt_rows",
    "build_default_manifests",
    "build_positive_images",
    "build_prompt_rows",
    "build_safety_prompts",
    "download_danbooru_images",
    "download_metadata_file",
    "hand_crop_rows",
    "http_download",
    "is_anatomy_positive",
    "iter_metadata",
    "main",
    "manifest_setup_hints",
    "normalize_tags",
    "positive_image_rows",
    "prompt_anchor_texts",
    "prompt_from_tags",
    "record_id",
    "record_score",
    "register",
    "select_positive_targets",
    "split_prompt_rows",
    "split_safety_prompt_rows",
    "write_jsonl",
    "write_prompt_report",
    "write_safety_report",
]
