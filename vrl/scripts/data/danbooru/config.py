"""Danbooru dataset defaults and taxonomy projections.

The editable vocabulary remains in ``datasets/danbooru/config.yaml``. This
module loads that one source of truth and exposes the concrete runtime shapes
used by the anatomy, safety, asset, and CLI owners.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError, model_validator

from vrl.config.base import ConfigBase
from vrl.scripts.data.common import repo_root

ANIME_PROMPTS_COMMAND = "anime-prompts"
ANIME_SAFETY_PROMPTS_COMMAND = "anime-safety-prompts"
ANIME_POSITIVES_COMMAND = "anime-positives"
ANIME_FETCH_IMAGES_COMMAND = "anime-fetch-images"

OUTPUT_DIR = repo_root() / "datasets" / "danbooru"
ANATOMY_DIR = OUTPUT_DIR / "anatomy"
SAFETY_DIR = OUTPUT_DIR / "safety"

DANBOORU_REPO_ID = "nyanko7/danbooru2023"
DANBOORU_METADATA_FILE = "metadata/posts.tar.gz"

ANATOMY_TRAIN_OUTPUT = ANATOMY_DIR / "train_prompts.jsonl"
ANATOMY_EVAL_OUTPUT = ANATOMY_DIR / "eval_prompts.jsonl"
ANATOMY_REPORT_OUTPUT = ANATOMY_DIR / "prompt_report.json"
ANATOMY_TRAIN_LIMIT = 20_000
ANATOMY_EVAL_LIMIT = 1_000
ANATOMY_MIN_SCORE = 5.0
ANATOMY_PREFERRED_MIN_SCORE = 20.0
ANATOMY_PROMPT_STYLE = "mixed"
ANATOMY_SEED = 0

SAFETY_TRAIN_OUTPUT = SAFETY_DIR / "train.jsonl"
SAFETY_EVAL_OUTPUT = SAFETY_DIR / "eval_baseline.jsonl"
SAFETY_REPORT_OUTPUT = SAFETY_DIR / "report.json"
SAFETY_TRAIN_LIMIT = 2_000
SAFETY_EVAL_LIMIT = 200
SAFETY_MIN_SCORE = 0.0
SAFETY_CANDIDATE_POOL_FACTOR = 3
SAFETY_PROMPT_TAG_LIMIT = 24
SAFETY_MIN_RISK_TAGS = 1
SAFETY_SEED = 0

POSITIVE_IMAGES_OUTPUT = ANATOMY_DIR / "positive_images.jsonl"
POSITIVE_IMAGE_SOURCE = "danbooru2023"
POSITIVE_IMAGE_MIN_SCORE = 20.0
POSITIVE_IMAGE_LIMIT = 10_000

TEMPLATE_ID = "anime_anatomy_v1"
DOMAIN = "anime"


class _AnatomyTaxonomy(ConfigBase):
    """``anatomy`` block of datasets/danbooru/config.yaml."""

    exclude_tags: list[str]
    hand_focus_allowed_exclude_tags: list[str]
    subject_tags: dict[str, Any]
    pose_tags: dict[str, Any]
    hand_tags: list[str]
    hand_focus_bucket_tags: list[str]
    feet_tags: list[str]
    arm_tags: list[str]
    clothing_tags: list[str]
    scene_tags: list[str]
    action_bucket_tags: list[str]
    default_bucket_weights: dict[str, float]
    prompt_anchor_tags: list[str]

    @model_validator(mode="after")
    def _hand_focus_buckets_are_hand_tags(self) -> _AnatomyTaxonomy:
        stray = sorted(set(self.hand_focus_bucket_tags) - set(self.hand_tags))
        if stray:
            raise ValueError(
                "anatomy.hand_focus_bucket_tags contains tags absent from "
                f"anatomy.hand_tags: {stray}",
            )
        return self


class _SafetyTaxonomy(ConfigBase):
    """``safety`` block of datasets/danbooru/config.yaml."""

    target_ratings: list[str]
    excluded_tags: list[str]
    risk_tags: list[str]


class _Taxonomy(ConfigBase):
    anatomy: _AnatomyTaxonomy
    safety: _SafetyTaxonomy


def _load_taxonomy(path: Path) -> _Taxonomy:
    """Parse the tag taxonomy through the same closed-section contract as configs."""

    with path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    try:
        return _Taxonomy.model_validate(raw)
    except ValidationError as exc:
        from vrl.config.schema import _extract_error_message

        raise ValueError(f"{path}: {_extract_error_message(exc)}") from exc


_TAXONOMY_PATH = OUTPUT_DIR / "config.yaml"
_TAXONOMY = _load_taxonomy(_TAXONOMY_PATH)
_ANATOMY = _TAXONOMY.anatomy
_SAFETY = _TAXONOMY.safety

EXCLUDE_TAGS = set(_ANATOMY.exclude_tags)
HAND_FOCUS_ALLOWED_EXCLUDE_TAGS = set(_ANATOMY.hand_focus_allowed_exclude_tags)

SUBJECT_TAGS = dict(_ANATOMY.subject_tags)
SUBJECT_PROMPT_TAGS = tuple(SUBJECT_TAGS)
POSE_TAGS = dict(_ANATOMY.pose_tags)
HAND_TAGS = set(_ANATOMY.hand_tags)
HAND_FOCUS_BUCKET_TAGS = set(_ANATOMY.hand_focus_bucket_tags)
FEET_TAGS = set(_ANATOMY.feet_tags)
ARM_TAGS = set(_ANATOMY.arm_tags)
CLOTHING_TAGS = tuple(_ANATOMY.clothing_tags)
SCENE_TAGS = tuple(_ANATOMY.scene_tags)
ACTION_BUCKET_TAGS = set(_ANATOMY.action_bucket_tags)
DEFAULT_BUCKET_WEIGHTS = dict(_ANATOMY.default_bucket_weights)
PROMPT_ANCHOR_TAGS = tuple(_ANATOMY.prompt_anchor_tags)

SAFETY_TARGET_RATINGS = tuple(_SAFETY.target_ratings)
SAFETY_EXCLUDED_TAGS = set(_SAFETY.excluded_tags)
SAFETY_RISK_TAGS = set(_SAFETY.risk_tags)
