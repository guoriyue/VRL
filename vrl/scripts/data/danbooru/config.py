"""Danbooru dataset defaults and taxonomy projections.

The editable vocabulary remains in ``manifests/danbooru/config.yaml``. This
module loads that one source of truth and exposes the concrete runtime shapes
used by the safety prompt builder and the CLI.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from vrl.config.base import ConfigBase
from vrl.scripts.data.common import repo_root

ANIME_SAFETY_PROMPTS_COMMAND = "anime-safety-prompts"

OUTPUT_DIR = repo_root() / "manifests" / "danbooru"
SAFETY_DIR = OUTPUT_DIR / "safety"

DANBOORU_REPO_ID = "nyanko7/danbooru2023"
DANBOORU_METADATA_FILE = "metadata/posts.tar.gz"

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


class _TagVocabulary(ConfigBase):
    """``tags`` block of datasets/danbooru/config.yaml: prompt vocabulary shared by builders."""

    subject_tags: dict[str, Any]
    pose_tags: dict[str, Any]
    clothing_tags: list[str]
    scene_tags: list[str]
    prompt_anchor_tags: list[str]


class _SafetyTaxonomy(ConfigBase):
    """``safety`` block of manifests/danbooru/config.yaml."""

    target_ratings: list[str]
    excluded_tags: list[str]
    risk_tags: list[str]


class _Taxonomy(ConfigBase):
    tags: _TagVocabulary
    safety: _SafetyTaxonomy


def _load_taxonomy(path: Path) -> _Taxonomy:
    """Parse the tag taxonomy through the same closed-section contract as configs."""

    with path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    try:
        return _Taxonomy.model_validate(raw)
    except ValidationError as exc:
        from vrl.config.base import _extract_error_message

        raise ValueError(f"{path}: {_extract_error_message(exc)}") from exc


_TAXONOMY_PATH = OUTPUT_DIR / "config.yaml"
_TAXONOMY = _load_taxonomy(_TAXONOMY_PATH)
_TAGS = _TAXONOMY.tags
_SAFETY = _TAXONOMY.safety

SUBJECT_TAGS = dict(_TAGS.subject_tags)
SUBJECT_PROMPT_TAGS = tuple(SUBJECT_TAGS)
POSE_TAGS = dict(_TAGS.pose_tags)
CLOTHING_TAGS = tuple(_TAGS.clothing_tags)
SCENE_TAGS = tuple(_TAGS.scene_tags)
PROMPT_ANCHOR_TAGS = tuple(_TAGS.prompt_anchor_tags)

SAFETY_TARGET_RATINGS = tuple(_SAFETY.target_ratings)
SAFETY_EXCLUDED_TAGS = set(_SAFETY.excluded_tags)
SAFETY_RISK_TAGS = set(_SAFETY.risk_tags)
