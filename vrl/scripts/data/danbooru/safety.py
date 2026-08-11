"""Safety prompt dataset construction from normalized Danbooru metadata."""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from vrl.scripts.data.common import dedupe_text, write_jsonl
from vrl.scripts.data.danbooru.config import (
    CLOTHING_TAGS,
    DANBOORU_METADATA_FILE,
    DANBOORU_REPO_ID,
    POSE_TAGS,
    PROMPT_ANCHOR_TAGS,
    SAFETY_CANDIDATE_POOL_FACTOR,
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
)
from vrl.scripts.data.danbooru.manifest_rows import (
    interleave_manifest_rows,
    metadata_counts,
    proportional_group_counts,
)
from vrl.scripts.data.danbooru.metadata import (
    iter_metadata,
    normalize_tags,
    record_id,
    record_score,
    resolve_metadata_path,
)

# Danbooru metadata writes ratings as a single letter. This protocol projection
# is deliberately separate from the editable prompt taxonomy in config.yaml.
_SAFETY_RATING_SPELLINGS = {
    "general": ("g", "safe"),
    "sensitive": ("s",),
    "questionable": ("q",),
    "explicit": ("e",),
}
SAFETY_RATING_ALIASES = {
    spelling: canonical
    for canonical, spellings in _SAFETY_RATING_SPELLINGS.items()
    for spelling in (canonical, *spellings)
}


def build_safety_prompts(
    *,
    metadata: str | Path | None = None,
    download_metadata: bool = False,
    hf_repo: str = DANBOORU_REPO_ID,
    hf_file: str = DANBOORU_METADATA_FILE,
    hf_cache_dir: str | Path | None = None,
    train_output: str | Path = SAFETY_TRAIN_OUTPUT,
    eval_output: str | Path = SAFETY_EVAL_OUTPUT,
    report_output: str | Path = SAFETY_REPORT_OUTPUT,
    train_limit: int = SAFETY_TRAIN_LIMIT,
    eval_limit: int = SAFETY_EVAL_LIMIT,
    ratings: Sequence[str] = SAFETY_TARGET_RATINGS,
    min_score: float = SAFETY_MIN_SCORE,
    seed: int = SAFETY_SEED,
    candidate_pool_factor: int = SAFETY_CANDIDATE_POOL_FACTOR,
    max_metadata_rows: int | None = None,
    prompt_tag_limit: int = SAFETY_PROMPT_TAG_LIMIT,
    min_risk_tags: int = SAFETY_MIN_RISK_TAGS,
    allow_partial: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    metadata_path = resolve_metadata_path(
        metadata=metadata,
        download_metadata=download_metadata,
        hf_repo=hf_repo,
        hf_file=hf_file,
        hf_cache_dir=hf_cache_dir,
    )
    target_count = train_limit + eval_limit
    rows = build_danbooru_safety_prompt_rows(
        metadata_path,
        ratings=ratings,
        min_score=min_score,
        limit=target_count,
        seed=seed,
        candidate_limit=max(target_count, target_count * candidate_pool_factor),
        max_metadata_rows=max_metadata_rows,
        prompt_tag_limit=prompt_tag_limit,
        min_risk_tags=min_risk_tags,
    )
    train_rows, eval_rows = split_safety_prompt_rows(
        rows,
        train_limit=train_limit,
        eval_limit=eval_limit,
    )
    if not allow_partial and (len(train_rows) < train_limit or len(eval_rows) < eval_limit):
        raise RuntimeError(
            "Not enough safety prompt rows generated: "
            f"train={len(train_rows)}/{train_limit}, "
            f"eval={len(eval_rows)}/{eval_limit}. "
            "Increase max_metadata_rows, lower filters, or set allow_partial=True.",
        )
    write_jsonl(train_output, train_rows, sort_keys=False)
    write_jsonl(eval_output, eval_rows, sort_keys=False)
    if report_output:
        write_safety_report(report_output, train_rows, eval_rows)
    return train_rows, eval_rows


def build_danbooru_safety_prompt_rows(
    metadata_path: str | Path,
    *,
    ratings: Sequence[str] = SAFETY_TARGET_RATINGS,
    min_score: float = 0.0,
    limit: int | None = None,
    seed: int = 0,
    candidate_limit: int | None = None,
    max_metadata_rows: int | None = None,
    prompt_tag_limit: int = 24,
    min_risk_tags: int = 1,
) -> list[dict[str, Any]]:
    target_ratings = {_normalize_rating(rating) for rating in ratings}
    target_ratings.discard(None)
    if not target_ratings:
        raise ValueError("at least one valid safety rating is required")

    candidates: list[dict[str, Any]] = []
    scanned = 0
    seen_prompts: set[str] = set()
    for row in iter_metadata(metadata_path):
        scanned += 1
        if max_metadata_rows is not None and scanned > max_metadata_rows:
            break
        tags = normalize_tags(row)
        tag_set = set(tags)
        if not _is_active_post(row):
            continue
        if tag_set & SAFETY_EXCLUDED_TAGS:
            continue
        rating = _record_rating(row, tag_set)
        if rating not in target_ratings:
            continue
        if record_score(row) < min_score:
            continue
        nsfw_tags = _safety_nsfw_tags(tags, rating=rating)
        risk_tags = [tag for tag in nsfw_tags if not tag.startswith("rating:")]
        if len(risk_tags) < min_risk_tags:
            continue
        prompt = _safety_prompt_from_tags(
            tags,
            rating=rating,
            nsfw_tags=nsfw_tags,
            prompt_tag_limit=prompt_tag_limit,
        )
        if not prompt or prompt in seen_prompts:
            continue
        seen_prompts.add(prompt)
        candidates.append(
            {
                "prompt": prompt,
                "metadata": {
                    "category": f"danbooru_{rating}",
                    "source_post_ids": [record_id(row)] if record_id(row) is not None else [],
                    "source_tags": tags,
                    "source_score": record_score(row),
                    "rating": rating,
                    "nsfw_tags": nsfw_tags,
                },
            },
        )
        if candidate_limit is not None and len(candidates) >= candidate_limit:
            break

    rng = random.Random(seed)
    by_rating: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        by_rating[str(row["metadata"]["rating"])].append(row)
    for rows in by_rating.values():
        rng.shuffle(rows)
    return interleave_manifest_rows(by_rating, limit=limit or len(candidates))


def split_safety_prompt_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    train_limit: int,
    eval_limit: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        metadata = row.get("metadata") or {}
        key = str(metadata.get("rating") or metadata.get("category") or "unknown")
        groups[key].append(dict(row))

    eval_counts = proportional_group_counts(
        {key: len(value) for key, value in groups.items()},
        limit=eval_limit,
    )
    train_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    eval_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for key, group_rows in groups.items():
        eval_count = eval_counts.get(key, 0)
        eval_groups[key].extend(group_rows[:eval_count])
        train_groups[key].extend(group_rows[eval_count:])
    return (
        interleave_manifest_rows(train_groups, limit=train_limit),
        interleave_manifest_rows(eval_groups, limit=eval_limit),
    )


def write_safety_report(
    path: str | Path,
    train_rows: Sequence[Mapping[str, Any]],
    eval_rows: Sequence[Mapping[str, Any]],
) -> None:
    report = {
        "train_count": len(train_rows),
        "eval_count": len(eval_rows),
        "train_categories": metadata_counts(train_rows, key="category"),
        "eval_categories": metadata_counts(eval_rows, key="category"),
        "train_ratings": metadata_counts(train_rows, key="rating", default="unrated"),
        "eval_ratings": metadata_counts(eval_rows, key="rating", default="unrated"),
        "train_nsfw_tags_top": _nsfw_tag_counts(train_rows, limit=50),
        "eval_nsfw_tags_top": _nsfw_tag_counts(eval_rows, limit=50),
    }
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _nsfw_tag_counts(rows: Sequence[Mapping[str, Any]], *, limit: int) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for row in rows:
        metadata = row.get("metadata") or {}
        for tag in metadata.get("nsfw_tags", []) or []:
            counter[str(tag)] += 1
    return dict(counter.most_common(limit))


def _is_active_post(row: Mapping[str, Any]) -> bool:
    return not any(bool(row.get(key)) for key in ("is_deleted", "is_flagged", "is_banned"))


def _normalize_rating(value: Any) -> str | None:
    text = str(value or "").strip().lower()
    return SAFETY_RATING_ALIASES.get(text)


def _record_rating(row: Mapping[str, Any], tags: set[str]) -> str | None:
    rating = _normalize_rating(row.get("rating"))
    if rating is not None:
        return rating
    for tag in tags:
        if not tag.startswith("rating:"):
            continue
        rating = _normalize_rating(tag.split(":", 1)[1])
        if rating is not None:
            return rating
    return None


def _safety_nsfw_tags(tags: Sequence[str], *, rating: str) -> list[str]:
    out = [f"rating:{rating}"]
    out.extend(tag for tag in tags if tag in SAFETY_RISK_TAGS)
    return dedupe_text(out)


def _safety_prompt_from_tags(
    tags: Sequence[str],
    *,
    rating: str,
    nsfw_tags: Sequence[str],
    prompt_tag_limit: int,
) -> str:
    tag_set = set(tags)
    parts: list[str] = [f"rating:{rating}", "adult"]
    for tag in SUBJECT_PROMPT_TAGS:
        if tag in tag_set:
            parts.append(tag)
            break
    if "solo" in tag_set:
        parts.append("solo")
    parts.extend(nsfw_tags)

    context_tags = tuple(POSE_TAGS) + PROMPT_ANCHOR_TAGS + CLOTHING_TAGS + SCENE_TAGS
    for tag in context_tags:
        if tag in tag_set:
            parts.append(tag)
    cleaned = dedupe_text(parts)
    return ", ".join(cleaned[: max(1, prompt_tag_limit)])
