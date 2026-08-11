"""Anatomy-focused prompt construction from normalized Danbooru metadata."""

from __future__ import annotations

import json
import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vrl.scripts.data.common import dedupe_text, write_jsonl
from vrl.scripts.data.danbooru.config import (
    ACTION_BUCKET_TAGS,
    ANATOMY_EVAL_LIMIT,
    ANATOMY_EVAL_OUTPUT,
    ANATOMY_MIN_SCORE,
    ANATOMY_PREFERRED_MIN_SCORE,
    ANATOMY_PROMPT_STYLE,
    ANATOMY_REPORT_OUTPUT,
    ANATOMY_SEED,
    ANATOMY_TRAIN_LIMIT,
    ANATOMY_TRAIN_OUTPUT,
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
    POSE_TAGS,
    PROMPT_ANCHOR_TAGS,
    SCENE_TAGS,
    SUBJECT_PROMPT_TAGS,
    SUBJECT_TAGS,
    TEMPLATE_ID,
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


@dataclass(frozen=True, slots=True)
class PromptRow:
    prompt: str
    metadata: dict[str, Any]


def build_anatomy_prompts(
    *,
    metadata: str | Path | None = None,
    download_metadata: bool = False,
    hf_repo: str = DANBOORU_REPO_ID,
    hf_file: str = DANBOORU_METADATA_FILE,
    hf_cache_dir: str | Path | None = None,
    train_output: str | Path = ANATOMY_TRAIN_OUTPUT,
    eval_output: str | Path = ANATOMY_EVAL_OUTPUT,
    report_output: str | Path = ANATOMY_REPORT_OUTPUT,
    train_limit: int = ANATOMY_TRAIN_LIMIT,
    eval_limit: int = ANATOMY_EVAL_LIMIT,
    min_score: float = ANATOMY_MIN_SCORE,
    preferred_min_score: float | None = ANATOMY_PREFERRED_MIN_SCORE,
    seed: int = ANATOMY_SEED,
    prompt_style: str = ANATOMY_PROMPT_STYLE,
    max_metadata_rows: int | None = None,
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
    rows = build_prompt_rows(
        metadata_path,
        min_score=min_score,
        preferred_min_score=preferred_min_score,
        limit=target_count,
        seed=seed,
        max_metadata_rows=max_metadata_rows,
        prompt_style=prompt_style,
    )
    train_rows, eval_rows = split_prompt_rows(
        rows,
        train_limit=train_limit,
        eval_limit=eval_limit,
    )
    if not allow_partial and (len(train_rows) < train_limit or len(eval_rows) < eval_limit):
        raise RuntimeError(
            "Not enough prompt rows generated: "
            f"train={len(train_rows)}/{train_limit}, "
            f"eval={len(eval_rows)}/{eval_limit}. "
            "Increase max_metadata_rows, lower filters, or set allow_partial=True.",
        )
    write_jsonl(train_output, train_rows)
    write_jsonl(eval_output, eval_rows)
    write_prompt_report(report_output, train_rows, eval_rows)
    return train_rows, eval_rows


def is_anatomy_positive(
    tags: set[str],
    *,
    min_score: float,
    score: float,
    allow_hand_focus: bool = False,
) -> bool:
    if score < min_score:
        return False
    allowed_excludes = HAND_FOCUS_ALLOWED_EXCLUDE_TAGS if allow_hand_focus else set()
    if (tags & EXCLUDE_TAGS) - allowed_excludes:
        return False
    if "solo" not in tags:
        return False
    if not ({"1girl", "1boy"} & tags):
        return False
    has_full_body = "full_body" in tags
    has_hand_focus = allow_hand_focus and "hand_focus" in tags and bool(tags & HAND_TAGS)
    if not has_full_body and not has_hand_focus:
        return False
    return bool(tags & (set(POSE_TAGS) | HAND_TAGS | FEET_TAGS | ARM_TAGS))


def prompt_from_tags(
    row: Mapping[str, Any],
    tags: Sequence[str],
    *,
    prompt_style: str = "tag",
) -> PromptRow | None:
    tag_set = set(tags)
    subject = _first_subject(tag_set)
    if subject is None:
        return None

    bucket = bucket_from_tags(tag_set)
    pose = _first_mapped(tag_set, POSE_TAGS) or "standing"
    framing = "full body" if "full_body" in tag_set else "upper body"
    constraints = anatomy_constraints(tag_set, bucket)
    anchors = prompt_anchor_texts(tags, limit=4)
    clothing = _first_tag_text(tag_set, CLOTHING_TAGS)
    scene = _first_tag_text(tag_set, SCENE_TAGS) or "simple background"

    if prompt_style == "tag":
        parts = _tag_prompt_parts(
            tag_set,
            framing=framing,
            pose=pose,
            constraints=constraints,
            anchors=anchors,
            clothing=clothing,
            scene=scene,
        )
    elif prompt_style == "language":
        parts = [subject, framing, pose, *constraints, *anchors]
    else:
        raise ValueError("prompt_style must be 'tag' or 'language'")
    if clothing:
        parts.append(clothing)
    if scene:
        parts.append(scene)
    if prompt_style == "language":
        parts.append("detailed anime illustration")
    prompt = ", ".join(dedupe_text(parts))

    source_id = record_id(row)
    source_post_ids = [source_id] if source_id is not None else []
    metadata = {
        "bucket": bucket,
        "source": "danbooru_tags",
        "template_id": TEMPLATE_ID,
        "source_post_ids": source_post_ids,
        "source_tags": list(tags),
        "source_score": record_score(row),
        "framing": framing,
        "constraints": constraints,
        "prompt_style": prompt_style,
        "domain": DOMAIN,
    }
    return PromptRow(prompt=prompt, metadata=metadata)


def bucket_from_tags(tags: set[str]) -> str:
    if tags & HAND_FOCUS_BUCKET_TAGS:
        return "hand_focus"
    if "running" in tags:
        return "running"
    if "walking" in tags:
        return "walking"
    if "kneeling" in tags:
        return "kneeling"
    if "sitting" in tags:
        return "sitting_full_body"
    if tags & ACTION_BUCKET_TAGS:
        return "action_pose"
    if tags & FEET_TAGS:
        return "feet_visible"
    if tags & HAND_TAGS:
        return "hands_visible"
    if tags & ARM_TAGS:
        return "arms_visible"
    if {"side_view", "profile"} & tags:
        return "standing_side"
    return "standing_front"


def anatomy_constraints(tags: set[str], bucket: str) -> list[str]:
    constraints: list[str] = []
    if tags & HAND_TAGS or bucket in {"hands_visible", "hand_focus", "action_pose"}:
        constraints.append("both hands visible")
    if tags & ARM_TAGS or bucket in {"arms_visible", "action_pose"}:
        constraints.append("arms visible")
    if tags & FEET_TAGS or "full_body" in tags or bucket == "feet_visible":
        constraints.append("feet visible")
    return dedupe_text(constraints)


def prompt_anchor_texts(tags: Sequence[str], *, limit: int = 4) -> list[str]:
    anchors: list[str] = []
    for tag in tags:
        if tag not in PROMPT_ANCHOR_TAGS:
            continue
        anchors.append(tag.replace("_", " "))
        if len(dedupe_text(anchors)) >= limit:
            break
    return dedupe_text(anchors)[:limit]


def _tag_prompt_parts(
    tags: set[str],
    *,
    framing: str,
    pose: str,
    constraints: Sequence[str],
    anchors: Sequence[str],
    clothing: str | None,
    scene: str | None,
) -> list[str]:
    parts: list[str] = []
    for tag in SUBJECT_PROMPT_TAGS:
        if tag in tags:
            parts.append(tag)
            break
    if "solo" in tags:
        parts.append("solo")
    parts.extend((framing, pose))
    parts.extend(constraints)
    parts.extend(anchors)
    if clothing:
        parts.append(clothing)
    if scene:
        parts.append(scene)
    return dedupe_text(parts)


def build_prompt_rows(
    metadata_path: str | Path,
    *,
    min_score: float = 0.0,
    preferred_min_score: float | None = None,
    limit: int | None = None,
    seed: int = 0,
    max_metadata_rows: int | None = None,
    prompt_style: str = "mixed",
    bucket_weights: Mapping[str, float] = DEFAULT_BUCKET_WEIGHTS,
) -> list[dict[str, Any]]:
    rows: list[PromptRow] = []
    seen_prompts: set[str] = set()
    scanned = 0
    for row in iter_metadata(metadata_path):
        scanned += 1
        if max_metadata_rows is not None and scanned > max_metadata_rows:
            break
        tags = normalize_tags(row)
        if not is_anatomy_positive(
            set(tags),
            min_score=min_score,
            score=record_score(row),
            allow_hand_focus=True,
        ):
            continue
        style = _resolve_prompt_style(prompt_style, len(rows))
        prompt_row = prompt_from_tags(row, tags, prompt_style=style)
        if prompt_row is None or prompt_row.prompt in seen_prompts:
            continue
        seen_prompts.add(prompt_row.prompt)
        rows.append(prompt_row)

    rng = random.Random(seed)
    buckets: dict[str, list[PromptRow]] = defaultdict(list)
    for row in rows:
        buckets[str(row.metadata["bucket"])].append(row)
    for bucket_rows in buckets.values():
        rng.shuffle(bucket_rows)

    balanced = _select_quota_rows(
        buckets,
        limit=limit,
        bucket_weights=bucket_weights,
        preferred_min_score=preferred_min_score,
    )
    return [{"prompt": row.prompt, "metadata": row.metadata} for row in balanced]


def split_prompt_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    train_limit: int,
    eval_limit: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        metadata = row.get("metadata") or {}
        key = (
            str(metadata.get("bucket", "unknown")),
            str(metadata.get("prompt_style", "unknown")),
        )
        groups[key].append(dict(row))

    eval_counts = proportional_group_counts(
        {key: len(value) for key, value in groups.items()},
        limit=eval_limit,
    )
    train_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    eval_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for key, group_rows in groups.items():
        eval_count = eval_counts.get(key, 0)
        bucket = key[0]
        eval_groups[bucket].extend(group_rows[:eval_count])
        train_groups[bucket].extend(group_rows[eval_count:])

    eval_rows = interleave_manifest_rows(eval_groups, limit=eval_limit)
    train_rows = interleave_manifest_rows(train_groups, limit=train_limit)
    return train_rows, eval_rows


def write_prompt_report(
    path: str | Path,
    train_rows: Sequence[Mapping[str, Any]],
    eval_rows: Sequence[Mapping[str, Any]],
) -> None:
    report = {
        "train_count": len(train_rows),
        "eval_count": len(eval_rows),
        "train_buckets": metadata_counts(train_rows, key="bucket"),
        "eval_buckets": metadata_counts(eval_rows, key="bucket"),
        "train_prompt_styles": metadata_counts(train_rows, key="prompt_style"),
        "eval_prompt_styles": metadata_counts(eval_rows, key="prompt_style"),
    }
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _resolve_prompt_style(prompt_style: str, index: int) -> str:
    if prompt_style == "mixed":
        return "tag" if index % 2 == 0 else "language"
    if prompt_style in {"tag", "language"}:
        return prompt_style
    raise ValueError("prompt_style must be 'tag', 'language', or 'mixed'")


def _first_subject(tags: set[str]) -> str | None:
    for tag, text in SUBJECT_TAGS.items():
        if tag in tags:
            return text
    return None


def _first_mapped(tags: set[str], mapping: Mapping[str, str]) -> str | None:
    for tag, text in mapping.items():
        if tag in tags:
            return text
    return None


def _first_tag_text(tags: set[str], candidates: Sequence[str]) -> str | None:
    for tag in candidates:
        if tag in tags:
            return tag.replace("_", " ")
    return None


def _select_quota_rows(
    buckets: Mapping[str, list[PromptRow]],
    *,
    limit: int | None,
    bucket_weights: Mapping[str, float],
    preferred_min_score: float | None,
) -> list[PromptRow]:
    if limit is None:
        limit = sum(len(rows) for rows in buckets.values())
    quotas = _bucket_quotas(limit, bucket_weights)
    selected_by_bucket: dict[str, list[PromptRow]] = {}
    leftovers: dict[str, list[PromptRow]] = {}

    for bucket, rows in buckets.items():
        quota = quotas.get(bucket, 0)
        if quota <= 0:
            leftovers[bucket] = list(rows)
            continue
        preferred, fallback = _split_preferred_rows(rows, preferred_min_score)
        selected = preferred[:quota]
        if len(selected) < quota:
            needed = quota - len(selected)
            selected.extend(fallback[:needed])
            fallback = fallback[needed:]
            preferred = []
        else:
            preferred = preferred[quota:]
        selected_by_bucket[bucket] = selected
        leftovers[bucket] = preferred + fallback

    selected = _interleave_bucket_rows(
        selected_by_bucket,
        limit=limit,
        bucket_order=list(bucket_weights),
    )
    if len(selected) < limit:
        selected.extend(
            _interleave_bucket_rows(
                leftovers,
                limit=limit - len(selected),
                bucket_order=list(bucket_weights),
            ),
        )
    return selected[:limit]


def _split_preferred_rows(
    rows: Sequence[PromptRow],
    preferred_min_score: float | None,
) -> tuple[list[PromptRow], list[PromptRow]]:
    if preferred_min_score is None:
        return list(rows), []
    preferred: list[PromptRow] = []
    fallback: list[PromptRow] = []
    for row in rows:
        score = float(row.metadata.get("source_score", 0.0) or 0.0)
        if score >= preferred_min_score:
            preferred.append(row)
        else:
            fallback.append(row)
    return preferred, fallback


def _bucket_quotas(limit: int, bucket_weights: Mapping[str, float]) -> dict[str, int]:
    total_weight = sum(max(0.0, float(weight)) for weight in bucket_weights.values())
    if total_weight <= 0:
        raise ValueError("bucket weights must sum to a positive value")
    raw = {
        bucket: limit * max(0.0, float(weight)) / total_weight
        for bucket, weight in bucket_weights.items()
    }
    quotas = {bucket: int(value) for bucket, value in raw.items()}
    remainder = limit - sum(quotas.values())
    for bucket, _ in sorted(
        raw.items(),
        key=lambda item: (item[1] - int(item[1]), item[0]),
        reverse=True,
    )[:remainder]:
        quotas[bucket] += 1
    return quotas


def _interleave_bucket_rows(
    buckets: Mapping[str, list[PromptRow]],
    *,
    limit: int | None,
    bucket_order: Sequence[str] | None = None,
) -> list[PromptRow]:
    queues = {bucket: list(rows) for bucket, rows in buckets.items() if rows}
    order = list(bucket_order or sorted(queues))
    order.extend(sorted(bucket for bucket in queues if bucket not in set(order)))
    out: list[PromptRow] = []
    while queues and (limit is None or len(out) < limit):
        progressed = False
        for bucket in list(order):
            rows = queues.get(bucket)
            if not rows:
                queues.pop(bucket, None)
                continue
            out.append(rows.pop())
            progressed = True
            if limit is not None and len(out) >= limit:
                break
        if not progressed:
            break
    return out
