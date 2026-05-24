"""Utilities for building anime experiment manifests.

The helpers in this module intentionally operate on local JSON/JSONL metadata
or deterministic prompt banks. Anatomy helpers keep prompt provenance explicit
so reward failures can be traced back to tag buckets and source posts.
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import random
import re
import tarfile
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vrl.scripts.data.common import default_data_root, emit

TEMPLATE_ID = "anime_anatomy_v1"
DOMAIN = "anime"

EXCLUDE_TAGS = {
    "cropped",
    "portrait",
    "upper_body",
    "cowboy_shot",
    "multiple_girls",
    "multiple_boys",
    "text",
    "watermark",
    "lowres",
}
HAND_FOCUS_ALLOWED_EXCLUDE_TAGS = {"upper_body", "cowboy_shot"}

SUBJECT_TAGS = {"1girl": "single anime girl", "1boy": "single anime boy"}
SUBJECT_PROMPT_TAGS = ("1girl", "1boy")
POSE_TAGS = {
    "standing": "standing",
    "walking": "walking",
    "running": "running",
    "sitting": "sitting",
    "kneeling": "kneeling",
    "dancing": "dancing",
    "fighting_stance": "fighting stance",
    "dynamic_pose": "dynamic action pose",
    "action_pose": "action pose",
    "jumping": "jumping",
    "kicking": "kicking",
    "high_kick": "high kick",
    "squatting": "squatting",
    "crouching": "crouching",
    "all_fours": "on all fours",
    "bent_over": "bent over",
    "standing_on_one_leg": "standing on one leg",
    "stretching": "stretching",
}
HAND_TAGS = {
    "hands",
    "hand",
    "hand_up",
    "outstretched_arm",
    "arms_up",
    "arms_behind_back",
    "waving",
    "holding",
    "hand_focus",
    "reaching",
    "spread_fingers",
    "open_hand",
    "clenched_hand",
    "clenched_hands",
    "interlocked_fingers",
    "finger_gun",
    "peace_sign",
    "v",
    "salute",
}
HAND_FOCUS_BUCKET_TAGS = {
    "hand_focus",
    "spread_fingers",
    "open_hand",
    "clenched_hand",
    "clenched_hands",
    "interlocked_fingers",
    "finger_gun",
    "peace_sign",
    "v",
    "salute",
}
FEET_TAGS = {"feet", "foot", "barefoot", "boots", "shoes", "sneakers", "sandals"}
ARM_TAGS = {
    "arms",
    "arm",
    "arms_up",
    "outstretched_arm",
    "arms_behind_back",
    "crossed_arms",
    "arm_up",
    "arm_support",
}
CLOTHING_TAGS = (
    "boots",
    "shoes",
    "dress",
    "coat",
    "armor",
    "school_uniform",
    "sportswear",
    "jacket",
    "skirt",
)
SCENE_TAGS = (
    "simple_background",
    "city_street",
    "forest",
    "studio_background",
    "stage",
    "street",
    "indoors",
    "outdoors",
)
ACTION_BUCKET_TAGS = {
    "dancing",
    "fighting_stance",
    "dynamic_pose",
    "action_pose",
    "jumping",
    "kicking",
    "high_kick",
    "squatting",
    "crouching",
    "all_fours",
    "bent_over",
    "standing_on_one_leg",
    "stretching",
    "holding_weapon",
    "holding_sword",
    "holding_gun",
    "dual_wielding",
}
DEFAULT_BUCKET_WEIGHTS = {
    "hands_visible": 0.14,
    "hand_focus": 0.10,
    "action_pose": 0.16,
    "running": 0.08,
    "walking": 0.10,
    "kneeling": 0.10,
    "standing_front": 0.07,
    "feet_visible": 0.07,
    "sitting_full_body": 0.07,
    "arms_visible": 0.08,
    "standing_side": 0.03,
}
PROMPT_ANCHOR_TAGS = (
    "long_hair",
    "short_hair",
    "medium_hair",
    "very_long_hair",
    "twintails",
    "ponytail",
    "braid",
    "wavy_hair",
    "black_hair",
    "brown_hair",
    "blonde_hair",
    "red_hair",
    "blue_hair",
    "pink_hair",
    "white_hair",
    "silver_hair",
    "green_hair",
    "purple_hair",
    "black_eyes",
    "brown_eyes",
    "blue_eyes",
    "green_eyes",
    "red_eyes",
    "purple_eyes",
    "yellow_eyes",
    "shirt",
    "blouse",
    "sweater",
    "hoodie",
    "jacket",
    "coat",
    "dress",
    "skirt",
    "pants",
    "shorts",
    "school_uniform",
    "sportswear",
    "armor",
    "gloves",
    "hat",
    "cap",
    "hood",
    "scarf",
    "cape",
    "ribbon",
    "bow",
    "belt",
    "backpack",
    "bag",
    "sword",
    "staff",
    "umbrella",
    "microphone",
    "musical_instrument",
    "looking_at_viewer",
    "looking_back",
    "side_view",
    "profile",
    "from_side",
    "from_behind",
    "smile",
    "serious",
    "open_mouth",
    "closed_mouth",
)
FAILURE_LABELS = {
    "bad_hands",
    "missing_hand",
    "missing_feet",
    "extra_fingers",
    "missing_fingers",
    "melted_fingers",
    "extra_limbs",
    "disconnected_arm",
    "impossible_leg_bend",
    "implausible_action_pose",
    "cropped_full_body",
    "wrong_person_count",
}
SAFETY_TEMPLATE_ID = "anime_safety_danbooru_v1"
SAFETY_RATING_ALIASES = {
    "g": "general",
    "general": "general",
    "safe": "general",
    "s": "sensitive",
    "sensitive": "sensitive",
    "q": "questionable",
    "questionable": "questionable",
    "e": "explicit",
    "explicit": "explicit",
}
SAFETY_TARGET_RATINGS = ("questionable", "explicit")
SAFETY_EXCLUDED_TAGS = {
    "loli",
    "shota",
    "child",
    "toddler",
    "baby",
    "infant",
    "kindergarten",
    "elementary_school_student",
    "middle_school_student",
    "underage",
}
SAFETY_RISK_TAGS = {
    "ass",
    "ass_focus",
    "areolae",
    "bikini",
    "bottomless",
    "bra",
    "breast_grab",
    "breasts",
    "cameltoe",
    "cleavage",
    "condom",
    "cum",
    "cum_on_body",
    "cum_on_breasts",
    "cum_on_face",
    "cum_on_tongue",
    "cumdrip",
    "cunnilingus",
    "erection",
    "exhibitionism",
    "fellatio",
    "female_masturbation",
    "fingering",
    "fondling",
    "genitals",
    "grabbing_own_breast",
    "handjob",
    "hetero",
    "imminent_sex",
    "intercrural_sex",
    "lingerie",
    "masturbation",
    "nakadashi",
    "nipple_slip",
    "nipples",
    "nude",
    "open_clothes",
    "oral",
    "paizuri",
    "panties",
    "pantyshot",
    "penis",
    "pussy",
    "pussy_juice",
    "sex",
    "sex_from_behind",
    "spread_legs",
    "straddling",
    "swimsuit",
    "testicles",
    "thigh_gap",
    "topless",
    "undressing",
    "underwear",
    "vaginal",
    "vaginal_sex",
}

@dataclass(frozen=True, slots=True)
class PromptRow:
    prompt: str
    metadata: dict[str, Any]


def iter_metadata(path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield metadata rows from JSONL, JSON, gzip, or tarred JSON containers."""

    p = Path(path)
    suffixes = p.suffixes
    if suffixes[-2:] == [".tar", ".gz"] or p.suffix == ".tgz":
        yield from _iter_tarred_json_metadata(p)
        return

    if p.suffix == ".gz":
        with gzip.open(p, "rt", encoding="utf-8") as handle:
            yield from _iter_json_records(handle, source=str(p))
        return

    if p.suffix in {".json", ".jsonl"}:
        with p.open(encoding="utf-8") as handle:
            yield from _iter_json_records(handle, source=str(p))
        return

    payload = json.loads(p.read_text(encoding="utf-8"))
    yield from _iter_json_payload(payload, source=str(p))


def _iter_tarred_json_metadata(path: Path) -> Iterator[dict[str, Any]]:
    with tarfile.open(path, "r:*") as archive:
        for member in archive:
            if not member.isfile():
                continue
            if not member.name.endswith((".json", ".jsonl", ".json.gz")):
                continue
            extracted = archive.extractfile(member)
            if extracted is None:
                continue
            if member.name.endswith(".gz"):
                with gzip.open(extracted, "rt", encoding="utf-8") as handle:
                    yield from _iter_json_records(handle, source=f"{path}:{member.name}")
            else:
                with io.TextIOWrapper(extracted, encoding="utf-8") as handle:
                    yield from _iter_json_records(handle, source=f"{path}:{member.name}")
            return
    raise ValueError(f"{path}: expected a JSON or JSONL file inside the archive")


def _iter_json_records(handle: io.TextIOBase, *, source: str) -> Iterator[dict[str, Any]]:
    first = ""
    for line in handle:
        if line.strip():
            first = line
            break
    if not first:
        return
    try:
        first_payload = json.loads(first)
    except json.JSONDecodeError:
        payload = json.loads(first + handle.read())
        yield from _iter_json_payload(payload, source=source)
        return

    if isinstance(first_payload, dict):
        yield first_payload
        for line_number, line in enumerate(handle, start=2):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{source}:{line_number}: JSONL rows must be objects")
            yield row
        return

    yield from _iter_json_payload(first_payload, source=source)


def _iter_json_payload(payload: Any, *, source: str) -> Iterator[dict[str, Any]]:
    if isinstance(payload, list):
        for row in payload:
            if not isinstance(row, dict):
                raise ValueError(f"{source}: JSON array entries must be objects")
            yield row
        return
    if isinstance(payload, dict):
        for key in ("posts", "items", "data", "rows"):
            rows = payload.get(key)
            if isinstance(rows, list):
                for row in rows:
                    if not isinstance(row, dict):
                        raise ValueError(f"{source}: {key} entries must be objects")
                    yield row
                return
        yield payload
        return
    raise ValueError(f"{source}: expected JSON object, JSON array, or JSONL")


def write_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> int:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with p.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True) + "\n")
            count += 1
    return count


def write_jsonl_preserve_order(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> int:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with p.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row)) + "\n")
            count += 1
    return count


def normalize_tags(row: Mapping[str, Any]) -> list[str]:
    tags: list[str] = []
    for key in (
        "tags",
        "tag_string",
        "tag_string_general",
        "tag_string_character",
        "tag_string_meta",
    ):
        value = row.get(key)
        if value is None:
            continue
        if isinstance(value, str):
            tags.extend(value.split())
        elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
            tags.extend(str(item) for item in value)
    out: list[str] = []
    seen: set[str] = set()
    for tag in tags:
        normalized = str(tag).strip().lower().replace(" ", "_")
        if normalized and normalized not in seen:
            seen.add(normalized)
            out.append(normalized)
    return out


def record_id(row: Mapping[str, Any]) -> Any:
    for key in ("id", "post_id", "danbooru_id"):
        if key in row and row[key] not in (None, ""):
            return row[key]
    return None


def record_score(row: Mapping[str, Any]) -> float:
    try:
        return float(row.get("score", 0) or 0)
    except (TypeError, ValueError):
        return 0.0


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
    prompt = ", ".join(_dedupe_text(parts))

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
    return _dedupe_text(constraints)


def prompt_anchor_texts(tags: Sequence[str], *, limit: int = 4) -> list[str]:
    anchors: list[str] = []
    for tag in tags:
        if tag not in PROMPT_ANCHOR_TAGS:
            continue
        anchors.append(tag.replace("_", " "))
        if len(_dedupe_text(anchors)) >= limit:
            break
    return _dedupe_text(anchors)[:limit]


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
    return _dedupe_text(parts)


def build_prompt_rows(
    metadata_path: str | Path,
    *,
    min_score: float = 0.0,
    preferred_min_score: float | None = None,
    limit: int | None = None,
    seed: int = 0,
    candidate_limit: int | None = None,
    max_metadata_rows: int | None = None,
    prompt_style: str = "mixed",
    bucket_weights: Mapping[str, float] | None = DEFAULT_BUCKET_WEIGHTS,
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
        if candidate_limit is not None and len(rows) >= candidate_limit:
            break

    rng = random.Random(seed)
    buckets: dict[str, list[PromptRow]] = defaultdict(list)
    for row in rows:
        buckets[str(row.metadata["bucket"])].append(row)
    for bucket_rows in buckets.values():
        rng.shuffle(bucket_rows)

    if bucket_weights:
        balanced = _select_quota_rows(
            buckets,
            limit=limit,
            bucket_weights=bucket_weights,
            preferred_min_score=preferred_min_score,
        )
    else:
        balanced = _interleave_bucket_rows(buckets, limit=limit)

    return [{"prompt": row.prompt, "metadata": row.metadata} for row in balanced]


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

    eval_counts = _proportional_group_counts(
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

    eval_rows = _interleave_manifest_rows(eval_groups, limit=eval_limit)
    train_rows = _interleave_manifest_rows(train_groups, limit=train_limit)
    return train_rows, eval_rows


def _proportional_group_counts(
    group_sizes: Mapping[tuple[str, str], int],
    *,
    limit: int,
) -> dict[tuple[str, str], int]:
    total = sum(group_sizes.values())
    if total <= 0 or limit <= 0:
        return {key: 0 for key in group_sizes}
    raw = {
        key: min(size, limit * size / total)
        for key, size in group_sizes.items()
    }
    counts = {key: int(value) for key, value in raw.items()}
    remainder = min(limit, total) - sum(counts.values())
    for key, _ in sorted(
        raw.items(),
        key=lambda item: (item[1] - int(item[1]), item[0]),
        reverse=True,
    ):
        if remainder <= 0:
            break
        if counts[key] >= group_sizes[key]:
            continue
        counts[key] += 1
        remainder -= 1
    return counts


def _proportional_text_group_counts(
    group_sizes: Mapping[str, int],
    *,
    limit: int,
) -> dict[str, int]:
    total = sum(group_sizes.values())
    if total <= 0 or limit <= 0:
        return {key: 0 for key in group_sizes}
    raw = {
        key: min(size, limit * size / total)
        for key, size in group_sizes.items()
    }
    counts = {key: int(value) for key, value in raw.items()}
    remainder = min(limit, total) - sum(counts.values())
    for key, _ in sorted(
        raw.items(),
        key=lambda item: (item[1] - int(item[1]), item[0]),
        reverse=True,
    ):
        if remainder <= 0:
            break
        if counts[key] >= group_sizes[key]:
            continue
        counts[key] += 1
        remainder -= 1
    return counts


def _interleave_manifest_rows(
    groups: Mapping[str, list[dict[str, Any]]],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    queues = {bucket: list(rows) for bucket, rows in groups.items() if rows}
    out: list[dict[str, Any]] = []
    while queues and len(out) < limit:
        progressed = False
        for bucket in sorted(list(queues)):
            rows = queues[bucket]
            if not rows:
                del queues[bucket]
                continue
            out.append(rows.pop(0))
            progressed = True
            if len(out) >= limit:
                break
        if not progressed:
            break
    return out


def write_prompt_report(path: str | Path, train_rows: Sequence[Mapping[str, Any]], eval_rows: Sequence[Mapping[str, Any]]) -> None:
    report = {
        "train_count": len(train_rows),
        "eval_count": len(eval_rows),
        "train_buckets": _bucket_counts(train_rows),
        "eval_buckets": _bucket_counts(eval_rows),
        "train_prompt_styles": _prompt_style_counts(train_rows),
        "eval_prompt_styles": _prompt_style_counts(eval_rows),
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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
    return _interleave_manifest_rows(by_rating, limit=limit or len(candidates))


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

    eval_counts = _proportional_text_group_counts(
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
        _interleave_manifest_rows(train_groups, limit=train_limit),
        _interleave_manifest_rows(eval_groups, limit=eval_limit),
    )


def write_safety_report(
    path: str | Path,
    train_rows: Sequence[Mapping[str, Any]],
    eval_rows: Sequence[Mapping[str, Any]],
) -> None:
    report = {
        "train_count": len(train_rows),
        "eval_count": len(eval_rows),
        "train_categories": _category_counts(train_rows),
        "eval_categories": _category_counts(eval_rows),
        "train_ratings": _rating_counts(train_rows),
        "eval_ratings": _rating_counts(eval_rows),
        "train_nsfw_tags_top": _nsfw_tag_counts(train_rows, limit=50),
        "eval_nsfw_tags_top": _nsfw_tag_counts(eval_rows, limit=50),
    }
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def positive_image_rows(
    metadata_path: str | Path,
    *,
    image_root: str | Path | None = None,
    min_score: float = 0.0,
    limit: int | None = None,
    source: str = "danbooru",
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in iter_metadata(metadata_path):
        tags = normalize_tags(row)
        if not is_anatomy_positive(set(tags), min_score=min_score, score=record_score(row)):
            continue
        image_path = _image_path(row, image_root=image_root)
        if image_path is None:
            continue
        out.append(
            {
                "image_path": image_path,
                "source": source,
                "post_id": record_id(row),
                "score": record_score(row),
                "tags": tags,
                "domain": DOMAIN,
            },
        )
        if limit is not None and len(out) >= limit:
            break
    return out


def hand_crop_rows(
    manifests: Sequence[str | Path],
    *,
    label: str,
    source: str,
    fallback_whole_image: bool = False,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for manifest in manifests:
        for row in iter_metadata(manifest):
            image_path = row.get("image_path")
            if not image_path:
                continue
            boxes = _hand_boxes(row)
            if not boxes and fallback_whole_image:
                boxes = [None]
            for index, box in enumerate(boxes):
                crop_row = {
                    "image_path": str(image_path),
                    "source": row.get("source", source),
                    "parent_image_path": str(image_path),
                    "labels": [label],
                    "domain": row.get("domain", DOMAIN),
                }
                if box is not None:
                    crop_row["bbox"] = box
                    crop_row["crop_id"] = f"{Path(str(image_path)).stem}_{index:02d}"
                rows.append(crop_row)
    return rows


def hard_negative_rows(
    candidates_path: str | Path,
    *,
    min_severity: int = 1,
    labels: set[str] | None = None,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    requested = labels or FAILURE_LABELS
    for row in iter_metadata(candidates_path):
        row_labels = {str(label) for label in row.get("labels", [])}
        if not row_labels & requested:
            continue
        severity = int(row.get("severity", 1) or 1)
        if severity < min_severity:
            continue
        out = dict(row)
        out.setdefault("source", "anima_base")
        out.setdefault("domain", DOMAIN)
        out["labels"] = sorted(row_labels)
        out["severity"] = severity
        selected.append(out)
    return selected


def label_queue_rows(hard_negative_path: str | Path) -> list[dict[str, Any]]:
    questions = (
        "Are both arms anatomically connected?",
        "Are the legs plausible for the requested action?",
        "Are hands visible when requested?",
        "Are fingers plausible enough for the image scale?",
        "Which failure labels apply?",
    )
    rows: list[dict[str, Any]] = []
    for row in iter_metadata(hard_negative_path):
        rows.append(
            {
                "image_path": row.get("image_path", ""),
                "prompt": row.get("prompt", ""),
                "candidate_labels": row.get("labels", []),
                "severity": row.get("severity", None),
                "questions": list(questions),
                "metadata": {
                    "source": row.get("source", "anima_base"),
                    "domain": row.get("domain", DOMAIN),
                },
            },
        )
    return rows


def add_common_limit_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--min-score", type=float, default=0.0)


def _bucket_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counter = Counter(
        str((row.get("metadata") or {}).get("bucket", "unknown"))
        for row in rows
    )
    return dict(sorted(counter.items()))


def _prompt_style_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counter = Counter(
        str((row.get("metadata") or {}).get("prompt_style", "unknown"))
        for row in rows
    )
    return dict(sorted(counter.items()))


def _category_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counter = Counter(
        str((row.get("metadata") or {}).get("category", "unknown"))
        for row in rows
    )
    return dict(sorted(counter.items()))


def _rating_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counter = Counter(
        str((row.get("metadata") or {}).get("rating", "unrated"))
        for row in rows
    )
    return dict(sorted(counter.items()))


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
    return _dedupe_text(out)


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

    context_tags = (
        tuple(POSE_TAGS)
        + PROMPT_ANCHOR_TAGS
        + CLOTHING_TAGS
        + SCENE_TAGS
    )
    for tag in context_tags:
        if tag in tag_set:
            parts.append(tag)
    cleaned = _dedupe_text(parts)
    return ", ".join(cleaned[: max(1, prompt_tag_limit)])


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


def _dedupe_text(parts: Iterable[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for part in parts:
        text = re.sub(r"\s+", " ", str(part).strip())
        if text and text not in seen:
            seen.add(text)
            out.append(text)
    return out


def _image_path(row: Mapping[str, Any], *, image_root: str | Path | None) -> str | None:
    for key in ("image_path", "file_path", "path"):
        value = row.get(key)
        if value:
            return str(Path(str(value)).expanduser())
    if image_root is None:
        return None
    post_id = record_id(row)
    if post_id is None:
        return None
    ext = str(row.get("file_ext") or row.get("ext") or "jpg").lstrip(".")
    return str(Path(image_root).expanduser() / f"{post_id}.{ext}")


def _hand_boxes(row: Mapping[str, Any]) -> list[Any]:
    for key in ("hand_boxes", "hands", "bboxes", "boxes"):
        value = row.get(key)
        if isinstance(value, list):
            return value
    bbox = row.get("bbox")
    return [bbox] if bbox is not None else []


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_prompts = subparsers.add_parser(
        "build-prompts",
        help="Build quota-balanced train/eval prompt manifests from metadata.",
    )
    build_prompts.add_argument("--metadata", default="")
    build_prompts.add_argument("--download-danbooru-metadata", action="store_true")
    build_prompts.add_argument("--hf-repo", default="nyanko7/danbooru2023")
    build_prompts.add_argument("--hf-file", default="metadata/posts.tar.gz")
    build_prompts.add_argument("--hf-cache-dir", default="")
    build_prompts.add_argument("--train-output", required=True)
    build_prompts.add_argument("--eval-output", required=True)
    build_prompts.add_argument("--report-output", required=True)
    build_prompts.add_argument("--train-limit", type=int, default=20_000)
    build_prompts.add_argument("--eval-limit", type=int, default=1_000)
    build_prompts.add_argument("--min-score", type=float, default=5.0)
    build_prompts.add_argument("--preferred-min-score", type=float, default=20.0)
    build_prompts.add_argument("--seed", type=int, default=0)
    build_prompts.add_argument(
        "--prompt-style",
        choices=("tag", "language", "mixed"),
        default="mixed",
    )
    build_prompts.add_argument(
        "--bucket-balance",
        choices=("quota", "natural"),
        default="quota",
    )
    build_prompts.add_argument("--candidate-pool-factor", type=int, default=2)
    build_prompts.add_argument("--max-metadata-rows", type=int, default=None)
    build_prompts.add_argument("--allow-partial", action="store_true")
    build_prompts.set_defaults(func=_cmd_build_prompts)

    safety = subparsers.add_parser(
        "build-safety-prompts",
        help="Build deterministic anime safety train/eval prompt manifests.",
    )
    safety.add_argument("--train-output", required=True)
    safety.add_argument("--eval-output", required=True)
    safety.add_argument("--report-output", default="")
    safety.add_argument("--metadata", default="")
    safety.add_argument("--download-danbooru-metadata", action="store_true")
    safety.add_argument("--hf-repo", default="nyanko7/danbooru2023")
    safety.add_argument("--hf-file", default="metadata/posts.tar.gz")
    safety.add_argument("--hf-cache-dir", default="")
    safety.add_argument("--train-limit", type=int, default=None)
    safety.add_argument("--eval-limit", type=int, default=None)
    safety.add_argument("--rating", action="append", default=[])
    safety.add_argument("--min-score", type=float, default=0.0)
    safety.add_argument("--seed", type=int, default=0)
    safety.add_argument("--candidate-pool-factor", type=int, default=3)
    safety.add_argument("--max-metadata-rows", type=int, default=None)
    safety.add_argument("--prompt-tag-limit", type=int, default=24)
    safety.add_argument("--min-risk-tags", type=int, default=1)
    safety.add_argument("--allow-partial", action="store_true")
    safety.set_defaults(func=_cmd_build_safety_prompts)

    positives = subparsers.add_parser(
        "build-positives",
        help="Build anime positive image manifest for reward calibration.",
    )
    positives.add_argument("--metadata", required=True)
    positives.add_argument("--output", required=True)
    positives.add_argument("--image-root", default=None)
    positives.add_argument("--source", default="danbooru")
    positives.add_argument("--min-score", type=float, default=0.0)
    positives.add_argument("--limit", type=int, default=10_000)
    positives.set_defaults(func=_cmd_build_positives)

    crops = subparsers.add_parser(
        "build-hand-crops",
        help="Build anime hand crop manifest from positive/generated manifests.",
    )
    crops.add_argument("--positive-manifest", action="append", default=[])
    crops.add_argument("--generated-manifest", action="append", default=[])
    crops.add_argument("--output", required=True)
    crops.add_argument("--fallback-whole-image", action="store_true")
    crops.set_defaults(func=_cmd_build_hand_crops)

    hard_negatives = subparsers.add_parser(
        "mine-hard-negatives",
        help="Filter generated candidates into an anatomy hard-negative manifest.",
    )
    hard_negatives.add_argument("--candidates", required=True)
    hard_negatives.add_argument("--output", required=True)
    hard_negatives.add_argument("--min-severity", type=int, default=1)
    hard_negatives.add_argument("--label", action="append", default=[])
    hard_negatives.set_defaults(func=_cmd_mine_hard_negatives)

    label_queue = subparsers.add_parser(
        "export-label-queue",
        help="Export anatomy hard negatives into a focused label queue JSONL.",
    )
    label_queue.add_argument("--hard-negatives", required=True)
    label_queue.add_argument("--output", required=True)
    label_queue.set_defaults(func=_cmd_export_label_queue)

    calibrate = subparsers.add_parser(
        "calibrate-rewards",
        help="Run offline audit for the anime anatomy pose reward.",
    )
    calibrate.add_argument("--positive-manifest", required=True)
    calibrate.add_argument("--hard-negative-manifest", required=True)
    calibrate.add_argument("--output", required=True)
    calibrate.add_argument("--device", default="cuda")
    calibrate.add_argument("--max-samples", type=int, default=512)
    calibrate.set_defaults(func=_cmd_calibrate_rewards)

    args = parser.parse_args(argv)
    args.func(args)


def _cmd_build_prompts(args: argparse.Namespace) -> None:
    metadata = args.metadata or ""
    if args.download_danbooru_metadata:
        metadata = _download_metadata(
            repo_id=args.hf_repo,
            filename=args.hf_file,
            cache_dir=args.hf_cache_dir or None,
        )
    if not metadata:
        raise ValueError("Provide --metadata or pass --download-danbooru-metadata")

    target_count = args.train_limit + args.eval_limit
    bucket_weights = DEFAULT_BUCKET_WEIGHTS if args.bucket_balance == "quota" else None
    candidate_limit = (
        None
        if bucket_weights
        else max(target_count, target_count * args.candidate_pool_factor)
    )
    rows = build_prompt_rows(
        metadata,
        min_score=args.min_score,
        preferred_min_score=args.preferred_min_score if bucket_weights else None,
        limit=target_count,
        seed=args.seed,
        candidate_limit=candidate_limit,
        max_metadata_rows=args.max_metadata_rows,
        prompt_style=args.prompt_style,
        bucket_weights=bucket_weights,
    )
    train_rows, eval_rows = split_prompt_rows(
        rows,
        train_limit=args.train_limit,
        eval_limit=args.eval_limit,
    )
    if not args.allow_partial and (
        len(train_rows) < args.train_limit or len(eval_rows) < args.eval_limit
    ):
        raise RuntimeError(
            "Not enough prompt rows generated: "
            f"train={len(train_rows)}/{args.train_limit}, "
            f"eval={len(eval_rows)}/{args.eval_limit}. "
            "Increase --max-metadata-rows, lower filters, or pass --allow-partial.",
        )
    write_jsonl(args.train_output, train_rows)
    write_jsonl(args.eval_output, eval_rows)
    write_prompt_report(args.report_output, train_rows, eval_rows)


def _cmd_build_safety_prompts(args: argparse.Namespace) -> None:
    metadata = args.metadata or ""
    if args.download_danbooru_metadata:
        metadata = _download_metadata(
            repo_id=args.hf_repo,
            filename=args.hf_file,
            cache_dir=args.hf_cache_dir or None,
        )

    if not metadata:
        raise ValueError("Provide --metadata or pass --download-danbooru-metadata")

    train_limit = args.train_limit if args.train_limit is not None else 2_000
    eval_limit = args.eval_limit if args.eval_limit is not None else 200
    target_count = train_limit + eval_limit
    rows = build_danbooru_safety_prompt_rows(
        metadata,
        ratings=args.rating or SAFETY_TARGET_RATINGS,
        min_score=args.min_score,
        limit=target_count,
        seed=args.seed,
        candidate_limit=max(target_count, target_count * args.candidate_pool_factor),
        max_metadata_rows=args.max_metadata_rows,
        prompt_tag_limit=args.prompt_tag_limit,
        min_risk_tags=args.min_risk_tags,
    )
    train_rows, eval_rows = split_safety_prompt_rows(
        rows,
        train_limit=train_limit,
        eval_limit=eval_limit,
    )
    if not args.allow_partial and (
        len(train_rows) < train_limit or len(eval_rows) < eval_limit
    ):
        raise RuntimeError(
            "Not enough safety prompt rows generated: "
            f"train={len(train_rows)}/{train_limit}, "
            f"eval={len(eval_rows)}/{eval_limit}. "
            "Increase --max-metadata-rows, lower filters, or pass --allow-partial.",
        )
    write_jsonl_preserve_order(args.train_output, train_rows)
    write_jsonl_preserve_order(args.eval_output, eval_rows)
    if args.report_output:
        write_safety_report(args.report_output, train_rows, eval_rows)


def _download_metadata(*, repo_id: str, filename: str, cache_dir: str | None) -> str:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required for --download-danbooru-metadata",
        ) from exc

    return hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type="dataset",
        cache_dir=str(Path(cache_dir).expanduser()) if cache_dir else None,
    )


def _cmd_build_positives(args: argparse.Namespace) -> None:
    rows = positive_image_rows(
        args.metadata,
        image_root=args.image_root,
        min_score=args.min_score,
        limit=args.limit,
        source=args.source,
    )
    write_jsonl(args.output, rows)


def _cmd_build_hand_crops(args: argparse.Namespace) -> None:
    rows = []
    rows.extend(
        hand_crop_rows(
            args.positive_manifest,
            label="hand_ok",
            source="anime_positive",
            fallback_whole_image=args.fallback_whole_image,
        ),
    )
    rows.extend(
        hand_crop_rows(
            args.generated_manifest,
            label="bad_hand",
            source="anima_generated",
            fallback_whole_image=args.fallback_whole_image,
        ),
    )
    write_jsonl(args.output, rows)


def _cmd_mine_hard_negatives(args: argparse.Namespace) -> None:
    labels = set(args.label) if args.label else FAILURE_LABELS
    rows = hard_negative_rows(
        args.candidates,
        min_severity=args.min_severity,
        labels=labels,
    )
    write_jsonl(args.output, rows)


def _cmd_export_label_queue(args: argparse.Namespace) -> None:
    write_jsonl(args.output, label_queue_rows(args.hard_negatives))


def _cmd_calibrate_rewards(args: argparse.Namespace) -> None:
    import asyncio

    report = asyncio.run(
        _run_reward_calibration(
            positive_manifest=args.positive_manifest,
            hard_negative_manifest=args.hard_negative_manifest,
            device=args.device,
            max_samples=args.max_samples,
        ),
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


async def _run_reward_calibration(
    *,
    positive_manifest: str,
    hard_negative_manifest: str,
    device: str,
    max_samples: int,
) -> dict[str, Any]:
    from vrl.rewards.functions.anime_anatomy import AnimeAnatomyStructureReward

    positives = _load_reward_rollouts(positive_manifest, max_samples=max_samples)
    negatives = _load_reward_rollouts(hard_negative_manifest, max_samples=max_samples)

    anatomy_reward = AnimeAnatomyStructureReward(device=device)
    pose_pos = await anatomy_reward.score_batch(positives)
    pose_neg = await anatomy_reward.score_batch(negatives)

    return {
        "positive_count": len(positives),
        "hard_negative_count": len(negatives),
        "anime_anatomy_structure": _metric_report(pose_pos, pose_neg),
    }


def _load_reward_rollouts(path: str, *, max_samples: int) -> list[Any]:
    from PIL import Image

    from vrl.rewards.types import RewardRollout, RewardTrajectory

    rows = []
    for row in iter_metadata(path):
        image_path = row.get("image_path")
        if not image_path:
            continue
        p = Path(str(image_path)).expanduser()
        if not p.exists():
            continue
        rows.append(
            RewardRollout(
                request=None,
                trajectory=RewardTrajectory(
                    prompt=str(row.get("prompt", "")),
                    seed=0,
                    steps=[],
                    output=Image.open(p).convert("RGB"),
                ),
                metadata={"manifest_row": row},
            ),
        )
        if len(rows) >= max_samples:
            break
    return rows


def _metric_report(positive_scores: list[float], negative_scores: list[float]) -> dict[str, Any]:
    return {
        "positive_mean": _mean(positive_scores),
        "hard_negative_mean": _mean(negative_scores),
        "auc": _pairwise_auc(positive_scores, negative_scores),
    }


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _pairwise_auc(positive_scores: list[float], negative_scores: list[float]) -> float:
    if not positive_scores or not negative_scores:
        return 0.0
    wins = 0.0
    total = 0
    for pos in positive_scores:
        for neg in negative_scores:
            total += 1
            if pos > neg:
                wins += 1.0
            elif pos == neg:
                wins += 0.5
    return wins / total


# --------------------------------------------------------------------------- #
# populate-facing commands (wired into `python -m vrl.scripts.data.populate`)  #
# --------------------------------------------------------------------------- #


def register(subparsers: Any) -> None:
    prompts = subparsers.add_parser("anime-prompts")
    prompts.add_argument("--metadata", type=Path, default=None)
    prompts.add_argument(
        "--train-output",
        type=Path,
        default=Path("datasets/danbooru/anatomy/train_prompts.jsonl"),
    )
    prompts.add_argument(
        "--eval-output",
        type=Path,
        default=Path("datasets/danbooru/anatomy/eval_prompts.jsonl"),
    )
    prompts.add_argument(
        "--report-output",
        type=Path,
        default=Path("datasets/danbooru/anatomy/prompt_report.json"),
    )
    prompts.add_argument("--train-limit", type=int, default=20_000)
    prompts.add_argument("--eval-limit", type=int, default=1_000)
    prompts.add_argument("--min-score", type=int, default=5)
    prompts.add_argument("--preferred-min-score", type=int, default=20)
    prompts.add_argument("--prompt-style", default="mixed")
    prompts.set_defaults(func=_cmd_anime_prompts)

    positives = subparsers.add_parser("anime-positives")
    positives.add_argument("--metadata", type=Path, required=True)
    positives.add_argument("--image-root", type=Path, default=None)
    positives.add_argument(
        "--output",
        type=Path,
        default=Path("datasets/danbooru/anatomy/positive_images.jsonl"),
    )
    positives.add_argument("--hand-crops-output", type=Path, default=None)
    positives.add_argument("--source", default="danbooru2023")
    positives.add_argument("--min-score", type=int, default=20)
    positives.add_argument("--limit", type=int, default=10_000)
    positives.add_argument("--fetch-images", action="store_true")
    positives.add_argument("--overwrite", action="store_true")
    positives.set_defaults(func=_cmd_anime_positives)

    fetch = subparsers.add_parser("anime-fetch-images")
    fetch.add_argument("--metadata", type=Path, required=True)
    fetch.add_argument("--image-root", type=Path, default=None)
    fetch.add_argument("--min-score", type=int, default=20)
    fetch.add_argument("--limit", type=int, default=10_000)
    fetch.add_argument("--source", default="danbooru2023")
    fetch.add_argument("--overwrite", action="store_true")
    fetch.set_defaults(func=_cmd_anime_fetch_images)


def _cmd_anime_prompts(args: argparse.Namespace) -> None:
    command = [
        "build-prompts",
        "--train-output",
        str(args.train_output),
        "--eval-output",
        str(args.eval_output),
        "--report-output",
        str(args.report_output),
        "--train-limit",
        str(args.train_limit),
        "--eval-limit",
        str(args.eval_limit),
        "--min-score",
        str(args.min_score),
        "--preferred-min-score",
        str(args.preferred_min_score),
        "--bucket-balance",
        "quota",
        "--prompt-style",
        args.prompt_style,
    ]
    if args.metadata:
        command.extend(["--metadata", str(args.metadata)])
    else:
        command.append("--download-danbooru-metadata")
    main(command)


def _cmd_anime_positives(args: argparse.Namespace) -> None:
    image_root = Path(args.image_root or (default_data_root() / "danbooru" / "images")).expanduser()
    fetched: dict[str, int] = {"selected": 0, "downloaded": 0, "skipped_existing": 0, "failed": 0}
    if args.fetch_images:
        image_root.mkdir(parents=True, exist_ok=True)
        targets = _select_positive_targets(
            args.metadata,
            image_root,
            min_score=args.min_score,
            limit=args.limit,
            source=args.source,
        )
        downloaded, skipped, failed = download_danbooru_images(
            args.metadata,
            targets,
            fetch=_http_download,
            overwrite=args.overwrite,
        )
        fetched = {
            "selected": len(targets),
            "downloaded": downloaded,
            "skipped_existing": skipped,
            "failed": failed,
        }

    main(
        [
            "build-positives",
            "--metadata",
            str(args.metadata),
            "--image-root",
            str(image_root),
            "--output",
            str(args.output),
            "--source",
            args.source,
            "--min-score",
            str(args.min_score),
            "--limit",
            str(args.limit),
        ],
    )
    hand_crops_written = 0
    if args.hand_crops_output:
        main(
            [
                "build-hand-crops",
                "--positive-manifest",
                str(args.output),
                "--output",
                str(args.hand_crops_output),
                "--fallback-whole-image",
            ],
        )
        hand_crops_written = _count_lines(Path(args.hand_crops_output))

    emit(
        {
            "dataset": "danbooru_positives",
            "image_root": str(image_root.resolve()),
            "positive_manifest": str(args.output),
            "positives_written": _count_lines(Path(args.output)),
            "hand_crops_output": str(args.hand_crops_output) if args.hand_crops_output else "",
            "hand_crops_written": hand_crops_written,
            "fetched": fetched,
        },
    )


def _cmd_anime_fetch_images(args: argparse.Namespace) -> None:
    image_root = Path(args.image_root or (default_data_root() / "danbooru" / "images")).expanduser()
    image_root.mkdir(parents=True, exist_ok=True)
    targets = _select_positive_targets(
        args.metadata,
        image_root,
        min_score=args.min_score,
        limit=args.limit,
        source=args.source,
    )
    downloaded, skipped, failed = download_danbooru_images(
        args.metadata,
        targets,
        fetch=_http_download,
        overwrite=args.overwrite,
    )
    emit(
        {
            "dataset": "danbooru_images",
            "image_root": str(image_root.resolve()),
            "selected": len(targets),
            "downloaded": downloaded,
            "skipped_existing": skipped,
            "failed": failed,
        },
    )


def download_danbooru_images(
    metadata_path: str | Path,
    targets: Mapping[str, Path],
    *,
    fetch: Callable[[str, Path], None],
    overwrite: bool = False,
) -> tuple[int, int, int]:
    """Download selected Danbooru images into their build-positives target paths.

    ``targets`` maps post id -> local image path (as produced by
    ``positive_image_rows``). ``fetch`` is injectable so the selection/IO wiring
    can be tested without network access. Returns (downloaded, skipped, failed).
    """

    remaining: dict[str, Path] = {str(pid): Path(path) for pid, path in targets.items()}
    downloaded = skipped = failed = 0
    for row in iter_metadata(metadata_path):
        post_id = str(record_id(row))
        target = remaining.pop(post_id, None)
        if target is None:
            continue
        if target.exists() and not overwrite:
            skipped += 1
        else:
            url = _danbooru_file_url(row)
            if not url:
                failed += 1
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                try:
                    fetch(url, target)
                    downloaded += 1
                except Exception:
                    failed += 1
        if not remaining:
            break
    return downloaded, skipped, failed


def _select_positive_targets(
    metadata_path: str | Path,
    image_root: Path,
    *,
    min_score: int,
    limit: int,
    source: str,
) -> dict[str, Path]:
    selected = positive_image_rows(
        metadata_path,
        image_root=image_root,
        min_score=min_score,
        limit=limit,
        source=source,
    )
    return {
        str(row["post_id"]): Path(row["image_path"])
        for row in selected
        if row.get("post_id") is not None and row.get("image_path")
    }


def _danbooru_file_url(row: Mapping[str, Any]) -> str:
    for key in ("file_url", "large_file_url", "preview_file_url"):
        value = row.get(key)
        if isinstance(value, str) and value.startswith(("http://", "https://")):
            return value
    return ""


def _http_download(url: str, target: Path) -> None:
    import requests

    with requests.get(url, stream=True, timeout=30) as response:
        response.raise_for_status()
        with target.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1 << 16):
                handle.write(chunk)


def _count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


__all__ = [
    "DEFAULT_BUCKET_WEIGHTS",
    "FAILURE_LABELS",
    "SAFETY_TARGET_RATINGS",
    "build_danbooru_safety_prompt_rows",
    "build_prompt_rows",
    "download_danbooru_images",
    "hand_crop_rows",
    "hard_negative_rows",
    "iter_metadata",
    "label_queue_rows",
    "main",
    "positive_image_rows",
    "register",
    "split_prompt_rows",
    "split_safety_prompt_rows",
    "write_jsonl",
    "write_jsonl_preserve_order",
    "write_prompt_report",
    "write_safety_report",
]


if __name__ == "__main__":
    main()
