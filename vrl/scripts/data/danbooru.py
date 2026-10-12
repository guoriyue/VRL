"""Danbooru-derived dataset builders: metadata parsing, safety prompts and the CLI.

The editable vocabulary remains in ``manifests/danbooru/config.yaml``; this
module loads that one source of truth. ``register`` and
``manifest_setup_hints`` are the setup-CLI surface.
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
from collections.abc import Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from vrl.config.base import ConfigBase
from vrl.utils.json_files import write_json, write_jsonl
from vrl.utils.paths import repo_root

# ── Defaults and taxonomy ─────────────────────────────────────────────────

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
    """``tags`` block of manifests/danbooru/config.yaml: prompt vocabulary shared by builders."""

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


# Parse the tag taxonomy through the same closed-section contract as configs.
_TAXONOMY_PATH = OUTPUT_DIR / "config.yaml"
with _TAXONOMY_PATH.open(encoding="utf-8") as _handle:
    _RAW_TAXONOMY = yaml.safe_load(_handle)
try:
    _TAXONOMY = _Taxonomy.model_validate(_RAW_TAXONOMY)
except ValidationError as exc:
    from vrl.config.base import _extract_error_message

    raise ValueError(f"{_TAXONOMY_PATH}: {_extract_error_message(exc)}") from exc
del _handle, _RAW_TAXONOMY
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


def dedupe_text(parts: Iterable[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for part in parts:
        text = re.sub(r"\s+", " ", str(part).strip())
        if text and text not in seen:
            seen.add(text)
            out.append(text)
    return out


# ── Metadata archive parsing ──────────────────────────────────────────────


def resolve_metadata_path(
    *,
    metadata: str | Path | None,
    download_metadata: bool,
    hf_repo: str,
    hf_file: str,
    hf_cache_dir: str | Path | None,
) -> str:
    if metadata:
        return str(metadata)
    if not download_metadata:
        raise ValueError("Provide metadata or set download_metadata=True")
    from vrl.utils.hf_hub import HuggingFaceRepoRevision

    return str(
        HuggingFaceRepoRevision(hf_repo, "main").download_file(
            hf_file, repo_type="dataset", cache_dir=hf_cache_dir or None
        ),
    )


def iter_metadata(path: str | Path) -> Iterator[dict[str, Any]]:
    metadata_path = Path(path)
    suffixes = metadata_path.suffixes
    if suffixes[-2:] == [".tar", ".gz"] or metadata_path.suffix == ".tgz":
        with tarfile.open(metadata_path, "r:*") as archive:
            for member in archive:
                if not member.isfile():
                    continue
                if not member.name.endswith((".json", ".jsonl", ".json.gz")):
                    continue
                extracted = archive.extractfile(member)
                if extracted is None:
                    continue
                source = f"{metadata_path}:{member.name}"
                if member.name.endswith(".gz"):
                    with gzip.open(extracted, "rt", encoding="utf-8") as handle:
                        yield from _iter_json_records(handle, source=source)
                else:
                    with io.TextIOWrapper(extracted, encoding="utf-8") as handle:
                        yield from _iter_json_records(handle, source=source)
                return
        raise ValueError(f"{metadata_path}: expected a JSON or JSONL file inside the archive")

    if metadata_path.suffix == ".gz":
        with gzip.open(metadata_path, "rt", encoding="utf-8") as handle:
            yield from _iter_json_records(handle, source=str(metadata_path))
        return

    if metadata_path.suffix in {".json", ".jsonl"}:
        with metadata_path.open(encoding="utf-8") as handle:
            yield from _iter_json_records(handle, source=str(metadata_path))
        return

    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    yield from _iter_json_payload(payload, source=str(metadata_path))


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


# ── Manifest row balancing ────────────────────────────────────────────────


def metadata_counts(
    rows: Sequence[Mapping[str, Any]],
    *,
    key: str,
    default: str = "unknown",
) -> dict[str, int]:
    counter = Counter(str((row.get("metadata") or {}).get(key, default)) for row in rows)
    return dict(sorted(counter.items()))


def proportional_group_counts[GroupKey](
    group_sizes: Mapping[GroupKey, int],
    *,
    limit: int,
) -> dict[GroupKey, int]:
    total = sum(group_sizes.values())
    if total <= 0 or limit <= 0:
        return {key: 0 for key in group_sizes}
    raw = {key: min(size, limit * size / total) for key, size in group_sizes.items()}
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


def interleave_manifest_rows(
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


# ── Safety prompts ────────────────────────────────────────────────────────

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

    context_tags = tuple(POSE_TAGS) + PROMPT_ANCHOR_TAGS + CLOTHING_TAGS + SCENE_TAGS
    candidates: list[dict[str, Any]] = []
    scanned = 0
    seen_prompts: set[str] = set()
    for row in iter_metadata(metadata_path):
        scanned += 1
        if max_metadata_rows is not None and scanned > max_metadata_rows:
            break
        tags = normalize_tags(row)
        tag_set = set(tags)
        if any(bool(row.get(key)) for key in ("is_deleted", "is_flagged", "is_banned")):
            continue
        if tag_set & SAFETY_EXCLUDED_TAGS:
            continue
        # The rating field, else a rating:<x> tag.
        rating = _normalize_rating(row.get("rating"))
        if rating is None:
            for tag in tag_set:
                if tag.startswith("rating:"):
                    rating = _normalize_rating(tag.split(":", 1)[1])
                    if rating is not None:
                        break
        if rating not in target_ratings:
            continue
        if record_score(row) < min_score:
            continue
        nsfw_tags = dedupe_text(
            [f"rating:{rating}", *(tag for tag in tags if tag in SAFETY_RISK_TAGS)],
        )
        risk_tags = [tag for tag in nsfw_tags if not tag.startswith("rating:")]
        if len(risk_tags) < min_risk_tags:
            continue
        parts: list[str] = [f"rating:{rating}", "adult"]
        for tag in SUBJECT_PROMPT_TAGS:
            if tag in tag_set:
                parts.append(tag)
                break
        if "solo" in tag_set:
            parts.append("solo")
        parts.extend(nsfw_tags)
        parts.extend(tag for tag in context_tags if tag in tag_set)
        prompt = ", ".join(dedupe_text(parts)[: max(1, prompt_tag_limit)])
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
    """Eval is proportional per rating (or category); ordering interleaves the same key."""

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        metadata = row.get("metadata") or {}
        key = str(metadata.get("rating") or metadata.get("category") or "unknown")
        groups[key].append(dict(row))
    eval_counts = proportional_group_counts(
        {key: len(value) for key, value in groups.items()}, limit=eval_limit
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
    write_json(output, report)


def _nsfw_tag_counts(rows: Sequence[Mapping[str, Any]], *, limit: int) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for row in rows:
        metadata = row.get("metadata") or {}
        for tag in metadata.get("nsfw_tags", []) or []:
            counter[str(tag)] += 1
    return dict(counter.most_common(limit))


def _normalize_rating(value: Any) -> str | None:
    text = str(value or "").strip().lower()
    return SAFETY_RATING_ALIASES.get(text)


# ── Command line ──────────────────────────────────────────────────────────


def build_default_manifests(
    *,
    metadata: str | Path | None = None,
    download_metadata: bool = True,
    hf_cache_dir: str | Path | None = None,
) -> None:
    resolved = resolve_metadata_path(
        metadata=metadata,
        download_metadata=download_metadata,
        hf_repo=DANBOORU_REPO_ID,
        hf_file=DANBOORU_METADATA_FILE,
        hf_cache_dir=hf_cache_dir,
    )
    build_safety_prompts(metadata=resolved)


def manifest_setup_hints() -> tuple[tuple[str, tuple[str, ...]], ...]:
    return (("manifests/danbooru/safety/", (ANIME_SAFETY_PROMPTS_COMMAND,)),)


def main(argv: Sequence[str] | None = None) -> None:
    if argv is None:
        import sys

        argv = sys.argv[1:]
    if not argv:
        build_default_manifests()
        return
    parser = argparse.ArgumentParser(
        description="Build Danbooru-derived anime datasets in place.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    register(subparsers)
    args = parser.parse_args(argv)
    args.func(args)


def register(subparsers: Any) -> None:
    safety = subparsers.add_parser(ANIME_SAFETY_PROMPTS_COMMAND)
    safety.add_argument("--metadata", type=Path, default=None)
    safety.set_defaults(func=_cmd_anime_safety_prompts)


def _cmd_anime_safety_prompts(args: argparse.Namespace) -> None:
    build_safety_prompts(
        metadata=args.metadata,
        download_metadata=args.metadata is None,
    )


__all__ = ["main", "manifest_setup_hints", "register"]

if __name__ == "__main__":
    main()
