"""Manifest row balancing and reporting primitives shared by Danbooru datasets."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from typing import Any


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


def split_rows_proportionally[GroupKey](
    rows: Sequence[Mapping[str, Any]],
    *,
    group_key: Callable[[Mapping[str, Any]], GroupKey],
    interleave_bucket: Callable[[GroupKey], str],
    train_limit: int,
    eval_limit: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split rows into (train, eval) with eval drawn proportionally from each group.

    ``group_key`` decides the proportional-sampling groups; ``interleave_bucket``
    maps a group to the bucket the final train/eval order round-robins over.
    """

    groups: dict[GroupKey, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[group_key(row)].append(dict(row))

    eval_counts = proportional_group_counts(
        {key: len(value) for key, value in groups.items()},
        limit=eval_limit,
    )
    train_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    eval_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for key, group_rows in groups.items():
        eval_count = eval_counts.get(key, 0)
        bucket = interleave_bucket(key)
        eval_groups[bucket].extend(group_rows[:eval_count])
        train_groups[bucket].extend(group_rows[eval_count:])
    return (
        interleave_manifest_rows(train_groups, limit=train_limit),
        interleave_manifest_rows(eval_groups, limit=eval_limit),
    )
