"""Shared resolution for the public data-loader contract."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal, cast, get_args

DataLoaderName = Literal[
    "pickapic_preference",
    "prompt_manifest",
    "prompt_image_manifest",
]


def resolve_data_loader(
    loader: object | None,
    preprocessing_format: object | None,
) -> DataLoaderName:
    """Resolve one loader name and reject explicit format conflicts."""

    fmt = "" if preprocessing_format is None else str(preprocessing_format)
    if loader is None:
        return "prompt_image_manifest" if fmt == "image_caption_jsonl" else "prompt_manifest"

    name = str(loader)
    allowed = frozenset(get_args(DataLoaderName))
    if name not in allowed:
        raise ValueError(
            f"unknown data.loader={name!r}; expected one of {sorted(allowed)}",
        )
    if name == "prompt_image_manifest" and fmt != "image_caption_jsonl":
        raise ValueError(
            "data.loader='prompt_image_manifest' requires "
            "data.preprocessing.format='image_caption_jsonl'",
        )
    if name == "prompt_manifest" and fmt == "image_caption_jsonl":
        raise ValueError(
            "data.preprocessing.format='image_caption_jsonl' requires "
            "data.loader='prompt_image_manifest'",
        )
    return cast("DataLoaderName", name)


def manifest_sources(manifest: object) -> dict[str, int | None]:
    """Normalize ``data.manifest`` into ``{path: count}``.

    Two spellings, one shape for every reader (loader, config validation, the
    dataset bootstrap): a single path string, or a mapping of path to how many
    prompts to draw from it — the recipe-level way to say "85% anatomy, 15%
    safety-stress" without materializing a mixed manifest on disk. ``None``
    count means every row in the file.
    """

    if isinstance(manifest, str):
        return {manifest: None}
    if isinstance(manifest, Mapping):
        if not manifest:
            raise ValueError("data.manifest mapping must name at least one manifest")
        sources: dict[str, int | None] = {}
        for path, count in manifest.items():
            if not isinstance(count, int) or isinstance(count, bool) or count < 1:
                raise ValueError(
                    f"data.manifest[{path!r}] must be a positive prompt count, got {count!r}",
                )
            sources[str(path)] = count
        return sources
    raise ValueError(
        "data.manifest must be a manifest path or a {path: prompt count} mapping, "
        f"got {type(manifest).__name__}",
    )


__all__ = ["DataLoaderName", "manifest_sources", "resolve_data_loader"]
