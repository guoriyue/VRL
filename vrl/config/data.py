"""Shared resolution for the public data-loader contract."""

from __future__ import annotations

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


__all__ = ["DataLoaderName", "resolve_data_loader"]
