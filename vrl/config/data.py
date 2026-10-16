"""Shared resolution for the public data-loader contract."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

DataLoaderName = Literal[
    "layered_scenes",
    "pickapic_preference",
    "prompt_manifest",
    "prompt_image_manifest",
]


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


__all__ = ["DataLoaderName", "manifest_sources"]
