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
    """Normalize the ``layered_scenes`` loader's ``data.manifest`` into ``{root: count}``.

    Two spellings, one shape for every reader (config validation, the scene
    dataset, the cache encoder and the evaluation script): a single dataset
    root, or a mapping of root to how many of its scenes to take (the first
    ``count`` in sorted order, ``scene_cache_paths``). A ``None`` count, in
    either spelling, means every scene under the root.
    """

    if isinstance(manifest, str):
        return {manifest: None}
    if isinstance(manifest, Mapping):
        if not manifest:
            raise ValueError("data.manifest mapping must name at least one dataset root")
        sources: dict[str, int | None] = {}
        for path, count in manifest.items():
            if count is not None and (
                not isinstance(count, int) or isinstance(count, bool) or count < 1
            ):
                raise ValueError(
                    f"data.manifest[{path!r}] must be a positive scene count or null, "
                    f"got {count!r}",
                )
            sources[str(path)] = count
        return sources
    raise ValueError(
        "data.manifest must be a dataset root or a {root: scene count} mapping, "
        f"got {type(manifest).__name__}",
    )


__all__ = ["DataLoaderName", "manifest_sources"]
