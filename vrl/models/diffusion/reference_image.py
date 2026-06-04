"""Shared reference-image loading helper for diffusion runtimes."""

from __future__ import annotations

from typing import Any


def load_reference_image(reference_image: Any) -> Any:
    """Load a reference image path into an RGB PIL image; pass non-string/empty values through unchanged."""
    if not isinstance(reference_image, str) or not reference_image:
        return reference_image
    from PIL import Image

    return Image.open(reference_image).convert("RGB")


__all__ = ["load_reference_image"]
