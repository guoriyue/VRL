"""Janus-Pro family — DeepSeek's autoregressive multimodal model.

Currently supports Janus-Pro-1B for text-to-image generation under
the visual-rl GRPO pipeline.

Requires the upstream package ``deepseek-ai/Janus`` (not on PyPI):
    git clone https://github.com/deepseek-ai/Janus
    cd Janus && pip install -e .
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

# Public transition table imported by model/runtime during package initialization.
JANUS_R1_SEGMENTS = ("initial_image", "selfcheck_text", "final_image")

if TYPE_CHECKING:
    from vrl.models.families.janus_pro.config import (
        JanusProConfig as JanusProConfig,
    )
    from vrl.models.families.janus_pro.config import (
        JanusProModelSection as JanusProModelSection,
    )
    from vrl.models.families.janus_pro.model import (
        JanusProModel as JanusProModel,
    )
    from vrl.models.families.janus_pro.model import (
        image_token_logits_from_hidden as image_token_logits_from_hidden,
    )
    from vrl.models.families.janus_pro.runtime import (
        JanusProChunkExecutor as JanusProChunkExecutor,
    )
    from vrl.models.families.janus_pro.runtime import (
        JanusProR1ChunkExecutor as JanusProR1ChunkExecutor,
    )
    from vrl.models.families.janus_pro.runtime import (
        JanusProR1ChunkGatherer as JanusProR1ChunkGatherer,
    )


# Public lazy-import boundary: config discovery must not import the model runtime.
_PUBLIC_EXPORTS = {
    "JanusProChunkExecutor": (
        "vrl.models.families.janus_pro.runtime",
        "JanusProChunkExecutor",
    ),
    "JanusProConfig": ("vrl.models.families.janus_pro.config", "JanusProConfig"),
    "JanusProModel": ("vrl.models.families.janus_pro.model", "JanusProModel"),
    "JanusProModelSection": (
        "vrl.models.families.janus_pro.config",
        "JanusProModelSection",
    ),
    "JanusProR1ChunkExecutor": (
        "vrl.models.families.janus_pro.runtime",
        "JanusProR1ChunkExecutor",
    ),
    "JanusProR1ChunkGatherer": (
        "vrl.models.families.janus_pro.runtime",
        "JanusProR1ChunkGatherer",
    ),
    "image_token_logits_from_hidden": (
        "vrl.models.families.janus_pro.model",
        "image_token_logits_from_hidden",
    ),
}

__all__ = ["JANUS_R1_SEGMENTS", *_PUBLIC_EXPORTS]


def __getattr__(name: str) -> Any:
    """Load a public Janus-Pro symbol only when requested."""

    try:
        module_name, symbol_name = _PUBLIC_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name), symbol_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
