"""LlamaGen family — FoundationVision's llama-style autoregressive T2I model.

Currently supports the t2i_XL_stage1_256 checkpoint (775M GPT-XL + VQ-16
tokenizer, flan-t5-xl caption conditioning) under the visual-rl GRPO pipeline.

Upstream has no HuggingFace integration; the GPT and VQGAN definitions are
vendored under ``vendor/`` (MIT, FoundationVision) — see the vendor headers
for why the GPT cannot be mapped onto ``LlamaForCausalLM``.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vrl.models.families.llamagen.config import (
        LLAMAGEN_CAPTION_TOKEN_NUM as LLAMAGEN_CAPTION_TOKEN_NUM,
    )
    from vrl.models.families.llamagen.config import (
        LLAMAGEN_IMAGE_TOKEN_NUM as LLAMAGEN_IMAGE_TOKEN_NUM,
    )
    from vrl.models.families.llamagen.config import LlamaGenConfig as LlamaGenConfig
    from vrl.models.families.llamagen.config import (
        LlamaGenModelSection as LlamaGenModelSection,
    )
    from vrl.models.families.llamagen.model import (
        LLAMAGEN_IMAGE_VOCAB_SIZE as LLAMAGEN_IMAGE_VOCAB_SIZE,
    )
    from vrl.models.families.llamagen.model import LlamaGenModel as LlamaGenModel
    from vrl.models.families.llamagen.model import (
        LlamaGenReplayModel as LlamaGenReplayModel,
    )
    from vrl.models.families.llamagen.runtime import (
        LlamaGenChunkExecutor as LlamaGenChunkExecutor,
    )


# Public lazy-import boundary: config discovery must not import the model runtime.
_PUBLIC_EXPORTS = {
    "LLAMAGEN_CAPTION_TOKEN_NUM": (
        "vrl.models.families.llamagen.config",
        "LLAMAGEN_CAPTION_TOKEN_NUM",
    ),
    "LLAMAGEN_IMAGE_TOKEN_NUM": (
        "vrl.models.families.llamagen.config",
        "LLAMAGEN_IMAGE_TOKEN_NUM",
    ),
    "LLAMAGEN_IMAGE_VOCAB_SIZE": (
        "vrl.models.families.llamagen.model",
        "LLAMAGEN_IMAGE_VOCAB_SIZE",
    ),
    "LlamaGenChunkExecutor": (
        "vrl.models.families.llamagen.runtime",
        "LlamaGenChunkExecutor",
    ),
    "LlamaGenConfig": ("vrl.models.families.llamagen.config", "LlamaGenConfig"),
    "LlamaGenModel": ("vrl.models.families.llamagen.model", "LlamaGenModel"),
    "LlamaGenModelSection": (
        "vrl.models.families.llamagen.config",
        "LlamaGenModelSection",
    ),
    "LlamaGenReplayModel": (
        "vrl.models.families.llamagen.model",
        "LlamaGenReplayModel",
    ),
}

__all__ = list(_PUBLIC_EXPORTS)


def __getattr__(name: str) -> Any:
    """Load a public LlamaGen symbol only when requested."""

    try:
        module_name, symbol_name = _PUBLIC_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name), symbol_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
