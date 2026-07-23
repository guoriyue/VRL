"""NextStep-1 family — StepFun's continuous-token autoregressive model.

NextStep-1 is a 14B AR transformer paired with a 157M flow-matching head
for continuous image tokens. ICLR 2026 Oral.

Requires the upstream package ``stepfun-ai/NextStep-1`` (not on PyPI):
    git clone https://github.com/stepfun-ai/NextStep-1
    cd NextStep-1 && pip install -e .

The wrapper here mirrors ``vrl.models.families.janus_pro`` so the same
``OnlineTrainer + TokenGRPO`` machinery works without changes — the only
substantive difference is that "logits" become per-token Gaussian
log-probabilities (continuous tokens, no codebook).
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vrl.models.families.nextstep_1.config import (
        NextStep1Config as NextStep1Config,
    )
    from vrl.models.families.nextstep_1.config import (
        NextStep1ModelSection as NextStep1ModelSection,
    )
    from vrl.models.families.nextstep_1.model import (
        NextStep1Model as NextStep1Model,
    )
    from vrl.models.families.nextstep_1.runtime import (
        NextStep1ChunkExecutor as NextStep1ChunkExecutor,
    )
    from vrl.models.families.nextstep_1.runtime import (
        NextStep1ChunkGatherer as NextStep1ChunkGatherer,
    )


# Public lazy-import boundary: config discovery must not import the model runtime.
_PUBLIC_EXPORTS = {
    "NextStep1ChunkExecutor": (
        "vrl.models.families.nextstep_1.runtime",
        "NextStep1ChunkExecutor",
    ),
    "NextStep1ChunkGatherer": (
        "vrl.models.families.nextstep_1.runtime",
        "NextStep1ChunkGatherer",
    ),
    "NextStep1Config": (
        "vrl.models.families.nextstep_1.config",
        "NextStep1Config",
    ),
    "NextStep1Model": ("vrl.models.families.nextstep_1.model", "NextStep1Model"),
    "NextStep1ModelSection": (
        "vrl.models.families.nextstep_1.config",
        "NextStep1ModelSection",
    ),
}

__all__ = list(_PUBLIC_EXPORTS)


def __getattr__(name: str) -> Any:
    """Load a public NextStep-1 symbol only when requested."""

    try:
        module_name, symbol_name = _PUBLIC_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name), symbol_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
