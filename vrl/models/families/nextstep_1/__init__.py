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

# Deliberately exports nothing. The family registry dispatches by dotted
# submodule path (vrl/models/families/registry.py), so a package-root re-export is a
# second surface nothing imports; keeping this module empty is also what stops
# config discovery from pulling the torch-backed model runtime.
__all__: list[str] = []
