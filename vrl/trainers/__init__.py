"""RL trainers: training loop orchestration and weight sync.

Deliberately exports nothing. The package root previously mirrored eleven
symbols spanning both the torch-free config contracts and the torch-heavy
trainer/weight-sync runtime, so no import of it could be cheap and nothing
imported it. The real boundary is one level down: ``vrl.trainers.online``
defers ``OnlineTrainer`` behind its config, and ``vrl.trainers.core.types`` is
torch-free on its own. Import from the owning module.
"""

from __future__ import annotations

__all__: list[str] = []
