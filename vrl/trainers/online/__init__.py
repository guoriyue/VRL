"""Online training loop public facade."""

from __future__ import annotations

from typing import TYPE_CHECKING

from vrl.utils.config import install_lazy_exports

if TYPE_CHECKING:
    from vrl.trainers.online.config import OnlineBatchPlan as OnlineBatchPlan
    from vrl.trainers.online.config import TrainerConfig as TrainerConfig
    from vrl.trainers.online.trainer import OnlineTrainer as OnlineTrainer


_PUBLIC_EXPORTS = {
    "OnlineBatchPlan": "vrl.trainers.online.config",
    "OnlineTrainer": "vrl.trainers.online.trainer",
    "TrainerConfig": "vrl.trainers.online.config",
}
install_lazy_exports(globals(), _PUBLIC_EXPORTS)
