"""Architecture guard for the continuous batch-identity contract.

SPRINT_continuous_stage_contracts_and_baseline T0: batch identity has exactly
one construction site (the producer). Trainer and reward code must never
re-derive ``batch_id``/``group_slot`` — they only read the identity carried by
the item and its stats. Until the versioned lookahead (Sprint 2) gives
``batch_id`` a selection consumer, these tests are its validation consumer.
"""

from __future__ import annotations

import re
from pathlib import Path

import vrl

_SRC_ROOT = Path(vrl.__file__).resolve().parent

# Assignment or keyword construction of an identity field. Word-prefix
# boundary keeps unrelated names (continuous_batch_id=...) out of scope.
_IDENTITY_WRITE = re.compile(r"(?<![\w.])(batch_id|group_slot)\s*=")


def _python_sources(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*.py") if "__pycache__" not in path.parts)


def test_continuous_item_is_constructed_only_by_the_producer() -> None:
    offenders = [
        path
        for path in _python_sources(_SRC_ROOT)
        if "ContinuousRolloutItem(" in path.read_text(encoding="utf-8")
        and path.name != "producer.py"
    ]
    assert not offenders, (
        "ContinuousRolloutItem must have one production construction site "
        f"(the producer); also constructed in: {offenders}"
    )


def test_trainer_and_reward_never_write_batch_identity() -> None:
    offenders: list[str] = []
    for subpackage in ("trainers", "rewards"):
        for path in _python_sources(_SRC_ROOT / subpackage):
            for line_number, line in enumerate(
                path.read_text(encoding="utf-8").splitlines(),
                start=1,
            ):
                if _IDENTITY_WRITE.search(line):
                    offenders.append(f"{path}:{line_number}: {line.strip()}")
    assert not offenders, (
        "trainer/reward code must not re-derive continuous batch identity "
        "(read it from the rollout item instead):\n" + "\n".join(offenders)
    )
