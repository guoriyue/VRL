"""Per-rank ownership contract for the multi-rank online loop.

The training rollout is data-parallel: every rank drives its OWN colocated Ray
runtime over a disjoint prompt shard. The single-writer / collective contract the
loop keeps is:

    training rollout  -> EVERY rank generates its local prompt shard
    fixed eval        -> EVERY rank generates its local eval shard,
                         rank0 ALONE writes eval_metrics.csv
    checkpoint        -> EVERY rank runs the trainable-state gather (a collective),
                         rank0 ALONE writes the files

The single writer is always ``context.is_primary``. The only place this contract
is extracted into a standalone, unit-drivable seam is the checkpoint path
(``save_training_checkpoint``), which this module exercises directly below. The
fixed-eval / training-rollout write-gates live inline inside ``run_online_recipe``
and are covered indirectly: the per-rank ``is_primary`` flag they branch on is
pinned by ``test_primary_flag_is_rank0_only``, and the gather-before-gate ordering
is pinned by the real checkpoint test. (Re-asserting that gate inside a test-local
helper would only test the helper, not the recipe, so those shadow tests were
removed.)
"""

from __future__ import annotations

import torch

from vrl.trainers.checkpointing import TRAINING_CHECKPOINT_NAME, save_training_checkpoint
from vrl.trainers.distributed import resolve_training_context
from vrl.trainers.strategy import SingleProcessStrategy

_FSDP_CFG = {
    "distributed": {"training": {"strategy": "fsdp", "num_nodes": 1, "gpus_per_node": 2}},
}


class _GatherRecordingStrategy:
    """Fake training strategy recording the trainable-state export.

    Under FSDP2 ``export_trainable_state`` is the all-gather collective every rank
    must enter; the fake lets the test assert it ran on all ranks regardless of
    ``is_primary``.
    """

    def __init__(self) -> None:
        self.export_calls = 0

    def export_trainable_state(self, bundle: object) -> dict[str, object]:
        del bundle
        self.export_calls += 1
        return {}


class _StubTrainer:
    def state_dict(self) -> dict[str, object]:
        return {}


def _fsdp_context(rank: int):
    return resolve_training_context(
        _FSDP_CFG,
        device=torch.device("cpu"),
        env={"RANK": str(rank), "LOCAL_RANK": str(rank), "WORLD_SIZE": "2"},
    )


def test_checkpoint_gathers_on_every_rank_but_only_rank0_writes(tmp_path) -> None:
    """``save_training_checkpoint`` gathers on EVERY rank then writes on rank0 alone.

    Drives the production ``vrl.trainers.checkpointing.save_training_checkpoint`` so
    the ordering contract is actually exercised: the trainable-state export (the
    FSDP collective) runs before the ``if not is_primary: return`` gate. Deleting
    that gate (rank1 would then write a file) or moving the gather below it (rank1
    would skip the collective and deadlock FSDP) makes this test fail.
    """
    strategy = _GatherRecordingStrategy()
    for rank in range(2):
        context = _fsdp_context(rank)
        save_training_checkpoint(
            tmp_path / f"checkpoint-{rank}",
            trainer=_StubTrainer(),
            bundle=object(),
            family="test",
            progress={},
            rng_state={"seed": 0},
            strategy=strategy,
            is_primary=context.is_primary,
        )

    # Collective on all ranks: both rank0 and rank1 must enter the gather.
    assert strategy.export_calls == 2
    # Single writer: only rank0 materializes the checkpoint; rank1 returns pre-write.
    assert (tmp_path / "checkpoint-0" / TRAINING_CHECKPOINT_NAME).exists()
    assert not (tmp_path / "checkpoint-1").exists()


def test_primary_flag_is_rank0_only() -> None:
    assert _fsdp_context(0).is_primary is True
    assert _fsdp_context(1).is_primary is False


def test_barrier_seam_is_callable_and_noop_in_single_process() -> None:
    """The barrier seam exists for every rank (no-op now, real under FSDP)."""
    strategy = SingleProcessStrategy()
    for rank in range(2):
        _fsdp_context(rank)  # env resolves cleanly for each rank
        assert strategy.barrier() is None
