"""A failed persistence step cannot masquerade as a resumable save."""

import os
import stat

import pytest

from tests.trainers._checkpoint_helpers import UNIT_IDENTITY, _Bundle, _Trainer
from vrl.trainers.checkpointing import (
    TRAINING_CHECKPOINT_NAME,
    TrainingCheckpoint,
    save_training_checkpoint,
)


def test_failed_staging_directory_flush_preserves_previous_published_checkpoint(
    tmp_path, monkeypatch
):
    target = tmp_path / "checkpoint-1"
    save_training_checkpoint(
        target,
        trainer=_Trainer(),
        bundle=_Bundle(),
        family="unit",
        model_identity=UNIT_IDENTITY,
        progress={"global_step": 1},
        rng_state={},
    )
    original = (target / TRAINING_CHECKPOINT_NAME).read_bytes()
    fsync = os.fsync

    def fail_directory_flush(descriptor):
        if stat.S_ISDIR(os.fstat(descriptor).st_mode):
            raise OSError("injected directory flush failure")
        return fsync(descriptor)

    monkeypatch.setattr(os, "fsync", fail_directory_flush)
    with pytest.raises(OSError, match="directory flush failure"):
        save_training_checkpoint(
            target,
            trainer=_Trainer(),
            bundle=_Bundle(),
            family="unit",
            model_identity=UNIT_IDENTITY,
            progress={"global_step": 2},
            rng_state={},
        )
    assert (target / TRAINING_CHECKPOINT_NAME).read_bytes() == original
    assert TrainingCheckpoint.load(target).payload["progress"]["global_step"] == 1
    assert not list(tmp_path.glob("checkpoint-1.tmp-*"))
