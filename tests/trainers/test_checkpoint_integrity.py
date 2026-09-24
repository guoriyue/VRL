"""Payload corruption and failed persistence cannot masquerade as a resumable save."""

import json
import os
import stat
import struct
import zipfile

import pytest
import torch

from tests.trainers._checkpoint_helpers import UNIT_IDENTITY, _Bundle, _Trainer
from vrl.trainers.checkpointing import (
    CHECKPOINT_META_NAME,
    TRAINING_CHECKPOINT_NAME,
    TrainingCheckpoint,
    is_complete_checkpoint,
    save_training_checkpoint,
)


def test_same_size_tensor_corruption_is_rejected_before_restore_and_legacy_still_loads(tmp_path):
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
    checkpoint = target / TRAINING_CHECKPOINT_NAME
    original = checkpoint.read_bytes()
    before = TrainingCheckpoint.load(target)
    with zipfile.ZipFile(checkpoint) as archive:
        storage = next(info for info in archive.infolist() if info.filename.endswith("/data/0"))
    # Change a real tensor-storage byte while retaining a readable, same-size
    # torch archive. The structural discovery scan alone cannot detect this.
    data = bytearray(original)
    name_length, extra_length = struct.unpack_from("<HH", data, storage.header_offset + 26)
    offset = storage.header_offset + 30 + name_length + extra_length
    data[offset] ^= 1
    checkpoint.write_bytes(data)
    assert len(data) == len(original) and is_complete_checkpoint(target)
    unchecked = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert not torch.equal(
        unchecked["model"]["owned_state"]["module"]["weight"],
        before.payload["model"]["owned_state"]["module"]["weight"],
    )
    with pytest.raises(ValueError, match="integrity mismatch"):
        TrainingCheckpoint.load(target)
    checkpoint.write_bytes(original)
    metadata_path = target / CHECKPOINT_META_NAME
    metadata = json.loads(metadata_path.read_text())
    metadata.pop("checkpoint_file_sha256")
    metadata_path.write_text(json.dumps(metadata))
    assert TrainingCheckpoint.load(target).payload["progress"]["global_step"] == 1


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
