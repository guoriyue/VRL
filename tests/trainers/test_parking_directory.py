"""distributed.resources.trainer_parking_directory: the trainer's parking destination.

The config accepts only an absolute node-local path; the parking layer refuses
a directory that is missing or sits on a RAM-backed mount, since tmpfs would
keep the "disk" files in host RAM and defeat the point silently. A real CPU
park shows the frozen shards move into files under the directory and come back
byte-identical, with the files gone after restore.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from vrl.config.schema import parse_config
from vrl.models.frozen_disk_store import FrozenDiskStore
from vrl.models.parking import TrainingMemoryState, TrainingStateParking
from vrl.ray.resources import DistributedResourceConfig
from vrl.trainers.strategy import FSDPStrategy, build_strategy


def test_parking_directory_is_optional_and_must_be_absolute() -> None:
    assert DistributedResourceConfig().trainer_parking_directory is None
    assert (
        DistributedResourceConfig(
            trainer_parking_directory="/mnt/nvme/vrl"
        ).trainer_parking_directory
        == "/mnt/nvme/vrl"
    )
    for bad in ("", "   ", "relative/dir"):
        with pytest.raises(ValueError, match="trainer_parking_directory"):
            DistributedResourceConfig(trainer_parking_directory=bad)


def test_parking_directory_reaches_the_fsdp_strategy(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_init(self, context, **kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(FSDPStrategy, "__init__", fake_init)
    root = parse_config(
        OmegaConf.create(
            {
                "model": {"family": "sd3_5"},
                "distributed": {
                    "training": {"strategy": "fsdp", "fsdp": {"mesh": ["dp_shard"]}},
                    "resources": {"trainer_parking_directory": "/mnt/nvme/vrl-parking"},
                },
            }
        )
    )
    context = SimpleNamespace(strategy="fsdp", rank=0, world_size=1, device="cpu")
    build_strategy(root, context)

    assert captured["parking_directory"] == "/mnt/nvme/vrl-parking"


def _cpu_training_state() -> tuple[torch.nn.Module, TrainingMemoryState]:
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Linear(4, 2))
    model[0].requires_grad_(False)  # the frozen shard that goes to disk
    return model, TrainingMemoryState(
        model=model,
        ref_model=None,
        optimizer=None,
        ema=None,
        grad_scaler=None,
        device=torch.device("cpu"),
    )


def test_disk_parking_refuses_missing_and_ram_backed_directories(tmp_path, monkeypatch) -> None:
    mounts = tmp_path / "mounts"
    nvme = tmp_path / "nvme"
    shm = tmp_path / "shm"
    nvme.mkdir()
    (shm / "inner").mkdir(parents=True)
    mounts.write_text(
        f"/dev/root / ext4 rw 0 0\nnvme0n1 {nvme} ext4 rw 0 0\ntmpfs {shm} tmpfs rw 0 0\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(FrozenDiskStore, "_MOUNTS", str(mounts))
    _, state = _cpu_training_state()

    with pytest.raises(ValueError, match="does not exist"):
        TrainingStateParking(
            state, parking_directory=str(tmp_path / "absent")
        ).park_training_state()
    with pytest.raises(ValueError, match="tmpfs mount"):
        TrainingStateParking(state, parking_directory=str(shm)).park_training_state()
    # A directory under the tmpfs mount is caught by the longest-prefix match.
    with pytest.raises(ValueError, match="tmpfs mount"):
        TrainingStateParking(state, parking_directory=str(shm / "inner")).park_training_state()


def test_disk_parking_moves_frozen_shards_to_files_and_restores_them(
    tmp_path, monkeypatch
) -> None:
    mounts = tmp_path / "mounts"
    nvme = tmp_path / "nvme"
    nvme.mkdir()
    mounts.write_text(f"/dev/root / ext4 rw 0 0\nnvme0n1 {nvme} ext4 rw 0 0\n", encoding="utf-8")
    monkeypatch.setattr(FrozenDiskStore, "_MOUNTS", str(mounts))
    model, state = _cpu_training_state()
    before = {name: tensor.clone() for name, tensor in model.state_dict().items()}
    frozen_weight = model[0].weight
    trainable_weight = model[1].weight

    parking = TrainingStateParking(state, parking_directory=str(nvme))
    parking.park_training_state()

    files = sorted(path for path in nvme.rglob("*.bin"))
    assert len(files) == 2  # frozen weight and bias, one file each
    assert (
        model[0].weight is frozen_weight
        and model[0].weight.data_ptr() != before["0.weight"].data_ptr()
    )
    assert model[1].weight is trainable_weight  # trainable rows stay in host RAM
    for name, tensor in model.state_dict().items():
        assert torch.equal(tensor, before[name]), name

    parking.restore()

    assert not list(nvme.iterdir())
    for name, tensor in model.state_dict().items():
        assert torch.equal(tensor, before[name]), name


def test_partial_file_mapping_failure_can_restore_and_clean_up(tmp_path, monkeypatch) -> None:
    """A failed second file must not lose the first parameter or leak its file."""
    mounts = tmp_path / "mounts"
    mounts.write_text("/dev/root / ext4 rw 0 0\n", encoding="utf-8")
    monkeypatch.setattr(FrozenDiskStore, "_MOUNTS", str(mounts))
    directory = tmp_path / "storage"
    directory.mkdir()
    model, state = _cpu_training_state()
    before = {name: tensor.clone() for name, tensor in model.state_dict().items()}
    parameters = list(model.parameters())
    from_file = torch.from_file
    calls = 0

    def fail_second_mapping(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("disk full")
        return from_file(*args, **kwargs)

    monkeypatch.setattr(torch, "from_file", fail_second_mapping)
    parking = TrainingStateParking(state, parking_directory=str(directory))
    with pytest.raises(OSError, match="disk full"):
        parking.park_training_state()
    assert len(list(directory.rglob("*.bin"))) == 1

    parking.restore()
    parking.restore()

    assert not list(directory.iterdir())
    assert all(old is current for old, current in zip(parameters, model.parameters(), strict=True))
    for name, tensor in model.state_dict().items():
        assert torch.equal(tensor, before[name]), name
