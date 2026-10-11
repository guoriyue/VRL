"""distributed.resources.trainer.parking.disk_directory: the trainer's parking destination.

The config accepts only an absolute node-local path; the parking layer refuses
a directory that is missing or sits on a RAM-backed mount, since tmpfs would
keep the "disk" files in host RAM and defeat the point silently. A real CPU
park shows every parked tensor move into files under the directory and come
back byte-identical, with the files gone after restore.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from vrl.config.schema import parse_config
from vrl.models.parking import ModelParking, TrainingMemoryState, TrainingStateParking
from vrl.models.parking_disk_store import ParkingDiskStore
from vrl.ray.resources import DistributedResourceConfig, ParkingConfig
from vrl.trainers.strategy import FSDPStrategy, build_strategy


def test_parking_directory_is_optional_and_must_be_absolute() -> None:
    assert DistributedResourceConfig().trainer.parking == ParkingConfig("move", None)
    for bad in ("", "   ", "relative/dir"):
        with pytest.raises(ValueError, match="disk_directory"):
            ParkingConfig("move", bad)


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
                    "resources": {
                        "trainer": {"parking": {"disk_directory": "/mnt/nvme/vrl-parking"}}
                    },
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
    model[0].requires_grad_(False)  # frozen and trainable shards park alike
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
    monkeypatch.setattr(ParkingDiskStore, "_MOUNTS", str(mounts))
    _, state = _cpu_training_state()

    with pytest.raises(ValueError, match="does not exist"):
        TrainingStateParking(
            state, disk=ParkingDiskStore(str(tmp_path / "absent"))
        ).park_training_state()
    with pytest.raises(ValueError, match="tmpfs mount"):
        TrainingStateParking(state, disk=ParkingDiskStore(str(shm))).park_training_state()
    # A directory under the tmpfs mount is caught by the longest-prefix match.
    with pytest.raises(ValueError, match="tmpfs mount"):
        TrainingStateParking(
            state, disk=ParkingDiskStore(str(shm / "inner"))
        ).park_training_state()


def test_disk_parking_moves_every_parked_tensor_to_files_and_restores_them(
    tmp_path, monkeypatch
) -> None:
    mounts = tmp_path / "mounts"
    nvme = tmp_path / "nvme"
    nvme.mkdir()
    mounts.write_text(f"/dev/root / ext4 rw 0 0\nnvme0n1 {nvme} ext4 rw 0 0\n", encoding="utf-8")
    monkeypatch.setattr(ParkingDiskStore, "_MOUNTS", str(mounts))
    model, state = _cpu_training_state()
    before = {name: tensor.clone() for name, tensor in model.state_dict().items()}
    frozen_weight = model[0].weight
    trainable_weight = model[1].weight

    parking = TrainingStateParking(state, disk=ParkingDiskStore(str(nvme)))
    parking.park_training_state()

    files = sorted(path for path in nvme.rglob("*.bin"))
    assert len(files) == 4  # both frozen and trainable weight/bias are reclaimable
    assert (
        model[0].weight is frozen_weight
        and model[0].weight.data_ptr() != before["0.weight"].data_ptr()
    )
    assert model[1].weight is trainable_weight  # storage changes, identity does not
    for name, tensor in model.state_dict().items():
        assert torch.equal(tensor, before[name]), name

    parking.restore()
    assert not list(nvme.iterdir())
    for name, tensor in model.state_dict().items():
        assert torch.equal(tensor, before[name]), name


def test_disk_belongs_to_the_move_mechanism_not_to_the_trainer(tmp_path, monkeypatch) -> None:
    """Any role's move ledger can park to disk; the default stays in RAM."""

    mounts = tmp_path / "mounts"
    mounts.write_text("/dev/root / ext4 rw 0 0\n", encoding="utf-8")
    monkeypatch.setattr(ParkingDiskStore, "_MOUNTS", str(mounts))
    directory = tmp_path / "storage"
    directory.mkdir()
    model, _ = _cpu_training_state()
    before = {name: tensor.clone() for name, tensor in model.state_dict().items()}

    in_ram = ModelParking()
    in_ram.park(model, restore_device=torch.device("cpu"))
    assert not list(directory.iterdir())
    in_ram.restore()

    on_disk = ModelParking(disk=ParkingDiskStore(str(directory)))
    on_disk.park(model, restore_device=torch.device("cpu"))
    assert len(list(directory.rglob("*.bin"))) == 4
    on_disk.restore()

    assert not list(directory.iterdir())
    for name, tensor in model.state_dict().items():
        assert torch.equal(tensor, before[name]), name


@pytest.mark.gpu
def test_disk_parking_maps_each_tensor_straight_from_the_gpu(tmp_path, monkeypatch) -> None:
    """No anonymous host copy of the whole module: the store receives CUDA tensors."""

    mounts = tmp_path / "mounts"
    mounts.write_text("/dev/root / ext4 rw 0 0\n", encoding="utf-8")
    monkeypatch.setattr(ParkingDiskStore, "_MOUNTS", str(mounts))
    directory = tmp_path / "storage"
    directory.mkdir()
    model, _ = _cpu_training_state()
    model.cuda()
    before = {name: tensor.cpu() for name, tensor in model.state_dict().items()}
    stored_from: list[str] = []
    store = ParkingDiskStore.store

    def recording_store(self, tensors):
        tensors = list(tensors)
        stored_from.extend(ModelParking.tensor_device(tensor).type for tensor in tensors)
        store(self, tensors)

    monkeypatch.setattr(ParkingDiskStore, "store", recording_store)
    parking = ModelParking(disk=ParkingDiskStore(str(directory)))
    parking.park(model, restore_device=torch.device("cuda"))

    assert stored_from == ["cuda"] * 4
    assert all(p.untyped_storage().filename is not None for p in model.parameters())
    parking.restore()
    assert not list(directory.iterdir())
    for name, tensor in model.state_dict().items():
        assert tensor.is_cuda and torch.equal(tensor.cpu(), before[name]), name


def test_disk_parking_preserves_full_model_adam_update(tmp_path) -> None:
    """Real master/moment/gradient mappings must survive the next optimizer step."""
    from vrl.trainers.optimizer import FP32MasterWeightOptimizer

    torch.manual_seed(9)
    model = torch.nn.Linear(4, 2, dtype=torch.bfloat16)
    reference = torch.nn.Linear(4, 2, dtype=torch.bfloat16)
    reference.load_state_dict(model.state_dict())

    def optimizer(module):
        return FP32MasterWeightOptimizer(
            module.parameters(), lambda p: torch.optim.AdamW(p, lr=1e-3)
        )

    opt, ref_opt = optimizer(model), optimizer(reference)
    x = torch.ones(2, 4, dtype=torch.bfloat16)

    def step(module, optim):
        optim.zero_grad(set_to_none=True)
        module(x).float().square().mean().backward()
        optim.prepare_gradients()
        optim.step()

    step(model, opt)
    step(reference, ref_opt)
    state = TrainingMemoryState(model, None, opt, None, None, torch.device("cpu"))
    identities = [id(p) for p in model.parameters()] + [id(p) for p in opt.parameters()]
    parking = TrainingStateParking(state, disk=ParkingDiskStore(str(tmp_path)))
    parking.park_training_state()
    # Every large optimizer tensor has file-backed storage, including masters.
    tensors = list(opt.parameters())
    tensors += [v for s in opt.state.values() for v in s.values() if isinstance(v, torch.Tensor)]
    assert tensors
    for tensor in tensors:
        assert tensor.untyped_storage().filename is not None
    parking.restore()
    assert not list(tmp_path.iterdir())
    assert identities == [id(p) for p in model.parameters()] + [id(p) for p in opt.parameters()]
    step(model, opt)
    step(reference, ref_opt)
    for actual, expected in zip(model.parameters(), reference.parameters(), strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for actual, expected in zip(opt.parameters(), ref_opt.parameters(), strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_partial_file_mapping_failure_can_restore_and_clean_up(tmp_path, monkeypatch) -> None:
    """A failed second file must not lose the first parameter or leak its file."""
    mounts = tmp_path / "mounts"
    mounts.write_text("/dev/root / ext4 rw 0 0\n", encoding="utf-8")
    monkeypatch.setattr(ParkingDiskStore, "_MOUNTS", str(mounts))
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
    parking = TrainingStateParking(state, disk=ParkingDiskStore(str(directory)))
    with pytest.raises(OSError, match="disk full"):
        parking.park_training_state()
    assert len(list(directory.rglob("*.bin"))) == 1

    parking.restore()
    parking.restore()

    assert not list(directory.iterdir())
    assert all(old is current for old, current in zip(parameters, model.parameters(), strict=True))
    for name, tensor in model.state_dict().items():
        assert torch.equal(tensor, before[name]), name
