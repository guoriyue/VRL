"""distributed.resources.parking_directory: where a parked role's host copy lives.

The config accepts only an absolute node-local path; the parking layer refuses
a directory that is missing or sits on a RAM-backed mount, since tmpfs would
keep the "disk" files in host RAM and defeat the point silently.
"""

from __future__ import annotations

import pytest
from omegaconf import OmegaConf

from vrl.config.schema import parse_config
from vrl.models.parking import require_disk_backed_directory
from vrl.ray.resources import DistributedResourceConfig
from vrl.trainers.strategy import FSDPStrategy, build_strategy


def test_parking_directory_is_optional_and_must_be_absolute() -> None:
    assert DistributedResourceConfig().parking_directory is None
    assert DistributedResourceConfig(parking_directory="/mnt/nvme/vrl").parking_directory == (
        "/mnt/nvme/vrl"
    )
    for bad in ("", "   ", "relative/dir"):
        with pytest.raises(ValueError, match="parking_directory"):
            DistributedResourceConfig(parking_directory=bad)


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
                    "resources": {"parking_directory": "/mnt/nvme/vrl-parking"},
                },
            }
        )
    )
    from types import SimpleNamespace

    context = SimpleNamespace(strategy="fsdp", rank=0, world_size=1, device="cpu")
    build_strategy(root, context)

    assert captured["parking_directory"] == "/mnt/nvme/vrl-parking"


def test_disk_parking_refuses_missing_and_ram_backed_directories(tmp_path) -> None:
    with pytest.raises(ValueError, match="does not exist"):
        require_disk_backed_directory(str(tmp_path / "absent"))

    mounts = tmp_path / "mounts"
    nvme = tmp_path / "nvme"
    shm = tmp_path / "shm"
    nvme.mkdir()
    shm.mkdir()
    mounts.write_text(
        f"/dev/root / ext4 rw 0 0\nnvme0n1 {nvme} ext4 rw 0 0\ntmpfs {shm} tmpfs rw 0 0\n",
        encoding="utf-8",
    )

    require_disk_backed_directory(str(nvme), mounts=str(mounts))
    with pytest.raises(ValueError, match="tmpfs mount"):
        require_disk_backed_directory(str(shm), mounts=str(mounts))
    # A directory under the tmpfs mount is caught by the longest-prefix match.
    (shm / "inner").mkdir()
    with pytest.raises(ValueError, match="tmpfs mount"):
        require_disk_backed_directory(str(shm / "inner"), mounts=str(mounts))
