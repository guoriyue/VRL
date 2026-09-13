"""One torchrun rank of the node probe: vrl importable, NCCL all-reduce on its GPU."""

import os
import pathlib
import re

import torch
import torch.distributed as dist

import vrl  # noqa: F401  (the artifact must carry the package)
from tools.python.nvidia_preload import preload


def main() -> None:
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    assert preload(), "rank did not preload the locked NVIDIA libraries"
    maps = pathlib.Path("/proc/self/maps").read_text()
    assert not [m for m in re.findall(r"\S+/libcudart\.so\S*", maps) if "/usr/" in m]
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl", rank=rank, world_size=world)
    total = torch.tensor([rank + 1.0], device="cuda")
    dist.all_reduce(total)
    assert int(total.item()) == world * (world + 1) // 2
    dist.destroy_process_group()
    print(f"rank {rank}/{world} nccl ok", flush=True)


if __name__ == "__main__":
    main()
