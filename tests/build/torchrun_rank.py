"""One torchrun rank: prove the Bazel-managed closure reaches every worker."""

import os
import pathlib
import sys

import torch
import torch.distributed as dist

import vrl


def main() -> None:
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    assert ".venv" not in pathlib.Path(sys.executable).parts, sys.executable
    assert any(part.endswith(".runfiles") for part in pathlib.Path(vrl.__file__).parts), (
        vrl.__file__
    )
    # The startup hook ran in this child too: the locked NVIDIA libraries are
    # mapped, and no host CUDA installation is.
    from tools.python.nvidia_preload import preload

    assert preload(), "sitecustomize did not preload NVIDIA libraries in the rank"
    maps = pathlib.Path("/proc/self/maps").read_text()
    host = sorted({line.split()[-1] for line in maps.splitlines() if "/usr/local/cuda" in line})
    assert not host, f"rank mapped host CUDA libraries: {host}"
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    total = torch.tensor([rank + 1], dtype=torch.int64)
    dist.all_reduce(total)
    assert int(total.item()) == world_size * (world_size + 1) // 2
    dist.destroy_process_group()
    print(f"rank {rank}/{world_size} ok", flush=True)


if __name__ == "__main__":
    main()
