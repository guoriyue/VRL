"""Prove a delivered VRL artifact works on a node that never saw this repo.

Run from the python zip (``bazel build --build_python_zip //tools/delivery:node_probe``)
on the target host. It checks the interpreter and CUDA libraries come from the
artifact, runs torch on every visible GPU, then launches one torchrun rank per
GPU that imports vrl and completes an NCCL all-reduce.
"""

from __future__ import annotations

import json
import os
import pathlib
import re
import subprocess
import sys

import torch

from tools.python.nvidia_preload import preload


def main() -> None:
    report: dict[str, object] = {
        "host": os.uname().nodename,
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "preloaded": len(preload()),
    }
    maps = pathlib.Path("/proc/self/maps").read_text()
    host_cuda = sorted(
        {m for m in re.findall(r"\S+/lib(?:cudart|cublas|cudnn)\.so\S*", maps) if "/usr/" in m}
    )
    report["host_cuda_libraries_mapped"] = host_cuda
    count = torch.cuda.device_count()
    report["gpus"] = [torch.cuda.get_device_name(i) for i in range(count)]
    for index in range(count):
        x = torch.randn(1024, 1024, device=f"cuda:{index}")
        torch.cuda.synchronize(index)
        assert torch.isfinite((x @ x).sum()).item()
    if count:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--standalone",
                "--nnodes=1",
                f"--nproc-per-node={count}",
                "--max-restarts=0",
                "--module",
                "tools.delivery.node_rank",
            ],
            capture_output=True,
            text=True,
            timeout=600,
        )
        report["torchrun_returncode"] = result.returncode
        report["torchrun_ranks"] = sorted(re.findall(r"rank \d+/\d+ nccl ok", result.stdout))
        if result.returncode != 0:
            report["torchrun_stderr"] = result.stderr[-3000:]
    print(json.dumps(report, indent=2))
    if host_cuda or (count and report["torchrun_returncode"] != 0):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
