"""Prove a delivered VRL artifact works on nodes that never saw this repo.

Run from the python zip (``bazel build --build_python_zip //tools/delivery:node_probe``)
on the target host. It checks the interpreter and CUDA libraries come from the
artifact, runs torch on every visible GPU, then launches one torchrun rank per
GPU that imports vrl and completes an NCCL all-reduce. With ``--nnodes`` and a
``--master-addr`` the same command on every node forms one multi-node job;
with ``--ray-address`` it instead runs one task per node of a Ray cluster.

Set ``RULES_PYTHON_EXTRACT_ROOT`` to the same path on every node so the
extracted interpreter has one path cluster-wide (Ray workers reuse it).
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import subprocess
import sys

import torch

from tools.python.nvidia_preload import preload


def _host_cuda_libraries() -> list[str]:
    maps = pathlib.Path("/proc/self/maps").read_text()
    return sorted(
        {m for m in re.findall(r"\S+/lib(?:cudart|cublas|cudnn)\.so\S*", maps) if "/usr/" in m}
    )


def _torchrun(args: argparse.Namespace, count: int) -> dict[str, object]:
    command = [sys.executable, "-m", "torch.distributed.run", f"--nproc-per-node={count}"]
    if args.nnodes > 1:
        command += [
            f"--nnodes={args.nnodes}",
            f"--node-rank={args.node_rank}",
            f"--master-addr={args.master_addr}",
            f"--master-port={args.master_port}",
        ]
    else:
        command += ["--standalone", "--nnodes=1"]
    command += ["--max-restarts=0", "--module", "tools.delivery.node_rank"]
    result = subprocess.run(command, capture_output=True, text=True, timeout=900)
    report: dict[str, object] = {
        "torchrun_returncode": result.returncode,
        "torchrun_ranks": sorted(re.findall(r"rank \d+/\d+ nccl ok on \S+", result.stdout)),
    }
    if result.returncode != 0:
        report["torchrun_stderr"] = result.stderr[-3000:]
    return report


def _ray_cluster(address: str) -> dict[str, object]:
    import ray

    ray.init(address=address)

    @ray.remote(num_gpus=1)
    def on_node() -> dict[str, str]:
        import vrl  # noqa: F401
        from tools.python.nvidia_preload import preload as node_preload

        node_preload()
        return {
            "node": ray.get_runtime_context().get_node_id()[:8],
            "host": os.uname().nodename,
            "executable": sys.executable,
            "gpu": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "host_cuda_libraries": ",".join(_host_cuda_libraries()),
        }

    nodes = [n for n in ray.nodes() if n["Alive"]]
    gpus = int(sum(n["Resources"].get("GPU", 0) for n in nodes))
    results = ray.get([on_node.remote() for _ in range(gpus)])
    ray.shutdown()
    return {
        "ray_nodes": len(nodes),
        "ray_gpus": gpus,
        "ray_tasks": sorted(results, key=lambda r: (r["host"], r["node"])),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nnodes", type=int, default=1)
    parser.add_argument("--node-rank", type=int, default=0)
    parser.add_argument("--master-addr", default="127.0.0.1")
    parser.add_argument("--master-port", type=int, default=29500)
    parser.add_argument(
        "--ray-address", default=None, help="run one task per GPU of this Ray cluster"
    )
    args = parser.parse_args()

    report: dict[str, object] = {
        "host": os.uname().nodename,
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "preloaded": len(preload()),
        "host_cuda_libraries_mapped": _host_cuda_libraries(),
    }
    count = torch.cuda.device_count()
    report["gpus"] = [torch.cuda.get_device_name(i) for i in range(count)]
    for index in range(count):
        x = torch.randn(1024, 1024, device=f"cuda:{index}")
        torch.cuda.synchronize(index)
        assert torch.isfinite((x @ x).sum()).item()
    failed = bool(report["host_cuda_libraries_mapped"])
    if args.ray_address:
        report.update(_ray_cluster(args.ray_address))
        failed |= any(r["host_cuda_libraries"] for r in report["ray_tasks"])
    elif count:
        report.update(_torchrun(args, count))
        failed |= report["torchrun_returncode"] != 0
    print(json.dumps(report, indent=2))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
