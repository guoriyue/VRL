"""Run the Bazel-built flash-attn on a GPU and compare with torch attention.

Executed through //third_party/magi_1:python (or its zip) on an Ampere/Ada/
Hopper host: flash-attn 2.4.2 ships sm_80/sm_90 kernels only.
"""

import json
import os
import pathlib
import re
import sys

import flash_attn
import torch
from flash_attn import flash_attn_func


def main() -> None:
    maps = pathlib.Path("/proc/self/maps").read_text()
    host_cuda = sorted(
        {m for m in re.findall(r"\S+/lib(?:cudart|cublas|cudnn)\.so\S*", maps) if "/usr/" in m}
    )
    torch.manual_seed(0)
    q, k, v = (torch.randn(2, 128, 8, 64, device="cuda", dtype=torch.bfloat16) for _ in range(3))
    out = flash_attn_func(q, k, v, causal=True)
    reference = torch.nn.functional.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True
    ).transpose(1, 2)
    max_abs_diff = (out.float() - reference.float()).abs().max().item()
    report = {
        "host": os.uname().nodename,
        "executable": sys.executable,
        "torch": torch.__version__,
        "flash_attn": flash_attn.__version__,
        "flash_attn_file": flash_attn.__file__,
        "gpu": torch.cuda.get_device_name(0),
        "capability": ".".join(map(str, torch.cuda.get_device_capability(0))),
        "max_abs_diff_vs_sdpa": max_abs_diff,
        "host_cuda_libraries_mapped": host_cuda,
    }
    print(json.dumps(report, indent=2))
    if host_cuda or max_abs_diff > 2e-2:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
