"""Build a diffusers Qwen-Image-2.1 directory whose transformer comes from a BF16 GGUF file.

Used on 2026-10-09 for 0xSojalSec/Qwen-Image-2.1-Uncensored-HF@d94be5d7 (qwen-image-2.1-UC-BF16.gguf,
sha256 f151c683...). Its 297 tensor names match the upstream diffusers transformer exactly; only the 128
attention to_q/to_k/to_v/to_out.0 tensors differ (relative difference 1e-4 to 4e-4), and the text encoder
and VAE are byte-identical to upstream b3179ad3. Everything except the transformer is symlinked.

usage: convert_uc_gguf_to_diffusers.py <upstream_snapshot_dir> <bf16.gguf> <out_dir>
"""

import json
import os
import shutil
import sys

import numpy as np
import torch
from gguf import GGUFReader
from safetensors import safe_open
from safetensors.torch import save_file


def gguf_bf16_to_torch(tensor) -> torch.Tensor:
    raw = np.array(tensor.data).view(np.uint16)
    shape = list(reversed([int(x) for x in tensor.shape]))
    return torch.from_numpy(raw.astype(np.int16)).view(torch.bfloat16).reshape(shape).clone()


def main() -> None:
    base, gguf_path, out = sys.argv[1:4]
    os.makedirs(os.path.join(out, "transformer"), exist_ok=True)
    for name in os.listdir(base):
        if name == "transformer":
            continue
        dst = os.path.join(out, name)
        if not os.path.lexists(dst):
            os.symlink(os.path.realpath(os.path.join(base, name)), dst)

    tensors = {t.name: t for t in GGUFReader(gguf_path).tensors}
    index_path = os.path.join(
        base, "transformer", "diffusion_pytorch_model.safetensors.index.json"
    )
    weight_map = json.load(open(index_path))["weight_map"]
    if set(weight_map) != set(tensors):
        raise SystemExit("GGUF tensor names do not match the upstream transformer")
    shards: dict[str, list[str]] = {}
    for key, shard in weight_map.items():
        shards.setdefault(shard, []).append(key)
    for shard, keys in shards.items():
        upstream = safe_open(os.path.join(base, "transformer", shard), "pt")
        out_tensors = {
            k: gguf_bf16_to_torch(tensors[k]).to(upstream.get_tensor(k).dtype) for k in keys
        }
        save_file(out_tensors, os.path.join(out, "transformer", shard), metadata={"format": "pt"})
        print("wrote", shard, len(keys), flush=True)
    for name in ("config.json", "diffusion_pytorch_model.safetensors.index.json"):
        shutil.copy(
            os.path.join(base, "transformer", name), os.path.join(out, "transformer", name)
        )


if __name__ == "__main__":
    main()
