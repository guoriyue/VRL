"""SigLIP text-image audit of GPT reference images: each image's best-matching prompt must be its own.
usage: audit_gpt_refs.py <prompts.jsonl> <refs_dir> <n_prompts> -> prints counts, writes <refs_dir>/audit.json and regen list"""

import json
import os
import sys

import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

torch.set_num_threads(16)
prompts, refs, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
name, rev = "google/siglip-so400m-patch14-384", "9fdffc58afc957d1a03a25b10dba0329ab15c2a3"
proc = AutoProcessor.from_pretrained(name, revision=rev)
m = AutoModel.from_pretrained(name, revision=rev).eval()


def vec(o):
    return (
        o
        if torch.is_tensor(o)
        else (
            o.pooler_output
            if getattr(o, "pooler_output", None) is not None
            else o.last_hidden_state.mean(1)
        )
    )


rows = [json.loads(l) for l in open(prompts)][:n]
ids = [i for i in range(n) if os.path.exists(f"{refs}/{i:04d}.png")]
with torch.no_grad():
    t = proc(
        text=[rows[i]["prompt"][:200] for i in ids],
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    T = vec(m.get_text_features(**t))
    T = T / T.norm(dim=-1, keepdim=True)
    I = torch.stack(
        [
            vec(
                m.get_image_features(
                    **proc(
                        images=Image.open(f"{refs}/{i:04d}.png").convert("RGB").resize((384, 384)),
                        return_tensors="pt",
                    )
                )
            )[0]
            for i in ids
        ]
    )
    I = I / I.norm(dim=-1, keepdim=True)
S = (I @ T.T).numpy()
own = np.diag(S)
best = S.argmax(1)
mis = [ids[k] for k in range(len(ids)) if best[k] != k]
weak = [ids[k] for k in range(len(ids)) if own[k] < 0.10 or (S[k].max() - own[k]) > 0.03]
regen = sorted(set(mis) | set(weak))
print(f"AUDIT total {len(ids)} mismatched {len(mis)} weak {len(weak)} regen {len(regen)}: {regen}")
json.dump(
    {
        "ids": ids,
        "best": [int(ids[b]) for b in best],
        "own": own.tolist(),
        "bestsim": S.max(1).tolist(),
    },
    open(f"{refs}/audit.json", "w"),
)
json.dump(regen, open(f"{refs}/regen_ids.json", "w"))
