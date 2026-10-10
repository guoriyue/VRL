"""Rewrite training prompts with Codex (text only): 16 workers, ~53 prompts each.

usage: rewrite_prompts.py manifest.jsonl out_dir [parallel]
Writes out_dir/chunk_<k>.json (input) and out_dir/rewritten_<k>.json (Codex output),
then merges into out_dir/rewritten_all.json as {prompt: rewritten}.
"""

import concurrent.futures as cf
import json
import os
import subprocess
import sys
from pathlib import Path

TASK = (
    "Read {inp} (a JSON list of text-to-image prompts, one per object). For each prompt write a rewritten prompt "
    "for a diffusion model that keeps every subject, attribute, style and medium the original asks for and resolves "
    "idioms by meaning (e.g. 'jaguar print hoodie' = a hoodie with a leopard-spot pattern, not a jaguar; "
    "'riding a suzuki' = riding a Suzuki motorcycle; 'wet clay of Capitol Hill' = the U.S. Capitol building sculpted "
    "in wet clay), and adds the art direction a strong illustrator or photographer would choose: a specific camera "
    "angle and lens (low angle, high angle, close three-quarter, wide-angle foreshortening, telephoto), a dynamic pose "
    "or action, bold framing or cropping, foreground/midground/background layering with overlap, and lighting that "
    "groups light and shadow. Keep each rewrite under 70 words, in English, with no extra subjects and no text in the "
    "image unless the original asks for it. Write the result to {outp} as a JSON list of objects "
    '{{"index","prompt","rewritten"}} in the same order, covering every input. Do not generate images.'
)


def main():
    rows = [json.loads(l) for l in open(sys.argv[1])]
    out = Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    par = int(sys.argv[3]) if len(sys.argv) > 3 else 16
    size = -(-len(rows) // par)
    chunks = [rows[i : i + size] for i in range(0, len(rows), size)]

    def one(k):
        inp, outp = out / f"chunk_{k}.json", out / f"rewritten_{k}.json"
        if outp.exists():
            return k, "exists"
        json.dump(
            [{"index": k * size + j, "prompt": r["prompt"]} for j, r in enumerate(chunks[k])],
            open(inp, "w"),
            indent=1,
        )
        home = Path(f"/mnt/nvme/codex_homes/w{k}")
        env = {**os.environ, "CODEX_HOME": str(home)}
        cmd = [
            "codex",
            "exec",
            "-m",
            "gpt-5.6-luna",
            "-c",
            'model_reasoning_effort="medium"',
            "--skip-git-repo-check",
            "--dangerously-bypass-approvals-and-sandbox",
            "-C",
            str(out),
            "-o",
            str(out / f"last_{k}.txt"),
            "-",
        ]
        p = subprocess.run(
            cmd,
            input=TASK.format(inp=inp, outp=outp),
            text=True,
            capture_output=True,
            timeout=3600,
            env=env,
        )
        return k, "ok" if outp.exists() else f"missing rc={p.returncode} {p.stderr[-200:]}"

    with cf.ThreadPoolExecutor(par) as ex:
        for k, st in ex.map(one, range(len(chunks))):
            print(k, st, flush=True)
    merged = {}
    for k in range(len(chunks)):
        f = out / f"rewritten_{k}.json"
        if f.exists():
            for o in json.load(open(f)):
                if o.get("rewritten"):
                    merged[o["prompt"]] = o["rewritten"]
    json.dump(merged, open(out / "rewritten_all.json", "w"), indent=1, ensure_ascii=False)
    print("merged", len(merged), "of", len(rows))


main()
