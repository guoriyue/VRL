"""Codex pass over Danbooru safety prompts: clothed safe_prompt + minor/real-person flag.

usage: codex_safe_rewrite.py in.jsonl out_dir [parallel]
Writes out_dir/chunk_<k>.json, out_dir/out_<k>.json, merged out_dir/merged.json {prompt: {...}}.
"""

import concurrent.futures as cf
import json
import os
import subprocess
import sys
from pathlib import Path

TASK = (
    'Read {inp}: a JSON list of {{"index","prompt"}} where each prompt is a comma-separated list of Danbooru '
    "tags for an anime image generator. These prompts are used to train a safety filter that REMOVES nudity "
    'and sexual content from the model. For every item output {{"index", "flag", "reason", "safe_prompt"}}:\n'
    "- flag = true if ANY tag names a real person, OR names a fictional character who is under 18 in canon or "
    "usually drawn as a child or young teen (e.g. most school-age anime characters, Hatsune Miku, Kagamine Rin), "
    "OR uses any youth-coded tag (school uniform, randoseru, petite/flat_chest with childlike framing, child, teen, "
    "loli, shota, aged_down, etc.). When in doubt, flag. reason = short explanation (empty if not flagged).\n"
    "- safe_prompt = the same tag list rewritten so the image is fully clothed and non-sexual: keep the character "
    "count (1girl, 2girls...), hair, eyes, setting, pose (made non-suggestive), framing, art style and colors; "
    "replace every nude, underwear, exposed-body, sexual-act, bodily-fluid or fetish tag with ordinary clothing "
    "tags (e.g. 'dress', 'jacket', 'jeans') or drop it; remove 'rating:explicit' / 'rating:questionable' and "
    "write 'rating:general'. Keep the 'adult' tag. Output tags only, comma-separated.\n"
    "Write the result to {outp} as a JSON list in the same order, covering every input. Do not generate images."
)


def main():
    rows = [json.loads(line) for line in open(sys.argv[1])]
    out = Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    par = int(sys.argv[3]) if len(sys.argv) > 3 else 16
    size = -(-len(rows) // par)
    chunks = [rows[i : i + size] for i in range(0, len(rows), size)]

    def one(k):
        inp, outp = out / f"chunk_{k}.json", out / f"out_{k}.json"
        if outp.exists():
            return k, "exists"
        json.dump(
            [{"index": k * size + j, "prompt": r["prompt"]} for j, r in enumerate(chunks[k])],
            open(inp, "w"),
            indent=1,
        )
        env = {**os.environ, "CODEX_HOME": f"/mnt/nvme/codex_homes/w{k}"}
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
        f = out / f"out_{k}.json"
        if not f.exists():
            continue
        src = json.load(open(out / f"chunk_{k}.json"))
        for o in json.load(open(f)):
            j = o["index"] - k * size
            if 0 <= j < len(src):
                merged[src[j]["prompt"]] = o
    json.dump(merged, open(out / "merged.json", "w"), indent=1, ensure_ascii=False)
    print("merged", len(merged), "of", len(rows))


main()
