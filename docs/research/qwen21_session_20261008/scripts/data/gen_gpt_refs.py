"""Generate GPT reference images for a prompt manifest through the Codex CLI (image tool), N in parallel.

usage: gen_gpt_refs.py <prompts.jsonl> <out_dir> [limit] [parallel]
Writes <out_dir>/<i:04d>.png (i = line index) and <out_dir>/log.jsonl; skips existing files.
"""

import concurrent.futures as cf
import json
import subprocess
import sys
from pathlib import Path

prompts, out = sys.argv[1], Path(sys.argv[2])
limit = int(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[3] != "-" else None
par = int(sys.argv[4]) if len(sys.argv) > 4 else 6
only = (
    set(json.load(open(sys.argv[5]))) if len(sys.argv) > 5 else None
)  # explicit index list (regeneration)
out.mkdir(parents=True, exist_ok=True)
rows = [json.loads(l) for l in open(prompts)]
if limit:
    rows = rows[:limit]
todo = [(i, r) for i, r in enumerate(rows) if only is None or i in only]
work = out / "_work"
work.mkdir(exist_ok=True)
(out / "_bad").mkdir(exist_ok=True)


def one(i, row):
    target = out / f"{i:04d}.png"
    if target.exists() and only is None:
        return {"i": i, "status": "exists"}
    if target.exists():
        target.rename(out / "_bad" / f"{i:04d}.png")
    wd = work / f"{i:04d}"
    wd.mkdir(exist_ok=True)
    prompt = row["prompt"].replace('"', "'")
    q = (
        f'Generate one square 1024x1024 image for exactly this text-to-image prompt, following it literally and in the medium it implies: "{prompt}". '
        f"Save it as {wd / 'image.png'} (PNG). Do not add text or watermarks. Reply with the absolute path of the saved file only."
    )
    cmd = [
        "codex",
        "exec",
        "-m",
        "gpt-5.6-luna",
        "-c",
        'model_reasoning_effort="low"',
        "--skip-git-repo-check",
        "--dangerously-bypass-approvals-and-sandbox",
        "-C",
        str(wd),
        "-o",
        str(wd / "last.txt"),
        "-",
    ]
    try:
        p = subprocess.run(cmd, input=q, text=True, capture_output=True, timeout=900)
        local = wd / "image.png"
        if local.exists() and local.stat().st_size > 10000:
            local.replace(target)
        ok = target.exists() and target.stat().st_size > 10000
        return {
            "i": i,
            "status": "ok" if ok else "missing",
            "rc": p.returncode,
            "msg": (wd / "last.txt").read_text()[:200]
            if (wd / "last.txt").exists()
            else p.stderr[-200:],
        }
    except Exception as e:
        return {"i": i, "status": "error", "msg": str(e)[:200]}


with open(out / "log.jsonl", "a") as f, cf.ThreadPoolExecutor(par) as ex:
    for r in ex.map(lambda t: one(*t), todo):
        f.write(json.dumps(r) + "\n")
        f.flush()
        print(r["i"], r["status"], r.get("msg", "")[:80], flush=True)
