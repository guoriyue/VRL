import hashlib
import json
from pathlib import Path

root = Path("outputs/qwen21_aesthetic_qualification")
rows = []
for p in sorted((root / "generation").glob("q*/s*.json")):
    r = json.loads(p.read_text())
    assert hashlib.sha256(Path(r["path"]).read_bytes()).hexdigest() == r["sha256"]
    rows.append(
        {k: r[k] for k in ["sample_id", "prompt_id", "prompt", "path", "sha256"]}
        | {"metadata": {"category": r["category"], "seed": r["seed"]}}
    )
assert len(rows) == 512, len(rows)
assert len({r["sample_id"] for r in rows}) == 512
(root / "media.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
(root / "stress_source.jsonl").write_text(
    "".join(json.dumps(r) + "\n" for r in rows if r["sample_id"].endswith("-s0"))
)
print("Verified 512 image hashes across 64 prompts.")
