"""Freeze unseen test prompts before continuation results, without generating images."""

import hashlib
import json
import unicodedata
from pathlib import Path

REPO = Path("/home/ubuntu/VRL")
ROOT = Path("/mnt/nvme/outputs/qwen21_aesthetic_lora")
OUT = ROOT / "fresh_test256_window15"


def norm(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def strings(obj):
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for value in obj.values():
            yield from strings(value)
    elif isinstance(obj, list):
        for value in obj:
            yield from strings(value)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    assert not OUT.exists(), "Frozen output already exists; do not resample."
    train = REPO / "manifests/pickscore_sfw/train.txt"
    test = REPO / "manifests/pickscore_sfw/test.txt"
    prompts = {}
    for p in test.read_text().splitlines():
        if p.strip():
            prompts.setdefault(norm(p), p.strip())
    excluded = {norm(p) for p in train.read_text().splitlines()}
    audit = []
    roots = [
        ROOT,
        Path("/mnt/nvme/outputs/qwen21_aesthetic_fullparam"),
        REPO / "outputs/qwen21_aesthetic_qualification",
        REPO / "docs/research/qwen21_aesthetic",
    ]
    # Scan all string leaves, not just one prompt field: batch records may nest prompts.
    for root in roots:
        for path in sorted(root.rglob("*.jsonl")):
            matched = set()
            lines = path.read_text().splitlines()
            for index, line in enumerate(lines):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError:
                    # An active writer can leave its final line incomplete. Do not
                    # ignore incomplete evidence when selecting a fresh holdout.
                    raise RuntimeError(f"Incomplete JSONL; retry audit later: {path}:{index + 1}")
                matched.update(norm(s) for s in strings(value) if norm(s) in prompts)
            excluded.update(matched)
            if matched:
                audit.append(
                    {
                        "path": str(path),
                        "sha256": digest(path),
                        "matched_test_prompts": len(matched),
                    }
                )
    available = [k for k in prompts if k not in excluded]
    # Hash ordering is fixed before scores; no outcome-dependent selection.
    available.sort(
        key=lambda k: hashlib.sha256(("qwen21-fresh-20260926-v1\0" + k).encode()).digest()
    )
    assert len(available) >= 256, len(available)
    selected = available[:256]
    OUT.mkdir()
    rows = [{"id": f"fresh-{i:04d}", "prompt": prompts[k]} for i, k in enumerate(selected)]
    manifest = OUT / "prompts.jsonl"
    manifest.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    for gpu in range(4):
        (OUT / f"test_gpu{gpu}.jsonl").write_text(
            "".join(
                json.dumps(r, ensure_ascii=False) + "\n" for r in rows[gpu * 64 : (gpu + 1) * 64]
            )
        )
    protocol = {
        "candidate": str(ROOT / "window2_trial/train_to_15/checkpoint-final"),
        "candidate_fixed_before_result": True,
        "control": "base",
        "prompt_count": 256,
        "samples_per_prompt": 1,
        "resolution": 1024,
        "steps": 40,
        "denoise_mode": "native",
        "seed": 20261001,
        "seed_rule": "seed + global prompt index",
        "train_manifest_sha256": digest(train),
        "source_test_manifest_sha256": digest(test),
        "manifest_sha256": digest(manifest),
        "available_after_exclusions": len(available),
        "excluded_matches": audit,
        "normalization": "NFKC, casefold, collapse whitespace; not semantic deduplication",
        "selection": "sha256 ordering with fixed salt qwen21-fresh-20260926-v1",
        "evaluation": "Paired prompt-level aesthetic and independent pinned PickScore, bootstrap 10000; report all prompts. No candidate or prompt reselection using these outcomes. Visible improvement requires separate blinded review; score changes alone are insufficient.",
        "status": "Frozen only; no image generation launched. Recheck non-overlap immediately before evaluation; reject contamination instead of silently replacing prompts.",
    }
    (OUT / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: protocol[k]
                for k in [
                    "prompt_count",
                    "available_after_exclusions",
                    "manifest_sha256",
                    "candidate",
                ]
            },
            indent=2,
        )
    )
