"""Build the Wan 2.2 T2V GRPO prompt splits from the VideoPhy manifests.

Three disjoint sets, all plain one-prompt-per-line ``.txt`` manifests:

* ``train_small.txt``: a small learning-verification training set drawn from
  the official VideoPhy train split.
* ``val.txt``: prompts from the same train pool that are never trained on;
  used for checkpoint selection / generalization reads during the run.
* ``test.txt``: the official VideoPhy eval split, held out entirely; scored
  once at the end and never used to tune anything.

Dedup is exact after normalization plus a token-Jaccard near-duplicate screen
against the training prompts so a val/test prompt cannot be a paraphrase of a
trained one. The report records every decision so the split is reproducible.
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "videophy"
SEED = 20260914
TRAIN_SMALL = 16
VAL = 24
NEAR_DUP_JACCARD = 0.6


def normalize(prompt: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", " ", prompt.lower()).strip()


def tokens(prompt: str) -> set[str]:
    stop = {"a", "an", "the", "of", "in", "on", "into", "onto", "to", "and", "is", "with", "by"}
    return {t for t in normalize(prompt).split() if t not in stop}


def jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if a | b else 0.0


def read(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    train_pool = read(SOURCE / "train.txt")
    test = read(SOURCE / "eval.txt")
    assert len(train_pool) == 309 and len(test) == 35, (len(train_pool), len(test))

    seen: set[str] = set()
    pool: list[str] = []
    for prompt in train_pool:
        key = normalize(prompt)
        if key in seen:
            continue
        seen.add(key)
        pool.append(prompt)
    assert not (seen & {normalize(p) for p in test}), "official train/eval overlap"

    rng = random.Random(SEED)
    order = list(range(len(pool)))
    rng.shuffle(order)
    train_small = [pool[i] for i in order[:TRAIN_SMALL]]
    train_tokens = [tokens(p) for p in train_small]

    def near(prompt: str) -> tuple[float, str]:
        t = tokens(prompt)
        best = max(((jaccard(t, tt), train_small[k]) for k, tt in enumerate(train_tokens)))
        return best

    val: list[str] = []
    rejected_val: list[dict[str, object]] = []
    for i in order[TRAIN_SMALL:]:
        if len(val) == VAL:
            break
        score, nearest = near(pool[i])
        if score >= NEAR_DUP_JACCARD:
            rejected_val.append({"prompt": pool[i], "jaccard": score, "nearest_train": nearest})
            continue
        val.append(pool[i])
    test_kept: list[str] = []
    rejected_test: list[dict[str, object]] = []
    for prompt in test:
        score, nearest = near(prompt)
        if score >= NEAR_DUP_JACCARD:
            rejected_test.append({"prompt": prompt, "jaccard": score, "nearest_train": nearest})
            continue
        test_kept.append(prompt)

    sets = {"train_small": train_small, "val": val, "test": test_kept}
    for a in sets:
        for b in sets:
            if a < b:
                assert not ({normalize(p) for p in sets[a]} & {normalize(p) for p in sets[b]}), (
                    a,
                    b,
                )
    for name, prompts in sets.items():
        (HERE / f"{name}.txt").write_text("\n".join(prompts) + "\n", encoding="utf-8")
    report = {
        "source": {"train": str(SOURCE / "train.txt"), "eval": str(SOURCE / "eval.txt")},
        "seed": SEED,
        "near_duplicate_jaccard_threshold": NEAR_DUP_JACCARD,
        "counts": {k: len(v) for k, v in sets.items()},
        "rejected_val_near_duplicates": rejected_val,
        "rejected_test_near_duplicates": rejected_test,
        "max_cross_jaccard": {
            f"{a}->{b}": max(jaccard(tokens(p), tokens(q)) for p in sets[a] for q in sets[b])
            for a in sets
            for b in sets
            if a != b
        },
    }
    (HERE / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "source"}, indent=1))


if __name__ == "__main__":
    main()
