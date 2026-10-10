"""Rescore (prompt, image, reference) triples through the live judge; input jsonl rows: prompt, image, ref[, intent]."""

import asyncio
import json
import sys
import uuid

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, "/home/ubuntu/VRL")
from vrl.rewards.inference import RewardInferenceArtifact, RewardInferenceRequest
from vrl.rewards.service.client import HttpRewardScorer


def tensor(p):
    return (
        torch.from_numpy(np.array(Image.open(p).convert("RGB"))).permute(2, 0, 1).float() / 255.0
    )


async def main():
    rows = [json.loads(l) for l in open(sys.argv[1])]
    sc = HttpRewardScorer("http://127.0.0.1:8318", expected_model="vlm-preference-qwen3vl32b")
    await sc.activate()
    try:
        with open(sys.argv[2], "w") as out:
            import itertools
            import os

            by_prompt = os.environ.get("RESCORE_BY_PROMPT") == "1"
            chunks = (
                [list(g) for _, g in itertools.groupby(rows, key=lambda r: r["prompt"])]
                if by_prompt
                else [rows[i : i + 4] for i in range(0, len(rows), 4)]
            )
            for i, chunk in enumerate(chunks):
                arts = tuple(
                    RewardInferenceArtifact(
                        artifact_id=f"rs-{uuid.uuid4().hex[:8]}",
                        sample_id=f"{i}-{j}",
                        path="",
                        prompt=r["prompt"],
                        media=tensor(r["image"]),
                        metadata={
                            "judge_reference": r["ref"],
                            **({"judge_intent": r["intent"]} if r.get("intent") else {}),
                            **({"judge_checklist": r["checklist"]} if r.get("checklist") else {}),
                        },
                    )
                    for j, r in enumerate(chunk)
                )
                res = await sc.score_batch(RewardInferenceRequest(f"rs-{uuid.uuid4().hex}", arts))
                for r, x in zip(chunk, res):
                    out.write(
                        json.dumps({**r, "scores": {k: float(v) for k, v in x.scores.items()}})
                        + "\n"
                    )
                    out.flush()
    finally:
        await sc.shutdown()


asyncio.run(main())
