# Anima General-Quality Canary Prompts

This dataset is a small, reviewed canary for reward-model training. It is
not a replacement for a production-scale caption corpus.

The prompts are complete natural-language scene descriptions. They avoid the
previous Danbooru renderer's repeated tag clauses such as `1girl, solo, full
body`, contradictory attributes, and blanket requirements that every hand and
foot remain visible. The sixteen buckets cover portraits, body mechanics,
object contact, multi-character interaction, difficult viewpoints,
environments, materials, and lighting. Back views and naturally occluded parts
are intentionally valid cases.

- `train_prompts.jsonl`: 64 prompts, four per bucket.
- `eval_prompts.jsonl`: 32 disjoint prompts, two per bucket.

The training rows are round-robin ordered across buckets. With the recipe's
four prompts per update and `sequential_window` sampler, every update covers
four distinct buckets and sixteen updates make one complete 64-prompt pass.
The one-update canary intentionally consumes only the first four rows.

Build deterministic synthetic clean anchors from the untouched Anima base:

```bash
python -m vrl.scripts.families.cosmos.anima.generate \
  --config experiment/anima_preview3/online_grpo_codex_quality_ddrl_canary \
  --manifest datasets/anima/quality_v1/train_prompts.jsonl \
  --output-dir data/external/anima/quality_v1 \
  --steps 40 --guidance-scale 4.5 --samples-per-prompt 1

python -m vrl.scripts.denoise.encode_targets \
  --experiment anima_preview3/online_grpo_codex_quality_ddrl_canary \
  --out data/external/anima/quality_v1/sft_latents_bf16.pt \
  --storage-dtype bf16
```

The generation command writes `anchor_manifest.jsonl` next to the images. The
DDRL recipe consumes that generated manifest and latent shard. Review the base
anchors before a long run; this canary deliberately keeps generated assets out
of Git under `data/external/`.
