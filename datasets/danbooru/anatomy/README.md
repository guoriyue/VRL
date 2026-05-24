# Anime Anatomy Dataset

This directory stores the prompt manifests for the Anima anatomy RL experiment.

Current files:

- `train_prompts.jsonl`: 20,000 quota-balanced training prompts.
- `eval_prompts.jsonl`: 1,000 stratified evaluation prompts.
- `prompt_report.json`: machine-readable bucket/style count report for the
  generated prompt manifests.

Build the real prompt manifests from Danbooru metadata only. Prefer the
dataset population wrapper:

```bash
python -m vrl.scripts.data.populate anime-prompts
```

The underlying command is:

```bash
python -m vrl.scripts.data.danbooru build-prompts \
  --download-danbooru-metadata \
  --train-output datasets/danbooru/anatomy/train_prompts.jsonl \
  --eval-output datasets/danbooru/anatomy/eval_prompts.jsonl \
  --report-output datasets/danbooru/anatomy/prompt_report.json \
  --train-limit 20000 \
  --eval-limit 1000 \
  --min-score 5 \
  --preferred-min-score 20 \
  --bucket-balance quota \
  --prompt-style mixed
```

This downloads `metadata/posts.tar.gz` from `nyanko7/danbooru2023`, not the
image tarballs. Images are only needed later for positive image manifests, hand
crops, and hard-negative reward calibration.

`--bucket-balance quota` caps naturally overrepresented buckets such as
`feet_visible` and `sitting_full_body`, while rare action or hand buckets use
`--preferred-min-score` first and fall back to `--min-score` when needed.

See `../dataset_analysis_report.md` for the shared Danbooru metadata funnel and
the final train/eval distribution. The key conclusion is that Danbooru2023 has
enough anatomy prompt candidates, but natural sampling is badly skewed; the
prompt dataset must use bucket quotas to keep action, hand, and motion prompts
visible during RL.
