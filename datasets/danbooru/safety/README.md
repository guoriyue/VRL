# Anime Safety Prompt Manifests

These manifests are prompt-only datasets for online safety tuning and baseline
evaluation of anime image models.

This dataset lives under `datasets/danbooru/safety` for anime dataset
organization. The primary build path derives prompts from Danbooru metadata
rows with `rating:questionable` and `rating:explicit`, while filtering obvious
underage-risk tags. Each generated row records `rating`, `nsfw_tags`,
`source_tags`, `source_post_ids`, and `safety_target`.

The prompt text may contain explicit Danbooru tags. The training objective for
these rows is still safety: `metadata.safety_target` is `avoid_nsfw`.

Files:

- `train.jsonl`: prompt mix for RL training.
- `eval_baseline.jsonl`: fixed prompt set for baseline and post-training
  NSFW-safety checks.

Rows use the native `PromptExample` JSONL format. The `metadata.category` field
is used only for reporting.

Rebuild the Danbooru-derived manifests with:

```bash
python -m vrl.scripts.data.anime_anatomy build-safety-prompts \
  --download-danbooru-metadata \
  --train-output datasets/danbooru/safety/train.jsonl \
  --eval-output datasets/danbooru/safety/eval_baseline.jsonl \
  --report-output datasets/danbooru/safety/report.json \
  --train-limit 2000 \
  --eval-limit 200
```
