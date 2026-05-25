# Anime Safety Prompt Manifests

These manifests are prompt-only datasets for online safety tuning and baseline
evaluation of anime image models.

This dataset lives under `datasets/danbooru/safety` for anime dataset
organization. The primary build path derives prompts from Danbooru metadata
rows with `rating:questionable` and `rating:explicit`, while filtering obvious
underage-risk tags. Each generated row records sample-level fields:
`rating`, `nsfw_tags`, `source_tags`, `source_post_ids`, and `source_score`.

The prompt text may contain explicit Danbooru tags. The builder is intentionally
task-neutral: it does not write training targets, model policy labels, or
dataset-wide constants into each row or report.

Files:

- `train.jsonl`: prompt mix for RL training.
- `eval_baseline.jsonl`: fixed prompt set for baseline and post-training
  NSFW-safety checks.

Rows use the native `PromptExample` JSONL format. The `metadata.category` field
is used only for reporting.

Rebuild the Danbooru-derived manifests with:

```bash
python -m vrl.scripts.data.populate anime-safety-prompts
```
