# Anime Safety Prompt Manifests

These manifests are prompt-only datasets for online safety tuning and baseline
evaluation of anime image models.

The prompts intentionally avoid minors, explicit sex acts, and illegal content.
Every stress prompt uses adult subjects and targets common anime-generation
failure modes where models may drift into unsafe exposure: swimsuit styling,
stage costumes, bedroom lighting, close camera framing, and fantasy armor.

Files:

- `train.jsonl`: prompt mix for RL training.
- `eval_baseline.jsonl`: fixed prompt set for baseline and post-training
  NSFW-safety checks.

Rows use the native `PromptExample` JSONL format. The `metadata.category` field
is used only for reporting.
