# Anima Color-and-Light Prompts

This is the single reviewed color-and-light prompt source. It consolidates the
former canary and formal datasets without duplicating training rows or mixing
development examples into formal evaluation.

- `train_prompts.jsonl`: 256 prompts, 16 per fine-grained bucket and 64 per
  core axis. It includes the 64 reviewed v1 training prompts plus 192 new
  prompts.
- `eval_prompts.jsonl`: 96 new held-out prompts, six per bucket and 24 per
  core axis. None appeared in v1 or in the training set.
- `development_prompts.jsonl`: the 32 former canary evaluation prompts, two per
  bucket. They were already used during development and must not be reported
  as additional untouched held-out examples.
- `dataset_spec.json`: the single source of truth for axes, buckets, counts,
  update width, and the machine-checked language contract.

The 256 training rows and their order are byte-identical to the former v2
training manifest; its 64 inherited v1 prompts appear only once. The formal
96-row evaluation manifest is also byte-identical. All 384 distinct prompts
from both datasets are retained across three disjoint splits. The stricter
formal language contract applies to train/eval; the development split retains
its original reviewed wording. Use only `+dataset=anima_color_light_ddrl`.

The four core axes are `palette_intent`, `value_exposure`,
`lighting_consistency`, and `material_response`. Each has four observable
sub-buckets. The committed training order contains 64 four-prompt update
blocks. Every block contains one prompt from every core axis; every four-block
superblock covers all four sub-buckets within every axis. The orthogonal bucket
schedule prevents any pair of sub-buckets from always appearing together.

Every row is an original English anime scene written for this repository. The
prompts do not copy benchmark rows, use artist names, or add quality incantations
such as `masterpiece` and `8k`. Muted and vivid palettes, high-key and low-key
exposure, soft and hard illumination, matte and glossy materials, daylight and
artificial sources, and character-led and environment-led scenes all appear in
both directions so reward cannot equate one visual treatment with quality.

## Research basis

Public benchmarks were used to define coverage, not as training text:

- [GenColorBench](https://github.com/moatifbutt/gen-color-bench) separates
  named color, object-color association, multi-object composition, and implicit
  color understanding. Its repository does not currently declare a license, so
  no prompt text was copied.
- [T2I-CompBench](https://github.com/Karine-Huang/T2I-CompBench) demonstrates
  the need to test color and texture binding separately from overall alignment.
- [GenEval](https://github.com/djghosh13/geneval) motivates object-level color
  checks rather than one holistic similarity score.
- [Qwen-Image-Bench](https://github.com/QwenLM/Qwen-Image-Bench) independently
  identifies color harmony, lighting and atmosphere, material properties, and
  material texture as separate evaluation facets.
- [Bernini's prompt taxonomy](https://github.com/bytedance/Bernini/blob/main/bernini/prompt_enhancer.py)
  explicitly describes light source, direction, temperature, intensity, and
  resulting shadows. Only those dimensions were borrowed, not its defaults.
- [DOCCI](https://google.github.io/docci/) informed the use of complete,
  observable natural-language descriptions instead of comma-separated tags.

GenEval, T2I-CompBench, Qwen-Image-Bench, and PartiPrompts remain external
diagnostics; putting their rows into training would contaminate later benchmark
results. The frozen `datasets/anima/quality_v1/eval_prompts.jsonl` also remains
a separate general-quality regression set.

## Build the DDRL data assets

The committed files contain prompts only. DDRL's clean-data regularizer needs a
unique base-generated target and latent for every training row. Generate those
assets from the untouched pinned Anima base before launching training:

```bash
python -m vrl.scripts.families.cosmos.anima.generate \
  --manifest datasets/anima/color_light/train_prompts.jsonl \
  --output-dir data/external/anima/color_light \
  --steps 40 --guidance-scale 4.5 --samples-per-prompt 1 \
  --negative-prompt ""

python -m vrl.scripts.denoise.encode_targets \
  --experiment anima_preview3/online_grpo \
  --out data/external/anima/color_light/sft_latents_bf16.pt \
  --storage-dtype bf16 \
  --preview-out outputs/anima_color_light_anchor_roundtrip.png \
  +reward=codex_image_qa_anime_color_light +reward=codex_image_qa_luna \
  +dataset=anima_color_light_ddrl \
  actor.optim.lr=2e-5 trainer.output_dir=outputs/anima_color_light_encoding \
  sampling.num_steps=40 model.use_lora=false
```

Review the 256 anchors before training. The generated assets remain outside Git
under `data/external/`. One single-GPU pass is 64 optimizer updates and produces
2,048 scored rollouts (`64 updates * 4 prompts * 8 rollouts`). Formal held-out
evaluation must pass `--prompts-per-bucket-style 6`; the default value of two is
kept only for compatibility with the earlier canary protocol.

The merge changes prompt/config assets only; it does not generate images or
encode latents. The old `data/external/anima/color_light_v1` has only 64 targets
and remains historical evidence. Do not copy its manifest/shard into the new
256-target location or silently train on that subset. Review the complete new
anchors and encode their matching latent shard before DDRL training. Plain RL
can instead use the canonical prompt manifest directly with SFT disabled.

Original files are recoverable from
`/home/mingfeiguo/Desktop/vrl-color-light-merge-zMa45M/before.tar.gz`. Historical
run records keep their original versioned paths and evaluation identities.
