# Few-Shot Rollout Preview

This directory owns a small, opt-in image preview for RL experiment configs. It
does not score images, compare against a native pipeline, or make a training
decision. Nothing under `vrl/` imports this package, and `vrl-train` does not
pause or resume based on its output.

Run it with a real bundled experiment name or an absolute YAML path:

```bash
uv run --no-sync pytest tests/quality/test_rollout_preview.py -q \
  --rollout-preview-config experiment/sana/online_grpo_aesthetic \
  --rollout-preview-dir /tmp/sana-rollout-preview
```

The output directory must not already exist. The preview takes up to the first
four prompts from the experiment's real `data.manifest`, generates one image
per prompt with deterministic seeds, and writes one to four numbered PNGs plus
`preview.json`:

```text
000.png
...
preview.json
```

Open the PNG files individually. `preview.json` records each prompt, seed, exact
request sampling values, model identity, and resolved precision so the visible
result can be traced back to the YAML settings.

The execution path is the registered production rollout path:

- family, task, model builder, and executor come from `FAMILY_REGISTRY`;
- sampling values come from the composed `rollout` and `sampling` YAML blocks;
- prompts and per-row overrides come from the configured training manifest;
- generation uses the registered executor's `plan()` and `forward_plan()` path.

There is no SANA adapter or second family support table. Any registered `t2i`
family with a usable experiment YAML uses the same preview. Video and
reference-conditioned tasks are intentionally outside this first image-only
slice.

`trainer.resume_from` is rejected because the direct rollout builder cannot
restore trainer state. To preview trained inference weights, point the model
config at the exact checkpoint or LoRA adapter through its supported model
fields; silently showing fresh base weights would be misleading.

Run the CPU tests with:

```bash
CUDA_VISIBLE_DEVICES="" uv run --no-sync pytest tests/quality -q
```

The old evidence schema, hashes, scores, corruption controls, native/BF16
comparisons, replay bundle, status machine, producer coverage table, and contact
sheet were removed. Objective model-math parity remains in model-owned tests;
visual quality remains a human judgment of the generated files.
