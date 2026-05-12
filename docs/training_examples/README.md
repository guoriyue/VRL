# Training Examples

This directory is for specific training runs that are useful as examples.

Use one subdirectory per experiment config:

```text
docs/training_examples/<experiment_config_name>/
  README.md
  qualitative_ocr_comparison.jpg
```

Rules:

- Keep raw run artifacts in `outputs/<run_name>/`.
- Keep only curated, small images in the matching example directory.
- Name image files by what they show, e.g. `qualitative_ocr_comparison.jpg`.
- Label images in docs as `[Image #N]` so the prose can refer to them consistently.
- State whether the result is qualitative, metric-backed, or both.
- Do not describe an example as a benchmark or paper reproduction unless the exact evaluation protocol is documented.
