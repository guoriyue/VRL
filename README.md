# visual-rl

RL-style post-training infrastructure for visual generative models.

This matrix tracks repository integration status, not model quality or benchmark
claims. It lists canonical training recipes only; reward swaps, model-size
swaps, and short-lived ablations should be passed as config overrides instead
of becoming new `configs/experiment/*.yaml` files.

## Model / Algorithm Matrix

Legend:

- `[x]` active: experiment YAML, training entrypoint, rollout adapter, and
  structural tests exist.
- `[~]` wired: code path exists, but the model binding still needs real
  checkpoint smoke validation.
- `[ ]` not wired.
- `-` not a target pairing for the current model family.

| Model | Modality | GRPO | TokenGRPO | Diffusion DPO | Current progress |
| --- | --- | --- | --- | --- | --- |
| SD3.5 | text-to-image diffusion | `[x]` `sd3_5_ocr_grpo` | - | `[ ]` | Canonical active recipe: OCR GRPO. |
| Wan 2.1 1.3B | text-to-video diffusion | `[x]` `wan_2_1_1_3b_ocr_grpo` | - | `[x]` `wan_2_1_1_3b_dpo` | Canonical active recipes: OCR GRPO and offline DPO. |
| Cosmos Predict2 2B | video-to-world diffusion | `[x]` `cosmos_predict2_2b_grpo` | - | `[ ]` | Canonical Predict2 GRPO wiring. |
| Janus-Pro 1B | autoregressive image | - | `[x]` `janus_pro_1b_ocr_grpo` | - | Canonical TokenGRPO recipe: OCR. |
| NextStep-1 1.1 | continuous-token autoregressive image | - | `[~]` `nextstep_1_ocr_grpo` | - | Wired, but still marked pre-smoke for real checkpoint binding. |

## Algorithm Kinds

| Algorithm kind | Used by | Config base |
| --- | --- | --- |
| `grpo` | SD3.5, Wan 2.1, Cosmos Predict2 | `configs/base/algorithm/grpo.yaml` |
| `token_grpo` | Janus-Pro, NextStep-1 | `configs/base/algorithm/token_grpo.yaml` |
| `diffusion_dpo` | Wan 2.1 offline DPO | `configs/base/algorithm/dpo.yaml` |

Run any active experiment with:

```bash
python -m vrl.scripts.train --config experiment/<config_name>
```

Example:

```bash
python -m vrl.scripts.train --config experiment/sd3_5_ocr_grpo
```

Use overrides for one-off reward/model/data changes:

```bash
python -m vrl.scripts.train \
  --config experiment/sd3_5_ocr_grpo \
  reward.components.ocr=0.0 \
  reward.components.aesthetic=1.0 \
  reward.kwargs.aesthetic.model_name=openai/clip-vit-large-patch14 \
  data.manifest=datasets/drawbench/train_192.txt \
  trainer.output_dir=outputs/sd3_5_aesthetic_ablation
```
