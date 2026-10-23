# Qwen Image 2.1

The `qwen_image_21` family supports text generation, ordered reference-image
editing, and RGB/RGBA output through the existing denoise executor and replay
interfaces. It is separate from the older `qwen_image` family.

Use `model/qwen_image_21/base`. The frozen dependency lock includes the upstream
Qwen 2.1 implementation; sync with the cosmos extra before running it. The base
preset keeps the frozen text encoder on CPU for a 32 GB GPU. Sampling uses
`reference_resolution` (default 1024) and `output_mode` (`rgb` or `rgba`).

A prompt manifest supplies an ordered `reference_images` list; a row may spell
a single image as `reference_image`, which the loader maps to a one-element list. Paths resolve against the existing
artifact data root. Reference latents are fixed conditioning: replay stores them,
but only generated target latents enter the SDE transition likelihood.

PNG output preserves alpha. Ordinary RGB reward models view RGBA composited over
white; transparent layers need their own alpha-aware checks. A successful RGBA
output alone does not establish correct layer decomposition.

## Integration check

The native editing path was verified against the upstream pipeline (shared first
latent, RGB/RGBA output, replay log-probability parity, one LoRA backward step)
with a one-off probe, `vrl.scripts.generation.qwen_image_21_edit_probe`; the
results are in `docs/reference/qwen_image_21_editing.md` and
`archive/main-20261016:docs/reports/visual_rl_engine_20260922/PROGRESS.md`. The probe was removed
on 2026-10-16; recover it from commit `47f276be7` to rerun the comparison.

## Bundled experiments

- `experiment/qwen_image_21/online_grpo_ocr`: text-to-image GRPO on the OCR
  reward (`reward/ocr`, `dataset/ocr`) with the wide LoRA on one 32 GB card.
- `experiment/qwen_image_21/online_grpo_smoke_single_gpu`: the smallest OCR
  end-to-end probe (256px, `model/qwen_image_21/base`).
- `experiment/qwen_image_21/online_grpo_local_edit`: reference-conditioned
  editing, described below.

## Reference-conditioned online training

`experiment/qwen_image_21/online_grpo_local_edit` composes
`model/qwen_image_21/lora_wide`, `reward/editreward_http`, and
`dataset/local_edit_plain_remove_add`. The dataset is the unhinted removal and
addition rows of `manifests/local_edit/` (written with the full manifests by
`python -m vrl.scripts.data.local_edit`), with a matching held-out subset; source
images live under the artifact data root and must be provisioned separately.
The reward is the EditReward judge alone, served by an operator-run service;
the experiment points `reward.inference.editreward.endpoint` at port 18316
instead of the preset's 8315, matching the service fields below.

The EditReward adapter loads the released checkpoint through its upstream
implementation in an isolated Transformers 4.57 environment. The generator
continues to use this repository's frozen Transformers 5.17 environment.
Start the standard reward service in the isolated environment, with this checkout
on PYTHONPATH, and configure these service fields:

```yaml
host: 127.0.0.1
port: 18316
model_name: editreward-qwen25-7b
model_version: 51b92ab5246295637c4ab3bd71e54a26f0a5189d
max_request_bytes: 67108864
artifact_roots: [/absolute/path/to/outputs]
worker_config:
  model_factory: vrl.rewards.models.editreward:EditRewardModel
  device: cuda:0
  data_root: /absolute/path/to/source-images
  upstream_root: /absolute/path/to/EditReward
  base_model: Qwen/Qwen2.5-VL-7B-Instruct
  base_revision: cc594898137f460bfe9f0759e9844b3ce807cfb5
  reward_model_name: TIGER-Lab/EditReward-Qwen2.5-VL-7B
  revision: 51b92ab5246295637c4ab3bd71e54a26f0a5189d
```

The adapter only reads cached model snapshots. The qualified upstream checkout
was `77a93aaa461fe9187e0ff841b59ecc0d0620bb7f`; preserve its dependencies separately.
Its initial base-model warning about uninitialized reward heads precedes a strict
load of the released reward checkpoint. A successful strict load is required.

```bash
HF_HUB_OFFLINE=1 .venv/bin/python -m vrl.scripts.train \
  --config experiment/qwen_image_21/online_grpo_local_edit \
  model.path=/absolute/path/to/cached/Qwen-Image-2.1-snapshot \
  data.artifact_data_root=/absolute/path/to/source-images
```

Trainer, rollout worker and external reward explicitly park between phases on a
shared GPU. No Ray GPU bundle is reserved for the operator-owned service. A service
must advertise and fulfill the existing parking protocol; this is not permission
to overlap an unverified external GPU workload.

The adapter exports `editreward` and `editreward_log_sigma`. The latter is the
checkpoint's raw log-scale output, not a calibrated correctness probability.
Reference images are resized to candidate geometry and transparency is composited
on white. It supports one reference per judgment and rejects ambiguous/multiple
references. These preprocessing choices must be frozen for comparisons.
