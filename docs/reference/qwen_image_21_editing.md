# Qwen-Image-2.1 editing and transparent assets

The `qwen_image_21` family supports text-only generation, single-image editing,
ordered multi-reference editing (up to ten images), and RGBA subject extraction.
They use the same VRL batch executor, denoise loop, trajectory collection, and
replay model. The family retains task `t2i` to select image reward transport.

## Inputs and output modes

Use `reference_image` for one source, or `reference_images` for an ordered list.
Do not supply both. Image numbers in the prompt follow list order.
`PromptExample.references` remains reward-only; it is not a generation input.
Relative reference paths resolve against `data.artifact_data_root` / `VRL_DATA_ROOT`.

Example JSONL rows:

```jsonl
{"prompt":"Change the text on the wooden sign to say \"Closed Today\". Keep everything else unchanged.","target_text":"Closed Today","reference_image":"source.png"}
{"prompt":"Place the maple leaf from image 2 onto the wooden sign in image 1, keeping everything else unchanged.","reference_images":["source.png","leaf.png"]}
{"prompt":"This is an RGBA image with transparency. Extract only the wooden bookshop sign from the input image, preserving its lettering and appearance. The image has alpha channel and the background is transparent.","reference_image":"source.png","request_overrides":{"output_mode":"rgba"}}
```

Family-specific sampling options:

```yaml
sampling:
  reference_resolution: 1024
  output_mode: rgb
```

`reference_resolution` controls the reference preprocessing area, matching the
official pipeline's `output_resolution`. Each reference retains its aspect
ratio with dimensions rounded to multiples of 32. Target `height` and `width`
remain independent. Lowering reference resolution also reduces vision tokens
and reference latent storage. The resized area must stay within the checkpoint
vision processor's pixel limits (65,536 to 16,777,216 for the pinned checkpoint).
A 128px reference is too small: the vision encoder would enlarge it while the
VAE kept it small. VRL rejects that mismatch before encoding. Use 512 or 1024
for references; target resolution can still be smaller. Extremely narrow images
that round to zero size must be resized/padded before use.

`rgb` retains the existing white-background output. `rgba` preserves all four
channels through generation, PNG saving, and in-memory/tensor reward transport.
It does not force the model to generate transparency: use the explicit RGBA
prompt shown above. RGB image reward adapters composite over white. An alpha-aware
reward can read channel 4 directly; this integration does not add a new reward
objective or claim that binary alpha is always desirable.

The collector accepts these sampling fields in YAML or per-row
`request_overrides`. Training manifests use the existing `prompt_manifest`
loader, with a `.jsonl` manifest. For text editing, the existing OCR reward can
score `target_text`; preservation and extraction quality need their own rewards.

## Reproducible probe

The lock pins a diffusers commit containing Qwen-Image-2.1 and its compatible
Transformers dependency. Synchronize an environment from the lock:

```bash
uv sync --frozen --group test --group lint --extra cosmos
python -m vrl.scripts.generation.qwen_image_21_edit_probe \
  --source /path/to/bookshop.png --reference /path/to/leaf.png \
  --out outputs/qwen21_edits --resolution 512 --reference-resolution 512 \
  --steps 20 --compare-reference --check-lora-backward
```

Set `HF_HUB_OFFLINE=1` to require locally cached weights. `--block-offload` reduces transformer
residency on CUDA; `--device cpu` runs without GPU memory. It saves three VRL
outputs, optional official outputs, and `report.json` with transparency and
replay checks. Official comparison uses identical initial latent values and
disables KV caching on both paths. Small precision differences may remain.
The optional backward check attaches a fresh rank-4 LoRA, generates a two-step
SDE trajectory with two references, checks replay log-probability parity, and
verifies finite gradients and an actual optimizer update. It does not save or
change any existing policy checkpoint.

The image probe uses deterministic native Euler; RL's SDE sampling is a separate
distribution and should not be presented as an equivalent configuration.

References are encoded once per sample batch and expanded across that prompt's
samples. Only target-image latents are updated, stored as actions, and scored
with denoise log-probabilities. Reference latents and geometry are restored for
replay, including the unconditional branch when true CFG is enabled. The
transformer KV cache remains disabled because LoRA updates change prefix keys
and values.

The generic fixed-grid `image_checkpoint_eval` script remains text-only and
rejects reference-conditioned manifests; use the edit probe or generation
requests for editing. Full-scene decomposition into an editable layer stack is
not implemented by these operations.

## Validation recorded on 2026-09-20

The visual probe used the real 7B model, 512px targets/references, 20 native
Euler steps, and identical starting latents. This initial GPU suite used a bf16
VAE on both sides and the VRL executor/collector/replay methods:

| Operation | Observed result | RGBA mean absolute difference vs official (0–255) |
| --- | --- | --- |
| Localized edit | Sign reads `Closed Today` | 0.0635 |
| Two references | Reference leaf appears on the source sign | 0.0641 |
| Subject extraction | Sign on transparent background; 85.86% of alpha values below 128 | 0.0323 |

Every first-step replay action matched exactly. The layer contains small
near-zero alpha values; 1.53% of pixels have alpha in [8,247]. The visual
comparison page is `outputs/qwen_image_21_edit_integration/index.html`.

A separate registry-built CPU test used the production fp32 VAE boundary,
256px inputs, and two-reference SDE sampling with noise level 0.7. A fresh
rank-4 LoRA produced finite nonzero gradients and an actual optimizer update:
replay log-probability max error `1.19e-7`, gradient norm `0.09315`, parameter
max change `4.23e-6`. These are integration checks, not a long RL training run
or a quality benchmark. Existing checkpoints were not modified.

The final precision-aligned registry probe (CPU, fp32 VAE, 128px targets,
256px references, two Euler steps) matched the official loop byte-for-byte
for all three cases. That small-budget run checks implementation parity, not
usable image quality. The comparison adapter preserves fp32 reference pixels
at the VAE boundary on both sides; upstream otherwise rounds them to text dtype.

Relevant regression runs passed: the broad config/generation/data/family/media
selection (763 passed, 1 skipped), rollout and evaluator checks (112 passed),
and final focused checks after the geometry guard (10 passed). These selections
overlap and should not be added together as a unique test count.

## Localized attribute and object-replacement probe (2026-09-21)

A separate, untrained official `QwenImage21Pipeline` probe uses two real source
photographs to distinguish two task definitions:

- **Attribute editing:** recolor an armchair or sweater; replace fabric upholstery
  with leather while retaining the chair's silhouette.
- **Whole-item replacement:** replace the upholstered chair with a bamboo chair
  or a wooden stool, allowing geometry changes and newly visible background.

Artifacts are saved locally under `outputs/qwen_image_21_local_edits/`:
`index.html` contains original/edit pairs and opacity overlays; `cases.json`
contains the exact prompts; `sources.json` attributes the photographs;
`run_probe.py` reproduces/resumes the official-pipeline experiment;
`report.json` records settings and timings. The experiment uses bf16, native
Euler, CFG 1, 40 steps, seed 42 per case, KV cache, and
`output_resolution=1024` (832 × 1248 outputs for these portraits). The text
encoder and untiled VAE run on CPU; the transformer uses streamed block offload.
No masks, LoRA, or post-edit background restoration are used.

The first tiled-VAE attempt introduced colored vertical marks on the wall.
A source-photo VAE round trip reproduced them without any diffusion steps:
bf16 tiled RGBA MAE was 1.8083/255, versus 1.1375/255 untiled. Changing the
*tiled* VAE to fp32 did not resolve them (1.8465/255). All main-gallery results
therefore use the untiled VAE. This isolates a tiled-path artifact in this
setup; it is not a general diagnosis of the upstream implementation. The
initial output and round-trip checks are retained with the artifacts.

Protected-patch pixel differences are descriptive diagnostics, not rewards or
semantic success scores. The unchanged-room instruction control also passes
through generation; it is not an identity operation. Two photos and one seed
per edit do not establish reliability or demonstrate a training gain.

In the completed probe, the chair/sweater recolors, leather upholstery, bamboo
chair, and wooden stool followed their principal instructions visually.
Bamboo/stool changed the silhouette and exposed background, as expected for
replacement tasks. Fine print and alignment were not exact copies. The
unchanged-room control explicitly failed: it added a small flower vase to the
side table. Keep this failure in the gallery rather than presenting only
successful edits. `observations.json` records the per-image visual review.
