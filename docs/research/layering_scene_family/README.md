# Layering on a randomized scene family

Supervised (not RL) training of a LoRA on Qwen-Image-Layered so that one photo of a room decomposes
into a clean room layer plus one layer per object, with occluded parts completed where they can be
inferred from the image (inferable regions are scored against truth; unknowable regions only for
plausibility). Training data: procedurally rendered rooms (Blender, outside this repository) with full per-layer truth, optionally mixed
with RORD real photos. Evaluation: rendered held-out scenes, RORD held-out photos (true background),
12 real room photos and 24 diverse photos (no truth).

No RL update has been run. The step-4 RL gate (held-out object match >= 0.8) is not met; the work is
parked, see [`docs/sprints/parked/SPRINT_layering_scene_family.md`](../../sprints/parked/SPRINT_layering_scene_family.md).

Reports in this folder:

- [`single_scene_20261008.md`](single_scene_20261008.md): the earlier single-scene (room_chair) phase.
- [`scene_family_20261009.md`](scene_family_20261009.md): the scene family up to arm D2, and the
  2026-10-11 summary against the goal's step criteria.
- [`run_arm.sh`](run_arm.sh): one training arm with every evaluation.

Results, data and checkpoints live outside git.

## Pipeline

Every entry is a CLI (`--help`). Run the `python -m` modules from the repository root with the repo
installed editable.

| Step | Command | Notes |
| --- | --- | --- |
| Cache scenes | `python -m vrl.scripts.denoise.encode_layered_scenes --root <dataset>` | rendered scenes and RORD samples alike: condition tokens, prompt embeds, target latents, `weights_weighted` / `weights_control`, plus `empty_frame.pt` (the spare-layer target); `model.use_lora=false` skips the adapter, `precision.rollout.quantization.format=fp8` fits a 32 GB card |
| Cache photos | `... encode_layered_scenes --photos <jpg>... --out <dir> --layers N` | real rooms, diverse photos, COCO photos; truth-free; fixed layer budget |
| Train | `vrl-train --config experiment/qwen_image_layered/offline_sft_layered_scenes` | one GPU; `data.manifest` (root or `{root: scene count}`), `data.token_weights=weighted\|control`, `data.spare_layers_max`, `model.lora.path`, `trainer.seed`, `trainer.resume_from`; `actor.stream_frozen_blocks=true` on 32 GB cards; schedule indices `[0, 48)`, lr 1e-4 |
| Evaluate | `python -m vrl.scripts.eval.layered_decompose_eval` | `--experiment --dataset --split --adapter <PEFT dir or checkpoint dir> \| --base --seeds --steps --max-scenes --no-truth --output`; fp8 override on 32 GB cards |

The Blender renderers (the randomized rooms, their v3 one-object-per-layer variant, and the
Infinigen-Indoors layering) are data generation, not training, and live outside this repository;
they produce the scene directories (`source.png`, `layer_<i>.png`, `inferable_mask.png`,
`unknowable_mask.png`) that `encode_layered_scenes` reads. Deleted from this repository, recoverable
from git history: the renderers themselves, the SAM 2.1 / SAM 3 refine, clean and fragment-stats
ladder for RORD foregrounds, `decide_continue.py` (the 1000-update continue rule of 2026-10-09),
`make_bundle.sh` and the `runs/*.sh` queues of the original host.

## Setting up another machine

1. Environment (Python 3.12): torch 2.11 (CUDA 13.0 build), diffusers at commit
   `80c7ed262aeffbeb43ef13ae04baeb9b84515a69`, transformers 5.17.0, peft 0.19.1, accelerate 1.14,
   safetensors 0.8, numpy 2.3, Pillow 12, plus this repo installed editable (`uv pip install -e .`).
   Relabeling RORD with SAM is no longer in the repo; training does not need it.
2. Base model:
   `huggingface-cli download Qwen/Qwen-Image-Layered --revision 8f0ca708dfff6ba1dd5f2d85d78f8c108a040bcf`.
3. Data directory (`$DATA`, about 11 GB, copied with `rsync -aL`). Caches are pinned to the model
   revision, so caches built on another machine stay valid:

   | Path | Contents |
   | --- | --- |
   | `dataset/` | v1 rendered scenes, cached: 200 train / 24 held-out |
   | `dataset_v2/` | v2 rendered scenes, cached: 1000 train / 48 held-out |
   | `rord_layers/` | RORD indoor 2-layer, polygon foreground; evaluation (8 held-out scenes, 20 frames) |
   | `rord_layers_matte/` | RORD with difference-matte foreground; training (177 train / 19 held-out frames) |
   | `real_rooms/` | 12 OmniEdit/GEdit indoor photos (MIT), no truth |
   | `diverse_photos/` | optional: 24 photos, no truth (8 people and 8 outdoor from GEdit-Bench, MIT; 8 anime generated with Anima) |
   | `stable_layers_peft/` | official Stable-Layers adapter as a PEFT directory (training start) |

   `stable_layers_peft/` is the official Stable-Layers adapter, a flat `{name: tensor}` file whose
   names follow the diffusers transformer (`transformer_blocks.<n>.attn.to_q.lora_A.default.weight`),
   written the way `model.lora.path` loads it: `peft.LoraConfig(r=32, lora_alpha=64,
   target_modules=[...the model preset's...]).save_pretrained(dir)` for `adapter_config.json`, and
   `adapter_model.safetensors` holding the tensors under `base_model.model.<name>` with the
   `.default` adapter infix removed. Caches written by the pre-framework scripts (`cache.pt` without a
   `schema` key) are not read any more: re-encode the scenes.

## Arms

All arms start from the official adapter, train with lr 1e-4 over schedule indices [0, 48) and
evaluate with seed 4401, 25-step Heun, layer budget = truth layer count.

| Arm | Training data | Weighting | Spare empty layers | Updates | Seed |
| --- | --- | --- | --- | --- | --- |
| weighted | v1 train (200) | weighted (inferable hidden tokens carry 25% of the loss) | none | 1000 + 1000 (`--resume-from`) | 20261009 (default) |
| control | v1 train (200) | control (all ones) | none | 1000 + 1000 | 20261009 (default) |
| E | v1 (200) + RORD polygon (180) | weighted, RORD foreground weight 0.2 | 0-2 | 1000 | 20261010 |
| D2 | v2 (1000) + v1 (200) + RORD matte | weighted, RORD foreground weight 0.5 | 0-2 | 1000 + 1000 | 20261011 |
| sim_only | v2 (1000) + v1 (200) | weighted | 0-2 | 1000 + 1000 | 20261011 |
| sim_only_s2, sim_only_s3 | as sim_only | weighted | 0-2 | 1000 + 1000 | 20261012, 20261013 |
| sim_only_control | as sim_only | control | 0-2 | 1000 + 1000 | 20261011 |
| sim_only_spare4 | as sim_only | weighted | 0-4 (`--extra-empty-max 4`) | 1000 + 1000 | 20261011 |
| sim_rord_matte | v2 + v1 + RORD matte (D2's data) | weighted | 0-2 | 1000 + 1000 | 20261011 |
| sim_v1_only | v1 (200) | weighted | 0-2 | 1000 + 1000 | 20261011 |

Frozen controls: original Qwen-Image-Layered (`--base`: no adapter, which is what the earlier
`--zero-b` arm's zeroed `lora_B` amounted to) and the official Stable-Layers adapter. The `sim_only*` arms ran on a four-L40S host; their
results are not summarized here (see the cross-machine note below).

## Results (25-step Heun, seed 4401)

| Model | Rendered v1 held-out object match | RORD background recovery | 12 real rooms (visual) |
| --- | --- | --- | --- |
| Original Qwen (base) | 0.070 | n/a | worst: room layer blank or blurry, room split into holed fragments |
| Official Stable-Layers adapter | 0.001 | 0.153 | baseline |
| weighted, 2000 updates | 0.562 | 0.455 (1000 updates) | better than official on 9 of 12; clean room layer 10 of 12 |
| control, 2000 updates | 0.619 | 0.312 (1000 updates) | weighted better on 6, tied on 6, worse on 0 |
| E, 1000 | 0.296 | 0.479 | worse than weighted 2000 on 12 of 12 |
| D2, 2000 | 0.334 | 0.526 | vs weighted 2000: better 2, worse 6, tied 4 |

Rendered v1 held-out (24 scenes), all axes:

| Axis | Original Qwen | Official adapter | weighted 2000 | control 2000 |
| --- | --- | --- | --- | --- |
| Object match | 0.070 | 0.001 | 0.562 | **0.619** |
| Inferable completion | 0.842 | 0.821 | **0.958** | 0.929 |
| Unknowable plausibility | 0.627 | 0.618 | **0.735** | 0.723 |
| Visible retention | 0.886 | 0.827 | 0.975 | **0.987** |
| No holes after hide/move | 0.953 | 0.909 | 0.984 | **0.985** |
| Composite coverage | 1.000 | 1.000 | 1.000 | 1.000 |

Weighted minus control per scene: object match -0.057 (weighted better on 15/24), inferable
completion +0.030 (20/24), unknowable plausibility +0.012 (15/24).

Per-object rescore (matched = share of objects with layer quality >= 0.5; product = `layered_object_match`):

| Set | Arm | Matched | Mean quality | Product |
| --- | --- | --- | --- | --- |
| v1 (24) | control 2000 (v1 only) | 0.967 | 0.916 | 0.619 |
| v1 (24) | weighted 2000 (v1 only) | 0.941 | 0.899 | 0.562 |
| v1 (24) | D2 | 0.864 | 0.807 | 0.334 |
| v1 (24) | E | 0.834 | 0.792 | 0.296 |
| v1 (24) | official | 0.587 | 0.544 | 0.001 |
| v2 (24) | control 2000 (never saw v2) | 0.780 | 0.752 | 0.179 |
| v2 (24) | D2 (trained on v2) | 0.662 | 0.662 | 0.083 |
| v2 (24) | official | 0.579 | 0.524 | 0.032 |

RORD held-out (20 real photos, 2 layers):

| Arm | Background recovery | Object match | Visible retention | Edit holes |
| --- | --- | --- | --- | --- |
| E | 0.479 | 0.398 | 0.882 | 0.952 |
| weighted 1000 (v1 only) | 0.455 | 0.226 | 0.798 | 0.943 |
| control 1000 (v1 only) | 0.312 | 0.169 | 0.770 | 0.924 |
| official | 0.153 | 0.215 | | |

D2 has the best RORD background recovery (0.526) and, scored against the matte truth, the best
object match (0.552).

Step-2 criteria on the v1 held-out: (a) object match >= 0.8 and above both controls: **not met**
(best 0.619); (c) inferable completion error down >= 50%: met (0.179 -> 0.042, -77%); (e) visible
error not above the controls by 5%: met (0.173 -> 0.025); (f) composite coverage >= 0.995: met (1.000).

## Lessons

- Rendered data alone transfers to real photos; the clearest real-photo gain is a complete, clean
  room layer.
- Occlusion weighting is not the cause of the gain: the unweighted control has higher object match.
  Weighting gives a small, consistent inferable-completion gain and is better on real rooms (6 better, 0 worse).
- Shape generalization is limited: reserved test shapes (double-door cabinets) are often replaced by
  shapes seen in training.
- RORD coarse polygons taught object layers that carry chunks of wall and floor (arm E). RORD's
  background layer keeps every piece of furniture (only the person is removed), which tells the model
  that furniture belongs to the room, against the goal.
- D2 (matte instead of polygon) still regresses on rendered scenes and loses to the v1-only control
  even on the v2 distribution it trained on; it outputs fewer non-empty layers than the truth in 13/24
  v2 scenes (control 2000 8/24, official 0/24). The two suspects, RORD two-layer data and the 0-2
  spare empty layers, are not separated yet.
- Do not use RORD polygons as foreground truth or compare models on object match against them; do not
  count "no holes after hide" on real photos (1.0 for every model); do not count more layers as
  progress; do not use fully hidden parts of a training scene as completion evidence.
- Deviations from the plan: 1 seed x 25 Heun steps instead of 2 seeds x 50 (identical for every
  arm); reward-skill gates 1 and 3 were not run as separate reports (the axes are metrics only).

## Compare arms on one machine

The same adapter and seed give different per-scene scores on a remote L40S and on the local RTX 5090.
For the `sim_only_spare4` adapter, seed 4401, 25 steps, v2 held-out `scene_60000`: inferable completion
0.962 on the L40S and 0.854 on the 5090. Evaluate every arm being compared, including the frozen
controls, on the same machine and card type.

## Four GPUs

The trainer uses one GPU; run one arm per GPU with `CUDA_VISIBLE_DEVICES`. `run_arm.sh` trains
2 x 1000 updates (the second segment resumes from the first's final checkpoint) and then runs every
evaluation. Arm presets: `sim_only*` (v2 + v1), `sim_rord_matte*` (v2 + v1 + RORD matte),
`sim_v1_only*` (v1).

```bash
export DATA=/data/layering
cd <repo root>
R=docs/research/layering_scene_family/run_arm.sh
GPU=0 ARM=sim_only         SEED=20261011 $R &                     # D2's data without RORD: is RORD helping?
GPU=1 ARM=sim_only_s2      SEED=20261012 $R &                     # seed repeat
GPU=2 ARM=sim_only_spare4  SEED=20261011 SPARE_LAYERS_MAX=4 $R &  # more spare layers
GPU=3 ARM=sim_only_control SEED=20261011 WEIGHTING=control $R &   # weighting ablation on v2
wait
```

On 32 GB cards set `STREAM=true` (the frozen transformer blocks stay on the host; one update takes
about 12.5 s, so 2000 updates take about 7 hours, and the evaluations about 5 more) and
`EVAL_EXTRA=precision.rollout.quantization.format=fp8`. RORD is used for evaluation
(`rord_layers/`), not training, in the `sim_only*` arms. The arms in the tables above were trained
by the pre-framework scripts; the framework sampler draws its initial noise differently, so new
runs are compared with each other, not with those tables.
