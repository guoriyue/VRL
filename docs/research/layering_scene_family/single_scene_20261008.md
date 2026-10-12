# Layered RL progress, single scene (2026-10-08, following NEXT_MODEL_PROMPT_v2)

Condensed English translation of `PROGRESS_20261008_vrl.md`. Paths under `outputs/` refer to the
original host and are not in git; commit hashes are from the original `layering` branch.

Status: phase A supervised training finished 200 updates. A1 passed, A2 is 8/8, held-out rendered
scene 2/2 with the correct layout; real-photo regression shows composite holes (section 3). No RL
update was started. Every conclusion is limited to the single room_chair scene.

The commit hashes below predate the 2026-10-12 rebase. That cleanup removed this phase's
single-scene RL and probe scripts, the SFT regularizer weight map and the room_chair GRPO recipe,
and turned the `layered_*` rewards into evaluation metrics (`vrl/scripts/eval/layered_metrics.py`);
the local branches `layering-rl-diagnostics` and `layering-before-metrics` keep the removed code.

## 1. Step 0: VRL binding (done and committed)

| Commit | Content |
| --- | --- |
| `a71a01af3` | Register the `qwen_image_layered` family: reuses the `qwen_image` MMDiT and 4-channel VAE, loads through the checkpoint's own `QwenImageLayeredPipeline`; the source image is encoded as condition tokens appended after the generated tokens and cut from the output; the generated sequence is `layers+1` frames; `decode_latents` returns `[B, layers, 4, H, W]`; `encode_video_to_latents` is its exact inverse (verified on CPU with an invertible fake VAE); `sft_weight_map` places pixel masks on the packed token axis |
| `877393a35` | Fix: after PEFT wrapping, module paths carry a `base_model.model.` prefix, and the exclusion substring `_mod` matched `base_model`, so fp8/nvfp4 rollout quantization of LoRA families swapped no linear at all. Supervised probe gains `--schedule-indices` and `--checkpoint-every` |
| `37894db84` | `encode_targets` accepts an RGBA PNG frame directory as `target_video`; the family declares its frame count through `clean_target_frames` |
| `077a03bcb` | `normalized_sum` only takes components with a configured weight and ignores extra observation axes (a multi-axis reward otherwise failed under the default policy) |
| `bcb365276` | `layered_reference` reward with four components (hidden completion, object match, visible retention, edit holes), wrapping `layered_reference_score.score_layers` |
| `365d3f848`, `06a34d2ea` | room_chair dataset / reward / experiment preset skeleton (parses, but cannot run on one 32 GB card, see section 4) |

Real-checkpoint smoke (`vrl_family_smoke_20261008/`): VRL registry build path; rollout fp8 quantizes
720 attention/MLP linears, the text encoder stays on the host, peak memory 28.4 GB, 20-step Euler ODE
in 26 s. Four output layers: wall / floor + skirting / cabinet + lamp base / chair + lamp shade.
Composite vs source MAE 0.0100 (probe candidate00: 0.0217). Cabinet without feet and the lamp split
over two layers match the historical failure. This proves loading, conditioning, packing and RGBA
decoding are correct; it is not the probe's Heun/SDE sampler, and fp8 output is not bitwise
comparable with bf16.

## 2. Phase A: supervised candidate support (not RL)

Protocol written before launch (`occlusion_fullrange_support200/protocol.json`):

- Parent: the official Stable-Layers adapter mapped to the training LoRA layout (sha256 a4771baa...).
- Targets: `room_chair_codec/target_latents.pt` (composite_source, room, nightstand, lamp, chair).
- Spatial weights (`occlusion_spatial_weights.pt`): 12813 cabinet pixels hidden by the chair, i.e.
  81 tokens of the nightstand frame, weighted `1 + 53.3 x coverage`; they carry 25.8% of the loss
  mass. Other tokens weigh 1; overall mean 1.
- Timesteps: schedule index [0, 48), sigma 1.0 to 0.13 (index 48 is 0.094). The earlier fit32 used [1, 50).
- Budget: 200 updates, lr 1e-4, seed 20261008, checkpoints at 50/100/150/200; about 9 s per update.
- A1: `probe_known_layer_restart.py --starts 40` (teacher initialization, not normal inference),
  seeds 4201/4202, metric `hidden_feet_mean_alpha` (`score_cabinet_feet.py`; all three valid cabinet
  feet are behind the chair). Baseline: start 40 gives 0.012, start 48 gives 0.972. Pass: >= 0.5 and
  visible-region error up by <= 10% over the parent.
- A2: 8 unused seeds (4401-4408), 50-step native Heun, parent and final checkpoint.

## 3. Results

Training (`occlusion_fullrange_support200/training/`): all 200 updates done, 836 LoRA tensors changed,
largest parameter change 0.0138, reloaded velocity bitwise identical. Fixed-noise weighted loss
(before -> after): index 10: 0.00271 -> 0.00013; index 25: 0.00276 -> 0.00017; index 40: 0.00436 ->
0.00039. Mean training loss by block: updates 1-50 0.00086, 51-100 0.00053, 101-150 0.00046, 151-200
0.00026. A falling loss is not completion evidence (sft32 cut velocity MSE by 91% and still lost the
feet); the gates are A1/A2.

A1 (teacher-initialized restart at start 40, sigma 0.385; not normal inference). Hidden-feet mean
alpha / visible cabinet RGBA error:

| Checkpoint | Seed 4201 | Seed 4202 | Gap-between-feet alpha |
| --- | --- | --- | --- |
| Parent (official adapter, historical start40) | 0.012 / 0.0342 | - | 0.003 |
| adapter_50 | 0.010 / 0.0091 | 0.135 / 0.0090 | 0.003 / 0.027 |
| adapter_100 | 0.829 / 0.0088 | 0.990 / 0.0082 | 0.006 / 0.142 |
| adapter_150 | 0.972 / 0.0085 | 0.984 / 0.0083 | 0.006 / 0.013 |
| adapter_200 | 0.984 / 0.0085 | 0.976 / 0.0082 | 0.004 / 0.008 |

A1 passed (>= 0.5; visible error fell instead of rising). From 100 updates on, the three hidden feet
survive denoising from sigma 0.385, where the parent erases them. This refutes "the model keeps
occluded content only at sigma <~ 0.1" and supports the intervenable part of the hypothesis: erasure
can be removed by supervision that covers the whole trajectory and weights the occluded region. It is
still not evidence for normal full-noise inference.

A2 (normal 50-step native Heun, 640, 4 layers, 8 never-used seeds 4401-4408). Counting rules written
into protocol.json before the final results: complete hidden feet = hidden-feet mean alpha >= 0.5 and
gap alpha <= 0.2; lamp/cabinet separation = lamp-region alpha in the cabinet layer <= 0.5.

- Parent: complete hidden feet 0/8 (best 0.21), lamp/cabinet separation 0/8 (lamp alpha in the
  cabinet layer >= 0.99), visible cabinet error 0.028-0.041.
- Final (200 updates): complete hidden feet **8/8** (alpha 0.934-0.996, gap <= 0.055), separation
  **8/8** (lamp alpha in cabinet layer 0.188, matching truth), visible cabinet error 0.0076-0.0094,
  cabinet alpha IoU 0.987-0.992. Per-seed numbers in `eval_parent/feet_scores.json`,
  `eval_final/feet_scores.json`; figure `a2_seed4401_parent_vs_final_layers.png`.

Conclusion (training scene only): A2 lands in the planned 8/8 branch. On room_chair, supervision over
sigma in [0.13, 1] weighted on the occluded region makes normal full-noise inference reliably produce
complete hidden feet and separate the lamp from the cabinet, which no earlier RL or supervised attempt
did. By the NEXT_STEP_PLAN branch rule, phase B RL is not needed for this scene; RL's purpose shifts
to generalization. This was achieved on new seeds of the **training scene itself**: the model saw its
full truth layers, so it does not show general occlusion completion.

Regression and held-out:

- Regression (kitchen, bedroom, photo872, manga003; historical 8-layer protocol, seed 4401, no
  truth, visual check only; `regression_parent/`, `regression_final/`, `sheet_regression_*.png`).
  RGB reconstruction error (0-255) parent -> final: kitchen 15.3 -> 3.3, bedroom 10.9 -> 6.1,
  photo872 37.8 -> 10.5, manga003 13.8 -> 8.5. Composite alpha error: kitchen 0 -> 0.16, bedroom
  0 -> 0.37, photo872 0 -> 0.77, manga003 0 -> 0. Visually, kitchen and bedroom backgrounds are
  completed as empty rooms with the person/bed removed (plausible; a thin transparent seam by the
  bedroom window); photo872's background splits into a ground layer and a tree layer that does not
  cover fully, leaving small composite holes; manga003 is fine. The final model changes how the
  background is allocated and risks composite holes: a real regression that later objectives must
  constrain explicitly (opaque composite coverage).
- Held-out desk_crate (`heldout_desk_crate/`, same renderer: desk front legs hidden by a crate, a
  potted plant on the desk, different wall color and floor material; never used in training or
  calibration; seeds 4401/4402; `heldout_parent/`, `heldout_final/`,
  `sheet_heldout_desk_crate_seed4401.png`). `layered_reference` axes, parent -> final, mean of two
  seeds: object match 0.001 -> 0.88, visible retention 0.82 -> 0.99, hidden completion 0.92 -> 0.98,
  edit holes 0.92 -> 0.99, reconstruction 0.98 -> 0.99; hidden desk region alpha 0.96 -> 0.98.
  Visually the parent splits the floor into its own layer and merges the plant into the desk; the
  final model's four layers match the truth layout (room, desk with complete legs behind the crate,
  plant, crate). One held-out scene, two seeds, same rendering style (camera, lighting, materials):
  a generalization signal, not real-photo evidence.

## 3b. Overall

1. The causal hypothesis is supported: erasure of occluded parts can be removed by supervision that
   covers the whole trajectory and weights the occluded region; within 200 updates this holds on A1,
   A2 and one held-out rendered scene.
2. On room_chair supervision solves complete hidden feet plus lamp/cabinet separation, so phase B RL
   no longer targets this scene; RL's target becomes generalization and real photos.
3. Cost: composite holes on photo872 and bedroom (background split into layers that do not cover).
   Any later objective needs an opaque-coverage constraint or reward component (`layered_reference`
   has `nonredundancy`/`reconstruction` but no direct coverage term; `exterior_alpha_mean` and alpha
   coverage could become axes).
4. Next (not run): (a) >= 2 more held-out layouts from the same renderer; (b) add known-layer scenes
   beyond room_chair to supervision and see whether the holes go away; (c) run reward-skill gates 1-4
   on the `layered_*` rewards before online GRPO on many scenes; (d) close the trainer-memory and
   shared-initial-latent gaps of section 4.

## 4. VRL gaps for phase B

- Trainer memory: the 20B base in bf16 (40 GB) does not fit a 32 GB card. Rollout is solved by fp8
  quantization; for the trainer/replay side VRL has `distributed.training.strategy=fsdp` +
  `fsdp.cpu_offload` (FSDP2 CPUOffloadPolicy), which could stream the frozen base from the host, but
  it needs torchrun and was not verified here. The probe scripts use their own `FrozenCPUBlock`
  streaming.
- Shared initial latent within a group: the executor uses `seed + sample_start` per sample, so
  initial noise differs within a group; NEXT_STEP_PLAN needs a shared initial latent so advantages
  reflect actions, not noise. VRL has no such switch yet.
- Reward qualification: the `layered_*` rewards are registered with CPU tests, but reward-skill gates
  1-4 (repeatability, oracle agreement, shortcut pressure, in-group spread under the training
  sampler) were not run. They must not enter a training key before passing.
- SFT shard: `encode_targets` can encode RGBA frame directories, but
  `data/external/layered/room_chair/sft_latents.pt` was not generated on GPU.

## 5. Do not repeat

Gradient-existence checks, fresh Adam, single-timestep supervision, low-noise same-batch multi-epoch
PPO, continuing RL updates while every candidate lacks the cabinet feet. Confirmed here as well: VRL's
`qwen_image` t2i family has no image conditioning or multi-layer output; the `qwen_image_layered`
variant is required.
