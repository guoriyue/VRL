# Layering on a randomized scene family

Supervised (not RL) training of a LoRA on Qwen-Image-Layered so that one photo of a room decomposes
into a clean room layer plus one layer per object, with occluded parts completed where they can be
inferred. Training data: procedurally rendered rooms with full per-layer truth, optionally mixed with
RORD real photos. Evaluation: rendered held-out scenes, RORD held-out photos (true background) and
12 real room photos (no truth).

These are research scripts, kept outside the `vrl` package. Each one is a CLI (`--help`); run them
from this directory so sibling imports resolve. Results, data and checkpoints live outside git.

## Results so far (25-step Heun, seed 4401)

| Model | Rendered held-out object match | RORD background recovery | 12 real rooms (visual) |
| --- | --- | --- | --- |
| Official Stable-Layers adapter | 0.00 | 0.153 | baseline |
| Sim only, weighted, 2000 updates | 0.56 | 0.455 (1000 updates) | better on 10 of 12 |
| Sim only, unweighted control, 2000 | 0.62 | 0.312 (1000 updates) | weighted better on 6, tied on 6 |
| E: sim + RORD coarse polygons + spare layers, 1000 | 0.30 | 0.479 | worse on 12 of 12 |
| D2: sim v2 + v1 + RORD difference matte + spare layers, 2000 | running | running | running |

Lessons: rendered data alone transfers to real photos. RORD with coarse polygons taught object
layers that carry chunks of wall and floor. RORD's room layer also contains every piece of furniture
(only the person is removed), which tells the model that furniture belongs to the room, against the
goal. The step-4 RL gate (object match >= 0.8) is not met.

## Pipeline

| Step | Script | Notes |
| --- | --- | --- |
| Render scenes | `render_scene_family.py` (v1, 9 classes, orthographic), `render_scene_family_v2.py` (v2, 16 classes, perspective, real-photo paintings) | Blender 4.0.2, Cycles on GPU |
| Generate a split | `generate_scene_family.py` | rejects scenes without both inferable and unknowable hidden regions |
| Cache | `cache_scene_family.py` | condition tokens, prompt embeds, target latents, weighted/control token weights |
| Spare empty layers | `encode_empty_frame.py` | latent of a transparent frame for `--extra-empty-max` |
| Train | `train_scene_family.py` | single GPU, `--arm weighted|control`, `--resume-from`, `--extra-dataset` |
| Evaluate | `evaluate_scene_family.py` | layered_reference axes; `--no-truth` for real photos; RORD background recovery |
| Continue rule | `decide_continue.py` | checkpoint gate used in the 2026-10-09 runs |
| Real rooms | `prepare_real_rooms.py` | caches photos for truth-free evaluation |
| RORD 2-layer | `build_rord_layers.py`, `cache_rord_layers.py` | background = true empty-room photo |
| RORD foreground | `build_rord_matte.py` (difference matte), `refine_rord_sam.py` (SAM 2.1 + ViTMatte), `clean_rord_fragments.py`, `fragment_stats.py` | quality ladder: polygon < matte < SAM |
| RORD one layer per object | `build_rord_sam3_layers.py` | SAM 3 text prompts; not wired into `cache_rord_layers.py` yet (it assumes 2 layers) |

`runs/` holds the queue scripts that produced the results above (absolute paths of the original host).

## Setting up another machine

1. Environment (Python 3.12): torch 2.11 (CUDA 13.0 build), diffusers at commit
   `80c7ed262aeffbeb43ef13ae04baeb9b84515a69`, transformers 5.17.0, peft 0.19.1, accelerate 1.14,
   safetensors 0.8, numpy 2.3, Pillow 12, plus this repo installed editable (`uv pip install -e .`).
   Relabeling RORD additionally needs SAM 2.1 / SAM 3 (`SAM2_PATH`, `--sam3`); training does not.
2. Base model: `huggingface-cli download Qwen/Qwen-Image-Layered --revision 8f0ca708dfff6ba1dd5f2d85d78f8c108a040bcf`.
3. Data bundle (about 11 GB): on the source host run `./make_bundle.sh /path/bundle`, then
   `rsync -aL --info=progress2 /path/bundle/ newhost:/data/layering/`. It holds the cached rendered
   datasets (`dataset/`, `dataset_v2/`), RORD (`rord_layers/` for evaluation, `rord_layers_matte/`
   for training), `real_rooms/`, `empty_frame_latent.pt`, the official adapter as the training start
   (`initial_adapter.pt`), the source plan and the pinned upstream `decompose.py` (with its license).
4. Plan with local paths:
   `python make_plan.py --plan /data/layering/plan.source.json --base-path <snapshot dir> --upstream /data/layering/upstream/decompose.py --out /data/layering/plan.json`.

## Four GPUs

`train_scene_family.py` uses one GPU; run one arm per GPU. `run_arm.sh` trains 2 x 1000 updates and
then runs every evaluation; `VRL_GPU_LOCK` keeps arms from sharing a card.

```bash
export DATA=/data/layering PLAN=/data/layering/plan.json PY=$(which python)
GPU=0 ARM=sim_only            SEED=20261011 ./run_arm.sh &   # D2's data without RORD: is RORD helping?
GPU=1 ARM=sim_only_s2         SEED=20261012 ./run_arm.sh &   # seed repeat
GPU=2 ARM=sim_rord_matte_s2   SEED=20261012 ./run_arm.sh &   # D2 with a second seed
GPU=3 ARM=sim_only_control    SEED=20261011 WEIGHTING=control ./run_arm.sh &   # weighting ablation on v2
wait
```

On a 32 GB RTX 5090 (transformer blocks streamed from host memory) one update takes about 12.5 s,
so 2000 updates take about 7 hours, and the evaluations about 5 more.
