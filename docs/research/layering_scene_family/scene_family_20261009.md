# Layering generalization on a scene family (2026-10-09 to 2026-10-11, following NEXT_MODEL_PROMPT_v3)

Condensed English translation of `PROGRESS_20261009_scene_family.md` and `SUMMARY_20261011.md`.
Paths under `outputs/` and `scene_family/` refer to the original host and are not in git; script
names are the ones used at the time (see [README.md](README.md) for the current module paths).

## Summary (2026-10-11)

Steps 1 and 2 are done. Step 3 is refuted on its predeclared criterion (a): best object match 0.619
< 0.8. The precondition of step 4 (RL) is not met; no RL update was run. Every number below is
diagnostic inference after supervised updates.

Holds on the held-out set:

- Layering against the frozen controls: v1 held-out (24 scenes) object match original 0.070 /
  official 0.001 -> weighted 0.562 / control 0.619. Per object, the matched share goes from official
  0.587 to control 0.967. Far fewer merged and missing objects; not memorization (held-out class
  combinations and 6 parameter ranges never appear in training).
- Inferable occlusion completion: error official 0.179 -> weighted 0.042, down 77% (criterion (c) met).
- Composite coverage 1.000 ((f) met); visible-region error 0.173 -> 0.025 ((e) met); no holes after
  hide/move 0.909 -> 0.984.
- Unknowable-region plausibility 0.618 -> 0.735 (continuation and color continuity only, not truth).
- 12 real rooms (visual): weighted beats official on 9/12; clean, complete room layer on 10/12.

Does not hold or refuted:

- (a) object match >= 0.8: not met, step 3 is refuted here. The product metric is 0 for a whole
  scene when one object is missed; best 0.619.
- Occlusion weighting as the cause of the gain: not supported. The unweighted control on the same
  data has higher object match (0.619 vs 0.562); weighting is only slightly and consistently better
  on inferable completion (+0.030, 20/24) and better on real rooms (6 better / 0 worse).
- Shape generalization: limited; test-only shapes (double-door cabinets etc.) are often replaced by
  shapes seen in training.
- Adding real data (RORD) and random spare layers: E and D2 are better only on RORD background
  recovery (0.526) and regress on the rendered held-out. D2 loses to the v1-only control even on the
  v2 distribution it trained on (matched objects 0.662 vs 0.780) and outputs too few layers in 13/24
  scenes. E's regression comes from RORD polygon truth (object layers with wall/floor chunks); D2
  still regresses with a difference matte; the suspects are the two-layer fg/bg data and the random
  spare layers, not yet separated.

Deviations from the plan: evaluation uses 1 seed (4401) x 25 Heun steps per scene instead of 2 seeds
x 50 steps, identical for every arm. Reward-skill gates 1 and 3 were not run as separate reports;
reward components are evaluation metrics only and never entered a training key (no RL, so not
required). The v2 scenes, RORD, spare layers and Infinigen scenes are unplanned extensions for
diagnosing generalization; they do not change the step-3 verdict.

Do not repeat: RORD polygon regions as foreground truth, or object match against polygon truth to
compare models; "no holes after hide" as real-photo evidence (1.0 for every model); more layers as
progress (E's fewer layers are merging, not less fragmentation); "complete cabinet feet" on a single
scene as completion evidence (fully hidden parts are memorized).

Next (needs a decision): a v2 + v1 control without spare layers and without RORD, to separate the
two suspects (every arm on the four-GPU machine uses spare layers, so their conclusions wait on it).
Infinigen-Indoors scenes (`render_infinigen_layers.py`) are being generated in batches as a more
realistic multi-layer training distribution. No RL before object match reaches 0.8.

## 1. Scene distribution (`scene_family/`)

- Generator `render_scene_family.py` (Blender 4.0.2, Cycles GPU, 16 samples, 640x640): 9 parametric
  classes (cabinet, table, chair, lamp, potted plant, crate, bookshelf, sofa, stool), 2-6 foreground
  objects per scene, random materials, colors, positions, camera azimuth and lighting; one room layer,
  one layer per object, ordered far to near.
- Occlusion by design: the first two objects of each scene form a "target behind, occluder in front"
  pair, the occluder drawn from the wide classes.
- Labels: each part is rendered as its own ID pass to get its visible share. A part more than half
  hidden is *inferable* if >= 25% of it is still visible (continuation) or a same-class sibling part
  is >= 60% visible (symmetry), otherwise *unknowable*. Outputs `inferable_mask.png`,
  `unknowable_mask.png`, `occlusion.json`. Thresholds are geometric rules at generation time, no
  manual labels.
- Acceptance: >= 150 px each of inferable and unknowable; held-out class combinations must not
  appear in train.
- Size: train 200 (313 rejected), held-out 24 (49 rejected), zero combination overlap. Reserved
  held-out ranges: tapered table legs, double-door cabinets (door ratio 0.6-0.8), cylindrical lamp
  shades, 4-5 shelf bookcases, blue wall hue 0.55-0.65, red chairs.
- Check: 7 train + 3 held-out scenes sampled at random; 0 composite hole pixels, layer count = objects
  + 1, sensible labels (`dataset/acceptance_sheet.png`, `acceptance_sample.json`). Known bias: the
  rejection rule favors 5-6 object scenes (131/200 in train).
- Cache `cache_scene_family.py`: condition tokens, prompt, target latents (composite frame plus each
  layer encoded separately and concatenated), two token weightings (weighted arm: tokens of inferable
  hidden regions carry 25% of the loss, unknowable regions are not raised; control arm: all ones).

## 2. Evaluation protocol (frozen before training)

Axes come from `vrl/scripts/eval/layered_metrics.py` (then a reward model, commit `070c3989c`): object match, visible
retention, hide/move edit holes, reconstruction, non-redundancy, plus new composite coverage,
inferable-region completion (against truth) and unknowable-region plausibility (only whether the
matched layer continues and stays color-continuous with its own visible part, never truth pixels).
Controls: original Qwen (official adapter with `lora_B` zeroed) and the frozen official Stable-Layers
adapter. Layer budget = objects + 1. Success criteria: NEXT_MODEL_PROMPT_v3 step 2.

Deviation: held-out scenes mostly have 6-7 layers and one 50-step Heun sample takes about 8 minutes,
so the full queue would exceed 40 hours. Every arm therefore uses 24 held-out scenes x seed 4401 x
25 Heun steps (50 forward passes); the quick evaluation is 6 scenes x seed 4401 x 25 steps. Arms are
comparable with each other; absolute numbers are not comparable with the earlier 50-step protocol.

## 3. Multi-scene supervision

Queue `run_20261009/run_queue.sh`: weighted arm 1000 updates -> quick evaluation on 6 held-out scenes
and a predeclared decision (`decide_continue.py`: continue to 2000 if the 500 or 1000 checkpoint beats
the parent by > 0.02 on object match or inferable completion) -> control arm with the same budget ->
final held-out evaluation of both -> regression on kitchen/bedroom/photo872/manga003. About 10 s per
update.

Frozen controls (24 held-out, seed 4401, 25-step Heun): original Qwen object match 0.070, visible
retention 0.886, inferable completion 0.842, unknowable plausibility 0.627; official adapter 0.001,
0.827, 0.821, 0.618; both have composite coverage 1.0. Both controls nearly always merge or drop
objects in these 3-7 object scenes.

Decision at 1000 updates (6 held-out scenes, seed 4401), parent -> 500 -> 1000: object match 0.0002
-> 0.478 -> 0.577; inferable completion 0.910 -> 0.974 -> 0.977; visible retention 0.843 -> 0.976 ->
0.971; unknowable plausibility 0.624 -> 0.775 -> 0.771; composite coverage 1.0 -> 1.0. Decision:
continue to 2000 (`decision_1000.json`). A 6-scene, single-seed stop-loss check, not a conclusion.

Final held-out (24 scenes, seed 4401, 25-step Heun):

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

Step-2 criteria: (a) object match >= 0.8 and above both controls: **not met** (best 0.619), though far
above the controls (0.070, 0.001). (c) inferable completion error down >= 50%: official 0.179 ->
weighted 0.042, -77%, met. (f) composite coverage >= 0.995: met (1.000). (e) visible error no more
than 5% above the controls: met (0.173 -> 0.025).

Conclusions: multi-scene supervision clearly teaches more complete, more precise layering and
inferable completion on unseen rendered scenes, without memorization. Occlusion weighting is not the
main cause (the unweighted control has higher object match; weighting gives a small, consistent
inferable-completion gain), so weighting cannot be stated as a finding. Test-only shapes are
sometimes replaced by training shapes. Object match below 0.8 makes step 3 partial by the
predeclared criteria; step 4's precondition is not met, so no RL.

Real room photos (12 OmniEdit/GEdit indoor photos, MIT, no truth, judged one by one):

- Weighted vs official: 6 clearly better, 3 slightly better, 1 similar, 2 with problems (twin-bed room
  fragmented; loft bedroom keeps the bed in the room layer). The most consistent gain is a complete,
  clean room layer (10/12).
- Weighted vs control: 6 better (control leaves sofas, wardrobes or paintings in the room layer, or
  merges furniture with rugs), 6 tied, 0 worse. Weighting matters more on real photos than on
  rendered scenes.
- Original Qwen is worst: room layer mostly blank or blurry, room split into holed fragments.
- The truth-free "no holes after hide" is 1.0 for every model and is not used as evidence.

## 4. Next round (2026-10-10)

- Real data: from the RORD test mirror (MIT, 6.4 GB), 31 indoor scenes -> 23 train (180 frames) / 8
  held-out (20 frames) two-layer samples (true background + foreground). New metric
  `background_recovery`: 1 = true background fully recovered, 0 = same as leaving the foreground in.
  Full RORD (about 200 GB compressed) was not downloaded (384 GB disk left); VDOR needs Hugging Face
  access approval; MuLAn (real photos, pseudo truth, CC BY-NC-SA) not downloaded.
- Random spare layers: 0-2 fully transparent extra layers per step (target = encoded transparent
  frame), aimed at fragmentation on real photos.
- v2 scenes: perspective camera, 16 furniture classes, wood/fabric textures, real photos as
  paintings, windows/doors/ceiling, indoor lights; 1000 train + 48 held-out.
- Arm E: v1 + RORD + spare layers, 1000 updates. Arm D: v2 + v1 + RORD + spare layers, 2000 updates.

### Arm E (v1 scenes + RORD real + spare empty layers)

Weighted, 1000 updates from the official adapter, v1 train (200) + RORD indoor train (180), 0-2 spare
layers per step. RORD held-out (20 real photos, 2 layers, 25-step Heun, seed 4401):

| Arm | Background recovery | Object match | Visible retention | Edit holes |
| --- | --- | --- | --- | --- |
| E | 0.479 | 0.398 | 0.882 | 0.952 |
| weighted 1000 (v1 only) | 0.455 | 0.226 | 0.798 | 0.943 |
| control 1000 (v1 only) | 0.312 | 0.169 | 0.770 | 0.924 |
| official | 0.153 | 0.215 | | |

Paired: E beats weighted 1000 on object match 18/20 and visible retention 19/20, on background
recovery only 11/20 (median +0.006). Real data mainly improves foreground separation; inpainting the
background under people is barely better than weighting alone.

Interruption: an unattended NVIDIA driver update at 06:17 (userspace 580.178 vs kernel module
580.173) broke CUDA (error 804). The queue resumed without a reboot by putting the 580.173.02
userspace libcuda on `LD_LIBRARY_PATH`.

v1 rendered held-out (24): E object match 0.296, inferable 0.929, visible 0.935, vs weighted 2000
0.562 / 0.958 / 0.975. On the 6 scenes that also have a 1000-update weighted checkpoint: E 0.361 vs
weighted 1000 0.577. E regressed on rendered scenes.

12 real rooms, E vs weighted 2000 side by side: E is worse on all 12. Object layers carry chunks of
floor/wall and the same furniture appears in several layers; in 8 rooms the room layer still holds
beds, sofas or tables. Non-empty layers 5.25 vs 5.83 (fewer layers from merging, not less
fragmentation).

Cause: the RORD foreground layer was the photo inside RORD's coarse polygon, i.e. the person plus
large patches of floor and wall; training on it taught "object layers include room chunks". The
training layers confirm the polygons visually.

Fix: `build_rord_matte.py` replaces the polygon with a difference matte (photo vs gt after a
per-channel gain/offset fit outside the polygon, threshold 0.07, open/close/hole-fill, 2 px
feather); 177 train / 19 held-out frames kept. `cache_rord_layers.py` gained `--occluder matte`
(background weighting under the matte) and `--foreground-weight` (0.5 for mattes; 0.2 for polygons).

### Arm D2

v2 (1000) + v1 (200) + `rord_layers_matte`, 0-2 spare layers, weighted, 2000 updates. RORD held-out is
still scored on `rord_layers` (same true background and polygon region, comparable with E). The
original arm D had started with polygon RORD; it was stopped at step 7 and its output removed.

RORD held-out: D2 has the best background recovery (0.526) and, scored against the matte truth, the
best object match (0.552). 12 real rooms, D2 vs weighted 2000: better 2, worse 6, tied 4.

Rendered held-out, per-object rescore (matched = share of objects with layer quality >= 0.5; product
= `layered_object_match`):

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

D2 loses to the v1-only control even on the v2 distribution it trained on. D2 outputs fewer non-empty
layers than the truth in 13/24 v2 scenes (control 2000 8/24, official 0/24): it merges or drops
objects. D2 and E share RORD (two-layer fg/bg) and the 0-2 spare layers; both are suspects, and every
arm on the four-GPU machine also uses spare layers. Next: a no-spare-layer arm (v2 + v1, no RORD).
No arm meets the RL gate (object match >= 0.8) on the v2 held-out.

## 5. Infinigen-Indoors scenes (2026-10-11)

`render_infinigen_layers.py` + `generate_infinigen_batch.sh` turn Infinigen-Indoors rooms (BSD,
commit a0d1a0dd) into the same layered format: a room layer with all furniture invisible to camera
and light (lamps still shine), one layer per furniture group with its own shadow on a shadow-catcher
room, depth ordering, 24 px inferable rule. Composite vs full render MAE 0.011-0.042 on 4 probe views.
Room generation is CPU only (45-80 min per dense room).
