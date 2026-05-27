# SPRINT: Anatomy Plausibility Reward (v2)

Status: **proposal — awaiting user decision on rollout shape**

## Revision log

**Draft 2** (this version) — pivot on user feedback: "实际上画面中不一定
永远都能看得见五个手指，比如手会有遮挡、手会有动作". The initial draft
assumed 5 fingers always visible and treated "fewer than 5" as a defect.
That punishes legitimate poses: fists, V-signs, points, occluded
fingers, back-of-hand views. Three structural changes from draft 1:

1. **Top-level principle added**: *visibility-gated, occlusion never
   penalizes.* Each subscore (body / arms / legs / hand geom / hand
   morph) runs a visibility gate first and only contributes when
   keypoints clear confidence + on-silhouette checks. The top-level
   formula is now a geometric mean **over visible subscores only**, not
   a fixed 5-term product.

2. **Per-finger visibility gate in `S_hands_geom`**: each of the 5
   fingers is evaluated independently. Fingers that fail the gate
   (low keypoint confidence, collapsed phalanx chain, or keypoints off
   the silhouette) are simply not scored — they are not penalty
   contributions. Cross-finger checks (finger spread, thumb opposition)
   only apply between pairs of fingers that both pass the gate.

3. **`S_hands_morph` pivots to asymmetric tip-count penalty**: instead
   of "expect 5 defects in the silhouette", we infer the expected
   visible-tip count `N_kp` from how many fingers passed the
   `S_hands_geom` visibility gate. Then:
   - `N_silhouette > N_kp + 1` → **strong penalty** (this is the
     hallucinated-extra-finger / fused-blob case — the actual user
     complaint).
   - `N_silhouette < N_kp − 1` → **soft penalty** (ambiguous —
     occlusion vs fusion — can't distinguish without more signal).
   - Within ±1 → no penalty.

   Asymmetry is the key design choice: we bias to catch hallucinations,
   not to catch occlusions.

Two new open questions added (visibility confidence threshold,
asymmetric penalty weights).

**Draft 1** — initial design. Replaced existence-driven additive
penalties with geometric-mean plausibility subscores and OpenCV
silhouette analysis on hand crops. See git history for the diff.

---


## Why we're doing this

The current `anime_anatomy_structure` reward
(`vrl/rewards/models/pose/structure.py`) is dominated by **existence
penalties**: 0.35 weight on "missing body keypoints" and 0.30 on
"missing hands", versus only 0.40 combined on geometric plausibility
(impossible angles + limb asymmetry). The eval8 baseline-vs-RL report
(`outputs/anima_anatomy_reports/baseline_vs_rl_eval8.md`) shows the
signal is discretized to {0.0, 0.3, 0.4} and RL training produced
**zero delta** — there is no gradient information for the model to act
on.

User feedback (paraphrased):

- "The anime anatomy reward is bad. I want human-eye reasonable limb /
  body / pose / fingers, not counting if the point exists or not."
- DWPose detection failure is rare because prompts force the subject
  into frame; misaligned, ugly, anatomically wrong keypoints are the
  real failure mode.
- Hand evaluation should combine **keypoint geometry** with
  **morphology** (silhouette analysis on the original-image hand crop).

The clearest example of the disease: `_visible_hand_count` /
`_collapsed_hand_fraction` only check `_point_spread` — the bounding
box of all 21 hand keypoints lumped together. The 21-point hand layout
(wrist + 4 joints × 5 fingers, thumb → pinky) is completely unused.
We have rich per-finger structure we are throwing away.

## Goal

Replace the additive existence-driven score with a **plausibility-based
score** that responds smoothly to how anatomically reasonable the
detected pose looks, so RL gets a useful gradient and the eval set
spreads across the [0, 1] range instead of clumping at three values.

## Top-level principle: visibility-gated, occlusion never penalizes

A hand can be clenched (no fingers visible), a finger can be hidden
behind another, a back view hides arms, a sitting pose hides legs.
**None of these are anatomy bugs.** The reward must distinguish:

- **Visible and wrong** → penalize (this is the gradient we want).
- **Not visible** → no signal, drop from the score (don't impute, don't
  neutral-default).
- **Visible and right** → reward.

Every subscore — body, arms, legs, hand geometry, hand morphology —
runs a visibility gate first and only contributes if the relevant
keypoints clear a confidence + on-silhouette check.

## New scoring formula

```
final_score = geometric_mean(S_i for S_i in subscores if visible_i)
```

The set of subscores is `{S_body, S_arms, S_legs, S_hands_geom,
S_hands_morph}` and each subscore is itself a geometric mean over its
**visible** sub-components (per finger, per limb side, per body
landmark). A hand whose fingers are all occluded contributes nothing to
the total; a back view that hides arms is scored on body + legs only.

Each subscore is in [0, 1] and **soft / continuous** (no hard
thresholds). Geometric mean means any one badly-broken visible
subdimension drags the total down — matches the "ugly anywhere = ugly"
intuition and avoids the current bug where high body coverage masks
broken visible fingers.

If **zero** subscores are visible (e.g., DWPose returns nothing), the
reward returns 0 with `abstain=True` in diagnostics so the trainer can
choose to drop or downweight it. This is the only failure-mode fallback.

## Subscore designs

All subscores use the form `s = exp(-((v - expected)/tolerance)**2)`
where appropriate, or `1 - sigmoid(...)` — never a step function.

### `S_body` — torso / head plausibility

- **Spine straightness**: angle at neck (point 1) between head (0) →
  neck → mid-hip (mean of 8 and 11). Peaks within 15° of 180°.
- **Shoulder–hip parallelism**: shoulder line (2–5) ∥ hip line (8–11).
- **Head-neck alignment**: head (0) roughly above neck (1) in image-y,
  offset normalized by body scale.
- **Torso aspect ratio**: shoulder-width / torso-length ∈ [0.4, 0.9]
  (real human range).

### `S_arms` — geometric mean over **visible** arms

Per-arm visibility gate: shoulder, elbow, wrist all have confidence ≥
threshold. A back view or single-arm crop simply skips the hidden
arm — it does not contribute and does not penalize. Both arms hidden
→ `S_arms` drops out of the top-level mean.

For each visible shoulder → elbow → wrist chain:

- **Elbow flexion** in 0–170° physiological range. Peaks at ~150°
  (slight bend); 0° straightened is mildly penalized, hyperextension
  past 180° is harder penalized. Anime has lots of straight arms, so
  we should not punish them.
- **Forearm / upper-arm length ratio** ~ 0.85–1.05 (soft Gaussian
  around 0.95).
- **Wrist–hand continuity**: distance from body wrist keypoint (4 / 7)
  to the hand's wrist keypoint (index 0 of the 21-point hand)
  normalized by body scale. The "is the hand attached" check that's
  currently missing.

### `S_legs` — symmetric to arms

Same per-leg visibility gate (hip + knee + ankle). Sitting / cropped /
occluded legs drop out without penalty. Both legs hidden → `S_legs`
drops out of the top-level mean.

For each visible chain:
- Knee flexion in 0–135° (knees bend one way; hyperextension
  penalized harder than arms).
- Thigh / shin ratio ~ 1.0.
- Hip-knee-ankle chain plausibility.

### `S_hands_geom` — per-finger geometry (the big one)

Per-finger visibility gate first, then per-finger scoring. For each
hand's 21 keypoints (thumb 1–4, index 5–8, middle 9–12, ring 13–16,
pinky 17–20):

**Per-finger visibility** (each finger evaluated independently):
- All 4 phalanx keypoints have confidence ≥ `min_keypoint_confidence`.
- The 4 keypoints do not collapse onto each other (chain length
  > 0.5% of body scale — filters DWPose's "all four points at
  wrist" degenerate output).
- (When silhouette mask is available) at least 3 of the 4 phalanx
  keypoints lie inside the hand mask.

A finger that fails the gate **contributes no score** — it is excluded
from the per-finger geometric mean. This handles fists, clenched
fingers, occluded fingers, gesture poses (V-sign, point) cleanly.
Importantly, **failing the gate is not itself a penalty** — the
penalty for hallucinated fingers comes from `S_hands_morph` (silhouette
shows more tips than visible keypoints).

**Per-visible-finger scoring**:
- **Phalanx length monotonicity**: proximal ≥ middle ≥ distal.
- **Inter-knuckle bend direction**: DIP and PIP joints bend toward the
  palm consistently with finger curl.
- **Finger-length-to-palm-length** ratio in ~0.7–1.0.

**Cross-finger checks** (only between pairs of fingers that both pass
visibility):
- **Finger spread**: angles between adjacent visible proximal
  phalanxes in 0–60° range.
- **Thumb opposition**: thumb base (joint 1) on the radial side of the
  palm — only scored if thumb + at least one other finger visible.

A hand contributes to `S_hands_geom` iff ≥ 1 finger passes the
visibility gate. `S_hands_geom` is the geometric mean across the
contributing hands (0, 1, or 2). Both hands fully occluded → hand
geometry subscore is dropped from the top-level geometric mean.

### `S_hands_morph` — silhouette / morphology on the original crop

**Trigger only when `S_hands_geom < 0.85` for that hand**, to bound
per-image cost.

Pipeline (OpenCV only — no new deps needed; `scikit-image`,
`segment-anything`, `mediapipe` are all absent from `pyproject.toml`):

1. **Crop** the hand using 21 keypoints bbox, padded 25%.
2. **Skin / hand mask**: HSV skin range + adaptive Otsu on V. Take
   largest connected component.
3. **Hand-area sanity**: mask area / crop area ∈ [0.20, 0.85]; else
   morph subscore is **dropped** (not neutral) and that hand
   contributes only `S_hands_geom`.
4. **Convex hull + defects** (`cv2.convexHull` + `cv2.convexityDefects`):
   count "deep" defects (depth > 0.05 × crop diagonal).
5. **Pose-aware expected fingertip count** — this is the key change.
   Do **not** assume 5. Infer the expected visible-tip count `N_kp`
   from how many fingers passed the `S_hands_geom` visibility gate
   (range 0–5). Then derive expected silhouette signatures:
   - `N_kp ≥ 4` (open hand) → expect 3–4 deep defects.
   - `N_kp = 2–3` (partial / gesture) → expect 1–2 deep defects.
   - `N_kp ≤ 1` (fist / heavy occlusion) → silhouette analysis is
     uninformative; drop this hand's morph subscore entirely.
6. **Asymmetric penalty on tip count**:
   - `N_silhouette > N_kp + 1` → **strong penalty** (silhouette
     reveals fingertip features the keypoints didn't claim — this is
     the hallucinated-extra-finger / fused-blob case, the failure
     mode we actually want RL to fix).
   - `N_silhouette < N_kp − 1` → **soft penalty** that scales with
     keypoint confidence — if keypoints say "five fingers visible"
     but the silhouette shows two, either keypoints are wrong or
     fingers are fused; we can't tell which, so penalize gently and
     log to diagnostics.
   - Within ±1 of `N_kp` → no penalty.
7. **Keypoint-on-skin**: each of the 21 hand keypoints that passed
   visibility gating must lie inside the mask. Per-finger fraction of
   on-skin keypoints feeds back into the per-finger visibility gate
   (chicken-and-egg: first pass uses keypoint confidence only, second
   pass incorporates the mask).

Combine the defect-count score and keypoint-on-skin fraction by
geometric mean. Asymmetric handling means we are biased to **catch
hallucinated fingers** (the failure mode driving the user complaint)
without punishing legitimate occlusion and gestures.

## Reuse existing utilities

These live in `vrl/rewards/models/pose/` and stay as-is:

- `geometry.py`: `_people_from_result`, `_person_confidence`,
  `_body_scale`, `_distance`, `_angle_degrees`, `_PersonPose`,
  `_Keypoint`, `_BODY_SCALE_SEGMENTS`, `_ARM_CHAINS`, `_LEG_CHAINS`,
  `_PAIRED_LIMB_SEGMENTS`.
- `hints.py`: prompt hint parsing (repurposed for abstain fallback,
  not penalty gating).
- `structure.py:_extract_images`.
- `dwpose.py`: backbone untouched.

Deprecated (kept for v1 fallback only):
`_MISSING_KEYPOINT_PENALTY`, `_HAND_MISSING_PENALTY`,
`_COLLAPSED_HAND_PENALTY` and the additive branch of
`_score_pose_result`.

## Files to modify

| File | Change |
|---|---|
| `vrl/rewards/models/pose/plausibility.py` | **NEW.** Five subscore functions + `score_pose_plausibility(person, image_pil, body_scale) → (score, diagnostic)`. |
| `vrl/rewards/models/pose/hand_silhouette.py` | **NEW.** OpenCV hand silhouette pipeline (crop → mask → defects → keypoint-on-skin). |
| `vrl/rewards/models/pose/structure.py` | Add `scoring_mode: Literal["count", "plausibility"] = "plausibility"`. Route `_score_pose_result` to legacy or new path. Diagnostic schema becomes a union with `mode` field. |
| `vrl/rewards/functions/anime_anatomy.py` | Forward `scoring_mode` kwarg. Registry key `anime_anatomy_structure` stays unchanged so existing experiment YAMLs keep working. |
| `configs/reward/anime_anatomy_structure.yaml` | Default `scoring_mode: plausibility`. Old runs pin `count` for reproducibility. |
| `tests/rewards/test_anime_anatomy.py` | New `TestPlausibilityMode` class with fixtures: well-formed-finger pose, broken-finger pose (reversed phalanx lengths), detached-hand pose, **fist pose (all fingers fail visibility gate)**, **partial visibility (only thumb + index above confidence — V-sign)**, **back view (arms occluded)**. Synthetic skin-color PIL image to exercise the silhouette branch deterministically. Existing `TestCountMode` tests pinned with `scoring_mode="count"`. |
| `vrl/scripts/diffusion/cosmos/anima/mine_anatomy_prompts.py` | No code change; verify the new diagnostic keys don't break JSON serialization. |

## Rollout — decision needed

**Recommended: single registry key, mode flag, default flipped to
`plausibility`.**

- One source of truth — avoids two reward variants drifting apart.
- Existing experiment configs continue to load (behavior changes,
  not the wiring) — we don't have to touch every YAML.
- Old GRPO checkpoints stay reproducible via `scoring_mode: count`
  in their config.
- The `last_components` key the online trainer reads
  (`vrl/scripts/common/online.py:168`) keeps its name, so logs and
  comparison metrics keep flowing.
- Tradeoff: diagnostic schema changes shape between modes. We expose
  `mode` as a top-level field so downstream consumers can dispatch.

**Alternative if clean break wanted**: register sibling key
`anime_anatomy_plausibility`, leave the old one frozen. Costs two
parallel `RewardFunction` classes and parallel config family forever,
and we'd need to update the experiment YAML.

## Verification (before declaring done)

1. **Unit tests**: `pytest tests/rewards/test_anime_anatomy.py -v`.
   New cases must cover:
   - clean pose → score > 0.85;
   - broken-finger pose (reversed phalanxes) → < 0.5;
   - detached-hand pose → < 0.6;
   - **fist pose → score similar to clean pose** (occlusion is not a bug);
   - **V-sign (2 fingers visible, 3 occluded) → score reflects only the 2 visible fingers**;
   - **back-view pose (arms hidden) → score uses body + legs only, no arm penalty**;
   - **6-finger hallucination (silhouette shows extra tip not in keypoints) → score < 0.4**;
   - both modes still produce valid scores.
2. **Eval8 regression**: rerun the eval that produced
   `outputs/anima_anatomy_reports/baseline_vs_rl_eval8.md` with the
   new reward. Confirm:
   - Score distribution is **not** clumped at {0.0, 0.3, 0.4} —
     meaningful spread across [0, 1].
   - Baseline mean qualitatively tracks human judgment on the eight
     images (manual spot check of hands and limbs).
3. **Silhouette sanity**: 5 hand-only crops (2 clean, 1 fused-finger,
   1 extra-finger, 1 detector-misaligned). Confirm `S_hands_morph`
   rank-orders them correctly.
4. **No new package install**: `pip install -e .[pose]` resolves the
   same dependency list (only `opencv-python` is needed for silhouette
   work, already present).

## Open questions for the user

1. Rollout shape: recommended single-key + mode flag vs sibling key?
2. Is the `[0.20, 0.85]` mask-area sanity window calibrated for our
   anime art style, or should we widen it on a sample first?
3. For the geometric mean: should we floor each subscore at e.g. 0.05
   so a single broken subdimension doesn't crash the total to ~0?
4. **Visibility threshold**: per-finger gate uses
   `min_keypoint_confidence` (default 0.25). DWPose on anime is noisier
   than on real photos — should we raise this for the plausibility
   path (e.g. 0.40) so partially-detected fingers don't sneak in and
   get scored on garbage coordinates? Recommend collecting a small
   labeled set of (anime image, ground-truth-visible-finger-mask)
   pairs to calibrate before locking in a number.
5. **Asymmetric tip-count penalty weights**: how aggressive should the
   hallucinated-extra-finger penalty be vs the fused-finger soft
   penalty? Reasonable starting point: hallucination drops `S_hands_morph`
   to ≤ 0.3; fusion to ~0.6. Need to tune on examples.
