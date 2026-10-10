# Reward card: siglip-v2_5-qwen21-t2i-512-sde

- kind: learned
- axis: `aesthetic`
- ready for a training key: **no** (not run: spread)
- blind spots: Primary paired preference gate failed: utility 0.5703, 95% CI 0.4609–0.6797. See blind_verdict.json; pooled AUC is only supplementary.; Composition/counting subset: 0.1875 preference utility and -0.75 adherence points (n=8; exploratory).; Continuous spread measured in continuous_spread.json; no calibrated binary success threshold, so toolkit spread gate remains unqualified.

## 1. Repeatability -- PASS
max score range 0.0 (tolerance 1e-06) over 512 samples x 2 repeats

## 2. Agreement with labels -- FAIL
- pair_preferred_vs_rejected_pooled_diagnostic: AUC 0.523 [0.410, 0.628] (55 vs 55) -- FAIL

## 3. Shortcuts and damage -- FAIL
- black_rgb: scored at or above the genuine output on 11% of 64 sources -- FAIL
- blur_rgb: scored at or above the genuine output on 38% of 64 sources -- FAIL
- checkerboard_rgb: scored at or above the genuine output on 6% of 64 sources -- PASS
- noise_rgb: scored at or above the genuine output on 12% of 64 sources -- FAIL

## 4. Spread under the training sampler -- not run

## 5. After training
not yet trained on
