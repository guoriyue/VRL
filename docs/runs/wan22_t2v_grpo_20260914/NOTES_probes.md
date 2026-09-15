# Probe results (2026-09-14, base Wan 2.2 T2V-A14B, 320x320x17, no LoRA)

## Generation-setting probe (quality_probe/, 2 prompts, seed-fixed)
- 10 steps (ODE or CPS-SDE): frame 0 is mosaic garbage, frames 1-3 ghosted, clean by ~frame 8;
  the pencil prompt is blocky in every frame. Lossless PNG == mp4 (abs diff 1.6-3.8), so it
  is the model/VAE output, not the codec.
- No VAE tiling/slicing: identical to base -> not the cause. 480x480 @10 steps: still bad.
  Official negative prompt @10 steps: still bad.
- 20 steps: coherent scenes, no full-frame mosaic; first-frame residual speckle remains on
  roughly half of the 32 train_small base clips (frame0->1 abs diff 11-22 vs ~2-9 later).
  40 steps: fully clean first frame, but 2x the 20-step cost -> not affordable for training.
- Decision: train and evaluate at 20 steps (window_range [0,20], 19 replay steps).

## Kling VideoReward probe (eval_train_small/reward_probe.json, 32 base clips @20 steps)
- Repeat scoring: exact (max abs diff 0.0).
- visual_quality: frozen clip lower in 78%, noise s=20/60 lower in 78%/84% -> correctly signed.
  Frame-shuffle RAISES it in 91% (the corrupted first frame moves away from position 0).
- motion_quality: noise LOWERS it in only 9-25% (noise raises MQ), frozen lower in 22% ->
  inverted / not measuring motion at 17 frames. Same defect as the 480p_33f Cosmos finding.
- text_alignment: frozen/noise lower in 94-97%, correctly signed; largest spread (std 0.71).
- overall_reward = VQ+MQ+TA still falls under noise/frozen, but carries the inverted MQ term.
- Spread across base clips: VQ std 0.24 (within-prompt seed diff 0.15), TA 0.71, overall 0.84.
- Decision: train on score_key=visual_quality; TA/MQ/overall recorded as diagnostics.
