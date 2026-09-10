# SD3.5 FP16 acceptance investigation

The existing FP16 role policy is a simpler candidate than the rejected selective
BF16 intervention. The pinned Miles SD3 OCR launch script explicitly selects FP16
for both trainer and rollout (revision `87d2aafc714959221ed6dee6f760296f94da4a0e`,
`scripts/run_diffusion_grpo_sd3_ocr_sglang.py`). This comparison does not claim to
reproduce its SGLang/FSDP implementation or H200 deterministic reference metrics.

## Compiled real-checkpoint diagnostic

The [raw report](sd3_5_fp16_allsteps_20260909.json) and
[exact source](sd3_5_fp16_allsteps_20260909_probe.txt) preserve a fresh real SD3.5
medium trajectory with initialized LoRA. The diagnostic explicitly overrides
`precision.training.dtype=fp16` and
`precision.rollout.prompt_encoders.dtype=bf16`. Rollout transformer precision
inherits FP16; all 486 trainable adapters remain FP32. The 908 base parameter
tensors are FP16. BF16 prompt encoder computation is retained, while the model
projects its conditioning tensors to the transformer's FP16 input dtype.

Compilation remains enabled through the production default compile method.
There is no selective conditioning wrapper, BF16 reduction override or Inductor
cast-emulation override. This is a different precision configuration, not evidence
that the original BF16 recipe now passes. The source loads the original preset
with compile disabled only to build the model, then compiles it before generating
the trajectory, so both generation and replay use the compiled transformer.

The configured seed-17 sampler selects prompt index 1971. It generates 16 samples
at 512 pixels, guidance 4.5, noise 0.7 and ten scheduled steps, then compares the
nine trainable transitions against batch-1 replay with and without backward.
All 18 original rollout/replay comparisons are finite and below 0.01:

| Replay mode | Maximum original rollout/replay logprob difference |
|---|---:|
| No-grad | 0.000399172306060791 |
| With backward | 0.0005183592438697815 |

All nine backward arms observe finite existing gradients. Backward here is
unscaled and no optimizer step occurs; it does not establish absence of gradient
underflow or training stability. The production trainer already creates a CUDA
GradScaler for FP16 autocast, and that path needs actual training acceptance.
No reward, VAE decode, independent replay model, weight transport or trainer
parity gate runs in this diagnostic. One prompt is not a complete recipe test.

The probe started from clean `593d08c3f`. Subsequent `3a0874b9b` documentation and
`dbe5ad2ca` evidence-capture changes did not alter model execution. It used the
repository virtualenv, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, `PYTHONHASHSEED=17` and
`PYTHONPATH=.`. Local log: `/tmp/vrl_sd3_fp16_allsteps_probe.log`. This standalone
report is not an environment-bound training completion receipt.

## Next acceptance and preserved scope

Run the original OCR recipe for two complete epochs with the two explicit
precision overrides, original batch sizes, real configured reward/data and the
unchanged 0.01 parity threshold. Keep deterministic seed-17 settings and collect
the existing launch record, full-precision metrics, verdict and artifact receipt.
The run must exercise independent rollout/replay models, production gradient
scaling and actual optimizer updates. Only a successful run can justify a second
identical run for strict metric comparison. Neither is complete at this commit.

No runtime implementation or preset changed. The existing role precision policy,
model forward autocast boundary and GradScaler already express this experiment;
no new conditioning class, wrapper, precision flag or family table is necessary.
The original BF16 failure and rejected interventions remain recorded in the
[preceding investigation](sd3_5_execution_precision_20260909.md).
