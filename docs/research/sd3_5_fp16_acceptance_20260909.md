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

## Real OCR attempt A: shared-device capacity failure

The actual two-epoch attempt ran from clean `5bfbebe7b` with:

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED=17 \
  .venv/bin/python -m vrl.scripts.supervise \
  --config experiment/sd3_5/online_grpo_ocr --max-attempts 1 \
  trainer.total_epochs=2 trainer.seed=17 trainer.deterministic=true sampling.seed=17 \
  precision.training.dtype=fp16 precision.rollout.prompt_encoders.dtype=bf16 \
  trainer.output_dir=outputs/repro/sd3_5_ocr_fp16_a
```

Launch `f8208c33bcc5442bb97ccbdb6852e34f` records the explicit precision settings
and a clean checkout. The first generation batch exhausted memory while another
Anima training process occupied 12.13 GiB on the shared 32 GiB RTX 5090. Existing
OOM recovery split 16 samples into two batches of 8, completed those generations,
and subsequently generated the next group at batch 16. Real OCR scoring loaded
the configured cached Paddle models. CuMem parking passed with a 660 MiB worker
residual against the 498 MiB baseline.

A later generation failed even after existing recovery reached a single sample.
The error reports 17.15 GiB in the SD3 worker, 12.34 GiB in the other trainer,
626 MiB in its generation worker and 854 MiB in the SD3 trainer. Only 189.94 MiB
was free when another 256 MiB allocation was requested. This is actual shared-card
capacity exhaustion; no numerical parity conclusion follows from it.

The supervisor exited 1 after its single allowed attempt at 22:06:48 Pacific.
The [failed verdict](sd3_5_fp16_attempt_a_20260909_verdict.json) is preserved.
Both metrics files contain only headers; no successful final artifact receipt or
completed epoch is present. Local log: `/tmp/vrl-sd3-ocr-fp16-a.log`. After exit,
NVML listed only the pre-existing Anima processes, confirming the SD3 processes
released their GPU allocations. No other workload was stopped or modified.

The source batch configuration, reward and 0.01 threshold were not changed.
Runtime OOM splitting is nevertheless a real execution change, so this attempt
would not establish a fixed-batch deterministic baseline even if it had finished.
The next full training attempt needs a stable capacity window; repeatedly
relaunching alongside the same independently waking model would not supply that
evidence. FP16 remains a promising diagnostic result, not an accepted recipe.

## Real OCR attempt B completes two epochs

The same command was run with output directory
`outputs/repro/sd3_5_ocr_fp16_b` from clean `51b4e2cfe`, after the GPU became
available. It completed two epochs and exited zero at 22:34:42 Pacific.
`verify_run_completion` verified the actual launch, full-precision metrics,
checkpoint artifact hashes and matching supervised success verdict. Launch ID:
`6e83c2d569554c46a778ef50cae10a53`.

| Epoch | Loss | OCR reward mean | Gradient norm | Maximum logprob difference |
|---|---:|---:|---:|---:|
| 0 | 0.000020266533182520005 | 0.26027169823646545 | 0.002486748620867729 | 0.001454971730709076 |
| 1 | 0.000048537592455330956 | 0.35471808165311813 | 0.0021115136332809925 | 0.0019754618406295776 |

The persisted first-update gate passes with the unchanged 0.01 threshold.
The final checkpoint records completed epoch 2, trainer step 2 and global step 2.
Loading it through `load_training_checkpoint` finds 486 optimizer state entries,
all with Adam step 2, plus GradScaler scale 65536 and growth tracker 2. Thus the
checkpoint demonstrates two actual optimizer updates, not just two loop counters
or two scaler-skipped attempts. EMA had not reached its configured update
interval, so the export correctly uses raw checkpoint-owned weights.

The [full-precision metric rows](sd3_5_fp16_run_b_20260909.csv) and
[receipt/verdict/checkpoint observations](sd3_5_fp16_run_b_20260909.json) are
archived. The copied receipt's relative artifact paths refer to the original
output directory; this compact documentation copy is not a standalone replacement
for the checkpoint and launch files. Local log: `/tmp/vrl-sd3-ocr-fp16-b.log`.

The run is real short-training acceptance, not deterministic or learning-quality
acceptance. A transient external process later occupied 12.79 GiB and triggered
one 16-to-8-plus-8 OOM split at 22:26:57. The rest of training completed, but this
dynamic generation shape disqualifies the run as a fixed-batch reference. The
initial status update saying all first four groups used batch 16 was incorrect;
the complete log establishes this split. A two-point reward increase is not a
learning curve or held-out improvement claim.

This supports adopting the already-existing FP16 role policy for the single-GPU
OCR recipe while retaining BF16 prompt encoders. It does not justify changing
unverified FSDP or continuous descendants implicitly. Fixed-batch repeated runs,
held-out evaluation and longer curves remain outstanding.
