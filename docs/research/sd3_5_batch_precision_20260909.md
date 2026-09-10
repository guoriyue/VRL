# SD3.5 batch-size and precision isolation

A real-checkpoint, eager forward experiment on the shared RTX 5090 found that
changing the denoising batch from 16 to 1 can exceed the existing 0.01 logprob
parity limit under BF16. This is evidence for a numerical investigation, not a
complete reproduction or repair of the compiled training failure.

The experiment completed on September 9, 2026 Pacific, with repository revision
`30b75ae84`. The [raw measurements](sd3_5_batch_precision_20260909.json) and
[exact executed diagnostic source](sd3_5_batch_precision_20260909_probe.txt) are
preserved. The source is an archived one-off experiment, not a supported CLI.
It ran with the repository virtualenv using:

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED=17 PYTHONPATH=. \
  .venv/bin/python /tmp/vrl_sd3_batch_precision_probe.py \
  > /tmp/vrl_sd3_batch_precision_probe.log 2>&1
```

## Controlled comparison

The script loads `experiment/sd3_5/online_grpo_ocr`, initializes the trainer RNG
with seed 17 and deterministic mode, and explicitly disables compilation. It
selects prompt index 1971 through the configured sampler, encodes that one prompt
and repeats its embeddings 16 times. It uses the actual model with initialized
LoRA, 512-pixel sampling, guidance 4.5, 128-token conditioning, ten scheduled steps,
and flow-GRPO noise level 0.7. Those numerical settings agree with the failed
training attempt's resolved config. No training checkpoint is restored.

It records the first nine denoising transitions under the initial BF16 policy,
then reuses each step's latents, actions, timesteps and conditioning in three
precision arms. Within each arm it compares a batch-16 forward with the
concatenation of sixteen batch-1 forwards. Both use the model's replay state
restoration and production SDE log-density function. All calls run under no-grad;
text encoders and VAE move to CPU after conditioning is computed. No media decode,
OCR scoring, backward, optimizer update or Ray transport is performed.

| Transformer arm | Maximum batch-16 vs batch-1 logprob difference | Maximum original BF16 trajectory vs batch-1 difference |
|---|---:|---:|
| BF16, TF32 enabled | 0.018531210720539093 | 0.018531210720539093 |
| BF16, IEEE FP32 operations | 0.018531210720539093 | 0.018531210720539093 |
| FP32, IEEE, no outer autocast | 0.000000476837158203125 | 0.011572189629077911 |

All maxima occur at step index 8, the last recorded transition. For both BF16
arms the batch logprob difference grows from 0.00045955 at step 0 to 0.00503470
at step 7, then exceeds 0.01 at step 8. The two BF16 arms have identical recorded
results: disabling TF32 alone did not resolve this experiment's batch sensitivity.
The FP32 arm substantially reduces its own batch sensitivity but still differs
from the original BF16 rollout by more than 0.01. Switching only replay to FP32
therefore does not establish parity with existing BF16 trajectories.

## Limits and next decision

This experiment does not identify a particular sensitive parameter or kernel.
Whole-transformer dtype conversion also casts adapter tensors, whereas the actual
training source maintains FP32 trainable adapters. It does not reconstruct
checkpoint values more accurately when promoting an already BF16-loaded model.
The FP32 arm reuses BF16-generated actions and BF16 conditioning, so it is not a
fresh all-FP32 training or generation comparison. One prompt and nine steps are
insufficient to claim recipe-wide stability or exact deterministic repeatability.
The raw JSON has no independent environment/artifact binding and is not a training
completion receipt.

The failed production attempt used compilation and grad-enabled replay and
reported 0.02667667716741562 at its first-update parity gate. This eager no-grad
experiment shows a possible source of that class of failure; it does not isolate
all contributions to that particular failure. The next investigation must retain
actual adapter dtypes, compare production rollout/replay paths and inspect
sensitive operations before choosing a precision change. Matching the full rollout
batch in backward also needs a memory measurement, not an assumed free fix.

No production code, recipe, batch setting, precision policy or threshold changed.
The existing family forward/replay interface and shared SDE math remain necessary
boundaries. This evidence adds no contract dataclass, wrapper or algorithm table.
The original training acceptance and complete curves remain outstanding.
