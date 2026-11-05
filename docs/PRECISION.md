# Precision configuration

VRL separates each role's ordinary dtype from selective low-precision kernels.
The complete public shape is:

```yaml
precision:
  float32_precision: tf32

  training:
    dtype: bf16
    outer_autocast: true

  rollout:
    dtype: bf16
    outer_autocast: true
    quantization:
      format: fp8
    prompt_encoders:
      dtype: fp16
```

`precision.float32_precision` and `precision.training.dtype` are required.
`float32_precision` accepts `ieee` or `tf32`. `rollout.dtype` inherits the
training dtype, `training.outer_autocast` defaults to true, and
`rollout.outer_autocast` inherits the training value.
`rollout.prompt_encoders.dtype` inherits the resolved rollout dtype. Set prompt
encoders to FP16 explicitly when that memory/accuracy trade-off is desired;
quantization never changes their dtype implicitly.

## Role precision and the model forward boundary

`training.dtype` is the base dtype for the trainer/replay model.
`rollout.dtype` is the base dtype for the generation policy. A base dtype governs
parameter loading and the dtype used by the model's shared execution boundary.
After config resolution, each process receives one `RolePrecision` containing
that role's `dtype`, `float32_precision`, `outer_autocast`, and optional
`quantization`. This is the single precision object used by model loading,
backend setup, the shared diffusion forward boundary, quantization, and
diagnostics; rollout build options carry only generation-specific inputs such as
the prompt-encoder dtype and weight-sync lifecycle.

`outer_autocast` controls the shared transformer AMP scope for a role. It is a
public execution choice rather than a hidden family capability. Selecting false
keeps the model's native parameter execution while still allowing protected
operations inside the model to own their local precision. FP32 roles make the
shared autocast scope a no-op regardless of this setting.

SANA's canonical preset explicitly selects FP16 parameters, IEEE FP32 matmuls,
and `outer_autocast: false` for both roles. These are checkpoint-validated preset
values, not permanent restrictions attached to every model registered as SANA;
an explicit config override is accepted as an experiment choice.

An aligned BF16 run only needs:

```yaml
precision:
  float32_precision: tf32
  training:
    dtype: bf16
```

The allowed base dtypes are `fp32`, `bf16`, and `fp16`. FP8 and NVFP4 are not
ordinary model or autocast dtypes and are rejected in a `dtype` field.

## FP32 matmul backend

```yaml
precision:
  float32_precision: tf32
  training:
    dtype: bf16
```

`float32_precision` controls FP32 matrix multiplication kernels in every trainer
and rollout process. `tf32` permits TensorFloat-32 acceleration; `ieee` requires
strict IEEE FP32 behavior. This axis is independent of parameter storage and the
role's outer autocast setting: FP16/BF16 models may still contain FP32
attention or protected sub-operations whose matmuls consume this policy.

The setting is explicit because process defaults can differ. SANA's canonical
preset selects `ieee` because its linear-attention processor promotes Q/K/V to
FP32 and is numerically sensitive to TF32.

The same call also turns off cuBLAS reduced-precision reductions for FP16 and
BF16 GEMMs (`torch.backends.cuda.matmul.allow_{fp16,bf16}_reduced_precision_reduction`,
both default on in PyTorch). With them on, cuBLAS chooses a split-K kernel per
GEMM shape and may sum the partial products in the half dtype, so a sample's
output depends on the batch it is computed in, and a rollout batch replayed at a
different batch size drifts. Measured with the rollout/replay divergence probe
(now `vrl.scripts.generation.full_sequence_probe`; its reductions-on switch was
removed once this table was recorded) on an RTX 5090 (torch 2.11), rollout
batch 4 replayed as batch 1:

| Family (dtype) | reductions on | reductions off |
| --- | --- | --- |
| SD3.5 Medium (BF16) | max \|Δlogprob\| 1.1e-1, first divergence `context_embedder` | 3.7e-9 |
| SANA 1600M (FP16) | 1.5e-3 | 7.5e-4; the remainder comes from cuBLAS and cuDNN kernel choice for the LoRA projections and the 1x1 `conv_point` |

This covers the base model's GEMMs only. A LoRA adapter that has left its
zero-initialized state adds rank-sized GEMMs whose cuBLAS kernel still depends
on the batch: with the probe's `--lora-b-std 0.01`, SD3.5 rollout batch 8
replayed as batch 1 reaches max |Δlogprob| 5.1e-2 (first divergence `to_q.lora_A`), batch 8 replayed
as batch 8 is bitwise, and `batch_invariant_ops` brings batch 8→1 to 4.8e-7.
LoRA runs therefore keep the generation batch equal to the replay batch.

Attention backends are a separate source. Each SDPA backend is batch-invariant
on its own, but flash, efficient and cuDNN attention reduce in different orders:
on SD3.5 a flash rollout replayed with cuDNN or efficient attention reaches
max |Δlogprob| 8.8e-2 on the last step. Ring context parallelism forces
`_native_cudnn` on the trainer while the rollout lets SDPA pick, so such
recipes cannot be held to bitwise replay parity.

Denoise step time did not change: SD3.5 Medium 131 ms/step (batch 4, 512px) and
309-310 ms/step (batch 2, 1024px); SANA 1600M 225 ms/step (batch 4, 1024px).

## Selective rollout quantization

Quantization is layered on `rollout.dtype`; it does not replace that dtype.
Normalization, residuals, excluded or unsupported Linears, and quantized GEMM
inputs/outputs still need the rollout base dtype.

Quantization is rollout-only. The trainer has no autograd-capable quantized
Linear, so `precision.training` carries no `quantization` block; one is rejected
as an unknown key rather than accepted as a silent no-op.

### FP8

```yaml
precision:
  float32_precision: tf32
  training:
    dtype: bf16
  rollout:
    dtype: bf16
    quantization:
      format: fp8
```

FP8 replaces eligible attention projections and MLP Linears with row-wise
scaled `_scaled_mm` GEMMs. There is no `recipe` key: row-wise scaling is the
only variant a run has ever used, so the policy fixes it.

### NVFP4

```yaml
precision:
  float32_precision: tf32
  training:
    dtype: bf16
  rollout:
    dtype: bf16
    quantization:
      format: nvfp4
```

NVFP4 is an experimental Blackwell-only rollout path. It conservatively replaces
eligible MLP Linears; attention projections remain at `rollout.dtype` pending the
real rollout-to-BF16-replay drift and reward gate. NVFP4 is the complete scaling
scheme. The old generic `format: fp4` spelling is rejected; use `format: nvfp4`.

A quantized rollout differs from an unquantized replay even when both base dtypes
are BF16. VRL therefore treats the full role policy as different and enables its
rollout/replay precision correction; the replay-parity gate then warns on drift
beyond `trainer.replay_parity.max_abs_logprob_diff` and fails on a non-finite
replay or an absolute log-ratio above ln(10).

## Prompt encoders

```yaml
precision:
  float32_precision: tf32
  training:
    dtype: bf16
  rollout:
    prompt_encoders:
      dtype: fp16
```

This controls generation-only text/prompt encoders for model families whose
rollout builder exposes them. It defaults to `rollout.dtype` and is independent
of policy quantization.

It does not control the VAE. Current diffusion loaders actively materialize VAE
decode modules in FP32 for output fidelity. The previous
`frozen_components` name incorrectly implied that one public dtype controlled
every frozen pipeline module and has been removed.

## Protected denoise math

Custom denoise SDE, scheduler, and log-probability math outside the transformer
always runs in FP32, independent of training/rollout base dtypes, outer autocast,
and quantization. There is no key for it: no run has ever used another dtype,
so the SDE log-prob math defaults to it. `float32_precision` still chooses
which FP32 matmul backend executes any matrix multiplications inside that
boundary.

## Separate precision boundaries

`rollout.trajectory_storage.dtype` controls trajectory transfer and host-memory
storage after generation. It does not select model compute kernels.

`distributed.training.fsdp.precision_policy` controls parameter sharding and
gradient-reduction behavior. It consumes the resolved model parameter dtype and
does not replace the public precision policy.

## Removed forms

Scalar precision and the overloaded legacy fields fail with a migration error:

```yaml
precision: bf16  # removed

actor:
  optim:
    allow_tf32: true  # use precision.float32_precision: tf32

precision:
  train: bf16                 # use training.dtype
  rollout: fp8                # use rollout.dtype + rollout.quantization
  math: fp32                  # removed; protected denoise math is always FP32
  frozen: fp16                # use rollout.prompt_encoders.dtype
  frozen_components: fp16     # removed misleading name
  rollout_recipe: rowwise     # removed; FP8 rollout is always row-wise scaled
```

The nested legacy spelling is also removed:

```yaml
precision:
  float32_precision: tf32
  training:
    dtype: bf16
  rollout:
    frozen_components:
      dtype: fp16
```

Use `precision.rollout.prompt_encoders.dtype` instead.
