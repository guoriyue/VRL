# SPRINT: PT26-10 — Isolated quantized-training input-gradient pilot

Status: **planned — P2 research; start only with an isolated compatible environment, a measured replay bottleneck and PT26-00.**

Prepared 2026-10-24 against repository `c090e97db`. Re-read current definitions, producers, consumers, tests and recipe references before implementation. Parent: [execution program](SPRINT_pytorch2026_execution_program.md).

## Current state

FP8/NVFP4 rollout caches and four-over-six calibration already exist; they are not training quantization. Inspect `vrl/nn/quantization/base.py`, `vrl/nn/optimization/passes.py` and current config validation before experimentation. Keep the default training-precision boundary intact.

## Deliverables

- [ ] Start with one representative LoRA block and pinned tensor shapes. Define forward, backward, accumulation, optimizer/master-weight and export dtypes independently.
- [ ] Compare three explicit arms where supported: BF16 reference; quantized forward with BF16 backward; quantized forward with an experimental quantized dX estimator such as MS-EDEN. Do not label a changed forward as an unchanged BF16 gradient reference.
- [ ] Preserve input gradients through the frozen base: frozen base parameters do not eliminate dX. Use `needs_input_grad` to skip dW and saved activations only when genuinely unnecessary. Cover bias, non-aligned dimensions, nonfinite inputs and actual autograd registration.
- [ ] Validate CPU/CUDA/Python RNG and activation-checkpoint recomputation, saved tensors and selective-AC dispatch; verify custom operators actually hit the intended OpOverload policy rather than silently escaping it.
- [ ] Expand only after block checks to full replay/update, repeated RL training and held-out/multi-seed reward/evaluation. Measure memory, full-step throughput and quality jointly; retain immediate BF16 fallback.

## Acceptance and stop rule

A usable result needs correct gradients/update within predeclared gates, stable accumulation and training, compatible state export and a material full-step gain. Finite gradients or lower reconstruction MSE are insufficient. Stop or revert on instability or held-out regression even when kernel throughput improves.

## Boundaries

Keep rollout caches detached from master training state, checkpoint/export keys and default recipes. No optimizer replacement or full-precision change based on a pretraining headline. Extend existing `tests/nn/quantization` only for the chosen backend and real invariants.

References: [TorchAO training workflows](https://docs.pytorch.org/ao/stable/workflows/training.html), [Quartet II paper](https://proceedings.mlr.press/v306/panferov26a.html).
