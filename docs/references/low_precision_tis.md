# Reference: Low-precision rollout correction (TIS / MIS)

Curated reading guide for **truncated importance sampling (TIS)** correction,
extracted from the completed `SPRINT_low_precision_rollout_production.md` so the
slime source map survives the sprint's deletion. This is a reference, not active
work.

## Why this is parked (the trigger condition)

The fp16-rollout parity sprint is **done**: SD3.5 `fp16 rollout / fp16 replay /
fp32 math` passes the precision check with zero drift (2026-06-07 run), and the
guard enforces it (`vrl/trainers/online/precision_guard.py` +
`tests/trainers/online/test_precision_drift_guard.py::test_precision_drift_guard_checks_fp16_same_forward_precision`).
The project also defaults to bf16 (rollout == replay), so there is currently **no
behavior/proximal mismatch for TIS to correct** — building it now would be a
correction for a mismatch that has been engineered away.

TIS becomes load-bearing **only** when a deliberate rollout/replay forward
mismatch is introduced:

- **fp8 rollout** (faster but lower precision than the bf16 training forward), or
- a **disaggregated inference backend** (SGLang/vLLM-style rollout serving whose
  forward kernels differ from the HF training forward — the classic train/infer
  mismatch slime targets).

If the rollout-perf roadmap reaches either, this is the natural follow-on.

## Correct TIS semantics (the load-bearing formula)

Two importance corrections exist; **do not let one ratio play both roles**:

```text
proximal_ratio  = exp(current_compute_log_prob - old_compute_log_prob)   # PPO clip
mismatch_ratio  = exp(old_compute_log_prob - behavior_rollout_log_prob)  # train/infer gap

policy_loss = bounded(mismatch_ratio).detach() * PPO(proximal_ratio, advantage)
```

The mismatch weight is bounded, detached, and multiplies the PPO term — it is a
reweighting, not a second policy gradient. Conflating it with `exp(current -
behavior_rollout)` is the #1 bug.

Diffusion-specific caveat: slime's TIS is **LLM token-level**; the diffusion path
is **per-denoise-step log-prob**. The token→timestep mapping (sequence-level vs
per-step weighting) is real design work — do **not** copy slime's token-level
`tis_clip=2.0` verbatim.

## slime source reading map (`~/Desktop/slime`)

Read the actual source before implementing; do not work from names/READMEs.

- `slime/utils/arguments.py` — `--use-rollout-logprobs`, `--get-mismatch-metrics`,
  `--use-tis`, `--tis-clip` / `--tis-clip-low`, `custom_tis_function_path`.
  Invariants: `use_rollout_logprobs` and `use_tis` are mutually exclusive
  (`assert not args.use_tis`); `get_mismatch_metrics` forces `custom_tis_function_path`.
- `slime/ray/rollout.py` — how `rollout_log_probs` flow from rollout samples into
  `train_data` and split by data-parallel partition.
- `slime/backends/megatron_utils/actor.py` — when the training engine recomputes
  `compute_log_prob`; why `get_mismatch_metrics` forces an extra forward even
  under `use_rollout_logprobs`:
  `if not use_rollout_logprobs or get_mismatch_metrics: rollout_data.update(compute_log_prob(...))`.
- `slime/backends/megatron_utils/loss.py` — `old_log_probs = rollout_log_probs if
  use_rollout_logprobs else log_probs`; `vanilla_tis_function` weight direction;
  TIS weight detach; where `pg_loss` is multiplied; `train_rollout_logprob_abs_diff`
  reporting:
  `tis = exp(old - rollout); w = clamp(tis, tis_clip_low, tis_clip); pg_loss *= w`.
- `examples/train_infer_mismatch_helper/mis.py` — token/sequence/geometric levels;
  truncate/clip/mask modes; `SAFETY_BOUND = 20.0` for exp overflow; batch
  normalize to mean=1.0; rejection sampling + veto threshold; metrics aggregated
  on the pre-RS mask.

## Local plumbing already in place

The two log-prob sources TIS needs already exist:

- Rollout log-probs are written into the trajectory by `sde_step_with_logprob`
  (`vrl/generation/diffusion/executor.py` → `buffers.log_probs`).
- Replay recomputes log-probs via the evaluator in `vrl/trainers/online/trainer.py`.

So an implementation is mostly: gate behind a `correction_mode` flag (default
off, like `denoise_compile`), wire the rollout log-prob through as the mismatch
baseline, and add the bounded/detached reweight to the GRPO loss. Off-by-default
keeps the bf16 path untouched.

## External references

- slime train/infer mismatch helper: <https://github.com/THUDM/slime/blob/main/examples/train_infer_mismatch_helper/README.md>
- slime loss: <https://github.com/THUDM/slime/blob/main/slime/backends/megatron_utils/loss.py>
- verl FP8 RL: <https://verl.readthedocs.io/en/latest/low_precision/fp8.html>
- lmsys FP8 RL: <https://www.lmsys.org/blog/2025-11-25-fp8-rl/>
- SGLang for RL: <https://sgl-project.github.io/advanced_features/sglang_for_rl.html>
