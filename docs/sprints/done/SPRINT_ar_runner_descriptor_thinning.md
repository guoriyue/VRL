# SPRINT: AR runner and runtime descriptor thinning

**Date:** 2026-07-10
**Status:** DONE

## Goal

Remove repeated AR runtime construction and discrete-token loop bookkeeping
without flattening genuinely different cache, action-space, or trajectory
contracts.

## Changes

1. Added `ARDiscreteTokenState` and `ARDiscreteTokenRunner` as the single owner
   of discrete runner validation, step/finalize wiring, and debug counters.
2. Made `PagedCFGARState` / `PagedCFGTokenRunner` build on that shared contract.
3. Consolidated Emu3 and Janus cond/uncond prefill plus CFG-behavior sampling /
   conditional-policy log-prob scoring.
4. Added `ARFamilyBuild` registry descriptors and generic rollout, replay, and
   spec construction in `vrl.models.ar.build`.
5. Removed the 15 repeated family build/replay/spec facades from Emu3,
   GLM-Image, Janus-Pro, LlamaGen, and NextStep-1 runtime modules.
6. Closed the temperature parity follow-up: rollout and replay now normalize
   categorical logits with the same recorded positive temperature, all
   discrete runners reject zero/non-finite temperatures before prefill, and
   Janus, GLM-Image, Emu3, and LlamaGen persist that value in trajectory
   context. Greedy decoding remains an explicit policy mode rather than a
   near-zero categorical approximation.
7. Fixed the generic AR training entrypoint to resolve algorithm-selected
   online families, so `janus_pro + token_grpo_multisegment` reaches the
   registered `janus_pro_r1` runtime and replay contracts.

## Deliberate boundaries kept

- Family `runner.py` modules remain protocol adapters. Their cache advancement
  is not interchangeable: GLM-Image owns explicit 3-axis mrope positions,
  LlamaGen owns a static in-place KV cache, and NextStep emits continuous flow
  tokens with saved replay noise.
- Family `runtime.py` modules remain request/trajectory adapters and own their
  concrete chunk executors.
- Family LoRA default tables remain isolated configuration taxonomies; only the
  merge/conversion algorithm is shared.
- Janus R1 remains a multisegment executor/gatherer instead of being forced into
  the discrete image-only path.
- CFG and top-k/top-p remain behavior-distribution controls. Policy log-probs
  continue to use conditional logits; this follow-up only adds the missing
  temperature normalization shared by rollout and replay.
- Directly stored replay `log_probs` remain untouched because they were already
  normalized during rollout. Temperature is applied only when replay stores
  logits and must reconstruct log-probs.

## Verification

- `ruff check vrl tests`
- `python -m vrl.config.lint`
- `git diff --check`
- `pytest -q`: 1675 passed, 16 skipped
