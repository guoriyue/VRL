# SPRINT: PT26-02 — Reuse Qwen3-VL image prefixes across question chunks

Status: **planned — P1; CPU cache-lifecycle tests can start now; adoption requires the real judge on GPU.**

Prepared 2026-10-24 against repository `c090e97db`. Re-read current definitions, producers, consumers, tests and recipe references before implementation. Parent: [execution program](SPRINT_pytorch2026_execution_program.md).

## Current state

`vrl/rewards/models/qwen3vl_preference.py::answer_scores` already shares image-prefix KV within one chunk. When questions exceed `questions_per_forward`, recursive calls repeat image preprocessing and prefix forward per chunk. The new scope is reuse within one scoring call across chunks. Existing within-chunk reuse is not missing functionality.

## Deliverables

- [ ] Profile image fitting/processing, vision/prefix forward, cache cloning/repetition, suffix forward, total reward wall time and allocated/reserved memory for one and many chunks.
- [ ] Compute the image prefix once for an invocation. Retain an immutable base prefix and prefix M-RoPE state; give each chunk independent writable cache state because cache repeat and suffix forward can mutate it.
- [ ] Preserve per-chunk memory limits, question/answer ordering, yes/no and rating normalization, partial final chunks and reference-image ordering. Different crops, image sets and reversed pair order must get different prefixes.
- [ ] Release all prefix/chunk state on success and failure and preserve reward park/offload/wake behavior. Avoid global or cross-invocation caches.
- [ ] Extend `tests/rewards/test_qwen3vl_preference.py` with mutation-sensitive cache fakes, repeated calls with changed images, uneven chunks, reference ordering and injected suffix failure.

## Acceptance

Compare every question's logits/answer probability and final scores against the existing chunked path under pinned reduction settings. Declare a justified numerical tolerance; aggregate reward alone is insufficient. Require lower total judge latency on a representative multi-chunk workload, no meaningful one-chunk regression, acceptable peak/reserved VRAM, no OOM/retry regression and measured collection-level benefit before default enablement.

## Boundaries

Keep current chunk size and scorer interface. Use local helpers only for the prefix/cache lifecycle seam; preserve family/scorer facades. This judge evaluates suffix logits rather than autoregressively generating an answer, so decode-serving optimizations are not automatically applicable.

Existing umbrella: [rollout/reward throughput](SPRINT_rollout_reward_throughput_pt26.md), T2. This plan owns only the remaining across-chunk opportunity; it does not reopen completed within-chunk sharing.
