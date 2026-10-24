# SPRINT: PT26-00 — Recipe-driven optimization acceptance

Status: **planned — P0; start with CPU schema/fixture work; GPU acceptance needs an available target GPU.**

Prepared 2026-10-24 against repository `c090e97db`. Re-read current definitions, producers, consumers, tests and recipe references before implementation. Parent: [execution program](SPRINT_pytorch2026_execution_program.md).

## Current state and ownership

`vrl/scripts/generation/full_sequence_probe.py` already checks recipe-driven rollout/replay at every denoise step, supports nonzero LoRA B, and offers eager first-divergence hooks. Its forward checks run under no-grad; it does not validate backward or an optimizer update. Extend that tool and reuse trainer execution rather than building another forward-only benchmark. The retired reward-overlap benchmark is not an implementation dependency.

## Deliverables

- [ ] Define a small baseline/candidate run contract: resolved recipe, source revision and diff identity, dependencies, device/topology, prompts/reference assets, seeds, precision and attention settings, initial trainable state, scorer identities/weights and evaluation split.
- [ ] Reuse probe output for trajectory and per-step logprob checks. Add a training-side companion that exercises actual loss, backward, clipping, optimizer state and one update; do not turn the no-grad probe into a second trainer.
- [ ] Record per-component reward and held-out evaluation, gradient/update differences, pre-update clip fraction, mismatch/TIS/RS statistics, allocated/reserved peaks, retries/OOMs and outer wall time. Distinguish lag-zero numerical mismatch from stale-policy effects.
- [ ] Separate cold compile/autotune/cache-build cost from warmed throughput. Count accepted usable training trajectories and completed updates over the entire run, including failures; report raw samples/s separately.
- [ ] Add a fail-closed comparison reader: absent fields, incomplete runs, incompatible recipes, mismatched scorer provenance, nonfinite data and failed correctness gates cannot become a passing speed result. Tolerances must be explicit before comparing candidates.

## Acceptance and resources

CPU fixtures must reject missing verdicts/metrics, a changed workload and a deliberately perturbed update, and accept a complete matching pair. A GPU smoke comparison must execute an actual backward/update and reject a wrong candidate. Production adoption additionally needs repeated warm measurements and held-out/multi-seed quality evidence at the scope affected; a microbenchmark or one-update smoke is insufficient.

Use `tests/scripts/test_full_sequence_denoise_probe.py`, trainer tests and `vrl/scripts/eval/image_checkpoint_eval.py` as existing seams. `+reward=` evaluation overrides replace the scorer configuration: pin scorers and weights across arms and retain training-reward provenance. Report thresholds and measurement variability rather than promising a conference speedup.

## Boundaries and stop rule

Keep recipe parsing, trainer loss/strategy ownership and evaluator provenance authoritative. Add only a runner/report facade where it removes duplicated orchestration. No generic experiment platform, global instrumentation framework, deleted benchmark resurrection or automatic deployment. A candidate failing correctness remains rejected even when faster. Run artifacts live outside tracked sprint documents; keep only evidence needed for a pending decision.
