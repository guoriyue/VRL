# Profile phase taxonomy

The stage names passed to `profile_range` (`vrl/utils/profiling.py`) are a
**contract, not free text**. They are the only thing about a performance
measurement that survives a hardware change: absolute numbers from an RTX 5090
say nothing about an H100, but "reward scoring is 23% of the step" is a ratio
that transfers and can be compared run to run.

`SPRINT_trustworthy_profiling_api.md` specified the *mechanism* (torch range +
optional NVTX, the manifest as trust anchor). This file specifies the
*vocabulary*, so a new call site picks an existing phase instead of coining a
synonym that silently splits one phase into two names.

## Rules

1. **`namespace.phase`**, both lowercase snake_case. The namespace says which
   subsystem owns the wall-clock; the phase says what work it is.
2. **Add a name only when the phase can be scheduled or optimized separately.**
   If two blocks always run back-to-back on the same device with nothing able to
   fill the gap, they are one phase.
3. **Never rename a live phase for taste.** The name is the join key against
   every trace already on disk; renaming silently orphans that history.
4. **Nested ranges are attribution aids, not additive percentages.** A kernel
   inside two nested ranges is counted under both. Do not sum them to a wall
   fraction — the profiler summary repeats this warning for the same reason.

## The namespaces

| Namespace | Owner | What it covers |
|---|---|---|
| `generation.` | `vrl/generation/steps/`, `vrl/generation/bindings/` | Diffusion denoise loop and its per-step pieces |
| `engine.` | `vrl/generation/execution/`, family runtimes | Engine-level planning, prefill/decode, KV cache |
| `trainer.` | `vrl/trainers/online/` | Forward/backward/loss/optimizer of the update |
| `collector.` | `vrl/rollouts/collector/` | Post-rollout scoring |
| `weight_sync.` | `vrl/trainers/weight_sync.py` | Trainer -> rollout-worker weight handoff |

## Current phases

Derived from the call sites; regenerate with:

```bash
grep -rhno 'profile_range("[a-z_.]*"' vrl/ --include=*.py \
  | grep -o '"[a-z_.]*"' | tr -d '"' | sort -u
```

### `generation.`
| Phase | Meaning |
|---|---|
| `generation.prompt_encode` | Text/prompt encoding before the loop |
| `generation.prepare_sampling` | Sampler/scheduler setup |
| `generation.denoise_step` | One full denoise iteration (outer; encloses the rest) |
| `generation.denoise_forward` | Policy transformer forward inside a step |
| `generation.ref_denoise_forward` | Reference-model forward (KL / ratio terms) |
| `generation.scheduler_step` | Scheduler update from model output to next latent |
| `generation.latent_snapshot` | Capturing latents for the trajectory |
| `generation.latent_write` | Writing latents out |
| `generation.trajectory_buffer_write` | Appending the step to the trajectory buffer |
| `generation.decode_latents` | VAE decode to pixels |

### `engine.`
| Phase | Meaning |
|---|---|
| `engine.plan` | Execution planning before dispatch |
| `engine.forward_chunk` | One chunk of batched forward work |
| `engine.prefill` | Autoregressive prefill |
| `engine.decode_step` | One autoregressive decode step |
| `engine.cache_read` | KV cache read |
| `engine.cache_write` | KV cache write |
| `engine.vq_decode` | VQ token decode |

### `trainer.`
| Phase | Meaning |
|---|---|
| `trainer.replay` | Evaluator forward over stored trajectories |
| `trainer.loss` | Algorithm objective reduction |
| `trainer.backward` | Backward pass |
| `trainer.optimizer_update` | Optimizer step |
| `trainer.sft_regularizer` | SFT regularization term |

### `collector.`
| Phase | Meaning |
|---|---|
| `collector.reward_score` | Reward model scoring of rollout samples |

### `weight_sync.`
| Phase | Meaning |
|---|---|
| `weight_sync.state_to_cpu` | Device -> host copy of trainable state |
| `weight_sync.push` | Transport + worker-side load |

These two are deliberately split. They are one logical operation today because
rollout and training share a GPU (`generation_overlap_safe: false`), but they
have different costs and different overlap potential the moment they do not:
the copy is trainer-side GPU work, the push is transport plus remote load. A
fused range would hide which half to schedule against.

## Why this matters more than the absolute numbers

The single most transferable habit in DeepSeek's published infrastructure work
is not a kernel — it is that they decompose a step into *named* phases, schedule
against those names, and publish the traces so the schedule is checkable
(`deepseek-ai/profile-data`). The DeepSeek-V3 report's overlap design is stated
in exactly these terms: attention / all-to-all dispatch / MLP / all-to-all
combine, with the SM split tuned per phase.

Our phases are different (diffusion RL has no all-to-all), but the discipline is
the same: a phase you have not named is a phase you cannot schedule, and a ratio
you did not record before changing hardware is a comparison you cannot make
afterwards.

## References

- `vrl/utils/profiling.py` — `profile_range`, `capture_torch_trace`, manifest
- `docs/sprints/done/SPRINT_trustworthy_profiling_api.md` — mechanism and non-goals
- `tests/utils/test_profiling.py` — range transparency / NVTX pairing
- `tests/trainers/test_weight_sync.py` — pins the `weight_sync.*` ranges
