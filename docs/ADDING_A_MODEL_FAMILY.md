# Adding a Model Family

This guide shows how to make a new diffusion or autoregressive image/video model
RL-trainable in `visual-rl`. The point of the unified contract is that a new family
is a **registry entry + one model directory** — the trainer, algorithms, rollout
orchestration, rewards, and config system are untouched. Worked example: the
existing `sd3_5` family.

## 0. What you implement

A family is a directory under `vrl/models/<modality>/<family>/` exposing three
symbols, plus one line in the rollout registry. Nothing else in `vrl/` changes.

| You provide | For `sd3_5` |
| --- | --- |
| `<Family>ChunkExecutor` — runs the denoise/decode rollout, records the trajectory | `vrl.models.diffusion.sd3_5.runtime:SD3_5ChunkExecutor` |
| `build_<family>_runtime_bundle` — loads weights into a `RuntimeModel` (with `replay_forward`) | `…runtime:build_sd3_5_runtime_bundle` |
| `extract_<family>_runtime_spec` — projects the YAML config into a typed runtime spec | `…runtime:extract_sd3_5_runtime_spec` |
| one `register_rollout_family(...)` entry | `vrl/rollouts/families/registry.py:132` |

Typical directory (mirror an existing family):

```text
vrl/models/diffusion/<family>/
  __init__.py     # re-exports the public symbols
  model.py        # the nn.Module + replay_forward(batch, timestep_idx)
  runtime.py      # <Family>ChunkExecutor, build_*_runtime_bundle, extract_*_runtime_spec
  runner.py       # the per-step generation/denoise driver
```

## 1. The contract: `replay_forward`

The one method that makes RL work is `replay_forward(batch, timestep_idx)` on your
`RuntimeModel`. The rollout records a multi-step denoising (or AR) trajectory; the
evaluator **replays** it through the current weights to recompute `log_prob` at each
step. This is the correctness core — the recomputed `old_log_prob` must match the
rollout-time distribution (the verl rule: never corrupt `old_log_prob`). For
diffusion, `replay_forward` returns the per-step SDE signals
(`log_prob`, `prev_sample_mean`, `std_dev_t`); for AR, the per-token logits/log-prob.
The algorithm layer (`vrl/algorithms/`) consumes those signals unchanged — you do not
write algorithm code.

## 2. Register the family (the one line)

Diffusion families use the `_diffusion_entry` helper. Add to
`vrl/rollouts/families/registry.py`:

```python
register_rollout_family(
    _diffusion_entry(
        family="my_model",
        task="t2i",                       # t2i | t2v | i2v | v2w | t2w
        aliases=(),
        executor_cls="vrl.models.diffusion.my_model.runtime:MyModelChunkExecutor",
        runtime_builder="vrl.models.diffusion.my_model.runtime:build_my_model_runtime_bundle",
        runtime_spec_extractor="vrl.models.diffusion.my_model.runtime:extract_my_model_runtime_spec",
        request_prefix="my_model",
        default_task_type="text_to_image",
        supports_reference_conditioning=False,   # True for I2V / V2W (adds reference_image plumbing)
    ),
)
```

That single entry wires the collector, the diffusion gatherer, the family
capability, and the executor kwargs. AR families register analogously (see the
`janus_pro` / `nextstep_1` entries in the same file) with `kind="ar_*"`.

Set `supports_reference_conditioning=True` for image/video-conditioned families
(I2V, V2W): the registry then threads `reference_image` through the collector and
reward path for you.

## 3. Config layers

Recipes compose from layers under `configs/`. Add the family's defaults, then an
experiment recipe that selects them:

```text
configs/base/model/my_model.yaml              # weights path, dtype, LoRA, scheduler
configs/experiment/diffusion/my_model/online_grpo_<reward>.yaml
```

Mirror an existing recipe (`configs/experiment/diffusion/sd3_5/online_grpo_ocr.yaml`)
and swap the model/sampling layers. The algorithm, rollout, and distributed layers
are shared and need no family-specific change.

## 4. Validate (the promotion bar)

A family is **🧪 Runnable** once it loads and a rollout produces artifacts. It is
**✅ Validated** only after a real run shows optimizer updates, a **non-flat**
`reward_mean`, generated artifacts, and changed weights (see the README Status
Policy). Flat reward is a bug, not a result.

Add at minimum:
- a model-loading test under `tests/models/diffusion/<family>/test_model_loading.py`
  (mirror an existing one) — proves weights load and shapes are right;
- a backbone-parity test if you wrap a diffusers/upstream backbone — proves your
  `replay_forward` matches the reference forward within tolerance.

Then run the recipe (`vrl-train --config experiment/diffusion/my_model/...`) and,
once it clears the bar, flip the README Supported-Models row to ✅ with a link to a
reproducible curve under `docs/runs/`.

## 5. Checklist

- [ ] `vrl/models/<modality>/<family>/` with `model.py` (`replay_forward`),
      `runtime.py` (executor + `build_*` + `extract_*`), `runner.py`, `__init__.py`.
- [ ] `replay_forward` recomputes per-step `log_prob` consistent with rollout
      sampling (do not corrupt `old_log_prob`).
- [ ] one `register_rollout_family(...)` entry in `registry.py`.
- [ ] config layers + one experiment recipe.
- [ ] model-loading (+ backbone-parity) test.
- [ ] a real run clears the promotion bar before marking ✅.

See [`docs/NORTH_STAR.md`](NORTH_STAR.md) for why this contract is the moat: adding a
13th family here is a registry line, not a fork.
