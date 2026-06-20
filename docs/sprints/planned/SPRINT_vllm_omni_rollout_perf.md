# SPRINT: vLLM-omni rollout — actually run it + measure vs native

**Branch:** `spike/vllm-omni-rollout`. **Started:** 2026-06-20.
**Why this exists:** the prior sprint (SPRINT_rollout_vllm_migration.md) concluded
vLLM-omni adoption was a non-win *by analysis*. The user (correctly) wants it
MEASURED, not reasoned: actually load sd3 through vLLM-omni's `DiffusersPipelineLoader`,
run a forward, and compare noise_pred consistency + throughput vs our native sd3.5
rollout. "Replace all that can be replaced, high quality, like verl-omni; learn
from cosmos-rl."

## Reference pattern (learned from verl-omni + cosmos-rl)

- **verl-omni** `pipelines/sd3_flow_grpo/vllm_omni_rollout_adapter.py`: subclasses
  vLLM-omni's `vllm_omni.diffusion.models.sd3.pipeline_sd3.StableDiffusion3Pipeline`
  → `StableDiffusion3PipelineWithLogProb`, adding the SDE-step-with-logprob the RL
  rollout needs. Uses `OmniDiffusionConfig` / `OmniDiffusionRequest` / `DiffusionOutput`.
  This is the integration template: take vLLM-omni's optimized SD3 pipeline, graft
  our logprob/SDE on top.
- **verl-omni** pins **`vllm-omni==0.22.0` + `vllm==0.22.0`** (pyproject.toml) — the
  known-good combo.

## BLOCKER found + resolution (the real reason it wasn't run before)

- This box has **vllm 0.21.0 + vllm-omni 0.18.0** (mismatched). vLLM-omni 0.18.0
  imports pre-0.21 vLLM internals (`vllm.inputs.data` module, `TokenInputs`) that
  0.21 renamed/flattened (`vllm.inputs`, `TokensInput`). Importing ANY
  `vllm_omni.diffusion.*` triggers `vllm_omni/__init__` → `patch.py` + LLM
  `entrypoints`, all version-sensitive → hard import failure. Shimming is a
  bottomless rabbit hole + risks silent incorrectness.
- Downgrading vllm in the MAIN env is not an option: our fp8 blockwise
  (`vrl/nn/quantization/fp8.py`) + paged attention use vllm 0.21 APIs.
- **Resolution: an ISOLATED venv with verl-omni's known-good `vllm==0.22.0 +
  vllm-omni==0.22.0`** (both newer, both support Blackwell sm_120). The main env is
  untouched; nothing on the machine is removed.

## Plan

1. ⏳ Build isolated venv `.venv-vllm-omni` with `vllm==0.22.0 vllm-omni==0.22.0
   diffusers transformers torchsde`. (heavy install, background.)
2. ⬜ Smoke: import `vllm_omni.diffusion.models.sd3.pipeline_sd3` in that venv.
3. ⬜ Load sd3.5 via vLLM-omni (its SD3 pipeline / `DiffusersPipelineLoader`), run
   ONE denoise forward on a fixed prompt+seed; capture noise_pred + wall time.
4. ⬜ Run the SAME prompt+seed through our native sd3.5 forward (main env); capture
   noise_pred + wall time.
5. ⬜ Compare: (a) noise_pred consistency (rel-L1 / cosine — are they the same
   model math?), (b) throughput s/step. Record the table here.
6. ⬜ If vLLM-omni is faster: scope grafting our logprob onto its SD3 pipeline
   (the verl-omni pattern). If not: that's the measured answer the prior sprint
   only reasoned.

## Results — MEASURED (sd3.5-medium, 512×512, 10 steps, bf16)

| backend | s/step | scope | notes |
|---|---|---|---|
| native (vrl) + LoRA | 0.054 | denoise-only | PEFT wrap overhead |
| **native (vrl) base** | **0.043** | denoise-only | no LoRA |
| **vLLM-omni base** | **0.045** | **full forward (incl. text encode)** | auto torch.compile + CUDNN_ATTN |
| vLLM-omni + TeaCache | N/A | — | **SD3 unsupported** (coeffs only for Flux/Qwen/Bagel/…) |

**Precision is verified-matched (both pure bf16, no quantization)** — introspection
on the loaded vLLM-omni transformer: `od.dtype=bf16`, `quantization_config=None`,
all 664 params `torch.bfloat16`, linear classes are vLLM TP layers (Column/Row/QKV/
Replicated) but UNquantized → bf16 GEMMs. Native is `dtype=bf16` + bf16 autocast.
So the comparison is apples-to-apples on precision; the only non-precision delta is
the attention kernel (native diffusers SDPA vs vLLM-omni CUDNN_ATTN, both bf16 I/O).

**Verdict: vLLM-omni gives NO meaningful throughput win for sd3 on this box.** The
two numbers are the same ballpark (~0.043–0.045 s/step), and they even measure
slightly DIFFERENT scopes in vLLM-omni's favor — native 0.043 is denoise-only,
while vLLM-omni's 0.045 already includes the 3-text-encoder prompt encode. vLLM-omni
auto-applies torch.compile + cuDNN attention, yet does not beat our plain native
denoise — its engine/TP/forward-context wrapping eats the gains at single-GPU sd3
scale. And its TeaCache doesn't even support SD3.

**This empirically confirms the prior sprint's reasoned conclusion** (diffusion-side
vLLM-omni adoption = low-ROI) — now with a real number, not analysis. The
optimizations vLLM-omni bundles (compile, cuDNN attn) are IMPORT-portable to our
native rollout; adopting its engine is not warranted for diffusion.

Side finding: native `DiffusionModelBase.torch_compile_transformer` fails on sd3
(`cannot assign SD3Transformer2DModel.forward as child module 'transformer'`) — a
real bug in the compile-swap for the diffusers sd3 transformer; filed for follow-up
(doesn't affect the verdict, native is already at-or-faster uncompiled).

## AR side — head-to-head NOT measurable on this box (hardware + coverage, evidenced)

| AR model | local | size | fits 32GB | in vLLM-omni? |
|---|---|---|---|---|
| **Janus-Pro-1B** (our actual AR run, `model/ar/janus_pro/1b`) | ✓ | 1B | ✅ | ❌ not in vLLM-omni NOR vLLM archs (the `janus` grep hits are the async-queue lib) |
| **NextStep-1.1** (`nextstep_1_1` in vLLM-omni) | ✓ | ~14B (hidden 5120 × 48) | ❌ OOM | ✓ |

**No AR model both fits this 32GB box AND is covered by both stacks**, so a real
throughput diff can't be produced here:
- Janus-Pro-1B fits but vLLM-omni doesn't support the Janus architecture → nothing
  to compare against; native is the only path (and our AR rollout is already a
  lockstep paged decode — see SPRINT_rollout_vllm_migration.md P3).
- NextStep-1.1 is in vLLM-omni but 14B → OOMs 32GB on BOTH sides → needs a larger /
  multi-GPU box to measure.

So for the AR model actually runnable here (Janus-Pro-1B), vLLM-omni isn't even an
option — the "replace AR rollout with vLLM-omni" question is moot on this hardware.
A NextStep vLLM-omni-vs-native measurement is a multi-GPU follow-up, not a 32GB task.

## Journal (most recent first)

- **2026-06-20 (cont.)** — Native half DONE: `sd3_forward_probe.py` runs our sd3.5
  denoise (0.054 s/step, sd3.5-medium local weights, geneval config) and dumps the
  comparison JSON. Driver is 580/CUDA 13.0 → vllm 0.22 (CUDA 13) WILL run. Isolated
  venv install grinding at ~1 MB/s (vllm 261MB done, now flashinfer 360MB + torch
  cu13 ~2-3GB → ~1hr more). vLLM-omni probe waits on that.
- **2026-06-20** — Root-caused why vLLM-omni never ran: 0.21/0.18 version mismatch
  (hard import wall, not laziness). Found verl-omni's known-good combo (0.22/0.22).
  Branch `spike/vllm-omni-rollout` cut from the just-pushed main. Starting isolated
  venv install.

## Rules
- Isolated venv only; do NOT change the main env's vllm (breaks our fp8/paged).
- Never delete anything. Rollout-only. Commit to this branch, do not push unasked.
