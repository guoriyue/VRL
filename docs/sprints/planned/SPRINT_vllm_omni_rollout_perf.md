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

## Journal (most recent first)

- **2026-06-20** — Root-caused why vLLM-omni never ran: 0.21/0.18 version mismatch
  (hard import wall, not laziness). Found verl-omni's known-good combo (0.22/0.22).
  Branch `spike/vllm-omni-rollout` cut from the just-pushed main. Starting isolated
  venv install.

## Rules
- Isolated venv only; do NOT change the main env's vllm (breaks our fp8/paged).
- Never delete anything. Rollout-only. Commit to this branch, do not push unasked.
