# Reading: VeRL-Omni (verl-omni) — rollout, borrows, lessons for vrl

**What it is.** verl-omni (`import verl_omni`, v0.1.0, ~19k LOC) is a thin diffusion/omni-modal RL **override layer** built on top of upstream `verl`. It does not own a distributed-RL core: it pip-depends on `verl==0.8.0` (Ray single-controller, DataProto, checkpoint engine, FSDP2 utils, agent/reward-loop managers) and adds only a diffusion Ray trainer, two algorithm registries, per-model pipeline adapters, and rollout/reward adapters. Rollout runs on **vLLM-Omni** (`AsyncOmni`, the diffusion-capable analogue of vLLM's `AsyncLLM`); the trainer is FSDP2 or **VeOmni**, selectable by one config string. Its headline trick: treat "diffusion generation" as a single request-scoped job inside verl's existing async-rollout-server + agent-loop + DataProto contract, and push all diffusion specifics into a thin per-`(architecture, algorithm)` pipeline subclass.

Origin: built on verl (volcengine HybridFlow); rollout backend vLLM-Omni; trainer backend VeOmni/FSDP2. verl is a pip dependency, **not** vendored and **not** a submodule (`.gitmodules` is 0 bytes; `pyproject.toml:55-61` pins `verl==0.8.0`, `vllm-omni==0.22.0`, `vllm==0.22.0`).

---

## 1. How verl-omni does rollout

**One request = one prompt's complete denoise loop.** Rollout reuses verl's async rollout-server machinery and only swaps the engine. `verl_omni/workers/rollout/base.py:16` registers the diffusion backend into verl's existing registry:

```python
_ROLLOUT_REGISTRY[("vllm_omni", "async")] = "verl.workers.rollout.vllm_rollout.ServerAdapter"
```

Only `mode="async"` exists — in-process sync rollout is hard-removed (`verl_omni/workers/config/diffusion/rollout.py:169` raises on `mode=="sync"`). The HTTP server launches the diffusion engine, not vanilla vLLM:

```python
engine_client = AsyncOmni(**engine_args)   # vllm_omni_async_server.py:168
```

The per-pipeline **rollout adapter** is injected into the engine by dotted class path, keyed on `(architecture, algorithm)` resolved from the registry (`vllm_omni_async_server.py:149-156`, `enable_dummy_pipeline=True` + `custom_pipeline_args={"pipeline_class": ...}`). That adapter — e.g. `StableDiffusion3PipelineWithLogProb` — subclasses vLLM-Omni's stock SD3 pipeline, swaps the Euler scheduler for a `FlowMatchSDEDiscreteScheduler`, and **drives the denoise loop itself**, collecting the trajectory inside the SDE window:

```python
# sd3_flow_grpo/vllm_omni_rollout_adapter.py:270-281
latents, log_prob, _, _ = self.scheduler.step(
    noise_pred.float(), timestep_value, latents,
    generator=generator, noise_level=cur_noise_level, sde_type=sde_type,
    return_logprobs=logprobs, return_dict=False)
# Save fp32 trajectory BEFORE casting back to model dtype:
if i >= sde_window[0] and i < sde_window[1]:
    all_latents.append(latents.float()); all_log_probs.append(log_prob); all_timesteps.append(timestep_value)
```

It ships the trajectory + the prompt embeddings back through the rollout→trainer contract (`vllm_omni_rollout_adapter.py:423-437`):

```python
return DiffusionOutput(output=image, custom_output={
    "all_latents": all_latents, "all_log_probs": all_log_probs, "all_timesteps": all_timesteps,
    "prompt_embeds": prompt_embeds, "pooled_prompt_embeds": ..., "negative_*": ...}, to_cpu=True)
```

The contract itself is small and **algorithm-shaped** (`verl_omni/workers/rollout/replica.py:20-31`): a pixel/latent tensor, optional per-step `log_probs`, and an `extra_fields` bag for the rest. The shape varies by algorithm: FlowGRPO/DanceGRPO ship a reverse-SDE trajectory + log-probs, while the NFT adapter (forward-process objective) deliberately ships **no** trajectory — only `latents_clean` + `train_timesteps` (`qwen_image_diffusion_nft/vllm_omni_rollout_adapter.py:258-267`).

**Request-scoped, no engine batching.** Each agent-loop turn issues exactly one `generate` with a fresh uuid (`single_turn_agent_loop.py:62-70`); the engine async-generator is drained to one final `OmniRequestOutput` (`vllm_omni_async_server.py:378-386`). GRPO's `n>1` is interleaved row-repeat at the batch level before the agent loop fans out concurrent per-sample requests — not continuous batching inside the engine.

**The trainer re-uses the trajectory.** The FSDP/VeOmni diffusion engine reads `num_timesteps` from the saved `all_timesteps` and re-runs the transformer per saved step to recompute log-probs and backprop (`workers/engine/veomni/diffusion_impl.py:442,459`):

```python
num_timesteps = data["all_timesteps"].shape[1]
...
for step in range(num_timesteps):
    loss, meta_info = self.forward_step(micro_batch, loss_function=..., step=step)
```

**System mode is co-located, on-policy SYNC.** Actor (trainer) and rollout live in one Ray worker (`create_colocated_worker_cls`), time-sharing GPUs via a sleep/wake barrier: generate → sleep rollout → recompute/update actor → `update_weights` (wakes rollout + pushes weights) (`ray_diffusion_trainer.py:988-989, 1090-1091`). "Async" here means three orthogonal things, none of which makes the policy update off-policy: (a) the rollout *server* is always asyncio; (b) **sample-level reward overlap** — each finished sample streams to a reward actor while others still denoise; (c) a disaggregated weight-transport path for non-colocated setups. The policy-update boundary stays strictly on-policy (`docs/algo/async_reward.md:51`).

**Weight sync.** Co-located "naive" path: resume rollout weight memory → gather actor params → push over **ZMQ IPC** bucket-by-bucket (`engine_workers.py:918,923` `BucketedWeightSender.async_send_weights`, RPC carries only metadata) → offload actor to CPU → resume rollout kv-cache last. Resharding is one seam — `get_per_tensor_param` collapses any FSDP2/USP/TP/DP/EP layout to full bf16 per-tensor params with `transformer.`-prefixed keys via `DTensor.full_tensor()` (`diffusers_impl.py:762`). Sleep is **level-1** so non-trainable VAE/text-encoder (never synced) are offloaded-and-restored, not discarded (`vllm_omni_async_server.py:204-210`).

---

## 2. What verl-omni borrows

- **verl (volcengine HybridFlow)** — hard pip dependency + heavy subclass. The entire async-rollout-server, agent-loop, DataProto, single-controller/RayWorkerGroup, ResourcePoolManager, CheckpointEngine, PPO metric utils, FSDP2 utils (`apply_fsdp2`, `DTensor`), `BucketedWeightSender/Receiver`, `RewardLoopManager`, and the GSPO `loss_mode` are all verl's. verl-omni subclasses `vLLMHttpServer`/`vLLMReplica` and adds only the diffusion branch (~50 `from verl` import sites). The omni/GSPO track is *pure verl* — launched via `python3 -m verl.trainer.main_ppo` with verl-omni injected only through `VERL_USE_EXTERNAL_MODULES=verl_omni` (`examples/gspo_trainer/run_qwen3_omni_thinker_gspo_lora.sh:14`); `verl_omni/trainer/omni/` is empty.
- **vLLM-Omni** — pip dependency, the rollout engine. `AsyncOmni`, `OmniEngineArgs`, `OmniDiffusionSamplingParams`, `OmniRequestOutput`, and the stock diffusion pipelines (`StableDiffusion3Pipeline`, `QwenImagePipeline`, `Wan22Pipeline`) that the adapters subclass; sleep/wake/`collective_rpc` and bucketed `load_weights` are vLLM/vLLM-Omni mechanisms.
- **VeOmni (ByteDance-Seed)** — optional trainer backend wrapping `veomni.trainer.dit_trainer` + `parallel_state`, exposing combinable USP/TP/DP-replicate/DP-shard/EP, all resharded via the same `full_tensor()` path. (LoRA + USP not yet supported on this backend — `veomni/diffusion_impl.py:66-70`.)
- **FSDP2 (PyTorch `fully_shard`/`DTensor`)** — the default trainer backend, consumed via verl's `fsdp_utils` wrappers.
- **flow_grpo (yifan123/flow_grpo)** — adapted-with-attribution, **not** a dependency. The SDE-with-log-prob scheduler is vendored in-tree (`pipelines/schedulers/flow_match_sde.py:75` "Modified from …sd3_sde_with_logprob.py") and the FlowGRPO PPO loss is ported (`diffusion_algos.py:288-289`). README claims ~25% higher throughput than diffusers-based flow_grpo.
- **Algorithm papers, re-implemented in `diffusion_algos.py` registries**: FlowGRPO, DanceGRPO (same loss, different model/SDE), Flow-DPPO (exact-Gaussian-KL trust region), GRPO-Guard, Diffusion-DPO (adapted from diffusers `dpo/loss.py`), DiffusionNFT (NVlabs, arXiv:2509.16117 — implicit-negative-prediction trick), KL. GSPO is **not** here (lives in verl). diffusers + PEFT for model/scheduler/LoRA bases.

---

## 3. What vrl can learn

Grounded against the vrl spine (working dir `/home/mingfeiguo/Desktop/wm-infra/vrl`, branch `fp8-rollout-precision-tis`). vrl already matches verl-omni's request-scoped, on-policy, registry-driven shape on several axes — so several "learns" are *confirmations or gap-closures*, not new architecture.

### 3.1 Stream reward per-sample to a disjoint GPU pool (highest leverage)

**verl-omni does X** — after each sample finishes generation, its own asyncio task awaits a remote reward actor while other samples still denoise (`diffusion_agent_loop.py:297-300`):

```python
selected_reward_loop_worker_handle = random.choice(self.reward_loop_worker_handles)
result = await selected_reward_loop_worker_handle.compute_score.remote(data)
output.reward_score = result["reward_score"]
```

Overlap is only enabled when reward has its own GPU pool (`ray_diffusion_trainer.py:670` `enable_agent_reward_loop = not use_rm or reward_model.enable_resource_pool`); the trainer then skips its colocated reward path because `rm_scores` already arrived (`ray_diffusion_trainer.py:1000`). Policy update still waits for the full scored batch (`docs/algo/async_reward.md:51`).

**vrl today (post-reading update)** — the old Ray reward pool described by the original note was deleted. Rollout collectors now call the public `RewardRuntime.score(RewardRequest) -> RewardOutput` boundary. Model-backed components use either `InProcessRewardInferenceRuntime` or `HttpRewardInferenceRuntime`; overlap is admitted only when the runtime reports nonblocking execution and verified accelerator isolation. Shared-GPU in-process rewards deliberately remain serial.

**Borrow verdict updated.** The useful idea is the capability-gated overlap, not restoring a `RayRewardRuntime` pool. The current runtime boundary carries those capabilities without coupling rollout orchestration to a particular transport.

### 3.2 Make the rollout output schema algorithm-shaped, carrying the trajectory

**verl-omni does X** — the rollout→trainer contract is a tensor + optional `log_probs` + an `extra_fields` bag (`replica.py:20-31`), and what fills it **varies by algorithm**: FlowGRPO ships `{all_latents, all_log_probs, all_timesteps, prompt_embeds}` (`vllm_omni_rollout_adapter.py:423-437`), NFT ships only `{latents_clean, train_timesteps, prompt_embeds}` with no trajectory (`qwen_image_diffusion_nft/vllm_omni_rollout_adapter.py:258-267`). The trainer re-iterates whatever trajectory arrived (`diffusion_impl.py:442,459`).

**vrl today** — vrl's `EnginePlan` is a fixed envelope (`vrl/generation/execution/planner.py:98-111`: `sample_rows`, `chunks`, `execution_stages`, fixed `expected_axes`), and trajectory shape is driven by the family capability's axes, not by the algorithm. vrl already proves the trainer-re-iterates model: `DiffusionNFT.compute_loss` reads `latents_clean` and re-runs the transformer with positive/negative/reference forwards (`vrl/algorithms/diffusion_nft.py:161,236-267`).

**Borrow (partial).** The structural idea — let the rollout payload schema be *selected by the algorithm*, not forced into one fixed shape — is worth adopting for vrl's continuous-mode version-stamped groups so an NFT group carries `latents_clean` only while a GRPO group carries the full SDE trajectory. vrl is already most of the way there via per-family capabilities; the delta is keying the carried tensors on the algorithm (NFT vs flow-GRPO) rather than the family.

### 3.3 rollout_correction for fp8/bf16 rollout-vs-train log-prob drift (directly on the active branch)

**verl-omni does X** — `bypass_mode` reuses rollout log-probs as `old_log_probs` to skip the ~20% recompute pass (`rollout_correction.py:81` `batch.batch["old_log_probs"] = batch.batch["rollout_log_probs"]`), then corrects the induced off-policy bias with **IS** (clipped importance ratio) × **RS** (reject samples with extreme log-ratio), folded into one `rollout_is_weights` multiplier (`rollout_correction.py:151-159`). For `ppo_clip` the PPO ratio *is* the IS, so only the RS mask touches the gradient (`workers/utils/losses.py:50`). Diffusion has no padding mask, so rejection = weight 0. It logs off-policy diagnostics (KL, k3_kl, chi2, log_ppl_diff).

**vrl today** — vrl already started this independently and is arguably ahead on measurement. `vrl/algorithms/logprob_mismatch.py` is exactly the "bf16/fp8 rollout vs fp32 replay" drift toolkit: `compute_logprob_mismatch_stats` (`:40`) measures it and `apply_truncated_importance_weight` (TIS, `:110`) corrects it, with `PrecisionCorrectionConfig` living at `trainer.precision_correction` and injected into the algorithm.

**Borrow.** vrl has TIS (the IS half) and the drift metrics. What verl-omni adds and vrl lacks: (1) the **RS reject-sample mask** as a complement to truncation (drop extreme-ratio samples entirely rather than only clamp the weight); (2) the **bypass-vs-recompute split** (skip the recompute forward when drift is bounded — a direct FLOP saving for vrl's single-32GB GPU); (3) the diffusion-specific insight that with a short SDE window token-level RS is low-power, so prefer seq-mean/seq-max rejection. These are incremental additions to `logprob_mismatch.py`, not a rewrite. Honest caveat: verl-omni's IS/RS math (`compute_rollout_correction_and_rejection_mask`) lives in upstream verl; vrl would reimplement RS inline.

### 3.4 Transport weights over a bucketed side-channel, not the Ray object store

**verl-omni does X** — metadata-only RPC + tensors streamed bucket-by-bucket over a ZMQ IPC socket, bypassing Ray serialization (`engine_workers.py:918,923`), with a single resharding seam `get_per_tensor_param` collapsing any parallel layout to full bf16 (`diffusers_impl.py:762`).

**vrl today** — weight sync is a CPU state-dict round-tripped through the Ray object store: `RayGenerationWeightSync.push_to_rollout_workers` does `ray.put(state_ref)` then `update_weights.remote(shared_state, ...)` (`vrl/generation/ray/weight_sync.py:57-58`), landing in the worker as `model.load_trainable_state(state_ref)` (`vrl/generation/execution/worker.py:102`).

**Move (only if full-param sync becomes a bottleneck).** For LoRA the current `ray.put`-once-share-ref path is fine and already de-duplicates across workers (the comment at `weight_sync.py:51-56` shows they fixed the per-worker duplication). The ZMQ-IPC move matters only for the multi-GB full-param case the spine flags as a pain point. **Do NOT** copy this for LoRA — it would add a side-channel for no gain. Note vrl is *ahead* in one respect: it already supports **non-draining** weight sync via versioned trainable-state slots (`vrl/generation/execution/worker.py:42-47,78-84`, `vrl/rollouts/orchestration/continuous/schedule.py:123-133`), where verl-omni's colocated path always drains via sleep/wake.

### 3.5 Preserve frozen VAE/text-encoder across the rollout-memory release (correctness gap)

**verl-omni does X** — sleep is **level-1** specifically so non-trainable pipeline components (VAE, text-encoder) that are never in the weight sync get offloaded-and-restored, not discarded (`vllm_omni_async_server.py:204-210`).

**vrl today** — `release_rollout_runtime_memory` is `empty_cuda_cache()` plus an optional `release_runtime_memory()` hook (`vrl/rollouts/orchestration/lifecycle.py:117-125`), and the driver-model offload is a whole-model `self.model.to("cpu")` (`lifecycle.py:108`). MEMORY also notes `ReleasableRayGenerationRuntime` kills+relaunches actors per cycle (cold model reload). Whether vrl's release path *discards* the frozen VAE/text-encoder and pays to reload them is **unverified** from the spine alone — `release_runtime_memory` is a getattr hook whose body lives in the family runtime, which I did not read.

**Borrow (verify first).** If vrl's per-cycle release/kill drops the frozen encoder/VAE, adopt verl-omni's discipline: offload-and-restore the non-trainable components rather than discard+reload. The gap is real in shape (whole-model `to("cpu")` + actor kill-and-relaunch is the full-discard pattern verl-omni explicitly avoids), but confirming the cost requires reading the family runtime's `release_runtime_memory`.

### 3.6 Registry seams (already aligned — confirmation, not borrow)

verl-omni composes "model × algorithm" from config strings via two tiny registries — `(architecture, algorithm)` for the training adapter and the rollout adapter (`pipelines/model_base.py:51-55`), plus `@EngineRegistry.register(model_type, backend=[...])` + `EngineRegistry.new(model_type, backend=strategy)` for the trainer backend (`engine_workers.py:153`, `diffusers_impl.py:814`). vrl already uses the same shape: `train.py` dispatches by `trainer.entrypoint` import path (`vrl/scripts/train.py:32,46-47`), and families live under `models/{ar,diffusion}/<family>/`. **No move needed** — this validates vrl's existing design. The one idea worth lifting is the **loss-validates-its-own-inputs** contract (`required_model_output_keys`/`required_data_keys` failing fast with available-vs-missing diagnostics), which vrl's algorithm layer does not yet enforce.

### 3.7 Implicit-negative + previous-policy adapter (already done — do NOT re-derive)

verl-omni's DiffusionNFT computes the negative branch algebraically — `implicit_negative_prediction = (1+beta)*old - beta*forward` — to avoid a 4th forward (`diffusion_algos.py:812-846`), and maintains the "old" policy by copy/EMA of a LoRA adapter. **vrl already does both**: `vrl/algorithms/diffusion_nft.py:252-253` is the identical `negative_prediction = (1.0 + beta) * previous_prediction - beta * forward_prediction`, and `after_optimizer_step` refreshes the previous-policy adapter via `sync_previous_policy_adapter(decay=...)` (`vrl/algorithms/diffusion_nft.py:299-309`). Nothing to borrow here.

---

## What NOT to copy

- **Hard pip-pin to upstream internals.** verl-omni imports deep private paths (`verl.workers.engine.base.BaseEngine`, `verl.utils.fsdp_utils` internals) and pins exact versions. That brittleness is the cost of dependency-over-fork; vrl owns its trainer/rollout core and should keep it.
- **ZMQ-IPC weight transport for LoRA.** vrl's `ray.put`-once-share-ref path (`weight_sync.py:57-58`) is already correct and de-duplicated for the LoRA-heavy default; a side-channel adds complexity for no gain there. Reserve it for the multi-GB full-param case only.
- **Always-draining sleep/wake barrier.** verl-omni's colocated path always drains rollout via sleep before training; vrl's versioned-slot non-draining sync (`continuous/schedule.py:123-133`) is strictly more capable — keep it.
- **Implicit-negative / previous-policy adapter.** Already implemented in vrl's NFT — re-porting verl-omni's version would be churn.

## Suggested priority order

1. **Capability-gated reward overlap** (§3.1) — incorporated through the public `RewardRuntime`; do not restore the removed Ray reward pool.
2. **Extend `logprob_mismatch.py` with RS rejection + bypass-vs-recompute** (§3.3) — directly serves the active `fp8-rollout-precision-tis` branch; vrl already owns the TIS + drift-metrics half.
3. **Verify then fix frozen-encoder/VAE preservation on release** (§3.5) — a correctness trap if the per-cycle release discards non-trainable components; cheap once verified.
4. **Algorithm-shaped rollout payload + loss-validates-its-inputs contract** (§3.2, §3.6 tail) — structural tidiness that pays off as more algorithms land; not urgent.
