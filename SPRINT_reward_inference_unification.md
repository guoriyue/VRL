# Sprint: Unify Reward Model Inference

## Context

Train and rollout run the **policy** model through one path: the generation executor
pipeline (`PipelineExecutor` + `FAMILY_REGISTRY`) on the shared Ray substrate
(`vrl/ray/`). They share that path because they share the *same trainable weights*
(weight-sync, offload, policy-version tracking).

Reward models are different: they're **frozen, independent networks** that share no
weights with the policy, so they legitimately can't (and shouldn't) join the policy
pipeline. The real problem is that reward inference today has **two divergent contracts**
and ad-hoc per-reward glue ("patches"):

1. **In-process** `RewardFunction` (`vrl/rewards/base.py`): `async score(rollout)->float`,
   takes an in-memory `RewardRolloutExample`. Each subclass hand-rolls its own model loading
   (`from_pretrained().eval().to(device,dtype)`, `torch.load`, ONNX session, subprocess, …).
2. **Ray** `RewardModel` (`vrl/rewards/ray/model.py`): `__call__(*, artifact, request)
   -> Mapping[str,float]`, takes a *materialized* `RewardInferenceArtifact` (disk path),
   hosted by `RewardModelWorker` on `RayActorMethodRuntime` (which already wraps the shared
   `RayActorGroup` — `vrl/ray/runtime.py:89`).

Only `video_reward` bridges the two (it's a `RewardFunction` that materializes artifacts and
drives the Ray path internally). Consequences: defining a reward means picking a contract up
front; **only `video_reward` can run on Ray** (GPU-heavy rewards like pickscore/aesthetic are
stuck in the trainer process); and there's no shared model-loading helper anywhere.

**Intended outcome:** one way to define a reward model, runnable **in the current process or on
the shared Ray pool by config** (`inference_runtime: direct|ray`), plus a shared model-host that
kills the hand-rolled loading for the torch-nn rewards. Generation is untouched.

## Current state (grounding)

- Reward categories (migration is NOT one-size-fits-all):
  - **Torch nn on a device** (host helper applies): `aesthetic`, `clip`, `pickscore`, `nsfw_safety`
  - **Subprocess/CLI**: `codex_image_qa`, `ocr` (paddle)
  - **Dynamic import callable**: `geneval`
  - **ONNX**: `anime_anatomy`
  - **Ray**: `video_reward` (→ `KlingVideoRewardModel`)
- Inference contracts (`vrl/rewards/inference.py`): `RewardInferenceArtifact{artifact_id, path,
  media_type, prompt, sample_id, policy_version, metadata}` (path = on-disk),
  `RewardInferenceRequest{request_id, artifacts, reward_name, score_key, score_aggregation,
  policy_version, metadata}`, `RewardInferenceResult{…, scores, selected_score, …}`; helpers
  `select_score`, `shard_reward_request`, `validate_reward_results`.
- Wiring: `vrl/scripts/common/factory.py:build_reward_from_cfg` → `MultiReward.from_dict(weights,
  device, reward_kwargs)` → `reward_cls(device=…, **kwargs)`. Consumed by
  `RewardScorer` (`vrl/rollouts/collector/rewards.py`). `MultiReward`/`RewardScorer` only know
  the `RewardFunction` ABC (`score`/`score_batch`).
- Blast radius: ~13 `tests/rewards/*` files; ~4 core callers. No shared model-load helper exists.

## Target design

Keep `RewardFunction` (`rollout -> float`) as the **stable public API** (so `MultiReward` /
`RewardScorer` are untouched). Restructure what sits *under* it into three clean pieces:

1. **Canonical reward unit = `RewardModel`** (`inputs -> named scores`), with the existing
   `model_factory(worker_config) -> RewardModel` convention. This is the "define a reward once"
   surface; it already exists worker-side (`vrl/rewards/ray/model.py`).

2. **`RewardInput`** — generalize today's `RewardInferenceArtifact` so it carries an **in-memory
   media tensor OR a materialized path** (+ prompt + metadata). Lazy helpers `as_media()` /
   `as_path(dir)` convert on demand. This avoids forcing disk writes on direct torch rewards
   (the common case) while letting file-based models (Kling) materialize lazily.

3. **`RewardRuntime` (transport) protocol** with a uniform `score(request) -> [results]`.
   Two impls, selected by the `inference_runtime` config value:
   - **`direct`** → `DirectRewardRuntime` (**new**): run the `RewardModel` directly in the
     current Python process, without Ray actors. Builds the model lazily, calls it on in-memory
     inputs. No Ray, no disk.
   - **`ray`** → `RayRewardRuntime`: run the `RewardModel` through Ray reward actors with explicit
     resource placement — the existing `build_reward_actor_runtime` + `score_reward_request`
     (materializes, shards across actors).

4. **`HostedReward(RewardFunction)`** — generalize `VideoReward` into a transport-agnostic
   driver: builds `RewardInput`s from rollouts → dispatches to the chosen `RewardRuntime` →
   returns `selected_score`. `inference_runtime: direct|ray` picks the transport. `VideoReward`
   becomes `HostedReward` + the Kling factory (back-compat).

5. **`TorchRewardModel`** — shared base for the torch-backed nn rewards, replacing
   per-reward device/dtype/lazy-load boilerplate.

Net: a reward = one `RewardModel` + a `model_factory`; it runs direct or ray by config; nn
rewards share one loader. Heterogeneous rewards (CLI/ONNX/import-path) still implement
`RewardModel` for their *scoring* but keep their own loaders.

## Key decisions

- **Transport naming = `direct` (not `local`).** `direct` names the execution boundary (run in
  the calling process, no Ray actors) and avoids `local`'s ambiguity (local machine / local GPU /
  CPU-only / debug mode). Defined once, used everywhere.
- **In-memory inputs for the direct transport** (don't materialize to disk just to score
  in-process). Only models that intrinsically need a file (Kling video) call `as_path()` to
  materialize lazily.
- **`RewardFunction` public API stays.** `HostedReward` *is* a `RewardFunction`, so `MultiReward`
  composes it transparently — no change to `MultiReward.from_dict` callers/signatures.
- **`select_score` / `score_key` semantics are transport-independent** — both runtimes return
  `RewardInferenceResult`; selection logic stays in one place.
- **Default `inference_runtime: direct`**; `ray` is opt-in per component (for GPU offload).

## Phased plan (each phase independently shippable & green)

**MVP = P1–P3** (delivers the mechanism + one proof on each side). P4–P6 are incremental.

- **P1 — Input + transport seam.** Generalize `RewardInferenceArtifact` → `RewardInput`
  (in-memory media + lazy `as_path`/`as_media`); extract a `RewardRuntime` protocol; make the
  driver (`score_reward_request`) transport-agnostic. No behavior change; `video_reward` still
  works on Ray. Files: `vrl/rewards/inference.py`, `vrl/rewards/ray/runtime.py`.
- **P2 — Direct transport.** Add `DirectRewardRuntime` (new `vrl/rewards/runtime/direct.py`)
  that builds a `RewardModel` via the same `model_factory` and calls it in the current process.
  Parity test: same fake `RewardModel` returns identical scores direct vs ray.
- **P3 — Generalize the bridge + prove nn path.** Extract `HostedReward` (new
  `vrl/rewards/hosted.py`) from `VideoReward`; rebuild `VideoReward` as `HostedReward` + Kling
  factory (config back-compat). Add `TorchRewardModel`
  (`vrl/rewards/models/base.py`) and migrate **`pickscore`** as the representative nn
  reward (→ `RewardModel` + factory, run via `HostedReward(direct)`). Proves both transports +
  the host helper end-to-end.
- **P4 — Migrate remaining nn rewards:** `aesthetic`, `clip`, `nsfw_safety` (same pattern as P3).
- **P5 — Migrate heterogeneous rewards** to `RewardModel` (scoring only; keep their loaders):
  `ocr`, `codex_image_qa`, `geneval`, `anime_anatomy`. Optional / can defer.
- **P6 — Config/factory wiring.** Thread `inference_runtime: direct|ray` per component through
  `build_reward_from_cfg` / `MultiReward.from_dict` (default `direct`); add the field to
  `configs/reward/*.yaml` where relevant.

## Non-goals (explicit boundaries)

- Do **not** touch the generation pipeline, `PipelineExecutor`, or `FAMILY_REGISTRY`.
- Do **not** merge the reward worker with `GenerationWorkerCore` (the "maximal" option) — the
  generation worker carries policy weight-sync that reward doesn't need.
- Do **not** force CLI/ONNX/import-path rewards through the torch model-host.
- Do **not** change `RewardFunction` / `MultiReward` / `RewardScorer` public signatures.

## Risks & mitigations

- **Materialization cost** for direct torch rewards → mitigated by in-memory `RewardInput`; only
  file-needing models materialize lazily.
- **video_reward config back-compat** (`reward_name`, `worker_config.model_factory` derivation,
  `score_key`, `artifact_format`) — `VideoReward` must remain a drop-in; keep a config/alias shim
  and cover with the existing `tests/rewards/test_video_reward*.py`.
- **Blast radius** — ~13 reward test files; migrate one reward at a time, keep suite green per phase.
- **Score parity** — direct vs ray must produce identical `selected_score`; enforce with a parity test.

## Verification

- `tests/rewards/` stays green after every phase (esp. `test_video_reward*.py`, `test_multi.py`).
- New tests: `DirectRewardRuntime` unit; `HostedReward(direct)` output parity vs the pre-migration
  `RewardFunction` for `pickscore`; a fake-`RewardModel` parity test (direct == ray scores).
- Integration: run `build_reward_from_cfg` on an experiment config with `inference_runtime: direct`
  and again with `ray`; confirm identical selected scores on a small batch.
- `ruff check` clean on all touched files.

## Critical files

- `vrl/rewards/inference.py` — `RewardInput` generalization, transport-agnostic driver.
- `vrl/rewards/ray/{runtime,worker,model}.py` — `RewardRuntime` protocol; `RayRewardRuntime` conforms.
- `vrl/rewards/runtime/direct.py` *(new)* — `DirectRewardRuntime`.
- `vrl/rewards/hosted.py` *(new)* — `HostedReward`; `vrl/rewards/functions/video_reward.py` rebuilt on it.
- `vrl/rewards/models/base.py` — `TorchRewardModel`.
- `vrl/rewards/functions/{pickscore,aesthetic,clip,nsfw_safety}.py` — migrate to `RewardModel`.
- `vrl/scripts/common/factory.py`, `vrl/rewards/functions/registry.py` — transport wiring.
- `configs/reward/*.yaml` — `inference_runtime` field.
- `tests/rewards/*` — parity + direct-runtime tests.
