# SPRINT PROGRAM: VRL implementation and research backlog

Status: **planned (2026-10-24)**. This is the entry point for the requested sprint dump. It schedules actionable work from the conference review and connects the existing repository backlog. Each conditional item states its trigger; inclusion does not mean every experiment should ship.

Repository snapshot checked: `c090e97db`. Source code and current recipes win over historical statements in older plans. No implementation, training campaign, package upgrade or cloud machine launch is included in this documentation change.

## 1. Start here

1. **PT26-01: fix the middle-frame semantic bug.** It has a concrete reproducible code path and can proceed independently.
2. **PT26-00: establish comparable optimization acceptance.** Reuse the forward probe and actual trainer; add the missing backward/update and fail-closed verdict coverage.
3. **PT26-02: profile and reuse judge prefixes across chunks.** Within-chunk sharing already exists. Proceed with implementation only if chunked workloads justify it.
4. **PT26-03 and PT26-04: qualify parity and existing acceleration.** Diagnose the actual first divergence and measure real training; do not assume enabling existing features is new work.
5. Pick **one measured bottleneck** from checkpointing, accumulated-gradient communication or weight installation/transport. Keep compiler-cache, training-quantization and kernel trials conditional.

These are dependency waves rather than calendar promises. A sprint produces a verified change or an explicit go/no-go decision; GPU/topology availability and the measured workload determine the duration. Do not run all hardware experiments concurrently on a shared GPU.

## 2. Execution queue

| ID | Priority / scope | Concrete next deliverable | Dependency / unblock event |
|---|---|---|---|
| [PT26-00](SPRINT_pt26_00_optimization_acceptance.md) | P0, acceptance | Recipe-pinned baseline/candidate contract; real backward/update comparison; fail-closed verdict | CPU work now; real GPU for integration qualification |
| [PT26-01](SPRINT_pt26_01_reward_frame_decode.md) | P0, correctness then media | Correct middle frame; separate indexed-decoder trial | Semantic fix now; decoder only after measured cost |
| [PT26-02](SPRINT_pt26_02_reward_prefix_chunks.md) | P1, reward throughput | Immutable per-call prefix plus independent chunk caches | Real multi-chunk judge profile; PT26-00 for full-step result |
| [PT26-03](SPRINT_pt26_03_targeted_lora_parity.md) | P0 diagnosis, conditional kernel | Nonzero-LoRA first-divergence reproduction and targeted remedy | Existing forward probe now; PT26-00 for gradients/update |
| PT26-04, below | P1, existing capability qualification | Production quality/performance matrix for FP8/NVFP4/FSDP2/compile+AC | PT26-00 and relevant NCCL hardware |
| [PT26-05](SPRINT_pt26_05_selective_lowering.md) | P1 experiment | Compare selective lowering with eager/root/regional compile | PT26-03 identifies a relevant region; pinned compiler APIs |
| [PT26-06](SPRINT_pt26_06_checkpoint_staging.md) | P1 measurement, conditional implementation | Save-stall breakdown; immutable async staging if justified | Measured exposed save stall; PT26-00 restore comparison |
| [PT26-07](SPRINT_pt26_07_collection_gradient_sync.md) | P1 experiment | Backend-specific sync deferral with equivalent update | Multiple collections/update and exposed communication |
| [PT26-08](SPRINT_pt26_08_weight_install_effect.md) | P1 correctness; P2 transport | Two-payload live-weight/output test; measured transport decision | Tests now; alternative transport only after real stall profile |
| [PT26-09](SPRINT_pt26_09_distributed_compile_cache.md) | P2 experiment | Signed artifact manifest and single-writer publication | Duplicate distributed cold compile remains material |
| [PT26-10](SPRINT_pt26_10_training_quantization.md) | P2 research | One-block forward/dX pilot with master-weight/export contracts | Isolated compatible environment, replay bottleneck, PT26-00 |
| [PT26-11](SPRINT_pt26_11_profiled_graphs_kernels.md) | P3 experiment | One trace-justified graph, fusion or tuning candidate | Fresh production profile; previous graph no-go is not overturned by a toy shape |
| [PT26-12](SPRINT_pt26_12_public_material_followup.md) | P2 evidence | Work-ID-specific primary-material gap resolution | New primary release or concrete untried route; independent of implementation |

### PT26-04 — qualify existing precision and distributed paths

Owning plans: [rollout/reward throughput](SPRINT_rollout_reward_throughput_pt26.md), [FSDP compile](SPRINT_fsdp_replay_compile_unblock.md) and [parity](SPRINT_rollout_replay_parity_root_cause.md). Keep implementation and status changes with those owners; do not open duplicate feature work.

- [ ] Refresh the support matrix against current recipes and target hardware: BF16 reference, FP8 rollout, NVFP4/four-over-six rollout, current FSDP2, root/regional compile and AC/SAC. Test supported combinations explicitly; this is not a claim every Cartesian combination works.
- [ ] Use identical initial trainable states, prompts, seeds, effective global batch and evaluation scorers/weights. Report forward/backward/gradient-accumulation/optimizer dtypes separately.
- [ ] Measure full accepted-update throughput, allocated/reserved peaks, exposed communication, per-step logprobs, pre-update clipping, TIS/RS, gradients/update and repeated held-out/multi-seed reward. Separate cold cost from steady state and lag-zero error from staleness.
- [ ] Run real NCCL and representative model-duration trials after small tests; CPU/gloo success and reconstruction-MSE improvements cannot establish long-run RL quality. Record the remaining supported/unsupported combinations and a default/fallback decision.

Completion means a reproducible qualified matrix with explicit quality gates and go/no-go decisions. Historical speed/quality numbers in older plans are workload-specific evidence, not targets or current results.

### New distributed capabilities have separate triggers

**HSDP and sharded DCP** belong to [multi-node FSDP](SPRINT_multinode_fsdp_hsdp_dcp.md). Current 1D FSDP and CP do not implement HSDP. Trigger: an actual multi-node workload/topology and measured scaling or checkpoint limitations. Deliver mesh/config/launch design, implementation, rank-order and failure checks, equivalent update/restore and real multi-node NCCL scaling. Changing world size requires an explicit RNG/data-order resume contract; it is not an incidental benefit of DCP.

**Cross-rank CFG** belongs to [rollout/reward throughput T4](SPRINT_rollout_reward_throughput_pt26.md). Ordinary CFG already exists; parallel CFG is new work. Trigger: a two-forward CFG family with a measured multi-GPU rollout bottleneck. Deliver request/RNG/rank ownership, cond/uncond communication, cancellation/error handling and equivalence across seeds/batches, followed by net throughput/memory comparison including the extra GPU. Do not assume identical arithmetic or useful scaling before testing.

## 3. Existing repository portfolio

The table below routes the rest of what we can build to existing plans. It is navigation, not a declaration that every historic task is still pending. Before starting, identify the remaining unchecked action and refresh its code evidence. Completed subfeatures do not become new sprints merely because they appear in a conference talk.

| Workstream | Existing owner plans | Selection gate / intended outcome |
|---|---|---|
| Continuous execution | [three-stage pipeline](SPRINT_continuous_three_stage_pipeline_program.md), [finalize overlap](SPRINT_rollout_finalize_overlap_ga.md) | Complete the owner's ordered stages and failure tests before a long GA run; prove accepted-update throughput and bounded staleness |
| Scheduler/admission | [admission and failure attribution](SPRINT_rollout_admission_and_failure_attribution.md), [Ray dispatch](SPRINT_ray_actor_dispatch_scalability.md), [adaptive sampling](SPRINT_ngu_adaptive_sampling.md) | Attribute rejected/failed groups and demonstrate a real scheduling or sample-efficiency bottleneck before optimization |
| Weight/memory placement | [Wan 2.2 placement](SPRINT_wan22_grpo_throughput_weight_placement.md), [weight transport](SPRINT_weight_sync_transport_seam.md) | Measure PCIe/CPU snapshot/transport/install critical path; PT26-08 strengthens installation evidence |
| Parallel training | [video CP](SPRINT_video_context_parallel.md), [multi-node FSDP/HSDP/DCP](SPRINT_multinode_fsdp_hsdp_dcp.md), [FSDP compile](SPRINT_fsdp_replay_compile_unblock.md) | An available model/topology and an identified memory/scaling limit; pass distributed update/restore gates |
| Performance and stack | [throughput](SPRINT_rollout_reward_throughput_pt26.md), [Triton ledger](SPRINT_triton_map_and_perf_ledger.md), [upgrade readiness](SPRINT_torch_vllm_ray_upgrade_readiness.md) | New measurements or a deliberate version-upgrade decision; revalidate private APIs in an isolated environment |
| Reward/data quality | [reward qualification](SPRINT_reward_qualification.md), [production reward/data](SPRINT_production_reward_and_rl_data.md), [verifiable environments](SPRINT_verifiable_visual_environments.md) | Establish scorer validity, failure modes, calibration and held-out validity before optimizing reward throughput |
| Evaluation and teaching | [eval provenance](SPRINT_miles_eval_provenance.md), [multi-objective routing/teachers](SPRINT_multi_objective_reward_routing_and_teachers.md), [industry gaps](SPRINT_industry_post_training_gap_backlog.md) | Trace results to the correct checkpoint/scorer/data and qualify each proposed objective/teacher independently |
| Visual editing products | [general edits](SPRINT_qwen21_general_edit_rl.md), [accessory edits](SPRINT_qwen21_accessory_edit_rl.md), [text replacement](SPRINT_qwen21_text_replace_edit_rl.md), [production visual/text tasks](SPRINT_production_visual_editing_and_text.md) | Select one task/data/evaluator contract and complete its proof run before broader product claims |
| Episodes and video proof | [Qwen visual episodes](SPRINT_qwen_visual_episodes.md), [Wan I2V proof run](SPRINT_wan_2_1_i2v_proof_run.md) | A bounded task, credit assignment and verifiable evaluation; owner-defined hardware and data prerequisites |
| Contracts and development | [core hardening](SPRINT_core_contract_hardening_program.md), [fail-closed protocols](SPRINT_protocol_contracts_fail_closed.md), [dev guardrails](SPRINT_agent_dev_guardrails.md), [test cleanup](SPRINT_test_cleanup_night.md) | A named contract failure or repeated development problem; meaningful deterministic checks rather than LOC-driven cleanup |
| Historical benchmark disposition | [retired overlap benchmark plan](SPRINT_reward_overlap_benchmark_evidence_contract.md) | Explicitly marked obsolete after benchmark deletion. Do not schedule resurrection; use PT26-00 for current acceptance and reconcile/delete the stale plan when its owner handles lifecycle cleanup |

## 4. Shared implementation and acceptance rules

- Start CPU-possible work immediately: frame semantics, comparison schema/fixtures, cache mutation tests and two-payload installation checks. Hardware-dependent acceptance remains pending until executed; record resource requirements without launching AWS instances as a side effect of this plan.
- Give each experiment a baseline, one changed factor where feasible, explicit quality/performance tolerances and a stop condition before running it. Report both isolated component cost and full-step outcome, including retries/OOMs, installation and startup costs. Do not invent speedup targets from conference headlines.
- Keep existing prompt reuse, within-chunk judge prefix sharing, NVFP4 calibration, stage timing, media handoff and persistent Inductor cache as the baseline. Extend a missing boundary only after re-reading current code; do not reimplement finished work.
- Preserve checkpoints, state keys, effective policy versions, scorer provenance, recipe parsing, RNG, parking, admission and failure behavior. Hashes/ACKs alone do not prove numerical or policy equivalence.
- Reuse owner tests and add cases only for a meaningful invariant or failure mode. Documentation-only changes need link/path/diff validation; implementation uses targeted tests followed by the stated real-hardware gates.

### Architecture: changes, preserved boundaries and non-goals

**Change when encountered:** if new business vocabulary, prompt templates or backend/domain tables would become large module-level ALL_CAPS data mixed into workflow code, put that data in a clearly named domain/config asset with one owner. Remove a thin helper only when it has no protocol, reuse or readability purpose and its removal simplifies the actual call graph.

**Keep:** schema/protocol keys, environment-variable names, checkpoint/file names, model architecture dimensions, test fixture constants and deliberately isolated config/taxonomy tables are valid constants. Strategy methods, family adapters, public facades, lazy-import seams, framework callbacks and meaningful test fakes remain separate when they enforce a boundary or keep cross-family behavior consistent. A named prefix/cache-lifecycle helper is useful if it makes ownership explicit; a pass-through helper with no such role is not.

**Non-goals:** no broad flattening, renaming or constant migration to reduce LOC; no loss of uniform family shapes for a tiny local simplification; no automatic backend replacement, Megatron migration, package upgrades, external outreach or speculative platform build. Review the exact function/constant and state its reason before any cleanup.

## 5. Sources and lifecycle

The local review inputs are `~/Desktop/pytorch2026-vrl-review/recommendations.json`, `integration-decisions.json`, the implementation/prefix/kernel briefs and `deeper-audit.json`. They are external research context, not a portable repository dependency. Each focused plan carries the source/code evidence necessary for its action; raw papers and review archives remain outside this tree.

Coverage of the ten actionable review recommendations is explicit: **1 → PT26-01; 2 → PT26-03; 3 → PT26-04 plus the HSDP/CFG owner triggers; 4 → PT26-05; 5 → PT26-06; 6 → PT26-07; 7 → PT26-08 and transport owner; 8 → PT26-09; 9 → PT26-10; 10 → PT26-11.** PT26-00, PT26-02 and PT26-12 cover acceptance, the newly identified prefix gap and bounded materials follow-up.

Use the [sprint lifecycle](../README.md): only executable active plans live here. When a focused sprint's final action lands, remove its file and replace this index link with the remaining owner's plan or remove the completed row in the same commit. Keep only measurements that support a current decision; do not append completed-run archives or DONE banners. A conditional experiment can finish with a no-go decision; unresolved hardware or missing evidence is not a passing result.

## 6. Replay 重叠执行试点（2026-10-29）

[Forward/backward 重叠与延后 dW](SPRINT_replay_forward_backward_overlap.md) 记录本次 DeepSeek 调度借鉴的独立实现、命令和硬件验收门槛。128px/512px CUDA 更新一致性通过，但独占性能与 trace 未证明可采用的净收益，因此不进入生产集成。它衔接 PT26-00 验收和 PT26-07 通信测量，保留原 owner。
