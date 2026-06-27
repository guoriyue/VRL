# SPRINT: Speculative Diffusion Rollout — target-exact acceleration for RL trajectories

状态：**ABANDONED — exact-coupling 路线确认死亡（2026-06-25）**。实现代码
（`vrl/math/diffusion/speculative.py` + 单测）已删除：whole-latent 单步 accept/reject
被高维 maximal coupling 的方差墙击穿（接受率≈0），这是基本规律不是调参问题，不再投入。
保留为**负结果档案**：本 sprint 文档 + reading note（实测数字记在 §9 P3）。
探针 `speculative_draft_probe.py` 已删除——它回答的问题已记录在这里。

---

（历史状态）**P0–P2 done / P3 measured = NEGATIVE / P4–P5 gated-off（2026-06-24）**。
**P3 实测在真实 SD3.5 上 acceptance≈0、省算 0%**：whole-latent 单步 speculative coupling
被 latent 高维（sqrt(D)≈512）的方差墙击穿，frozen-target 和 previous-policy draft 都过不去
（详见 §9 P3，实测数字已记录；探针已删除）。toy（D=8）严重高估。
P4/P5 因此不跑。下面 P0–P2 仍是有效资产（exact-coupling 数学 + RL contract 已证），
但「省 target NFE」这个 killer 诉求在 exact-coupling 路线上**当前不成立**。

已落地：reading note
`docs/sprints/reading/speculative_diffusion_sampling.md`（P0）、核心模块
`vrl/math/diffusion/speculative.py`（reflection maximal coupling +
`speculative_sde_step`，P1/P2）、CPU 证明测试
`tests/math/test_speculative_diffusion.py`（10 passed）。关键结论：flow-matching
单步是“同协方差高斯对”，speculative sampling 退化成 **reflection maximal
coupling**——输出精确等于 target 转移、不需 residual 采样、`old_log_prob` contract
干净（= target 转移 logprob，可 replay 复算）。toy 实测在分布统计检验通过的前提下
**省 ~72% 串行 target 评估**（draft 贴近 target 时）。P3+ 需要 GPU + 真模型，未在本轮跑。

这份 sprint 回答一个更严格的问题：能不能像 speculative decoding 一样，用便宜 draft 先猜，再用 expensive target policy 校正，从而 **保持 target rollout distribution 不变**，但减少 target diffusion model 的 function evaluations。

这比 `SPRINT_signal_paged_rollout` 更接近“PagedAttention 级别”的诉求：不是靠改变采样相关性省算，而是用接受/拒绝校正保留目标分布。它仍然不是 bitwise cache；它保证的是 **same target distribution / same target policy semantics**，不是“同一个 seed 下逐 tensor 完全相同”。

## 0. 一句话

LLM speculative decoding 的结构是：

```text
cheap draft model 先生成多个候选 token
expensive target model 并行 verify
accept / reject / residual sample
最终输出分布 == target model 原本的输出分布
```

diffusion rollout 的对应版本是：

```text
cheap draft transition / draft path 先提出 denoise 候选
expensive target policy verifier 校正候选 transition/path
accepted path 的分布 == target diffusion policy 的采样分布
同时输出 RL 需要的 behavior logprob / timestep / trajectory fields
```

目标不是“近似快一点”，而是：

```text
减少 target DiT evaluations
保留 target rollout distribution
保留 RL 训练所需的 behavior trajectory contract
```

## 1. 先讲清 exactness：不是 bitwise exact，是 distribution exact

Speculative decoding / sampling 的“exact”通常指：

```text
算法输出的随机变量分布与 target sampler 相同
```

不是：

```text
同一个 random seed 下，每个中间 tensor 都和 serial target sampler bitwise 相同
```

这个区别很重要：

- 如果要求 **bitwise same trajectory for the same seed**，diffusion 不存在明显的大头 compute reuse。不同 latent 输入必须重新跑 target forward。
- 如果接受 **same target distribution**，speculative sampling 才有空间：draft 可以猜，target 用概率校正，最终分布仍是 target。

对 RL 来说，distribution exact 通常是足够的，因为 rollout 本来就是随机采样；但必须记录：

```text
behavior_policy_version
target transition logprob / old_log_prob
acceptance metadata
trainable mask / timestep fields
```

否则 trainer 不能知道这条 trajectory 到底对应哪个 behavior policy。

## 2. LLM speculative decoding 的机制

以 autoregressive target model `p` 和 cheap draft model `q` 为例：

```text
1. draft q 连续生成 K 个 candidate tokens
2. target p 对这些候选位置做并行 verification
3. 对每个 token 按 p/q 的接受概率保留
4. 一旦 reject，就从 residual distribution 采样一个 token，然后停止接受后续 draft
```

关键性质：

```text
最终 token 分布仍等于 target p
target model 用一次并行 verification 覆盖多个候选 token
```

系统化版本如 SpecInfer 会把候选组织成 token tree，让 target verifier 一次验证更多分支。

## 3. diffusion speculative sampling 的对应物

diffusion / flow rollout 是连续 Markov chain：

```text
x_T -> x_{T-1} -> ... -> x_0
```

每一步 target transition 可以抽象成：

```text
p_t(x_{t-1} | x_t, prompt, policy)
```

draft transition 是：

```text
q_t(x_{t-1} | x_t, prompt, cheap_draft)
```

speculative diffusion sampling 的核心是：用 `q_t` 或一段 draft path 先提出候选，再用 `p_t` 做接受/拒绝校正，使最终 Markov chain 的分布仍然是 target chain。De Bortoli et al. 2025 把 speculative sampling 从离散序列推广到连续 vector-valued Markov chains，并报告在保持 target exact samples 的同时减少 diffusion model function evaluations。

## 4. 为什么这比 shared-prefix 更符合“killer capability”

| 方法 | 是否保持 target distribution | 省的是 target compute 吗 | 对 RL 语义的影响 |
|---|---:|---:|---|
| shared prefix / TreeGRPO | 否，会改变 group 相关性 | 是 | algorithmic tradeoff |
| TeaCache / DeepCache | 否，近似 reuse | 是 | rollout-vs-replay drift |
| fp8 / kernel patch | 否，数值近似 | 单次 forward 更快 | drift guard / TIS |
| speculative diffusion | **是，若校正正确** | **目标就是少 target NFE** | 可保留 target policy semantics |

所以如果标准是“不要改 rollout distribution，只少算 expensive target”，speculative diffusion 比 shared-prefix 更像正确方向。

## 5. VRL 的独特点：不是发明 paper，而是做 RL-ready speculative rollout

Speculative decoding/sampling 已经有 paper。VRL 值得做的不是“我也实现一个 sampler”，而是补上 paper 通常没有解决的 RL 系统面：

```text
target policy 每个 train step 都变
draft policy 可能是 previous policy / EMA / cheap solver / smaller model
rollout 输出不是 final media，而是 trainable trajectory
trainer 要 old_log_prob / prev_sample_mean / timestep / policy_version
reward 需要 decoded artifact
Ray rollout workers 要和 weight sync / staleness contract 对齐
video rollout 要处理巨大的 latent / VAE / reward payload
```

这才可能是我们的 capability：

```text
Speculative decoding : LLM serving
Speculative diffusion rollout : video/world-model RL trajectory generation
```

## 6. Draft choices

候选 draft 不止一种。按最值得试的顺序：

### 6.1 Previous-policy draft

RL 天然有 previous policy / stale rollout policy：

```text
draft = previous rollout worker weights
target = current rollout policy
```

如果每步 policy update 很小，draft 和 target 接近，acceptance rate 可能高。这是 RL 独有优势。风险是 draft 不能变成 behavior policy；必须由 target 校正，并记录 target behavior logprob。

### 6.2 Cheap solver / coarser transition draft

不用训练小模型，用便宜 deterministic/low-order solver 预测候选 transition，再 target 校正。De Bortoli et al. 提到有 out-of-the-box draft strategy，不需要训练 draft model。这个最适合先做 proof。

### 6.3 Lower precision / quantized draft

draft 用 fp8 / cheaper attention / approximate cache；target 用 exact rollout policy verify。这个比直接用 fp8 rollout 更干净，因为最终分布由 target 校正。

### 6.4 Smaller distilled draft model

最像 LLM speculative decoding，但成本最高：需要维护小模型或 distilled denoiser。先不作为 MVP。

## 7. RL trajectory contract

当前 diffusion trajectory 是 dense：

```text
observations
actions
old_log_prob
timesteps
kl
optional old_prev_sample_mean
replay_tensors
```

speculative rollout 必须额外记录：

```text
target_policy_version
draft_policy_version / draft_kind
accepted_mask
target_nfe
draft_nfe
acceptance_rate
residual_sample_count
target_old_log_prob
```

关键原则：

```text
trainer 只消费 target-policy behavior facts
draft facts 只用于 debugging / acceptance metrics
```

如果某个 accepted transition 的最终分布确实等于 target transition，那么 old_log_prob 应该是 target transition logprob，而不是 draft logprob。若论文算法的校正路径使某些 residual sample 来自 residual distribution，也必须能还原 trainer 所需的 target behavior logprob，或者该路径不能进入 policy-gradient training。

这条是 P0 必须读 paper pin down 的核心问题：**final sample exact 不自动等于 every training-step logprob contract 已经清楚**。

## 8. API 草案

先内部实验，不进入 public schema：

```text
SpeculativeDiffusionConfig
  enabled: bool
  draft_kind: previous_policy | cheap_solver | low_precision | smaller_model
  verify_block_size: int
  max_draft_steps: int
  min_acceptance_rate: float
  exact_mode: bool
  fallback_to_target: bool
```

内部 payload：

```text
DraftTransition
  x_t
  x_prev_candidate
  timestep
  draft_log_prob
  draft_aux

TargetVerification
  accepted: bool
  x_prev_final
  target_log_prob
  residual_used: bool
  target_aux
```

输出仍然先走 dense fallback：

```text
DiffusionChunkResult(
  observations,
  actions,
  log_probs = target_old_log_prob,
  timesteps,
  replay_tensors,
  context={
    speculative_acceptance_rate,
    speculative_target_nfe,
    speculative_draft_nfe,
  }
)
```

## 9. Phase plan

### P0 — paper reading + math pin-down ✅ done → `docs/sprints/reading/speculative_diffusion_sampling.md`

目标：把 De Bortoli et al. 的 diffusion speculative sampling 算法读到能实现 toy version，明确：

```text
transition density 需要哪些量
target NFE 为什么能少
accept/reject 后如何记录 target logprob
continuous residual sample 如何处理
哪些 sampler/scheduler 满足条件
```

产出：

```text
docs/sprints/reading/speculative_diffusion_sampling.md
```

验收：

```text
能用自己的话写出 algorithm step-by-step
能指出它是否适配当前 flow_matching SDE transition
能回答 old_log_prob contract 是否可保留
```

### P1 — toy exactness test ✅ done → `tests/math/test_speculative_diffusion.py`（exactness KS + ~72% NFE saved）

目标：先在 toy Markov chain 上实现，不碰模型。

候选：

```text
1D / low-D Gaussian Markov chain
known target transition p_t
known draft transition q_t
```

测试：

```text
speculative sampler 的 marginal distribution 与 target sampler 匹配
acceptance_rate / target_nfe 统计正确
logprob contract 可记录
```

验收：

```text
CPU pytest，不依赖 GPU
固定 random seed 下不要求 bitwise 等于 target sampler
统计分布检验在合理样本数下通过
```

### P2 — single-step VRL transition prototype ✅ done → `vrl/math/diffusion/speculative.py::speculative_sde_step`（disabled==baseline 已钉住）

目标：把 speculative transition 接到 `sde_step_with_logprob` 附近，而不是先改 executor。

路径：

```text
draft proposes prev_sample
target transition computes correction / target logprob
return final prev_sample + target_logprob + acceptance metadata
```

验收：

```text
baseline exact path unchanged
speculative disabled == current sde_step_with_logprob
metadata shape aligns with [B, T]
```

### P3 — real model micro-probe ❌ FAILED THE GATE（探针已删除，数字见下）

**实测结论（2026-06-24，SD3.5-medium，RTX 5090，28 步 / 1024^2，D=262144，sqrt(D)=512）：
whole-latent 的 speculative coupling 在真实图像 diffusion 上 acceptance ≈ 0，省算 0%。
toy（D=8）严重高估，被高维方差墙击穿。**

根因是接受率用的是**全维**标准化均值距离：`a = 2·Φ(-‖Δ‖/2)`，
`‖Δ‖ = (逐元素 RMS drift / σ) × sqrt(D)`。sqrt(D)=512 把任何不微小的逐元素 drift 放大成
‖Δ‖≫1 → acceptance 直接 underflow 到 0。两类 draft 都过不去：

| draft | 逐元素 drift / σ | ‖Δ‖/σ | acceptance |
|---|---:|---:|---:|
| frozen-target（offset 1，论文 Eq.7） | ~3.6% | ~18 | **0.000** |
| previous-policy（相对权重更新 1e-3） | ~1.0% | ~5.1 | **0.011** |
| previous-policy（相对权重更新 1e-4） | ~0% | ~0 | 1.000 |
| previous-policy（相对权重更新 1e-2） | ~1.6% | ~8.2 | 0.000 |

逐步看（offset 1）：最好的中段（step 7–13）也才 ‖Δ‖/σ≈10（acceptance 仍 0）；去噪后期
σ→0，‖Δ‖/σ 爆到几千。previous-policy draft 的接受率悬崖在相对权重更新 **1e-4 与 1e-3 之间**——
也就是只有当两份 policy 几乎完全一样（相对漂移 ≤1e-4）时才接受，而这种「几乎没动」的 draft
对 rollout 省算没有意义。一旦 policy 正常移动（≥1e-3）acceptance 立刻归零。

**这是 F2 的实证答案：whole-latent 单步 speculative coupling 的 NFE 省算机制（靠整步接受来
跳过 target forward）被 latent 维度击穿。** 子向量/逐坐标耦合能保持 exactness 但**不省 NFE**
（仍要每步 target velocity 才能决定反射），所以救不了省算目标。

**因此 P4/P5 被 gate 掉，本轮不跑**（sprint 自己的门槛：P3 acceptance 要「高到 verifier
overhead 值得」——现在是 0）。唯一可能的活路需要新研究，不是多跑：
- 近乎精确的 distilled draft（逐元素 drift < ~1/sqrt(D) ≈ 0.2%），成本最高（§6.4）；
- 改变省算机制本身：不是「整步接受跳过 target」，而是别的结构（如 latent 分块的低维子问题、
  或非耦合的 importance-corrected reuse），但那已不是本 sprint 的 exact-coupling 路线。

---

原始计划（保留作参考）。目标：只测 acceptance 和 target NFE，不训练。

对象：

```text
one small SD3 / Wan / Cosmos prompt
short rollout
draft_kind = cheap_solver or previous_policy
target = current exact rollout model
```

指标：

```text
acceptance_rate
target_nfe_saved
wall-clock speedup
rollout-vs-target distribution sanity
memory overhead
```

验收：

```text
target_nfe_saved >= 25%
wall-clock speedup >= 1.2x
acceptance rate high enough that verifier overhead is worth it
```

### P4 — RL integration gate ⛔ gated-off（P3 未过：acceptance≈0）

> 不跑：P3 已证明 whole-latent coupling 在真实模型上 acceptance≈0、0% 省算。把一个零接受率
> 的机制接进 GRPO 跑数小时只会得到「和 baseline 一样、还更慢」的结论。等出现接受率非零的
> draft（distilled 或别的省算机制）再开。下面 contract 保留——P0–P2 已证明它是干净的。

目标：证明它不只是 inference trick，而是能给 RL 训练用。

必须记录：

```text
old_log_prob = target behavior logprob
behavior_policy_version = target rollout policy
accepted_mask / draft metadata in context
```

训练 gate：

```text
small online GRPO run
reward curve not worse than baseline
rollout mismatch guard clean
advantage / reward variance unchanged within tolerance
```

### P5 — video/world-model path

只有 P4 通过才考虑视频。视频里真正难的是：

```text
target verifier batching
latent memory
VAE/reward artifact lifetime
cross-node Ray payload
```

这是可能产生系统 moat 的部分。

## 10. Failure modes

### F1. Acceptance rate 太低

draft 和 target 差太远，几乎都 reject。结果比 baseline 更慢。必须用 `min_acceptance_rate` 自动 fallback。

### F2. target NFE 没省

如果 verifier 仍然需要每个 denoise step 都跑 target forward，而且不能 batch/skip，那么 speculative 只增加 overhead。P0/P3 必须先回答 target NFE 为什么会减少。

### F3. final sample exact 但 RL logprob 不清楚

这最危险。paper 可能只关心 final sample distribution；VRL 关心每个 trainable transition 的 behavior logprob。如果无法给 trainer 一个清晰 target old_log_prob，这条不能用于 policy-gradient rollout，只能用于 eval/serving。

### F4. policy changes break draft

previous-policy draft 在 early training 可能和 target 差很远。需要 acceptance-aware scheduling：policy 更新小的时候启用，acceptance 低时关闭。

### F5. short schedules收益不足

本 repo 很多 diffusion config 只有 10–35 steps。即使理论能省 NFE，absolute gain 可能不够覆盖 verifier overhead。必须实测。

## 11. Non-goals

- 不做 approximate draft-only rollout。
- 不把 previous-policy draft 当 behavior policy 直接训练。
- 不承诺 bitwise same trajectory。
- 不先做 smaller draft model training。
- 不把 speculative 与 fp8/TeaCache/shared-prefix 同时打开；先隔离变量。
- 不在 P0/P1 通过前改 public config。

## 12. 与现有 sprint 的关系

- `SPRINT_signal_paged_rollout`：shared-prefix 是 algorithmic tradeoff，可能省算但改变相关性；本 sprint 是 target-exact distribution path。若目标是“same target semantics”，本 sprint 优先级更高。
- `SPRINT_rollout_vllm_migration`：TeaCache 是 approximate within-sample cache；本 sprint 只有在 target correction 明确时才算 exact。
- `SPRINT_rollout_optimization_layer`：fp8/kernel patch 降低单次 forward 成本；本 sprint 尝试减少 target forward 次数。
- `SPRINT_rollout_correction_rejection` / low-precision TIS：如果 speculative path 退化成 approximate rollout，就必须回到这些 correction；exact path 不应依赖它们兜底。

## 13. 参考代码

- `vrl/math/diffusion/flow_matching.py`：`sde_step_with_logprob`，当前 target transition/logprob 地基。
- `vrl/generation/diffusion/executor.py`：denoise loop，未来 speculative transition 接入点。
- `vrl/generation/diffusion/gather.py`：`DiffusionChunkResult` 聚合，加入 speculative counters。
- `vrl/trajectory/builders.py`：`old_log_prob` / `mask` / replay input contract。
- `vrl/rollouts/evaluators/diffusion/sde_logprob.py`：trainer replay logprob，必须与 speculative rollout logprob contract 对齐。
- `vrl/trainers/weight_sync.py`：previous-policy draft / target-policy version 的同步边界。

## 14. 论文 / 外部参考

- Fast Inference from Transformers via Speculative Decoding — https://arxiv.org/abs/2211.17192
- SpecInfer: Accelerating Generative Large Language Model Serving with Tree-based Speculative Inference and Verification — https://arxiv.org/abs/2305.09781
- Accelerated Diffusion Models via Speculative Sampling — https://arxiv.org/abs/2501.05370
- ICML/PMLR page: Accelerated Diffusion Models via Speculative Sampling — https://proceedings.mlr.press/v267/de-bortoli25a.html
- vLLM / PagedAttention for contrast, not as the mechanism here — https://arxiv.org/abs/2309.06180
