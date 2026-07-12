# SPRINT: Async rollout/train overlap for DiffusionNFT（parked）

状态：**parked / design-decided（2026-06-17；2026-06-20 复核仍 parked）**。这是"小 rollout 吃不满 GPU → 能不能边采边训"
这个问题的设计裁决 + continuous-for-cosmos 接线排查。结论先行:**async overlap 对我们当前的
DiffusionNFT 没有"理论安全"版本,只有"实测不伤就用"的经验路;且任何真 overlap 都需要 ≥2 卡。**

> **边界更新（2026-07-11）**：
> `docs/sprints/planned/SPRINT_miles_phase_lease_and_one_continuous.md` 已把这里提到的单卡
> resident continuous harness 连同 `require_separate_gpus`、`persistent_colocated_workers` 和
> role `memory_fraction` 一起删除。当前单卡 shared topology 只能走 strict phase lease；唯一
> production continuous 是 disjoint-GPU owner loop，且 typed staleness window 必须 `>=1`。
> 下文关于旧 harness/escape hatch 的描述只保留为历史，不再表示当前代码能力。

> **复核更新（2026-06-20）**：本 doc 的 §1 核心裁决「DiffusionNFT 无理论安全 async」**现已被代码强制**——
> `vrl/algorithms/diffusion_nft.py:51 tolerates_off_policy_staleness = False` + `vrl/rollouts/orchestration/schedule.py:124`
> 的 `max_stale>0 且不容忍 → ValueError` fail-fast。相关基础设施也已落地并归档 `done/`：非-drain 权重同步
> （`lifecycle.py:87 supports_non_draining_weight_sync` + versioned trainable-state slots）、`SPRINT_continuous_scheduler_redesign`、
> `SPRINT_shadow_model_weight_sync`。**但本 sprint 自己的交付物**（NFT-specific async overlap 的 Option A/B/C 菜单：
> mask-offpolicy-prefix / collect_rollout_logprobs / v_old / WeightSyncThread —— `vrl/` 全部零命中）**仍未开工、仍卡 ≥2 卡**。
> 即：相关工作落地反而**强化**了这个 park（裁决已成 code-enforced），而非取代它。保持 parked。

**触发事件(满足才解 park)**:**有 ≥2 张可用 GPU**(trainer 常驻一张、rollout actor 常驻另一张)。
单卡到此为止——三家(slime / cosmos-rl / 我们)一致,单卡没有真 overlap(显存墙)。

**两个已锁定的前提(2026-06-17 与用户确认)**:
1. **算法锁 DiffusionNFT**(论文用的,likelihood-free)。不换 flow-matching GRPO。
2. **暂单卡、能搞到多卡** → 现在只做卡数无关的 prep + 本设计文档;决定性实验(Option A)等多卡。

**Scope guard（2026-06-17）**：这里的 async 只指 **rollout actor / ready queue** 和 **trainer**
的跨阶段重叠。sync mini/microbatch 是内存切片与梯度累积机制，不属于本 sprint，也不作为 async
验收标准。

来源:对 `~/Desktop/slime`、`~/Desktop/cosmos-rl`、本仓库逐文件读出的证据(2026-06-17 workflow,
file:line 已核)。

> **更新(2026-06-17,源码复核后修正)**:仓库里已有一个**单卡 continuous resident 配方**
> `configs/experiment/diffusion/sd3_5/online_grpo_ocr_single_gpu_async_debug.yaml`(commit `061cfb2`)。
> 它把 `require_separate_gpus:false` + `persistent_colocated_workers:true` + `release_after_collect:false`
> 组合起来,让 continuous 编排**能在单卡启动且不 RAISE**。这**不改变本 doc 的核心裁决**(真 wall-clock
> overlap 仍需 ≥2 卡;DiffusionNFT 仍无可证安全的 async),但下面 §2/§3/§6/§8 中"单卡机制上跑不了 /
> `_validate_allowed` 已禁掉单卡"的措辞**是错的,已据此更正**——`_validate_allowed` 禁的是 **offload 运行时**
> 和 **`require_separate_gpus AND colocated`** 这两个**具体条件**,不是"单卡"本身。这个单卡配方是 **GRPO**(有
> IS 比值),只验证 continuous 机制(producer/queue/consumer、version-stamp、weight-sync barrier、常驻
> worker 的 LoRA 推送=Gap 5、`max_stale=1` 准入),**不**验证 DiffusionNFT-under-staleness,也**不**产生真
> overlap(单卡只切显存不切时间)。

---

## 1. 核心裁决:为什么 async 对 DiffusionNFT 不可证安全

让陈旧 rollout 变安全的"修正",两家靠的**全是 logprob 比值** `exp(logp_new − logp_rollout)`:

- **slime** = 算法修正、浅管线:1 步双缓冲(`train_async.py:35-53`,disaggregated-only,
  `train_async.py:11` 断言 `not colocate`),用 **TIS**(truncated importance sampling)给 `pg_loss`
  乘 `clamp(exp(train_logp − rollout_logp), clip_low, clip)`(`slime/backends/megatron_utils/loss.py:744-765`)。
- **cosmos-rl** = 系统修正、深管线:中央 controller 给每个 prompt 盖**预测的 weight_version**
  (`controller.py:273-276`),三道 version gate 把 staleness 卡在 `allowed_outdated_steps`(默认 4),
  外加 WeightSyncThread 在独立 CUDA stream 上做 buffer-model swap 隐藏同步延迟。

**最硬的"tell"**:cosmos-rl **自己的 NFT diffusion trainer 零算法修正,出厂就是 `mode='colocated'`
(serial,on-policy by construction)**(`cosmos-predict2-5-2b-720-nft.toml`)。连 NVIDIA 都没把
diffusion NFT 做成 async。

**我们的 DiffusionNFT 是 verified likelihood-free**:loss 是 normalized_mse on positive/negative
denoise 预测,advantage 只经 `reward_mix` 进入,"previous" adapter 是**预测目标不是 behavior 分母**
(`vrl/algorithms/diffusion_nft.py:228-258, 305-319`)。对比 `vrl/algorithms/grpo/continuous.py:85` 才有
`ratio = exp(log_prob − old_log_prob)`。**所以 TIS / ICEPOP / use-rollout-logprobs / AIPO / PPO-clip
一个都搬不过来。**

## 2. 能搬 / 不能搬

| 机制 | 依赖 logprob? | 对我们 |
|---|---|---|
| 分卡 disaggregation | ❌ | **能搬,已有**(`require_separate_gpus` 默认 True=默认分卡,但 `getattr(config,...,True)` 可覆盖成 false→单卡共卡,**非强制**) |
| weight-sync barrier(pause→drain→sync→resume) | ❌ | **能搬,已有**(`schedule.py:110-119`) |
| version-gating 准入(bounded staleness) | ❌ | **能搬,已有**(`StalenessPolicy.admit`,`staleness.py:38-44`) |
| async WeightSyncThread(独立 stream + buffer swap) | ❌ | 能搬(隐藏同步延迟),未实现 |
| mask-offpolicy-prefix(陈旧 denoise 步 loss 置零) | ❌ | 能搬(唯一不靠比值的 sound 修正),未实现 |
| TIS / ICEPOP / use-rollout-logprobs / AIPO / PPO-clip | ✅ | **不能搬**(DiffusionNFT 无比值) |

## 3. 现状:已有 vs 缺

**已有(orchestration 骨架全在,默认分卡 + 默认关;但有单卡 resident debug 配方,见顶部更新)**:
- continuous 三件套 producer/queue/consumer(`vrl/rollouts/orchestration/continuous/`),producer 提交时盖
  `current_policy_version()`(`producer.py:167`)。
- 版本门 `StalenessPolicy.admit`(`staleness.py:38-44`),默认 `max_stale=0`(= 同版本才训 ≈ strict)。
- weight-sync barrier(`schedule.py:110-119`,`after_train_step` 在 718)。
- 权重推送:CPU state_dict → `ray.put`(单副本)→ fan-out `update_weights.remote`
  (`vrl/generation/ray/weight_sync.py:50-59`);worker 对版本不匹配 chunk 软拒绝
  (`vrl/generation/execution/worker.py:119-136`)。

**缺**:
- **DiffusionNFT loss 没有任何 off-policy 修正项**(见 §1)。
- **producer 侧 freshness gate + receipt-time discard**(cosmos 有,我们只在 consumer 端丢太旧 item)
  —— 防 rollout 跑太超前白生成。GPU 无关、sound、小改。
- async WeightSyncThread、mask-offpolicy-prefix:未实现。

## 4. 三个做法菜单(假设有 ≥2 卡)

| | 是什么 | 卡数无关? | 何时做 |
|---|---|---|---|
| **A** 紧 staleness + 实测 | `max_stale=1`,纯 config,量"偏差伤不伤" | 需 ≥2 卡跑 | **多卡到位先做这个** |
| **B** mask-offpolicy-prefix | streaming 里陈旧 denoise 步 loss 置零、只留新版本步 | ✅ 代码可先写 | 仅当 A 证明伤 |
| **C** 记录生成时目标 `v_old` | rollout 存生成时预测进 trajectory,训练对真目标算 loss | ✅ 架构可先设计 | 让大 staleness 真站得住 |

- **A** = slime 的 1 步双缓冲,经我们现有 producer + barrier 实现,**不加任何修正,直接 benchmark 偏差**。
- **B** = slime `mask-offpolicy-in-partial-rollout`(`sglang_rollout.py:247-248`)的 diffusion 类比——唯一
  不靠 logprob 的 sound 修正;需给 `compute_batch_timestep_loss` 加按版本的 step mask。
- **C** = cosmos `collect_rollout_logprobs` 的 diffusion 类比(存 `v_old` 不是 logprob);改 trajectory
  schema(`replay_tensor_dict("denoise")`,消费点 `diffusion_nft.py:140`)+ rollout collector。真研究,非旋钮。

## 5. 排序(async 在"证明 on-policy 会学"之后)

async overlap 是**吞吐优化**,对 DiffusionNFT 它拿"还没证明的学习信号"去冒**无补偿 off-policy 偏差**的险。
正确顺序:
1. **现在(单卡)**:strict on-policy 证明 DiffusionNFT **真能学**(固定 eval / `lr=1e-4` + advantage,
   block-test 曲线)。
2. **多卡到位**:跑 **Option A**(`max_stale=1` vs strict,同 seed/prompt/reward)——量出 staleness=1 伤不伤。
   不伤 → 白捡 overlap;伤 → 再建 B/C。
3. async 是**多卡 follow-on,不是现在该建的东西**。在证明会学之前加不可证安全的 staleness,会搅浑学习判断。

## 6. continuous-for-cosmos 接线排查(2026-06-17,在 VRL@origin/main 上读)

cosmos 现在**从没跑过 continuous**(continuous 唯一示例是 sd3_5 OCR 文本,见 `continuous.yaml` 注释)。
flip 到 continuous 会撞下面的 gap:

- **Gap 1(config)**:`configs/experiment/diffusion/cosmos_predict2_5/online_nft_kling_video_reward.yaml`
  没 include `/base/rollout/orchestration`,默认走 `strict_on_policy`(`schedule.py:46-66` 默认 STRICT)。
  且单卡走 `ray_rollout_colocated_single_gpu`。要 continuous 需加
  `/base/rollout/orchestration=continuous` + 一个**分卡** distributed preset(`ray_rollout_cross_node` 已存在)。
- **Gap 2(运行时不兼容,最硬)**:continuous 的 `_validate_allowed` 有两道 raise(`schedule.py:198-208`):
  (a) `requires_driver_model_offload()` → raise;(b) `require_separate_gpus and runtime_is_colocated()` → raise。
  cosmos 的 colocated 单卡运行时**训练模型 collect 时 offload 到 CPU**(offload→collect→restore→train),
  所以**对 cosmos** **(a) 直接命中、continuous RAISE**。cosmos 需要**两卡常驻运行时**(trainer 常驻
  GPU0、rollout worker 常驻 GPU1,均不 offload),从没在此模式跑过。**注:这是 cosmos offload 运行时的限制,
  不是 continuous 的普遍限制**——sd3_5 的 resident 配方(`online_grpo_ocr_single_gpu_async_debug.yaml`)用
  `release_after_collect:false` + `persistent_colocated_workers:true` 走 **resident 路径**,那里
  `requires_driver_model_offload` 硬编码 `False`(`runtime.py:59`;会翻 True 的 `gpus_per_worker>0` 只在
  on-demand 路径 `runtime.py:87`),所以 raise (a) 不触发;再加 `require_separate_gpus:false` 短路 raise (b),
  **单卡 continuous 可合法启动**(见顶部更新)。
- **Gap 3(sync streaming × continuous 接线未验证)**:cosmos 用同步 streaming microbatch
  (`microbatch_size=1` → `gas = rollout_batch_size`)。`_run_streaming_optimizer_update`(`online.py:283`)
  每个 optimizer update 会多次调用 `collect_training_batch`，每次进入
  `rollout_schedule.next_iteration`(`trainer.py:439`)。continuous queue 设计是 policy-versioned ready
  rollout groups / iterations；需要验证这些重复 strict calls 是否都保持同一 optimizer target 的版本边界、
  metrics 聚合和 weight-sync 时机正确。**这不是 microbatch async 计划**：不要把 consumer 边界降到
  microbatch，也不要在 `_run_streaming_optimizer_update` 里加 prefetch。continuous 示例(sd3_5 OCR)
  是非 streaming(gas=0)，所以这仍是多卡到位前要先 smoke 的接线 gap。
- **Gap 4(DiffusionNFT × continuous 从未跑)**:continuous 只跑过 GRPO(sd3_5/AR,**有** IS 比值)。
  DiffusionNFT + continuous + `max_stale≥1` 正是 §1 的不安全组合;`max_stale=0` 才 behavior-equiv strict。
- **Gap 5(weight-sync 模型不同)**:continuous 用 barrier + **常驻 worker** 的 `update_weights` 推送;
  cosmos 现在是 `release_after_collect`(kill+relaunch+reload)。continuous 下 `release_after_collect` 必须**关**,
  rollout worker 常驻;需验证常驻 worker 对 DiffusionNFT LoRA adapter 的 `load_trainable_state` 推送可用。

**多卡到位后的接线最小集**:新增一个 cosmos `continuous` override 配方(orchestration=continuous +
分卡 preset + `release_after_collect=false` + `max_stale=0` 起步)→ 先把 Gap 2/5 跑通到"不 RAISE、能同步"→
再查 Gap 3(streaming×continuous)→ 才到 Option A 的 `max_stale=1` 实测。

## 7. 推荐第一步(多卡到位时)

**Option A,2 卡,`max_stale=1` vs strict baseline**,同 seed/prompt/reward,看 block-test reward 曲线
(别用单 t 点,见 first-trustworthy-curve 教训)。纯配置、零算法改动,直接把"理论说没补偿"量成"实际伤不伤"。
伤了再建 B/C,不伤就白捡 overlap。

## 8. 非目标
- **不在单卡上追求真 overlap**(三家一致:显存墙,colocated 只能时间片;单卡能跑 continuous 机制 smoke,
  但只切显存不切时间,**不产生真 wall-clock overlap**)。注:`_validate_allowed` 禁的是 **offload 运行时** +
  **`require_separate_gpus AND colocated`** 两个条件,**不是"单卡"本身**;单卡机制 smoke 是被允许的(见顶部更新)。
- **不做 microbatch/minibatch async**。sync streaming accumulation 保持 `collect -> backward -> release`;
  本 sprint 只讨论 rollout actor 与 trainer 的跨阶段 overlap。
- **不造假 logprob head 去复用 TIS**(那是 DiffusionNFT 刻意避开的 tractability 假设)。
- **不在证明 on-policy 会学之前**引入不可证安全的 staleness。
- **不换 flow-matching GRPO**(已与用户确认锁 DiffusionNFT;若改主意,整个 slime 修正菜单解锁,本 doc 重写)。

## 9. 关键文件引用
- 我们:`vrl/algorithms/diffusion_nft.py:228-258,305-319`(likelihood-free)、
  `vrl/algorithms/grpo/continuous.py:85`(我们有、NFT 没有的比值)、
  `vrl/rollouts/orchestration/continuous/{producer.py:167,staleness.py:38-44,schedule.py:110-119,198-208}`、
  `vrl/rollouts/orchestration/schedule.py:33-66`(mode dispatch)、
  `vrl/generation/ray/weight_sync.py:50-59`、`vrl/generation/execution/worker.py:119-136`、
  `vrl/scripts/common/online.py:283`(streaming update)、`vrl/trainers/online/trainer.py:439`(next_iteration)、
  `configs/base/rollout/orchestration/{strict,continuous}.yaml`、`configs/base/distributed/ray_rollout_cross_node.yaml`、
  **单卡 resident 配方** `configs/experiment/diffusion/sd3_5/online_grpo_ocr_single_gpu_async_debug.yaml`(commit `061cfb2`)、
  release-flag 派生 `vrl/ray/resources.py:312-352,1149-1152`、launcher 路由 `vrl/generation/ray/launcher.py:190-202`、
  resident vs on-demand offload flag `vrl/generation/ray/runtime.py:59`(resident,硬编码 False)vs `:87`(on-demand)、
  `require_separate_gpus` 可覆盖 `vrl/rollouts/orchestration/schedule.py:91` + 默认值 `vrl/trainers/core/types.py:163`。
- slime:`~/Desktop/slime/train_async.py:11,35-53`、`slime/backends/megatron_utils/loss.py:744-765`(TIS)。
- cosmos-rl:`~/Desktop/cosmos-rl/cosmos_rl/dispatcher/controller.py:273-276`、NFT 配方 `mode='colocated'`。
- 方向研究存档:`docs/sprints/reading/{slime.md,cosmos-rl.md}`。
