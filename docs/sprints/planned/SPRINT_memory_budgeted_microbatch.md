# SPRINT: 内存预算驱动的 microbatch 切片(单根派生,不设两次)（planned）

状态：**T1 + T2 已落地 VRL main**(`58e0c13` 旋钮+RSS guard);随后 4 个 commit 收尾(见落地记录)。T3 未做、T4 待真机。
本 sprint 是 `SPRINT_streaming_rollout_accumulation.md`(流式累积,已落地 VRL
main `4c85f3b`)的**收尾精炼**:把"切多少"从一个**手填的次数**改成一个**大小/内存预算**,派生其余,
并用**真实 RSS** 而不是低估的字节公式来预测/自动定档。对标 slime 的"单根 + 派生 + 预算式切分"。

> **落地记录(2026-06-16,VRL main)**:
> - `58e0c13` T1 `microbatch_size` 旋钮 + T2 host-RAM RSS fail-fast(cosmos predict2.5 启用)。
> - `b89f8e2` 关 footgun:`host_memory_budget_fraction>0` 但没开 streaming → 直接报配置错(原本静默失效)。
> - `34c32df` wan_2_1/online_grpo_ocr(唯一 rbs>1 的 legacy **视频**配置)采用 streaming。
> - `57755b0` AR(janus_pro rbs=8、nextstep_1 rbs=4)采用 streaming。**更正前一轮误判**:AR 不需要改
>   TokenGRPO loss——rollout 按 prompt-group 切批(`split_batch_by_group`),`compute_loss` 是**逐组**调用、
>   `loss_scale` 是全局(`total_groups×timesteps`),`mask.sum()` 归一是**逐组**不是逐 microbatch;故 streaming
>   只是把"逐组循环"分片,梯度等价(条件仅 `global_std=false`,两个 AR 配置都满足)。
> - `39ead74` 加构建期告警:`global_std=true` + streaming 且每刀 >1 组时,advantage 的 global std 变成
>   **逐 microbatch** 而非全批 → 梯度偏离。命中 sd3 的 `online_grpo_ocr/geneval/pickscore`(gas=4、8 组、
>   global_std=true)。按 §6"不机械改 sd3"只**告警**,不改其配置、不硬失败;`microbatch_size=1` 豁免。

> **进度(2026-06-15)**:T1 旋钮 + cosmos 迁移 + 配置 schema + T2 RSS fail-fast + 测试全绿(238 passed)。
> **更正一处误判**:T2 曾被以"与 `SPRINT_generation_memory_system.md` 重叠"为由暂缓——经多 agent 对抗审计
> 核实,该理由**不成立**:那个 sprint 管的是 **GPU 字节预算**(`MemoryContract` 按卡/角色派生),只在"已有机制"
> 表里提过一次 host memory snapshot,并不拥有 **host RSS** 预算/fail-fast 这条轴;且 RSS fail-fast 在 vrl 里
> 全无实现(6 处 `log_host_memory` 都是纯日志),所需原语 `capture_host_memory()`(rss/available/used_fraction)
> 已存在却没人消费。故 T2 在本分支落地**不与任何在飞工作冲突**。"可观测/legacy 告警"那半确实已由上游
> `d400638`(`_log_rollout_memory_plan`)落地,但那只是日志/告警,不是 T2 要的 **RSS 外推 fail-fast**。

> 方法:核了 slime 源码(`/home/mingfeiguo/Desktop/slime`)+ reading doc,逐条带 `path:line`;核了 vrl
> 现状(`estimate_batch_bytes`、`log_host_memory`、per-timestep backward)与 torch 的 host/GPU 内存 API。

---

## 0. Core Decision

**不新增第二个切片旋钮。** 流式累积已经让**一个 microbatch 同时充当 host 采集切片和训练 backward 单元**
(采完即 backward 再 release),GPU 那一轴由 `for j in train_indices` 逐去噪步**自动**变细。所以 vrl
**本来就只切一次**,不需要像 slime 那样把 rollout 和 train 各设一个。

要改的只有两点,都向 slime 看齐:

1. **把 `gradient_accumulation_steps`(次数)降级为派生量,新增 `microbatch_size`(大小)做唯一手填切片旋钮**——你设"一刀放得下几组",次数 = `rollout_batch_size ÷ microbatch_size` 自己掉出来。语义从"切几刀"变成"每刀多大",这才是你真正关心、且能直接对内存的量。
2. **内存预测用真实 RSS,不用 `estimate_batch_bytes`**——后者是队列背压启发式,系统性低估(见 §2);改成在**第一个 microbatch** 上量 `log_host_memory` 的 RSS 增量,据此 fail-fast 或自动定档。

非目标:不动已落地的梯度等价数学;不为 GPU 再加一个 micro-batch 旋钮(per-timestep 已经管住 GPU)。

---

## 1. 现状(已落地的基线 + 两处别扭)

流式累积(`4c85f3b`)已经做到:`online.py` 的 `_run_streaming_optimizer_update` 把 `rollout_batch_size`
切成 `gradient_accumulation_steps` 个 microbatch,逐个 `collect → backward → release`,一次 optimizer
step,host 峰值 ≈ 一个 microbatch。梯度与整批路径**等价**(已有测试钉死)。

两处别扭:

- **`gradient_accumulation_steps` 是"次数",要拿 `rbs` 反算。** 用户真正想表达的是"一刀放得下多大",
  却被迫填一个换算后的次数;`micro = rollout_batch_size // gas`(`online.py`)。slime 的教训是反过来:
  设大小、派生次数(§3)。
- **没有开跑前的内存判断;`estimate_batch_bytes` 不能用来预测 OOM**(§2)。现在只能"跑到崩才知道",
  而视频 clip 下崩得很晚(攒到第 ~24–32 组,白生成约两小时)。

---

## 2. 为什么不能用 `estimate_batch_bytes` 预测 OOM(已核)

`vrl/rollouts/orchestration/continuous/types.py:15` 只 sum 顶层 tensor 字段
`[observations, actions, rewards, dones, group_ids, videos]` + `extras` 里**直接是 tensor 的**(浅层),
docstring 自己写明 *"deliberately avoid a deep traversal of arbitrary extras"*。问题:

1. **漏 `batch.trajectory`**(replay 轨迹:逐 timestep `old_log_prob`/`prev_sample_mean`/mask,`prev_sample_mean`
   是 latent 尺寸 × 去噪步)——正是 OOM 的大头,**完全不计**。还漏 `batch.training_view`、`context`、嵌套 extras。
2. **tensor 字节 ≠ 进程 RSS**:Ray OOM monitor 杀的是 **RSS**(日志 `trainer process 76.41GB`),而 RSS 因
   CPU malloc 碎片 + Python 开销**通常大于** Σ`tensor.nbytes`。

torch 也帮不上 host 这半:`dir(torch)` 里没有 CPU 的 allocated 计数器(只有 `OutOfMemoryError`/
`_pin_memory`/`memory_format`);`torch.cuda.*` 那套(`max_memory_allocated` 等)只对 **GPU** 精确,
`torch.cuda.*host_memory_stats*` 仅指 **pinned** host 内存,不是任意 CPU RAM。`tensor.nbytes` 精确但只是
**单 tensor 逻辑字节**。所以拿这公式做 fail-fast 会出现最坏的错:预测"放得下"、实际 OOM。

→ 正确的真相源是**进程 RSS**,仓库已有 `vrl/utils/memory.py:log_host_memory()`(直接读 `/proc`,无 psutil 依赖,给 `rss_mb`)。

---

## 3. slime 怎么做(对标的样板,已核源码)

slime 只声明**一个根**,其余全派生,GPU 那刀切**预算**不是次数:

| 层 | slime 的量 | 谁定 | 证据 |
|---|---|---|---|
| 数据/rollout 步 | `rollout_batch_size × n_samples_per_prompt`(样本) | **用户手填(唯一根)** | `slime/utils/arguments.py:575-586` |
| 数据/optimizer 步 | `global_batch_size`(样本) | **派生**:`= rollout_batch_size × n_samples ÷ num_steps_per_rollout`;若同时手填则 **assert 必须相等** | `arguments.py:1768-1776` |
| GPU backward 切片 | `micro_batch_size`(固定)**或** `max_tokens_per_gpu`(动态 token 预算) | 设**大小/预算**,次数派生:`num_microbatches = (global÷dp)÷micro` 或 token 装箱 | `megatron_utils/data.py:322-376`、`model.py:325-331,483-492` |
| 生成 GPU batch | —— | **不是 slime 旋钮**,交给 SGLang continuous batching;slime 只设并发 semaphore | `sglang_rollout.py:94-96` |

三条原则:**(a) 一个根、其余派生 + assert(不重复声明同一个量);(b) GPU 那刀切"预算"(`max_tokens_per_gpu`)、次数掉出来;(c) 生成批交给推理引擎。**

vrl 与 slime 的结构差异:slime 的 rollout 和 train 是**独立进程**,train 侧 `get_data_iterator` 重新把
数据切成 GPU micro-batch(所以它"看着像两个旋钮"但其实派生);vrl 是**一个 loop 采完即训**,
**一个 microbatch 两用**——所以 vrl 比 slime 更不需要第二个旋钮。

---

## 4. 目标设计

### 4.1 旋钮:设大小,派生次数

```text
rollout.rollout_batch_size        # 根:每次优化器更新的目标组数(prompt 条件数)。不变。
rollout.microbatch_size   # 新:每个采集/训练切片几组(= 你一刀放得下多少)。
actor.gradient_accumulation_steps # 降级为派生量 = rollout_batch_size // microbatch_size
```

- 用户**只填 `microbatch_size`**(或继续填 `gradient_accumulation_steps`,二选一)。
- 同时填两个 → **assert 一致**(slime 式,`arguments.py:1768-1776` 的同款防漂移),不让两者并存打架。
- 校验沿用已落地的 `TrainerConfig.__post_init__`:`rollout_batch_size % microbatch_size == 0`、
  `>= 1`、`ppo_epochs == 1`(流式不能跨 epoch 重放)。
- **`gradient_accumulation_steps == 0` 兼容路径保留**(legacy 整批)。

> 关键:这是**纯语义换壳,不改数学**。`loss_scale = total_groups × timesteps` 与切法无关,梯度仍等价
> (已有 `test_streaming_matches_full_batch_gradient` 守着)。

### 4.2 vrl 已经"切一次两用",不加第二刀

- **host 轴** = microbatch 组数(`microbatch_size` 控制,采完即 release);
- **GPU 轴** = `for j in train_indices` 逐去噪步,GPU 同时只活"一个 timestep × microbatch 样本"——
  这等价于 slime 的 micro_batch 独立于 data batch,**vrl 免费拿到**(扩散天然按 timestep 切)。

所以**不需要**为 GPU 再加一个 micro-batch 旋钮。除非以后"单 timestep × 一组样本"本身都放不下 GPU,
那才是另一个问题(前向内 gradient checkpointing / 帧切分),不在本 sprint。

### 4.3 内存预算:量 RSS,不用字节公式

两档,推荐都做:

- **fail-fast(开跑前/第一刀)**:第一个 microbatch `collect+backward` 前后各 `log_host_memory()` 一次,
  RSS 增量 = "一个 microbatch 真实占多少 host"(含碎片/Python/pinned 全部)。外推峰值 ≈
  `microbatch_groups × per_microbatch_rss`(流式下其实就是 ~1 个 microbatch),超 host 预算 → 清晰报错
  "这台机放不下,调小 `microbatch_size`",**一刀内就告诉你**,不再白生成两小时。
- **auto-tune(可选)**:据 RSS 增量反推**放得下的最大 `microbatch_size`**(rbs 的最大因子),
  自动定档——把内存↔吞吐的甜点交给测量而不是手调(microbatch 越大越摊薄 Ray 往返 / reward 批 / 生成 GPU 利用率)。

---

## 5. 实施计划

### T1 — 旋钮:`microbatch_size` 为主,`gradient_accumulation_steps` 派生 ✅ 已实现
- `TrainerConfig` 加 `microbatch_size`(`metadata={"yaml":"rollout"}`);`__post_init__`:二选一、同填则 assert
  相等、互相派生。复用已落地的整除/`ppo_epochs==1` 校验。`vrl/trainers/core/types.py`。
- **`online.py` 无需改**:`_run_streaming_optimizer_update` 读的是派生后的 `gradient_accumulation_steps`,
  `micro = rbs // gas` 已正确(纯换壳,梯度数学不动)。比原计划更省 churn。
- **配置 schema**:`vrl/config/schema.py:RolloutConfig` 加 `microbatch_size`(否则 unknown-key 校验拒收)。
- 配置迁移:cosmos kling `gas=32/rbs=32` → `microbatch_size=1`、cosmos motion `1/1 → 1`(等价)。
- 测试:`test_reward_update_flow` 加派生/assert 用例;`test_load_all_experiments` 的 kling 几何测试从
  "断言原始 yaml gas==32" 改为"断言 schema 声明 size + TrainerConfig **派生** gas==32"(去 exact-config)。

### T2 — RSS 预算 fail-fast(用现成 `capture_host_memory`)✅ 已实现
- **配置项**:`TrainerConfig.host_memory_budget_fraction`(yaml:`rollout`,`field(default=0.0)`,`__post_init__`
  校验 `[0.0,1.0)`;`0.0` = 关)。schema 同步登记,unknown-key 校验通过。`vrl/trainers/core/types.py`。
- **fail-fast**:`online.py:_check_host_memory_budget` —— 流式累积下"持有 ~1 个 microbatch"即 host 峰值,故
  **采到第一个 microbatch 后**量真实 RSS(`capture_host_memory` 读 `/proc`,非字节估算,因 Ray OOM 监控按 RSS 杀),
  `used_fraction > budget` 即 `raise MemoryError`,报错含快照 + "调小 `microbatch_size`"的建议。
  在 `_run_streaming_optimizer_update` 的 `mb_index==0` 处调用(每个 update 起点查一次,catch host-RAM creep)。
- **配置启用**:两个 cosmos_predict2_5 配置加 `host_memory_budget_fraction: 0.9`(就是报"延迟 OOM"的那批受限显存视频跑)。
- 测试:`test_host_memory_budget_fail_fast`(注入假 RSS:超预算 raise、≤预算放行、读不到内存不误杀)+
  `test_host_memory_budget_fraction_bounds`(边界校验)。238 passed。
- **可观测那半**已由上游 `d400638`(`_log_rollout_memory_plan`)落地(streaming plan 日志 + legacy 告警),与本 fail-fast 互补。

### T3 —(可选)auto-tune microbatch ⏸ 未做(可基于 T2 的 RSS 量测)
- 据 RSS 增量反推最大可行 `microbatch_size`(rbs 因子),日志说明选了哪档、为什么。
- 默认关闭(显式 opt-in),避免"自动改 batch 形状"的意外。

### T4 — 真机 smoke(你来跑,需 GPU)
- cosmos kling:确认换语义后**仍不 OOM、一行 metric、梯度行为不变**;T2 在故意调大 microbatch 时**一刀内报错**而不是跑半天。

---

## 6. 配置迁移(在线 + gas>0 的)

| 配置 | 现 `gas/rbs` | `microbatch_size`(= rbs/gas) | 备注 |
|---|---|---|---|
| cosmos kling | 32 / 32 | 1 | 等价,1 组一刀 |
| cosmos motion | 1 / 1 | 1 | 等价 |
| sd3 geneval/ocr/pickscore | 4 / 8 | 2 | ⚠️ 见下:已落地的 gas 重定义已把它们从 2 步/rollout 变 1 步 |
| anima aesthetic(±nsfw) | 4 / 4 | 1 | 等价 |
| sd3 ocr_crossnode_debug | 1 / 2 | 2 | ⚠️ 同上,2 步 → 1 步 |

⚠️ **承接基线的副作用**:流式 sprint 落地时 `gradient_accumulation_steps` 已被重定义,sd3 / ocr-debug 的
optimizer 步数从 2 变 1。本 sprint **不引入新副作用**(只换壳),但迁移时要么接受、要么按 slime 思路用
`rollout_batch_size` 表达原步数(streaming sprint §3.3)。这是 sd3 实验所有者的决定,不机械改。

---

## 7. 非目标

- **不加第二个 GPU micro-batch 旋钮**:per-timestep 已经管住 GPU 轴;vrl 采完即训,一个 microbatch 两用。
- **不用 `estimate_batch_bytes` 做 OOM 预测**:它是背压启发式、系统性低估(漏 `trajectory`/`training_view`、tensor 字节 ≠ RSS)。保留它原用途(队列字节上限)。
- **不改已落地的梯度等价数学**:`loss_scale = total_groups × timesteps`、一次 optimizer step / 更新,保持不变。
- **不引入 psutil**:`log_host_memory` 已用 `/proc`。
- **不在本轮重写采集为多步异步**(那是 cosmos-rl 风格的更大改,见 `SPRINT_continuous_scheduler_redesign.md`)。

---

## 关键文件引用

- 已落地基线:VRL main `4c85f3b`、`docs/sprints/done/SPRINT_streaming_rollout_accumulation.md`;
  `vrl/scripts/common/online.py:_run_streaming_optimizer_update`、`vrl/trainers/online/trainer.py`
  (`begin_optimizer_update`/`backward_on_training_batch`/`finish_optimizer_update`)、`vrl/trainers/core/types.py:__post_init__`
- 内存:`vrl/rollouts/orchestration/continuous/types.py:15`(`estimate_batch_bytes`,**别用于预测**)、`vrl/utils/memory.py:log_host_memory`(RSS 真相源)
- GPU 侧(若需):`torch.cuda.max_memory_allocated()` / `reset_peak_memory_stats()`
- slime 对标:`/home/mingfeiguo/Desktop/slime` —— `utils/arguments.py:575-586`(rollout/n_samples 根)、`:1768-1776`(global 派生 + assert)、`backends/megatron_utils/data.py:322-376`(num_microbatches 派生 / `max_tokens_per_gpu` 装箱)、`backends/megatron_utils/model.py:325-331,483-492`(Megatron micro-batch backward)、`backends/sglang_utils/arguments.py:39` / `sglang_rollout.py:94-96`(生成交给引擎,只设并发)
