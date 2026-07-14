# SPRINT: Reward execution —— 成本分层放置 + 异步打分

> **Superseded status (2026-07-12):** this document preserves an earlier,
> withdrawn design discussion. Its claims that VRL has no standalone reward
> service, that reward execution uses a local/Ray switch, and that strict
> streaming adds no capability are no longer current. The implemented source of
> truth is `docs/sprints/SPRINT_reward_service.md`: reward inference now supports
> typed `in_process` and `http` runtimes, strict streaming is gated by both
> non-blocking execution and verified accelerator isolation, and unverified HTTP
> endpoints fail closed. The cost-aware auto-placement proposal below remains a
> non-implemented historical proposal, not a config contract.

状态：**design / not-started（P1 + P2 都已撤回，2026-06-29）；P3 仍未实现**。本轮 CPU 落地了 P1（流式打分）+ P2（reward_cost 成本感知放置），评审后**两个都撤回**，只保留分析结论（§13/§13.1）和一个顺带的 stale 测试修复（`VRL_PROFILE_COLLECT`→`VRL_PROFILE`）。
- **P2 撤回**：现形态只做了"显式标注"那半，**和已有的显式 `gpu_pool: rollout` 重复**；真正价值在自动测量（需 GPU run，未做），且和 P1 互斥（单卡 share = 串行卸载，不是并发）。
- **P1 撤回**：reward 的 async **早已存在于 `continuous` 模式**（reward ∥ 别的 group 生成 + ∥ 训练，见 §13.1）；P1 改的 `collect_prompt_batches` 在 continuous 下每次只收一个 group → no-op，只对 strict_on_policy 默认路径有那一条窄的 call 内重叠，niche 太小、未实测、和 continuous 高度重叠。**没有引入新能力。**
此前为 design / not-started。注意 §4.1/§4.2/§6/§10 旧文引用的 `inference_runtime: local|ray` 已被 8de3e63 改名为 `execution: inline|pool`；§9.1.1 的 factory reward-key 硬编码已由 `active_pool_reward_keys`（按 `execution` 派生）消除，无需再改。
- **位置变更（2026-06-29）**：`active_pool_reward_keys`（"哪些 active reward 走 GPU pool"，含 `default_execution` 类默认回退）已从 `vrl/ray/resources.py` **搬到 reward 领域** `vrl/rewards/functions/registry.py`——通用 Ray 底座 `ray/resources.py` 不再 import reward registry（修 `test_shared_ray_substrate_stays_domain_neutral`）。`resolve_distributed_resources` 改为接收 keyword-only `pool_reward_count`（领域感知入口 factory/online 算好传入），底座只拿"要几块 reward GPU"这个数，不懂 reward 的事。见 `done/SPRINT_weak_test_cleanup.md` 收尾记录。

关联：
- [[SPRINT_global_ray_placement_owner]]（reward 的 GPU 怎么来：owner 的 bundle）
- [[SPRINT_framework_lessons_vrl]]（非阻塞 barrier 这条主题的同源）

## 0. Core Decision（先看这一段）

一句话：**reward 不该有一个全局统一的放置策略。正确的是按 reward 的成本分层放置
——轻的进程内跑、中等/突发的和 rollout 共享一张卡、只有重且能喂满卡的才独占——
而"这个 reward 能不能填满一张卡"就是"独占还是共享"的判据本身。不要给 VRL 建一个
独立的 reward 服务。**

三条具体结论：

1. **不建独立 reward 服务（HTTP / 独立 Ray 池）。** slime/cosmos 走外部服务是因为它们
   的 reward 对框架是外部的；VRL 是 Ray-native，且已经是"落盘 artifact + Ray actor 池"
   形态，用灵活 placement 处理异质 reward 比"统一拆出去"更合适。
2. **放置是利用率问题，判据是"能否喂满一张卡"。** 能 → dedicate（自然忙）；不能 →
   colocate/share（混合工作填满卡，别空转）；不要 GPU → in-process。VRL 的
   `reward.share_with_rollout` 三态已经是这个形状。
3. **真正要补的是异步打分 + 让 `auto` 成本感知**，不是改放置模型。现在
   `collect 全部 → score 全部` 是硬 barrier；`auto` 只看"有没有空闲卡"。这两点是这个
   sprint 的主线。

## 1. 触发这次讨论的两个约束（都成立）

1. **reward 是异质的。** OCR（rapidocr，纯 CPU）很轻，Kling VideoReward（视频理解模型）
   很重。给所有 reward 配专属卡荒谬。
2. **独立卡很难一直喂满。** reward 是突发的——一批 rollout 出来才打分，打完就闲。
   专门给它一张卡，卡在两次突发之间空转。这正是 colocation（共享分时）存在的理由：
   某角色填不满一张卡时，让它和别人分时，卡才一直忙。

任何"reward 架构"方案必须同时回答这两点；"统一拆成独立服务/独立卡"两点都答不好。

## 2. 前人怎么做（关键：是 spectrum，不是 all-dedicate）

两家都按 reward 成本分档，只有重模型那一档才走独立服务：

| | 轻 / 规则 reward | 重 / 模型 reward |
|---|---|---|
| **slime** | 规则类（math/f1/gpqa）在 rollout 的 asyncio 事件循环里 CPU 跑，**在生成 semaphore 之外**，慢 reward 不挡 GPU 准入（`reading/slime.md:161,697,788`） | `remote_rm` = HTTP 打到 `args.rm_url`（`reading/slime.md:797`），**一个你自己另起的外部服务，完全在 slime 的 placement group 之外** |
| **cosmos-rl** | 文本/规则在 rollout worker 上的 `ProcessPoolExecutor`/`ThreadPoolExecutor`（非文本用线程池，视频张量不 pickle），**非阻塞 FIFO，生成照常继续**（`reading/cosmos-rl.md:431,928`） | `use_remote_reward` = HTTP octet-stream 打到独立的 `cosmos_rl_reward` 包（`reading/cosmos-rl.md:11,943`），视频先 VAE 编码成 fp16 latent 再发 |

共同点：
1. reward **异步、和生成交错、在 GPU 准入路径之外**——绝不是 collect-all→score-all 的硬 barrier。
2. **只有重模型 reward** 走独立服务/独立 GPU；轻 reward 留在 worker 上（CPU 池）。
3. 没有一家给"所有 reward"配专属 GPU，更没有把 reward 模型塞进 rollout 的 placement
   group 当共管 bundle。

## 3. VRL 现状盘点（带证据，避免边讨论边猜）

VRL 其实已经有大半分层能力：

```text
Tier 0 轻/CPU reward → 进程内
  vrl/rewards/runtime.py  InProcessRewardRuntime（OCR 等函数 reward 在采集进程内打分）
  factory.py _with_resolved_reward_runtime_kwargs：reward_key 非 kling/video 直接 return
                                                  （OCR 根本不进 Ray 路径）

Tier 1/2 GPU reward → 共享 or 独占（开关已存在）
  vrl/ray/resources.py  RewardResourceConfig.share_with_rollout 三态
                        （None=auto / true=共享 / false=独占）
  _resolve_reward_devices：auto = "有空闲卡就 dedicate，没有就 share"

数据面已是"落盘 + Ray 池"（disaggregation-friendly，对位 cosmos）
  vrl/rewards/base.py _init_disk_artifact_reward：VideoRewardArtifactStore 写盘 + RayRewardRuntime 池打分
  vrl/rewards/ray/runtime.py  RayRewardRuntime → RayActorMethodRuntime

reward 的 GPU bundle（MR4）
  vrl/ray/resources.py build_bundle_layout / vrl/ray/placement.py reward_placement
```

两个真实 gap：

```text
G1. collect→score 是硬 barrier
    vrl/rollouts/collector/core.py：collect_unscored(全部) → score_rollouts(全部) → train
    reward 不和生成并发，dedicated reward 卡更难喂满。

G2. auto 只看"有没有空闲卡"，不看 reward 成本
    _resolve_reward_devices 的 auto 一旦发现 spare 就 dedicate，不判断 reward 能否喂满它。
```

## 4. 目标设计：成本分层放置（Tier 0/1/2）

```text
Tier 0  轻 / CPU reward（OCR、规则类）
        → 进程内，不要 GPU / actor / placement
        → VRL 已这么做，别动（对齐 slime 规则 reward、cosmos 文本 reward）

Tier 1  GPU reward 但填不满一张卡 / 突发
        → colocate/share：和 rollout 分时复用一张卡，让卡做混合工作别空转
        → share_with_rollout=share；是约束②的解法

Tier 2  重 reward 且能喂满一张卡（大模型 + 足够深的 rollout buffer 持续供给）
        → dedicated 卡 + 异步流水线打分，reward(N) 与 rollout(N+1) 并发
        → share_with_rollout=false；只有这一档才"独占"
```

判据（核心）：**"这个 reward 的吞吐能不能撑起一张卡？"**
- 能（重 + buffer 深）→ dedicate，靠持续 backlog 喂满。
- 不能（中等 / 突发）→ share，混合工作填满卡。
- 不要 GPU → in-process。

### 4.1 更好的抽象：派生，不要新建一张配置矩阵

原来的写法把三件事揉进一个枚举：

```yaml
tier: local_cpu | gpu_shared | gpu_dedicated
placement: auto | share_rollout | dedicated
```

诊断对（执行方式 / 资源需求 / 利用率不该揉在一起），但**药方不是再造一张多轴矩阵**。
一个看着"更通用"的 4 字段 spec——

```yaml
execution:
  backend: local | ray_actor | external_service
  accelerator: none | gpu
  concurrency: inline | async_pool | actor_pool | service
  saturation: none | bursty | sustained | auto
```

——其实是 overengineering，三条理由：

1. **`backend` 和 `concurrency` 几乎是同一个轴**：`ray_actor ⟺ actor_pool`、
   `external_service ⟺ service`，只在 `local` 时分叉一次。一个自由度拆成两个字段。
2. **`accelerator`/`backend` 是 reward 实现的属性，不是用户该配的**：OCR 用 CPU、Kling 用
   GPU，代码自己知道。让用户在 YAML 声明它，正撞 AGENTS.md "不要在配置里复制 typed
   structure" 那条。
3. **枚举笛卡尔积大半非法**：`local+service`、`none+sustained` 都没意义；一个大半无效的
   组合枚举比它要取代的 `tier` 还脆。

而且 **slime/cosmos 都没有这种多轴 spec**（见 §4.2）：slime 是一个 `rm_type` 字符串、
cosmos 是一个 `use_remote_reward` 布尔，执行方式**搭在 reward 身份上**，不是另开一张矩阵。

**关键：VRL 已经有那个"一个 switch"了。**

```text
vrl/rewards/runtime.py  make_reward_runtime(inference_runtime: "local" | "ray")
configs/reward/kling_video_reward.yaml  inference_runtime: ray
```

`inference_runtime: local | ray` 和 slime 的 `rm_type` / cosmos 的 `use_remote_reward`
完全对位——backend / accelerator / concurrency 全部从它 + reward 类型**派生**，不需要新字段：
- 函数 reward（OCR）→ `local` → 进程内、CPU、inline。
- 模型 reward（Kling）→ `ray` → actor 池、GPU。

**唯一真正缺的输入是"能不能喂满一张卡"——而它该测，不该配。**只加这一个，且默认 auto：

```yaml
reward:
  kwargs:
    kling_video_reward:
      inference_runtime: ray
      # 唯一新增：一个数(per-item 打分耗时 ÷ 生成耗时)，默认 auto 测，
      # 不是 none|bursty|sustained 这种人为分桶。
      reward_cost: auto        # auto | <float ratio>
```

resource resolver 用 `inference_runtime`（已有）+ `reward_cost`（新增、auto 测）+
topology + queue depth 派生 placement：

```text
inference_runtime=local              → 无 GPU，InProcessRewardRuntime / CPU
inference_runtime=ray + cost 低/突发  → 即使有 spare 也优先 share（别空转）
inference_runtime=ray + cost 高 + backlog 够 → dedicated bundle，异步队列喂满
（未来）external service              → 不进 VRL placement，VRL 只管 client/backpressure/error
```

`Tier 0/1/2` 只是文档里的解释标签，永不进 config：

```text
Tier 0 = inference_runtime=local
Tier 1 = inference_runtime=ray + cost 低/未知 + colocatable
Tier 2 = inference_runtime=ray + cost 高 + dedicated backlog
```

净结论：**不加 `tier`，也不加 4 字段 `execution`。复用已有的 `inference_runtime`（= slime/
cosmos 的那个 switch），只补一个 auto 测量的 `reward_cost` 给 resolver 判 share/dedicate。**
比 4 轴矩阵和原 tier 枚举都简单，且对齐"从源头派生、别在配置里复制"。

### 4.2 slime / cosmos 实际不是 tier 配置，而是执行路径选择

slime 没有 `gpu_shared/gpu_dedicated` 这种 reward placement knob。它的选择面是
`rm_type` / `custom_rm_path`：

```text
custom_rm_path / math / f1 / gpqa / ifbench / random
  → rollout manager 的 asyncio reward path

remote_rm + rm_url
  → HTTP 调外部 reward model service
```

证据：slime 的 `async_rm` 是 reward hub 字符串分发，优先级是 `custom_rm_path`、
per-sample `metadata["rm_type"]`、全局 `rm_type`；`remote_rm` 只是其中一个分支
（`reading/slime.md:771-803`）。更关键的是，reward HTTP 调用发生在 generation semaphore
之外，所以慢 reward 不挡 GPU generation 准入（`reading/slime.md:153-162,697`）。
换句话说，slime 的抽象是 **execution backend**，不是 **placement enum**。

cosmos-rl 也没有全局 tier 配置。它的选择面是 `use_remote_reward`：

```text
use_remote_reward=false
  → rollout worker 上的 RewardDispatcher，本地 ProcessPool / ThreadPool

use_remote_reward=true
  → reward_service/ 独立包，HTTP octet-stream 远程打分
```

证据：本地 reward 在 rollout worker 上跑，`RewardDispatcher` 按 64 payload/task 分片，
文本走 process pool，非文本走 thread pool；dequeue 是非阻塞 FIFO，慢 reward 会 head-of-line
block reporting，但 generation 继续（`reading/cosmos-rl.md:924-943,1273`）。远程 reward
路径把 payload 通过 HTTP 发给独立 `reward_service/` 包，视频会先编码成 fp16 latent
（`reading/cosmos-rl.md:11,943`）。

所以 VRL 不应该复制一个 `tier` 字段；应该复制这条原则：

```text
reward function declares execution capability;
scheduler/resource layer derives placement;
remote service is one backend, not the default architecture.
```

## 5. 两个约束的回答

**约束①（异质 reward）** → Tier 分层直接解决：OCR 落 Tier 0（无 GPU），Kling 落 Tier 1/2。
放置是 per-reward 决策，不是全局开关。

**约束②（独立卡难喂满）** → 这正是"独占 vs 共享"的判据本身，不是反对意见：
- reward 填不满卡 → 不 dedicate，share 让卡忙。
- reward 能填满卡 → dedicate，且靠异步流水线（深 buffer 持续供 group）让它一直有 backlog。
- 补充：`auto` 现在 dedicate 的是**空闲卡**——那张卡本来就闲，给 reward 至少避免了
  "为打分暂停 rollout"。真正的浪费是**从 rollout 抢卡喂突发 reward**，而 auto 不会这么干
  （没 spare 就 share）。所以方向没错，缺的是"成本感知"那一层。

## 6. 实施计划（主线不是改放置，是这两件）

```text
P1  异步打分：干掉 collect→score barrier
    - group 一产出即异步丢 reward 打分，与下一轮生成并发
    - 轻 reward 也不该卡住生成（对位 slime "RM 在 semaphore 之外"、
      cosmos "非阻塞 FIFO，生成继续"）
    - 让 dedicated reward 卡靠持续 backlog 喂满（正面解约束②）
    落点：vrl/rollouts/collector/core.py（score_rollouts 改成流式/异步）、
          vrl/rollouts/orchestration/continuous/（把 reward 接进 producer/consumer 流水线）

P2  auto 改成成本感知（不只看 spare 有无）
    - 不新增 execution 配置；复用已有 inference_runtime(local|ray) 当 backend switch，
      只补一个 auto 测量的 reward_cost（per-item 打分耗时 ÷ 生成耗时）
    - tier 不进配置；由 inference_runtime + reward_cost + topology + queue depth 派生
    - 决策：cost 高 + buffer 深 → dedicate；cost 低/未知 → 即使有 spare 也优先 share；
      inference_runtime=local → in-process
    落点：vrl/rewards/*（reward_cost hint，默认 auto 测）、
          vrl/scripts/common/factory.py（按 inference_runtime 派发 runtime，去掉 reward-key 硬编码）、
          vrl/ray/resources.py（_resolve_reward_devices 的 auto 分支消费 reward_cost）

P3  ref-based 数据面（接最早的"数据过 driver 两次"）
    - rollout 产物走对象引用/磁盘，reward 直接 pull，不绕 driver 再切片重发
    - VRL 已有 artifact store，把这条接成 ref-based 即可
    落点：vrl/generation/ray/executor.py（产物 ref）、vrl/rewards/ray/*（按 ref 取）
```

P1 是吞吐主线；P2 让分层自动化；P3 省 driver 往返。三者都不需要新建独立 reward 服务。

## 7. 单卡 fallback

单卡上 reward 必然和 rollout/trainer 共享那一张卡——share + release 分时（MR4 已支持）。
这不是"错"，是 Tier 1 在 GPU=1 时的特例：卡只有一张，分时才能让它一直忙。保留，但
**不当多卡的默认心智模型**。

## 8. 多节点的开放问题（不在本 sprint 落地）

reward 的卡由谁分，两条路：
- **同一个 `GlobalRayPlacementOwner` 给 dedicated bundle**（单机多卡最简单，MR4 已支持）。
- **独立的 reward replica / placement**（多节点、独立扩展与重启更干净，更贴 cosmos 的
  独立 reward 服务）。

倾向：**单机多卡用同一个 owner 的 dedicated bundle；真上多节点再拆独立 reward replica**
（配合 `SPRINT_cosmos_rl_scaling_learnings` 的 controller/NCCL 路线）。本 sprint 不决策。

## 9. Non-Goals

```text
不给 VRL 建独立 reward HTTP 服务（VRL 是 Ray-native，灵活 placement 更合适）
不"所有 reward 都 dedicate 一张卡"（OCR 这种纯 CPU reward 配 GPU 是浪费）
不"所有 reward 都 share"（重且能喂满卡的 reward 该独占并发）
不把 tier 或 4 字段 execution 写成一级配置枚举（tier 是 inference_runtime + reward_cost 派生出来的解释标签）
不把 reward 模型当 rollout placement group 里的"共管 bundle"长期化（那是单卡特例）
不改 GRPO/advantage 数学、不改现有 reward 注册表（vrl/rewards/ 的 registry 保留）
不在没有 reward 成本画像数据前就硬编码 dedicate/share 阈值（P2 先标注后自动测）
```

## 9.1 架构卫生：哪些要改，哪些不要改

应该改：

1. `vrl/scripts/common/factory.py` 里按 reward key 判断 Ray runtime 注入的硬编码。
   现在逻辑等价于：

   ```python
   for key in ("kling_video_reward", "video_reward")
   ```

   这会让下一个 GPU reward 必须改公共 factory。它不是协议边界，也不是稳定 taxonomy；
   应改成按已有的 `inference_runtime` 派发，factory 不再 hardcode reward-key。

   2026-06-14 closeout：Ray placement owner 收尾只记录这个 smell，不在该短 sprint
   展开奖励派发重构；保留为 P2 follow-up，和成本感知 auto placement 一起处理。

2. `vrl/ray/resources.py` 的 `share_with_rollout=None` auto 分支。
   现在 auto 只看 spare GPU；P2 后应消费 `inference_runtime + reward_cost + backlog`，把
   "spare exists" 降级为一个 topology fact，而不是最终策略。

3. `vrl/rollouts/orchestration/continuous/producer.py` 与 collector 的边界。
   现在 continuous 的 in-flight item 是完整 `collect+score`；P1 后应允许
   `generated but unscored` 进入 reward stage queue，再由 scorer stage 产出 trainer-ready batch。

应该保留：

1. `InProcessRewardRuntime`。它是 Tier 0 / `inference_runtime=local` 的协议边界，不是多余薄层。
2. `RayRewardRuntime` / `RayActorMethodRuntime`。它们是 model-backed reward 的 transport
   边界，应继续承载 actor lifecycle、release-after-score、owner-managed placement。
3. `GlobalRayPlacementOwner` / `BundleLayout`。它们是 run-level GPU bundle 的 source of truth，
   取代 per-role PG + offset math；不要为了少文件把它们塞回 resource resolver。
4. `MultiReward` registry。它是 reward composition 的公共 API；P2 增加 reward_cost
   不应改变 weighted reward 的数学语义。

非目标：

- 不为了少几行把 reward runtime 薄层 flatten 掉；这些薄层承担 protocol/interface boundary。
- 不用大写硬编码表维护 reward capability；capability 应来自 reward class/config 的 source of truth。
- 不把 slime/cosmos 的 HTTP 服务形态当默认结论；它们的共同结论是 execution backend 可替换，
  不是所有 reward 都要服务化。

## 10. 验收（设计落地后怎么验）

```text
- 多卡 dedicated reward：reward 卡利用率 > 单纯 burst（异步 backlog 喂满的证据）
- 异步打分：一个 training step 内 reward 打分与下一轮生成 wall-clock 重叠（profiling）
- auto 成本感知：轻 reward(OCR) 永不占 GPU；重 reward(Kling) 在有 spare 且能喂满时才 dedicate
- 新 reward 接入：新增一个 GPU reward 不需要编辑 factory 的 reward-key if/elif
- backend 派生：同一个 reward 在 local/ray 两种 inference_runtime 下不改变 reward math（external 为未来 backend）
- 单卡：share + release 分时仍跑通（MR4 既有测试不回归）
```

## 11. References

读物（证据）：
- `docs/sprints/reading/slime.md`（:161,:697,:788,:797 reward 在事件循环、semaphore 外、remote_rm、rm_hub）
- `docs/sprints/reading/slime.md`（:771-803 reward hub / remote_rm 分支）
- `docs/sprints/reading/cosmos-rl.md`（:11,:431,:928,:943 reward_service 独立包、rollout worker 上算、ProcessPool/ThreadPool、use_remote_reward）
- `docs/sprints/reading/cosmos-rl.md`（:1273-1278 RewardDispatcher / remote reward evidence table）
- `docs/sprints/reading/SPRINT_framework_lessons_vrl.md`（非阻塞 barrier 同源主题）

VRL 代码（落点）：
- `vrl/rewards/runtime.py`（InProcessRewardRuntime —— Tier 0）
- `vrl/rewards/base.py`（_init_disk_artifact_reward —— artifact store + Ray 池）
- `vrl/rewards/ray/runtime.py`（RayRewardRuntime → RayActorMethodRuntime）
- `vrl/ray/resources.py`（RewardResourceConfig.share_with_rollout、_resolve_reward_devices、build_bundle_layout）
- `vrl/ray/placement.py`（reward_placement）
- `vrl/rollouts/collector/core.py`（collect_unscored / score_rollouts —— G1 barrier）
- `vrl/rollouts/orchestration/continuous/`（异步流水线落点）
- `vrl/generation/ray/executor.py`（ref 数据面落点）

## 12. verl-omni 证据：逐样本流式打分（P1 的实现参考）

verl-omni 已经把本 sprint §6 的 P1（异步打分主线）跑成了产品代码：reward **不是**在 rollout 全收完后一次性打的，而是**每个 sample 一产出就立刻 fire 一个 per-sample reward task**，与后续样本的生成并发；同时 reward 拿到**自己独立的 resource pool**，trainer 主循环据此**跳过 colocated 打分**（因为分数已经随 rollout 流回来了）。这正好是本 sprint P1 想要的"group 一产出即异步丢 reward 打分"（§6:256-262）。关键区别要写清楚：**verl-omni 是复用上游 verl 的 reward-loop 机制**（`verl.experimental.reward_loop.RewardLoopManager` + `verl.experimental.agent_loop.AgentLoopManager`），而 **vrl 不引入这套，而是在已有的 `RayRewardRuntime` 里把"整批 `score_batch`"改写成"逐 group 流式 score"**——落点仍是 §6 P1 写的 `collector/core.py` + `orchestration/continuous/`，不新建 reward 服务。**on-policy 边界不变**：流式只改变"何时把分打出来"，policy 更新依旧等整批打完（见下文边界说明）。

### 12.1 逐样本流式打分：per-sample `compute_score.remote`

verl-omni 现在 在 agent loop 里对**单个样本**直接 `await ...compute_score.remote(data)`，且只在异步 reward 开启（拿到 worker handle）时才走这条路：

```python
# diffusion_agent_loop.py:273-277
async def _compute_score(self, output, prompts, responses, kwargs):
    """Compute reward score for single sample."""
    enable_async_reward = self.reward_loop_worker_handles is not None
    if output.reward_score is None and enable_async_reward:
```
```python
# diffusion_agent_loop.py:297-300
selected_reward_loop_worker_handle = random.choice(self.reward_loop_worker_handles)
result = await selected_reward_loop_worker_handle.compute_score.remote(data)
output.reward_score = result["reward_score"]
output.extra_fields["reward_extra_info"] = result["reward_extra_info"]
```
注意打包给 reward 的 `batch` 是 `batch_size=1`（`diffusion_agent_loop.py:280-286`，`prompts [1, L]` / `responses [1, C, H, W]`），即**一个样本一个 task**、从一个 reward worker 池里随机挑 handle——天然把打分摊到 reward 池并和别的样本的生成重叠。这就是 P1 §6:257 "group 一产出即异步丢 reward 打分，与下一轮生成并发" 的具体机制。
（verl_omni/agent_loop/diffusion_agent_loop.py:273-300）

vrl 现状 是**整批、阻塞**：先 generate-all 再 score-all（本 sprint §0 已点名的 G1 barrier）：

```python
# prompt_collection.py:23-26
"""Two phases: generate every prompt group first (the generation runtime stays
resident throughout), then score all groups through one reward call."""
```
而 vrl 真正打分的入口是 `RayRewardRuntime.score_batch`，它接的是**整个 request**、内部按 worker 数 shard 后 `map`：
```python
# rewards/ray/runtime.py:53-64
async def score_batch(self, request: RewardInferenceRequest) -> list[RewardInferenceResult]:
    ...
    shards = shard_reward_request(request, num_shards=self._actor.num_workers)
    ...
    nested = await self._actor.map(shards)
```
P1 的 vrl 落地 = **不改 transport seam，改驱动方式**：让 `orchestration/` 在每个 group `collect_unscored` 完成后立刻对该 group 调 `score_batch`（或新增一个 per-group 的 `score_group`），把结果 future 挂回流水线，而不是攒满 `unscored_groups` 后一次 score-all。`RayRewardRuntime` 已经是 actor 池 + `_actor.map`，**逐 group 喂正是它擅长的**——这对位 verl-omni 的 per-sample handle，只是 vrl 的粒度是 group 而非单样本（vrl 的 reward 模型按 group 算 GRPO 优势更自然）。
（vrl/rollouts/orchestration/prompt_collection.py:23-26,74-80；vrl/rewards/ray/runtime.py:53-64）

### 12.2 disjoint-pool gate：reward 有独立 pool 时 trainer 跳过 colocated 打分

verl-omni 现在 用一个显式开关决定"流式 reward 是否生效"，条件就是**没有 RM、或 RM 拿到了独立 resource pool**：

```python
# ray_diffusion_trainer.py:670-674
enable_agent_reward_loop = not self.use_rm or self.config.reward.reward_model.enable_resource_pool
# if enable_agent_reward_loop, we directly pass reward_loop_workers to agent loop manager
reward_loop_worker_handles = self.reward_loop_manager.reward_loop_workers if enable_agent_reward_loop else None
```
reward 的独立 pool 来自 `ResourcePoolManager` 按 `Role.RewardModel` 取：
```python
# ray_diffusion_trainer.py:646-650
resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel) if self.use_rm else None
self.reward_loop_manager = RewardLoopManager(config=self.config, rm_resource_pool=resource_pool)
```
handle 只在 `enable_agent_reward_loop` 时塞进 agent loop manager（`ray_diffusion_trainer.py:684` 的 `reward_loop_worker_handles=...`）。而 trainer 主循环里的 colocated 打分**带一个 short-circuit**：分数已经随 rollout 流回来（`rm_scores` 已在 batch 里）时就**不再** colocated 打：

```python
# ray_diffusion_trainer.py:1000-1005
with marked_timer("reward", timing_raw, color="yellow"):
    if self.use_rm and "rm_scores" not in batch.batch.keys():
        batch_reward = self._compute_reward_colocate(batch)
        batch = batch.union(batch_reward)
    reward_tensor, reward_extra_infos_dict = extract_reward(batch)
```
读法：流式开启 → reward worker 在自己的 pool 上、和 rollout 并发把 `rm_scores` 填进 batch → trainer 看到 `rm_scores` 已存在 → **跳过** `_compute_reward_colocate`（不在 rollout 卡上分时打分）。这就是本 sprint §6:260 "让 dedicated reward 卡靠持续 backlog 喂满" 的对位：reward 占独立 pool 是流式打分能并发的前提，二者同一个开关。
（verl_omni/trainer/diffusion/ray_diffusion_trainer.py:646-650,670-674,684,1000-1005）

vrl 现状 已经有同样语义的 backend 开关，只是还没接到"流式"上：`make_reward_runtime` 按 `execution: inline|pool` 选 in-process 还是 Ray 池：

```python
# rewards/runtime.py:62-81
def make_reward_runtime(execution: Literal["inline", "pool"], *, model_factory, worker_config=None):
    ...
    if runtime == "inline":
        return InProcessRewardRuntime(worker_cfg)
    if runtime == "pool":
        from vrl.rewards.ray.runtime import RayRewardRuntime
        ...
        return RayRewardRuntime(ray_cfg)
```
对位关系：verl-omni 的 `enable_agent_reward_loop`（要不要流式 + 要不要独立 pool）≈ vrl 的 `execution == "pool"`（reward 在独立 Ray actor 池）。**P1 在 vrl 不需要新开关**——`execution=pool` 已经把 reward 摆到独立池，P1 只要让 orchestration 在 pool 模式下逐 group 流式 `score_batch`；`execution=inline`（单卡 fallback，§7）保持整批阻塞即可。这与本 sprint §6 P2 "复用已有 execution(inline|pool) 当 backend switch" 是同一条线。
（vrl/rewards/runtime.py:62-81）

### 12.3 on-policy 边界：策略更新仍等整批打完

流式打分**不**破坏 on-policy。verl-omni 里逐 sample 打的 `reward_score` 只是更早 ready；真正进训练前，`_postprocess`（`diffusion_agent_loop.py:303` 起）把所有样本 `torch.cat` 回一个 batch，trainer 在 `marked_timer("reward")` 块里统一 `extract_reward(batch)`（`ray_diffusion_trainer.py:1004-1005`）后才进 `old_log_prob`/优势/更新。也就是：**"何时打分"被异步化，"何时用分更新策略"没有**——policy step 依旧消费一个完整、已全部打分的 batch。

对 vrl 的含义：P1 落在 `strict_on_policy`（默认 serial）下时，收益是**把 reward 的 wall-clock 藏到生成后续 group 背后**（§6 验收 §10:356 "reward 打分与下一轮生成 wall-clock 重叠"），但 OnlineTrainer.step 仍等整批 scored batch——on-policy 不变。只有切到 `continuous` 模式、允许版本错配时，才会引入额外 staleness，那由现有 `StalenessPolicy` 管，**不属于 P1 改的语义**。所以 P1 是纯吞吐优化，不是 on/off-policy 的语义切换——这点要在落地时写进 §9.1 的边界说明，避免被误读成"流式 = off-policy"。

### 12.4 小结：抄什么、不抄什么

| 维度 | verl-omni（参考） | vrl 的 P1 落地 |
| --- | --- | --- |
| 流式机制载体 | 复用上游 `verl.experimental.reward_loop` + `agent_loop`（`ray_diffusion_trainer.py:641,661`） | **在已有 `RayRewardRuntime.score_batch` 内重写流式**，不引入 verl reward-loop（`rewards/ray/runtime.py:53-64`） |
| 打分粒度 | per-sample（`batch_size=1`，`diffusion_agent_loop.py:280-298`） | per-group（GRPO 按 group 算优势更自然），逐 group 调 `score_batch` |
| 独立 pool 开关 | `enable_agent_reward_loop`（`ray_diffusion_trainer.py:670`） | 复用已有 `execution=pool`（`rewards/runtime.py:62`），不新增开关 |
| colocated 跳过 | `if use_rm and "rm_scores" not in batch`（`ray_diffusion_trainer.py:1000`） | orchestration 在 pool 模式下不再走 generate-all→score-all（替换 `prompt_collection.py:23-26` 的两段式） |
| on-policy | 更新前 `_postprocess` 把整批 cat 回（`diffusion_agent_loop.py:303`） | OnlineTrainer.step 仍等整批 scored batch，语义不变 |

## Non-Goals（本节追加部分）

- **不引入 verl 的 `RewardLoopManager` / `AgentLoopManager`**。verl-omni 复用上游那套是因为它本就建在 verl 之上；vrl 的 reward transport 已经是 `RayRewardRuntime` + `RayActorMethodRuntime`，P1 在这层内做流式即可，不为对齐 verl 而搬一套 manager 进来。
- **不改打分粒度到 per-sample**。vrl 按 group 流式就够覆盖 P1 收益；per-sample task 数更多、调度开销更大，且与 GRPO group 优势计算的天然边界不符——未验证 per-sample 在 vrl 下更快，不做。
- **不在本节改 on/off-policy 语义**。流式只动"何时打分"；continuous 模式的 staleness 由现有 `StalenessPolicy` 负责，不在 P1 范围。
- **不新增放置/开关配置**。沿用 §6 既定结论：`execution=inline|pool` 就是 backend switch，P1 不加新 config key。

## References（本节追加部分）

读物（证据，本 run 实读）：
- `verl_omni/agent_loop/diffusion_agent_loop.py:273-300`（per-sample `compute_score.remote`、`enable_async_reward` gate、`batch_size=1` 打包）
- `verl_omni/agent_loop/diffusion_agent_loop.py:303-322`（`_postprocess` 整批 cat 回 —— on-policy 边界）
- `verl_omni/trainer/diffusion/ray_diffusion_trainer.py:646-650`（reward 取 `Role.RewardModel` 独立 resource pool → `RewardLoopManager`）
- `verl_omni/trainer/diffusion/ray_diffusion_trainer.py:670-674,684`（`enable_agent_reward_loop` gate、handle 仅在开启时传入 AgentLoopManager）
- `verl_omni/trainer/diffusion/ray_diffusion_trainer.py:1000-1005`（colocated 打分的 `"rm_scores" not in batch` short-circuit）

vrl 代码（落点，本 run 实读）：
- `vrl/rollouts/orchestration/prompt_collection.py:23-26,74-80`（generate-all → score-all，P1 要替换的两段式）
- `vrl/rewards/runtime.py:62-81`（`make_reward_runtime(execution: inline|pool)` —— 对位 `enable_agent_reward_loop`）
- `vrl/rewards/ray/runtime.py:53-64`（`RayRewardRuntime.score_batch` → `shard_reward_request` → `_actor.map` —— P1 流式改写的载体）
## 13. 落地记录（2026-06-28 P1 落地 / 2026-06-29 P2 撤回，CPU-only，未 commit）

本轮 CPU 落地了 P1（流式打分）+ P2（auto 成本感知放置），**评审后两个都撤回**（2026-06-29），代码全部
还原到 committed 状态；只保留下面的分析结论 + 一个顺带的 stale 测试修复。P3（ref 数据面）不在本轮。

**决定性边界发现（撤回 P1 的根据）：** reward 的重叠**早已存在于 `continuous` 模式**——
`ContinuousRolloutProducer` 维持 `max_inflight_groups` 个并发 `_collect_group`（各自 generate+score 一个
group），且 `after_train_step` 的 non-draining weight sync 让 in-flight 的 collect（含 reward）**与训练并发**
（`continuous/schedule.py:130-145`、`producer.py:164-228`）。所以 continuous 里 reward 同时和「其他 group
的生成」+「训练」重叠。**而 P1 改的 `collect_prompt_batches` 在 continuous 下每次只收一个 group
（`producer.py:214` 传 `prompts=[prompt]`）→ P1 的 call 内 streaming 是 no-op。** P1 只对
**strict_on_policy 默认路径**生效，收益是窄窄的 "score(N) ∥ generate(N+1)" call 内重叠 + 仍有硬 train
barrier，且和 continuous 高度重叠（continuous(max_stale=0)≈strict 还自带 producer 重叠）。**P1 没有引入
新能力，niche 太小且未实测，故撤回。** 完整重叠表见 §13.1。

### P1 — 流式打分（曾实现，已撤回 2026-06-29）

曾落点 `collect_prompt_batches` + `RolloutCollector.can_stream_scoring`（按 topology 派生 stream/batch 两条
路），已全部还原。下面记录当时的根因判断，供未来评估同类改动参考。

根因判断（仍成立）：**collect→score barrier 不是无条件错的**。`score_rollouts` 里有个
`_should_release_runtime_before_reward_model()` gate——共享单卡时 rollout 必须先 release 才能让
reward model 占同一张卡（§7 fallback），批量打分把 release + actor cold start 摊一次，是**对的**。
barrier 只在 reward **不和 rollout 争卡**（disjoint pool 或 CPU-inline）时纯属多余串行。所以：

- `can_stream_scoring = not _should_release_runtime_before_reward_model()`——由 GPU topology 派生，
  **不新增 config key**（对位 verl-omni 的 `enable_agent_reward_loop` = reward 有独立 pool）。
- 流式路径：每个 group `collect_unscored` 一完成，立刻 `asyncio.ensure_future(score_rollouts([group]))`
  作为后台 task，与下一个 group 的生成并发；末尾 `await` 收齐，按输入序 remap+split。
- 批量路径（共享单卡）：完全保留原行为（generate-all → release → score-all）。

**on-policy 不变**（§12.3）：流式只改"何时打分"，`collect_prompt_batches` 仍在返回前收齐全部已打分
batch，trainer.step 消费的还是完整 scored batch。**分数正确性由构造保证**：asyncio 单线程协作式，
sync inline reward 在两次 await 间原子执行、async pool reward 只在 Ray await 处让出，分数按值返回——
只有 reward_fn 的 telemetry（last_results）在并发 pool 打分下可能 stale，已在 docstring 标注 best-effort。

测试（`tests/rollouts/orchestration/test_prompt_collection.py` + `tests/rollouts/collector/test_runtime.py`）：
- `test_streaming_overlaps_scoring_with_next_generation`：用 `asyncio.Event` 做**确定性 overlap 证明**——
  group1 的生成阻塞到 group0 打分开始，只有流式（打分∥生成）才不死锁；断言 `score:p0` 早于 `generate:p1`
  + 每 group 一次 score 调用 + 结果与批量路径逐 group 等价、有序。
- `test_streaming_accumulates_per_group_phase_times`：每 group 自带 score/build 计时累加（无重复计 wall）。
- `can_stream_scoring` 属性：shared-GPU lifecycle → False（保持批量）；disjoint lifecycle → True（放行流式）。

### P2 — auto 放置消费 `reward_cost`（成本感知）—— 已撤回（2026-06-29）

曾落地 `RewardResourceConfig.reward_cost` + `_parse_reward_cost` + `_reward_can_saturate_card` +
`_resolve_reward_devices` auto 分支 cost gate + 阈值 `_REWARD_SATURATION_COST=1.0`（cost<1→share、
cost>=1→dedicate、auto→保持 topology 默认），全部撤回。撤回理由：

1. **和已有的显式 `gpu_pool: rollout` 重复**：用户知道 reward 轻，直接显式写 rollout 就行；手动 cost
   knob 没给独立价值。`reward_cost` 真正值钱的是**自动测量**（per-item 打分耗时 ÷ 生成耗时自动回填），
   而那半要 GPU run，没做——所以现形态是给未来自动测量留的脚手架，独立看价值很薄。
2. **可能和 P1 互相抵消**：单卡上 rollout 和 reward 内存装不下（`release_rollout_before_reward`），
   "share" 其实是卸载→打分→装回的串行，不是真并发。把轻 reward 从 dedicated（P1 流式能藏到生成背后）
   改成 share，反而把藏好的 reward 时间又变回串行 bubble。cost<1→share 这条规则太粗。
3. 一个用户看得见、却基本是 no-op 的配置键，正是 AGENTS.md 点名要避免的。

净结论：轻 reward 的干净答案是 **Tier 0（纯 CPU、进程内、`execution=inline`，根本不碰 GPU）**；GPU 轻
reward 这个中间档本就没有干净的纯放置答案，要靠"深 backlog 喂满"或"多 reward 混着填卡"，那都是 ≥2-GPU
的事，不是一个 cost 阈值能解决。`reward_cost` 等真能上多卡自动测时再建。

### 最终状态（P1 + P2 都撤回后，2026-06-29）

P1/P2 的源码与新增测试全部还原到 committed 状态（`vrl/ray/resources.py`、`vrl/rollouts/collector/core.py`、
`vrl/rollouts/orchestration/prompt_collection.py` 及其测试 git 无 diff）。**唯一保留的代码改动**是一个
独立于 P1/P2 的 stale 测试修复：`test_collect_phase_timings_are_per_call_not_shared` 用的环境变量
`VRL_PROFILE_COLLECT` 在 2026-06-27 重命名为 `VRL_PROFILE` 时漏改（源码读 `VRL_PROFILE`），改正后转绿。

```text
pytest tests/rollouts/collector/test_runtime.py
  tests/rollouts/orchestration/test_prompt_collection.py   -> 16 passed（env 修复 + 原有 deferred-scoring 测试）
```

注：rebase 后发现 `tests/scripts/test_online_lifecycle.py` 里 FSDP 策略守卫测试仍按旧预期要求阻断
`fsdp`；当前 recipe/launcher 已支持 symmetric colocated FSDP 路径，测试已改为验证 `fsdp` 放行。

### 仍未做（P3 + 后续）

- P3 ref-based 数据面（rollout 产物走 ref，reward 直接 pull，省 driver 往返）——未动。
- `reward_cost` 整个知识点（含自动测量）——已撤回，等能上多卡自动测时再建（见 P2 小节）。
- 流式的 ≥2-GPU 吞吐验收（§10 "reward 卡利用率 > burst"、"打分与下轮生成 wall-clock 重叠"）——需多卡实测。

### 13.1 reward 到底能和什么重叠（评审确认，带证据）

| 模式 | reward ∥ 其他 group 生成？ | reward ∥ 训练？ | 机制 / 证据 |
|---|---|---|---|
| **continuous**（只允许 trainer/rollout 分卡的 async 路径） | **是** | **是** | pre-launch topology validator 拒绝共卡与 reward mid-iteration handoff；独立 owner 维持 `max_inflight_groups` 个并发 `_collect_group`（各跑 generate+score 一个 group），non-draining weight sync 让 in-flight collect 与训练并发。 |
| **strict_on_policy**（默认、严格 on-policy、cosmos 用） + 共享单卡 | 否 | 否 | `release_rollout_before_reward=True` → 卸载 rollout 才能打分；generate-all→release→score-all→train 全串行。P1 在此 `can_stream_scoring=False`，no-op（保留批量是对的） |
| **strict_on_policy** + reward 独立卡 + **P1** | **是（仅 call 内 group 间）** | 否 | P1 让 `collect_prompt_batches` 里 score(group N) 作为后台 task ∥ generate(group N+1)；但 collect 之后仍硬 train barrier |

读法：
- **想要 reward 和「生成 + 训练」都重叠，continuous 模式早就给了**（代价是 staleness，由 `StalenessPolicy` 管）。
- production continuous 要求 `max_stale_policy_versions>=1`；零 staleness 必须使用
  `strict_on_policy`，不能再用 continuous 的后台队列伪装成 async。
- **P1 唯一独占的 niche**：strict_on_policy + reward 独立卡 + 多 group + 坚持零 staleness（不切 continuous）——
  此时 P1 给 call 内 score∥next-gen 重叠（不含训练）。窄、未实测、且和 continuous 能做的高度重叠。
- 净判断：**P1 没有引入 reward async（早有）；它只是给 strict 默认路径补一条 call 内的窄重叠。** 是否值得保留，
  取决于是否真有"strict + 独立 reward 卡 + 不接受 continuous staleness"的实跑场景；若没有，P1 也可一并 park。
