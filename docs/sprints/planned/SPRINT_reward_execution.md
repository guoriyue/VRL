# SPRINT: Reward execution —— 成本分层放置 + 异步打分

状态：design / not-started —— P1/P2/P3 全部未实现（P1 硬 barrier 仍在 collector/core.py:107-108 与 prompt_collection.py:23-26,74-80 的 collect-all→score-all；P2 `reward_cost` 全仓 0 命中、_resolve_reward_devices auto 仍只看 spare GPU（resources.py:814-823）；P3 无 ref 数据面）。注意 §4.1/§4.2/§6/§10 引用的 `inference_runtime: local|ray` 已被 8de3e63 改名为 `execution: inline|pool`，该核心抽象论证需按新 key 重写。

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
  vrl/rewards/runtime.py  LocalRewardRuntime（OCR 等函数 reward 在采集进程内打分）
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
inference_runtime=local              → 无 GPU，LocalRewardRuntime / CPU
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

1. `LocalRewardRuntime`。它是 Tier 0 / `inference_runtime=local` 的协议边界，不是多余薄层。
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
- `vrl/rewards/runtime.py`（LocalRewardRuntime —— Tier 0）
- `vrl/rewards/base.py`（_init_disk_artifact_reward —— artifact store + Ray 池）
- `vrl/rewards/ray/runtime.py`（RayRewardRuntime → RayActorMethodRuntime）
- `vrl/ray/resources.py`（RewardResourceConfig.share_with_rollout、_resolve_reward_devices、build_bundle_layout）
- `vrl/ray/placement.py`（reward_placement）
- `vrl/rollouts/collector/core.py`（collect_unscored / score_rollouts —— G1 barrier）
- `vrl/rollouts/orchestration/continuous/`（异步流水线落点）
- `vrl/generation/ray/executor.py`（ref 数据面落点）
