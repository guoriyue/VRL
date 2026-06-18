# SPRINT: Scaling Learnings from cosmos-rl

状态：research/proposed（基于 `~/Desktop/cosmos-rl` 源码精读,2026-06-03)。本 sprint 是"学什么 + 怎么往 vrl 加"的路线图,不含实现。

> **更新（2026-06-18，对照已落地的执行臂复核）**：本 roadmap 的 trainer-多卡那条（§0 #1 / §2.1）
> 当时建议"先 FSDP2"。**之后 trainer 多卡的实际落点改成了 DDP（symmetric colocated）**——当前模型是
> **2B + LoRA、单卡放得下**,DDP 每卡复制全量、只 all-reduce LoRA 梯度,比 FSDP 每次 forward all-gather
> 冻结 base 更快更简单;**FSDP2 只在模型单卡塞不下（≥7-13B）时才需要**。DDP / FSDP2 两条策略层都已落地、
> CPU 全绿、online 多-rank 编排都 gated 在硬件上（`online.py:_require_supported_online_strategy` 放行
> single_process+ddp、fail-fast fsdp）。详见
> [`SPRINT_symmetric_colocated_ddp.md`](../planned/SPRINT_symmetric_colocated_ddp.md)（DDP,Slice 1-4 已实现）
> 与 [`SPRINT_multi_gpu_training.md`](../parked/SPRINT_multi_gpu_training.md)（FSDP2 策略层）。
> **"怎么花每张卡"（throughput vs overlap）+ DiffusionNFT 的 async 约束见新增 §3a。**

## 0. Core Decision

一句话:**vrl 的 continuous-rollout 编排在逻辑上已经等同 cosmos-rl 的异步设计(producer/queue/consumer + staleness bound + weight-sync barrier),差距是物理层面的——分离进程/GPU、NCCL 权重同步、FSDP 多卡。** cosmos-rl 教给我们的不是"换架构",而是几个可单独移植的能力,按"对当前单卡 diffusion RL 的价值 ÷ 移植成本"排序:

1. **trainer 多卡**(最高价值、最可行;**当前 2B+LoRA 走 DDP,FSDP2 留给塞不下的模型**——见 §2.1 更新)——cosmos-rl 的 `diffusers/parallelize.py` 是一份 ~40 行、**FSDP-only、可直接照抄**的模板,模型变大要分片时直接用。
2. **NCCL 权重同步**(替掉"Ray 推 CPU state_dict")——control plane 走 HTTP/Redis、data plane 走 GPU↔GPU NCCL。`pynccl.py` 几乎可整段搬。
3. **分离 policy/rollout 进程**(真并行,而非单事件循环的交错)——vrl 已有逻辑骨架,缺的是物理分离。
4. **FP8 训练**(建立在你刚落地的 precision config 上)——torchao 一行 module-swap,机械活小、数值调参尾巴长。
5. **Single-controller / elastic NCCL**(只有上多节点才需要)——registry + 命令总线 + heartbeat + 动态 NCCL group。最重,最后做。

cosmos-rl 的定位是**多节点、上千卡、LLM+diffusion**;vrl 是**单卡/少卡、diffusion/video**。所以不是全盘照搬——下面每条都标了"对 vrl 值不值"。

## 1. cosmos-rl 架构速览 + 与 vrl 的映射

cosmos-rl = **一个中心 controller(FastAPI + 内嵌 Redis)+ 自注册的 Policy(训练)replicas + Rollout(生成)replicas**。control plane 走 HTTP(register/heartbeat/next_prompt/rollout/train_ack)+ Redis streams(typed `Command`);**data plane(权重张量)完全走 NCCL,不碰 CPU/broker**。

| cosmos-rl | vrl 现状 | 差距 |
| --- | --- | --- |
| Controller(中心 registry + 命令总线 + heartbeat) | `OnlineTrainer` 事件循环 + `RayGenerationLauncher`(无中心 registry) | 无 elastic/fault-tolerance |
| Policy / Rollout 分离进程(可不同 GPU SKU) | 单进程 trainer + Ray 生成 actor(同事件循环编排) | 单事件循环只**交错**不真并行 |
| 异步 producer/consumer + `allowed_outdated_steps`(3 层 staleness) | `continuous/{producer,queue,consumer,staleness}.py`(**已有,逻辑等价!**) | ✅ 对齐,只是单进程 |
| 权重同步:NCCL P2R unicast → R2R broadcast(GPU→GPU) | `weight_sync.py:RayRuntimeWeightSyncer.push`(**CPU state_dict 经 Ray 推**) | ← 主要瓶颈 |
| `ParallelDims` + `DeviceMesh` + FSDP2/TP/CP/PP | **无**(单卡) | ← 多卡主要缺口 |
| FP8/FP4(torchao / TransformerEngine / vLLM) | 刚落地 precision config(bf16) | FP8 是下一步 |

**重要结论**:vrl 的 `continuous/` 编排(我会前解释过的 producer→bounded queue→consumer + `StalenessPolicy` + pause/drain weight-sync barrier)和 cosmos-rl 的异步设计是**同一套思想**。cosmos-rl 的 `RolloutTaskScheduler`(`max_concurrent_requests`、`paused()` 排空再同步、per-sample weight-version 标记)几乎是 vrl producer 的镜像——**这验证了 vrl 的设计方向是对的**。

## 2. 逐特性:cosmos-rl 怎么做 / vrl 怎么加

### 2.1 [Tier 1·最高价值] FSDP2 多卡

> **现状更新（2026-06-18）**：trainer 多卡的实际落点是 **DDP（symmetric colocated），不是 FSDP2**——
> 2B+LoRA 单卡放得下,DDP 拿的是**数据并行吞吐**（每卡全量、只 all-reduce LoRA 梯度,**不省显存**）,比
> FSDP 每次 forward all-gather 冻结 base 更快更简单。`DDPStrategy` 已落、CPU 全绿
> （`tests/trainers/test_ddp.py` 7 passed）;online 多-rank 编排 gated 在硬件上。FSDP2 策略层
> （`FSDPStrategy`/fully_shard/DTensor）也已落、同样 online-gated,**仅在模型单卡塞不下时才是首选**。
> 下面这套 cosmos `diffusers/parallelize.py` 模板**仍是模型变大要上 FSDP 时的照抄对象**,只是不是当前 2B
> 的选择。详见 [`SPRINT_symmetric_colocated_ddp.md`](../planned/SPRINT_symmetric_colocated_ddp.md) /
> [`SPRINT_multi_gpu_training.md`](../parked/SPRINT_multi_gpu_training.md)。

**cosmos-rl 怎么做**:`ParallelDims`(`utils/parallelism.py`)持 5 个并行度,`init_device_mesh` 建 N-D mesh;每个模型暴露 `parallelize_fn`,在 **meta device** 上建模 → 应用并行 → 物化。**diffusion 走的是 FSDP-only 路径**(`policy/model/diffusers/parallelize.py`):按 `transformer_blocks` 逐块 `fully_shard(block, mesh, MixedPrecisionPolicy)`,根上再 `fully_shard`,明确 `assert pp_size==1`、无 TP。就 ~40 行。

**对 vrl 值不值**:**最值**。这正是 backward-efficiency sprint 的"唯一实质杠杆"+ 你申请多卡的目的。Agent 评估:**2.5B 模型上 FSDP2 ≈ 一天 plumbing**(纯 FSDP 不需要 cosmos 那套 5 轴 `ParallelDims`,一维 mesh 足够)。

**怎么加**(落地点 `vrl/trainers/fsdp.py` + sd3_5 model):
1. 建进程组 + 1-D `DeviceMesh(("dp_shard",))`。
2. 照抄 `diffusers/parallelize.py`:遍历 `pipeline.transformer.transformer_blocks` 逐块 `fully_shard` + `MixedPrecisionPolicy`(直接接你的 `precision` config!),尊重 `_no_split_modules`。
3. **三个真实 blocker**(Agent 明确):
   - **LoRA + FSDP2**:必须**先注入 LoRA 再 shard**,让 adapter 在分片单元内;FSDP2 支持单元内 mixed `requires_grad`(FSDP1 不行)。cosmos 的做法是物化后 `reinitialize_lora_params`。
   - **meta-init + 物化**:别 `from_pretrained().cuda()`(那等于先把整模塞一张卡,白搭)。要 meta 建模 → shard → `to_empty` → 把权重写进 `param.to_local()` 分片(cosmos `load_hf_weights` 就这么干)。
   - **checkpoint**:变成 DTensor state dict(`torch.distributed.checkpoint`);optimizer 必须在 shard **之后**建。
4. **不要碰 TP/CP/PP**:Agent 明确,2.5B+LoRA 下 TP 要求每层 sharding plan(`tp_plans.py`)+ 模型按 plan 重写,而且 **cosmos 自己在 GRPO+LoRA 下禁用 TP**(`tp_size==1`)。TP/CP/PP 是 ≥7-13B 或超长序列才值;authoring 成本(不是 torch API)才是真代价。

参考:`cosmos_rl/policy/model/diffusers/parallelize.py`(最贴你的模板)、`utils/parallelism.py`、`policy/model/base.py`。

### 2.2 [Tier 1·高价值] NCCL 权重同步(替掉 Ray CPU-state_dict 推送)

**cosmos-rl 怎么做**:权重 **GPU→GPU 走 NCCL**,从不碰 CPU/broker。两跳:一个 policy replica **NCCL-send** 给一个 leader rollout(P2R unicast),leader 再 **NCCL-broadcast** 给其余 rollout(R2R)。NCCL communicator 按需建——rank0 `ncclGetUniqueId` → 经 controller 把 UID 交换给对面(`post_nccl_comm_initiator/acceptor`)。`parallelism_map.py` 预算每个 rank 收发哪些分片,所以**训练侧 FSDP/TP 布局 ≠ 生成侧 vLLM TP 布局也能同步**(各收各的 slice,无全量 all-gather)。可选 `WeightSyncThread`(独立 CUDA stream)让传输和生成重叠。

**对 vrl 值不值**:**高**——尤其 cross-node。vrl 现状 `RayRuntimeWeightSyncer.push(state_dict)` 每步交四笔税:GPU→CPU 拷贝、cloudpickle 序列化、过 Ray broker、CPU→GPU 重载;多 GB 模型下能吃掉大半 step 时间,还 O(#gen-actor) 线性放大。

**怎么加**(最小路径,Agent 给的):
1. 控制消息继续走现有 Ray/编排;加一个 rendezvous:让训练侧 rank0 `ncclGetUniqueId` 把 bytes 交给生成侧(复刻 `post_nccl_comm_initiator/acceptor`)。`cosmos_rl/utils/pynccl.py` 是自包含 ctypes wrapper,**几乎可整段 lift**。
2. 建一个跨"训练 GPU + 生成 GPU"的 NCCL communicator,把 CPU-state_dict-through-Ray 换成逐 tensor `nccl_broadcast`。**光这一步就去掉了拷贝+序列化+broker 三笔税**。
3. 布局不同再加 `WeightSyncInstructionsGroup` 分片指令。
4. 只有要 elastic 才上 `HighAvailabilitylNccl`(见 2.5)。

落地点:`vrl/trainers/weight_sync.py`(`RayRuntimeWeightSyncer`)+ `vrl/generation/ray/weight_sync.py`。参考:`cosmos_rl/utils/pynccl.py`、`collective/collective.py:P2RCollectiveManager`、`utils/parallelism_map.py`。

### 2.3 [Tier 2] 分离 policy/rollout 进程(真并行)

**cosmos-rl 怎么做**:Policy 和 Rollout 是**不同进程、不同 GPU(甚至不同 SKU——生成用 L40,训练用 H100)**,各自独立 `torchrun`/并行配置/scale。rollout 不停拉 prompt 生成,policy 不停拉完成的 rollout 训练,**互不阻塞、真并行**。

**对 vrl 值不值**:**中**。Agent 关键点:vrl 单事件循环的 producer/consumer 只能**交错**(同 CUDA context + GIL),一个长 backward 和一个长 generate **永远只交替、不真重叠**;cosmos 的"rollouts never sleep while policy trains"必须靠物理分离。vrl 的 `ContinuousRolloutSchedule` 已带 `require_separate_gpus`,方向对——但目前生成仍在 Ray actor、训练在驱动进程同一 loop 编排。

**怎么加**:这是把 vrl 现有的 continuous 编排"物理化"——生成 actor 独占 GPU、训练独占 GPU,中间靠 2.2 的 NCCL 同步权重 + 现有 bounded queue 传 rollout。**先做 2.1+2.2,这条自然水到渠成**(它依赖 NCCL 同步才有意义)。可借鉴 `RolloutTaskScheduler` 的 `max_concurrent_requests` 在途上限、`paused()` 排空再同步。参考:`rollout/worker/asynchronous/rollout_task_scheduler.py`、`rollout/worker/weight_sync.py:WeightSyncThread`。

### 2.4 [Tier 2] FP8 训练

**cosmos-rl 怎么做**:`enable_fp8` → 用 **torchao** `convert_to_float8_training(model, Float8LinearConfig.from_recipe_name("rowwise"), module_filter_fn)` 把合格 `nn.Linear` 换成 FP8 linear,**动态 rowwise scaling**,master weight/optimizer 仍 bf16/fp32。敏感层(`lm_head`、vision tower)按 FQN 过滤掉,只转 dim%16==0 的。rollout 侧 FP8 **不是 cosmos 写 kernel**——是给 vLLM `quantization="fp8"` + monkey-patch 走 `torch._scaled_mm`;权重同步时 rollout **动态把收到的 bf16 权重重量化**成 FP8。

**对 vrl 值不值**:**中**,且**建立在你刚落地的 precision config 上**(FP8 是 `compute` 轴再降一档)。Agent 评估:机械集成 ≈ 一天,但**数值调参尾巴是多天**。

**怎么加**(diffusion 特有的坑,Agent 强调):
1. meta device 上 `convert_to_float8_training(...)`,在 FSDP wrap 之前。
2. **FQN 过滤是成败关键**:DiT/MMDiT 的 timestep/AdaLN/modulation 投影、最终输出投影、patch-embed、喂 softmax 的 q/k/v 比 LLM MLP 敏感得多——量化 modulation 或最终层通常直接毁图。要自己 tune `filter_fqns`。
3. **数值退化在 diffusion 里更难发现**:没有 token-level loss 报警,artifact 在多步去噪里累积——必须用**生成样本质量/FID 做 FP8-vs-bf16 A/B**,不能只看 loss。
4. rowwise 起步;`tensorwise + FSDP2 all-gather` cosmos 自己都还有 bug(禁了 `enable_fsdp_float8_all_gather`)。
5. 硬件门槛 CC≥8.9(Ada/Hopper+)、torch≥2.7。FP8 省的是**吞吐不是显存**(master weight 仍高精度)。

参考:`utils/fp8/fp8_util.py`、`utils/model_converter.py`、`policy/trainer/llm_trainer/llm_trainer.py:91`、`rollout/vllm_rollout/monkey_patch_for_fp8.py`。先决条件:你的 `precision` config 加一档 `fp8`。

### 2.5 [Tier 3·只有上多节点才需要] Single-controller + elastic NCCL

**cosmos-rl 怎么做**:中心 controller 持全局 registry,replica 自注册 + heartbeat;typed `Command`(`BuildMesh`/`P2RUnicast`/`R2RBroadcast`/`DataFetch`)经 Redis stream 下发;`HighAvailabilitylNccl` 在成员变化时 `ncclCommAbort` 旧 group → 建新 group。掉一个 replica(heartbeat 丢失/NCCL timeout)→ unregister → `BuildMesh` 重组,训练继续;加一个 replica 同路径。

**对 vrl 值不值**:**低**,除非真上多节点/上千卡。Agent 给的最小版:把现有 asyncio trainer 当 controller,但 (a) 把 Ray actor 静态 handle 列表换成**带 heartbeat 超时的命名 registry**,(b) `actor.generate.remote()` 换成**typed command 放共享队列**,(c) 每个 prompt batch 打 **weight_version** 标记、controller 中心管 prompt-fetch + rollout-buffer + throttle。这三件给你"可寻址 worker + 显式命令 + elastic 重组"——其余(FastAPI、内嵌 Redis、shard mapper)是上多节点才需要的。

参考:`dispatcher/{controller,status,command,replica}.py`、`utils/distributed.py:HighAvailabilitylNccl`。

## 3. 优先级路线图(对 vrl)

```
Tier 1（直接服务"多卡"目标,和 AWS quota 同步推进）
  1. trainer 多卡       ← 当前 2B+LoRA=DDP（已落 Slice 1-4）;FSDP2 留给单卡塞不下的模型（抄 diffusers/parallelize.py + MixedPrecisionPolicy）
  2. NCCL 权重同步      ← lift pynccl.py;替掉 Ray CPU-state_dict push
Tier 2（Tier 1 之上自然延伸）
  3. 分离 policy/rollout 进程（依赖 #2 的 NCCL 同步）
  4. FP8 训练（接 precision config 加 fp8 档;重在 FQN 过滤 + 样本质量 A/B）
Tier 3（只有上多节点才做）
  5. single-controller（registry + 命令总线 + heartbeat）
  6. elastic / 动态 NCCL group（HighAvailabilitylNccl 式 abort→rebuild）
```

建议:**trainer 多卡先用 DDP**——2B+LoRA 单卡放得下,DDP 拿的是**数据并行吞吐**(不省显存,每卡仍全量),已落 Slice 1-4、正在 2×1 首验;**FSDP2 等模型单卡塞不下(≥7-13B)再上**,那时分片才解决显存(~一天 + checkpoint/LoRA 收尾)。#2(NCCL 权重同步)紧随(让多卡间权重同步不走 CPU)。#3/#4 看吞吐需求再上。#5/#6 等真要跨节点弹性时再说。

## 3a. "disaggregate 做对" 到底指什么:throughput vs overlap（DiffusionNFT 锁定下）

"disaggregate" 在这里其实是**三件被混在一起的事**,对 DiffusionNFT 结论不同,必须拆开:

| 维度 | 是什么 | 卡用来 | 对 DiffusionNFT | 执行臂 |
|---|---|---|---|---|
| **吞吐（数据并行）** | N 卡都训同一 policy,各 collect,一次同步 DDP/FSDP optimizer step。 | **并行训练** | ✅ sound、算法无关,直接拿。DDP 已落（2B+LoRA）;FSDP 等塞不下再上。 | [`SPRINT_symmetric_colocated_ddp.md`](../planned/SPRINT_symmetric_colocated_ddp.md) |
| **物理分离 trainer/rollout（D1）** | trainer 常驻一卡、rollout actor 常驻另一卡,**同步/on-policy**。买:独立 scale、无 offload 抖动、更大 rollout batch、无显存争抢。 | 训练卡 + 生成卡（时间不重叠） | ✅ on-policy 下 sound;需 ≥2 卡;放置面已支持。 | [`SPRINT_placement_surface_disaggregated_default.md`](../planned/SPRINT_placement_surface_disaggregated_default.md) |
| **async overlap（D2）** | 在 D1 之上让"训第 N 步"与"采第 N+1 步"真重叠 → rollout 必然 off-policy/stale。 | 两卡时间也重叠 | ❌ **不可证安全**:DiffusionNFT likelihood-free,无 `exp(logp_new−logp_old)` 比值,TIS/AIPO/PPO-clip 一个都搬不过来。需 ≥2 卡 + 实测"伤不伤"。 | [`SPRINT_async_rollout_train_overlap.md`](../parked/SPRINT_async_rollout_train_overlap.md)、[`SPRINT_shadow_model_weight_sync.md`](../planned/SPRINT_shadow_model_weight_sync.md) |

**从 cosmos-rl 学到的最硬一条**:cosmos-rl 自己的 **LLM/GRPO 默认 `mode=disaggregated` + async**（D1+D2,靠
IS 比值 + `allowed_outdated_steps`≈4 兜底）;但它**自己的 diffusion NFT 配方出厂是 `mode=colocated`
（serial、on-policy by construction）**——**连 NVIDIA 都没把 diffusion NFT 做成 async overlap**。这正面
印证 §1/§5:对 likelihood-free 的 DiffusionNFT,"disaggregate 做对" = 做 **吞吐 + D1（on-policy 物理分离）**,
**不做 D2**;D2 只有换回有 IS 比值的 GRPO 才解锁（用户已锁 DiffusionNFT,不换）。

**按 GPU 预算的落地决策**:

| 卡数 | 做什么 | 为什么 |
|---|---|---|
| **1 卡** | 单卡 colocated 时分;先证明 **on-policy DiffusionNFT 真能学**（固定 eval / lr=1e-4,block-test 曲线） | 真 overlap 单卡做不到（显存墙,只切显存不切时间） |
| **2 卡（你现在）** | **首选 DDP 吞吐**（2×1,已落,正在首验）;证明会学后再加 **D1 分卡 on-policy**（trainer 一卡 + rollout 一卡,`max_stale=0`） | DDP 是 sound 白捡的吞吐;D1 给独立 scale + 大 batch,不冒 off-policy 险 |
| **≥2 卡 + 想要 overlap** | 才碰 **D2**:`max_stale=1` vs strict,同 seed/prompt/reward 实测偏差（parked doc Option A） | 纯 config、零算法改动,把"理论说没补偿"量成"实际伤不伤";伤了别开 |
| **更大模型 / 更多卡** | FSDP2（塞不下才上,§2.1）+ NCCL 权重同步（替 Ray CPU-state_dict,§2.2） | 见 §2.1 / §2.2 / §3 |

一句话:**先把卡花在吞吐（DDP）和 on-policy 物理分离（D1）上——这两样对 DiffusionNFT 都 sound;async
overlap（D2）是最后、需多卡、且对 DiffusionNFT 要先实测才敢开的东西。**

## 4. Ray 去留与扩展性边界

**结论:Tier 1/2 不去 Ray;去 Ray 是 Tier 3(上千卡 + elastic)才评估的一次性大重写。**

cosmos-rl 的"没有 Ray"是它**自研 controller** 的结果(瞄准上千卡 + "零外部编排依赖"),不是这些技术的前提。它真正的 lesson 是 **control plane / data plane 分离**——控制消息走轻量通道,重张量走 GPU→GPU NCCL。vrl **已有 control plane**(Ray + asyncio);问题是把重张量也塞进了 control plane(`RayRuntimeWeightSyncer.push` 经 object store 推 CPU state_dict)。修法是**加一条 NCCL data plane**,不是换掉 Ray。

**两个收益和 Ray 共存**:
- **FSDP2** → trainer 改成 **Ray Train**(`ray.train.torch`)拉的多进程 worker group,FSDP 跑在里面(Ray 原生支持)。Ray 不动,trainer 结构变。
- **NCCL 权重同步** → Ray 继续管 actor 生命周期/放置;权重张量走 actor 间 NCCL(`ray.util.collective` 或手动 UID 交换),**绕开 object store**。这就是 cosmos 的 control/data 分离,只是 **Ray 当 control plane**。

**"Ray 能上千卡吗?"分两层答**:
- **Ray 这个系统:能**,有生产先例(OpenAI 大规模 RLHF / Anyscale 上万核)。
- **vrl 当前用法:不能**——两个瓶颈在远早于上千卡时就卡:
  1. **单进程 asyncio trainer 当唯一 driver/协调者**(GIL 绑定、一个 loop 串行发上千路 RPC)。
  2. **权重经 object store 推**(多 GB × O(N) 个生成 actor,几十卡就开始疼)。
  → 当前架构瓶颈约在**几十–一两百卡**显现,到不了上千。

**扩展性边界 + 修法**:

| 规模 | 状态 / 修法 | 要不要去 Ray |
| --- | --- | --- |
| 1–几十卡(1–几节点) | Ray 完全够,别过早优化 | 否 |
| 几十–几百卡 | 权重换 **NCCL**(control/data 分离),Ray 当 control plane → 撑住 | 否(Ray 内可做) |
| 上千卡 + elastic 容错 | Ray 单 driver + 静态 handle 模型绷;cosmos DIY controller(动态 NCCL group / 掉卡重组)有真优势 = **Tier 3 大重写**,届时要把 Ray 的编排自己补回来 | 那时才评估 |

**真正逼你去 Ray 的是"协调模型 + 弹性容错",不是算力本身**:权重那条在 Ray 内就能修(NCCL,立竿见影);协调那条(中心 driver 持 actor handle)只在极大规模才绷,而 cosmos 的"自注册 replica + 命令总线 + heartbeat"天生为这个设计。

**建议**:① 先在 Ray 内把**权重同步换 NCCL**(成本小、把"几十卡就卡"提到"几百卡舒服");② FSDP 用 Ray Train,Ray 留着;③ 只有真要**跨多节点、上千卡、掉卡自愈**时,才评估换 cosmos 式 controller(=去 Ray)。一句话:**Ray 不是上千卡的天花板,你当前的"单 driver + object-store 权重同步"才是;先修后者,Ray 能陪你走很远。**

## 5. 明确不抄的(Non-Goals)

- **TP / CP / PP**:2.5B+LoRA+diffusion 下不值——要求模型按 per-layer plan 重写,cosmos 自己在 GRPO+LoRA 下都禁 TP。≥7-13B 再议。
- **cosmos 整套 `ParallelDims`(5 轴)**:纯 FSDP 一维 mesh 够,别引入 5 轴编排的复杂度。
- **FastAPI controller + 内嵌 Redis + shard mapper**:单/少卡用不上;vrl 的 asyncio + Ray 编排已够。
- **vLLM FP8/FP4 rollout monkey-patch**:那是 LLM 推理后端的事;vrl 的 diffusion 生成路径不同。
- **改 GRPO 数学 / 现有 continuous 编排**:vrl 的 producer/queue/consumer + StalenessPolicy 已和 cosmos 对齐,保留。

## 6. 关键参考文件(cosmos-rl,供实现时查)

```text
# 并行 / FSDP（Tier 1）
cosmos_rl/policy/model/diffusers/parallelize.py   # FSDP-only 模板,最贴 vrl
cosmos_rl/utils/parallelism.py                    # ParallelDims / DeviceMesh（取一维即可）
cosmos_rl/policy/model/base.py                    # parallelize_fn / load_hf_weights 契约
cosmos_rl/policy/trainer/llm_trainer/llm_trainer.py  # meta-init → parallelize → 物化 调用点
# NCCL 权重同步（Tier 1）
cosmos_rl/utils/pynccl.py                         # ctypes NCCL wrapper,可整段 lift
cosmos_rl/collective/collective.py               # P2RCollectiveManager
cosmos_rl/utils/parallelism_map.py               # 跨布局分片指令
cosmos_rl/rollout/worker/weight_sync.py          # WeightSyncThread（重叠传输）
# 异步分离（Tier 2）
cosmos_rl/rollout/worker/asynchronous/rollout_task_scheduler.py
docs/async/overview.rst
# FP8（Tier 2）
cosmos_rl/utils/fp8/fp8_util.py ; utils/model_converter.py ; docs/quantization/fp8.rst
# controller / elastic（Tier 3）
cosmos_rl/dispatcher/{controller,status,command,replica}.py
cosmos_rl/utils/distributed.py:HighAvailabilitylNccl
```

## 7. vrl 侧落地点(touch points)

```text
vrl/trainers/fsdp.py                       # #1 FSDP2 落地（建 mesh + 抄 parallelize）
vrl/models/diffusion/sd3_5/model.py        # #1 meta-init + 逐块 fully_shard + LoRA-before-shard
vrl/config/precision.py                    # #1 MixedPrecisionPolicy 接 compute 轴；#4 加 fp8 档
vrl/trainers/weight_sync.py                # #2 NCCL 同步替 RayRuntimeWeightSyncer.push
vrl/generation/ray/weight_sync.py          # #2 生成侧 NCCL recv
vrl/rollouts/orchestration/continuous/     # #3 物理化（已是逻辑等价骨架）
```

## 8. 相关 VRL sprint（本 roadmap 的执行臂）

本文是"学什么 + 怎么往 vrl 加"的总路线图;具体落地分散在这些 sprint:

- **trainer 多卡吞吐** → [`planned/SPRINT_symmetric_colocated_ddp.md`](../planned/SPRINT_symmetric_colocated_ddp.md)（DDP,Slice 1-4 已落）、[`parked/SPRINT_multi_gpu_training.md`](../parked/SPRINT_multi_gpu_training.md)（FSDP2 策略层）、[`done/SPRINT_multi_gpu_readiness.md`](../done/SPRINT_multi_gpu_readiness.md)（地基已落）。
- **物理分离放置面（D1）** → [`planned/SPRINT_placement_surface_disaggregated_default.md`](../planned/SPRINT_placement_surface_disaggregated_default.md)（disaggregated 默认,P0-P2 已落）、[`done/SPRINT_colocation_config_simplification.md`](../done/SPRINT_colocation_config_simplification.md)。
- **async overlap（D2）裁决 + 安全权重同步** → [`parked/SPRINT_async_rollout_train_overlap.md`](../parked/SPRINT_async_rollout_train_overlap.md)（DiffusionNFT 约束 + Option A/B/C）、[`planned/SPRINT_shadow_model_weight_sync.md`](../planned/SPRINT_shadow_model_weight_sync.md)、[`planned/SPRINT_slime_overlap_strategy.md`](../planned/SPRINT_slime_overlap_strategy.md)。
- **cosmos-rl 全架构精读** → [`reading/cosmos-rl.md`](./cosmos-rl.md)。
