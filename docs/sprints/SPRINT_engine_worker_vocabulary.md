# SPRINT：engine / worker 抽象 + 多卡引擎全量支持

状态：**P1/P2 已实施，P3 待做，P4/P5 待多卡后端接线（2026-08-14）**。
实施记录：P1 = commit `5fbb1470`（num_engines 键 + RolloutRuntimeSection）；
P2 = 本 commit（GenerationRankActor 协议、RayGenerationEngine 组合体、driver
全链改单位、BundleLayout 引擎分组、N=2/3 假件单测、reward 双子改名）。

原始计划：前置事实：`gpus_per_worker` 配置旋钮与
`rollout_gpus_per_worker` 派生视图已删除（commit `8c4ffd23`、`b50892bb`）——
删除的是"只有一种合法取值的旋钮"，不是概念的位置。

决策（owner 拍板，三轮迭代后定稿）：多卡 / 多节点生成是近期现实（当前单卡
只是省钱），本 sprint **全量支持** `gpus_per_engine > 1`：抽象、配置键、进程组
运行时、Ulysses 序列并行执行一起做。"有害旋钮"的顾虑由 **capability gate**
化解：不支持多卡引擎的 family 对 >1 **硬报错**，绝不静默退化为重复计算。

硬件事实（影响验收方式，不影响交付范围）：当前开发机单卡（RTX 5090 32GB）。
NCCL 多 rank 需要每 rank 一张卡，因此**整模型多卡验收挂多卡机器**；但
Ulysses 的数值正确性用 **CPU + gloo** 在 N=2 下与单 rank 参考 attention
逐元素比对——本机就能钉死，不是妥协而是更强的单测。

## 0. 现状诊断（为什么"两边都不是"）

今天已有两层，但切的是 transport 维度，不是 engine/rank 维度：

- `RayGenerationWorker`（204 行）只是 Ray RPC 外壳（并发组、pipelined 进度锁），
  一切委托 `self.core`；
- `GenerationWorkerCore`（773 行）把两种职责熔在一起：
  - **engine 级**（副本的）：`update_weights` 权重版本状态、`load_policy`/
    `release_policy` 生命周期、`sleep`/`wake` 入口、请求编排、
    `probe_batch_size`、capability 上报；
  - **rank 级**（一张卡的）：`_build_executor`、forward chunk 执行、显存读数、
    CPU pinning、CuMem 原语。

engine 合同不存在（driver 靠鸭子类型调 actor 方法），rank 边界也不存在。

## 1. 目标结构（vLLM 形状：engine 在 driver 侧组合 rank actor）

```
driver 组件(dispatcher / weight_sync / schedule)
   │        只面对 GenerationEngine 协议
   ▼
RayGenerationEngine            driver 侧对象（非 actor），实现协议，
   │                           持 ranks: list[RayActorHandle]
   ▼ 扇出 / 聚合
RayGenerationWorker            rank actor（词归位：vLLM 语义的 per-GPU worker）
   └─ GenerationWorkerCore     rank 程序：单卡执行核（P5 起可加入引擎进程组）
```

**两个基数，不要混**：

- **engine 数**（`num_engines`）= 副本数 = 数据并行度：= rollout 卡数 ÷
  `gpus_per_engine`（robotics `[1,2]` + 每引擎 1 卡 → 2 engine；将来 6 卡 +
  每引擎 2 卡 → 3 engine × 2 rank）。
- **每 engine 的 rank 数**（`gpus_per_engine` / `len(ranks)`）：被删除的
  `gpus_per_worker` 的正名回归——这次带着真消费者（P5 的并行执行）回来。

各方法的扇出/聚合语义（N=1 全部退化为"调唯一 rank"）：

| 协议方法 | 语义 |
|---|---|
| `execute_batch` / `execute_request_pipelined` / `probe_batch_size` | 全 rank 协同调用（各 rank 算同一批的自己那份序列），结果取 rank 0 |
| `update_weights` | 广播全 rank，校验回声版本一致 |
| `load_policy` / `release_policy` / `sleep` / `wake` | 全 rank，快照聚合 |
| `health` | rank 粒度机器（探针/杀进程/重启），engine 层派生裁决：任一 rank 死 = engine 死 |
| `metadata` / `supports_versioned_trainable_state` | 聚合，且全 rank 必须一致 |

词汇结论：**engine** = 副本（DP 单元）；**worker** = rank（`RayGenerationWorker`
名字不动、语义归位为每卡 actor）；**actor** = 通用基建成员词。白名单不动：
`dataloader_num_workers`、SANA 擦除表历史拼写、docs 历史文档。

## P1 配置面

| 键 | 处置 | 理由 |
|---|---|---|
| `distributed.resources.rollout.num_workers` | → `num_engines` | 数的是副本；`0`=无舰队；`num_gpus: 0` 时缺省 1 个 CPU 引擎 |
| `distributed.rollout.gpus_per_engine` | **新增**（P4 落地） | slime `--rollout-num-gpus-per-engine` 对齐；缺省 1 |
| `distributed.rollout.cpus_per_worker` | 保留 | CPU 发给 rank actor，新词汇下语义本来就对 |
| `distributed.rollout.worker_rpc_timeout_s` | 保留 | RPC 打到 rank actor 上，同理 |

- resolved 层：`rollout_num_engines` + `resolve_num_engines`；对账行
  `rollout_workers=` → `engines=`（P4 后追加 `gpus_per_engine=`）。
- `RolloutWorkerSection` → `RolloutRuntimeSection`（混装 engine/rank 级旋钮，
  字段注释逐条标归属层）。注意 `sana_aesthetic_report.py` default-equivalent
  表直接 import 该类——同步 import（代码级；协议指纹不变，canonical 链不含
  `num_workers`）。
- Presets：仅 `recipe/offline/diffusion_dpo.yaml`（`num_workers: 0` →
  `num_engines: 0`）；擦除器加 `num_workers`→`num_engines` 翻译。

## P2 抽出两层（结构核心）

1. **`GenerationEngine` 协议**（风格对齐 `GenerationRuntime` / `RewardScorer` /
   `RolloutCollectorControl`）：上表 12 个方法。pool 成员变 driver 侧对象后，
   派发是普通方法调用（engine 内部才翻译成远程调用）——remote_method
   字符串间接层在 pool 层整个消失。
2. **`RayGenerationEngine`**（driver 侧对象）：实现协议，持
   `ranks: list[RayActorHandle]`，按上表扇出/聚合。异步面返回**聚合 future**：
   全 rank 完成、结果取 rank 0、任一 rank 失败立即整体失败——不能只等
   rank 0 的 ref（非 0 rank 崩溃会让集合通信挂死 rank 0）。
3. **rank 层归位**：`RayGenerationWorker` 名字不动；`GenerationWorkerCore`
   即 rank 程序；类内标注 engine 态 / rank 态成员分界。
4. **driver 全链改单位，生命周期留 rank 层**：dispatch / weight_sync /
   session / runtime 面对 engine；`batch_placement.py` `worker_id` →
   `engine_id`；health/liveness 机器保持 rank 粒度，engine 层出裁决。
5. **`BundleLayout` 引擎分组**：bundle 按 engine 分组，probe 校验按组，
   组内 bundle 同节点 PACK 亲和。分组逻辑对组大小 N 泛型。
6. **通用层分家**：`actor_group`（launch/kill/liveness）成员 = rank actor；
   `actor_pool`/dispatcher（LPT 派发、inflight 记账）成员 = engine。
7. reward 侧对齐双子：`RewardWorkerLaunchContract` → `RewardRuntimeLaunchContract`；
   `from_worker_config`/`worker_config` 改 component/runtime 措辞。
8. **对 N 泛型 + N>1 假件单测**（无 GPU）：录音假 rank 句柄在 N=2/3 下驱动
   RayGenerationEngine，钉死广播顺序、版本回声校验、fail-closed、快照聚合、
   按组 bundle 校验。

## P3 rank 进程组运行时

- launcher 为每个引擎组分配 rendezvous（MASTER_ADDR/MASTER_PORT、rank/world
  env），rank actor 启动时若组大小 > 1 则初始化 torch.distributed
  （GPU 上 nccl；测试路径 gloo + CPU）。
- 组的生死一体：任一 rank 启动失败 → 整组销毁重建（有界重启沿用现有
  supervisor 策略，单位从 actor 升为组）。
- **gloo CPU 冒烟测试**：两进程组网、all-to-all 张量往返、优雅退组——本机
  无多卡也能全部验证。

## P4 `gpus_per_engine` 键 + capability gate

- 键落 `distributed.rollout.gpus_per_engine`（缺省 1）；resolution 派生
  `num_engines = rollout 卡数 ÷ gpus_per_engine`（整除校验；`num_engines`
  显式给出时必须相等）。cross_node 暂拒 >1（跨节点引擎组见非目标）。
- **capability gate（发键而不发地雷的关键）**：family
  `runtime_capabilities.supports_multi_gpu_engine`，缺省 False；>1 且家族
  不支持 → 配置解析期硬报错（指名家族与键），**绝不**静默跑 N 遍重复计算。
- `BundleLayout` 组大小接通真值；formatter 对账行追加 `gpus_per_engine=`。

## P5 Ulysses 序列并行执行（第一个真消费者）

- **注入缝 = diffusers attention processor**：所有 family 的 transformer 都用
  diffusers attention 模块（processor 可换）。Ulysses processor：attention 前
  按 head 维 all-to-all 换序列分片、算完再换回——模型无关，一次实现全家族
  复用；每家族只需接线（processor 安装 + 序列切分/还原 + rope/位置编码偏移
  核对）。候选依赖：优先评估 para-attn / xfuser 直接复用，评估不过再手写
  （all-to-all 数百行量级）。
- 非 attention 部分：VAE / text encoder 不切——rank 0 编码，广播 latent /
  embedding；去噪循环各 rank 持全量权重（USP 权重复制，**现有权重同步模型
  原样成立**：P2 的广播即成品）；采样噪声按 seed 广播保证全 rank 一致。
- **数值验证（本机、CPU+gloo、无卡也跑）**：N=2 Ulysses attention 输出与
  单 rank 参考 attention 逐元素比对（容差 0）；整层 transformer block 前向
  N=1 vs N=2 一致性。
- 首个接线家族：**sd3_5**（图像、最便宜、MMDiT 是 xDiT 官方支持的结构，
  也是仓库的默认试验田），翻真其 capability；wan_2_1 随后按 audit-on-touch。
- `probe_batch_size` 语义在 N>1 下重审：探测的是"引擎"吞吐（序列并行摊薄
  激活显存），全 rank 协同探测。

## P6 多卡验收（硬件门，非功能门）

本机单卡（RTX 5090 ×1），NCCL 多 rank 物理不可行；多卡机器可用时执行：

- 2 卡 sd3_5：同 seed 下 N=2 输出与 N=1 数值等价（容差内逐像素）；
- 吞吐 / 峰值显存对比报告（序列并行的收益本证）；
- 在线小跑：2 engine × 1 rank 与 1 engine × 2 rank 各一轮，收敛曲线无异常。

功能在 P1–P5 已交付并单测完毕；P6 是物理回归，不阻塞合并。

## 非目标

- 跨节点单引擎（引擎组跨节点）：等多节点拓扑设计，P4 先硬拒。
- Megatron 式真·权重切分与 TP 感知权重 scatter（slime
  `update_weight_from_tensor` 路线）：USP 权重复制已覆盖近期模型；触发器 =
  单卡装不下权重的模型进场。
- `dataloader_num_workers`、`distributed.rollout` section 路径不动。
- resources 角色级事实（设备集、交集共享、phase lease、sharing facts）不变；
  引擎分组只发生在 rollout 卡集内部。
