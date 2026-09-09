# Ray 还是 Rayless：VRL 的实测与决定

日期：2026-09-08。触发：外部评论"Rayless 的一大好处是显著更便于检视训练过程……
古早的 RL infra 圈子把 Ray 使用的过于厚重"。本文是把那句话对到本仓库上逐条核实
的结果，以及由此得出的决定。

结论先行：**这不是一个全局二选一的问题，本仓库不该做全局选择。**Rayless 对 75 个
experiment 里的 66 个更好，Ray 对其中 9 个是必需的。正确的形态是一个 Protocol 后面
挂两个 runtime，而这个 Protocol 已经存在。

---

## 0. 最容易被跳过的那一步逻辑

争论常常停在"Ray 重不重"，但真正的分界在更前面一步：

**`torchrun` 给你的是若干进程和一个集合通信组，它不给你"进程 A 在任意时刻请求进程 B
做一件事"。**

RL 的 rollout 和 trainer 恰恰是这种关系：各自有独立的节奏（trainer 步进时 rollout 可能
在生成，也可能在睡觉）、独立的放置（可能不在同一张卡、不在同一台机器）。这是
request/response，不是 SPMD 的集合通信。要在 torchrun 上得到它，你只有两条路：

1. 自己写 RPC —— 那正是 Ray 已经提供的东西；
2. 自己写队列，让控制流变成"数据可达性" —— 那正是 Meshy 做的事。

没有第三条"免费得到"的路。所以真正的问题不是"用不用 Ray"，而是：

> **trainer 和 rollout 之间那层协调，由谁来写和维护？**

- 用 Ray：别人写好了，你付可观测性的税。
- 自己造队列：你拥有一个分布式数据面（要处理 GPU 张量、背压、失败）。
- 都不要：你放弃独立节奏和独立放置，回到"一个同步程序"。

第三条对某些项目是完全正确的选择（见 §1 的 RL2）。对本仓库不是，因为那些能力在用。

## 1. "Rayless" 是三种不同的东西

| | 主张 | 代表 | 协调层是什么 |
| --- | --- | --- | --- |
| A | Ray 对小规模是过度工程，用 torchrun，代码越少越好 | [RL2](https://github.com/ChenmienTan/RL2)（Ray-Less RL for LLMs，<1K 行 PPO/REINFORCE） | 没有——同步、共置、一个程序 |
| B | 控制流应当由数据可达性驱动，不需要编排者 | [OpenBMB/Meshy](https://github.com/OpenBMB/Meshy) + [TransferQueue](https://github.com/Ascend/TransferQueue) | 一个列式队列数据面 |
| C | rollout 是一个推理服务，driver 是客户端 | vLLM / SGLang server 模式 | HTTP |

三者的论证完全不同。A 说的是"税太高"，B 说的是"这件事根本不需要"，C 说的是"换个
边界"。把它们混成一个"Rayless 阵营"是这场讨论最大的混淆源。

**A 为什么对本仓库不成立**：`torchrun` 是 SPMD，每个 rank 跑同一个程序。"rollout 睡下去
把物理显存让给 trainer、两者在同一张卡上轮流"这件事表达不了。而这正是你能在一张
共用的 5090 上跑训练的原因。

**B 为什么现在不划算**：见 §4。

**C 是真正的长期出口**：见 §6。

## 2. 本仓库的 Ray 用量（实测，非印象）

| | verl / OpenRLHF 那类 | VRL |
| --- | --- | --- |
| trainer | Ray worker group，被 single controller 用 RPC 驱动 | **torchrun**（`RANK`/`LOCAL_RANK`/`WORLD_SIZE`；FSDP/DDP 都是 torchrun 策略） |
| reward | 常见是独立 Ray actor | driver 进程内（`build_reward_runtime`） |
| critic / value | 有，需要编排 | **没有**（GRPO 系） |
| Ray actor 类 | 每个角色一组 | **一个**（generation worker）+ 一个 placement 探针 |

实际调用面十来个 API：`ray.init/shutdown/is_initialized`、`ray.remote`、`ray.get/put`、
`ray.get_gpu_ids/nodes`、`ray.method`、`ray.kill/cancel`。Ray 在这里只是
**actor 运行时 + 放置 + 传输**。

关键事实：**逻辑全在 Ray-free 的代码里**。`vrl/generation/execution/worker.py::GenerationWorkerCore`
不 import ray，`vrl/generation/ray/worker.py` 只是薄壳（`def sleep: return self.core.sleep()`）。
`vrl/ray/resources.py` 那 1000 行是纯 config→拓扑计算，同样不依赖 ray。

**所以"古早圈子用得过于厚重"那部分债，本仓库没有。**

## 3. Ray 到底买到了什么，以及每条的替代品

早先的一版说这四条"没有替代品"，那是过头了。逐条纠正：

| Ray 提供 | 替代品 | 用它的 preset |
| --- | --- | --- |
| 跨节点 rollout | `torchrun --nnodes`；或推理服务 + HTTP（§1 的 C） | 2 |
| 异步重叠（continuous） | **独立进程**即可，不必是 Ray：`torch.multiprocessing` + Queue，或子进程说 HTTP | 6 |
| 解耦放置 / 多引擎 | 自写 launcher + 每进程 `CUDA_VISIBLE_DEVICES`（拓扑你已经算好了） | 1 |
| 进程隔离（独立 CUDA context / allocator，崩溃不连坐） | 任何子进程 | 全部 |

**不属于 Ray 的功劳**（早先记错过）：sleep/wake 显存 parking 是 CuMem，实现在 Ray-free 的
core 里，进程内同样能做；资源解析同理。

**自制等价物真正要写的东西**：进程启动器、健康检查与存活判定、崩溃重启、带零拷贝的
张量传输、原子的成组资源预留（gang scheduling）、节点发现、失败向上传播。Ray 是这些的
现成且已调试实现。

**只有一件事自制 launcher 伪造不了**：在你**不拥有**的机器上协商资源。单机你知道有几张
卡；共享集群上你不知道哪个节点空着。`ray_rollout_cross_node.yaml` 正是这种形状
（`visible_devices: auto` + `cross_node: true`，谁在哪台机器运行时才知道）。

## 4. 数字：谁在为谁付钱

75 个 experiment 中：

- continuous（异步）：6
- cross_node：2
- dedicated pool / 多引擎：1
- 走 FSDP 多 rank：10（但 FSDP 是 torchrun，每 rank 只是在自己卡上挂一个 rank-local rollout）

去重后**约 9 个真的需要 §3 前三行**。**其余约 66 个是单卡共置同步**——在那些跑法里 Ray
只给第 4 行（进程隔离），而你付的是可观测性的税。

税的形态是具体的：`docs/sprints/done/` 里 12 个 Ray 形状的 sprint
（`rollout_worker_liveness`、`ray_rollout_operation_deadlines`、`global_ray_placement_owner`、
`ray_cluster_ownership_and_shared_host_isolation`、`ray_oom_degradation`、
`ray_phase_lifecycle_plan`、`ray_lifecycle_colocation_dedup`、`one-real-ray-cluster` 等），
以及 `worker_rpc_timeout_s` / `generation_stall_timeout_s` / `health_check_*` 这几个旋钮——
它们本质上是"远程调用没有活栈可看，只能靠超时判断死活"的补偿。

### 可观测性差异的真实机制

不在功能，在**你的代码跑在哪个进程里**。

- `torchrun`：rank0 的 stdout 就是终端；异常是普通 traceback；`pdb.set_trace()` 直接进；
  `py-spy dump --pid` 直接看；Ctrl-C 停在真实栈；`CUDA_LAUNCH_BLOCKING=1` 崩在真实行号。
- Ray：代码在远程 actor 里。异常序列化后在 driver 重抛成 `RayTaskError`，你拿到的是远端
  traceback 的**文本**，活的栈帧和局部变量没了，做不了 post-mortem。断点要走 `ray debug`/rpdb。
  stdout 被捕获转发、带前缀、有时缓冲。hang 最难受：driver 卡在 `ray.get`，得靠 `ray stack`
  或 dashboard 去猜哪个 actor 卡在哪。
- 额外一层：故障模式从"我的代码错了"变成"分布式系统出事了"（actor 死亡、placement group
  满足不了、object spilling、worker 被 OOM killer 干掉后表现为不相干的 `ActorDiedError`、
  driver 与 worker 代码版本不一致）。

## 5. Meshy 值不值得学

它的三条原则（README 原文）：每个角色是独立服务、`TransferQueue` 是唯一通道
（"column readiness is the only control signal, so services never handshake directly"）、
控制流由数据可达性驱动。启动方式是 launcher 起队列后 `torchrun` 每 GPU 一进程，拓扑由
声明式 recipe 给出，每台机器各自算出相同放置，因此不需要服务发现。

**它为什么能不要 Ray**：不是嫌重，是**Ray 提供的那件事（controller 远程调用对象）不在这个
设计里了**。没有人调用任何人，服务只是"读输入列 → 算 → 写输出列"。这是 choreography
取代 orchestration。

**最实的收益**：同步 / 异步流式 / OPD（one-step off-policy，rollout 用落后一次更新的权重继续
生成）是**同一份代码换一个 `pacing_window` 数字**（1 = 同步，2 = OPD，≥2 = 更深异步）。
而 verl 那边同步、fully-async、one-step-off 是三个不同 recipe。

**为什么本仓库现在不做**：

- 你异步只有 6/75。为这个收益去拥有一个分布式数据面（GPU 张量、背压、失败）投产比是反的。
- 它假设**拓扑是声明好的、机器是你的**。你在一台和别人共用的 5090 上跑，这个前提不成立；
  §3 里"自制 launcher 伪造不了"的那件事它没解决，是绕开了。
- 没有 controller，谁发现服务死了？队列的 ready 永远不来，你就挂着。这和"driver 卡在
  `ray.get`"是同一类问题换了个位置。
- `TransferQueue` 不是反 Ray 的表态：它出自 Ascend，是独立数据面，已进 verl（2026-04，
  128×H100 多模态端到端 +49.1%）、ROLL、腾讯混元 UniRL，slime 在讨论集成——
  **那些框架是 Ray + TransferQueue 一起用的**。

### "既然要队列，为什么不直接用 Ray？"

对本仓库而言这个反问就是正确答案。Meshy 不这么做，是因为目标冲突而非能力不足：
队列若是 Ray actor，每个服务都得是 Ray 客户端，整个运行时又被拖回每个进程，
"No Ray" 就不成立了；而且 `ray.util.queue.Queue` 是**单 actor 包一个 deque**，是串行化点，
不是分片数据面；他们要的也不是 FIFO，是**按格就绪的列式存储**；另外 TransferQueue 要跑
在 NPU 上，而 Ray 的 GPU 直传只支持 CPU 和 NVIDIA。

**Ray 现在的数据面能力**（容易被低估）：Ray 2.55+ 的
[Ray Direct Transport](https://docs.ray.io/en/latest/ray-core/direct-transport/direct-transport.html)
支持 actor 之间直接传 `torch.Tensor`，数据留在显存，底层走 NCCL / Gloo / NIXL RDMA，
绕开 CPU 对象存储；Compiled Graphs 也支持 NCCL 传 CUDA tensor。也就是说 TransferQueue
存在的核心理由（GPU 张量不绕主存）Ray 原生已有。

## 6. 决定

**保持 Ray，加一条进程内 rollout 路径，不造队列。**

**现在做的一件事**：把 `tests/e2e/test_real_checkpoint_rl.py` 里那 55 行的
`_DirectExecutorGenerationRuntime` 提升为正式实现，配一个 `distributed/inprocess_rollout`
preset。`GenerationRuntime` 已经是 Protocol（`vrl/generation/protocols.py`），
`GenerationWorkerCore` 本来就 Ray-free，seam 全在，只是没接出来。单卡冒烟、e2e、交互式
调试走它；多卡 / 异步 / 跨节点继续走 Ray。估计 100–150 行生产代码加一个 preset 加测试。

**明确不做**：不重写 `vrl/ray/resources.py`（本来就不依赖 ray，换什么架构都要它）；
不加深 Ray 层；不自建队列数据面。

**Compiled Graphs / RDT 现状**：版本有（`.venv` ray 2.55.1，pin 是宽松的 `ray>=2.9.0`），
**仓库一处没用**。唯一可能受益的是权重同步——今天是
`flatten_trainable_module_state()` → `ray.put`（`vrl/generation/ray/weight_sync.py:84`）
→ `actor.update_weights.remote(...)`，即 GPU→主存→plasma→主存→GPU。但三个理由让它
现在不划算：LoRA 跑法只同步 adapter（`vrl/run.py:174`
`base_weight_sync = sync_trainable_state and not use_lora`），负载本就很小；
**RDT 目前不兼容 asyncio**，而 `push_to_rollout_engines` 是 `async def`；单卡共置省下的
只是同卡的主存绕行。记在账上，条件到了再看。

**什么条件会改变这个决定**：

1. 异步 rollout 变成主线（那 6 个 preset 变成大多数）→ 重新评估队列数据面。
2. 需要在**共享集群**上按需申请卡 → Ray 的 gang scheduling 变成不可替代，更该留着。
3. **全参数微调 + 多卡分离放置**成为常态 → 权重同步变成多 GB 跨卡传输，RDT 值得，
   前提是它支持 asyncio 或那条路径改成同步调用。
4. 想彻底把 Ray 移出 rollout 路径 → 出口不是 Meshy 的队列，是
   `docs/sprints/parked/SPRINT_sglang_diffusion_execution_provider.md` 里已经画好的
   **推理服务 + HTTP 客户端**（业界正在往那边走）。

---

## 参考

- [RL2: Ray-Less RL for LLMs](https://github.com/ChenmienTan/RL2) ·
  [作者宣布贴](https://x.com/simon_ycl/status/1934661877239566554)
- [OpenBMB/Meshy](https://github.com/OpenBMB/Meshy) · [Ascend/TransferQueue](https://github.com/Ascend/TransferQueue)
- [verl: One Step Off Policy Async Trainer](https://verl.readthedocs.io/en/latest/advance/one_step_off.html) ·
  [Fully Async Policy Trainer](https://verl.readthedocs.io/en/latest/advance/fully_async.html)
- [AsyncFlow: An Asynchronous Streaming RL Framework](https://arxiv.org/pdf/2507.01663)
- [Keep the Tokens Flowing: Lessons from 16 Open-Source RL Libraries](https://huggingface.co/blog/async-rl-training-landscape)
- [Single vs Multi-Controller in veRL](https://langcopilot.com/posts/2025-07-14-single-vs-multi-controller-verl-pathways-rl)
- [Ray Direct Transport](https://docs.ray.io/en/latest/ray-core/direct-transport/direct-transport.html) ·
  [Ray Compiled Graph](https://docs.ray.io/en/latest/ray-core/compiled-graph/ray-compiled-graph.html)
- 本仓库内证据：`vrl/generation/execution/worker.py`、`vrl/generation/ray/worker.py`、
  `vrl/ray/resources.py`、`vrl/generation/ray/weight_sync.py:84`、`vrl/run.py:174`、
  `vrl/config/presets/base/distributed/`、`docs/sprints/done/SPRINT_ray_*`、
  `tests/e2e/test_real_checkpoint_rl.py::_DirectExecutorGenerationRuntime`
