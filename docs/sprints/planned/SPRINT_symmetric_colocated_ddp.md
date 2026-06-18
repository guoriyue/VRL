# SPRINT: Symmetric colocated DDP（cosmos-rl mode=colocated 风格）

状态：**Slice 1–4 的代码已实现，CPU 可验证部分全绿（2026-06-18，trainers+config+ray+online-lifecycle 396
passed）。多节点执行（nccl all-reduce 跨服务器 / GPU 共卡显存 / DDP backward 跨 2 机）我无法在本机验证——
用户的 2×1 跑就是首次真实验证。** 旧的 hybrid head/remote 方案已整套移除（用户决定走 DDP）。

## 首跑 runbook（2 服务器 × 1 GPU）

配方：`configs/experiment/diffusion/cosmos_predict2_5/online_nft_kling_video_reward_ddp_2x1.yaml`
（继承基础单卡 colocated 计划不变，只把 trainer 翻成 DDP across 2 ranks）。两台都跑（同 `--master_addr`=服务器 A IP）：
```bash
# server A (rank 0, 输出节点)：
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 torchrun --nnodes=2 --node_rank=0 --nproc_per_node=1 \
  --master_addr=<A_IP> --master_port=29500 -m vrl.scripts.train \
  --config experiment/diffusion/cosmos_predict2_5/online_nft_kling_video_reward_ddp_2x1
# server B (rank 1)：同命令 --node_rank=1
```
**不要 `ray start` 共享集群**——每 rank 自己 `ray.init()` 本地、独占本机 GPU。

**首跑 watch-list（本机测不到、最可能首跑出问题的点）**：
1. **DDP reducer × LoRA/多次 forward**：DiffusionNFT 一步内对 transformer 做**多次 forward**（positive/negative
   x0）再一次 backward。DDP reducer 默认期望"每个 requires_grad 参数每步恰好拿一次梯度"。若报
   `Expected to mark a variable ready only once` / `parameters that didn't receive grad`，把配方里
   `distributed.training.ddp.find_unused_parameters` 设 true，或考虑 DDP `static_graph`。**这是首跑最可能的坑。**
2. **NCCL 跨机连通**：两台要能在 `--master_port` + NCCL 端口互通；多网卡设 `NCCL_SOCKET_IFNAME` /
   `GLOO_SOCKET_IFNAME`。
3. **lockstep**：rank0-only IO 靠每步 `trainer.step` 的 grad all-reduce 自同步；若 eval 很长、非 rank0 久等是正常
   （会阻塞在下一步 all-reduce），不是死锁。
4. **weight sync**：每 rank 把（all-reduce 后相同的）权重推**本地** rollout actor，复用现有 Ray 拷贝路径。

## 目标

把训练拓扑做成**对称共卡**（对标 cosmos-rl `mode=colocated`）：N 张卡，**每张卡 = 1 个 DDP 训练 rank +
1 个共卡 rollout worker（时分）**，两者读同一 policy version、并行 collect，然后一次同步的 DDP optimizer step。
**没有"head 卡"特殊化**——每张卡角色相同。这取代了早先那个不对称的
head/remote 方案（那个为绕开多-rank 训练而生、单卡 single_process 即可跑；对称 DDP 反过来需要多-rank 训练，
换来干净对称的架构）。

**为什么 DDP 不是 FSDP**：2B transformer + LoRA 单卡放得下，不需要 FSDP 的分片；FSDP 的整模型 ZeRO-3 每次
forward 会 all-gather 冻结的 base，对 LoRA 纯浪费。DDP 在每卡复制全量、只 all-reduce 可训练（LoRA）梯度，更快更简单。
FSDP 仅在模型单卡塞不下时才需要。

## cosmos-rl 参考（已读源码）

- 对称、**无中央 controller 进程**：每个 GPU rank 跑同一 `policy_entry`，`mode=colocated` 时每 rank 建一个
  `ColocatedRLControlWorker`，**同时持有 policy + rollout，且二者共用同一进程、同一 `nn.Module`** →
  weight sync 是**指针交换**（零拷贝），`colocated/rl_worker.py:132-211` 主循环 generate→report→train→sync。
- **VRL 的硬约束**：VRL 的 rollout 跑在 **Ray actor（独立进程，`ReleasableRayGenerationRuntime` kill+relaunch）**，
  **跨不了进程做指针交换** → 要么 (a) 复用现有 Ray 拷贝式 weight sync，要么 (b) 造同进程 colocated runtime。见 §4。

## 实现切片

### Slice 1 — DDPStrategy（已完成 + CPU 单测）

- `vrl/trainers/strategy.py`：新增 `DDPStrategy`——`prepare_model` 只 DDP-wrap 可训练 `transformer`（冻结 base
  `requires_grad=False` 自动排除出 reducer bucket），`backward=loss.backward()`（DDP hook 自动 all-reduce =
  同步步），`clip_grad_norm` 本地裁剪（梯度已 all-reduce、各 rank 相同）。export/load 复用 FSDP 同款 full-state
  helper（DDP 保全量参数，`unwrap_compile_and_ddp` 剥 `.module` 后即普通 module，`gather_full_state_dict` 退化为
  普通 state）。抽出共享守卫 `_single_transformer_handle`（FSDP/DDP 共用）。
- 接线：`schema.py` strategy Literal += `ddp` + 新 `DDPConfig(find_unused_parameters)`；`distributed.py`
  `resolve_training_context` 让 ddp 走 fsdp 同款 torchrun 分支；`strategy.py` `build_strategy` ddp 分支；
  `resources.py` `trainer_default_auto` / `_validate_trainer_device_count` 让 ddp 与 fsdp 同等（多卡可解析）。
- 测试：`tests/trainers/test_ddp.py`（CPU gloo ws=1，7 passed）——build_strategy、prepare_model wrap、
  export 与单进程 key-space 一致（无 `.module.` 泄漏）、frozen 过滤、export→load round-trip、守卫 reject。
- **online 仍 gated**：`online.py` `_require_supported_online_strategy` 对非 single_process 仍 raise
  NotImplementedError（多-rank online 编排 = Slice 2）。这与 FSDP2 当年"策略层先 CPU 单测落地、online gated"完全同构。

### Slice 2 — rank-aware online loop（Blocker 1，最大块，attended/多卡）

`run_online_recipe` 当前是单进程单 asyncio loop，假定 rank0/world1。改造点（`vrl/scripts/common/online.py`）：
- 设备：用 `training_context.device`（`cuda:local_rank`）取代单一 `trainer_torch_device`。
- **Ray 协调（最高风险）**：rank0 建 placement group + 驱动；非主 rank `ray.init(address='auto')` 挂靠。
  当前无"rank0-建-PG、其余挂靠"协调，必须新增（调查点名 `placement.py` `pg.ready()` 在所有 rank 各自建 PG 时会
  死锁到 600s 超时）。
- 仅 rank0：`save_checkpoint` / `write_metric_row` / `_fixed_eval_and_log` / profiler 路径（否则 N 个 writer
  撞同一 metrics.csv / checkpoint dir）。
- advantage 归一化：GRPO global-std/group 统计当前是 per-rank-collected；需 all-reduce 或显式接受 per-rank。
- RNG/prompt 采样：各 rank 同 seed → 同 batch（对"同版本并行 collect"是对的；若要数据并行分片需显式切分）。

### Slice 3 — 对称共卡放置（resolver/placement，attended）

`resources.py` 加一个 fsdp/ddp-colocated 分支：strategy=ddp + `rollout.colocate` 时，rollout 解析为**正好 trainer
设备集**（每 rank 一个 rollout worker 共卡），`gpu_memory_fraction` 强制；把 `_validate_fsdp_trainer_disjoint`
换成 colocate-aware 检查（允许 trainer==rollout，前提是有显式 memory_fraction）。每 rank 一个本地 rollout worker
pin 到 `LOCAL_RANK` 的卡，复用现有 resident + memory_fraction 机制（**不是** offload/release 路径）。

### Slice 4 — weight-sync 岔路（决策）

- **(a) 复用现有 Ray 拷贝**（增量，推荐先做）：每 rank all-reduce 后权重相同，各 rank 把权重推给**本地**共卡
  rollout worker，走现有 `weight_sync.py` 的 CPU state_dict → ray.put → load。比指针交换贵，但复用全部现有 Ray
  rollout 基建。`weight_sync.py` 当前假定单一 pusher，需让每 rank 推自己本地 worker（或仅 rank0 推、广播）。
- **(b) 同进程 colocated runtime**（大改，才有零拷贝）：在 torchrun rank 进程内直接持有 rollout runtime + 同一
  `nn.Module`，weight sync = 指针交换（cosmos-rl 路线）。需新建一个同进程 generation runtime（平行于
  `vrl/generation/ray/runtime.py`）。**首版走 (a)**，(b) 作为后续吞吐优化。

## 取代的 hybrid head/remote 方案（已移除）

曾有一个不对称的 `hybrid head/remote`（head 卡共卡 1 个 rollout + 远程其余，single_process 即可跑，为绕开多-rank
训练而生）。对称 DDP 是更干净的终态，用户决定走 DDP，故该 hybrid 实现已整套移除（resolver/preflight/placement
的 hybrid 分支、`head_colocated_rollout_count`/`remote_rollout_count` 字段、hybrid 测试、hybrid 配方与其 sprint
文档），代码回到干净原版；DDP 不依赖它们。

## 非目标

- 不做 in-process pointer-swap weight sync（Slice 4-b，后续吞吐优化）。
- 不在单卡上追求真 overlap（对称共卡是时分；真并行 collect 的吞吐来自多卡）。
- 不改 FSDP2 行为（DDP 与 FSDP 共用 export/load helper + `_single_transformer_handle` 守卫，FSDP 路径行为不变，
  `tests/trainers/test_fsdp.py` 仍全绿）。
- 不在本 sprint 解 Slice 2 的多-rank online 编排（attended + 真实多卡 + 单独 PR）。

## 关键文件 / 引用

- Slice 1（已实现）：`vrl/trainers/strategy.py`（`DDPStrategy` + `_single_transformer_handle` + build_strategy）、
  `vrl/trainers/distributed.py`（resolve_training_context ddp 分支）、`vrl/config/schema.py`（strategy Literal +
  `DDPConfig`）、`vrl/ray/resources.py`（trainer_default_auto / `_validate_trainer_device_count` ddp）、
  `vrl/scripts/common/online.py:64-84`（online gate，仍 raise）、`tests/trainers/test_ddp.py`（7 tests）。
- Slice 2：`vrl/scripts/common/online.py`（run_online_recipe）、`vrl/ray/placement.py`（GlobalRayPlacementOwner /
  pg.ready）、`vrl/generation/ray/launcher.py`（ray.init / preflight）。
- Slice 4：`vrl/trainers/weight_sync.py`（现有 Ray 拷贝路径）、`vrl/generation/ray/runtime.py`（resident vs release）。
- cosmos-rl 参考：`cosmos_rl/rollout/.../colocated/rl_worker.py:132-211`（同进程 generate/train/sync 主循环）、
  `policy_entry.py:43-61`（每 rank ColocatedRLControlWorker）。
- 复用先例：`SPRINT_multi_gpu_training.md`（FSDP2 策略层 CPU 单测先落、online gated 的同构路径）、
  `docs/sprints/parked/SPRINT_async_rollout_train_overlap.md`（≥2 卡门槛 + DiffusionNFT 约束）。
