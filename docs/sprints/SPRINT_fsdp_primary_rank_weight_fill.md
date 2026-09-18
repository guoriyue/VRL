# SPRINT: FSDP 初始权重只由 rank 0 读取，其余 rank 用骨架接收广播

状态：已实施，CPU 双进程验证通过（2026-09-17）。分支 `review/combined-20260916`。
GPU 上的多卡验证尚未做，见 §6。

## 1. 问题

FSDP 训练时每个 torchrun 进程各自 `from_pretrained` 完整 checkpoint 到自己的
CPU 内存，再交给 `fully_shard` 逐块分片。分片发生在每个进程读完整模型**之后**，
所以有一个瞬间 N 份完整拷贝同时存在于主机内存：

| 模型 | 4 rank 峰值主机内存 | 本机（91 GB） |
| --- | --- | --- |
| SD3.5 medium（约 5 GB） | 20 GB | 可以 |
| MiniMax-H3（66 GB） | 264 GB | 不可能 |

这是 H3 至今只能走手工分卡（`placement.py` 的 `device_map`）而不能走完整分片
FSDP 的真正阻碍；GPU 侧 66 / 4 ≈ 17 GB 每卡本来放得下。

Miles Diffusion（`ebd55fc1`，`actor.py:113-149`、`checkpoint.py:31-65`）的做法：
rank 0 在 CPU 加载真权重，其余 rank 用 `init_empty_weights` 建 meta 骨架，
各自 `fully_shard` 之后由 `set_model_state_dict(broadcast_from_rank0=True)`
从 rank 0 逐张量广播填充，buffer 单独广播。主机内存峰值 1 份而不是 N 份。

## 2. 边界

- 谁需要真权重是**训练策略**的事实，不是模型的：`Strategy.materialize_weights`
  属性，非分片策略（单进程 / DDP / CP，每个 rank 都复制整个策略）恒为真，
  FSDP 只在 rank 0 为真。
- 训练入口把它作为关键字传给 `ResolvedModel.materialize(materialize_weights=...)`
  → `ModelFamilyEntry.build_replay(build, materialize_weights=...)` →
  各家族 replay 构建器 → `load_diffusers_transformer(..., materialize_weights=...)`。
  它不进 `ModelBuild`，不过 Ray wire 边界，和刚删掉的 `defer_trainable_device_move`
  是同一条原则：策略事实不投影成构建参数。
- 采样路径不受影响：`build_rollout` 没有这个关键字。

## 3. 改动

### 3.1 加载器（`vrl/models/loader.py`）

`load_diffusers_transformer(..., materialize_weights=False)` 走
`skeleton_from_config`：`load_config` + `init_empty_weights(include_buffers=False)`
下 `from_config`，再 `.to(dtype=build.parameter_dtype)`。参数在 meta，buffer
在 `__init__` 里正常算出。不打开任何权重文件（测试用 `from_pretrained` 抛异常
来钉住这一点）。

没有用 Miles 的"`from_pretrained` 套 `init_empty_weights`"做法：实测 diffusers
在那种模式下仍会把整个 state dict 读一遍再做空拷贝，并对每个参数发一条警告。

### 3.2 家族构建器

| 家族 | `materialize_weights=False` 时 |
| --- | --- |
| 所有走 `DenoiseFamilyBuild` 通用 recipe 的家族 | 骨架 |
| MiniMax-H3 / VDN-H3 共用的 `load_h3_replay_components` | 骨架（`block_devices` 分卡路径始终加载真权重，它的 device map 就是放置本身） |
| Anima | 骨架（同一个 `_cosmos_t2i_transformer_config()`） |
| VDN-H3 | 真权重：graft 要把 VDN checkpoint 的分支权重装进骨干 |
| Echo | 真权重：vendored LTX wrapper 整体加载 |
| Cosmos3 | 真权重：暂无骨架路径 |

保持真权重的家族在 FSDP 下行为与之前完全一致（每 rank 加载，无广播），
因为 §3.3 的填充是按"是否有 rank 持有骨架"集体判定的。

### 3.3 策略（`vrl/trainers/strategy.py::FSDPStrategy.prepare_model`）

顺序变为：结构校验 → 建进程组和 mesh → 集体判定是否需要填充
（`collectives.max_int(any(p.is_meta))`）→ 若需要，rank 0 用
`broadcast_object_list` 广播每个参数和 buffer 的 dtype，骨架按此调整
（diffusers 的 fp32 钉住只有加载真权重的 rank 知道）→ 原有 dtype 归一化 →
CP → `apply_fsdp`。分片之后对每个 handle 调 `_fill_from_primary`：

1. 仍有 meta 张量的 rank 调 `to_empty`（`cpu_offload` 时目标是 CPU，buffer
   再搬回计算设备，与 Miles 一致）。
2. `load_full_state_dict(wrapped, rank0 的分片前 state dict 或 {})`，就是
   checkpoint 恢复已在用的 `set_model_state_dict(broadcast_from_rank0=True)`。
3. buffer 逐个经 CPU 协调组广播：不依赖 NCCL/gloo 后端，也不依赖 buffer
   在 offload 下的具体位置。

adapter-only（`shard_trainable_only`）分支原来在分片前对整个 handle
`.to(device)`；骨架没有存储可搬，跳过，由第 1 步的 `to_empty` 放置。

dtype 归一化从进程组创建之前挪到之后。结构性的 `_trainable_module_handles`
校验仍在进程组之前，"坏模型无需进程组即可失败"的性质不变。

## 4. 验证

- `tests/models/test_loader.py::test_transformer_skeleton_reads_only_the_config`
- `tests/trainers/test_fsdp_primary_weight_fill.py`：两个 gloo rank，真 tiny
  `SD3Transformer2DModel` 存成 checkpoint，rank 1 用骨架；`prepare_model`
  后两个 rank 上每个参数和 buffer 与 rank 0 分片前的值逐一相等，无 meta 残留，
  可训练集合仍是 LoRA。完整分片和 adapter-only 两种布局都覆盖。
- `tests/trainers/test_fsdp.py` / `test_strategy.py` 各加一条属性断言。
- 其余测试 stub 因 loader 多了关键字而调整签名。

## 5. 代价

一次性的启动广播：每个非 rank 0 进程收到完整模型的字节数，只留自己那份分片。
H3 4 卡约 198 GB 互联流量，PCIe 上十几秒；对比每个 micro-step 前向和反向各
约 50 GB 的基座 all-gather，可以忽略。换来的是主机内存峰值从 N 份降到 1 份。

## 6. 未做

- GPU 多卡验证（本机只有一张 5090）。需要在 4 卡上跑一次随机权重的 H3
  完整分片 preflight，对照现有分卡结果。
- H3 端到端配方。训练侧现在可以走完整分片（`shard_trainable_only: false`、
  `precision_policy: none`），但 H3 的 rollout 也放不进一张卡：分卡生成
  （`partitioned_generation.py`）目前只有 `partitioned_h3_probe.py` 这个探针在用，
  还不是公开的 Ray rollout 策略，所以现有 colocated 4×1 布局（每 rank 一张
  rollout 卡）对 H3 不成立。没有为此加不能跑的 preset。之后退掉训练侧
  `placement.py` 分卡加载也要等这一步。
- 每个 rank 直接从 safetensors 读自己分片（互联流量为零）的更省方案。
