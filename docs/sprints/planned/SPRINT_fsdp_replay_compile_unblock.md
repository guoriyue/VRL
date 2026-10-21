# SPRINT：FSDP replay 下的 torch.compile——先在当前栈重测两条硬冲突

状态：**planned（2026-10-20）**。来源：PyTorch Conference NA 2026 Day 1。torchtitan 的 benchmark 一直在同时用 FSDP2、compile、Float8 和 AC/SAC（海报 P85、P92、P124）；097 讲 nested graph breaks；海报 P94 讲 Mega-Cache；005 是 PyTorch 2.14 的 compile 改进。

## 0. 结论先行

VRL 有两条 compile 硬冲突：compile × FSDP 和 compile × gradient checkpointing。前者**从来没有在本仓库测出过失败**；后者只在 torch 2.11 上测过 selective AC。另外，VRL 现在的包装顺序是 compile → AC → fully_shard，和 torchtitan 正好相反（torchtitan 是 AC → compile → fully_shard）。

第一步是在当前栈上按 torchtitan 的顺序重测。测通了才放开冲突；测不通，就把失败记录写进 validation 的注释。

FSDP replay 能 compile 之后，有两个好处：大视频家族的 replay 会更快；rollout 和 replay 能跑同一张编译图（`ReplayParityConfig` 的 docstring 写过，这种条件下 parity 可以达到 1e-6）。

## 1. 当前代码现实（HEAD a914644c6 核对）

**两条冲突**都在 `vrl/config/validation.py`：
- compile × checkpointing（:98-113）。证据：`archive/main-20261016:docs/sprints/parked/SPRINT_training_mfu_selective_checkpointing.md:105-125`，2026-06-27，RTX 5090，torch 2.11。表里只测了 selective，batch 1/2/4 都报 `InternalTorchDynamoError`。代码注释说 full AC 也失败，但没有找到对应的测量表。
- compile × FSDP（:115-130）。另外 `vrl/trainers/strategy.py:978-989` 在运行时还会再拒绝一次。来源是 commit `232e29e22`，理由只是引用 cosmos-rl 的一个 assert，没有本仓库的实测失败。

**当前顺序**：
1. `vrl/scripts/common/online.py:582` 调 `build_replay`，在 `vrl/models/steps/denoise/build.py:145-147` 做 compile；
2. `online.py:587` 开 AC，即 diffusers 的 `enable_gradient_checkpointing`（`vrl/trainers/activation_checkpointing.py:95-124`）；
3. `trainer.py:581` 调 `prepare_model`，再由 `fsdp.py:203-217` 做 `fully_shard`。

**compile 的形态**：
- 整根编译：`vrl/models/steps/denoise/base.py:446`，`fullgraph=False`。
- 分区编译：`vrl/nn/optimization/regional_compile.py:62`，逐 block 调 `block.compile(fullgraph=False)`。
- `dynamic` 没有设置；mode 默认 `"default"`。
- 没有任何 graph-break 日志，也没有 recompile 上限的配置。

**已有的实测**：`experiment/wan_2_1/online_grpo_hpsv3_fsdp_4x_l40s.yaml:89-93` 只 compile rollout，每组 446s → 324s。

**冷编译**：约 30 分钟（`vrl/config/schema.py:701-704`，`generation_stall_timeout_s=3600` 就是为它设的）。全仓没有用 `save_cache_artifacts`/`load_cache_artifacts`。`TORCHINDUCTOR_CACHE_DIR` 只在测试 runner 里设置（`tools/pytest/main.py:19`）。

**锁定版本**：torch 2.11.0，triton 3.6.0，diffusers 0.41.0.dev0。

## 2. 阶段

### P0. 重测，只写探针，不改生产代码

1. 选一个有 FSDP preset 的家族：Wan2.1 1.3B，或 Cosmos2.5 2B。2 卡，LoRA。
2. 三种组合，各跑 ≥3 次 update，都开 parity gate：
   - (a) FSDP + 逐 block regional compile，不开 AC；
   - (b) 按 torchtitan 顺序：每个 block 先 AC（先试 diffusers 的 full，再试 SAC），再 `block.compile`，最后 `fully_shard`；
   - (c) 当前顺序，作为对照，确认它确实失败，并记下错误信息。
3. 每组记录：是否成功；replay+backward 的耗时（用 trainer 的 `profile` 阶段）；显存峰值；首次 update 的 |Δlogprob|。
4. **验收**：得到一张实测表，写进本文件。

### P1. 根据 P0 放开冲突（只在 P0 证明可行时做）

1. 把顺序改成 AC → compile → fully_shard。改动点是 `build.py:145-147` 的 compile 时机和 `online.py:587`。注意 `torch_compile_for_role` 和 rollout 侧 `CompilePass`（`passes.py:229-272`）的验证不能退化。
2. 删除或收窄 `validation.py` 的两条冲突，以及 `strategy.py:978-989` 的运行时拒绝。保留一个条件：如果某个组合在 P0 里仍然失败，就留下这一条，并写明实测证据。
3. 更新 `tests/config/test_validation_tiers.py`、`tests/config/test_rollout_drift_guard.py` 里相关的用例。
4. **验收**：P0 那个家族的 FSDP preset 开启 compile 后，parity gate 通过，replay+backward 耗时下降。如果 rollout 和 replay 用同一张编译图，阈值就收紧到 1e-6。

### P2. 冷编译缓存（独立于 P1）

1. 第一个 rollout batch 成功后，primary rank 调 `torch.compiler.save_cache_artifacts()`，用 `vrl/utils/artifacts.atomic_file` 原子写入 run 目录。resume 时，以及其他 Ray worker 编译之前，先调 `load_cache_artifacts()`。
2. **不要**让多个 worker 共享同一个可写的 inductor cache 目录（海报 P94 的教训：会有竞态）。
3. **验收**：同一 preset 第二次启动时，记录"首个 batch 完成时间"，并和冷启动对比。缓存命中后，parity gate 仍然通过。如果收益不到 5 分钟就不合入（Wan/Cosmos 大分辨率的冷编译是 25–41 秒到 30 分钟不等）。

### P3. 让 graph break 可见（一次性测量）

1. 用 `TORCH_LOGS=graph_breaks,recompiles` 跑一遍各家族的 rollout compile，记录每个 block 的 break 数量。
2. 对于 0 break 的家族，把对应的 regional compile 改成 `fullgraph=True`，以后一回归就会直接报错。仍然有 break 的家族，只把 break 位置记在本文件里。
3. torch 2.14 的 `nested_graph_breaks` 和"小编译区域开销"优化，放到升级 sprint 之后再看（`SPRINT_torch_vllm_ray_upgrade_readiness.md`）。

## 3. 不做

- 不引入 CUDA graph 和 reduce-overhead。这一块归 `SPRINT_rollout_reward_throughput_pt26.md` 的 CUDA graph 探针。
- 不做 TorchTitan GraphTrainer 那种整步成图。

## 4. 实施记录（2026-10-20，单卡 RTX 5090，torch 2.11）

### P0 重测（replay 前向 + 反向，6 个 SDE 步，LoRA，单 rank NCCL）
SD3.5 Medium，batch 2，512px：

| 组合 | 能否运行 | 每次 replay（warm） | 峰值显存 | 相对 eager rollout 的 \|Δlogprob\|（最后一步） |
|---|---|---|---|---|
| eager，无 FSDP | 是 | 1.25 s | 25.4 GB | 0 |
| eager + FSDP | 是 | 1.35 s | 25.4 GB | 0 |
| regional compile，无 FSDP | 是 | 0.91 s | 21.8 GB | 9.7e-2 |
| compile → FSDP（VRL 现有顺序） | 是 | 1.00 s | 21.8 GB | 1.0e-1 |
| FSDP → compile | 是 | 0.98 s | 21.7 GB | 9.8e-2 |
| root compile + FSDP | 是 | 0.97 s | 21.7 GB | 1.0e-1 |
| eager + full AC + FSDP | 是 | 1.85 s | 16.6 GB | 0 |
| regional compile + full AC + FSDP | 是 | 1.30 s | 16.4 GB | 1.0e-1 |
| root compile + full AC | 是 | 1.11 s | 16.3 GB | 1.2e-1 |
| regional compile + selective AC（±FSDP） | 是 | 1.14–1.24 s | 16.7–16.9 GB | 1.0e-1 |

另外两个家族：
- Cosmos-Predict2.5 2B（256px，9 帧）：regional 和 root compile 加 full AC、regional 加 selective AC 都能跑。
- Wan2.1 1.3B：regional compile 加 full AC 能跑；**regional compile 加 selective AC 复现了** `InternalTorchDynamoError: checkpoint not implemented for NestedUserFunctionVariable context_fn`。

冷编译 16.6–18.6 s（regional）、62–72 s（root）；同机 warm 缓存是 3.9–6.1 s。

### P1 放开的冲突（`config(validation): allow torch.compile with full activation checkpointing`）
- compile × full AC：放开。
- compile × selective / full_cpu：保留。selective 在 Wan 上失败；full_cpu 没有测过。
- compile × FSDP：**保留**，但报错信息改成实测结论：单 rank 上两种包装顺序都能跑，多 rank 的 reshard-after-forward all-gather 还没验证（这台机器只有一张卡）。
- 删掉了 `enable_transformer_gradient_checkpointing` 和 `build_strategy` 里重复的运行时检查。所有入口都经过 `BuiltConfigs.from_cfg` → `require_training_config`，配置加载时就会拒绝。
- 两个 Cosmos preset 里过时的注释已更新。

### 多 rank 验证与放开 compile × FSDP（2026-10-21）
单卡机器上用两个 Gloo rank 在 CPU 上验证。上面那条保留理由说的是：多 rank 下 reshard-after-forward 的 all-gather 和编译图配合还没测过。这里测的就是这一点：
- 对象：tiny diffusers SD3 transformer（3 层）；按 online 的构建顺序（compile → full AC → `FSDPStrategy.prepare_model`，reshard_after_forward=True）；每个 rank 用 batch 的一半，跑 4 步 AdamW。
- 参照：单进程、不分片、同一种 compile 的模块用整个 batch 训练。
- 结果：与参照的最大差异，第一步梯度 1.5–2.6e-8（梯度量级 0.10），4 步后参数 1.8–6.0e-7。eager、regional、root、regional+AC、root+AC 五种在同一个噪声水平，分片没有引入额外误差。
- 编译确实生效：rank 0 上 regional 有 2 张图，root 有 5 张，加 AC 后都是 3 张。root+AC 在 checkpoint 的 `disable` 处有 1 个 graph break，只影响速度，不影响数值。
- 原因：FSDP2 的 all-gather 和 reduce-scatter 都在 hook 里，dynamo 不追踪这些 hook，它们在编译图外 eager 执行。GPU 上 stream 的调度也走同一段 eager 代码。

据此 `config(validation)` 删掉了 compile × FSDP 冲突。现在 FSDP replay 和单进程 replay 的 compile 规则一致，selective/full_cpu AC 和 SP 仍然拒绝。回归测试：`tests/trainers/test_fsdp_compile_distributed.py`（regional 和 root 两种，各加 full AC，约 40 s）。写这个测试时还发现一个陷阱：spawn 出来的 rank 不在 conftest 的 CUDA pin 范围内，inductor 会在第一个 checkpoint 前向里初始化 CUDA，`torch.utils.checkpoint` 不允许这样，所以测试里的 rank 隐藏了 CUDA。

仍未测：NCCL 下多卡的速度和显存。上面单 rank 的表说明 compile 能快 25–30%，多卡是否保持这个比例要有第二张卡才能测。各 FSDP preset 仍然用 eager，注释已改成"允许但该配方未实测"。

### 数值上的限制（决定了 compile 能用在哪里）
compile 后的 replay 和 eager rollout 不是同一张图，最后一步差 0.1。两侧都 compile 也不行：replay 开 grad 时是 autograd 图，和 rollout 的 inference 图仍差 7e-2（Wan 1.3e-2）。所以：
- 只在开了 TIS/RS（精度标签不同）的配方里，replay compile 是安全的；
- 严格 on-policy 且要求首步 parity 的配方，应该 replay 用 eager。

### P2 冷编译缓存（`scripts(train): keep the torch.compile cache across reboots`）
`vrl-train` 在 Ray 启动前把 `TORCHINDUCTOR_CACHE_DIR` 默认设为 `~/.cache/vrl/torchinductor`，launcher 已设置的值优先。`save_cache_artifacts` / mega-cache 没有做：它解决的是跨机器分发，目前没有多机。

### P3 graph break 普查
SD3.5、SANA、Wan2.1 1.3B 的 regional compile 都是 **0 个 graph break**（`TORCH_LOGS=graph_breaks`）。所以 `fullgraph=True` 不会带来速度收益；按家族加开关又属于保护性机制，不做。
