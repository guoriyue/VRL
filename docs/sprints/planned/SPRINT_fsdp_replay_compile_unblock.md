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
