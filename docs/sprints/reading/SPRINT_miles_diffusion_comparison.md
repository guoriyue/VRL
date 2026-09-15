# 阅读：miles / miles_diffusion 与 VRL 的对照（2026-09-14）

来源：`/home/ubuntu/miles`（radixark/miles @2ef603a）、`/home/ubuntu/miles_diffusion`
（@ebd55fc，30 个提交，17k 行，62 个测试文件 / 192 个测试）。VRL 同日：110k 行，
384 个测试文件 / 3320 个测试，27 个模型家族，76 个 experiment recipe，24 个 reward。

## 进程模型（本次对照的起因：driver 侧 GIL 争用）

| | miles_diffusion | VRL |
|---|---|---|
| driver | 纯控制平面，只 `await` actor | torchrun rank 既训练又跑 producer |
| 训练 | `TrainRayActor`（FSDP2 + USP） | rank 进程内 FSDP2/DDP/single |
| rollout | sglang-diffusion 服务进程（引擎侧 TP/SP、CUDA graph） | 自研 denoise loop，Ray actor，eager（SD3.5 512px 发射绑定，SM 44%） |
| 反序列化 | msgpack 原始字节 + `RolloutImageResponseParserActor` 池 | driver 内 pickle |
| reward | Ray actor 池（PickScore GPU 池、OCR CPU 池） | 托管服务子进程 + park/wake 租约（2026-09-14 落地） |
| 流水 | 每 microgroup 一个 asyncio task：generate → deserialize → reward | strict 串行 / continuous prefetch |
| 大 reward 与 trainer 共卡 | 无（独立 reward GPU，或常驻小模型 colocate 0.05 GPU） | CuMem park/wake 时分（4 卡跑 7B HPSv3） |

## 他们领先的

1. 进程模型本身：计算从不进 driver，我们刚花一天证明并修补的问题在那里不存在。
2. rollout 引擎：sglang-diffusion（引擎侧并行与优化），并用 monkey patch + deterministic
   模式压 train/rollout 不一致；我们的 parity 门在 compile / batch>1 下都失败。
3. 训练侧 USP（Ulysses × Ring，来自 diffusers `_cp_plan`）；我们只有 rollout 侧手写 Ulysses（SD3）。
4. LoRA 权重经 CUDA IPC 直接同步到 colocated 引擎。
5. 已验证的多节点配方（2×8 + 1 reward GPU）、dashboard 时间线。

## 我们领先的

1. 广度：27 家族（含 AR token 模型 janus/llamagen/emu3/glm_image 和 cosmos 2/2.5/3、hunyuan、
   mochi、magi、H3），76 recipe，24 reward（Kling、VideoCon physics、UnifiedReward、HPSv3、机器人 IDM）。
2. 算法：GRPO / flash-GRPO / continuous GRPO、DiffusionNFT、V-GRPO、DPO、TIS/RS 精度修正、
   replay parity 与 drift 门；他们是 Flow-GRPO、NFT、SFT。
3. GPU 时分：4 卡跑 Wan + 7B reward 的 phase-cycled 布局，他们要第 5 张卡。
4. 类型化配置、resume 的 RNG 逐位一致、run evidence、测试量级（17×）。

## 结论

不是"整体学"而是"进程模型学、其余保留"。按收益顺序：

1. artifact 材料化与反序列化出 driver（parser actor 池的形状），
   见 `planned/SPRINT_reward_service_isolation.md` 之后。
2. sglang-diffusion 作为 rollout provider：`parked/SPRINT_sglang_diffusion_execution_provider.md`
   的触发条件应重估——miles_diffusion 已证明它能跑 Wan2.2/LTX/Cosmos3。
3. 训练侧 USP（视频大模型）。
4. driver 拆成控制 + 训练 actor（`TrainRayActor` 形状），最后做。

## 附录 A（2026-09-14）：训练侧 USP 与 parity 工具链的逐行对照

### A.1 训练侧 USP（对应本仓库 WS-C）

| 维度 | miles_diffusion | VRL（WS-C 落地形状） |
|---|---|---|
| 配置 | `--sequence-parallel-size` + `--ulysses-degree`（0=auto，ring 派生）| `fsdp.mesh: [dp_shard, cp]` + `fsdp.context_parallel.{ulysses_degree, ring_degree}` |
| mesh | `world._unflatten` 成 `(dp, sp)`，再 `sp → (ring, ulysses)`；**SP rank 在 FSDP 分片轴内** | FSDP 仍在整个 world 上 1D 分片；另建 3D `("dp_shard","ring","ulysses")` 给 diffusers |
| 注意力 | 复用 diffusers `_cp_plan` 元数据，但自写 hook + 自写 `usp_attention`（all-to-all + torch 私有 ring） | 直接调 diffusers `enable_parallelism`（同一份 plan，官方 Ulysses/ring 实现） |
| 样本分发 | rollout manager 只按 dp 切，SP 同组按 `dp` mesh local rank 取**同一个 Ray object** | leader 采样、组内 gloo 广播（VRL 的 rollout 是 rank 本地 Ray runtime） |
| loss / log-prob | 边界 gather 后全序列，loss 代码不感知 SP | 同 |
| 梯度 | gather 的 backward 先在 SP 组 all-reduce，配合 FSDP 的 1/(dp·sp) 平均 | diffusers gather backward 只取本地分片；`FSDPStrategy.backward` 把 loss 乘 cp，配合 FSDP 的 1/(dp·cp) 平均。数学等价，少一次集合通信 |
| 裁剪 | 普通 `clip_grad_norm_` + `full_tensor()` | 普通 DTensor 感知 `clip_grad_norm_` |
| 指标 | 只在 dp 组归约；`grad_norm` 标记 REPLICATED | world 归约；CP 同组值相同，max/all_true 不变，sum 类只用作比值 |
| rollout SP 与训练 SP | 完全解耦（引擎 `--sglang-sp-degree`，`tp*sp == gpus_per_engine`） | 解耦；follower 的 rollout 引擎目前空转，后续可与 `gpus_per_engine == cp` 配对 |
| 一致性守卫 | `validate_same_microbatch_counts_across_train_ranks` 在 dp×sp gloo 组 all-gather 微批数 | `_TrainingMicrobatch.plan_balanced` 已用 `max_int` 拉平 |

### A.2 rollout/train parity（对应 WS-F 与 batch-16 / compile 漂移问题）

它们的原则：**不放宽 ε，把两个前向做到一致**。`--diffusion-clip-range` 默认 1e-4（与我们相同），没有 TIS/mask 修正。

1. `--diffusion-recompute-old-log-prob`：训练侧用更新前权重重算 old log-prob；第 0 个优化窗口直接
   `old = new.detach()`（ratio 恒 1，零额外前向）。这正是 rollout 核 vs 训练核漂移的逃生口。
2. rollout microgroup 与训练 micro-batch 形状对齐（`train-dp-split-mode contiguous`），承认 batch 形状
   决定数值（`patch_qwen_image.py` 的 `split_seqs` 连续化就是 batch>1 才出现的漂移源）。
3. 引擎侧 `SGLANG_BATCH_INVARIANT_OPS_ENABLE_MM_FALLBACK_VARIANT=true`（batch 不变 matmul）；rollout
   路径**不用** torch.compile 与 CUDA graph。
4. 融合核的中途 bf16 舍入：Wan 每个 norm/residual 站点 ~3e-3 相对误差，30 块累积正好落在我们测到的
   0.014–0.03；修法是两侧在**同样的位置**舍入（fp32 LN/residual/RoPE + 一次 `.type_as`）。
5. 边界 dtype 显式声明（`input_dtype_policy`：latents/cond/timestep），timestep/sigma 网格保持 fp32。
6. 诊断模式：`--diffusion-debug-mode --debug-skip-optimizer-step` + `MILES_VERIFY_WEIGHT_SYNC=1`，引擎
   回传每步 `model_output`，训练侧记 `model_output_{mean,max}_abs_diff / rel_max`、
   `log_prob_mean_abs_diff`、`ratio_abs_minus_1`、`approx_kl`、`clipfrac`；把权重同步、前向路径、
   优化器三种漂移分开看。验收带：H3 mean|Δlogp| 1.4e-5..7.1e-5；Wan2.2 全参 + norm patch +
   `torch_sdpa` 做到 diff 恰好 0、ratio 恰好 1（200 rollout 持续）。CI 用确定性模式做逐位对比。

对 VRL 的落地顺序（先便宜后贵）：
- 立刻：SD3.5 batch-16 配方开 `precision_correction.recompute_old_logprob`（ppo_epochs=1 时第一窗口
  零成本），预期 pre-update clip 从 54–60% 归零；这同时解锁 compile。
- 其次：replay 批形状 = rollout 批形状（内存允许时 `training_microbatch_size` 对齐 16），已知等形状漂移为 0。
- 再次：Wan 家族对齐 norm/residual/RoPE 舍入位置，作为 B spike 引擎 provider 的 patch 组。
- 长期：frozen-weight 诊断模式 + `model_output_*` 指标 + 确定性模式下的逐位 CI 标准。

## 附录 B（2026-09-14）：VRL driver 进程里的 rollout 后处理清单（A 第 2 步 / D 的输入）

Ray worker 返回后、trainer 拿到 `RolloutBatch` 前，driver 进程要做的事（都在 rank 进程的
事件循环线程或 collector 线程里）：

| 步骤 | 位置 | 性质 |
|---|---|---|
| Ray object 反序列化（GenerationBatchResult，含 `[B,T,*latent]` 轨迹张量） | `vrl/generation/ray/executor.py` `ray.get` | pickle，持 GIL |
| 多 batch 拼接、覆盖校验、replay 张量对齐、轨迹构建 | `vrl/generation/bindings/full_sequence_denoise/gather.py:36-115` → `vrl/trajectory/builders.py:30-135` | 纯 CPU 张量 cat/校验 |
| 存储策略再应用（dtype/device） | `vrl/trajectory/storage.py` | 张量拷贝 |
| reward 样本构建、artifact 采纳 | `vrl/rollouts/collector/batch_builder.py`、`vrl/rewards/artifacts.py` | 已在 WS-A 第 1 步瘦身 |
| 训练 batch 组装（rewards/group_ids/extras） | `vrl/rollouts/collector/core.py` `prepare_training_batches` | 小 |

worker 侧对应的是 `worker.py:910-948` `_copy_output_to_cpu`（pinned 异步拷贝）。
A 第 2 步的形状：把 gather + builders + storage 三步放进 CPU actor（每个 rank 一个池），driver 只
`ray.get` 一个已构建好的 `RolloutBatch` 引用；门是 py-spy 里 driver 侧这些帧的占比。

## 附录 C：每小时架构笔记

### 已研究主题（避免重复）
- 2026-09-14 23:40 — 并行状态记账：dp×sp 全体 rank 的微批计数守卫 / CP 同组样本一致性
- 2026-09-15 00:50 — 生命周期握手与失败处理：子进程就绪等待、死亡检测、身份核对
- 2026-09-15 01:55 — 指标归约范围：dp 组 vs world，CP 副本的重复计数
- 2026-09-15 02:45 — 权重更新事务与版本门：版本回执、内容核对、LoRA 合并时机
- 2026-09-15 03:45 — checkpoint/resume 身份：tracker 文件、RNG 与进度恢复、身份校验
- 2026-09-15 04:45 — staleness / off-policy 记账：过期样本是断言还是策略
- 2026-09-15 05:40 — 进程/GIL 隔离：训练进程里到底跑了什么（两次 py-spy 的证据）
- 2026-09-15 06:45 — 多节点放置：bundle → 物理卡的确定性映射，跨集群占用不可见
- 2026-09-15 07:45 — 反序列化与 batch 构建出事件循环（parser actor 池）：按三次 py-spy 判定不做
- 2026-09-15 08:45 — 时分租约的相位切换：sleep/offload/onload/wake 的顺序、失败组合与健康探测
- 2026-09-15 09:50 — reward 的媒体契约：生成输出到 reward 模型之间只允许一处归一化

### hourly note 2026-09-14 23:40 — 集合通信计数守卫与 CP 同组一致性

**它们解决的问题**：FSDP 的 forward/backward 是集合通信；若某个 rank 的微批数不同，其余 rank
会在 all-gather 上无限等待，错误表现为"挂死"而非报错。miles_diffusion 在训练前把每个 rank 的
微批计划在 dp×sp 的 gloo 组上 all_gather 一次，不一致直接抛错
（`miles/utils/train_data_utils.py:316-333`，调用点 `miles/backends/fsdp_utils/actor.py:394`）。

**VRL 今天怎么做**：更强——不是校验而是修复。`_TrainingMicrobatch.plan_balanced`
（`vrl/trainers/online/trainer.py:353-396`）用 `collectives.max_int` 取全体 rank 的最大槽数，
本地不足的用零权重 dummy 槽补齐，因此本地零优势过滤造成的不均衡不会挂死；跳过决策也用
`all_true` 统一（`trainer.py:1302,1452`）。

**CP 引入的新缺口**：CP 同组两 rank 的槽数由 leader 广播保证相等，但 token 分片前向假设两边
喂的是**同一批样本、同一顺序**；若内容不同（广播异常、leader 重试后返回不同列表、未来的
非广播共享路径），不会挂死，而是静默训练垃圾。这类"计数相等但身份不同"的失败是 VRL 现有守卫
覆盖不到的。

**改动（已实现）**：`ContextParallelRolloutSchedule.next_iteration` 在广播后于 CP 对象组
all_gather 一个便宜的批身份指纹（每批样本数、group_ids、reward 和），不一致即抛
`RuntimeError`（`vrl/rollouts/orchestration/context_parallel.py`）。成本：每次迭代一次 gloo
all_gather_object，字节量为 KB 级。风险：无（同组指纹恒等时零副作用）。门：cp=2 验收 run
的两个 update 通过且日志无 fingerprint 错误。

### hourly note 2026-09-15 00:50 — 子进程就绪握手：活着、健康，还要"是它"

**问题**：driver 拉起的子服务（reward 服务、引擎）在"健康"之前有三种失败：子进程死了、
永远不健康、以及**健康但不是我拉起的那个**。第三种最阴险：今天 cp=2 首次启动时两个 rank 把
服务 YAML 写到同一路径，rank 0 的子进程读到 rank 1 的端口，`/ready` 在 rank 1 的端口上返回
200，模型名和版本都一致，rank 0 却在自己的端口上等到超时。

**参照实现**：miles_diffusion 的引擎启动只做前两种——循环 GET 健康端点，同时检查
`process.is_alive()`，死了就抛（`miles/backends/sglang_diffusion_utils/sglang_diffusion_engine.py:87-104`），
没有 deadline，也不核对身份。

**VRL 今天**：`ManagedRewardScorer._wait_ready`（`vrl/rewards/service/managed.py`）已经有
子进程死亡检测 + deadline 超时即终止，比参照更严；`HttpRewardScorer` 的 preflight 还核对
`/info` 的 `expected_model` / `expected_model_version`（`client.py:270-282`）。缺的是"每次启动
唯一"的身份：同名同版本的另一个服务能通过所有检查。

**改动（已实现）**：`RewardServiceConfig.launch_token`（launcher 写入，随机 32 hex）→
`RewardServiceInfo.launch_token`（服务在 `/info` 回显）→ `ManagedRewardScorer` 在 ready 后核对，
不匹配抛 `LaunchTokenMismatch` 并终止自己的子进程。风险：外部 `http` 模式不受影响
（token 为空时不核对）。门：端到端服务测试（真实子进程）通过；重放今天的双 rank 场景应在
数秒内报错而非 30 分钟超时（已由 rank 后缀修复根因，token 是纵深防御）。

**推广**：任何 driver 拥有的子进程（未来的引擎 provider、并行 reward 池）都应遵循同一三元组：
死亡检测 + deadline + 每次启动的身份令牌；`/ready` 只回答"能服务"，"是谁"由 `/info` 回答。

### hourly note 2026-09-15 01:55 — 指标归约的范围：谁该被算一次

**问题**：训练指标（reward 均值/方差、clip 比例、parity 最大值、loss）需要跨 rank 归约；
一旦 world 里出现"同一份样本的副本"（CP 同组的两个 rank 持有相同 batch），按 world 求和的
指标会把副本重复计数。

**参照实现**：每个指标声明归约类型（`MetricReduce.MEAN / SUM / REPLICATED`），归约只在
**dp 组**上做（`miles/backends/fsdp_utils/metrics.py:20-60`；`actor.py:455-460` 用
`parallel_state.dp_group` 建 metric buffer），`grad_norm` 标为 REPLICATED（各 rank 相同，不归约）。

**VRL 今天**：`_global_reward_stats`（`vrl/trainers/online/trainer.py:74-99`）在 world 上
all-reduce (Σx, Σx², n)；parity/clip 用 `collectives.all_true / max_float / sum`
（`trainer.py:406-448`）；`TrainingCollectives` 没有组的概念（`vrl/trainers/distributed.py`）。
CP 下的实际影响：均值/方差与比值类指标**不变**（副本让分子分母同倍放大），max/all_true 幂等，
只有绝对计数（`clip_total`、`total_weight` 等）翻倍——目前这些只用于比值。因此现有 cp=2 门
用"reward_mean 与单卡基线相等"就能验证这一点（基线 epoch 0：−4.5297）。

**VRL 该改什么（后续，非本小时）**：给 `TrainingCollectives` 一个可选的归约组，CP 下由策略在
`prepare_model` 后把它设为 dp 组（world 按 cp 步长取 rank），绝对计数指标就一次都不多算；
`grad_norm` 天然 REPLICATED（DTensor 裁剪已在整个 world 上算出同一个数）。风险：所有调用点
都在 rank 对称路径上，换组不改变调用次数；门：cp=2 下 `pre_update_clip_fraction` 的分子分母
日志与基线一致，绝对计数减半。

### hourly note 2026-09-15 02:45 — 权重更新：版本回执之外还要"内容回读"

**问题**：训练侧把新权重推给 rollout 引擎后，怎么知道引擎真的在用它们？错误的表现不是崩溃，
而是 rollout 用旧策略生成、ratio 悄悄偏离；诊断漂移时它和 kernel 差异混在一起。

**参照实现**：更新按 bucket 推送、LoRA 逐层即时合并（`diffusion_update_weight_utils.py:337-360`），
并有一个环境变量门控的**回读核对**：推送后把引擎侧权重读回与训练侧比较
（`MILES_VERIFY_WEIGHT_SYNC`，`:338-339`），只在诊断时开。

**VRL 今天**：`RayGenerationWeightSync.push_to_rollout_engines`（`vrl/generation/ray/weight_sync.py:68-98`）
把 `policy_version` 随 payload 一起推，要求每个引擎所有 rank **回执同一版本**才算成功；
worker 侧 `update_weights(..., verify_content=)`（`vrl/generation/execution/worker.py:188-250`）
支持内容核对，且有 `begin/receive/commit/abort_weight_transfer` 的分段事务和
`verify_active_weights`。即版本门与内容核对 VRL 都有，且核对是请求参数而非环境变量。

**差异与可做的事**：(1) VRL 的 `verify_content` 默认关，与参照一致；建议在 `trainer.debug.first_step`
的首个 update 自动打开一次（首次 sync 后核对一次，之后关闭），把"权重没到位"从 parity 诊断里
排除，成本一次读回；(2) LoRA 路径：VRL 传扁平化的可训练状态、引擎侧 `install_trainable_state`，
不做合并，合并 dtype 问题不存在；参照的即时合并只在引擎不支持 adapter 时才需要。
风险：仅 debug 首步多一次核对；门：首步日志出现一次 `verify_content=True` 的通过记录。

### hourly note 2026-09-15 03:45 — resume 的三件事：找到最新、恢复"从哪继续"、确认"是同一个实验"

**参照实现**（LLM RL 的 FSDP 后端，`miles/backends/fsdp_utils/checkpoint.py:85-200`）：
`latest_checkpointed_iteration.txt` tracker 指向最新 step；分别 `dcp.load` 模型/优化器/LR 调度器；
`rng.pt` 恢复 torch/cuda RNG（可用 `no_load_rng` 跳过）；`iteration` / `next_rollout_id` 决定从哪个
rollout 继续。**没有身份校验**：换了配置或模型指向同一目录也会静默加载。

**VRL 今天**（`vrl/trainers/checkpointing.py:180-232, 387-400`；`vrl/scripts/common/online.py:638-726`）：
schema-v2 checkpoint 携带 `family` 与 `model.identity`（构建期的模型身份字典），`strict` 模式下
家族不符或身份不符直接 `ValueError`，缺身份也拒绝；进度（epoch/step）和 RNG（含 prompt 采样器的
generator）随 checkpoint 捕获与恢复，`resume_epoch` 由 `build_configs` 解析。VRL 在"是同一个实验"
这一项上比参照严格得多；tracker 文件的"最新"语义 VRL 用 `checkpoint-final` 与显式路径承担。

**可做的事（后续，非本小时）**：(1) 把 rollout 编排状态也纳入身份/进度——continuous 模式的
`policy_version` 与 staleness 窗口在 resume 后应从 checkpoint 恢复而不是从 0 开始，否则第一批
rollout 的 staleness 记账错位；(2) `strict=False` 路径只 warning，多 rank 下应至少 `all_true`
统一所有 rank 的判定，避免一个 rank 拒绝、其余 rank 继续导致集合通信挂死。风险：只加校验；
门：resume 单元测试覆盖身份不符 + 多 rank 判定一致。

### hourly note 2026-09-15 04:45 — 过期样本：断言、丢弃还是回炉

**问题**：异步/连续采样下，一个 prompt 组从生成到进训练之间策略可能已经更新了 k 次；k 超过
允许窗口时怎么办，决定了系统在"reward 慢了一拍"这类抖动下是继续还是崩溃。

**参照实现**（LLM RL 的全异步模式，`miles/utils/arguments.py:717-775`）：`--max-weight-staleness`
是**过滤器**而不是断言——超窗的组按 `--async-unused-samples-handler` 处理：`drop`（默认丢弃）或
`retry`（prompt 回炉重生成）；同时 `--async-data-buffer-capacity-factor` 给成品缓冲区设上限，
缓冲满时生产者阻塞，保证生成不会无限领先训练；`--async-max-concurrent-samples` 把并发生成量与
训练 batch 解耦。

**VRL 今天**（`vrl/rollouts/orchestration/continuous/staleness.py:14-42`，
`producer.py:536-552, 736-752`）：`StalenessPolicy.too_stale` 是一个**不变量**：组在 reward 前或
完成时超窗直接 `RuntimeError`，经槽位失败预算（3 次）后整个 run 失败。VRL 的连续调度只有
`max_stale_policy_versions` 一个窗口（默认要求 ≥1），单槽生产者，版本屏障由消费者校验（负
staleness 视为违规）。这在 max_stale=1、单槽的形状下等价于断言"生产者永远不会落后超过一个
版本"，正常运行确实成立；但一次 reward 服务超时或 HTTP 重试就会把它变成 run 级失败。

**VRL 该改什么（后续）**：把"超窗"从断言改成策略：`continuous.stale_policy: raise | drop | retry`
（默认保持 `raise` 以不改变现有语义），`drop` 记一个 `continuous.dropped_stale_groups` 计数并
释放槽位，`retry` 把 prompt 批放回生产队列头部。风险：`drop` 会让一次 update 的样本数少于
计划（需要与 `plan_balanced` 的跨 rank 对齐配合，本已支持不等槽数）；门：人为拖慢 reward
服务（sleep）时 `drop` 模式的 run 继续且计数递增，`raise` 模式行为不变。

**顺带**：本小时发现合并跑 CP 的 2 rank spawn 测试与 orchestration 测试时，pytest 在解释器退出
的 `multiprocessing._exit_function` 上挂住（join 存活子进程）；测试改为 join 超时后 kill。

### hourly note 2026-09-15 05:40 — 训练进程里到底跑了什么：用采样数据回答"要不要拆 driver"

**参照实现的形状**：driver 是纯控制平面——`actor_group.py:78-116` 用 `ray.remote(num_gpus=1)` 起
训练 actor，driver 只 `ray.get([actor.train.remote(...)])`、`update_weights.remote()`、
`wake_up.remote()`；rollout manager（`ray/rollout.py:47`）和引擎也是 actor。训练进程里除了训练
没有别的 Python 工作，GIL 争用在结构上不可能发生。

**VRL 今天**：torchrun 的 rank 进程既是训练进程，也是 rollout 采集的 driver（Ray RPC、pickle
反序列化、batch 构建、reward 客户端、continuous 模式的 producer 线程）。这正是 SD3.5 上进程内
CPU reward 与训练争 GIL（+61 s/epoch）的结构根源，已由"reward 一律独立服务"解决。

**这次的证据（Wan 1.3B + HPSv3 服务，严格模式，单 rank）**：生成阶段 25 分钟里 driver 的 Python
线程只活跃 3.5 s（0.2%），其中 81% 是 Ray pickle/torch.load；训练阶段 20 分钟里 driver 活跃 30%，
其中 49% 是 FSDP `wait_for_unshard`（等 GPU，无争用）、**17.5% 是训练 batch 上设备的同步 H2D 拷贝**
（`vrl/rollouts/batch/ops.py:80-100` → `vrl/trajectory/device.py:55-75`，`leaf.to(device)`），
pickle 只有 1.6%。

**结论**：严格模式下 VRL 的 rank 进程没有可观的 GIL 争用来源，拆 driver（D）在此没有收益；
driver 侧唯一值得动的是训练 batch 的 H2D 拷贝——但它是 CUDA 拷贝而非 GIL 问题，且 Ray 反序列化
后的张量不在 pinned 内存里，`non_blocking` 要先付一次 pin 拷贝，收益要先测再改（门：单 rank
训练阶段 `move_training_batch_to_device` 的 py-spy 占比从 17.5% 降到 <5% 且 epoch 墙钟不劣化）。
D 的最终判定等 continuous 模式（producer 线程与 backward 同进程）的 py-spy：那是 GIL 争用唯一还
可能出现的形状。

### hourly note 2026-09-15 06:45 — 放置：谁决定"rank i 在哪张卡"，以及别人占着的卡

**参照实现**（`miles/ray/placement_group.py:11-55`）：一个 PACK 的 placement group，每 bundle
1 GPU；用探针 actor（`InfoActor.get_ip_and_gpu_id`）读出每个 bundle 实际落在的节点 IP 与 GPU id，
再按（IP 数值, GPU id）排序，得到 rank → (节点, 卡) 的**确定性**映射；多节点脚本在此之上配
引擎 sp_degree 与训练 dp×sp（17 卡脚本 `:114-121` 把 reward 单独放一张卡）。

**VRL 今天**（`vrl/ray/placement.py:200-300, 397, 473-494, 540-560`）：同样是"探针 + bundle
匹配"的形状：`GlobalRayPlacementOwner` 探测 bundle 的 GPU id，按 role 的设备计划
（`resources.rollout.devices` 等）匹配 bundle；跨节点用 `SPREAD` 与 `cross_node_preflight`
（校验非 driver 节点的 GPU 数），并校验 actor 实际拿到的 GPU id。差别在**匹配方向**：参照实现
接受 Ray 给的卡再排序命名，VRL 要求 Ray 给的卡覆盖配置点名的物理卡——当同一台机器上还有
别的 Ray 集群占着卡时，Ray 的调度器看不见那份占用，把 bundle 放到了 0-1，而配置点名 2-3，
于是 fail-closed（今天 05:5x 连续踩到两次）。正确的启动方式是 `CUDA_VISIBLE_DEVICES` 窄化到
`visible_devices`，让 Ray 只能探到这些卡；`_local_torch_ordinal`（`vrl/ray/resources.py:343-365`）
再把计划序号按位置翻译成 torch 序号。**这条规则此前只在记忆和一次 4 rank smoke 的注释里**，
本小时把它写进了报错信息（`placement.py:556`）。

**后续（多节点前必做）**：把"每个 bundle 的 (节点 IP, GPU id)"写进 `run_evidence`，多节点时按
(IP, id) 排序生成 rank 映射并与 `num_nodes × gpus_per_node` 对账；单机上则允许一个
"接受 Ray 所给的卡"的模式（配置只写数量不写卡号），避免与其他集群的占用打架。
风险：只改错误文案与文档；门：多节点 2×4 冒烟时 `run_evidence` 记录的映射与实际 nvidia-smi 一致。

### hourly note 2026-09-15 07:45 — 反序列化出事件循环：证据说不需要

**参照实现**：引擎响应是 msgpack，交给 `RolloutImageResponseParserActor` 池在事件循环外解码并
切成训练样本，driver 只 `ray.get` 结果——原因是它们的 driver 是单线程 asyncio 控制平面，
几百 MB 的解码会卡住事件循环。

**VRL 今天**：Ray worker 返回 `GenerationBatchResult`，driver 在 `ray.get` 处 pickle 反序列化，
随后 gather/轨迹构建/batch 组装都在 rank 进程（附录 B 清单）。

**证据**（三次 py-spy，Wan 严格生成期 / Wan 严格训练期 / SD3.5 continuous）：driver 侧 pickle +
torch.load 分别占活跃时间 81%（但绝对值 3.5 s/25 min）、1.6%（6 s/20 min）、0.2%（1 s/17 min）。
VRL 的 rank 进程并不是单线程控制平面：反序列化发生在训练循环的间隙，且 WS-A 第 1 步已把
最大的负载（视频 artifact）搬到 worker 侧落盘。

**判定**：不做 parser actor 池；A 工作流关闭。保留的观察：训练 batch 的 H2D 拷贝
（`vrl/trajectory/device.py:66`）在两次训练期测量里占 7–17%，是 driver 侧唯一剩余的可优化项，
门是先测 pinned+non_blocking 的净收益。

### hourly note 2026-09-15 08:45 — 相位切换：谁拥有顺序，失败时谁先回来

**参照实现**（`miles/backends/fsdp_utils/actor.py:261-280`，`miles/ray/rollout.py:241-262`）：
训练 actor 的 `sleep`/`wake_up` 是把模型和优化器整体 `.cpu()/.cuda()` 加一个 gloo barrier；rollout
manager 的 `offload`/`onload` 逐引擎调 `release/resume_memory_occupation`，`onload(tags=[weights])`
可以只恢复权重（更新权重前不必唤醒全部显存）；`offload` 前 `health_monitoring_pause()`——
卸载窗口内不做健康探测。顺序由 driver 脚本逐行写出，失败没有组合语义。

**VRL 今天**（`vrl/rollouts/orchestration/rollout_runtime.py:196-260`）：`rollout_phase` 上下文
管理器**拥有整个握手的顺序**：park trainer（仅拓扑要求时）→ activate 生成 runtime → 采样+打分 →
release 生成侧显存（含 reward 服务的 `/park`）→ 只有 release 成功才 restore trainer；body 失败与
cleanup 失败合并成 `RolloutPhaseCleanupError`，release 失败时 trainer 保持 parked 让终态 shutdown
先回收 rollout 的卡。park 本身在所有 rank 上用 CPU 协调组统一成败（`FSDPStrategy.park_training_state`），
并在 unmap 前 `cuda.synchronize` + 协调 barrier，避免在别的 rank 还在 unmap 时发 NCCL kernel
（2026-08-16 Xid 79 事故的对策）。VRL 的租约在顺序和失败语义上更完整。

**参照里的两点在 VRL 已有对应**：(1) 卸载窗口内的健康探测——VRL 的健康监视线程在 parking 切换
和 parked 期间暂停探测（`vrl/generation/ray/health_monitor.py:39,59-89`，启动即暂停直到 fleet 激活）；
(2) 分标签部分唤醒——VRL 的权重同步走 versioned slot，不需要唤醒引擎即可安装，没有"只恢复
权重"的需求。结论：相位切换这一项 VRL 不缺东西，不改代码。

### hourly note 2026-09-15 09:50 — 媒体契约：生成到 reward 之间只能有一处归一化

**问题**：生成侧为了传输把媒体量化成 uint8，reward 模型各自假设一种输入（[0,1] 浮点、uint8 HWC、
PIL……）。表示法在两者之间被"顺手"转换的次数越多，越容易出现今天这种静默错误：worker 侧直接
落盘 uint8，`to_uint8` 再乘 255 饱和成白图，OCR reward 掉到 1/3 却不报任何错。

**参照实现**（`miles/rollout/rm_hub/ocr.py:90-94`、`hps.py`、`pickscore.py`）：所有 reward 池都经过
**同一个**转换函数 `generated_output_to_rgb_hwc_uint8_frames(..., round_normalized=True)`，把
引擎输出统一成 RGB HWC uint8 帧，并写明"与参考实现逐位一致"的舍入方式；reward 模型不再各自
处理 dtype。

**VRL 今天**：reward 模型通过 `artifact.as_media()` 取媒体（`vrl/rewards/models/*.py` 十余处），
各自再做 `to_uint8` / `_extract_images` 等转换；磁盘表示由写入方决定（driver 路径 float k/255，
WS-A 后的 worker 路径曾是 uint8）。契约分散在写入方、accessor 和每个模型三处。

**改动（已实现）**：把归一化收口到 accessor——`RewardInferenceArtifact.as_media()` 对 `.pt` 里的
整数张量统一还原成 [0,1] 浮点（`vrl/rewards/inference.py`），写入方可自由选择 uint8 存盘
（4× 更小），reward 模型只面对一种表示；`to_uint8` 对 uint8 透传作纵深防御（91c73a5b）。
风险：无（浮点 `.pt` 与内存路径行为不变）；门：SD3.5 recompute arm 重跑的 epoch 0 reward 回到
0.3–0.4 区间，`tests/rewards` 全绿。后续：materialize 改回 uint8 存盘以省 4× 磁盘与 IO。
