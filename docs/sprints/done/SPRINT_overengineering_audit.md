# SPRINT：全仓过度设计审计（投机功能 / 防御性代码 / 单实现接缝）

状态：**第一、二档与 §5 已执行（2026-09-28，52 个提交，每项一个）**；第三档按计划未动。
净删约 7.8k 生产行（vrl −8.7k/+0.9k，reward −0.9k）与约 10k 测试行；全量 CPU 套件与 48 个真实 Ray 用例通过。
刻意保留：`online.py` 里 reward parking 的早期校验（在模型加载与 Ray 启动之前失败，且逐 preset 的配置测试依赖它）；
写入时的 `validation_summary`（唯一能带出 train/eval episode 重叠警告的字段）；`draw_initial_latents` 钩子
（`qwen_image_21_edit_probe --compare-reference` 在用）。行为变化：训练与 rollout 共用 GPU 且 `offload.rollout=false`
现在在资源解析阶段就被拒绝（以前在运行时拒绝）；Wan 不再在 pipeline 加载时复查本地目录的 `boundary_ratio`；健康监控已删，
worker 故障由前台 RPC/stall 截止与 `RayActorError` 负责；被取消的 activate 直接让 runtime 进入终态；NFT / V-GRPO 与改前逐位一致
（tiny Wan DiT，8 组 × 4 步，loss、梯度与参数 `torch.equal`）。
原审计状态：planned（2026-09-27 审计完成，未改代码）。审计对象为 `main` @ `15bf62b2`（与 origin `5038f500` 只差 local-edit
常量的位置）。六条并行只读审计道覆盖 `vrl/` 全部子包、`reward/` 与对应测试，另一条专门评估 reward 栈外移。每条 finding
都读过函数体与调用点；标 ✅ 的是主线复核过的（grep / 运行时实测）。

## 0. 结论先行

- 生产约 9.6 万行、测试约 10.2 万行。按本轮标准可删 **约 9k 生产行（~9%）+ 约 10k 测试行**，其中高置信、低风险的
  第一档约 **4k 生产 + 4k 测试**。
- 与 2026-08 的 bloat audit 不同，这次几乎没有"死代码"：问题是**活着但没人用的机器**——默认关闭且无 preset 打开的功能、
  只有一个实现的 Protocol、同一条规则在三到六层重复校验、为从未发生的交错写的并发保护。
- 最值钱的单项不是删行，而是 **NFT / V-GRPO 每个 replay step 省一次 transformer forward**（§3 第 1 条）。
- reward 外移：**现在不建议拆成独立仓库**；先做两件小事即可拿到大部分收益（§6）。

## 1. 裁决规则（与 memory `no-speculative-protective-machinery` 一致）

一行代码留下，必须能指出它回答的实测问题或真实消费者。具体判据：

| 代码 | 判据 |
|---|---|
| A 投机功能 | 没有任何 preset / 实验 yaml / 记录在案的 run 打开它，只有测试到达 |
| B 单实现接缝 | Protocol / ABC / 注册表 / 策略表只有一个实现，包外无消费者 |
| C 角落管道 | 一个值穿过 3 层以上只为一个罕见情况 |
| D 防御代码 | 同一规则多层重复校验；内部边界的运行时类型检查；为不会发生的交错加锁 |
| E 无人设置的旋钮 | 所有调用方与 preset 都用默认值 |
| F 过度推导 | 复杂计算与一条简单规则等价 |
| G 过早优化 | 没有测过的瓶颈 |

## 2. 第一档：高置信、低风险，建议直接删

| # | 位置 | 类别 | 问题与证据 | 可删（生产/测试） |
|---|---|---|---|---|
| 1 | `vrl/nn/optimization/{fused_rms_norm,frame_shared_adaln,fused_gelu_projection,fused_lora_branch}.py`、`passes.py:236-341` 及 schema / ModelBuild 字段 | A/G | 四个 rollout 融合 kernel，全部默认关闭；✅ 528 行，preset 与 run 文档 0 引用；`SPRINT_gemm_utilization.md:68,82` 已判定"不值得碰" | ~700 / ~820 |
| 2 | `vrl/generation/execution/worker.py:322-507`、`batch_memory.py` 全文件及 executor/runtime/planner 接线 | A | `samples_per_generation_batch: auto` 启动期批大小探测；✅ 65 个 preset 全部写死整数，无一用 `auto` | ~480 / ~550 |
| 3 | `vrl/rollouts/orchestration/continuous/producer.py:100-110,493-581`、`thread.py:143-176` | A | `split_generation_reward` 生成/奖励双泵、`PendingRewardCapacity`、多批窗口；✅ 唯一 yaml 提及是注释；配置自述"opt-in until acceptance passes"，sprint 已 parked | ~290 / ~580 |
| 4 | `vrl/rewards/deployment.py`、`vrl/rewards/functions/registry.py` 的 combination/axis_mapping 分支、`vrl/config/reward_calibration.py`、`reward qualify` | A/H | 偏好拟合组合接入训练的"部署回执"；✅ 0 个 yaml 设 `reward.calibration`，磁盘上 0 个回执；同一规则在 registry 与 deployment 各查一遍。删掉还顺带切断 §6 的依赖泄漏 | ~330 / ~230 |
| 5 | `vrl/scripts/perf/common/baseline.py` + 两个 probe 的 `--record-baseline` | A/G | 追加式 JSONL 性能基线；✅ `read_baseline` 只有测试调用，默认输出 `docs/perf/BASELINE.jsonl` 从未存在 | ~300 / ~184 |
| 6 | 训练→评测证据链：`scripts/common/online.py:1104-1117,1234-1242` 的 seal、`vrl/trainers/trace.py:202-282`、`image_checkpoint_eval.py:301-430,849-876`、`GeneratorRuntimeIdentity` | A/G | 每次训练都对 `checkpoint-final` 做内容哈希并封回执；✅ 唯一读者 `verify_*` 只能经 `--verify-training-evidence` 到达，无任何记录使用；checkpoint 在评测里还被哈希两次 | ~340 / ~350 |
| 7 | `vrl/scripts/data/jrdb.py` | A | ✅ 从未在真实数据上跑过，preset / run 文档 0 引用，参考目录里只有 droid | ~236 / ~150 |
| 8 | `vrl/scripts/data/video_world/lerobot.py:229-276,574-629`（IDM 动作标签）与 `:279-313,403-493`（v2.1 回退） | A/F | 动作标签写进 manifest 但无人读（消费者已在 `da6b255c` 删除）；v2.1 回退用错模板键，真实数据上会 `KeyError` | ~235 / 0 |
| 9 | `vrl/models/families/wan_2_1/model.py:437-470,511-532` + `scripts/generation/qwen_image_21_edit_probe.py:158` | A | Wan `offload_mode=block`；✅ preset 只用 `none`/`sequential`；唯一调用方把它设在 qwen_image_21 上，而只有 Wan 有安装器，所以该脚本开关**现在就是坏的** | ~80 / ~150 |
| 10 | `vrl/nn/quantization/fp8.py` blockwise 分支、`passes.py:165-180` | A | 实测 0.19x、32 GB 上 OOM、标记为 TRAP 的 fp8 blockwise；0 preset | ~65 / ~60 |
| 11 | `resume_strict`（`config/schema.py:524` 穿到 `fsdp.py:385`、`trainer.py:2177-2290`、`offline/dpo.py:456`） | C/E | 非严格恢复穿过约 8 层、约 46 个分支；✅ 0 个 yaml 设置，记录的 run 全是 `true` | ~110 / ~120 |
| 12 | `vrl/rollouts/collector/batch_builder.py:61-69`、`types.py:17-20` 的 `REWARD_GROUP_ID_METADATA_KEY` | C | ✅ 三处写入、零处读取（唯一读者 codex judge 与 group_pick 都已删除） | ~15 / ~12 |
| 13 | `vrl/trajectory/views.py` 与 `batch_builder.py:80-131` 的 reward view 字典 | H | 三个 builder 产出同一个 spec（`output_ref=output`、range `unit`、无 refs）；`"tanh"` 值域从未产生 | ~110 / ~40 |
| 14 | `vrl/algorithms/trajectory.py:28-53` `AlgorithmAdapter` | H | trainer 总是传入 advantages，adapter 只是转调 `compute_loss`，却被当参数穿过两层 | ~50 / ~4 |
| 15 | `vrl/rollouts/stats.py:140-310` `StatsSink` Protocol + `MultiStatsSink` | B/D | 唯一组合方式固定在 `trainer.py:568`；折叠函数重查调用方已过滤的后缀 | ~50 / ~30 |
| 16 | 小的单实现接缝与无人设置的参数（逐条都复核过调用点） | B/E | `GenerationRankActor` Protocol（只有一个名字一致性测试在用）；`GenerationWeightSync` Protocol 与 `weight_sync=None` 分支；`DistributedExecutionPlanner` 单方法类；launcher 的 `init_ray`/`ray_init_kwargs`；`park_trainer_for_rollout` 穿 5 层；`RewardFunctionRuntime(None)` 无 reward 路径；空标记类 `CumemRewardFunction` 及三处子类检查；生成 `preflight()`（启动刚做过有界 RPC）；`WeightSyncer` ABC；reward 服务 `max_concurrency`（6 个 yaml 全是默认值 1，且单线程执行） | ~300 / ~250 |

## 3. 第二档：需要你拍板（语义或行为有取舍）

1. **NFT / V-GRPO 的"上一策略"前向恒等于当前前向的 detach。** ✅ 两个 base preset 都是 `weight_copy_decay: 0.0`，
   `optimizer_steps_per_batch` 默认 1，快照在每次 optimizer step 后整份刷新（`previous_policy.py:124-128`、`trainer.py:1555,1785`），
   所以计算 loss 时 previous == current；V-GRPO 的比值恒为 1，`clip_ratio`/`kl_coef` 分支永不触发。改用
   `forward_prediction.detach()` 可删快照、`after_optimizer_step`、decay 与首步不变量检查（~190 / ~150），**并且每个 replay step
   少一次 transformer forward**（NFT 3→2、V-GRPO 2→1）。代价：放弃 `ppo_epochs>1` 或 decay>0 的配置。
2. **首步漂移双门。** `precision_guard.py` 整个文件（额外的 no-grad 前向）与 replay-parity 门比较的是同一对 log-prob；唯一显式
   开它的 preset 两门都设 1e-6。建议 parity 门在开启 precision correction 时直接失败，删 guard（~330 / ~450）。
3. **生成 runtime 的 single-flight 保护**（`vrl/generation/ray/runtime.py:174-203,351-393,424-439,505-581`）：`asyncio.shield`、
   兄弟取消、57 行清理重试，保护的并发调用只有测试会构造（~170 / ~800）。
4. **流水线进度侧信道**（`actor_pool.py:401-544`、`executor.py:555-659`、`pipeline_protocol.py`）：worker 发布进度、driver 每秒轮询、
   按批重置 stall 截止。替代：多批 RPC 的超时 = `stall_timeout × 批数`（~330 / ~650）。代价：卡住的首批要 n 倍时间才暴露。
5. **健康监控本身。** 死 actor 下一次调用就会抛 `RayActorError`，挂住的会撞上已有的 RPC/stall 截止；监控只是早一个训练 step
   发现，结局相同（~200 / ~550）。今天刚简化过它，是否整删由你定。
6. **离线 reward 工具箱里需要盲标签的部分**：`reward/card.py`（从未写出过一张卡，`ready_for_training_key` 无人读）与
   `reward/calibration.py`（磁盘上没有任何拟合组合）。与已删的 audit 门同一个前提缺失（~755 / ~390）。
7. **`reward_collection_mode` 验收开关 + `reward_overlap_benchmark.py`**：`PER_GROUP_SERIAL` 对照臂只为已完成的测量存在；唯一设置它的
   yaml 要的是 collector 本来就会自动选的模式（~90 + 414 行脚本 / ~200）。
8. **其它防御层**：`vrl/models/checkpoint_identity.py` 每次解析都重查静态类元数据、哈希期间的文件签名复查（~200 / ~80）；
   `validate_driver_state` 遍历活模型树重查资源解析已决定的重叠（~110 / ~150）；`DatasetProvenance` 读取时重跑写入时已校验的规则
   （~350 / ~340，跨 `vrl/trainers/data/provenance.py`）；sana 报告每次读取都重哈希多 GB checkpoint（`5ee89674` 曾明确保留）；
   `tests/real_cover.py` 的 AST 扫描器可换成 conftest 里 ~50 行的收集期检查（测试侧 ~450）。

## 4. 第三档：研究在途，只标记不动

- Cosmos context-parallel 栈（`cosmos/context_parallel.py`、`strategy.py:954-1068`，故意不进 config 分派）：
  `docs/research/cosmos_cp_runtime_integration_20260913.md` 仍是 OPEN。
- TeaCache（实测约 0% 收益，但 `SPRINT_rl_safe_feature_cache_probe.md` 明确保留）、`ModulePrecisionTrace`（planned sprint 点名）、
  多 segment 轨迹通用性（刻意的 schema 选择）、28 个 family 别名中约 22 个无引用（旧 run 配置可能在用）。

## 5. 跨仓模式：同一规则多层校验

每条审计道都独立发现了同一个模式，这是根因，比逐条删更重要：

| 规则 | 重复位置 |
|---|---|
| reward 结果身份/顺序 | 最多 6 层：Ray actor、Ray driver、HTTP server、HTTP client、`evaluation.py`、`base.py:332` |
| 生成 fleet 身份 | 6 层：`actor_pool`、`engine`、`session`、`executor`、`weight_sync`、`batch_placement` |
| `sft_weight>0` 需要 `sft_latents` | `rules.py:105`、`online.py:474`、`trainer.py:528` + assert |
| reward 显存 parking 合法性 | `online.py:864`、`factory.py:218`、`registry.py:252`，参数相同 |
| Wan offload / topology | config、`wan_topology_from_build`、构造函数、pipeline 加载 |
| supervise 边界 | argparse、`RunSupervisor`、`HealthGateConfig`、`ContinuousHealthPolicy` |
| 轨迹合法性 | ✅ `TrajectoryReader.__post_init__` 每次构造都整批重校验，每个 microbatch 多次；batch 构造时已校验过（这条还有运行时开销） |

规则：**只在配置边界（schema / `rules.py`）或数据构造处校验一次**，下游按契约直接读。内部边界的 `isinstance`、
`getattr(..., default)` 与"再查一遍"一律删除。

## 6. reward 栈外移评估

**现状（✅ 均已复核）**
- 训练循环真正需要的契约很窄：`RewardRuntime` Protocol（`vrl/rewards/protocols.py:29`：preflight / activate / score / park_memory /
  shutdown）加 `RewardSample`、`RewardOutput`。只导入这两个模块不会加载 torch，也不会拉入 reward 以外的 vrl 子包。
- 进程外路径已存在：Ray reward actor（默认）与 HTTP reward service。72 个带 reward 的实验里 67 个走 Ray、5 个走 HTTP、0 个进程内；
  约 26 个把 reward 放在训练 GPU 上，靠 CuMem park/wake 分时。
- **依赖泄漏**：`vrl/rewards/deployment.py:19` 导入 `vrl.config.builders`，于是 `import vrl.rewards.deployment` 会拉进
  algorithms、generation、trainers、trajectory 全套；架构测试只查直接导入，漏掉了它。
- Ray reward actor 的 `runtime_env` 只设 `CUDA_VISIBLE_DEVICES`（`vrl/rewards/ray.py:243,261`），所以所有 reward 依赖都必须装进训练环境；
  `pyproject.toml` 里 `ar-vllm` 与 `reward` 声明冲突，vLLM 只能 `--no-deps` 手装，这是目前真正的环境痛点。

**三个选项**

| 选项 | 做什么 | 收益 | 代价与阻碍 |
|---|---|---|---|
| (a) 仓内独立包 | reward schema 类型搬进 `vrl/rewards`，切断 deployment 泄漏，reward 自带 extras | 依赖声明诚实，HTTP-only 时训练环境变小 | 3–5 天；默认 Ray 路径仍共用一个环境 |
| (b) 独立仓库，训练端只走 HTTP | 模型、服务、`ray.py`、`reward/`、10 个评测脚本全部迁出 | 独立发布、独立依赖、reward GPU 随意放 | 2–3 周并重新验收；丢掉 Ray object store 零拷贝媒体；HTTP 把视频转成 float32 base64（4 倍大小，默认 16 MB 上限）；丢掉单卡 CuMem 分时（26 个实验）；每次 run 要手动起服务 |
| (c) 不动 | — | 零成本 | 环境冲突继续手工绕 |

**建议**：不拆仓库。先做 §2 第 4 条（删 deployment 绑定，泄漏随之消失）并把 reward schema 类型搬进 `vrl/rewards`；若目标是依赖隔离，
下一步是让 Ray reward actor 可配置独立 Python 环境，而不是换成 HTTP。若"verifier"指的是离线工具箱 `reward/`，它已经单向依赖、
可以低成本迁出，但应先按 §3 第 6 条删掉从未跑过的 card / calibration 再迁。**主要阻碍**：单卡 5090 上的 CuMem park/wake 与
object store 零拷贝媒体，外置 HTTP 进程会同时放弃两者。

## 7. 执行顺序建议

1. 第一档 16 项，按包分组提交，每组跑受影响测试；预计 ~3.9k 生产 + ~3.7k 测试行。
2. §5 的重复校验按规则一次性收口（每条规则一个提交）。
3. 第二档逐条由用户裁决；第 1 条（NFT / V-GRPO）建议先在一个 NFT preset 上确认 loss 数值与改前逐位一致再删。
4. reward 外移只做 (a) 的前两步；拆仓库等出现多机 HTTP-only 拓扑的实际需求再议。
