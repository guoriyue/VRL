# SPRINT：编排核心按"线程与调度"模型收口，测试改为真实对象

状态：**进行中（2026-10-04 起）**。目标由 Mingfei 定义：核心编排里不再有为某个场景硬塞的
参数和能力位（"飞线"），结构像操作系统的线程与调度器一样可以一句话说清；测试不再靠
假对象，用真实的小规模组件。

## 0. 目标模型

把在线训练的每轮迭代看成三个"线程"在一个调度器下轮流拿 GPU：

| 角色 | 实体 | 状态 |
|---|---|---|
| trainer | driver 进程里的策略模型、优化器、EMA | resident / parked |
| rollout | Ray 生成 actor 队伍（`RayGenerationRuntime`） | not launched / active / parked |
| reward | Ray reward actor 或外部 HTTP 服务（`RewardFunctionRuntime`） | active / parked；HTTP 永远 resident |

三个事实决定一切调度行为，而且只在一个地方声明：

1. **放置**：`distributed.resources.<role>.devices`。不写就共 trainer 的卡；交集即共享。
   由它推出 `RayLifecyclePlan` 的三个 offload 位和四个 `park_<role>_for_<phase>` 边界位。
2. **调度模式**：`rollout_orchestration.schedule_mode`，strict（生成、训练、同步权重三段串行）
   或 continuous（producer 线程持续生成，consumer 按同版本凑整一批）。
3. **策略版本**：trainer 推一次权重版本加一；每个请求盖章；continuous 按版本窗口判新鲜。

角色自己不决定要不要 park；它们只实现 `activate` / `offload`(`park_memory`) / `shutdown`
这几个状态迁移，由 coordinator 的 `rollout_phase` 按计划排序调用。

## 1. 飞线清单与处置

"飞线"的判据：某个布尔或参数不是上面三个事实的直接读数，而是在中间层为了某个调用方
再算一遍、再传一遍，或者只有测试会设。

| # | 现状 | 处置 | 状态 |
|---|---|---|---|
| 1 | collector 每次打分时算 `require_reward_release`，以 `score(require_memory_release=)` 和 `park_memory(required=)` 传给 reward runtime，并用 `_reward_phase_started` 记相位 | reward runtime 在构造时拿到计划位（`parks_after_score`），自己记"是否欠一次 park"；`score(samples)` 与 `park_memory()` 不带参数 | 完成 ffb35ce8 |
| 1b | 第 1 行的中间方案里 reward 由构造参数 `parks_after_score` 自己决定打分后 park，而 rollout、trainer 的让出由调度器按放置计划决定，不对称 | collector 读放置计划的 `offload_reward`，打分后（无论成败）与阶段末调用 `park_memory()`；reward runtime 只记录自身是否占着设备内存；`memory_parking_required` 只取计划位（全 HTTP reward 在计划里本就不占 GPU） | 完成 ffb35ce8 |
| 2 | coordinator 用 `getattr(runtime, "supports_non_draining_weight_sync", False)` 探能力 | 先改为 family 能力 + launch contract 下发 + runtime 协议属性（f12ee2d0）；之后发现 family 能力位对全部 21 个家族都等于 `supports_policy_replay`，删除；决定改由 `TrainerConfig.from_root` 按调度模式与 `model.use_lora` 算一次（`versioned_weight_sync`），launch contract 与调度器都读它，runtime 不再暴露该属性 | 完成 f12ee2d0 |
| 3 | `OwnedCollection` 在三处用 `callable(getattr(item, "collect", None))` 鸭子判断 | 协议 `runtime_checkable`，`isinstance` | 完成 |
| 4 | `runtime_debug` 和 `group_size` 从 trainer 经 schedule、collector、request builder 逐层透传（63 处） | 它们是请求数据而非能力位，可接受；若要收口，合成一个每轮迭代的请求描述对象一次下发 | 待定 |
| 5 | `memory_parking_required` 在 registry 里被翻译成组件 kwargs 里的 `sleep_offload` 字符串 | 核对后判定可接受：计划位 → `ModelRewardFunction(sleep_offload=)` 构造参数 → `RewardRuntimeLaunchContract` 跨 Ray 进程边界，这是唯一会被序列化到 actor 的载体；YAML 侧已拒绝手写该键 | 不改 |
| 6 | `online.py` 1168 行同时承担 Ray 会话、资源计划、模型构建、训练循环、检查点 | 拆成"构建角色"和"驱动调度"两段，每段只读上面三个事实 | 第三批 |
| 7 | `_RayClusterSession` / `_OnlineRecipeLifecycle` 等私有生命周期类 | 随 #6 评估 | 第三批 |
| 8 | 策略版本在 coordinator 里从两个 provider（collector 的 runtime、syncer）探测，可能为 None，再 `require_int` 复验；staleness、consumer、continuous types 全带"版本缺失"分支，而生产从不产生 None（launch contract 发布 0，每次 push 发布 int） | 协议 `current_policy_version: int`；Ray runtime 从 contract 版本起（默认 0）；syncer 直读、只校验自己的 resume 覆盖；coordinator 只读 collector runtime 的版本；StalenessPolicy/PromptBatch/ScoredRollout 只收 int | 完成 f12ee2d0 |
| 9 | `RayRuntimeWeightSyncer.if_supported` 鸭子探测 runtime 有没有 `update_weights`（"进程内生成器可能没有"） | `update_weights` 进协议，recipe 无条件建 syncer | 完成 f12ee2d0 |
| 11 | trainer 用 `getattr` 探测算法与 evaluator 的可选能力：`prepare_update`（还从 evaluator 里取 scheduler/noise/sde 转交）、`after_optimizer_step`、`precision_correction`（trainer 构造后改写算法对象）、`replay_granularity`、`supports_deferred_replay_tensor_move`、`adv_clip_max` | 像调度器调用线程的固定入口：`Algorithm` 协议声明 `prepare_update(update_timesteps)`、`after_optimizer_step(global_step)`，两个根目标（GRPO、PreviousPolicy）给空实现，FlashGRPO/VGRPO 覆写；FlashGRPO 的 SDE 与精度校正由 factory 构造时注入（`trainer.precision_correction` 是唯一来源，trainer 读自己的配置，协议不声明该字段）；`Evaluator` 协议与 `ReplayEvaluatorBase` 声明两项 evaluator 属性 | 完成（见本批提交） |
| 12 | `kl_coef` / `sft_weight` 在 trainer、recipe、factory 里用 `getattr(config, ..., 0.0)` 探测；strategy 的 `context_parallel_groups`、SFT 用的 evaluator scheduler 同样靠探测 | 目标在 `Algorithm` 协议上声明两项权重（没有该项的目标声明 0）；`Strategy` 协议声明 `context_parallel_groups`（非分片后端为 None）；SFT 直接读 SDE evaluator 的 scheduler | 完成 c2a9fcf5 |
| 13 | 是否把 `supports_non_draining_weight_sync` 从 runtime 协议挪到 schedule 构造 | 起初判定不挪（TrainerConfig 不知道 family 能力与 LoRA）。family 能力位删除后只剩调度模式与 `use_lora` 两个配置事实，`TrainerConfig.from_root` 能直接算出 `versioned_weight_sync`，于是挪了：调度器读配置，不再询问 runtime | 完成 f12ee2d0 |
| 10 | 没有不经 Ray 的生成 runtime：测试只能造假 runtime；e2e 用私有 `_DirectExecutorGenerationRuntime` | 测试工具 `tests/generation/_in_process_runtime.py`：同一个 `GenerationWorkerCore`，actor 式单线程串行执行，slot 决策来自 launch contract（起初放在生产包里，因为生产代码从未使用而移到 tests） | 完成 f12ee2d0 |

已在本 sprint 之前完成的同类清理见 `placement-grammar-collapse` 记忆与提交
f114d3ba…06407954：gpu_pool、offload 覆盖位、reward.device 三态、scoring_is_nonblocking、
generation_overlap_safe、HTTP park/wake 租约、collector 的两个转发属性、runtime 的
`_transition` / `_force_shutdown` / `_installed_policy_version`。

## 2. 测试：从假对象到真实对象

现状（2026-10-04 统计）：51 个显式命名的 Fake/Stub 类，私有辅助类 rollouts 45、trainers 61、
generation 58、models 45，`monkeypatch.setattr` 在 scripts 116 处、generation 和 models 各 105 处。

真实替身的来源，仓库里已经有：

- `tests/scripts/eval/fixtures.py`：`write_tiny_sana_snapshot` / `write_tiny_cosmos25_snapshot` /
  `write_tiny_wan_snapshot` 写出能被真实 family 加载的小权重；`tiny_sana_online_config` 给出完整在线配置。
- CPU reward：`image_sharpness`（纯 numpy）、`ocr`（CPU-only 类）可作为真实 reward 组件。
- `tests/e2e/test_real_checkpoint_rl.py` 的 `_DirectExecutorGenerationRuntime`：不经 Ray 直接驱动真实
  family executor。把它升格为生产模块 `InProcessGenerationRuntime`（单进程调试也用得上），
  测试就有了真实的生成 runtime。
- `local_ray` fixture：真实本地 Ray 集群，已被 `tests/ray`、`tests/generation/ray` 使用。

替换顺序（每项一个提交，替换后删除对应假类）：

| 批 | 被替换的假对象 | 真实替身 | 状态 |
|---|---|---|---|
| A | `tests/rollouts/collector/test_runtime.py` 的 `_RewardRuntime` | 真 `RewardFunctionRuntime` 包一个最小 `RewardFunction` 子类（只记录调用） | 完成 ffb35ce8 |
| B | collector 测试的 `_Runtime`、`_RequestBuilder` | `InProcessGenerationRuntime` + 真 `GenerationRequestBuilder`（tiny SANA family） | 完成 7ba1c8ad |
| C | `PromptCollectionFake` 在 rollouts 下的子类（collector、strict、CP owner、prompt collection、continuous schedule/thread/contracts、owned collection） | 真 `RolloutCollector`（`real_collector` bench）+ 真 coordinator / strategy / syncer（`trainer_side`） | 完成 7ba1c8ad |
| D | trainer 测试的 `_EvaluatorAlgorithmFake` / `_Evaluator` / `_Algorithm` 以及 `tests/trainers/online` 里剩下的 16 个 `PromptCollectionFake` 子类 | `real_trainer`：recipe 同款接线（配置出的算法/evaluator、replay bundle、`SingleProcessStrategy`、真 syncer）配 tiny SANA | 完成 c2a9fcf5；保留：chunk 轨迹 evaluator 替身（CPU 上没有 chunk-autoregressive family）、Cosmos GPU worker、VDN（MiniMax-H3 加载坏了，见下） |
| E | `tests/scripts/test_online_lifecycle.py` 的 `_install_ray_side_fakes` 六件套 | 真本地 Ray + tiny SANA 真跑 `run_online_recipe`；Ray worker 通过 `worker_process_setup_hook` 装 tiny 管线 | 完成 c2a9fcf5 |
| F | `tests/generation/ray` 的 `_FakeSession` / `FakeRayActor` 族，以及包级 conftest 往 worker 注册表塞的假 family | 包级集群 worker 经 setup hook 服务 tiny SANA；`ray_sana_runtime` 走真 launcher + placement + `RayGenerationWorker`；故障靠保留真 actor 体的 worker 子类、`ray.kill`、真短 deadline、真坏 payload | 完成 f12ee2d0…（含 lease/sleep）；保留：CPU 上不可能出现的回包故障注入在真结果上、跨节点媒体探针、on-demand 拓扑用 `RolePlacement` 值（CPU 集群上 GPU 探针会崩） |

保留的注入点：`environ=`、超时参数、`monkeypatch` 掉的 CUDA 探测（`cuda_devices` fixture）。
它们让测试在无 GPU、无网络、一秒内跑完，不是业务替身。

真实栈的共用件（`tests/rollouts/collector/_helpers.py`，`tests/scripts/eval/fixtures.py`）：

- `tiny_sana_stack(monkeypatch, tmp_path, overrides=)`：解析 tiny SANA 在线配置，建 `InProcessGenerationRuntime`；
  `trainer_bundle()` 物化 replay bundle 作为 trainer 侧策略。`TinySanaPipeline` 现在带真的
  `encode_prompt` / `prepare_latents` / `image_processor`，走真实 denoise 执行器。
- `real_collector(...)`：`RolloutCollector.from_family` + 真 reward runtime；`versioned_slots=True`
  把 run 解析成 continuous LoRA，launch contract 的 `versioned_weight_sync` 为真。
- `trainer_side(bench)`：真 bundle + `SingleProcessStrategy` + 初始化标志；`coordinator_kwargs(bench)`
  就是 online recipe 给 coordinator 的那套接线。
- `Trace`：包真对象的方法记录顺序、线程、结果，并可排队一次性失败（含类型化异常）。这是唯一的
  "探针"，不是替身：被包的方法照常执行。
- 对时间敏感的用例先 `activate_generation_runtime()` 把模型加载排除在计时之外。
- `tests/e2e/test_real_checkpoint_rl.py` 的 `_DirectExecutorGenerationRuntime` 没有并入
  `InProcessGenerationRuntime`：它让生成直接跑在 trainer 自己的模块上（同一份权重、同一张卡），
  是另一种所有权形状，不是 worker 体；要并就得给进程内 runtime 加"借用 trainer 模块"的构造方式，
  等 GPU e2e 再评估。

D 批的难点：trainer 测试用 `nn.Linear(1, 1)` 策略和手造 batch 断言精确的损失 / EMA / 优化器步数值。
换成真 GRPO + 真 evaluator + tiny SANA bundle 后，这些数值断言要重新推导成结构断言（步数、
版本、形状、单调性），6.5k 行、16 个模块，需要单独一批评审。

## 3. 验收

- 核心编排模块（`rollouts/orchestration`、`rollouts/collector/core.py`、`generation/ray/runtime.py`、
  `rewards/runtime.py`）里不再有按调用方传入的能力布尔；每个布尔能指到 §0 三个事实之一。
- `tests/rollouts` 与 `tests/trainers/online` 不再定义 collector / reward runtime / generation
  runtime 的假类。（已达成；`PromptCollectionFake` 只剩 Cosmos GPU worker 与 VDN 两处，原因见 D 行。）
- G 批（`tests/ray` 的手写假 ray 模块）与 `continuous/test_thread.py` 两个 Ray quarantine 用例：已改为真实 Ray
  （b47b1432、7ba1c8ad）。保留：`test_dependencies.py` 两个需要三个不同 IP 节点的拓扑用例用
  `fake_ray(nodes)`，单机集群给不出这样的 `nodes()`。
- 每批改动后全量非 GPU 套件通过；按项提交，不推送。

## 4. 真实测试暴露的生产问题

- ~~SANA 不能在 DDP 下训练~~：已修，forward 经 `unwrap_compile_and_ddp` 读 `timestep_scale`，
  由单 rank gloo DDP 真实训练一步的测试覆盖。
- ~~MiniMax-H3 / VDN rollout 加载失败~~（已修 aa459b2a，测试改为加载真实 tiny modular snapshot）：`ModularPipeline.from_pretrained(path, workflow="t2va")`
  返回 `SequentialPipelineBlocks`，随后 `load_components(workflow="t2va")` 抛
  `NotImplementedError: workflows is not supported because _workflow_map is not set`。
  现有测试把 `from_pretrained` 换成了假的，所以没发现。
- **进程内 runtime 共享调用方的随机数流**：策略在调用方进程里惰性构建，LoRA 初始化消耗调用方的
  全局 torch RNG；Ray actor 用的是自己的进程。只影响测试与单进程调试。
- **CPU 主机上的 GPU 奖励**：在无卡主机上解析 aesthetic 预设，奖励设备解析为 `cuda:0` 且要求 parking；
  这种配置应在解析时报错，而不是在 actor 里失败。
- ~~已完成的 actor 调用可能被报告为已取消~~（已修 b47b1432：取消时先向 Ray 查询已完成的在途调用并交付其结果）：调用已在 actor 侧完成、但 driver 事件循环还没处理 Ray 的
  完成回调时，调用方取消会赢，dispatcher 抛 `RayOperationCancelled`，已完成的成功（可能是一次已提交的
  权重安装）或失败被丢弃。真实 dispatch 测试中 8/8 复现。取消时对在途 ref 做 `ray.wait(..., timeout=0)`
  可以让 actor 完成成为真正的线性化点。

## 5. 配置 / 运行状态 / 运行命令三分审查（2026-10-06）

判据：一个值什么时候能知道、知道之后会不会变。

- **配置**：启动前确定，解析时算一次，之后只读。
- **运行状态**：运行中会变，只由一个对象持有和修改。
- **运行命令**：某一时刻的动作，参数是这一刻的数据。

已修（每项一个提交，未推送）：

| 越线 | 处置 | 提交 |
|---|---|---|
| 单条 prompt 的 `request_overrides` 能改 `noise_level`/`sde_type`，回放仍用 run 级值 | denoise 参数只在 run 级配置里设置 | b8bb452a |
| 初始权重版本两个来源：契约写 0，resume 时 syncer 另取步数 | runtime 唯一持有版本号，syncer 取当前加一 | f12ee2d0 |
| 算法能力挂在运行对象上，经调度器传参只为抛配置错误；trust-region 检查分在工厂和训练器 | 两个能力进 `AlgorithmRequirements`（当时名为 `AlgorithmConfigContract`），检查移到 `TrainerConfig.from_root` 和 `rules.py` | c2a9fcf5 |
| 每次流式更新用 `getattr(..., False)` 探测 global_std | 直接读算法 config | c2a9fcf5 |
| `ray_launch_inputs` 事后补写 `base_weight_sync` | 在 `resolve_model_build` 里按 `use_lora` 定 | f12ee2d0 |
| `worker_config.sleep_offload` 绕过配置检查；构建器覆盖用户值 | 配置层拒绝两种写法，构建器只按拓扑注入 | ffb35ce8 |
| `_rollout_weights_initialized` 两个写入方（训练器字段 + 协调器回调） | 协调器持有，训练器只发 `require_weight_resync` 命令 | 7ba1c8ad |
| strict 调度每轮、关闭时重复检查训练状态能否停放 | 只在启动前检查一次 | 7ba1c8ad |
| rollout profiler 目录被无条件覆盖，worker 另有兜底 | 解析时按训练器规则定一次 | f12ee2d0 |
| resume 时改写已解析配置里的 LoRA 路径 | 改为 `rules.py` 交叉校验，删除改写 | 55777e41 |
| `PromptBatch.results` 三个写入方，controller 另存一份 batch 副本 | producer 唯一写入，`release_results` + `consumed` | 7ba1c8ad |
| recipe 与训练器判断是否需要 reference 模型的条件不同 | 都读算法的 `uses_evaluator` | c2a9fcf5 |
| 通用执行器的 `family`/`task` 由 worker 与 preview 各自补 | 由注册表放进启动契约 | f12ee2d0 |

审查后决定不改：

- **worker 对 `uses_pipeline_cpu_offload` 的探测**：它核实 hook 是否真的装上，换成读配置会丢掉这层核实。
- **`VRL_PROFILE`**：有文档的进程级信号，只在启动时写一次，外部 nsys 工作流直接使用。
- **recipe epoch 与 `TrainState.step`**：两者按构造相等。合并要改检查点格式，需要先决定旧检查点怎么兼容。
- **reward 停放状态的多层记录**：只在激活已失败的路径上不一致。
- **构建器里再跑一次 reward 停放校验**：这是 `MultiReward` 的构造不变量，预检脚本也走这条入口。

## 6. 未推送提交的复审（2026-10-07）

对 origin/main 之后的 80 个本地提交做了一轮复审（生成层、编排与 reward、训练器与配置、测试四路），
修复如下，每项一个提交：

| 问题 | 处置 |
|---|---|
| reward runtime 在 `activate()` 成功前就标记持有显存：激活失败时阶段末 park 抛"no active owner"，盖掉真实错误并跳过 trainer 恢复 | 标记改在函数激活成功之后；新增失败激活用例 |
| dispatcher 的"已完成调用优先于取消"修复在 `CancelledError` 处理器里多了一个 await，第二次取消会跳过归还 worker、取消在途 ref 和关闭池 | 该 await 吞掉重复取消；后续循环按"未完成即在途"处理 |
| 启动契约 `policy_version` 允许 None，而 runtime 协议只接受 int | 契约字段改为 `int = 0` |
| 无 trainer 的 root 下 rollout profiler 目录退化为当前目录 | 无 `output_dir` 时回退到 `outputs/torch_profiler` |
| 三个 Ray 模块用 1500 步请求加固定睡眠断言竞态；continuous 调度用例睡 120 ms 断言有进展 | `SlowGenerationWorker` 在 actor 内固定持有 1 秒；调度用例阻塞到 owner 报告新完成项 |
| 已删替身的残留：训练器 helper 里三个无调用者符号、collector helper 里只剩一个 GPU 模块用的 `PromptCollectionFake`、回显构造字段的用例、同路径参数矩阵、绕过 `bare_trainer` 的 SimpleNamespace 壳、打三处私有补丁的四 rank 行 | 删除或移到唯一使用者；矩阵改为基线加单轴翻转 |
| 三处重复的 tiny-SANA Ray 集群 fixture；共享 launcher 吞掉 runtime shutdown 异常 | 合成 `tiny_sana_ray_cluster`；shutdown 异常只在 runtime 已 TERMINATED 时跳过 |
| 文档：`TRAINING_GATES`、runtime 协议成员、sprint 表中已被后续提交推翻的行、几处引用已删名字的注释 | 更正 |

复审确认但不改：resume 后权重版本从 1 重新编号（没有消费者跨重启比较版本）；
`recompute_old_logprob` 的检查现在对所有目标生效（更严，预设无一命中）；Flash-GRPO 的
`prepare_update` 在流式模式下按每个 collection 而非整个 update 取均值（范围之外的既有行为，待定）。

## 7. 构造器与签名的全仓清扫（2026-10-07 / 10-08）

判据：一个参数如果能从同一调用里的另一个参数推出、只是转交给被构造的对象、唯一生产调用方永远传同一个值，
或者是对非可选类型参数的 None / isinstance 守卫，就删掉。按包分组，每组一个提交。

| 层 | 删除的参数 / 守卫 | 读取来源 |
|---|---|---|
| config / run | `ResolvedRun` 多个只转手的属性；`TrainerConfig.from_root(root)`、`ResourcePlan.from_root(root)` 的第二个"派生"参数；`normalize_precision` 的默认值 | 被调方自己从 root 推导 |
| generation | 执行器 `pipelined` 标志（有 finalizer 即流水）；`GenerationWorkerParking.sleep` 的设备参数；`EnginePlan.from_request` 外的 request 字段；`BundleLayout.rollout_gpus_per_engine`、`StagedBatchRefs.policy_version`、`measurement_scope`；`from_rank_results` 的可选 `expected_worker_ids`；若干 isinstance 守卫 | executor / 请求 / 契约本身 |
| generation | `merge_generation_batches` / `forward_plan` / `forward_plan_pipelined` / `sort_and_validate_batch_coverage` / finalizer `merge_request` 的 `sample_rows`；Ray 执行器不再把行随请求发给 finalizer | `request.sample_rows()`（确定性，contracts 用例钉住） |
| generation | `GenerationWorkerCore` 的 gatherer（rank 只产 batch，从不合并）和执行器"是否实现 forward_batch/merge"的可调用检查 | 合并在 driver 侧的注册表 gatherer |
| rollouts / rewards | `ContinuousRolloutConsumer`/`Producer`/`collect_iteration` 只转交的参数；`RolloutCollector` 构造改为 `from_family(entry, …)`；`RayRewardScorer` 三个常量化的超时 / CPU 参数；`RewardFunctionRuntime(reward_function)`；协调器的 `weights_initialized`；预检改调 `reward.validate_parking()` | 对象自己持有 |
| trainer / strategy / data | 流式更新 helper 的 `batch_plan`；`OnlineTrainer` 的 `ref_model`（KL 项或契约要求时策略即参考）与 `device`（恒等于 `strategy.context.device`）；回放选择 helper 的 config 字段参数；`init_training_process_group` 的 backend；策略各自建 collectives；FSDP mesh 词表由 schema 持有；`ContextParallelStrategy` 读 `context.cp_size`；`select_trainable_state`、`ImageCaptionPromptDataset.from_section`、`DatasetFileReport.artifact_count` | trainer config / strategy context / 模块自身 |
| scripts | `AlgorithmEvaluatorPair.from_built` 的 `family_entry`（`built.family`）；`OnlineRecipeRun.initialize_metrics` 的 `training_context` / `output_dir`；`_load_sft_latents_from_config(built, *, sft_weight)` | run 持有的 strategy / trainer config |
| checkpointing | `restore_training_checkpoint` / `restore_model_checkpoint` 两个 None 容忍的转发包装；所有生产调用方本来就先判空或手持已加载的检查点 | `TrainingCheckpoint.restore_training` / `.restore_model` |
| rewards | `HttpRewardScorer` 的双模构造（URL + 关键字，或 `RewardInferenceConfig` 外加"不许同时传关键字"的守卫） | 单一构造器 + `from_config`（schema 已校验 origin 和 `expected_model`） |
| data | `PromptExample.references`（进 reward metadata 但没有任何 reward 读它） | 删除 |

审查后保留：

- **`OnlineTrainer.weight_syncer` / `sync_state_getter`**：syncer 在所有真实调用方都是 `RayRuntimeWeightSyncer(collector.generation_runtime)`，
  但 getter 需要 `bundle`（`strategy.export_rollout_state(bundle)`），训练器只持有 `model`，无法派生；两者成对出现是协议要求。
- **`RuntimeBundle.raw_handle`**：离线 DPO 脚本读它取 pipeline。
- **`RayGenerationLauncher`**：无状态类，改成模块函数属于命名层面的整理，不计入本轮。
- **`TrainingCheckpoint.restore_training` 里对 `trainer._strategy` 的 getattr**：离线 DPO 训练器没有策略对象。
- **`restore_model_checkpoint` 的 None 容忍在 Qwen probe 里确实被用到**：改为调用方判空。

## 8. 算法层：声明、构造与求值器基类（2026-10-08）

| 问题 | 处置 |
|---|---|
| `AlgorithmConfigContract` 名字不说明是什么；`tolerates_off_policy_staleness` 在所有在线目标上都等于 `not requires_previous_policy`（行为策略是当前权重的目标才吃不下 staleness） | 不再有单独的类：五个事实是每个**目标类**的 ClassVar（`needs_sde_rollout`、`sft_source`、`requires_previous_policy`、`requires_reference_policy`、`requires_active_trust_region`，离线 DPO 的 `DiffusionDPO` 多一个 `consumed_sections`），和已有的 `uses_evaluator` 并排，词汇在 `Algorithm` 基类上写一次；配置层经 `vrl/config/algorithm.py` 的 kind 表拿到目标类（同一张表也给 factory 用，factory 自己的 kind→类字典删除）；删掉 `tolerates_off_policy_staleness`，continuous 检查直接读 `requires_previous_policy` |
| GRPO / FlowDPPO / GRPOGuard / DiffusionNFT 各自实现一遍"绑定 advantage estimator"和两个 advantage 方法；FlowDPPO / GRPOGuard 复制 GRPO 的构造器而不是继承；`_initialize_*` 两个只在构造器里用的 helper | `GroupRelativeObjective`（`advantages.py`）一次持有归一化配置、reward 权重与两个 advantage 方法（原 `GroupAdvantageEstimator` 并入其中），GRPO / NFT / V-GRPO 继承；构造器收 reward 的 `component_weights` 而不是调用方先 build 好的 estimator；FlowDPPO / GRPOGuard 不再写构造器 |
| `kl_coef = 0.0` / `sft_weight = 0.0` 在 FlowDPPO 和 GRPOGuard 各写一遍 | 两个权重只在真有这一项的 config 上作为 knob 存在（`kl_coef`：GRPO 族、NFT；`sft_weight`：GRPO、DPO；Flash 继承后钉 0）。trainer 不再读 `kl_coef`：它要的是"要不要参考策略"，GRPO 把 `requires_reference_policy` 作为 property（`kl_coef > 0`）给出，NFT 是常量 True；`sft_weight` 在 `Algorithm` 基类默认 0，GRPO 用 property 从 config 读 |
| `VGRPOConfig` 重复 `GroupAdvantageConfig` 的 `eps / adv_clip_max / global_std`，V-GRPO 自己再调一次 `group_relative_advantages` | `VGRPOConfig(GroupAdvantageConfig)`，默认 `advantage_combine="weighted_sum_raw"`（与原行为逐位相同），advantage 走同一条 estimator 路径 |
| `config: ... | None = None` 让基类替子类选默认配置，子类因此各写构造器 | 所有目标的 `config` 必填；factory 本来就总是传 |
| `Evaluator` Protocol 与 `ReplayEvaluatorBase` ABC 各声明一遍 `replay_granularity` / `supports_deferred_replay_tensor_move`；trainer 再在运行时校验字符串取值 | 只剩 ABC `Evaluator`，两个属性是 `ClassVar[Literal[...]]`；删掉 trainer 的字符串校验和对应用例 |
| GRPO / GRPOGuard 的 loss 里 `keep is None` 分支：每个 signal 都带 mask，`combine_keep_masks` 永远返回张量 | 删掉死分支；`_broadcast_sample_values` 去掉对非张量的 getattr 探测 |
| trainer 判断是否需要参考策略：`(uses_evaluator and kl_coef > 0) or requires_reference_policy` | 只读 `algorithm.requires_reference_policy`：GRPO 作为 property 回答（`config.kl_coef > 0`），NFT 常量 True，其余继承默认 False |

一个基类，先分在线 / 离线（2026-10-09 定稿）：`Algorithm`（`base.py`）只持有七个事实 ClassVar（`uses_evaluator`、`needs_sde_rollout`、`sft_source` 必填，`requires_*` / `consumed_sections` 有默认）、config、`sft_weight` 默认 0。第一层分叉和 `trainers/online` / `trainers/offline` 一致：`OnlineAlgorithm` 与 `GroupAdvantageConfig` 配对，持有组相对 advantage 的两个方法、两个生命周期空实现、抽象 `compute_loss(AlgorithmInput)`；`OfflineAlgorithm` 声明无 rollout、无 evaluator，loss 签名归具体算法。`DiffusionDPO(OfflineAlgorithm)` 与 `DiffusionDPOConfig` 配对，`compute_loss(model_pred, ref_pred, target)` 和 `sft_loss` 是它的方法（原 `diffusion_dpo_loss` / `diffusion_sft_loss` 自由函数），离线 trainer 持有 `algorithm` 并调用它，`OfflineDPOTrainerConfig` 不再复制 `beta` / `sft_weight`。`GroupRelativeObjective`、`PreviousPolicyObjective`、`ComponentAdvantageAlgorithm`、`AlgorithmRequirements`、`ClippedPolicyConfig` 五个名字删除（最后一个只有 `GRPOConfig` 一个子类，`clip_ratio` / `kl_coef` 直接放进 `GRPOConfig`）；词汇只剩三种：`*Config` 是 YAML 一节解析成的数据（解析一次、只读），对象（`Algorithm`、trainer、model）是行为，`typing.Protocol` 是行为对象要满足的接口，不再有第四个名词。

```
Algorithm                                  （七个事实 + config + sft_weight）
├─ OnlineAlgorithm          ↔ GroupAdvantageConfig
│  ├─ ImportanceRatioAlgorithm ↔ ImportanceRatioConfig
│  │  ├─ GRPO ↔ GRPOConfig，FlashGRPO ↔ FlashGRPOConfig
│  │  ├─ FlowDPPO ↔ FlowDPPOConfig
│  │  └─ GRPOGuard ↔ GRPOGuardConfig
│  └─ ForwardProcessAlgorithm ↔ ForwardProcessConfig
│     ├─ VGRPO ↔ VGRPOConfig
│     └─ DiffusionNFT ↔ DiffusionNFTConfig
└─ OfflineAlgorithm
   └─ DiffusionDPO ↔ DiffusionDPOConfig
```

两条在线分支各一层继承：重要性比分支 `ImportanceRatioAlgorithm` 持有三条共同事实和 TIS / RS / masked-mean 脚手架（`precision_keeps`、`masked_mean`、`masked_fraction`），GRPO / FlowDPPO / GRPOGuard 的 loss 只写各自的比与 mask；前向过程分支 `ForwardProcessAlgorithm`（`ForwardProcessConfig` 只放 `eps=1e-8`）持有 `compute_loss` 转发、`renoised`、`noise` 覆盖点、`flow_time`、`normalized_mse`，V-GRPO 只覆盖 `noise`（组内共享噪声），NFT 只声明自己独有的两条事实。`forward_process.py` 里没有独立函数；算法包里剩下的模块级函数只有三个 loss 共用的 `_broadcast_sample_values`。加噪仍读轨迹记录的 timestep 而不是 scheduler：`vrl/math/denoise` 的 `diffusion_pretraining_pair` 要求 trainer 进程里的 scheduler 状态等于 rollout 网格，这两个算法不带 SDE evaluator，轨迹是唯一保证一致的状态。

算法的构造器只收自己的 config（2026-10-08 续）：`component_weights`（来自 `reward.components`）、`precision_correction`（来自 `trainer.precision_correction`）和 Flash-GRPO 的 `noise_level` / `sde_type`（来自 rollout denoise 选项）原来由 factory 在运行时一根根接进算法；按"配置解析一次、由读它的对象持有"的规则，`build_configs._bridge_online_algorithm` 现在把它们解析进算法 config 的 `init=False` 字段（不是 YAML 键，`root` 保持未桥接的公开副本），GRPO / FlowDPPO / GRPOGuard / NFT / V-GRPO 都是 `Algorithm(config)`，FlashGRPO 多一个运行时对象 `scheduler`。

MAGI-1 适配器（`magi_1/model.py`）的六个单调用方 helper——`_preflight_local_installation` / `_probe_runtime_environment` / `_validate_single_process_config` / `_resolve_runtime_paths` / `_resolve_weight_components` / `_unused_local_port`——并回各自唯一的调用方：安装检查 + 读官方 JSON 是 `Magi1SubprocessConfig.load_official_config()`，权重路径解析在 `from_build` 里，运行时路径解析在模型构造器里，端口在环境构造里。留下的模块函数都有两个以上调用方或是公开入口。

`precision_correction` 保留：它不是修精度损失，而是在 rollout 与 replay 精度不一致（如 FP8 rollout、fp32 replay）时把 `exp(replay - rollout)` 这个重要性权重截断（TIS）并整条拒绝越界样本（RS），让少数漂移样本不能主导梯度；`TrainerConfig.from_root` 在精度分裂时自动开启。它只对 GRPO 族（有重要性比）有意义，所以只在 `GRPO` 构造器上出现一次。

## 9. 单调用方私有函数清扫（2026-10-08 / 10-09）

判据（MAGI-1 那次定下的）：模块级私有函数只有一个生产调用方就并回调用方；只在四种情况下保留——(a) 按名字注册的回调（argparse `func=`、注册表钩子、Triton 内核、torch SAC policy），(b) 递归 / 线程或进程目标，(c) 测试直接钉它自己的数学而调用方需要真模型或完整 recipe，(d) 长 CLI `main` 的阶段，并回会超过 ~120 行。并回时顺手删掉调用方已经做过的检查和对非可选类型参数的守卫。扫描脚本：`single_callers.py`（私有顶格 `def _x(`，统计 vrl/ 内除定义外的引用次数）。

| 组 | 提交 | 并回 | 保留 |
|---|---|---|---|
| rewards | fffdd5e7a | 19 | 10 |
| scripts：common / data / denoise / rewards / generation / lint / supervise | 2ca7e37ad | 21 | 21 |
| scripts：eval / perf | bde3c772a | 46 | 42 |
| config / generation / models / ray / trajectory / utils / trainers | 3614bbd56 | 38 | 20 |

清扫前 218 个候选，清扫后重扫剩 92 个，全部是上面四类保留项（每个提交说明里列了名字和理由）。顺带修到的真问题：`vrl/scripts/data/bootstrap.py` 的 `--run` 切片比错一位，setup 命令从未真正执行；`vrl/rewards/service/wire.py` 解码里一个局部变量遮住了请求体；`video_reward_suite` 一段不可达的 optional-model 循环。

## 10. 独立的 resolve / validate / build 函数并回其所有者（2026-10-09）

单调用方 helper 并回后，仍有一批公开的 `resolve_* / validate_* / build_*` 自由函数：它们产出或检查的东西本来就有一个对象。判据：**函数的结果就是某个类的实例 → 做成该类的构造方法；函数读的只是某个对象的字段 → 做成该对象的方法**。

| 原函数 | 现在 |
|---|---|
| `bootstrap.resolve_experiment_dataset_plan` → dict | `ExperimentDataFiles.from_mapping` + `DataFile`（角色、路径、在不在、行数、该有几行、`fetch_command`），`to_dict` / `run_missing` |
| `build_configs(cfg)` | `BuiltConfigs.from_cfg(cfg)`；`vrl.config` 包不再转发 |
| `resolve_online_run(cfg)` / `resolve_model(...)` | `ResolvedOnlineRun.from_cfg` / `ResolvedModel.from_root` |
| `resolve_data_loader` | 并入 `DataConfig` 的校验器 |
| `resolve_train_target` | 并入 `train.run_config` |
| `resolve_gradient_checkpointing_mode(root)` | `ActorSection.gradient_checkpointing_mode` property |
| `validate_reward_memory_parking_components(...)` | `RewardRuntimeConfig.validate_memory_parking(device)` |
| `resolve_required_reference_images_(examples, …)`（原地改列表） | `PromptExample.with_required_reference_images`，与其它 `with_*` 同型 |
| `validate_checkpoint_meta_compatibility(meta, …)` | `CheckpointTarget.validate_compatibility`，与 `TrainingCheckpoint.validate_compatibility` 同型 |
| `validate_rollout_schedule_topology(config, resources)` | `ResourcePlan.validate_schedule_mode(schedule_mode)`；拓扑用例改为真实 plan |

保留并说明理由：
- 按 kind 分发到多个实现类的工厂：`build_strategy`、`build_reward_scorer`、`build_optimizer`（产出不属于任何一个类）。
- 注册表按字符串路径引用的钩子：各 family 的 `build_*_replay_runtime_bundle`、`normalize_*_model_build`。
- 对 state-dict 树工作的一组函数：`capture_rng_state` / `validate_rng_state` / `restore_rng_state`。
- `resolve_checkpoint_model_identity(build)`：产出的是写进 checkpoint 元数据的可序列化 dict，算法独立成模块；做成 `ModelBuild` 方法只能是转发。
- `require_training_config(cfg)`：parse + 跨节规则 + 精度策略三件事的入口；`BuiltConfigs.from_cfg` 调它。
- 多调用方的纯函数：`resolve_model_root`、`resolve_metadata_path`、`normalize_precision`、`normalize_fsdp_parameter_dtype`、`resolve_echo_video_dimensions`。

## 11. eval 脚本共享骨架（2026-10-09）

内联单调用方函数不会减少代码；评估脚本之间真正的重复是三份同构的工具
（Wan 2.1 HPSv3、Wan 2.2 Kling、Cosmos Kling）各自复制了同一套数据类和循环。
本轮把这套骨架收进 `vrl/scripts/eval/_checkpoint_comparison.py`：

- `GeneratedVideo.from_file` / `artifact_id`：一张网格格子的记录（三个工具共用，
  cosmos 原来自带一份 `path: Path` 的变体和 `_artifact_id`）。
- `load_wan_run_config`、`reward_worker_config`：训练 run 的配置和奖励 worker
  配置（原来 5 处手写 `RewardInferenceArtifact.from_cli_values` / parking pop）。
- `CheckpointComparison.from_args` + `generate_arm`：参数校验、模型构建、
  checkpoint 兼容性检查、提示词/采样协议、按格子生成（两份 Wan 工具的
  `generate_grid` 前 60 行和 `_generate_arm` 逐字相同）。
- `score_rows` / `score_video`：按 sha 重校验、可选 scores.jsonl 缓存、
  惰性建模的打分循环（Wan 2.2 的版本带缓存，HPSv3 的不带，其余相同）。
- `score_report.summarize_prompt_level`：以 prompt 为单位的配对统计，从
  Wan 2.2 的 `score_grid` 内部搬出。
- `vrl.utils.json_files.write_csv`：三份手写 `csv.DictWriter` 收口；
  `write_scores` 改走 `write_jsonl` + `write_csv`；cosmos 的
  `_summarize_scores` 改走 `score_statistics.distribution`。

净变化：工具与测试 -504 / +185 行，另加 346 行共享模块（原来分散在三份工具里
的同一段代码只剩一份）。

## 13. 路径解析按“相对于哪个根”收口（2026-10-10）

同一件事写了几十遍的三类“解析”，各自只留一处：

| 根 | 以前 | 现在 |
|---|---|---|
| 用户写的路径（命令行参数、配置值） | 47 处 `Path(x).expanduser().resolve()` 散在 eval / data / trainer 脚本里，5 处 `args.data_root.expanduser().resolve() if args.data_root else default_data_root()` | `vrl.utils.artifacts.user_path(x)`；data root 一律 `coerce_data_root(args.data_root)` |
| Hugging Face Hub | 12 处直接调 `snapshot_download` / `hf_hub_download`（四个 reward judge、Kling 自己一份、echo 两份、magi、causvid 两份、editreward、SANA eval 三份、danbooru、videophy、lerobot、wd_tagger） | `vrl.utils.hf_hub.HuggingFaceRepoRevision`（从 `rewards/models/hub.py` 搬出）：`parse`、`download(local_files_only, allow_patterns)`、`download_file(filename, repo_type, cache_dir)`，下载失败的恢复提示在这一处 |
| reward judge 的 checkpoint（`worker_config` 的 `model_path` 或 `reward_model_name`） | 三个 judge 调 `resolve_model_root`，Kling 自己再包一层同名函数 | 一个函数 `hub.worker_model_root(worker_config, default_model=, family=)`，四个 judge 在构造器里直接调；Kling 的 repo 限制留在构造器里，目录布局检查是自己的 `_checkpoint_root(root)`，不再有包一层的同名函数，也不引入按“权重从哪来”分类的基类 |
| 数据根下的 manifest 文件 | `RootedPaths`（多根沙箱）和 `resolve_artifact_path`（数据根策略）两层；`with_resolved_references` / `with_resolved_artifacts` 两个只差字段集的方法；`with_required_reference_images` 自己再解析一遍；editreward / local_edit / dino / 报告各自再调一遍 | 照 vLLM 的 `MediaConnector` 收成一个对象：`RootedPaths(roots, absolute=)` 在构造时定策略（`"inside"` 是 reward 服务的沙箱，`"reject"` / `"anywhere"` 是 manifest 的两种），`resolve(raw, strict=)` 是唯一的分发点；`RootedPaths.data_root(value, allow_absolute=)` 从配置或 `VRL_DATA_ROOT` 建；`PromptExample.with_resolved_references(roots)` 只解析两个参考字段，干净目标按合约不解析，要读目标文件的工具自己 `roots.resolve` 那一个路径；reward 服务、editreward、local_edit、dino、报告、encode_targets、两个 eval 都拿同一个对象，没有第二套解析。`default_data_root` / `coerce_data_root` 两个自由函数也并进 `RootedPaths.data_root`（环境变量和本地默认在它里面），要 Path 的调用方读 `.root` |

没收的：相对于 run 目录或源码目录的路径（`sana_checkpoint_compare` 的 checkpoint、magi 的 `python_executable`、causvid 的 `third_party` 源码）各自有自己的根；`rewards/service`、`checkpoint_identity` 里保留原样的 `expanduser()`，因为那里的字符串进入身份或服务协议，不能顺手 `resolve()`。

## 14. 全仓库 resolve / normalize / require / load 自由函数普查（2026-10-10）

按“解析的是什么”分组后，真正重复的只有三组，其余各自解析不同的东西：

| 组 | 处置 |
|---|---|
| 训练 run 的 resolved config（eval 入口） | `load_resolved_run_config` 是唯一读者；image eval 和 SANA 曲线原来各自再 `load_config` 一遍，改为调它 |
| checkpoint 目标（eval 入口） | SANA 曲线自己的 `CheckpointTarget` dataclass 和“扫目录、读 meta、核对 epoch”循环删除，改用 `trainers.checkpointing.CheckpointTarget.load`（baseline 是 `.base("baseline")`，持久化曲线里仍标 epoch -1）；image eval 的 `load_target` 也走 `CheckpointTarget.from_cli_values`：两边标签规则的差别（Wan 工具跨 run 比较时 `checkpoint-final` 取 run 名，image eval 在同一个 run 里取叶名）由一个上下文参数 `run_dir` 表达，不是模式开关：`checkpoint-final` 只有在被评测的 run 之外时才取 run 名；LoRA 权重文件指向其所在目录的规则也只写在解析器里 |
| dtype | `resolve_torch_dtype` 是唯一原语；`normalize_precision` / `_normalize_plain_dtype` 是配置 token 层，`resolve_eval_dtype` 是 eval 的 `auto` 语义，`normalize_fsdp_parameter_dtype` 是对模块参数的改写，Kling 的 `_torch_dtype` 多处理 `auto` 和 None；都是在原语之上各做一件事，没有第二个原语 |

逐个看过、各自解析不同对象的：`ray/resources` 的四个 `_resolve_*_devices`（一个构造器的阶段，§10 决定保留）、causvid / magi 的源码目录与权重路径、`resolve_checkpoint_model_identity`、`resolve_eval_sampling`、`resolve_eval_device`、danbooru 的 `resolve_metadata_path`、`require_*` 一族（各自检查不同的事）、`load_*` 一族（各自读不同的文件；perf 的 `_load_prompts` 读纯文本行，和 `load_prompt_dataset_index` 的 `.txt` 约定不同）。

## 15. 按形状找包装函数，不按名字（2026-10-10）

按 `resolve_` / `validate_` 这类前缀搜只能找到起了那种名字的函数。包装函数的定义是形状：函数体只有一次对别的函数的调用，外加最多一行参数整理。用 AST 扫 `vrl/` 得到 209 个这种形状的函数，其中绝大多数不是包装：协议实现（各 family 的 `forward_step` / `restore_eval_state` / `encode_prompt_for_batch` / `build_branch`、`Strategy` 的导出方法）、框架回调（aiohttp 路由、pydantic 校验器、Ray actor 方法）、多模块共用的一行原语（`user_path`、`kv`、`to_uint8`、`read_json`）、以及测试钉住数学的函数（`group_relative_advantages`）。剩下的单调用方或双调用方的转发层这轮折掉：

| 原来 | 现在 |
|---|---|
| `image_checkpoint_eval.load_target` | `discover_targets` 自己调 `CheckpointTarget.from_cli_values` + `.load`，`uses_lora` 在一处循环里检查 |
| `ForwardProcessAlgorithm.compute_loss` → `compute_batch_timestep_loss` 两个入口 | 只剩 `compute_loss(inputs)`，V-GRPO / NFT 直接实现它，测试构造 `AlgorithmInput` |
| wan22 `_probe_score`、robotics `_json_sha256` / `_normalized_prompt`、curve verdict `_mean` | 调用处直接写 |
| HPSv3 `_frames`、服务端 `_overloaded_error`、causvid `_autocast_context`、loader `skeleton_from_config`、batch builder `_group_ids`、request builder `_apply_input_defaults`、优化器 `_make_master`、schema 的 `generation_request_rollout_fields` | 并进唯一的调用方 |
| `OnlineTrainer.state_dict` / `checkpoint_state_dict` → `_state_dict(checkpoint_primary_only=)` 三层 | `state_dict(*, checkpoint_primary_only=False)` 持有正文；`checkpoint_state_dict` 留着，是 checkpoint writer 按名字找的协议入口 |
| Cosmos Predict2 `postprocess_branch` → `_postprocess_branch(**拆开的字段)` | 一个方法 |
| Wan T2V / I2V 各一个 `forward_step` 转发到 `_forward_wan_backbone(extra_builder=)` | `forward_step` 本身是 T2V 基类上的那段正文，I2V 只覆盖 `_backbone_extra`（图像条件）；MRO 测试改为钉 `_backbone_extra` 的归属 |
| danbooru `split_safety_prompt_rows` → `split_rows_proportionally`（单调用方的泛型函数） | 正文并进唯一调用方 |

折错又恢复的一处：`DenoiseModelBase.forward` 不是 `forward_step` 的重复名字，它是 `nn.Module` 的入口，module hook 绕着它触发；留下并写明。
名字前缀本身不是判据。常见的几类：形状变换（resolve / normalize / coerce / parse）、门禁（validate / verify / check / ensure / require）、构造（build / make / create / prepare）、取得（get / fetch / load / read）、转换（to_ / from_ / as_）。每一类里都既有真正的多调用方原语，也有一层薄转发；区分的是调用方数量和它有没有自己的逻辑，不是动词。

## 16. 函数体完全相同的重复（2026-10-10）

按 AST 比对函数体（忽略 docstring，短于 120 字符的一行函数不算），vrl/ 里有 21 组函数体完全相同。处置：

| 组 | 处置 |
|---|---|
| `GRPO.sft_weight` / `DiffusionDPO.sft_weight` 两个 property 同体 | 一个 `Algorithm.sft_weight`，按 `sft_source` 事实决定读不读 config |
| `DenoiseBatchResult` / `DenoiseLoopResult` / `DenoiseTrajectoryBuffers` 各一份 `observations` / `actions` 视图 | 一个数据基类 `DenoisePath`（`latents` 加两个视图），三者继承 |
| qwen_image / qwen_image_21 解码里一样的 `_transform` 闭包 | `common.latent_decode.denormalize_video_vae_latents` 一处 |
| SingleProcess / DDP 两份 `backward`、`clip_grad_norm`；CP / DDP 两份"拒绝 parking"的 `park_training_state` / `restore_training_state` | 都放在 `_UnshardedStateStrategy` 上，SingleProcess 的 `TrainerParking` 在 MRO 更前面所以仍走真正的 parking |
| cosmos3 / minimax_h3 executor 各一份只为吞掉 `samples_per_generation_batch` 的构造器 | 生产代码从不传这个参数，两份删除；测试里那一处调用改掉 |
| lerobot 两个函数里各定义一遍同样的 `_dl` 闭包 | 一个模块函数，用 `functools.partial` 绑定仓库和缓存目录传给按版本的迭代器（它要的是回调） |
| `MultiReward.score` | 没有调用方，删除 |

试过又退回来的：
- 把 lumina2 / mochi / sana 逐行相同的 `build_branch` 上提为 `DenoiseBackboneRunnerBase` 的默认实现：这正是 2026-09-17 合并后又撤回的 `EncoderAttentionMaskRunnerBase`，违反 `SPRINT_functional_duplicates.md` 定下的边界（基础设施去重，model glue 不合并：每个 family 的 `encode_prompt` / `prepare_sampling` / `forward_step` / `build_branch` / 轨迹导出与恢复各留一份，即使逐行相同），退回。
- 把两个 denoise executor 基类相同的 `__init__(model, *, gatherer)` 上提到 `BatchExecutorBase`：三个测试替身证明那一层是故意没有 model 的（它只管分批和 gather），退回。
- 把 flux / qwen_image 相同的 `packed_token_count` 做成基类默认：`test_dynamic_shift_replay_without_a_token_count_fails_loud` 钉的是"动态 shift 的 family 必须自己声明打包方式，否则 replay 网格无声出错"这条契约，退回。

留下的同体函数（四个 masked-prompt family 的 `export_replay_tensors`、lumina2 / sana 的 `encode_prompt` 这类同体 model glue 按上面的边界一律保留，近似重复扫描会一直报它们，结论是“model glue, kept”）：`reward_media` 两个结果类的别名属性、geneval / wd_tagger 的 `__call__`、各 family 相同的协议方法（`encode_prompt_for_batch` 等）、`policy_cores` / `trainable_modules`（同体但概念不同，docstring 写明）、`requires_memory_parking` / `required`（两个协议）、lumina2 / mochi 的 `postprocess_branch`（两家都预测反向时间速度，各自声明）。

## 17. 产出某个类实例的自由函数并回该类（2026-10-10）

| 原函数 | 现在 |
|---|---|
| `create_context_parallel_groups(cp_size)` | `ContextParallelGroups.create(cp_size)` |
| `create_context_parallel_peer_group(context)` | `ContextParallelPeerGroup.create(context)` |

判断标准（按 C++ 的 static 成员函数）：函数并进类 `T`，要么它是具名构造函数——返回恰好一个 `T`，且普通构造表达不了（集体通信、字段由输入推导）；要么它要用 `T` 的不变量或内部表示。返回容器、`None`，或装的是别的模块的规则，就留在模块里当函数：Python 的模块就是命名空间。上表两项是具名构造函数。`build_adapter_exports`（返回 `dict | None`，决定的是检查点目录布局）和 `build_reward_model_definitions`（返回列表，写死的是这份 SANA 报告用哪两个 reward）按这条标准不并，留在 checkpointing 和报告模块里。

留下的 build_* / validate_*（80 个里的其余）：19 个脚本的 `build_parser` 是 CLI 入口约定；`build_*_replay_runtime_bundle` 是注册表按字符串引用的 family 钩子；`build_strategy` / `build_reward_scorer` / `build_optimizer` 按 kind 分发到多个类；`build_fsdp_mesh` / `build_context_parallel_mesh` 产出 torch 的 DeviceMesh；其余产出 dict / list / str，没有对应的类；`validate_*` 各检查一件自己的事，读的是文件或张量而不是某个对象的字段。两处同一角色两个名字的：`encode_targets` 和 `full_sequence_probe` 叫 `_build_arg_parser`，其它 19 个脚本叫 `build_parser`；改名要你定。

## 18. 构造器后缀按输入类型（2026-10-10）

把所有 `from_*` 按第一个参数的类型分组，同一种输入只留一个后缀：

| 输入 | 后缀 | 这次改的 |
|---|---|---|
| `DictConfig` | `from_cfg` | `RewardRuntimeConfig.from_cfg` 原来同时收 `DictConfig` 和 `RewardConfig`、函数体里按类型分支 → `from_section(reward: RewardConfig)`；手里只有 `DictConfig` 的 eval 脚本先 `RewardConfig.from_cfg(cfg)` |
| `RootConfig` | `from_root` | `DenoiseRequestOptions.from_sections(rollout, sampling)` → `from_root(root)` |
| 已解析的 `*Config` 对象 | `from_section`（`PrecisionPolicy` 已有 61 处） | `ContinuousRolloutSchedule` / `ImageCaptionPromptDataset` / `HttpRewardScorer` / `ProfilerActivitySelection` 的 `from_config` |
| `BuiltConfigs` | `from_built` | `AlgorithmEvaluatorPair.from_configs` |
| `argparse.Namespace` | `from_args` | `HealthGateConfig.from_cli` |
| 原始 mapping | `from_mapping` | `ExperimentDataFiles.from_data`、`BatchMemoryReading.from_metrics`、`RewardRuntimeLaunchContract.from_component_config`、`MultiReward.from_dict` |

删掉的：`TrajectoryStoragePolicy.from_config` 在生产代码里没有调用方，`rollout.trajectory_storage` 由 schema 直接解析成这个 dataclass；它的两个测试改为走 `RolloutConfig.model_validate`，测的是生产路径真正拒绝的那三种坏值。

没动的：按文件种类命名的 `from_path` / `from_yaml` / `from_capture` / `from_manifest` / `from_video_world_manifest` / `from_file`（后缀说的是文件是什么）；`TeaCacheConfig.from_sampling` 收的是 `sampling.teacache` 的原值（bool 或 mapping）。

## 19. 只读一个对象字段的自由函数（2026-10-10）

按形状扫：模块级函数，第一个参数的类型是 vrl 里定义的类，函数体只读它的字段。29 个命中，处置：

- 改了：`format_distributed_resource_plan(resolved)` 只读 `ResolvedDistributedResources` 的字段、纯格式化 → 它的 `__str__`（docstring 本来就写“给日志和报错看的紧凑一行”），两处日志改为 `logger.info("%s", resources)`。
- 有副作用的命令，不挂到数据类上：`init_rank_process_group(spec)`（初始化进程组）、`build_fsdp_mesh(context)` / `build_context_parallel_mesh`（建 DeviceMesh）。按“配置是数据、命令是调用”的划分，留在拥有这段生命周期的模块里。
- 框架回调：`tensor_to_bytes` / `tensor_from_bytes` 是注册给 Ray 的序列化器（反序列化器必须是普通模块函数，driver 端才不用注册）；`_candidate_sort_key` 是 `sorted(key=...)` 回调。
- 跨多个配置节的规则：`compile_conflicts(root)`、`validate_guarded_rollout_drift(root, precision)` 读 model / actor / distributed / sampling 多节，放在规则模块，不挂到整棵 `RootConfig` schema 上。
- 其余都还读别的参数（加载器、日志、按 kind 分发的工厂、family 自己的加载 glue），或是 §10 决定保留的资源解析阶段。

## 20. 同一种读法只写一份：diffusers config 的键（2026-10-10）

转发形状扫描按名字合计引用数，两个同名的私有函数会互相掩盖。按文件重数后，“config 是 dict 就 `.get`、否则 `getattr`”这个读法写了三遍：

| 原来 | 现在 |
|---|---|
| wan_2_1 `WanI2VDiffusersModel._encode_image_embeds` 读 `image_dim` 时内联了一遍这两个分支 | 调同文件的 `_config_value(config, "image_dim")`；回退到 PEFT 基模型 config 的判断不变 |
| `sana_inference._config_value`：先 `hasattr(config, "get")` 再二选一 | 删掉。`require_scheduler` 收到的永远是真实的 diffusers scheduler（生产加载 `DPMSolverMultistepScheduler`，测试也构造真实类），config 是 `FrozenDict`，直接 `config.get(key)`；`getattr` 分支从未执行 |

wan 的 `_config_value` 留着：它的测试替身 pipeline 的 config 既有 dict 也有 `SimpleNamespace`，两种读法都在用。

## 21. 仓库根只算一处（2026-10-10）

§13 按根收口时处理了用户路径、数据根和 Hub。仓库根虽然有 `vrl.utils.artifacts.repo_root()`，仍有 5 个模块 6 处自己用 `Path(__file__).resolve().parents[N]` 现算，N 跟着文件深度走，文件一挪就会悄悄指错：

| 位置 | 现在 |
|---|---|
| causvid `_resolve_source_root` 的默认 `third_party/CausVid`、`_resolve_configured_path` 的仓库相对回退 | `repo_root() / ...` |
| magi_1 `normalize_magi_1_model_build` 的相对 `python_executable` | `repo_root() / path` |
| `TrainingRunTrace` 记录代码快照的 git 目录 | `repo_root()` |
| perf `correction_ablation` 子进程的 cwd 和 `PYTHONPATH` | `repo_root()` |
| lint `dead_flags` 的扫描根 | `REPO_ROOT = repo_root()` |

同类的另外一处：`vrl.scripts.data.common` 为了“让数据脚本继续从 common 导入”再导出 `repo_root`，两个使用方（bootstrap、danbooru config）改为从所有者导入，再导出删除。

`vrl/scripts/data/setup.py` 的 init-dirs 为 pickapic 建缓存目录时又拼了一遍 `repo_root() / "data" / "cache" / "hf"`，这个目录必须与 pickapic `--cache-dir` 的默认值相同，改为调 `default_cache_dir()`。

替换前逐个核对过每个 `parents[N]` 与 `repo_root()` 是同一目录，行为不变。唯一可见的差别：`default_cache_dir()` 多做一次 `resolve()`，只有 `data` 是符号链接时，init-dirs 打印的路径会显示链接目标，建出的目录不变。`tests/conftest.py` 用 `os.path` 算仓库根是为了在导入 vrl 之前设置 `sys.path`，不在此列。

## 22. Contract / Plan / Requirement 类名（2026-10-10）

词汇只留三种（`*Config`、对象、`typing.Protocol`）。九个类名里带 Contract / Plan / Requirement，逐个看它实际是什么之后：

| 原来 | 现在 | 理由 |
|---|---|---|
| `RayLifecyclePlan` | `RoleLifecycle` | 三个角色的 GPU 集合和由此推出的 park / offload 决定；实例本来就叫 `lifecycle` |
| `RewardRuntimeLaunchContract`（`rewards/launch_contract.py`） | `RewardWorkerConfig`（`rewards/worker_config.py`） | 它就是 reward 组件的 `worker_config` 解析一次后的类型化形式 |
| `GenerationRuntimeLaunchContract`（`generation/launch_contract.py`） | `GenerationLaunchConfig`（`generation/launch_config.py`） | 生成 worker 启动时拿到的配置；不叫 `GenerationWorkerConfig`，因为 `RolloutWorkerConfig` 管的是同一批 worker 的驱动端设置 |
| `OnlineBatchPlan`，字段 `batch_plan` | `OnlineBatchConfig`，字段 `batch_config` | 在线一次更新的批次几何与内存上限，从 `rollout.*` / `actor.*` 解析一次 |
| `EvaluationPlan` 和自由函数 `resolve_plan(args)` | `CheckpointEvaluation` 和 `CheckpointEvaluation.from_args(args)` | 一次 checkpoint 评测的全部输入；产出实例的自由函数按 §2 并成构造方法 |
| `LatentDecodePlan` | 并进 `ChunkedLatentDecoder` | 它只是解码器的构造参数，每次都直接写在 `ChunkedLatentDecoder(...)` 里，只有解码器读它 |
| `EnginePlan` | 删除；`GenerationRequest.sample_batches()`，进程内入口 `forward_plan(request, plan)` 改为 `forward_request(request)` | 只包一个批次元组的外壳，所有使用处只读那一个字段；切批只读请求自己的字段，和 `sample_rows()` 放在一起；进程内入口原来要调用方传同一个请求切出的批次，这个参数可以从请求推出 |
| `ReplayRequestContract` | 这里不改名 | 只有 `DenoiseModelBase` 继承它；replay 模型层次重做时整个并进 `DenoiseModelBase`（§25），单独起新名没有意义 |
| `SequenceRequirement` | 不改 | 编辑序列要满足的条件，YAML 键和报告里都叫 `requirements`；只改类名反而让模块内不一致，改键会改输入格式 |

历史 sprint 和 research 文档保留当时的类名，按上表对照；`docs/ENGINE_ARCHITECTURE.md` 改成现名，并顺手更正它把 reward 那个类写成不存在的 `RewardWorkerLaunchContract`。

