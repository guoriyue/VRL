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
| config / run | `ResolvedRun` 多个只转手的属性；`TrainerConfig.from_root(root)`、`ResolvedDistributedResources.from_root(root)` 的第二个"派生"参数；`normalize_precision` 的默认值 | 被调方自己从 root 推导 |
| generation | 执行器 `pipelined` 标志（有 finalizer 即流水）；`GenerationWorkerParking.sleep` 的设备参数；`EnginePlan.from_request` 外的 request 字段；`BundleLayout.rollout_gpus_per_engine`、`StagedBatchRefs.policy_version`、`measurement_scope`；`from_rank_results` 的可选 `expected_worker_ids`；若干 isinstance 守卫 | executor / 请求 / 契约本身 |
| generation | `merge_generation_batches` / `forward_plan` / `forward_plan_pipelined` / `sort_and_validate_batch_coverage` / finalizer `merge_request` 的 `sample_rows`；Ray 执行器不再把行随请求发给 finalizer | `request.sample_rows()`（确定性，contracts 用例钉住） |
| generation | `GenerationWorkerCore` 的 gatherer（rank 只产 batch，从不合并）和执行器"是否实现 forward_batch/merge"的可调用检查 | 合并在 driver 侧的注册表 gatherer |
| rollouts / rewards | `ContinuousRolloutConsumer`/`Producer`/`collect_iteration` 只转交的参数；`RolloutCollector` 构造改为 `from_family(entry, …)`；`RayRewardScorer` 三个常量化的超时 / CPU 参数；`RewardFunctionRuntime(reward_function)`；协调器的 `weights_initialized`；预检改调 `reward.validate_parking()` | 对象自己持有 |
| trainer / strategy / data | 流式更新 helper 的 `batch_plan`；`OnlineTrainer` 的 `ref_model`（KL 项或契约要求时策略即参考）与 `device`（恒等于 `strategy.context.device`）；回放选择 helper 的 config 字段参数；`init_training_process_group` 的 backend；策略各自建 collectives；FSDP mesh 词表由 schema 持有；`ContextParallelStrategy` 读 `context.cp_size`；`select_trainable_state`、`ImageCaptionPromptDataset.from_config`、`DatasetFileReport.artifact_count` | trainer config / strategy context / 模块自身 |
| scripts | `AlgorithmEvaluatorPair.from_configs` 的 `family_entry`（`built.family`）；`OnlineRecipeRun.initialize_metrics` 的 `training_context` / `output_dir`；`_load_sft_latents_from_config(built, *, sft_weight)` | run 持有的 strategy / trainer config |
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
| `AlgorithmConfigContract` 名字不说明是什么；`tolerates_off_policy_staleness` 在所有在线目标上都等于 `not requires_previous_policy`（行为策略是当前权重的目标才吃不下 staleness） | 改名 `AlgorithmRequirements`（类属性 `requirements`），删掉 `tolerates_off_policy_staleness`，continuous 检查直接读 `requires_previous_policy` |
| GRPO / FlowDPPO / GRPOGuard / DiffusionNFT 各自实现一遍"绑定 advantage estimator"和两个 advantage 方法；FlowDPPO / GRPOGuard 复制 GRPO 的构造器而不是继承；`_initialize_*` 两个只在构造器里用的 helper | `GroupAdvantageObjective`（`advantages.py`）一次持有 estimator 与两个 advantage 方法，GRPO / NFT / V-GRPO 继承；构造器收 reward 的 `component_weights` 而不是调用方先 build 好的 estimator；FlowDPPO / GRPOGuard 不再写构造器 |
| `kl_coef = 0.0` / `sft_weight = 0.0` 在 FlowDPPO 和 GRPOGuard 各写一遍 | 共同父类 `TrustRegionGRPO` 声明一次（trust region 取代 KL 项，且无 SFT 项） |
| `VGRPOConfig` 重复 `GroupAdvantageConfig` 的 `eps / adv_clip_max / global_std`，V-GRPO 自己再调一次 `group_relative_advantages` | `VGRPOConfig(GroupAdvantageConfig)`，默认 `advantage_combine="weighted_sum_raw"`（与原行为逐位相同），advantage 走同一条 estimator 路径 |
| `config: ... | None = None` 让基类替子类选默认配置，子类因此各写构造器 | 所有目标的 `config` 必填；factory 本来就总是传 |
| `Evaluator` Protocol 与 `ReplayEvaluatorBase` ABC 各声明一遍 `replay_granularity` / `supports_deferred_replay_tensor_move`；trainer 再在运行时校验字符串取值 | 只剩 ABC `Evaluator`，两个属性是 `ClassVar[Literal[...]]`；删掉 trainer 的字符串校验和对应用例 |
| GRPO / GRPOGuard 的 loss 里 `keep is None` 分支：每个 signal 都带 mask，`combine_keep_masks` 永远返回张量 | 删掉死分支；`_broadcast_sample_values` 去掉对非张量的 getattr 探测 |
| trainer 判断是否需要参考策略：`(uses_evaluator and kl_coef > 0) or requires_reference_policy` | `kl_coef > 0 or requires_reference_policy`（无 evaluator 的目标 kl_coef 本来就是 0，NFT 由 requirements 声明） |

`precision_correction` 保留：它不是修精度损失，而是在 rollout 与 replay 精度不一致（如 FP8 rollout、fp32 replay）时把 `exp(replay - rollout)` 这个重要性权重截断（TIS）并整条拒绝越界样本（RS），让少数漂移样本不能主导梯度；`TrainerConfig.from_root` 在精度分裂时自动开启。它只对 GRPO 族（有重要性比）有意义，所以只在 `GRPO` 构造器上出现一次。
