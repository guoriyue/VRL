# SPRINT: Ray rollout 容错 —— actor 死亡从"整个 run 报废"降级为"丢一个 chunk"

状态:**planned**。前置:[`SPRINT_ray_cluster_ownership_and_shared_host_isolation.md`](../done/SPRINT_ray_cluster_ownership_and_shared_host_isolation.md)
已落地(归属显式、teardown 有序);本 sprint 在其之上加"运行中"的健壮性。

## 背景

2026-06 的 Janus 冒烟事故:一个 Ray generation worker 收到来源不明的外部
SIGTERM,整个 24h 长跑报废。现在的恢复模型是 `run-until-success` 守护脚本级别
的"全量重启 + 从 checkpoint 续"——正确但粒度太粗,in-flight rollout 全部陪葬。

cosmos-rl 的对应设计(dynamic NCCL process group 动态注册/注销、controller 的
JobPhase 状态机 + 显式 StopCommand,见其 PR #696)证明:**replica 死亡应当是
常态事件,不是致命错误**。VRL 的基建已有一半:lease 重拉起路径
(`vrl/generation/ray/runtime.py:370` 起)已经会在 `state.runtime is None` 时重建
runtime 并回放 `state.last_state` + `policy_version`。缺的只是把"actor 意外死亡"
接成这条路径的触发器,以及让关闭路径能区分"正常 DRAINING"和"运行中被杀"。

## 范围

### 1. Actor 死亡检测与重拉起

- 在 rollout 请求的 await 点捕获 `ray.exceptions.RayActorError`(actor 死亡的
  规范信号;确认 `ray.exceptions.ActorDiedError` 在当前 Ray 版本的继承关系后
  选择捕获的基类)。
- 死亡后走**已有的**重拉起机制:重建 actor(placement group 槽位复用)→ 回放
  最近同步的权重(`last_state`, `policy_version`)→ 重发当前 chunk 请求。
- 重试预算:连续 N 次(默认 3)**同因**死亡才放弃并抛出;不同因(如先 OOM 后
  SIGTERM)分开计数。放弃时抛出的异常必须携带全部死因记录。
- OOM 死因与现有 OOM-split 机制的关系:OOM 已有专门处理路径,本 sprint 的重拉
  起只负责非 OOM 死因;OOM 死因直接转交现有路径,不重复建设。

### 2. Run 级 phase 状态机(容错的判别前提)

- 引入显式 run phase:`RUNNING → DRAINING → STOPPED`。shutdown 入口先置
  DRAINING,再走既有的 teardown 顺序。
- 容错逻辑读 phase 决定行为:RUNNING 中 actor 死 → 重拉;DRAINING/STOPPED 中
  actor 死 → 放行(那是我们自己杀的)。
- 形状参考 cosmos-rl 的 JobPhase,但只做三态,不做 controller——单 driver 内
  一个字段够了,不引入新进程或消息系统。

### 3. 观察性:actor 命名 + 死因采集

- generation worker actor 用 `name=f"{family}-worker-{worker_id}"` 创建
  (namespace 用 run id 隔离,避免跨 run 撞名)。
- 死亡/关闭路径用 `ray.util.state.list_actors`(或 actor handle 的 death cause
  API)采集退出原因(exit code、OOM、被 kill、node 失联)写进日志。上次事故
  查不出凶手,一半原因是现场什么都没留下。

## 验收标准

- fake 层:actor 首死 → 重拉起 → 权重版本正确 → chunk 重发成功;连续同因死亡
  达预算 → 抛出且异常含死因链;DRAINING 中死亡 → 不重拉。
- **真 Ray 孪生(slow_test,无需真卡)**:`ray.init(num_cpus=4)` 本地集群,
  CPU actor 处理请求中途被 `ray.kill`,断言 run 完成、结果完整、日志含死因。
  这是本 sprint 的硬验收——fake 单独通过不算完成。
- phase 状态是行为消费的(容错分支读它),不是 log-only 字段(死字段规则)。
- 全套既有测试无回归。

## 测试策略(回应 fake 测试膨胀的关切)

- `_FakeRay` 目前在 `tests/scripts/test_online_ray_cluster.py`、
  `test_online_lifecycle.py`、`tests/generation/ray/test_runtime_config.py`
  三处各自实现——本 sprint **先合并成一个共享 fixture**(form-4:三份复制体),
  再在其上加新行为,不允许出现第四份。
- 纪律:每个新增编排**行为**必须有真 Ray `slow_test` 孪生;fake 只测协议边界
  的分支逻辑。fake 数量只减不增。

## 非目标

- 弹性伸缩(按队列深度增减 worker)——没有多节点用户前是伪需求。
- controller 进程 / 消息系统 / single-controller 重写。
- trainer 侧容错(FSDP rank 失联恢复)——训练进程死亡仍走 checkpoint 续跑。
- 事后追查 2026-06 事故的凶手。

## 参考

- 重拉起现成机制:`vrl/generation/ray/runtime.py:352-380`
- teardown 顺序:`vrl/scripts/common/online.py:_shutdown_online_recipe_runtime`
- cosmos-rl StopCommand/JobPhase/teardown:
  https://github.com/nvidia-cosmos/cosmos-rl/pull/696
- Ray actor 容错语义:
  https://docs.ray.io/en/latest/ray-core/fault_tolerance/actors.html
