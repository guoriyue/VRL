# SPRINT PROGRAM: Core contract hardening（planned）

状态：**planned / CPU-only implementation program**。

## 结论

本轮不新增“万能 runtime”“统一 manager”或新的 family 层。审计找到的是五个已经有
生产消费者、并且能用反例证明的基础契约缺口：

1. generation → models 的静态依赖 floor 可以被等价 import 语法绕过；
2. consumer-facing Protocol 被 concrete implementation nominally 继承，缺失实现会
   继承 `...` 空桩并静默返回 `None`；model guard 也只检查同名 callable，不检查生产
   调用是否能绑定；
3. reward-overlap acceptance 会把缺 verdict、缺承重 metric 的不完整 campaign 判为
   accepted；
4. Ray actor pool 每释放一个 slot 都重扫全部 pending jobs，CPU 基准呈稳定二次增长；
5. rollout worker 的 health endpoint 明确与 model residency 解耦，但 parked 时 monitor
   反而暂停，且 idle failure 没有交给最终 recipe verdict。

这五项按可审查里程碑独立提交。每项都有 true/false regression；不运行 Ray/GPU
campaign，不提交用户当前脏文件，不 push。

## 里程碑顺序

| 顺序 | Sprint | 为什么先后这样排 |
|---|---|---|
| 1 | [[SPRINT_generation_models_interface_floor]] | 先让架构 gate 本身可信，后续改动受同一依赖方向约束 |
| 2 | [[SPRINT_protocol_contracts_fail_closed]] | 让扩展 ABI 在构造/边界处失败，而不是深层执行后静默失败 |
| 3 | [[SPRINT_reward_overlap_benchmark_evidence_contract]] | acceptance verdict 是是否合入性能特性的证据边界，必须先 fail closed |
| 4 | [[SPRINT_ray_actor_dispatch_scalability]] | 在保持 Ray await/placement 语义的前提下消除已测得的二次调度成本 |
| 5 | [[SPRINT_rollout_worker_idle_liveness_correction]] | 修正跨线程、parking、terminal cleanup 的完整 ownership 链 |

## Principal-level 设计门槛

### LLVM

- 每个抽象是否代表稳定边界或不变量，而不是隐藏三行代码？
- verifier/gate 是否能拒绝语义等价的坏输入，而不是只匹配一种文本形状？
- 错误是否在最早拥有完整上下文的边界报告？

### SGLang / vLLM

- runtime capability 是否 structural、可替换、可通过 registry 扩展？
- scheduler 是否保持稳定 priority、placement 与 gather-order 语义？
- actor/RPC ownership 是否只有一个事实来源？

### PyTorch

- Protocol、ABC 与 concrete implementation 的职责是否分离？
- 是否避免依赖 Python typing 的“只查属性存在”行为去保证运行时调用 ABI？
- checkpoint 与 state ownership 是否尊重 distributed rank 边界？

### Triton

- 热路径是否避免随 queue depth 二次增长？
- 性能测试是否测算法增长而不是用易抖动 wall-time 断言？

### Ray

- ObjectRef 等待是否继续 async/non-blocking？
- worker process reachability 是否与 model/GPU residency 分开？
- monitor thread、event loop 和 actor cleanup 是否有明确单 owner 与失败交接？

## 明确不做

- 不合并 generation Protocol、implementation base 和 family executor。薄 facade/
  binding 在这里提供跨 family 一致形状，grepability 比少几行 LOC 更重要。
- 不把 `ChunkResult = Any` 强行类型化。它是 family-owned opaque transport；
  driver/gatherer 已在各自边界验证 payload。
- 不建立统一 perf-script registry 或万能 benchmark result。不同 probe 的测量对象与
  生命周期不同；这里只收紧会直接产出 merge verdict 的 overlap benchmark。
- 不替换 Ray ObjectRef 的直接 `await`，也不使用阻塞 `ray.get` 作为 generation wait。
- 不在本批实施 FSDP full-checkpoint rank0 ownership。审计已确认每 rank 预加载完整
  payload 是真实扩展瓶颈，但它同时涉及 model/optimizer/EMA/RNG/progress 的
  collective failure protocol。该项必须以 2-rank CPU Gloo、rank0 load failure
  broadcast 和 peak-RSS 基准为一个独立 program，不能夹在这些局部安全提交里。

## ALL_CAPS 与薄边界裁决

- **KEEP** architecture-test allowlist：刻意隔离的 boundary taxonomy。
- **KEEP** reward benchmark 的 A/B/C mapping、统计阈值与 t critical table：公开
  acceptance protocol / 隔离统计表。
- **KEEP** `HEALTH_CONCURRENCY_GROUP` 和 monitor join grace：Ray protocol name 与
  cleanup bound。
- **KEEP** worker `health()`：专用 Ray adapter，即使函数很薄也是真 framework boundary。
- **ADD** actor pending scheduler 私有类：它拥有 priority/capacity/worker-binding
  不变量，减少真实复杂度；不是 `_helper` 装饰。
- **不新增** protocol facade、manager、handler 或重复 field-name 常量。

## Program 验收

- 五个 child sprint 全部移入 `docs/sprints/done/`；
- 每个 child 有独立 commit 与 CPU true/false tests；
- touched Python 逐文件 Ruff check/format；
- architecture、generation contracts、model interfaces、perf verdict、Ray actor pool、
  health/lifecycle/collector/online cleanup 的组合 CPU suite 通过；
- `git log` 展示可逐个 revert 的里程碑；
- `git status` 只保留用户开始前已有的改动；没有 push。
