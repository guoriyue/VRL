# SPRINT: Resource/config resolution audit — resolve_distributed_resources 与 config 校验层裁决记录

**日期**: 2026-07-12  **状态**: PLANNED（主体是裁决记录；动作项只有 §3 的两小条）
**触发**: "resolve_distributed_resources 为什么在那里 / config 加载校验函数一大堆"。
本文回答 why、给出逐字段消费者核查结论，防止下轮审计重翻。

---

## 1. `resolve_distributed_resources` 为什么在那里（结论：保留原样）

**它是什么**：把 YAML 的 role 级资源声明（`distributed.resources.*` +
`distributed.training.*` + reward inference 种类）一次性解析成具体 CUDA 序号
和派生的生命周期计划的**唯一决策点**。它不是 wrapper——是 2026-06-17
"放置决策"轮（per-role GPU ids、overlap 由交集派生、release 标志由拓扑派生，
不设独立 YAML lifecycle 键）的落点。

**谁在消费**（4 个入口 + 1 个配置桥）：
- `vrl/scripts/common/online.py:769`（在线 recipe 主入口）
- `vrl/scripts/common/factory.py:99`（reward 设备）、`:330`（lifecycle）
- `vrl/scripts/diffusion/wan_2_1/train_dpo.py:215`（离线 DPO）
- `vrl/generation/ray/config.py`（`RayGenerationConfig.from_cfg` 携带
  `ResolvedDistributedResources`）

**逐字段消费者核查**（AGENTS.md Resolved* 死字段规则，2026-07-12 亲验）：
14 个字段全部有非日志读方——`lifecycle` 65 处、`cross_node` 10 处、
`rollout_devices` 7 处、`reward_num_workers`/`reward_devices` 各 5 处、
最少的 `reward_uses_trainer_device`/`reward_cpus_per_worker`/`rollout_num_gpus`
也各有行为消费者；`visible_devices` 已在定义处显式标注 display/provenance-only
（`format_distributed_resource_plan` 打印全池）。**无死字段，无动作。**

**为什么 200 行不拆**：函数体是五段直线决策（可见设备 → 逐 role 设备解析+校验 →
worker 数 → overlap/独占校验 → lifecycle 派生），每段的中间量（trainer_devices、
colocated、reward_execution_devices）被后段交叉消费——拆成私有函数就是制造
一串单调用 helper（违反 no-single-caller-helpers），决策也不再能自上而下读完。
fsdp 对称/非对称双拓扑、`gpu_pool=trainer` 即 overlap 许可等规则都有行内注释
交代出处。**保留单函数形态。**

## 2. config 加载/校验函数群为什么那么多（结论：分层各有职责，保留）

四层各管一件事，函数数量来自"每个生产门独立可测"的设计而非提取癖：

| 层 | 位置 | 职责 |
|---|---|---|
| typed schema | `config/schema.py`（pydantic） | 结构/枚举/跨字段合法性，parse 时机 |
| unknown-key walker | `config/unknown_keys.py` | 全树未知键单点告警（typo=死键=遗留键同一处理） |
| production gates | `config/validation.py` | `validate_production_*` 运行前合同（如 kling reward 必须真模型） |
| builders | `config/builders.py` | cfg → 运行时 dataclass 投影 |

本轮通读结论：`validation.py` / `builders.py` 全部函数有真调用方；
单调用者（`validate_reward_config`、`resolve_algorithm_kind`、
`build_trainer_config`、`build_reward_config` 等）都是被独立单测钉住的
公共 builder/gate 边界，在薄函数保留清单内。**无合并/删除项。**
（上一轮已删的 `sampling.cfg`、`task_variant` 死键不再赘述。）

## 3. 仅有的两条动作项

1. `resolve_distributed_resources` 体内五段决策补齐段落注释锚
   （`# -- visible devices / role devices / worker sizing / overlap checks /
   lifecycle derivation --`），成本 5 行，让"为什么在那里"从函数体自答。
2. 本文归档后，在 `vrl/ray/resources.py` 模块 docstring 加一行指向本 sprint
   （裁决记录防重翻，与 grab-bag audit 对 fsdp.py "一袋函数合法"的处理同款）。

## 非目标

- 不把 resources.py 类化/拆文件（同 grab-bag audit 对 fsdp.py 的裁定）。
- 不给校验层引入统一"ValidationPipeline"抽象——四层边界清楚，
  合并只会把 parse 时机和运行前门混在一起。
