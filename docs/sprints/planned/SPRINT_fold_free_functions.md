# SPRINT：零散的 build / resolve / validate / require 函数，能并进数据类的就并

状态：**本轮完成（2026-10-06）**。问题由 Mingfei 提出：仓库里有很多独立的构建、解析、校验函数，
它们是否必须独立存在，能否并进数据类。

## 0. 判据

扫描 `vrl/` 下顶层的 `build_* / resolve_* / validate_* / verify_* / require_* / check_*` 函数，共 178 个。
每一个只按"并进去之后具体消掉了什么"来判断（CLAUDE.md：只把分支挪进 classmethod 不算简化）：

- **并进数据类**：函数的主要输入是某个数据类，且它重新推导了该数据类已持有的事实，或把同一组参数反复传递。
- **内联**：私有函数只有一个调用方，名字没有带来额外信息。
- **删除**：死代码，或重复了别处已经做过的校验。
- **保留**：作用于普通值、被大量调用（`require_int`、`require_timeout`、`resolve_torch_dtype` 等），
  通过字符串注册的插件入口（各 family 的 runtime 构建函数），跨层会违反依赖方向，或并进去只是改名。
  CLI 的 `build_parser` 不在核心路径上，不动。

## 1. 已做的改动

| 改动 | 消掉的东西 | 提交 |
|---|---|---|
| `validate_model_family_aliases` 删除 | 从未被调用；别名一致性已由测试覆盖 | 本提交 |
| `TRAINING_GATES` 注册表 → `require_training_config` 直接调用两个检查 | 只有两个条目的注册类型、元组、追加约定和一个只为统一签名的适配函数 | 本提交 |
| `ResolvedRun.from_built` + `BuiltConfigs.family` | `resolve_run` / `resolve_online_run` 重复的四步核心链；预检脚本对私有 `_model_family` 的依赖 | 本提交 |
| `ResolvedReward.from_plan / validate_parking / actor_placement / build_function` | 围着三字段数据类的五个独立函数；`validate_reward_memory_parking` 对 park 需求、全 HTTP 豁免、设备的重复推导；`reward_inputs` → `resolve_reward_inputs` 的一层转发 | 本提交 |
| `_RoleResourceBase`（三个角色资源配置的共同基类） | `devices` 字段在四个函数里各自解析和校验；reward 配置类自带一份 `key_prefix` / `pins_devices`。现在构造时解析、校验一次并以规范形式存储 | 本提交 |
| `PromptExample.with_resolved_references / with_resolved_artifacts` | 两个对 `PromptExample` 的特性依恋函数；`data_root` / `allow_absolute` 在四到六处 `resolve_artifact_path` 里重复传递；没有再保留单独的解析器工厂函数 | 本提交 |
| checkpoint 来源只判断一次本地/远程 | 同一判断做两遍，以及为调和两遍而存在的不可达分支 | 本提交 |
| `build_family_runtime_bundle` 内联进 `ModelFamilyEntry.build_rollout` | 单调用方转发 | 本提交 |
| causvid 许可证只检查一次 | 每次构建检查两遍 | families 提交 |
| magi 单进程配置只在安装预检时检查 | 每个请求对未改动的配置重复检查 | families 提交 |
| `validate_checkpoint_compatibility` 删除 | 只做 None 判断的包装；调用方本来就有自己的 None 分支 | 本提交 |
| `build_rollout_schedule(config, lifecycle, *, …)` | 七个只转交给 coordinator 的参数 | orchestration 提交 |
| continuous 调度构造时的 reward 隔离检查删除 | 与启动前拓扑校验完全等价的重复校验 | orchestration 提交 |

## 2. 审查后保留的

- `BuiltConfigs` 与 `ResolvedRun` 分两层：reward 预检和 supervisor 只要校验过的配置，不能被迫解析 GPU 拓扑。
- `resolve_model` 不并进 `ResolvedRun`：评估脚本没有已解析的训练 run，在线训练用的是 rank 本地设备，
  做成方法会多出第二个入口。
- 各角色的设备解析函数（`_resolve_role_devices` / `_resolve_rollout_devices` / `_resolve_reward_devices`）：
  规则各不相同，做成同名不同签名的方法只是挪位置。
- `resolve_checkpoint_model_identity`、`validate_checkpoint_meta_compatibility`、`validate_rng_state`、
  `build_adapter_exports`、`build_strategy`、`check_cross_section_rules`、`validate_rollout_schedule_topology`：
  多调用方、作用于普通值，或并进数据类会违反层间依赖。
- 各 family 的私有解析步骤（causvid、magi、echo）：单调用方但各自是被直接测试的命名步骤，内联只会让构建函数变长。
