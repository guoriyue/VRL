# SPRINT: 减少引擎配置要手填的魔法字符串（Literal 化 + 派生默认）

状态：proposed / 设计（待批）

## 0. 结论 (TL;DR)

目标:**让引擎配置少要求用户精确填魔法字符串**。两条手段,跟仓里 `algorithm.kind` / `data.loader` 已经做过的 `Literal` 先例(`schema.py:88,116`)一致:

1. **Literal 化** —— 合法集是 vrl 自己拥有的小而固定集合、且现在靠**手写 membership 校验**的设置,改成 `Literal[...]`。类型即合法集 → 删手写校验集、报错更清晰、unknown-key 扫描也认得。
2. **派生默认** —— 取值能从已有信号(`model.family` / `task` / `algorithm.kind` / init-path / `artifact_format` / 是否独立 rollout worker)唯一确定的,给派生默认,用户不用填。**只填 UNSET,绝不覆盖显式配置**(护栏)。

编目了 ~16 个 user-facing string 设置:**6 个 both(Literal+派生)、2 个 Literal-only、1 个低优先 both、1 个直接删旋钮、6 个明确 leave**(外部依赖词汇 / per-reward 词汇 / 单值占位 / dead knob —— Literal 会 rot 或无意义)。

顺带挖出**两个该修的东西**:
- **`artifact_format` 有 latent bug**:代码默认 `'tensor'`,但 `.pt` decord 读不了,**每个能跑的 Kling run 都手动覆盖成 `mp4`**(`kling_video_reward.yaml:23` 注释明说)。派生默认应修成"video reward → mp4"。
- **`distributed.rollout.placement_strategy` 是 dead knob**:rollout 侧从不消费它(实际走 `GlobalRayPlacementOwner._strategy()` 的拓扑派生),5 个 base config 里设的全是 no-op。属删/接线决策,不在 Literal/派生范围。

---

## 1. 判据(什么改 Literal、什么派生、什么 leave）

| 处置 | 触发条件 |
|---|---|
| **Literal** | 合法集是 **vrl 自己拥有的、小而固定** 的集合,且现在用手写 membership check |
| **派生默认** | 取值能从已有信号唯一确定;只填 UNSET,不覆盖显式值 |
| **LEAVE** | (a) 合法集属**外部依赖**(torch.compile mode、vLLM cache_dtype)→ Literal 会随上游增值而 rot;(b) **per-reward 词汇 + 复合表达式**(score_key)→ 全局 Literal 跨 reward 误收;(c) **单值占位**(ar_engine/scheduling 的 vestigial)→ Literal 退化无意义;(d) **dead knob**(rollout placement_strategy)→ 该删/接线而非 typing |

---

## 2. 逐设置编目

### 2.1 both —— Literal + 派生默认（高价值）

| 设置 | 现状(手写校验在哪) | Literal | 派生默认从 | 风险 |
|---|---|---|---|---|
| `rollout.denoise_mode` | `Any=None`(schema:225)无类型;手查 `{"native","sde"}`(layout.py:270) | `Literal["native","sde"]`(标量字段,不用子模型) | **family**:只有 wan_2_1 用 `native`,其余继承 base `sde` → `family=="wan_2_1"→native, else sde` | 低(Literal)/ 中(派生:base 现在对所有 family 硬默认 sde,只填 UNSET 且现有 wan 配置都显式设了,已核对) |
| `distributed.rollout.sync_trainable_state` | 默认 `"disabled"`(config.py:34);手查 `{"disabled","lora_only"}`(config.py:54) | 实质是 **on/off**(消费者只判 `!= "disabled"`,`"lora_only"` 无语义,名字误导) → honest 形式是 bool | **派生开**:所有真实 online 配置都设 `lora_only`,默认 `"disabled"` 实际全错;online 入口要训练就要 resync → 默认 ON,无可训模块/无独立 rollout worker 才 OFF。**注意:`use_lora` 不是充分信号**(cosmos 全参也要 sync,online.py:181) | 中(收益大:删了 100% 真实配置都要填的魔法字符串 + 纠误导名;翻默认前确认无入口依赖静默 `disabled`,grep 显示 configs/ 里无) |
| `sampling.attention_backend` | 不是 SamplingConfig 字段(无类型无默认);手查 `_backend_builders` keys(ar_attention_backends.py:51) | `Literal["vllm_paged","torch_native"]` 加到 SamplingConfig | 默认 `"vllm_paged"`(现有 fallback);两个 AR family 都用它,无需按 family 派生 | 低(保行为 + **顺带修 false "unknown key" 警告**:它现在不是 schema 字段) |
| `reward.kwargs.*.execution` | OPEN dict;手查 `{"inline","pool"}` 三处(runtime.py:82 / base.py:174 / pickscore 默认) | `Literal["inline","pool"]` 作 **reward `__init__` 参数类型**(reward.kwargs 故意 OPEN,Literal 不能进 schema) | **init-path**:disk-artifact reward 硬要 pool(base.py:174 拒其他)→ 默认 pool;in-memory model reward 默认 inline | 低(保行为)**但有一个真 footgun**:factory.py:136 / resources.py:1378 数 pool reward 是读 **raw kwargs**,派生必须发生在 **resolution 层(factory)在计数之前**,否则 pool reward 不被计入 GPU 分配 |
| `reward.kwargs.*.media_type` | 手查 frozenset `MEDIA_TYPES={"image","video","tensor"}` 三处(inference.py:39 / artifacts.py:28 / _validate_media_shape) | `Literal["image","video","tensor"]`;**MEDIA_TYPES 用 `get_args()` 从 Literal 派生**(单一真相源,别同时维护常量+Literal) | `mp4→video` 已是硬蕴含(artifacts.py:32);disk-artifact 默认 video、in-memory 默认 image。只填 UNSET | 低-中(保留 `tensor` 第三值,别被 video-only 派生丢掉) |
| `reward.kwargs.*.artifact_format` | 手查 `{"tensor","mp4"}`(artifacts.py:30)+ mp4↔video 耦合 | `Literal["tensor","mp4"]` | **修 bug**:video reward 默认应是 `mp4`(现 Kling 代码默认 `tensor` 与唯一能跑配置矛盾) | 中(这条是 latent bug 不只是冗长;改默认是对依赖 `tensor` 默认者的行为变更,但那种 caller 本就被 decord 卡死) |

### 2.2 Literal-only（不派生,合法集是用户的 per-experiment 选择）

| 设置 | 现状 | 动作 | 为何不派生 |
|---|---|---|---|
| `rollout.sde.type` | flat ConfigBlock dict;手查 `{"sde","cps"}` **三处**(schema:457 / layout.py:262 / flow_matching.py:116) | 把 sde block 提成 typed 子模型 `SdeConfig(type: Literal["sde","cps"], ...)`(像 FSDPConfig);删 schema:457 重复校验;**保留** layout 请求边界 guard | sde vs cps 是 per-experiment 采样选择,同一 family 两边都出现,不可靠派生;默认 `sde` 已存在 |
| `distributed.rollout.chunk_placement_strategy` | 默认 `round_robin`;手查 `{"round_robin","dynamic"}` **两处**(config.py:56 + chunk_placement.py:34) | `Literal["round_robin","dynamic"]`;去重 `RayGenerationConfig.__post_init__` 的重复检查(**保留** `ChunkPlacementPolicy` 运行时 guard + `_PLACEMENT_STRATEGIES`) | 是 perf/调度选择;翻默认会静默改调度行为 |

### 2.3 低优先 / 单家族

| 设置 | 动作 | 备注 |
|---|---|---|
| `rollout.final_image_policy` | `Literal["always_generate","use_selfcheck"]` + 收掉 `rollout.*` 与 `sampling.r1.*` 的重复(让用户只填一处,另一处派生) | 只 janus_pro R1 一条路径,payoff 低;collapse 重复触及 collector/config.py 的 `_copy_first_present` + schema 相等校验,别过度 |

### 2.4 直接删旋钮

| 设置 | 动作 | 备注 |
|---|---|---|
| `reward.kwargs.*.scheduling` | 从 YAML 删(只接受 `"sync"`,默认就是 sync);**保留** base.py:169 运行时 guard 作 forward-compat | 单值集别造 Literal;两个 YAML 在填唯一合法值,纯噪声 |

### 2.5 明确 LEAVE（记录原因,免后续重提）

| 设置 | 为何 leave |
|---|---|
| `rollout.denoise_compile.mode` | torch.compile 的词汇(`default`/`reduce-overhead`/...)是 PyTorch 拥有的;Literal 会随上游增值 rot;已有默认 `"default"` 且 compile 默认关 |
| `sampling.ar_paged_cache_dtype` | vLLM 的 KV-cache dtype 词汇;仓内无 membership check(纯 pass-through);Literal 会复制 vLLM 集合并 rot;已有默认 `"auto"` |
| `sampling.ar_engine` | vestigial guard:只 `"native"` 合法,纯为拒绝 stale `vllm`/`hf` 报清晰错;单值 Literal 退化且会**恶化**报错信息;别升成 advertised 字段 |
| `sampling.ar_paged_block_size` | 是 int 不是魔法字符串(out of scope);已有默认 16 |
| `reward.kwargs.*.score_key` | per-reward 词汇(Kling vs VideoCon 集合不同)+ 复合表达式(`a+b`);全局 Literal 会跨 reward 误收;已有 per-reward 默认 + 缺键即报错;README 已记录 |
| `distributed.rollout.placement_strategy` | **dead knob**(rollout 从不消费,实际走 `GlobalRayPlacementOwner._strategy()` 拓扑派生);该删或接线(单独决策),**别 Literal**(typed 会误以为它生效) |

---

## 3. 实施计划（分阶段,每阶段独立 PR）

**P0 — 高价值、保行为、干净**(§2.1 的 6 个 + reward 一组):
- `denoise_mode`、`sync_trainable_state`、`attention_backend` 三个 rollout/sampling 设置:Literal + 派生默认。
- reward 一组(`execution` / `media_type` / `artifact_format` / `scheduling`):Literal 落 `__init__` 参数类型;派生从 init-path/artifact_format;`artifact_format` 默认修 mp4;`scheduling` 从 YAML 删。
  - ⚠️ **`execution` 派生必须在 factory resolution 层、在 pool-reward 计数之前**,否则 GPU 分配漏算(见 §2.1 风险)。

**P1 — 需要小结构改动**(§2.2):
- `sde.type` → 提 `SdeConfig` typed 子模型 + 删重复校验 + 保留 wire-boundary guard。
- `chunk_placement_strategy` → Literal + 去重 `__post_init__`(保留运行时 policy guard)。

**P2 — 低优先**(§2.3):
- `final_image_policy` → Literal + 收 rollout/sampling.r1 重复。

**分开决策**:`distributed.rollout.placement_strategy` dead knob —— 删 or 接线到 `GlobalRayPlacementOwner`,需 owner sign-off(归 [[SPRINT_design_smell_audit]] backlog 同类)。

---

## 4. Architecture Hygiene

- **单一真相源**:`MEDIA_TYPES` 若 Literal 化,frozenset 用 `typing.get_args()` 从 Literal 派生,**不要**常量+Literal 并存。
- **reward.kwargs 故意 OPEN**(schema:54-57,"config 层不复制 per-reward 知识"):Literal 不进 schema,落 reward `__init__` 参数类型。
- **保留所有运行时 guard**:`layout._parse_sde_type/_parse_denoise_mode`、`resolve_attention_backend`、`ChunkPlacementPolicy`、`base.py` 的 scheduling/execution 校验 —— rollout 请求过 Ray wire 是 **plain dict**,schema 的 Literal 管不到,运行时 guard 是 wire-boundary 防线。Literal 只是 user-facing allow-list,不取代运行时 guard。
- **不 Literal 外部依赖词汇**(torch mode / vLLM dtype):会 rot,违反"不手维护复制外部真相源的常量"。
- **不 flatten registry,不动跨 family 并行 shape**。

---

## 5. 非目标

- 不改 RL 学习 / 算法数学。
- 派生**只填 UNSET**,绝不覆盖用户显式设的值。
- 不给单值集造 Literal(`ar_engine` / `scheduling`)。
- 不删 registry-listed key without owner sign-off(`placement_strategy` dead-knob 单独决策)。
- 不把 reward.kwargs 收成 typed schema(它是故意 OPEN 的 per-reward 契约面)。

---

## 6. 验收

**单测**:
- Literal:非法值被拒,报错含 dotted path(`unknown <path>=...; expected ...`,对齐 schema:497-500 现有格式)。
- 派生:仅在 UNSET 时生效,与显式值一致;`denoise_mode` family 派生、`execution` init-path 派生、`media_type`/`artifact_format` 从 artifact_format 派生各一例。
- `attention_backend` 加字段后 unknown-key 扫描不再误报(`test_unknown_keys`)。
- 所有现存 `configs/*.yaml` 仍 load 通过(`test_load_all_experiments`)。

**回归(关键)**:
- `execution` 派生后 factory/resources 仍正确计 pool reward(GPU 分配不漏算)—— 这是唯一的真 footgun,必须有专门测试。
- `sync_trainable_state` 翻默认后,weight-sync 在所有 online 入口仍按预期开/关。

---

## 7. 参考

**代码(vrl2/VRL,当前 `feat/reward-exec-cost-hint` 分支,与 origin/main 一致）**
- `vrl/config/schema.py` — RolloutConfig(`:207-225` sde/denoise/final_image_policy)、SamplingConfig(`:235-258`)、`_cross_field_validate`(`:455-485`)、Literal 先例(`:88,116`)、错误格式(`:497-500`)
- `vrl/generation/diffusion/layout.py:122,261-274` — `_parse_sde_type`/`_parse_denoise_mode`(wire-boundary guard)
- `vrl/generation/ray/config.py:33-59,105-109` — `chunk_placement_strategy`/`sync_trainable_state`/`placement_strategy` + `__post_init__` 手查
- `vrl/generation/execution/chunk_placement.py:17,34,122` — `_PLACEMENT_STRATEGIES` + 运行时 policy guard
- `vrl/nn/modules/ar_attention_backends.py:25-68` — `_backend_builders` + `resolve_attention_backend`
- `vrl/rewards/runtime.py:71-84`、`vrl/rewards/base.py:143,152,169-175,181-189`、`vrl/rewards/inference.py:14,39-98`、`vrl/rewards/artifacts.py:28-104` — reward execution/media_type/artifact_format/scheduling/score_key
- `vrl/scripts/common/factory.py:136`、`vrl/ray/resources.py:1378` — pool-reward 计数(execution 派生 footgun 落点)
- `vrl/ray/placement.py:341-345` — `GlobalRayPlacementOwner._strategy()`(rollout placement_strategy dead-knob 的真实来源)

**相邻 sprint**
- `done/SPRINT_config_unknown_key_warning.md` — unknown-key 单遍走查(Literal 化要不破它)
- `planned/SPRINT_design_smell_audit.md` — `placement_strategy` dead-knob 归此类 backlog;`kind`/`loader` Literal 化先例
