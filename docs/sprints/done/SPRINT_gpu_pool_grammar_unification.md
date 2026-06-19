# SPRINT (done): 把 rollout 共卡统一到 `gpu_pool` 语法

状态：**done（2026-06-18 归档至 done/）**。已实现 + 测试（2026-06-18），纯语法/命名统一 + 兼容垫片，零行为变化（旧 `colocate` 等价已单测验证）。
ray+config+trainers+generation/ray+online-lifecycle 共 420 passed。

实现要点：`RolloutResourceConfig` 加 `gpu_pool`(auto|trainer|dedicated) + `memory_fraction`（对标
`reward.gpu_pool`）；`_parse_rollout_pool` 解析新键并把旧 `distributed.rollout.colocate` 映射成
`gpu_pool=trainer`+fraction（both-set 报错）；`_resolve_rollout_devices` 按 gpu_pool 分支（trainer=共卡 /
dedicated=严格分离 / auto=原行为，含 overlap fallback）；persistent = (gpu_pool==trainer ∧ memory_fraction)；
gpu_pool=trainer 本身即 overlap 许可。schema 由 dataclass 字段自动识别新键，无需改。**共享 preset
`ray_rollout_colocated_single_gpu` 不迁**（其 `auto` 在多卡上优先选空闲卡 = disaggregate，迁成 trainer 会变行为）——
只更新注释文档化新语法；`ddp_2x1` 用 `gpu_pool: trainer` 做 showcase。新增 7 个 gpu_pool 测试 + 旧 colocate
测试经垫片全过。

## 默认拓扑裁决（2026-06-18）

**online diffusion RL 的默认拓扑 = colocated**；disaggregated 是为 async / stale-tolerant 算法（GRPO）以后才加的
高级拓扑。理由：DiffusionNFT 是 on-policy、`max_stale=0`，吃不了 stale rollout；disaggregated 的核心价值是
rollout/train 重叠，但 NFT 下 remote 生成的下一步样本变 stale 被丢弃 → 退化成"两张卡轮流闲"，收益很差。colocated
是"同版本生成 → 训练 → 同步推进"，正合 paper / cosmos-rl。

落到本语法：默认 `rollout.gpu_pool` 不写 / `auto` 走 colocated 心智（单卡或 DDP 每 rank 本地共卡）；disaggregated
（`dedicated` 分池 + async 重叠）留给后续 async/GRPO/multi-pool sprint，不进默认。文档措辞：
*"Default topology for online diffusion RL: colocated. Disaggregated is an advanced/future topology for async GRPO
or stale-tolerant algorithms."*

## 问题

同一个概念——"一个 role 借另一个 role 的 GPU pool"——现在用两套不一致的语法表达：

| 概念 | 现写法 | 语法 / 位置 |
|---|---|---|
| reward 借 rollout 池 | `distributed.resources.reward.gpu_pool: rollout` | 声明式 pool 名，`resources` 块（`_parse_reward_gpu_pool`，`resources.py:1184`） |
| rollout 借 trainer 池（共卡） | `distributed.rollout.colocate: {memory_fraction: X}` | 动词块、无 pool 名，`rollout` runtime 块（`_parse_colocate`，`resources.py:1121`） |

`reward` 已有 `gpu_pool: auto\|rollout\|dedicated` 的 role-based 语法；rollout 共卡却是一次性的 `colocate` 块，
风格、位置都不同——读者看不出二者是同一类参数。`colocated`/`disaggregated` 适合做**文档拓扑名**，不适合做顶层
config key；VRL 的 role/resource grammar 里这件事应由 `gpu_pool` 表达。

## 目标语法（per-role `gpu_pool`）

```yaml
distributed:
  resources:
    rollout:
      num_gpus: 1
      gpu_pool: trainer        # 借 trainer 池 = 共卡（取代 rollout.colocate）
      memory_fraction: 0.4     # 可选：常驻时的显存上限；不写 = on_demand
    reward:
      gpu_pool: rollout        # 不变
```

- 三 role 统一心智：`gpu_pool` = "借谁的池"（`auto` / 另一 role 名 / `dedicated`）；可选 `memory_fraction` =
  "常驻时给该 worker 的显存上限"。
- rollout 取值 `auto|trainer|dedicated`（对应 reward 的 `auto|rollout|dedicated`——"借" 的值就是被借 role 的名）。
- 不写 / `auto` = 默认 disaggregated（= cosmos-rl `mode=disaggregated`）；`gpu_pool: trainer` = colocated
  （= cosmos-rl `mode=colocated`）。

### on_demand / persistent 映射（与现状同 toggle，换语法）

- `gpu_pool: trainer`（无 `memory_fraction`）→ **on_demand**：rollout 阶段间释放、trainer collect 时 offload，
  每阶段独占整卡（和 reward 借 rollout 池一样是 on_demand）。
- `gpu_pool: trainer` + `memory_fraction: X` → **persistent/resident**：rollout 常驻、与 trainer 同时占显存，
  故 `memory_fraction` 必填封顶。
- 即"写不写 `memory_fraction"`决定 on_demand vs 常驻——和今天 `colocate` 块完全一致，只是挂到统一语法下。
- on_demand vs persistent 仍是从拓扑派生的内部状态（`resources.py` release/lifecycle 派生），不是字面 key。

## 迁移 / 兼容

- **兼容垫片**：`distributed.rollout.colocate`（+ legacy `colocate_with_trainer`）继续被接受，解析时映射到
  `resources.rollout.gpu_pool=trainer` + `memory_fraction`（仿 reward 的 `share_with_rollout → gpu_pool` 垫片，
  `resources.py:1184-1200`）。旧配置/preset 不一改就废；设双源（新旧都写）= 报错，与现有 _parse 一致。
- **`allow_overlap` 衔接**：`rollout.gpu_pool: trainer` 本身即"允许和 trainer 共卡"的许可（同今 `colocate ... is
  itself the overlap permission`，`resources.py:193-195`）。gpu_pool:trainer 隐含许可；`allow_overlap` 退为更底层
  的"显式 device 重叠"逃生口，语义不变。
- **位置搬家**：共卡从 `distributed.rollout`（runtime）挪到 `distributed.resources.rollout`（placement）——更贴合
  它本就是放置决策。

## 改动面

- `vrl/config/schema.py`：`RolloutResourceConfig` 加 `gpu_pool: str` + `memory_fraction: float|None`（显式 field）。
- `vrl/ray/resources.py`：`_resolve_rollout_devices` 读 `rollout.gpu_pool`（trainer→借 trainer 池 = 现 colocate
  分支；dedicated/auto→现 disjoint 分支）；`_parse_colocate` 改为兼容垫片映射到新字段；persistent/memory_fraction
  派生不变。
- presets/configs：`ray_rollout_colocated_single_gpu.yaml`、`online_nft_kling_video_reward_ddp_2x1.yaml` 改用新语法
  （或保留旧语法靠垫片）。
- tests：reward+rollout 的 `gpu_pool` 对称解析、垫片等价、allow_overlap 衔接。

## 非目标

- 不引入 `mode: colocated/disaggregated` 顶层枚举（违背 role/resource grammar；colocated/disaggregated 仅作文档名）。
- 不改任何 运行时行为（on_demand/persistent/offload/release 全部不变；这是纯语法+垫片）。
- 不删旧 `colocate` key（保留为兼容输入；可在某个大版本后再清）。
- 不动 reward 侧语法（已是 `gpu_pool`，作为对齐目标）。

## 关键引用

- 对齐目标（reward）：`vrl/ray/resources.py:35-41`（`RewardResourceConfig.gpu_pool`）、`:1184-1200`
  （`_parse_reward_gpu_pool` + `share_with_rollout` 垫片）、`:809-899`（reward pool 解析分支）。
- 待统一（rollout colocate）：`vrl/ray/resources.py:1121-1148`（`_parse_colocate`）、`:193-226`（colocate=overlap
  permission + colocation 许可校验）、`:275-339`（persistent/memory_fraction 强制 + release/lifecycle 派生）。
- cosmos-rl 词汇来源：`mode=disaggregated|colocated`（`cosmos_rl/.../launch_all.py`）——给了拓扑名，VRL 用自己的
  role grammar 表达。
